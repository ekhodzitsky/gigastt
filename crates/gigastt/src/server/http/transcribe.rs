//! Synchronous file transcription (`POST /v1/transcribe`).

use axum::body::Bytes;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde::Serialize;
use std::str::FromStr;
use std::sync::Arc;

use super::super::config::RuntimeLimits;
use super::super::metrics::MetricsRegistry;
use gigastt_core::export::Segment;
use gigastt_core::inference::Engine;

use super::error::{
    ApiError, api_error, api_inference_timeout_error, api_pool_closed_error, api_timeout_error,
};
use super::export::{
    ExportParams, hotwords_from_export_params, overrides_from_export_params, render_export_response,
};
use super::state::AppState;

/// Transcription response.
#[derive(Serialize)]
pub struct TranscribeResponse {
    /// Full recognized transcript text.
    pub text: String,
    /// Word-level timing, confidence, and optional speaker annotations.
    pub words: Vec<gigastt_core::inference::WordInfo>,
    /// Duration of the submitted audio in seconds.
    pub duration: f64,
    /// Mean confidence across all words (duration-weighted average of
    /// `words[].confidence`): an average of per-word softmax scores, not a
    /// calibrated transcript probability. Additive; omitted when no words
    /// were decoded so the pre-field response shape is preserved.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    /// Transcript segments grouped from word timestamps, present only when the
    /// caller passed `?segments=true`. Additive: absent from the default response,
    /// so existing clients that read `text` / `words` / `duration` are unaffected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<Segment>>,
}

/// Resolve the `?codec=` / `?sample_rate=` query pair into a raw-decode recipe.
/// `Ok(None)` means no `codec` was given and the body is probed as a container
/// as before. Validation is done up front so a malformed request fails with a
/// 400 before a pool slot is reserved. Kept separate from the handler so the
/// status codes are unit-testable without a model.
#[allow(clippy::result_large_err)]
pub(super) fn resolve_raw_codec(
    params: &ExportParams,
) -> Result<Option<(gigastt_core::inference::audio::TelephonyCodec, u32)>, ApiError> {
    let Some(name) = params.codec.as_deref() else {
        return Ok(None);
    };
    let codec =
        gigastt_core::inference::audio::TelephonyCodec::from_name(name).ok_or_else(|| {
            api_error(
                StatusCode::BAD_REQUEST,
                "Unsupported codec. Supported: pcmu (ulaw), pcma (alaw), g722",
                "unsupported_codec",
            )
        })?;
    let sample_rate = params.sample_rate.ok_or_else(|| {
        api_error(
            StatusCode::BAD_REQUEST,
            "sample_rate is required when codec is set",
            "invalid_sample_rate",
        )
    })?;
    codec
        .validate_sample_rate(sample_rate)
        .map_err(|reason| api_error(StatusCode::BAD_REQUEST, &reason, "invalid_sample_rate"))?;
    Ok(Some((codec, sample_rate)))
}

/// Decode a raw telephony byte stream and re-wrap it as an in-memory PCM16
/// WAV, so every downstream engine path (mono, `channels=split`, diarized)
/// works unchanged. Runs inside the blocking inference closure; errors map to
/// the same 422 as container decode failures.
pub(super) fn raw_codec_to_wav(
    body: &[u8],
    codec: gigastt_core::inference::audio::TelephonyCodec,
    sample_rate: u32,
) -> Result<Bytes, gigastt_core::error::GigasttError> {
    let samples = gigastt_core::inference::audio::decode_telephony_raw(body, codec, sample_rate)
        .map_err(|e| gigastt_core::error::GigasttError::InvalidAudio {
            reason: format!("{e:#}"),
        })?;
    Ok(Bytes::from(
        gigastt_core::inference::audio::encode_wav_pcm16(&samples, 16000),
    ))
}

/// Check out a session triplet from the engine's batch pool with the configured
/// timeout and record the pool metrics, returning an owned reservation whose
/// lifetime is detached (`into_owned`) so it can travel through `spawn_blocking`.
///
/// Both `/v1/transcribe` and `/v1/transcribe/stream` reserve a slot *before*
/// decoding the upload; sharing the acquisition here keeps that backpressure —
/// and the "reserve before decode" ordering that caps concurrent decodes at the
/// pool size — identical across the two untrusted entry points. A saturated pool
/// yields a 503 (`timeout`, with `Retry-After`) or `pool_closed`; the caller
/// propagates it with `?`.
pub(super) async fn reserve_batch_slot(
    engine: &Engine,
    limits: &RuntimeLimits,
    metrics: Option<&Arc<MetricsRegistry>>,
) -> Result<
    gigastt_core::inference::OwnedReservation<gigastt_core::inference::SessionTriplet>,
    ApiError,
> {
    let checkout_start = std::time::Instant::now();
    let guard = match tokio::time::timeout(
        std::time::Duration::from_secs(limits.pool_checkout_timeout_secs),
        engine.pool_for_batch().checkout(),
    )
    .await
    {
        Ok(Ok(guard)) => guard,
        Ok(Err(_pool_closed)) => return Err(api_pool_closed_error()),
        Err(_timeout) => {
            if let Some(reg) = metrics {
                reg.counter_inc("gigastt_pool_timeouts_total", &[], 1);
                reg.histogram_record(
                    "gigastt_pool_checkout_duration_seconds",
                    &[],
                    checkout_start.elapsed().as_secs_f64(),
                );
            }
            return Err(api_timeout_error(limits));
        }
    };
    if let Some(reg) = metrics {
        reg.histogram_record(
            "gigastt_pool_checkout_duration_seconds",
            &[],
            checkout_start.elapsed().as_secs_f64(),
        );
    }
    Ok(guard.into_owned())
}

/// Shared file-transcription pipeline used by `/v1/transcribe` and the
/// OpenAI-compatible `/v1/audio/transcriptions` alias. Returns the engine
/// result so each surface can shape its own response envelope.
pub(super) async fn run_file_transcription(
    state: &AppState,
    body: Bytes,
    params: &ExportParams,
) -> Result<gigastt_core::inference::TranscribeResult, ApiError> {
    if body.is_empty() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "Empty request body",
            "empty_body",
        ));
    }

    // Defence-in-depth: `DefaultBodyLimit` already rejects oversized bodies
    // before they reach this handler, but a mis-ordered middleware stack or
    // a `Content-Length`-spoofing client could still deliver too many bytes.
    // The explicit 413 keeps the REST contract honest and gives clients a
    // machine-readable `payload_too_large` code alongside the spec-conformant
    // status. Cheap: `Bytes::len()` is a load, not a walk.
    let limits = state.limits.load();
    if body.len() > limits.body_limit_bytes {
        return Err(api_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "Request body exceeds the configured size limit",
            "payload_too_large",
        ));
    }

    let split_channels = params.channels.as_deref() == Some("split");
    let request_diarization = params.diarization == Some(true);
    if split_channels && request_diarization {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "channels=split and diarization=true are mutually exclusive",
            "conflicting_modes",
        ));
    }

    // Raw telephony upload (`?codec=`): validate the codec/rate pair up front
    // so a malformed request 400s before a pool slot is reserved. The actual
    // decode happens inside the blocking closure below.
    let raw_codec = resolve_raw_codec(params)?;

    // Snapshot the current engine once at request start; a concurrent hot-reload
    // swaps the `ArcSwap`, but this request keeps the engine (and its pool) it
    // began with for its whole lifetime.
    let engine = state.engine.load_full();

    // Per-request recognition-knob overrides (additive; all absent = the boot
    // defaults, byte-identical to the pre-feature response). Validate them
    // *before* checking out a session so an impossible request (a knob turned on
    // without its resource loaded, or a variant this single-model engine can't
    // serve) fails fast with a 409 instead of holding a pool triplet.
    if let Some(requested) = params.variant.as_deref() {
        // Forward-compat guard only: a single-variant engine can't switch heads,
        // so a `?variant=X` where X != the loaded variant is a 409. An unknown
        // token likewise can't match the loaded variant, so it 409s too.
        let matches = gigastt_core::model::ModelVariant::from_str(requested)
            .map(|v| v == engine.variant())
            .unwrap_or(false);
        if !matches {
            return Err(api_error(
                StatusCode::CONFLICT,
                "Requested model variant is not loaded",
                "variant_not_loaded",
            ));
        }
    }
    let overrides = overrides_from_export_params(params);
    if let Err(e) = engine.validate_overrides(&overrides) {
        return Err(api_error(StatusCode::CONFLICT, e.message(), e.code()));
    }
    let hotwords = hotwords_from_export_params(params);
    if let Some(ref hw) = hotwords
        && let Err(e) = engine.validate_hotwords(hw)
    {
        return Err(api_error(StatusCode::BAD_REQUEST, e.message(), e.code()));
    }

    // Checkout a session triplet from the batch pool (blocks if none available)
    // — this is a long file-transcription job, so it draws from the dedicated
    // batch pool when one exists (falling back to the interactive pool otherwise)
    // to avoid starving WebSocket / SSE streaming. Shared with the SSE handler
    // (`reserve_batch_slot`) so both reserve identically, and before decoding.
    let mut reservation =
        reserve_batch_slot(&engine, &limits, state.metrics_registry.as_ref()).await?;

    let inference_start = std::time::Instant::now();
    let span = tracing::Span::current();
    let handle = tokio::task::spawn_blocking(move || {
        let _enter = span.enter();
        // catch_unwind ensures triplet is returned to pool even on panic
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // Raw telephony upload: decode the headerless byte stream and
            // re-wrap it as an in-memory WAV so every downstream path (mono,
            // channels=split, diarized) works unchanged.
            let body = match raw_codec {
                Some((codec, rate)) => raw_codec_to_wav(&body, codec, rate)?,
                None => body,
            };
            if split_channels {
                let channels = gigastt_core::inference::audio::decode_audio_bytes_shared_channels(
                    body.clone(),
                )
                .map_err(|e| gigastt_core::error::GigasttError::InvalidAudio {
                    reason: format!("{e:#}"),
                })?;
                let fallback_reason = match channels.len() {
                    0 => Some("no channels"),
                    1 => Some("mono audio"),
                    2 if gigastt_core::inference::audio::is_dual_mono(&channels) => {
                        Some("dual-mono audio")
                    }
                    n if n > 2 => Some("more than two channels"),
                    _ => None,
                };
                if let Some(reason) = fallback_reason {
                    tracing::warn!(
                        "channels=split requested but {reason} detected; falling back to mono transcription"
                    );
                    engine.transcribe_bytes_shared(body, &mut reservation)
                } else {
                    engine.transcribe_channels(&channels, &mut reservation)
                }
            } else if request_diarization {
                // Diarization is opt-in (`?diarization=true`): only then run the
                // offline speaker pass. `body` is `axum::body::Bytes` (a
                // `bytes::Bytes` re-export) so the decode shares the upload buffer.
                engine.transcribe_bytes_shared_with_overrides_diarized_hotwords(
                    body,
                    &mut reservation,
                    &overrides,
                    hotwords.as_ref(),
                )
            } else {
                // No diarization requested: byte-identical to the pre-existing
                // zero-copy path. `overrides` is already validated above.
                engine.transcribe_bytes_shared_with_overrides_hotwords(
                    body,
                    &mut reservation,
                    &overrides,
                    hotwords.as_ref(),
                )
            }
        }));
        match r {
            Ok(inference_result) => inference_result,
            Err(_) => {
                tracing::error!("Panic in REST transcribe — triplet recovered");
                Err(gigastt_core::error::GigasttError::Inference {
                    source: anyhow::anyhow!("Inference thread panicked").into(),
                })
            }
        }
        // reservation dropped here automatically returns the triplet to the pool
    });

    // Guard the blocking ONNX run with the per-request inference timeout
    // (`0` disables). `spawn_blocking` can't be cancelled, so the detached task
    // keeps the triplet and returns the slot to the pool only when the run
    // finishes; the client gets a typed `inference_timeout` (504) immediately.
    let inference_timeout_secs = limits.inference_timeout_secs;
    let result = if inference_timeout_secs == 0 {
        handle.await
    } else {
        match tokio::time::timeout(
            std::time::Duration::from_secs(inference_timeout_secs),
            handle,
        )
        .await
        {
            Ok(r) => r,
            Err(_elapsed) => {
                if let Some(ref reg) = state.metrics_registry {
                    reg.counter_inc("gigastt_inference_timeouts_total", &[], 1);
                }
                tracing::error!("REST inference exceeded {inference_timeout_secs}s — aborting");
                return Err(api_inference_timeout_error());
            }
        }
    };
    if let Some(ref reg) = state.metrics_registry {
        reg.histogram_record(
            "gigastt_inference_duration_seconds",
            &[],
            inference_start.elapsed().as_secs_f64(),
        );
    }

    match result {
        Ok(Ok(result)) => Ok(result),
        Ok(Err(e)) => {
            tracing::error!("Transcription error: {e}");
            Err(api_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "Transcription failed. Check audio format.",
                "transcription_error",
            ))
        }
        Err(e) => {
            // spawn_blocking task itself failed (e.g., runtime shutdown).
            // The reservation was dropped inside the closure and the triplet
            // was returned to the pool automatically.
            tracing::error!("spawn_blocking join error: {e}");
            Err(api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal server error",
                "internal",
            ))
        }
    }
}

/// POST /v1/transcribe — upload audio file, get full transcript.
///
/// Accepts raw audio body. Supported formats: WAV (including G.711 A-law /
/// μ-law and G.722 inside WAV), MP3, M4A/AAC, OGG, FLAC — plus headerless
/// telephony streams when `?codec=pcmu|pcma|g722&sample_rate=N` is given.
/// Max body size enforced by the axum `DefaultBodyLimit` layer configured
/// from [`RuntimeLimits::body_limit_bytes`] (default 50 MiB).
pub async fn transcribe(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ExportParams>,
    body: Bytes,
) -> Result<Response, ApiError> {
    let result = run_file_transcription(&state, body, &params).await?;
    if let Some(rendered) = render_export_response(&result, &params)? {
        Ok(rendered)
    } else {
        // Default JSON response. `?segments=true` adds a cue-grouped
        // `segments` array (same boundaries as SRT/VTT) alongside the
        // unchanged top-level `text` / `words` / `duration`; absent
        // otherwise so existing clients see the exact same shape.
        let segments = if params.segments.unwrap_or(false) {
            Some(gigastt_core::export::to_transcript_segments(&result.words))
        } else {
            None
        };
        Ok(Json(TranscribeResponse {
            text: result.text,
            words: result.words,
            duration: result.duration_s,
            confidence: result.confidence,
            segments,
        })
        .into_response())
    }
}
