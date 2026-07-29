//! Synchronous file transcription (`POST /v1/transcribe`).

use axum::body::Bytes;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde::Serialize;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, OnceLock};

use super::super::config::RuntimeLimits;
use super::super::metrics::MetricsRegistry;
use gigastt_core::export::Segment;
use gigastt_core::inference::{DiarizationOutcome, Engine};

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
    /// Present only when `?diarization=true` was requested and speakers could
    /// **not** be labeled — the transcript is still complete, but the client
    /// would otherwise see empty speaker fields with no explanation. Additive:
    /// absent when diarization was not requested or succeeded, so existing
    /// clients see the exact same shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diarization: Option<DiarizationNotice>,
}

/// Capability notice attached to a transcription response when a
/// `?diarization=true` request produced no speaker labels. Makes the reason
/// visible to the client instead of returning an all-empty-speaker transcript
/// with HTTP 200 and no explanation.
#[derive(Serialize)]
pub struct DiarizationNotice {
    /// Always `"unavailable"` — the notice only appears when diarization was
    /// requested and did not label speakers.
    pub status: &'static str,
    /// Machine-readable cause: `"duration_ceiling"`, `"no_speaker_model"`, or
    /// `"pipeline_error"`.
    pub reason: &'static str,
    /// Human-readable, non-sensitive explanation.
    pub message: String,
    /// Length of the submitted audio in seconds — present only for the duration
    /// ceiling, so the client can see what was too long.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_seconds: Option<f64>,
    /// The clusterer's single-pass ceiling in seconds — present only for the
    /// duration ceiling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ceiling_seconds: Option<f64>,
}

/// Map a [`DiarizationOutcome`] to the client-facing [`DiarizationNotice`].
///
/// `Applied` (and any future success) yields `None` — no notice, so a
/// successful or unrequested diarization leaves the response shape untouched.
/// Every declined case carries a distinct `reason`, and the duration ceiling
/// also carries the input and ceiling seconds.
fn diarization_notice(outcome: DiarizationOutcome) -> Option<DiarizationNotice> {
    match outcome {
        DiarizationOutcome::Applied => None,
        DiarizationOutcome::NoSpeakerModel => Some(DiarizationNotice {
            status: "unavailable",
            reason: "no_speaker_model",
            message: "diarization was requested but no speaker model is loaded".to_string(),
            input_seconds: None,
            ceiling_seconds: None,
        }),
        DiarizationOutcome::DurationCeiling {
            input_secs,
            ceiling_secs,
        } => Some(DiarizationNotice {
            status: "unavailable",
            reason: "duration_ceiling",
            message: format!(
                "diarization skipped: input {input_secs:.0}s exceeds the {ceiling_secs:.0}s \
                 single-pass limit; the transcript is complete but has no speaker labels"
            ),
            input_seconds: Some(input_secs),
            ceiling_seconds: Some(ceiling_secs),
        }),
        DiarizationOutcome::Failed => Some(DiarizationNotice {
            status: "unavailable",
            reason: "pipeline_error",
            message: "diarization failed; the transcript is complete but has no speaker labels"
                .to_string(),
            input_seconds: None,
            ceiling_seconds: None,
        }),
        // `DiarizationOutcome` is `#[non_exhaustive]`; a future variant should
        // add its own notice above. Until then, prefer silence over a wrong one.
        _ => None,
    }
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

#[cfg(test)]
pub(super) use super::super::file_transcribe::raw_codec_to_wav;

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
    diar_sink: Option<Arc<OnceLock<DiarizationOutcome>>>,
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

    // Cooperative cancellation + real-progress plumbing, shared with the jobs
    // path. `abort` lets a client disconnect, a fired shutdown, or the
    // no-progress watchdog stop the detached ONNX run at its next window;
    // `progress` carries per-window processed-sample counts back to the watchdog
    // so a long file that keeps advancing never trips the timeout.
    let abort = Arc::new(AtomicBool::new(false));
    let progress = Arc::new(AtomicU64::new(0));

    let file_opts = super::super::file_transcribe::FileTranscribeOpts {
        overrides,
        hotwords,
        split_channels,
        diarization: request_diarization,
        raw_codec,
        abort: Some(abort.clone()),
        progress: Some(progress.clone()),
        diarization_outcome: diar_sink,
        max_audio_secs: limits.max_audio_secs_opt(),
    };

    // A client disconnect drops this handler future before it returns, dropping
    // the guard and flipping `abort`; the detached blocking run then cancels at
    // its next window and returns the triplet. On the normal return path the
    // guard fires after the result is already in hand, so it is a harmless no-op.
    let _abort_guard = super::super::file_transcribe::AbortOnDrop(abort.clone());

    let inference_start = std::time::Instant::now();
    let span = tracing::Span::current();
    // Route through the shared TaskTracker (like the WS / SSE paths) so a
    // SIGTERM drain can wait for an in-flight REST transcription up to
    // `--shutdown-drain-secs`; a bare `tokio::task::spawn_blocking` is untracked
    // and would be abandoned on shutdown, cutting the response mid-run.
    let handle = state.tracker.spawn_blocking(move || {
        let _enter = span.enter();
        // catch_unwind ensures triplet is returned to pool even on panic
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            super::super::file_transcribe::run_file_transcribe_blocking(
                &engine,
                body,
                &mut reservation,
                &file_opts,
            )
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

    // The per-request inference timeout is now a no-progress watchdog: the
    // deadline resets whenever a window completes, so a long file streaming
    // steady progress never trips it, while a genuinely stalled run still trips
    // at the same moment and returns a typed `inference_timeout` (504). On a
    // trip — and on shutdown — the watchdog flips `abort`, so the detached run
    // releases its triplet within one window instead of staying wedged for the
    // whole file. `0` disables the watchdog (shutdown still cancels).
    let inference_timeout_secs = limits.inference_timeout_secs;
    let outcome = super::super::file_transcribe::await_transcription_watchdog(
        handle,
        &progress,
        &abort,
        inference_timeout_secs,
        &state.shutdown,
    )
    .await;

    let result = match outcome {
        super::super::file_transcribe::WatchdogOutcome::Joined(r) => r,
        super::super::file_transcribe::WatchdogOutcome::TimedOut => {
            if let Some(ref reg) = state.metrics_registry {
                reg.counter_inc("gigastt_inference_timeouts_total", &[], 1);
            }
            tracing::error!(
                "REST inference made no progress for {inference_timeout_secs}s — aborting"
            );
            return Err(api_inference_timeout_error());
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
        Ok(Err(gigastt_core::error::GigasttError::Cancelled)) => {
            // Reached only when a shutdown cancelled the run while the client was
            // still connected (a disconnect drops this future before here). Be
            // honest that the server is going away rather than faking success.
            Err(api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "Server is shutting down",
                "cancelled",
            ))
        }
        Ok(Err(e)) if matches!(&e, gigastt_core::error::GigasttError::AudioTooLong { .. }) => {
            // Distinct from the generic decode failure below: a client must be
            // able to tell "too long" (raise `--max-audio-secs` or split) from
            // "corrupt". 413 + `audio_too_long`, with the observed/limit seconds.
            Err(api_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                &e.to_string(),
                e.code(),
            ))
        }
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
    // Only sink the diarization outcome when diarization was requested, so a
    // `?diarization=true` request that produces no speaker labels can be turned
    // into a visible notice instead of an all-empty-speaker transcript.
    let diar_sink: Option<Arc<OnceLock<DiarizationOutcome>>> =
        (params.diarization == Some(true)).then(|| Arc::new(OnceLock::new()));
    let result = run_file_transcription(&state, body, &params, diar_sink.clone()).await?;
    if let Some(rendered) = render_export_response(&result, &params)? {
        // Export formats (SRT/VTT/TSV/plain text) have no field for a capability
        // notice; it surfaces on the JSON path below.
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
        // Attach a diarization notice only when diarization was requested and
        // declined (ceiling / no model / pipeline error); `Applied` and the
        // not-requested case leave the field absent.
        let diarization = diar_sink
            .as_ref()
            .and_then(|sink| sink.get().copied())
            .and_then(diarization_notice);
        Ok(Json(TranscribeResponse {
            text: result.text,
            words: result.words,
            duration: result.duration_s,
            confidence: result.confidence,
            segments,
            diarization,
        })
        .into_response())
    }
}

#[cfg(test)]
mod diarization_notice_tests {
    use super::*;

    #[test]
    fn test_applied_yields_no_notice() {
        // A successful diarization needs no notice: speakers ride on the words.
        assert!(diarization_notice(DiarizationOutcome::Applied).is_none());
    }

    #[test]
    fn test_duration_ceiling_carries_numbers_and_reason() {
        let notice = diarization_notice(DiarizationOutcome::DurationCeiling {
            input_secs: 5400.0,
            ceiling_secs: 3600.0,
        })
        .expect("a declined ceiling must produce a notice");
        assert_eq!(notice.status, "unavailable");
        assert_eq!(notice.reason, "duration_ceiling");
        assert_eq!(notice.input_seconds, Some(5400.0));
        assert_eq!(notice.ceiling_seconds, Some(3600.0));

        // The wire JSON carries both numbers and the machine-readable reason.
        let json = serde_json::to_value(&notice).expect("serialize");
        assert_eq!(json["reason"], "duration_ceiling");
        assert_eq!(json["input_seconds"], 5400.0);
        assert_eq!(json["ceiling_seconds"], 3600.0);
    }

    #[test]
    fn test_no_speaker_model_reason_without_numbers() {
        let notice = diarization_notice(DiarizationOutcome::NoSpeakerModel).expect("notice");
        assert_eq!(notice.reason, "no_speaker_model");
        assert!(notice.input_seconds.is_none());
        assert!(notice.ceiling_seconds.is_none());
        // `skip_serializing_if` keeps the number fields out of the JSON entirely.
        let json = serde_json::to_value(&notice).expect("serialize");
        assert!(json.get("input_seconds").is_none());
        assert!(json.get("ceiling_seconds").is_none());
    }

    #[test]
    fn test_failed_maps_to_pipeline_error() {
        let notice = diarization_notice(DiarizationOutcome::Failed).expect("notice");
        assert_eq!(notice.reason, "pipeline_error");
        assert!(notice.input_seconds.is_none());
    }

    #[test]
    fn test_response_omits_diarization_field_when_absent() {
        // The default (not-requested / succeeded) response must be byte-shape
        // compatible: no `diarization` key at all.
        let resp = TranscribeResponse {
            text: "hi".to_string(),
            words: Vec::new(),
            duration: 1.0,
            confidence: None,
            segments: None,
            diarization: None,
        };
        let json = serde_json::to_value(&resp).expect("serialize");
        assert!(
            json.get("diarization").is_none(),
            "diarization must be omitted when there is no notice"
        );
    }
}
