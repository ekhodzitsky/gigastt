//! OpenAI-compatible audio transcriptions surface.

use axum::body::Bytes;
use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;
use std::sync::Arc;

use super::error::{ApiError, api_error};
use super::export::ExportParams;
use super::state::AppState;
use super::transcribe::{reserve_batch_slot, run_file_transcription};

/// POST /v1/audio/transcriptions — OpenAI-compatible file transcription.
///
/// Compatibility surface for clients that speak the OpenAI Audio Transcriptions
/// API (llama-swap, Hermes Agent, OpenAI SDKs with a custom `base_url`):
/// `multipart/form-data` with required `file` and optional
/// `model` / `response_format` / `language` / `timestamp_granularities[]` /
/// `stream`.
///
/// | `response_format` | Body |
/// |---|---|
/// | `json` (default) | `{"text":"..."}` |
/// | `text` | plain text |
/// | `srt` / `vtt` | captions |
/// | `verbose_json` | Whisper-style JSON (`task`, `language`, `duration`, `text`, optional `segments`/`words`) |
///
/// With `stream=true` (only with `json`/`text`): SSE of
/// `transcript.text.delta` events, a final `transcript.text.done`, then
/// `data: [DONE]`. Progressive deltas come from the real chunked encoder path.
///
/// Reuses the same inference pipeline as [`super::transcribe::transcribe`]. `model` is accepted
/// and ignored (single loaded head). For diarization, telephony codecs, or
/// native export knobs use `/v1/transcribe`.
pub async fn openai_transcriptions(
    State(state): State<Arc<AppState>>,
    multipart: Multipart,
) -> Result<Response, ApiError> {
    transcribe_multipart(state, multipart, None).await
}

pub(crate) async fn openai_transcriptions_admitted(
    State(state): State<Arc<AppState>>,
    axum::Extension(permit): axum::Extension<super::super::upload::UploadPermit>,
    multipart: Multipart,
) -> Result<Response, ApiError> {
    transcribe_multipart(state, multipart, Some(permit)).await
}

async fn transcribe_multipart(
    state: Arc<AppState>,
    multipart: Multipart,
    permit: Option<super::super::upload::UploadPermit>,
) -> Result<Response, ApiError> {
    let mut req = super::super::openai::parse_openai_multipart(multipart).await?;
    if let Some(permit) = permit {
        req.file = permit.retain(req.file);
    }
    if req.options.stream {
        return openai_transcriptions_stream(state, req.file).await;
    }
    // The OpenAI-compatible alias does not expose diarization, so no outcome sink.
    let result = run_file_transcription(&state, req.file, &ExportParams::default(), None).await?;
    Ok(super::super::openai::render_openai_response(
        &result,
        &req.options,
    ))
}

/// OpenAI `stream=true` path: chunked file transcription as SSE transcript events.
///
/// Uses the same lazy [`AudioChunks`] open as native `/v1/transcribe/stream`
/// (reserve pool first, probe duration, decode on demand) so large uploads
/// cannot expand full PCM before inference.
async fn openai_transcriptions_stream(
    state: Arc<AppState>,
    body: Bytes,
) -> Result<Response, ApiError> {
    // Same early guards as `/v1/transcribe/stream` — empty / oversized body.
    if body.is_empty() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "Empty request body",
            "empty_body",
        ));
    }
    let limits = state.limits.load();
    if body.len() > limits.body_limit_bytes {
        return Err(api_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "Request body exceeds the configured size limit",
            "payload_too_large",
        ));
    }

    let engine = state.engine.load_full();
    let reservation = reserve_batch_slot(&engine, &limits, state.metrics_registry.as_ref()).await?;
    let (chunks, upload_lifetime, mut reservation) = super::stream::open_stream_chunks(
        &state,
        body,
        reservation,
        limits.max_audio_secs_opt(),
        limits.inference_timeout_secs,
    )
    .await?;

    // Channel of pre-rendered SSE `data:` payloads (JSON events or `[DONE]`).
    let (tx, rx) = tokio::sync::mpsc::channel::<String>(32);

    let cancel = state.shutdown.child_token();
    let tracker = state.tracker.clone();
    let (abort, finished) =
        super::super::transcription_control::stream_abort(&tx, cancel.clone(), &tracker);
    let watchdog = super::stream_watchdog::StreamWatchdog::start(
        &state,
        limits.inference_timeout_secs,
        abort.clone(),
        cancel.clone(),
    );
    let response_watchdog = watchdog.clone();
    let partial = Arc::new(gigastt_core::inference::TranscriptSnapshot::default());
    let response_partial = partial.clone();
    let span = tracing::Span::current();
    tracker.spawn_blocking(move || {
        let _completion = watchdog.completion();
        let _upload_lifetime = upload_lifetime;
        let _finished = finished;
        let _enter = span.enter();
        use super::super::openai::{OpenAIStreamAssembler, sse_delta_payload, sse_done_payload};

        let runtime = tokio::runtime::Handle::current();
        let send = |payload: String| runtime.block_on(watchdog.send(&tx, payload, &cancel, &abort));

        let mut asm = OpenAIStreamAssembler::new();
        let report_error = |asm: &mut OpenAIStreamAssembler, code, message: &str| {
            if !watchdog.claim_finish() {
                return;
            }
            if let Some(segment) = partial.get()
                && let Some(delta) = asm.push_segment(&segment.text, false)
                && !send(sse_delta_payload(&delta))
            {
                return;
            }
            let _ = send(super::stream::sse_data_payload(&Err(
                super::stream::StreamError {
                    code,
                    message: message.into(),
                },
            )));
        };

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut stream_state = engine.create_state(false);
            stream_state.abort = Some(abort.clone());
            stream_state.partial = Some(partial.clone());
            let mut chunks = chunks;

            loop {
                if cancel.is_cancelled() {
                    tracing::info!("OpenAI SSE transcription cancelled by shutdown");
                    return;
                }
                let chunk = match chunks.next_chunk() {
                    Ok(Some(c)) => c,
                    Ok(None) => break,
                    Err(e) => {
                        tracing::error!("OpenAI SSE audio decode error: {e:#}");
                        report_error(&mut asm, "invalid_audio", "Failed to decode audio file.");
                        return;
                    }
                };
                match engine.process_chunk(chunk, &mut stream_state, &mut reservation) {
                    Ok(segs) => {
                        watchdog.progress();
                        for seg in segs {
                            if let Some(delta) = asm.push_segment(&seg.text, seg.is_final)
                                && !send(sse_delta_payload(&delta))
                            {
                                return; // client gone
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("OpenAI SSE transcription error: {e}");
                        report_error(&mut asm, e.code(), "Transcription failed.");
                        return;
                    }
                }
            }

            match engine.try_finish_stream(&mut stream_state, &mut reservation) {
                Ok(Some(segment)) => {
                    if !watchdog.claim_finish() {
                        return;
                    }
                    if let Some(delta) = asm.push_segment(&segment.text, segment.is_final)
                        && !send(sse_delta_payload(&delta))
                    {
                        return;
                    }
                }
                Ok(None) => {
                    if !watchdog.claim_finish() {
                        return;
                    }
                }
                Err(error) => {
                    tracing::error!("Final OpenAI SSE decode failed: {error:#}");
                    report_error(&mut asm, error.code(), "Failed to finish transcription.");
                    return;
                }
            }

            if send(sse_done_payload(asm.text())) {
                let _ = send("[DONE]".into());
            }
        }));

        if result.is_err() {
            tracing::error!("Panic in OpenAI SSE inference task — triplet recovered");
            report_error(
                &mut asm,
                "inference_panic",
                "Inference failed unexpectedly.",
            );
        }
        // reservation dropped → pool
    });

    let stream = response_watchdog
        .response(rx, move || {
            let mut error = serde_json::json!({
                "type": "error", "code": "inference_timeout",
                "message": "Transcription made no progress within the configured timeout.",
            });
            if let Some(segment) = response_partial.get() {
                error["partial"] = serde_json::to_value(segment).unwrap_or(serde_json::Value::Null);
            }
            vec![error.to_string()]
        })
        .map(|data| Ok::<_, std::convert::Infallible>(super::super::openai::sse_event_data(data)));

    Ok(Sse::new(stream)
        .keep_alive(
            KeepAlive::new()
                .interval(std::time::Duration::from_secs(15))
                .text(""),
        )
        .into_response())
}
