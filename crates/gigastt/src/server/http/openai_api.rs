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
    let req = super::super::openai::parse_openai_multipart(multipart).await?;
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
    let mut reservation =
        reserve_batch_slot(&engine, &limits, state.metrics_registry.as_ref()).await?;

    let samples = tokio::task::spawn_blocking(move || {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gigastt_core::inference::audio::decode_audio_bytes_shared(body)
        })) {
            Ok(inner) => inner,
            Err(_) => {
                tracing::error!("Panic in OpenAI SSE audio decode — treated as decode error");
                Err(anyhow::anyhow!("Audio decode thread panicked"))
            }
        }
    })
    .await
    .map_err(|e| {
        tracing::error!("spawn_blocking join error: {e}");
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Internal server error",
            "internal",
        )
    })?
    .map_err(|e| {
        tracing::error!("Audio decode error: {e:#}");
        api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "Failed to decode audio file. Check format (WAV, MP3, M4A, OGG, FLAC supported).",
            "invalid_audio",
        )
    })?;

    // Channel of pre-rendered SSE `data:` payloads (JSON events or `[DONE]`).
    let (tx, rx) = tokio::sync::mpsc::channel::<String>(32);

    let cancel = state.shutdown.clone();
    let tracker = state.tracker.clone();
    let span = tracing::Span::current();
    tracker.spawn_blocking(move || {
        let _enter = span.enter();
        use super::super::openai::{OpenAIStreamAssembler, sse_delta_payload, sse_done_payload};

        let send = |payload: String| -> bool { tx.blocking_send(payload).is_ok() };

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut stream_state = engine.create_state(false);
            let mut asm = OpenAIStreamAssembler::new();
            let chunk_size = 16000; // 1 s @ 16 kHz

            for chunk in samples.chunks(chunk_size) {
                if cancel.is_cancelled() {
                    tracing::info!("OpenAI SSE transcription cancelled by shutdown");
                    return;
                }
                match engine.process_chunk(chunk, &mut stream_state, &mut reservation) {
                    Ok(segs) => {
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
                        // Surface a final done with whatever we have so clients
                        // do not hang; OpenAI stream errors are not standardized.
                        let _ = send(sse_done_payload(asm.text()));
                        let _ = send("[DONE]".into());
                        return;
                    }
                }
            }

            if let Some(seg) = engine.finish_stream(&mut stream_state, &mut reservation)
                && let Some(delta) = asm.push_segment(&seg.text, seg.is_final)
            {
                let _ = send(sse_delta_payload(&delta));
            }

            let _ = send(sse_done_payload(asm.text()));
            let _ = send("[DONE]".into());
        }));

        if result.is_err() {
            tracing::error!("Panic in OpenAI SSE inference task — triplet recovered");
            let _ = send(sse_done_payload(""));
            let _ = send("[DONE]".into());
        }
        // reservation dropped → pool
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx)
        .map(|data| Ok::<_, std::convert::Infallible>(super::super::openai::sse_event_data(data)));

    Ok(Sse::new(stream)
        .keep_alive(
            KeepAlive::new()
                .interval(std::time::Duration::from_secs(15))
                .text(""),
        )
        .into_response())
}
