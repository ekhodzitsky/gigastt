//! Server-Sent Events streaming transcription (`POST /v1/transcribe/stream`).

use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use futures_util::StreamExt;
use futures_util::stream::Stream;
use std::sync::Arc;

use super::error::{ApiError, api_error};
use super::state::AppState;
use super::transcribe::reserve_batch_slot;

/// Per-segment error carried over the SSE channel: a stable machine-readable
/// code plus a sanitized message, mirroring the WebSocket error contract so
/// SSE clients get the same codes (`inference_error`, `inference_panic`,
/// `inference_timeout`, …) instead of one generic string.
pub(super) struct StreamError {
    pub(super) code: &'static str,
    pub(super) message: String,
}

/// Render one SSE segment-or-error result to the JSON payload string sent in
/// the `data:` field. Pure (no I/O) so the per-variant error `code`, the
/// `inference_panic` / `inference_timeout` events, and the partial/final
/// framing can be unit-tested without a model.
pub(super) fn sse_data_payload(
    result: &Result<gigastt_core::inference::TranscriptSegment, StreamError>,
) -> String {
    match result {
        Ok(seg) => {
            let ty = if seg.is_final { "final" } else { "partial" };
            let mut payload = serde_json::json!({
                "type": ty,
                "text": seg.text,
                "timestamp": seg.timestamp,
                "words": seg.words,
            });
            // Same omission contract as the WS segment payload: no words →
            // no `confidence` key at all.
            if let Some(confidence) = seg.confidence {
                payload["confidence"] = confidence.into();
            }
            payload.to_string()
        }
        Err(err) => serde_json::json!({
            "type": "error",
            "message": err.message,
            "code": err.code,
        })
        .to_string(),
    }
}

/// POST /v1/transcribe/stream — upload audio file, get SSE stream of partial/final results.
///
/// Real streaming: audio is processed chunk-by-chunk inside `spawn_blocking`,
/// and segments are sent to the SSE stream via an mpsc channel as they are produced.
pub async fn transcribe_stream(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Result<Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>>, ApiError> {
    if body.is_empty() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "Empty request body",
            "empty_body",
        ));
    }

    // Defence-in-depth early reject; matches `/v1/transcribe` — see that
    // handler for the rationale.
    let limits = state.limits.load();
    if body.len() > limits.body_limit_bytes {
        return Err(api_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "Request body exceeds the configured size limit",
            "payload_too_large",
        ));
    }

    // Checkout a session triplet from the batch pool BEFORE decoding. SSE file
    // transcription decodes and transcribes the *entire* upload (holding the
    // triplet for the whole file), so it is a batch workload, not interactive
    // streaming. Reserving first — via the same `reserve_batch_slot` the
    // synchronous `/v1/transcribe` uses — is load-bearing: it caps the number of
    // *concurrent decodes* at the pool size, so a burst of large (compressed)
    // uploads can't each expand into a full f32 PCM buffer at once and exhaust
    // memory. A saturated pool yields 503 / `Retry-After` here, before any decode.
    // Snapshot the current engine once; a concurrent hot-reload swap only affects
    // later requests, so this stream rides the pool it started on.
    let engine = state.engine.load_full();
    let mut reservation =
        reserve_batch_slot(&engine, &limits, state.metrics_registry.as_ref()).await?;

    // Decode audio now that a slot is reserved (in spawn_blocking since symphonia
    // is blocking). Holding the reservation across the decode is what bounds the
    // number of concurrent PCM expansions to the pool size. `body` is
    // `axum::body::Bytes`, so the move into the blocking closure is a refcount
    // bump and `decode_audio_bytes_shared` reads the upload buffer in place. On a
    // decode error the early `?` return drops `reservation`, returning the
    // triplet to the pool.
    // SSE materializes the whole buffer before chunking, so it is a whole-buffer
    // path: bounded by the operator `--max-audio-secs` (or the fixed safety
    // ceiling when unset), consistent with the single-shot REST path.
    let max_audio_secs = limits.max_audio_secs_opt();
    let samples = tokio::task::spawn_blocking(move || {
        // catch_unwind mirrors the REST handler: a panic inside the blocking
        // decode (e.g. a crafted container that trips an upstream arithmetic
        // panic) is absorbed and surfaced as a normal decode error instead of a
        // `JoinError`, so the SSE path returns a clean 422 rather than a 500.
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gigastt_core::inference::audio::decode_audio_bytes_shared_bounded(body, max_audio_secs)
        })) {
            Ok(inner) => inner,
            Err(_) => {
                tracing::error!("Panic in SSE audio decode — treated as decode error");
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
        // "Too long" is distinct from "corrupt": answer 413 `audio_too_long`
        // with the observed/limit seconds instead of the generic 422.
        if let Some(g @ gigastt_core::error::GigasttError::AudioTooLong { .. }) =
            e.downcast_ref::<gigastt_core::error::GigasttError>()
        {
            return api_error(StatusCode::PAYLOAD_TOO_LARGE, &g.to_string(), g.code());
        }
        tracing::error!("Audio decode error: {e:#}");
        api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "Failed to decode audio file. Check format (WAV, MP3, M4A, OGG, FLAC supported).",
            "invalid_audio",
        )
    })?;

    // Create mpsc channel for streaming segments from the inference task to SSE.
    let (tx, rx) = tokio::sync::mpsc::channel::<
        Result<gigastt_core::inference::TranscriptSegment, StreamError>,
    >(16);

    // The axum handler future has already returned by the time the SSE stream
    // starts flowing, so `with_graceful_shutdown` can't observe this task. Clone
    // the shutdown token and check it before every chunk so SIGTERM during a
    // long transcription drops cleanly.
    //
    // The whole file is transcribed in one blocking task, streaming each 1 s
    // chunk's segments out as they are produced. Each `process_chunk` is a small
    // bounded unit of work, so unlike the single-shot REST path it is not
    // wrapped by the per-request inference timeout; liveness on shutdown is
    // handled by the per-chunk cancellation check.
    let cancel = state.shutdown.clone();
    let tracker = state.tracker.clone();
    let span = tracing::Span::current();
    tracker.spawn_blocking(move || {
        let _enter = span.enter();
        // catch_unwind ensures the triplet is returned to the pool even on panic.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut stream_state = engine.create_state(false);
            let chunk_size = 16000; // 1 second at 16 kHz

            for chunk in samples.chunks(chunk_size) {
                if cancel.is_cancelled() {
                    tracing::info!("SSE transcription cancelled by shutdown");
                    return;
                }
                match engine.process_chunk(chunk, &mut stream_state, &mut reservation) {
                    Ok(segs) => {
                        for seg in segs {
                            if tx.blocking_send(Ok(seg)).is_err() {
                                // Receiver dropped (client disconnected).
                                return;
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.blocking_send(Err(StreamError {
                            code: e.code(),
                            message: "Transcription failed. Please check audio format.".into(),
                        }));
                        return;
                    }
                }
            }

            // Final decode of the sub-stride remainder, then flush — best-effort;
            // always emit so SSE clients receive a clean end-of-stream marker.
            if let Some(seg) = engine.finish_stream(&mut stream_state, &mut reservation) {
                let _ = tx.blocking_send(Ok(seg));
            }
        }));

        if result.is_err() {
            tracing::error!("Panic in SSE inference task — triplet recovered");
            // Mirror the WebSocket contract: surface a distinct `inference_panic`
            // code instead of ending the stream silently.
            let _ = tx.blocking_send(Err(StreamError {
                code: "inference_panic",
                message: "Inference failed unexpectedly.".into(),
            }));
        }
        // reservation dropped here automatically returns the triplet to the pool
    });

    // Convert receiver to SSE stream.
    let stream = tokio_stream::wrappers::ReceiverStream::new(rx)
        .map(|result| Ok(Event::default().data(sse_data_payload(&result))));

    // Explicit keep-alive: send a comment (`: \n\n`) every 15 s so nginx / ALB
    // do not close the connection during long transcriptions.
    Ok(Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text(""),
    ))
}
