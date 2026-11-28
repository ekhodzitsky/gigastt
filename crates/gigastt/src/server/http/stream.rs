//! Server-Sent Events streaming transcription (`POST /v1/transcribe/stream`).

use axum::body::Bytes;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use futures_util::StreamExt;
use futures_util::stream::Stream;
use gigastt_core::inference::CommitPolicy;
use std::sync::Arc;

use super::error::{ApiError, api_error};
use super::state::AppState;
use super::transcribe::reserve_batch_slot;

/// Samples per `process_chunk` call on file-stream paths (native SSE and
/// OpenAI `stream=true`): one second at 16 kHz. Frame-aligned, and pinned
/// because the streaming recognizer's state — and so the emitted segments —
/// depend on the chunk cadence.
pub(super) const STREAM_CHUNK_SAMPLES: usize = 16_000;

/// Maximum time a file-stream producer waits for a reader to drain its queue.
pub(super) const STREAM_SEND_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Keep FIFO delivery bounded without pinning an inference reservation behind
/// an unread response. On cancellation a final/error can still be queued if
/// capacity is immediately available; a full queue is never waited on again.
/// The token must belong to this request, not the server shutdown token.
pub(super) async fn send_stream_item<T>(
    tx: &tokio::sync::mpsc::Sender<T>,
    item: T,
    cancel: &tokio_util::sync::CancellationToken,
    abort: &std::sync::atomic::AtomicBool,
) -> bool {
    use tokio::sync::mpsc::error::TrySendError;
    let delivered = match tx.try_send(item) {
        Ok(()) => return true,
        Err(TrySendError::Closed(_)) => false,
        Err(TrySendError::Full(item)) => {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => false,
                result = tokio::time::timeout(STREAM_SEND_TIMEOUT, tx.send(item)) => {
                    match result {
                        Ok(result) => result.is_ok(),
                        Err(_) => {
                            tracing::warn!("SSE reader stalled; cancelling transcription");
                            false
                        }
                    }
                }
            }
        }
    };
    if !delivered {
        abort.store(true, std::sync::atomic::Ordering::Relaxed);
        cancel.cancel();
    }
    delivered
}

/// Per-request text commitment for the native SSE endpoint.
#[derive(Default, serde::Deserialize)]
pub struct StreamQuery {
    commit_policy: Option<String>,
}

/// Probe the container (optional duration ceiling) and open a lazy
/// [`AudioChunks`] iterator. Shared by native SSE and OpenAI stream so both
/// reserve a pool slot before expanding audio and never materialize full PCM.
pub(super) fn open_stream_chunks_blocking(
    body: Bytes,
    max_audio_secs: Option<f64>,
) -> Result<gigastt_core::inference::audio::AudioChunks, anyhow::Error> {
    // Enforce an operator length limit up front where the container
    // declares its duration (WAV / FLAC / M4A / OGG), so `--max-audio-secs`
    // keeps answering a clean 413 instead of tripping mid-stream. The
    // incremental budget inside the decode remains the backstop for
    // containers that declare nothing.
    if let Some(limit) = max_audio_secs
        && let Ok(Some(declared)) =
            gigastt_core::inference::audio::probe_duration_bytes(body.clone())
        && declared > limit
    {
        return Err(anyhow::Error::from(
            gigastt_core::error::GigasttError::AudioTooLong {
                observed_secs: declared,
                limit_secs: limit,
            },
        ));
    }
    gigastt_core::inference::audio::AudioChunks::from_bytes(
        body,
        STREAM_CHUNK_SAMPLES,
        max_audio_secs,
    )
}

/// Keep admission and upload ownership inside the tracked worker, including
/// after a cancelled handler or probe timeout has returned its HTTP response.
pub(super) async fn open_stream_chunks(
    state: &AppState,
    body: Bytes,
    reservation: gigastt_core::inference::OwnedReservation<gigastt_core::inference::SessionTriplet>,
    max_audio_secs: Option<f64>,
    timeout_secs: u64,
) -> Result<
    (
        gigastt_core::inference::audio::AudioChunks,
        Bytes,
        gigastt_core::inference::OwnedReservation<gigastt_core::inference::SessionTriplet>,
    ),
    ApiError,
> {
    use super::super::transcription_control::{
        AbortOnDrop, WatchdogOutcome, await_transcription_watchdog,
    };
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    let abort = Arc::new(AtomicBool::new(state.shutdown.is_cancelled()));
    let _cancel_on_drop = AbortOnDrop(abort.clone());
    let worker_abort = abort.clone();
    let handle = state.tracker.spawn_blocking(move || {
        let upload_lifetime = body.clone();
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if worker_abort.load(Ordering::Relaxed) {
                return Err(anyhow::Error::from(
                    gigastt_core::error::GigasttError::Cancelled,
                ));
            }
            let chunks = open_stream_chunks_blocking(body, max_audio_secs)?;
            if worker_abort.load(Ordering::Relaxed) {
                return Err(anyhow::Error::from(
                    gigastt_core::error::GigasttError::Cancelled,
                ));
            }
            Ok((chunks, upload_lifetime, reservation))
        }))
        .unwrap_or_else(|_| Err(anyhow::anyhow!("Audio decode thread panicked")))
    });
    match await_transcription_watchdog(
        handle,
        &AtomicU64::new(0),
        &abort,
        timeout_secs,
        &state.shutdown,
    )
    .await
    {
        WatchdogOutcome::TimedOut => {
            if let Some(metrics) = &state.metrics_registry {
                metrics.counter_inc("gigastt_inference_timeouts_total", &[], 1);
            }
            Err(super::error::api_inference_timeout_error(None))
        }
        WatchdogOutcome::Joined(result) => result
            .map_err(|error| {
                tracing::error!("SSE probe join error: {error}");
                api_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Internal server error",
                    "internal",
                )
            })?
            .map_err(map_stream_open_error),
    }
}

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
                "committed": seg.committed,
                "tentative": seg.tentative,
                "timestamp": seg.timestamp,
                "words": seg.words,
            });
            // Same omission contract as the WS segment payload: no words →
            // no `confidence` key at all.
            if let Some(confidence) = seg.confidence {
                payload["confidence"] = confidence.into();
            }
            // Same omission as the WebSocket segment: absent means not cut.
            if seg.truncated {
                payload["truncated"] = true.into();
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
    Query(query): Query<StreamQuery>,
    body: Bytes,
) -> Result<Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>>, ApiError> {
    let commit_policy = query
        .commit_policy
        .as_deref()
        .map_or(Some(CommitPolicy::Auto), CommitPolicy::parse_token)
        .ok_or_else(|| {
            api_error(
                StatusCode::BAD_REQUEST,
                "Unsupported commit_policy. Supported: auto, on_finalize, stable_prefix",
                "invalid_commit_policy",
            )
        })?;
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
    let reservation = reserve_batch_slot(&engine, &limits, state.metrics_registry.as_ref()).await?;
    let (chunks, upload_lifetime, mut reservation) = open_stream_chunks(
        &state,
        body,
        reservation,
        limits.max_audio_secs_opt(),
        limits.inference_timeout_secs,
    )
    .await?;

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
    // chunk's segments out as they are produced. The independent watchdog
    // closes stalled responses even during a native call. Output waits retain
    // their separate bounded backpressure policy.
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
        let runtime = tokio::runtime::Handle::current();
        let send = |item| runtime.block_on(watchdog.send(&tx, item, &cancel, &abort));
        let report_error = |code, message: &str| {
            if !watchdog.claim_finish() {
                return;
            }
            if let Some(segment) = partial.get()
                && !send(Ok(segment))
            {
                return;
            }
            let _ = send(Err(StreamError {
                code,
                message: message.into(),
            }));
        };
        // catch_unwind ensures the triplet is returned to the pool even on panic.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut stream_state = engine.create_state(false);
            stream_state.commit_policy = commit_policy;
            stream_state.abort = Some(abort.clone());
            stream_state.partial = Some(partial.clone());
            let mut chunks = chunks;

            // Fixed-size chunks (STREAM_CHUNK_SAMPLES), last one short — so the
            // streaming state advances with a stable cadence.
            loop {
                if cancel.is_cancelled() {
                    tracing::info!("SSE transcription cancelled by shutdown");
                    if !watchdog.claim_finish() {
                        return;
                    }
                    let segment = engine.flush_truncated(&mut stream_state);
                    let _ = send(Ok(segment));
                    return;
                }
                let chunk = match chunks.next_chunk() {
                    Ok(Some(c)) => c,
                    Ok(None) => break,
                    Err(e) => {
                        // A decode failure past the header (corrupt packet, or a
                        // length budget tripping on a container that declared no
                        // duration) can no longer be an HTTP status — the stream
                        // is already open — so it ends the stream with the same
                        // machine-readable code the status would have carried.
                        let code = e
                            .downcast_ref::<gigastt_core::error::GigasttError>()
                            .map_or("invalid_audio", |g| g.code());
                        tracing::error!("SSE audio decode error: {e:#}");
                        report_error(code, "Failed to decode audio file. Check format.");
                        return;
                    }
                };
                match engine.process_chunk(chunk, &mut stream_state, &mut reservation) {
                    Ok(segs) => {
                        watchdog.progress();
                        for seg in segs {
                            if !send(Ok(seg)) {
                                // Reader disconnected, stalled, or shutdown interrupted the send.
                                return;
                            }
                        }
                    }
                    Err(e) => {
                        report_error(e.code(), "Transcription failed. Please check audio format.");
                        return;
                    }
                }
            }

            // Final decode of the sub-stride remainder, then flush. Terminal
            // delivery follows the same bounded wait as ordinary segments.
            match engine.try_finish_stream(&mut stream_state, &mut reservation) {
                Ok(Some(segment)) => {
                    if !watchdog.claim_finish() {
                        return;
                    }
                    let _ = send(Ok(segment));
                }
                Ok(None) => {
                    let _ = watchdog.claim_finish();
                }
                Err(error) => {
                    tracing::error!("Final SSE decode failed: {error:#}");
                    report_error(error.code(), "Failed to finish transcription.");
                }
            }
        }));

        if result.is_err() {
            tracing::error!("Panic in SSE inference task — triplet recovered");
            // Mirror the WebSocket contract: surface a distinct `inference_panic`
            // code instead of ending the stream silently.
            report_error("inference_panic", "Inference failed unexpectedly.");
        }
        // reservation dropped here automatically returns the triplet to the pool
    });

    // Convert receiver to SSE stream.
    let stream = response_watchdog
        .response(rx, move || {
            let mut items = Vec::new();
            if let Some(mut segment) = response_partial.get() {
                segment.is_final = false;
                items.push(Ok(segment));
            }
            items.push(Err(StreamError {
                code: "inference_timeout",
                message: "Transcription made no progress within the configured timeout.".into(),
            }));
            items
        })
        .map(|result| Ok(Event::default().data(sse_data_payload(&result))));

    // Explicit keep-alive: send a comment (`: \n\n`) every 15 s so nginx / ALB
    // do not close the connection during long transcriptions.
    Ok(Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text(""),
    ))
}

/// Map probe/open failures for file-stream endpoints to HTTP errors.
pub(super) fn map_stream_open_error(e: anyhow::Error) -> ApiError {
    if matches!(
        e.downcast_ref::<gigastt_core::error::GigasttError>(),
        Some(gigastt_core::error::GigasttError::Cancelled)
    ) {
        return api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Server is shutting down",
            "cancelled",
        );
    }
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
        "Failed to decode audio file. Check format.",
        "invalid_audio",
    )
}
