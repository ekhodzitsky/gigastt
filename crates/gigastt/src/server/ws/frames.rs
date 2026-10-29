//! Per-frame WebSocket handlers (binary PCM, configure, stop, flush).

use super::super::config::SUPPORTED_RATES;
use super::super::json_text;
use anyhow::Result;
use axum::extract::ws::{Message as WsMessage, WebSocket};
use futures_util::SinkExt;
use gigastt_core::inference::{Engine, SessionTriplet};
use gigastt_core::protocol::ServerMessage;
use std::net::SocketAddr;
use std::sync::Arc;

/// Outcome returned by per-frame handlers. Keeps `handle_ws_inner` a thin
/// orchestration loop instead of a 250-line one-big-match.
pub(super) enum FrameOutcome {
    /// Continue consuming frames.
    Continue,
    /// Clean break — client asked to stop (Stop message) or the socket closed.
    Break,
}

pub(super) type WsSink = futures_util::stream::SplitSink<WebSocket, WsMessage>;

pub(super) struct StreamControl<'a> {
    pub abort: &'a Arc<std::sync::atomic::AtomicBool>,
    pub partial: &'a gigastt_core::inference::TranscriptSnapshot,
    pub shutdown: &'a tokio_util::sync::CancellationToken,
    pub disconnected: &'a tokio_util::sync::CancellationToken,
    pub timeout_secs: u64,
    pub deadline: tokio::time::Instant,
}

#[derive(Debug, PartialEq)]
enum StreamAbort {
    Cancelled,
    Shutdown,
    Timeout,
    SessionLimit,
}

async fn await_decode<T>(
    mut handle: tokio::task::JoinHandle<T>,
    control: &StreamControl<'_>,
) -> Result<Result<T, tokio::task::JoinError>, StreamAbort> {
    let timeout = async {
        if control.timeout_secs == 0 {
            std::future::pending::<()>().await;
        } else {
            tokio::time::sleep(std::time::Duration::from_secs(control.timeout_secs)).await;
        }
    };
    let reason = tokio::select! {
        biased;
        _ = control.shutdown.cancelled() => StreamAbort::Shutdown,
        _ = control.disconnected.cancelled() => StreamAbort::Cancelled,
        _ = tokio::time::sleep_until(control.deadline) => StreamAbort::SessionLimit,
        joined = &mut handle => return Ok(joined),
        _ = timeout => StreamAbort::Timeout,
    };
    // The encoder Run itself cannot be interrupted. Its owner publishes the
    // interrupted hypothesis and returns the triplet at the next decode step.
    control
        .abort
        .store(true, std::sync::atomic::Ordering::Relaxed);
    Err(reason)
}

async fn send_abort(
    sink: &mut WsSink,
    control: &StreamControl<'_>,
    reason: StreamAbort,
) -> Result<()> {
    let partial = control.partial.get();
    if let Some(segment) = &partial {
        send_server_message(sink, &ServerMessage::Partial(segment.clone())).await?;
    }
    let message = match reason {
        StreamAbort::Timeout => "Inference timed out.",
        StreamAbort::SessionLimit => "Maximum session duration exceeded",
        _ => "Transcription cancelled.",
    };
    let error = match reason {
        StreamAbort::Timeout => ServerMessage::Error {
            code: "inference_timeout".into(),
            message: message.into(),
            retry_after_ms: None,
        },
        StreamAbort::SessionLimit => ServerMessage::Error {
            code: "max_session_duration_exceeded".into(),
            message: message.into(),
            retry_after_ms: None,
        },
        _ => ServerMessage::Error {
            code: "cancelled".into(),
            message: message.into(),
            retry_after_ms: None,
        },
    };
    if matches!(reason, StreamAbort::Shutdown | StreamAbort::SessionLimit) {
        // The text goes out before the error so a client that stops reading
        // on `error` still has the committed utterance. `on_finalize`
        // partials publish an empty `committed` field; the truncated final
        // puts the recognized text back there.
        let segment = partial
            .map(gigastt_core::inference::TranscriptSegment::into_truncated_final)
            .unwrap_or_else(gigastt_core::inference::TranscriptSegment::empty_truncated_final);
        send_server_message(sink, &ServerMessage::Final(segment)).await?;
    }
    send_server_message(sink, &error).await?;
    sink.send(WsMessage::Close(Some(axum::extract::ws::CloseFrame {
        code: if reason == StreamAbort::SessionLimit {
            1008
        } else {
            1001
        },
        reason: message.into(),
    })))
    .await?;
    Ok(())
}

/// Send a serialized ServerMessage over the WebSocket sink. `?`-friendly so
/// handlers can delegate error propagation without duplicating the sink dance.
pub(super) async fn send_server_message(sink: &mut WsSink, msg: &ServerMessage) -> Result<()> {
    sink.send(WsMessage::Text(json_text(msg).into()))
        .await
        .map_err(Into::into)
}

/// Maximum number of empty binary frames accepted per WebSocket session.
/// Beyond this the connection is closed to prevent CPU / queue spam.
const MAX_EMPTY_FRAMES_PER_SESSION: usize = 1_000;

/// Handle a single PCM16 audio frame: resample if needed, run inference in a
/// `spawn_blocking` guarded by `catch_unwind`, and emit partial/final/error
/// payloads. The reservation is moved into the blocking closure and returned
/// on success; on spawn failure it is dropped inside the closure and the
/// triplet is returned to the pool automatically.
#[allow(clippy::too_many_arguments)]
pub(super) async fn handle_binary_frame(
    sink: &mut WsSink,
    engine: &Arc<Engine>,
    state_opt: &mut Option<gigastt_core::inference::StreamingState>,
    reservation: &mut Option<gigastt_core::inference::OwnedReservation<SessionTriplet>>,
    audio_received: &mut bool,
    empty_frame_count: &mut usize,
    client_sample_rate: u32,
    pending_byte: &mut Option<u8>,
    peer: SocketAddr,
    data: axum::body::Bytes,
    pcm_decode_buf: &mut Vec<f32>,
    metrics: Option<&Arc<super::super::metrics::MetricsRegistry>>,
    control: &StreamControl<'_>,
) -> Result<FrameOutcome> {
    if data.is_empty() {
        *empty_frame_count += 1;
        if *empty_frame_count > MAX_EMPTY_FRAMES_PER_SESSION {
            tracing::warn!("Empty binary frame spam from {peer}, closing connection");
            let err = ServerMessage::Error {
                message: "Empty frame limit exceeded".into(),
                code: "policy_violation".into(),
                retry_after_ms: None,
            };
            let _ = sink.send(WsMessage::Text(json_text(&err).into())).await;
            let _ = sink
                .send(WsMessage::Close(Some(axum::extract::ws::CloseFrame {
                    code: 1008,
                    reason: "policy violation".into(),
                })))
                .await;
            return Err(anyhow::anyhow!("Empty frame limit exceeded"));
        }
        tracing::debug!(
            "Empty binary frame from {peer}, skipping ({empty_frame_count}/{MAX_EMPTY_FRAMES_PER_SESSION})"
        );
        return Ok(FrameOutcome::Continue);
    }
    *audio_received = true;

    // Delegate carry-byte logic to the extracted pure function so it
    // can be property-tested independently of the async handler.
    gigastt_core::inference::audio::parse_pcm16_with_carry_into(
        &data,
        pending_byte,
        pcm_decode_buf,
    );
    if pending_byte.is_some() {
        tracing::warn!(
            "Odd-length PCM stream from {peer}: {} bytes, deferring 1 byte",
            data.len()
        );
    }
    let samples_16k = if client_sample_rate == 16000 {
        std::mem::take(pcm_decode_buf)
    } else {
        let state_ref = state_opt
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Streaming state lost"))?;
        gigastt_core::inference::audio::resample_with_cache(
            std::mem::take(pcm_decode_buf),
            gigastt_core::inference::audio::SampleRate(client_sample_rate),
            gigastt_core::inference::audio::SampleRate(16000),
            &mut state_ref.resampler,
            &mut state_ref.resample_output_buf,
        )?;
        std::mem::take(&mut state_ref.resample_output_buf)
    };

    let state = state_opt
        .take()
        .ok_or_else(|| anyhow::anyhow!("Streaming state lost"))?;
    let reservation_owned = reservation.take().ok_or_else(|| {
        tracing::error!("Reservation unexpectedly missing for {peer}");
        anyhow::anyhow!("Reservation lost")
    })?;

    let eng = engine.clone();
    let span = tracing::Span::current();
    let handle = tokio::task::spawn_blocking(move || {
        let _enter = span.enter();
        // Move ownership into the closure so state and reservation come back
        // unconditionally, including after a panic inside `process_chunk`.
        // Mirrors the pattern in src/server/http.rs.
        let mut state = state;
        let mut reservation = reservation_owned;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            eng.process_chunk(&samples_16k, &mut state, &mut reservation)
        }));
        (r, state, reservation)
    });

    let join_result = match await_decode(handle, control).await {
        Ok(joined) => joined,
        Err(reason) => {
            if reason == StreamAbort::Timeout
                && let Some(reg) = metrics
            {
                reg.counter_inc("gigastt_inference_timeouts_total", &[], 1);
            }
            send_abort(sink, control, reason).await?;
            return Ok(FrameOutcome::Break);
        }
    };

    match join_result {
        Ok((Ok(Ok(segments)), state_back, reservation_back)) => {
            *reservation = Some(reservation_back);
            *state_opt = Some(state_back);
            for seg in segments {
                let msg = if seg.is_final {
                    ServerMessage::Final(seg)
                } else {
                    ServerMessage::Partial(seg)
                };
                send_server_message(sink, &msg).await?;
            }
            Ok(FrameOutcome::Continue)
        }
        Ok((Ok(Err(e)), state_back, reservation_back)) => {
            *reservation = Some(reservation_back);
            *state_opt = Some(state_back);
            if matches!(e, gigastt_core::error::GigasttError::Cancelled) {
                send_abort(sink, control, StreamAbort::Cancelled).await?;
                return Ok(FrameOutcome::Break);
            }
            tracing::error!("Inference error for {peer}: {e:#}");
            send_server_message(
                sink,
                &ServerMessage::Error {
                    message: "Inference failed. Please check audio format.".into(),
                    code: "inference_error".into(),
                    retry_after_ms: None,
                },
            )
            .await?;
            Ok(FrameOutcome::Continue)
        }
        Ok((Err(_panic), state_back, reservation_back)) => {
            // Inference panicked: reservation is recovered, but the streaming
            // state (LSTM h/c buffers) may be mid-update and unsafe to reuse.
            // Drop it and install a fresh state so the session continues. The
            // per-session post-processing overrides are plain session policy
            // (never touched by inference), and configure-after-audio is
            // rejected, so they must survive the reset — the client has no
            // way to re-send them.
            tracing::error!(
                "Panic in WS inference for {peer} — triplet recovered, streaming state reset"
            );
            *reservation = Some(reservation_back);
            let mut fresh = engine.create_state(false);
            fresh.punctuation = state_back.punctuation;
            fresh.itn = state_back.itn;
            fresh.abort = state_back.abort;
            fresh.partial = state_back.partial;
            fresh.commit_policy = state_back.commit_policy;
            *state_opt = Some(fresh);
            send_server_message(
                sink,
                &ServerMessage::Error {
                    message: "Inference failed unexpectedly. Session reset.".into(),
                    code: "inference_panic".into(),
                    retry_after_ms: None,
                },
            )
            .await?;
            Ok(FrameOutcome::Continue)
        }
        Err(e) => {
            // spawn_blocking itself failed (runtime shutdown or cancellation).
            // The reservation was dropped inside the closure and the triplet
            // was returned to the pool automatically.
            tracing::error!("spawn_blocking join error for {peer}: {e}");
            Err(anyhow::anyhow!("Blocking task join failed"))
        }
    }
}

/// Lazy speaker loading (and waiting on another loader) must stay off Tokio
/// workers. Only the engine and the new state enter the blocking task: the
/// session retains its old state and pool reservation. If cancelled, native
/// loading may finish and populate the shared cache; its unused state is dropped
/// when the detached task completes.
#[cfg(feature = "diarization")]
async fn initialize_stream_state<T: Send + 'static>(
    create: impl FnOnce() -> T + Send + 'static,
    control: &StreamControl<'_>,
) -> Result<Result<T, tokio::task::JoinError>, StreamAbort> {
    let span = tracing::Span::current();
    let handle = tokio::task::spawn_blocking(move || {
        let _entered = span.enter();
        create()
    });
    await_decode(handle, control).await
}

/// Handle `{"type":"configure",…}`. Rejects configure-after-first-audio,
/// validates sample rate against `SUPPORTED_RATES`, (with diarization
/// feature) recreates the streaming state, and stores the per-session
/// post-processing overrides (`punctuation` / `itn`) on it.
#[allow(clippy::too_many_arguments)]
pub(super) async fn handle_configure_message(
    sink: &mut WsSink,
    engine: &Arc<Engine>,
    state_opt: &mut Option<gigastt_core::inference::StreamingState>,
    client_sample_rate: &mut u32,
    audio_received: bool,
    sample_rate: Option<u32>,
    diarization: Option<bool>,
    protocol_version: Option<String>,
    punctuation: Option<bool>,
    itn: Option<bool>,
    endpoint_mode: Option<String>,
    endpoint_silence_ms: Option<u32>,
    min_silence_ms: Option<u32>,
    commit_policy: Option<String>,
    peer: SocketAddr,
    control: &StreamControl<'_>,
) -> Result<FrameOutcome> {
    if audio_received {
        send_server_message(
            sink,
            &ServerMessage::Error {
                message: "Configure must be sent before first audio frame".into(),
                code: "configure_too_late".into(),
                retry_after_ms: None,
            },
        )
        .await?;
        return Ok(FrameOutcome::Continue);
    }
    if let Some(ref ver) = protocol_version
        && ![
            gigastt_core::protocol::MIN_PROTOCOL_VERSION,
            "1.1",
            "1.2",
            gigastt_core::protocol::PROTOCOL_VERSION,
        ]
        .contains(&ver.as_str())
    {
        send_server_message(
            sink,
            &ServerMessage::Error {
                message: format!(
                    "Unsupported protocol version: {ver}. Supported: {} through {}",
                    gigastt_core::protocol::MIN_PROTOCOL_VERSION,
                    gigastt_core::protocol::PROTOCOL_VERSION
                ),
                code: "unsupported_protocol_version".into(),
                retry_after_ms: None,
            },
        )
        .await?;
        return Ok(FrameOutcome::Break);
    }
    // Validate before changing any session fields or recreating its state.
    let endpoint_silence = match endpoint_silence_ms
        .map(gigastt_core::inference::EndpointSilence::new)
        .transpose()
    {
        Ok(duration) => duration,
        Err(error) => {
            send_server_message(
                sink,
                &ServerMessage::Error {
                    message: error.to_string(),
                    code: "invalid_endpoint_silence_ms".into(),
                    retry_after_ms: None,
                },
            )
            .await?;
            return Ok(FrameOutcome::Continue);
        }
    };
    let commit_policy = if let Some(token) = commit_policy {
        let Some(policy) = gigastt_core::inference::CommitPolicy::parse_token(&token) else {
            send_server_message(
                sink,
                &ServerMessage::Error {
                    message:
                        "Unsupported commit_policy. Supported: auto, on_finalize, stable_prefix"
                            .into(),
                    code: "invalid_commit_policy".into(),
                    retry_after_ms: None,
                },
            )
            .await?;
            return Ok(FrameOutcome::Continue);
        };
        Some(policy)
    } else {
        None
    };
    if let Some(rate) = sample_rate {
        if SUPPORTED_RATES.contains(&rate) {
            *client_sample_rate = rate;
            tracing::info!("Client {peer} configured sample rate: {rate}Hz");
        } else {
            send_server_message(
                sink,
                &ServerMessage::Error {
                    message: format!(
                        "Unsupported sample rate: {rate}Hz. Supported: {SUPPORTED_RATES:?}"
                    ),
                    code: "invalid_sample_rate".into(),
                    retry_after_ms: None,
                },
            )
            .await?;
        }
    }
    #[cfg(feature = "diarization")]
    if let Some(enable_dia) = diarization {
        tracing::info!("Client {peer} configured diarization: {enable_dia}");
        let engine = Arc::clone(engine);
        let mut new_state =
            match initialize_stream_state(move || engine.create_state(enable_dia), control).await {
                Ok(joined) => joined?,
                Err(reason) => {
                    send_abort(sink, control, reason).await?;
                    return Ok(FrameOutcome::Break);
                }
            };
        // The state is recreated wholesale; carry over any post-processing
        // overrides an earlier Configure already set on this session.
        if let Some(old) = state_opt.as_ref() {
            new_state.punctuation = old.punctuation;
            new_state.itn = old.itn;
            new_state.endpoint_mode = old.endpoint_mode;
            new_state.endpoint_silence = old.endpoint_silence;
            new_state.abort = old.abort.clone();
            new_state.partial = old.partial.clone();
            new_state.commit_policy = old.commit_policy;
        }
        *state_opt = Some(new_state);
    }
    #[cfg(not(feature = "diarization"))]
    let _ = (diarization, control);

    // Post-processing / endpoint overrides apply to whatever state the session
    // now holds (the diarization branch above may have just recreated it). An
    // absent field leaves the previous value, so repeated Configures compose
    // the same way `sample_rate` does.
    if let Some(state) = state_opt.as_mut() {
        if let Some(duration) = endpoint_silence {
            state.endpoint_silence = duration;
        }
        if let Some(policy) = commit_policy {
            state.commit_policy = policy;
        }
        if let Some(p) = punctuation {
            tracing::info!("Client {peer} configured punctuation: {p}");
            state.punctuation = Some(p);
        }
        if let Some(i) = itn {
            tracing::info!("Client {peer} configured itn: {i}");
            state.itn = Some(i);
        }
        if let Some(ref mode_s) = endpoint_mode {
            match gigastt_core::inference::EndpointMode::parse_token(mode_s) {
                Some(mode) => {
                    tracing::info!("Client {peer} configured endpoint_mode: {mode_s}");
                    state.endpoint_mode = mode;
                }
                None => {
                    send_server_message(
                        sink,
                        &ServerMessage::Error {
                            message: format!(
                                "Unsupported endpoint_mode: {mode_s}. Supported: auto, assistant, manual"
                            ),
                            code: "invalid_endpoint_mode".into(),
                            retry_after_ms: None,
                        },
                    )
                    .await?;
                    return Ok(FrameOutcome::Continue);
                }
            }
        }
        if let Some(ms) = min_silence_ms {
            if engine.has_vad() {
                let mut cfg = *engine.vad_config();
                cfg.min_silence_ms = ms;
                state.vad_endpointer = Some(gigastt_core::vad::VadEndpointer::new(&cfg));
                tracing::info!("Client {peer} configured min_silence_ms: {ms}");
            } else {
                tracing::debug!(
                    "Client {peer} set min_silence_ms={ms} but server has no VAD; ignored"
                );
            }
        }
    }
    let _ = engine;
    Ok(FrameOutcome::Continue)
}

/// Handle `{"type":"stop"}`. Flushes the streaming state, sends a final
/// segment (empty if there was nothing pending), and signals clean break.
///
/// The final ONNX decode runs in `spawn_blocking` + `catch_unwind`, matching
/// the binary-frame path — never on the async reactor worker.
pub(super) async fn handle_stop_message(
    sink: &mut WsSink,
    engine: &Arc<Engine>,
    state_opt: &mut Option<gigastt_core::inference::StreamingState>,
    reservation: &mut Option<gigastt_core::inference::OwnedReservation<SessionTriplet>>,
    peer: SocketAddr,
    control: &StreamControl<'_>,
) -> Result<FrameOutcome> {
    tracing::info!("Stop received from {peer}, finalizing");
    let Some(state) = state_opt.take() else {
        return Ok(FrameOutcome::Break);
    };
    // Move ownership into the blocking task so the triplet returns to the
    // pool even if `finish_stream` panics (mirrors `handle_binary_frame`).
    let eng = engine.clone();
    let reservation_owned = reservation.take();
    let span = tracing::Span::current();
    let handle = tokio::task::spawn_blocking(move || {
        let _enter = span.enter();
        let mut state = state;
        let mut reservation = reservation_owned;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            match reservation.as_mut() {
                // Final decode of audio buffered since the last strided decode
                // so trailing words aren't lost. A missing triplet cannot
                // establish that all pending audio was decoded.
                Some(res) => eng.try_finish_stream(&mut state, res),
                None => Err(gigastt_core::error::GigasttError::Cancelled),
            }
        }));
        (r, reservation)
    });

    let joined = match await_decode(handle, control).await {
        Ok(joined) => joined,
        Err(reason) => {
            send_abort(sink, control, reason).await?;
            return Ok(FrameOutcome::Break);
        }
    };
    let flush_seg = match joined {
        Ok((Ok(Ok(seg)), reservation_back)) => {
            drop(reservation_back);
            seg
        }
        Ok((Ok(Err(error)), reservation_back)) => {
            drop(reservation_back);
            tracing::error!("Final WS decode failed for {peer}: {error:#}");
            send_finish_error(sink, control.partial.get(), error.code()).await?;
            return Ok(FrameOutcome::Break);
        }
        Ok((Err(_panic), reservation_back)) => {
            drop(reservation_back);
            tracing::error!("Panic in WS final decode for {peer} — triplet recovered");
            send_finish_error(sink, control.partial.get(), "inference_panic").await?;
            return Ok(FrameOutcome::Break);
        }
        Err(error) => {
            tracing::error!("Blocking task join error on WS stop for {peer}: {error}");
            send_finish_error(sink, control.partial.get(), "inference_panic").await?;
            return Ok(FrameOutcome::Break);
        }
    };

    if control.abort.load(std::sync::atomic::Ordering::Relaxed) {
        send_abort(sink, control, StreamAbort::Cancelled).await?;
        return Ok(FrameOutcome::Break);
    }
    let final_msg = if let Some(seg) = flush_seg {
        ServerMessage::Final(seg)
    } else {
        ServerMessage::Final(gigastt_core::inference::TranscriptSegment::empty_final())
    };
    send_server_message(sink, &final_msg).await?;
    Ok(FrameOutcome::Break)
}

/// A failed stop preserves readable text but never masquerades as a successful final.
async fn send_finish_error(
    sink: &mut WsSink,
    partial: Option<gigastt_core::inference::TranscriptSegment>,
    code: &str,
) -> Result<()> {
    if let Some(partial) = partial.filter(|segment| !segment.text.is_empty()) {
        send_server_message(sink, &ServerMessage::Partial(partial)).await?;
    }
    send_server_message(
        sink,
        &ServerMessage::Error {
            code: code.into(),
            message: "Failed to finish transcription.".into(),
            retry_after_ms: None,
        },
    )
    .await?;
    sink.send(WsMessage::Close(Some(axum::extract::ws::CloseFrame {
        code: 1011,
        reason: "final transcription failed".into(),
    })))
    .await?;
    Ok(())
}

/// Flush any pending streaming state and emit a `Final` frame (even an empty
/// one) so e2e tests and clients can reliably assert that every session ends
/// with a Final before the Close. Used by the cancel and session-cap branches
/// of `handle_ws_inner`.
pub(super) async fn flush_and_final(
    sink: &mut WsSink,
    engine: &Arc<Engine>,
    state_opt: &mut Option<gigastt_core::inference::StreamingState>,
) -> Result<()> {
    let segment = match state_opt.as_mut() {
        Some(state) => engine.flush_truncated(state),
        None => gigastt_core::inference::TranscriptSegment::empty_truncated_final(),
    };
    send_server_message(sink, &ServerMessage::Final(segment)).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use tokio_util::sync::CancellationToken;

    #[cfg(feature = "diarization")]
    #[tokio::test]
    async fn test_configure_initialization_keeps_reactor_responsive() {
        // A current-thread runtime cannot service this timer if even the first
        // loader or a waiter on its lock runs synchronously on the reactor.
        let abort = Arc::new(AtomicBool::new(false));
        let partial = gigastt_core::inference::TranscriptSnapshot::default();
        let shutdown = CancellationToken::new();
        let disconnected = CancellationToken::new();
        let control = StreamControl {
            abort: &abort,
            partial: &partial,
            shutdown: &shutdown,
            disconnected: &disconnected,
            timeout_secs: 0,
            deadline: tokio::time::Instant::now() + std::time::Duration::from_secs(10),
        };
        let (release, gate) = std::sync::mpsc::channel();
        let gate = Arc::new(std::sync::Mutex::new(gate));
        let loaded = Arc::new(std::sync::OnceLock::new());
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let create = || {
            let gate = Arc::clone(&gate);
            let loaded = Arc::clone(&loaded);
            let attempts = Arc::clone(&attempts);
            move || {
                *loaded.get_or_init(|| {
                    attempts.fetch_add(1, Ordering::Relaxed);
                    // Bound the wait so the old synchronous implementation fails
                    // deterministically instead of deadlocking the test runtime.
                    gate.lock()
                        .unwrap()
                        .recv_timeout(std::time::Duration::from_secs(2))
                        .is_ok()
                })
            }
        };
        let timer = async {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            let _ = release.send(());
        };
        let (first, second, ()) = tokio::join!(
            initialize_stream_state(create(), &control),
            initialize_stream_state(create(), &control),
            timer,
        );
        assert!(first.unwrap().unwrap(), "loader blocked the reactor timer");
        assert!(second.unwrap().unwrap());
        assert_eq!(attempts.load(Ordering::Relaxed), 1);
    }

    #[cfg(feature = "diarization")]
    #[tokio::test]
    async fn test_configure_initialization_cancellation_drops_detached_result() {
        struct LoadedState(Arc<AtomicBool>);
        impl Drop for LoadedState {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Relaxed);
            }
        }
        for disconnect in [false, true] {
            let abort = Arc::new(AtomicBool::new(false));
            let partial = gigastt_core::inference::TranscriptSnapshot::default();
            let shutdown = CancellationToken::new();
            let disconnected = CancellationToken::new();
            let control = StreamControl {
                abort: &abort,
                partial: &partial,
                shutdown: &shutdown,
                disconnected: &disconnected,
                timeout_secs: 0,
                deadline: tokio::time::Instant::now() + std::time::Duration::from_secs(10),
            };
            let dropped = Arc::new(AtomicBool::new(false));
            let loaded = LoadedState(Arc::clone(&dropped));
            let (started, ready) = tokio::sync::oneshot::channel();
            let (release, gate) = std::sync::mpsc::channel();
            let cancel = async {
                ready.await.unwrap();
                if disconnect {
                    disconnected.cancel();
                } else {
                    shutdown.cancel();
                }
            };
            let initialize = initialize_stream_state(
                move || {
                    let _ = started.send(());
                    let _ = gate.recv_timeout(std::time::Duration::from_secs(2));
                    loaded
                },
                &control,
            );
            let (result, ()) = tokio::join!(initialize, cancel);
            assert!(
                matches!(result, Err(StreamAbort::Cancelled)) && disconnect
                    || matches!(result, Err(StreamAbort::Shutdown)) && !disconnect
            );
            assert!(
                !dropped.load(Ordering::Relaxed),
                "native load still owns its result"
            );
            release.send(()).unwrap();
            tokio::time::timeout(std::time::Duration::from_secs(2), async {
                while !dropped.load(Ordering::Relaxed) {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
        }
    }

    #[tokio::test(start_paused = true)]
    async fn test_stream_abort_sources_stop_in_flight_work() {
        for expected in [
            StreamAbort::Timeout,
            StreamAbort::Shutdown,
            StreamAbort::Cancelled,
            StreamAbort::SessionLimit,
        ] {
            let abort = Arc::new(AtomicBool::new(false));
            let partial = gigastt_core::inference::TranscriptSnapshot::default();
            let shutdown = CancellationToken::new();
            let disconnected = CancellationToken::new();
            let (tx, rx) = tokio::sync::oneshot::channel();
            let flag = abort.clone();
            let handle = tokio::spawn(async move {
                while !flag.load(Ordering::Relaxed) {
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
                let _ = tx.send(());
            });
            if expected == StreamAbort::Shutdown {
                shutdown.cancel();
            }
            if expected == StreamAbort::Cancelled {
                disconnected.cancel();
            }
            let control = StreamControl {
                abort: &abort,
                partial: &partial,
                shutdown: &shutdown,
                disconnected: &disconnected,
                timeout_secs: if expected == StreamAbort::Timeout {
                    1
                } else {
                    0
                },
                deadline: tokio::time::Instant::now()
                    + std::time::Duration::from_secs(if expected == StreamAbort::SessionLimit {
                        1
                    } else {
                        3600
                    }),
            };
            assert_eq!(await_decode(handle, &control).await.unwrap_err(), expected);
            assert!(abort.load(Ordering::Relaxed));
            tokio::time::timeout(std::time::Duration::from_secs(1), rx)
                .await
                .unwrap()
                .unwrap();
        }
    }
}
