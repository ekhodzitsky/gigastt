//! WebSocket handler: upgrade, session loop, PCM16 processing, and inference dispatch.

use std::net::SocketAddr;
use std::sync::Arc;
use anyhow::Result;
use axum::extract::State;
use axum::extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade};
use axum::response::Response;
use futures_util::{SinkExt, StreamExt};
use super::config::{DEFAULT_SAMPLE_RATE, RuntimeLimits, SUPPORTED_RATES, pool_retry_after_ms};
use super::http;
use super::json_text;
use crate::inference::{Engine, SessionTriplet};
use crate::protocol::{ClientMessage, ServerMessage};

/// Outcome returned by per-frame handlers. Keeps `handle_ws_inner` a thin
/// orchestration loop instead of a 250-line one-big-match.
enum FrameOutcome {
    /// Continue consuming frames.
    Continue,
    /// Clean break — client asked to stop (Stop message) or the socket closed.
    Break,
}

type WsSink = futures_util::stream::SplitSink<WebSocket, WsMessage>;

pub(super) async fn ws_handler(
    ws: WebSocketUpgrade,
    axum::extract::ConnectInfo(peer): axum::extract::ConnectInfo<SocketAddr>,
    State(state): State<Arc<http::AppState>>,
    headers: axum::http::HeaderMap,
) -> Response {
    // Origin allowlist is enforced by `origin_middleware` before the request
    // reaches this handler; anything that arrives here has already been cleared.
    //
    // V1-03: if shutdown has already been requested, refuse the upgrade
    // instead of handing the client a socket we're about to drain. Returning
    // a plain 503 with the `shutting_down` error code keeps the surface
    // consistent with the pool-saturation 503.
    if state.shutdown.is_cancelled() {
        use axum::http::StatusCode;
        use axum::response::IntoResponse;
        tracing::warn!(peer = %peer, "Rejecting WS upgrade after shutdown");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            axum::response::Json(serde_json::json!({
                "error": "Server shutting down",
                "code": "shutting_down",
            })),
        )
            .into_response();
    }

    let request_id = headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown")
        .to_string();

    let max_bytes = state.limits.load().ws_frame_max_bytes;
    let state_cloned = state.clone();
    ws.max_message_size(max_bytes)
        .max_frame_size(max_bytes)
        .on_upgrade(move |socket| {
            use tracing::Instrument;
            let span = tracing::info_span!("ws_session", request_id = %request_id, peer = %peer);
            async move {
                state_cloned
                    .tracker
                    .clone()
                    .track_future(handle_ws(socket, peer, state_cloned.clone()))
                    .await
            }
            .instrument(span)
        })
}

async fn handle_ws(socket: WebSocket, peer: SocketAddr, state: Arc<http::AppState>) {
    if let Some(ref reg) = state.metrics_registry {
        reg.gauge_inc("gigastt_ws_active_connections", vec![], 1);
    }
    struct WsMetricsGuard(Arc<http::AppState>);
    impl Drop for WsMetricsGuard {
        fn drop(&mut self) {
            if let Some(ref reg) = self.0.metrics_registry {
                reg.gauge_inc("gigastt_ws_active_connections", vec![], -1);
            }
        }
    }
    let _ws_guard = WsMetricsGuard(state.clone());
    let checkout_start = std::time::Instant::now();
    // `select!` the pool checkout against the shutdown token so SIGTERM
    // during pool saturation returns immediately instead of waiting the full
    // checkout window. `biased;` keeps cancel priority over progress.
    let guard = tokio::select! {
        biased;
        _ = state.shutdown.cancelled() => {
            tracing::info!(peer = %peer, "Shutdown requested before pool checkout");
            let (mut sink, _) = socket.split();
            let _ = sink
                .send(WsMessage::Close(Some(axum::extract::ws::CloseFrame {
                    code: 1001,
                    reason: "server shutdown".into(),
                })))
                .await;
            return;
        }
        res = tokio::time::timeout(
            std::time::Duration::from_secs(state.limits.load().pool_checkout_timeout_secs),
            state.engine.pool.checkout(),
        ) => match res {
            Ok(Ok(guard)) => guard,
            Ok(Err(_pool_closed)) => {
                tracing::info!("WebSocket pool closed for {peer} — server is shutting down");
                let (mut sink, _) = socket.split();
                let err = ServerMessage::Error {
                    message: "Server is shutting down".into(),
                    code: "pool_closed".into(),
                    retry_after_ms: None,
                };
                let _ = sink.send(WsMessage::Text(json_text(&err).into())).await;
                return;
            }
            Err(_) => {
                tracing::warn!("WebSocket pool checkout timeout for {peer}");
                if let Some(ref reg) = state.metrics_registry {
                    reg.counter_inc("gigastt_pool_timeouts_total", vec![], 1);
                    reg.histogram_record("gigastt_pool_checkout_duration_seconds", vec![], checkout_start.elapsed().as_secs_f64());
                }
                let (mut sink, _) = socket.split();
                let limits = state.limits.load();
                let err = ServerMessage::Error {
                    message: "Server busy, try again later".into(),
                    code: "timeout".into(),
                    retry_after_ms: Some(pool_retry_after_ms(&limits)),
                };
                let _ = sink.send(WsMessage::Text(json_text(&err).into())).await;
                return;
            }
        }
    };

    if let Some(ref reg) = state.metrics_registry {
        reg.histogram_record(
            "gigastt_pool_checkout_duration_seconds",
            vec![],
            checkout_start.elapsed().as_secs_f64(),
        );
    }
    // Strip the lifetime so the triplet can travel through `handle_ws_inner`,
    // which currently owns it directly. The reservation handles checkin on
    // the way back; if the inner loop loses the triplet to a spawn_blocking
    // panic, the reservation is dropped without sending and the pool
    // degrades gracefully — matches the pre-rewrite contract.
    let (triplet, reservation) = guard.into_owned();

    let limits = state.limits.load();
    let (triplet_opt, result) = handle_ws_inner(
        socket,
        peer,
        &state.engine,
        &limits,
        triplet,
        state.shutdown.clone(),
    )
    .await;
    if let Err(e) = result {
        tracing::error!("WebSocket error from {peer}: {e}");
    }

    if let Some(triplet) = triplet_opt {
        reservation.checkin(triplet);
    }
    // If triplet_opt is None, the triplet was lost due to a spawn_blocking panic.
    // The pool degrades gracefully with fewer available slots.
}

/// Send a serialized ServerMessage over the WebSocket sink. `?`-friendly so
/// handlers can delegate error propagation without duplicating the sink dance.
async fn send_server_message(sink: &mut WsSink, msg: &ServerMessage) -> Result<()> {
    sink.send(WsMessage::Text(json_text(msg).into()))
        .await
        .map_err(Into::into)
}

/// Handle a single PCM16 audio frame: resample if needed, run inference in a
/// `spawn_blocking` guarded by `catch_unwind`, and emit partial/final/error
/// payloads. Always returns the triplet to `triplet_opt` (or recovers a fresh
/// state after an inference panic) so the connection can keep serving.
#[allow(clippy::too_many_arguments)]
async fn handle_binary_frame(
    sink: &mut WsSink,
    engine: &Arc<Engine>,
    state_opt: &mut Option<crate::inference::StreamingState>,
    triplet_opt: &mut Option<SessionTriplet>,
    audio_received: &mut bool,
    client_sample_rate: u32,
    pending_byte: &mut Option<u8>,
    peer: SocketAddr,
    data: axum::body::Bytes,
) -> Result<FrameOutcome> {
    if data.is_empty() {
        tracing::debug!("Empty binary frame from {peer}, skipping");
        return Ok(FrameOutcome::Continue);
    }
    *audio_received = true;

    // V1-25: delegate carry-byte logic to the extracted pure function so it
    // can be property-tested independently of the async handler.
    let samples_f32 = crate::inference::audio::parse_pcm16_with_carry(&data, pending_byte);
    if pending_byte.is_some() {
        tracing::warn!(
            "Odd-length PCM stream from {peer}: {} bytes, deferring 1 byte",
            data.len()
        );
    }
    let samples_16k = if client_sample_rate == 16000 {
        samples_f32
    } else {
        let state_ref = state_opt
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Streaming state lost"))?;
        crate::inference::audio::resample_with_cache(
            &samples_f32,
            crate::inference::audio::SampleRate(client_sample_rate),
            crate::inference::audio::SampleRate(16000),
            &mut state_ref.resampler,
        )?
    };

    let state = state_opt
        .take()
        .ok_or_else(|| anyhow::anyhow!("Streaming state lost"))?;
    let triplet = triplet_opt.take().ok_or_else(|| {
        tracing::error!("Triplet unexpectedly missing for {peer}");
        anyhow::anyhow!("Triplet lost")
    })?;

    let eng = engine.clone();
    let join_result = tokio::task::spawn_blocking(move || {
        // Move ownership into the closure so state and triplet come back
        // unconditionally, including after a panic inside `process_chunk`.
        // Mirrors the pattern in src/server/http.rs.
        let mut state = state;
        let mut triplet = triplet;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            eng.process_chunk(&samples_16k, &mut state, &mut triplet)
        }));
        (r, state, triplet)
    })
    .await;

    match join_result {
        Ok((Ok(Ok(segments)), state_back, triplet_back)) => {
            *state_opt = Some(state_back);
            *triplet_opt = Some(triplet_back);
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
        Ok((Ok(Err(e)), state_back, triplet_back)) => {
            *state_opt = Some(state_back);
            *triplet_opt = Some(triplet_back);
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
        Ok((Err(_panic), _state_back, triplet_back)) => {
            // Inference panicked: triplet is recovered, but the streaming
            // state (LSTM h/c buffers) may be mid-update and unsafe to reuse.
            // Drop it and install a fresh state so the session continues.
            tracing::error!(
                "Panic in WS inference for {peer} — triplet recovered, streaming state reset"
            );
            *triplet_opt = Some(triplet_back);
            *state_opt = Some(engine.create_state(false));
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
            // Triplet is truly lost in this branch; bail out.
            tracing::error!("spawn_blocking join error for {peer}: {e}");
            Err(anyhow::anyhow!("Blocking task join failed"))
        }
    }
}

/// Handle `{"type":"configure",…}`. Rejects configure-after-first-audio,
/// validates sample rate against `SUPPORTED_RATES`, and (with diarization
/// feature) recreates the streaming state.
#[allow(clippy::too_many_arguments)]
async fn handle_configure_message(
    sink: &mut WsSink,
    engine: &Arc<Engine>,
    state_opt: &mut Option<crate::inference::StreamingState>,
    client_sample_rate: &mut u32,
    audio_received: bool,
    sample_rate: Option<u32>,
    diarization: Option<bool>,
    protocol_version: Option<String>,
    peer: SocketAddr,
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
        && ver != crate::protocol::PROTOCOL_VERSION
    {
        send_server_message(
            sink,
            &ServerMessage::Error {
                message: format!(
                    "Unsupported protocol version: {ver}. Supported: {}",
                    crate::protocol::PROTOCOL_VERSION
                ),
                code: "unsupported_protocol_version".into(),
                retry_after_ms: None,
            },
        )
        .await?;
        return Ok(FrameOutcome::Break);
    }
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
        *state_opt = Some(engine.create_state(enable_dia));
    }
    #[cfg(not(feature = "diarization"))]
    {
        let _ = (engine, state_opt, diarization);
    }
    Ok(FrameOutcome::Continue)
}

/// Handle `{"type":"stop"}`. Flushes the streaming state, sends a final
/// segment (empty if there was nothing pending), and signals clean break.
async fn handle_stop_message(
    sink: &mut WsSink,
    engine: &Arc<Engine>,
    state_opt: &mut Option<crate::inference::StreamingState>,
    peer: SocketAddr,
) -> Result<FrameOutcome> {
    tracing::info!("Stop received from {peer}, finalizing");
    let Some(mut state) = state_opt.take() else {
        return Ok(FrameOutcome::Break);
    };
    let flush_seg = engine.flush_state(&mut state);
    drop(state);
    let final_msg = if let Some(seg) = flush_seg {
        ServerMessage::Final(seg)
    } else {
        ServerMessage::Final(crate::inference::TranscriptSegment {
            text: String::new(),
            timestamp: crate::inference::now_timestamp(),
            words: vec![],
            is_final: true,
        })
    };
    send_server_message(sink, &final_msg).await?;
    Ok(FrameOutcome::Break)
}

/// Flush any pending streaming state and emit a `Final` frame (even an empty
/// one) so e2e tests and clients can reliably assert that every session ends
/// with a Final before the Close. Used by the cancel and session-cap branches
/// of `handle_ws_inner`.
async fn flush_and_final(
    sink: &mut WsSink,
    engine: &Arc<Engine>,
    state_opt: &mut Option<crate::inference::StreamingState>,
) -> Result<()> {
    let flush_seg = state_opt
        .as_mut()
        .and_then(|state| engine.flush_state(state));
    let final_msg = match flush_seg {
        Some(seg) => ServerMessage::Final(seg),
        None => ServerMessage::Final(crate::inference::TranscriptSegment {
            text: String::new(),
            timestamp: crate::inference::now_timestamp(),
            words: vec![],
            is_final: true,
        }),
    };
    send_server_message(sink, &final_msg).await
}

/// Runs the WebSocket session loop. Always tries to return the triplet so the
/// caller can check it back into the pool. Returns `None` only if the triplet
/// was lost due to a thread panic inside `spawn_blocking`.
async fn handle_ws_inner(
    socket: WebSocket,
    peer: SocketAddr,
    engine: &Arc<Engine>,
    limits: &RuntimeLimits,
    triplet: SessionTriplet,
    cancel: tokio_util::sync::CancellationToken,
) -> (Option<SessionTriplet>, Result<()>) {
    let (mut sink, mut source) = socket.split();
    tracing::info!("Client connected: {peer}");

    #[cfg(feature = "diarization")]
    let diarization_available = engine.has_speaker_encoder();
    #[cfg(not(feature = "diarization"))]
    let diarization_available = false;

    let ready = ServerMessage::Ready {
        model: "gigaam-v3-e2e-rnnt".into(),
        sample_rate: DEFAULT_SAMPLE_RATE,
        version: crate::protocol::PROTOCOL_VERSION.into(),
        supported_rates: SUPPORTED_RATES.to_vec(),
        diarization: diarization_available,
        min_protocol_version: None,
    };
    if let Err(e) = send_server_message(&mut sink, &ready).await {
        return (Some(triplet), Err(e));
    }

    let mut state_opt = Some(engine.create_state(false));
    let mut triplet_opt = Some(triplet);
    let mut client_sample_rate: u32 = DEFAULT_SAMPLE_RATE;
    let mut audio_received = false;
    // V1-25: carries the trailing odd byte across PCM16 frames so clients
    // that split their streams on odd boundaries don't accumulate a
    // 1-sample phase shift in the decoded audio.
    let mut pending_byte: Option<u8> = None;

    let idle_timeout = std::time::Duration::from_secs(limits.idle_timeout_secs);

    // V1-04: wall-clock deadline independent of `idle_timeout`. Setting
    // `max_session_secs = 0` disables the cap by parking the deadline far in
    // the future (u64::MAX / 2 ≈ 292 billion years) so `sleep_until` never
    // fires — callers who deliberately want unlimited sessions don't pay for
    // an additional branch in the select.
    let session_deadline = if limits.max_session_secs == 0 {
        tokio::time::Instant::now() + std::time::Duration::from_secs(u64::MAX / 2)
    } else {
        tokio::time::Instant::now() + std::time::Duration::from_secs(limits.max_session_secs)
    };

    let result: Result<()> = loop {
        // Fast-path deadline / cancel check: if a client streams frames
        // continuously (e.g. 20 ms silence every 100 ms) the `source.next()`
        // arm is always ready when we re-enter `select!`, and with `biased;`
        // the runtime still polls cancel / sleep_until first — but only if
        // they have a registered waker. `sleep_until` registers its waker
        // correctly, yet a subtle race on fast CI runners can let the frame
        // arm fire before the timer's waker is installed. A cheap
        // pre-check here guarantees the deadline / cancel wins.
        if cancel.is_cancelled() {
            tracing::info!(peer = %peer, "Shutdown signalled — flushing WS session");
            let _ = flush_and_final(&mut sink, engine, &mut state_opt).await;
            let _ = sink
                .send(WsMessage::Close(Some(axum::extract::ws::CloseFrame {
                    code: 1001,
                    reason: "server shutdown".into(),
                })))
                .await;
            break Ok(());
        }
        if tokio::time::Instant::now() >= session_deadline {
            tracing::warn!(
                peer = %peer,
                max_session_secs = limits.max_session_secs,
                "Session cap reached — closing WS"
            );
            let _ = send_server_message(
                &mut sink,
                &ServerMessage::Error {
                    message: "Maximum session duration exceeded".into(),
                    code: "max_session_duration_exceeded".into(),
                    retry_after_ms: None,
                },
            )
            .await;
            let _ = flush_and_final(&mut sink, engine, &mut state_opt).await;
            let _ = sink
                .send(WsMessage::Close(Some(axum::extract::ws::CloseFrame {
                    code: 1008,
                    reason: "max session duration".into(),
                })))
                .await;
            break Ok(());
        }

        tokio::select! {
            // `biased;` — cancel > deadline > frame. Guarantees that a
            // SIGTERM always wins a race against a pending frame, so the
            // drain path is deterministic.
            biased;

            _ = cancel.cancelled() => {
                tracing::info!(peer = %peer, "Shutdown signalled — flushing WS session");
                // Best-effort: the socket may already be dead if the peer
                // closed first, so every send is swallowed.
                let _ = flush_and_final(&mut sink, engine, &mut state_opt).await;
                let _ = sink
                    .send(WsMessage::Close(Some(axum::extract::ws::CloseFrame {
                        code: 1001,
                        reason: "server shutdown".into(),
                    })))
                    .await;
                break Ok(());
            }

            _ = tokio::time::sleep_until(session_deadline) => {
                tracing::warn!(
                    peer = %peer,
                    max_session_secs = limits.max_session_secs,
                    "Session cap reached — closing WS"
                );
                let _ = send_server_message(
                    &mut sink,
                    &ServerMessage::Error {
                        message: "Maximum session duration exceeded".into(),
                        code: "max_session_duration_exceeded".into(),
                        retry_after_ms: None,
                    },
                )
                .await;
                let _ = flush_and_final(&mut sink, engine, &mut state_opt).await;
                let _ = sink
                    .send(WsMessage::Close(Some(axum::extract::ws::CloseFrame {
                        code: 1008,
                        reason: "max session duration".into(),
                    })))
                    .await;
                break Ok(());
            }

            maybe_msg = tokio::time::timeout(idle_timeout, source.next()) => {
                let msg = match maybe_msg {
                    Ok(Some(Ok(msg))) => msg,
                    Ok(Some(Err(e))) => break Err(e.into()),
                    Ok(None) => break Ok(()),
                    Err(_) => {
                        tracing::info!(
                            "Client {peer} idle timeout ({}s)",
                            limits.idle_timeout_secs
                        );
                        break Ok(());
                    }
                };

                let outcome = match msg {
                    WsMessage::Binary(data) => {
                        handle_binary_frame(
                            &mut sink,
                            engine,
                            &mut state_opt,
                            &mut triplet_opt,
                            &mut audio_received,
                            client_sample_rate,
                            &mut pending_byte,
                            peer,
                            data,
                        )
                        .await
                    }
                    WsMessage::Text(text) => match serde_json::from_str::<ClientMessage>(&text) {
                        Ok(ClientMessage::Configure {
                            sample_rate,
                            diarization,
                            protocol_version,
                        }) => {
                            handle_configure_message(
                                &mut sink,
                                engine,
                                &mut state_opt,
                                &mut client_sample_rate,
                                audio_received,
                                sample_rate,
                                diarization,
                                protocol_version,
                                peer,
                            )
                            .await
                        }
                        Ok(ClientMessage::Stop) => {
                            handle_stop_message(&mut sink, engine, &mut state_opt, peer).await
                        }
                        Err(_) => {
                            tracing::debug!(
                                "Unrecognized text message from {peer}: {}",
                                &text[..text.len().min(100)]
                            );
                            Ok(FrameOutcome::Continue)
                        }
                    },
                    WsMessage::Close(_) => Ok(FrameOutcome::Break),
                    _ => Ok(FrameOutcome::Continue), // ignore ping/pong
                };

                match outcome {
                    Ok(FrameOutcome::Continue) => continue,
                    Ok(FrameOutcome::Break) => break Ok(()),
                    Err(e) => break Err(e),
                }
            }
        }
    };

    tracing::info!("Client disconnected: {peer}");
    (triplet_opt, result)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_catch_unwind_preserves_ownership_across_panic() {
        // Locks in the ownership contract used by `handle_ws_inner`'s spawn_blocking
        // block: moving captured values into the closure and wrapping the inner
        // computation in `catch_unwind(AssertUnwindSafe(_))` guarantees that the
        // values are observable after a panic, so the triplet can be returned to the
        // pool and the streaming state can be reset.
        use std::panic::{AssertUnwindSafe, catch_unwind};

        let mut state = 42u32;
        let mut triplet_marker = String::from("pool_slot");

        let result = catch_unwind(AssertUnwindSafe(|| {
            state = 99;
            triplet_marker.push_str("/taken");
            panic!("simulated inference panic");
        }));

        assert!(result.is_err(), "catch_unwind must report the panic");
        assert_eq!(state, 99, "state must remain accessible after panic");
        assert_eq!(
            triplet_marker, "pool_slot/taken",
            "triplet marker must survive panic"
        );
    }
}
