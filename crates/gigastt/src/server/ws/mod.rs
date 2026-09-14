//! WebSocket handler: upgrade, session loop, PCM16 processing, and inference dispatch.

use super::config::{DEFAULT_SAMPLE_RATE, RuntimeLimits, SUPPORTED_RATES, pool_retry_after_ms};
use super::http;
use super::json_text;
use anyhow::Result;
use axum::extract::State;
use axum::extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade};
use axum::response::Response;
use futures_util::{SinkExt, StreamExt};
use gigastt_core::inference::{Engine, SessionTriplet};
use gigastt_core::protocol::{ClientMessage, ServerMessage};
use std::net::SocketAddr;
use std::sync::Arc;

mod frames;
mod incoming;
use frames::{
    FrameOutcome, flush_and_final, handle_binary_frame, handle_configure_message,
    handle_stop_message, send_server_message,
};

/// Interval between server-initiated WebSocket pings. Keeps connections alive
/// through idle-dropping proxies and detects half-open TCP sessions faster than
/// the (much larger) idle timeout.
const WS_PING_INTERVAL_SECS: u64 = 30;

/// Close the socket after this many consecutive unanswered pings (no Pong and
/// no other frame in between). Because the first ping is sent one interval
/// after connect and the close fires on the tick *after* the counter reaches
/// this value, detection takes roughly `WS_PING_INTERVAL_SECS × (this + 1)`
/// seconds (≈ 90 s at the defaults).
const WS_MAX_MISSED_PONGS: u32 = 2;

/// Whether a ping tick should close the socket: `true` once `unanswered_pings`
/// has reached [`WS_MAX_MISSED_PONGS`]. Factored out of the session loop so the
/// close-threshold and counter-reset edges can be unit-tested without driving a
/// real socket and timers.
fn keepalive_should_close(unanswered_pings: u32) -> bool {
    unanswered_pings >= WS_MAX_MISSED_PONGS
}

pub(crate) async fn ws_handler(
    ws: WebSocketUpgrade,
    axum::extract::ConnectInfo(peer): axum::extract::ConnectInfo<SocketAddr>,
    State(state): State<Arc<http::AppState>>,
    headers: axum::http::HeaderMap,
) -> Response {
    // Origin allowlist is enforced by `origin_middleware` before the request
    // reaches this handler; anything that arrives here has already been cleared.
    //
    // If shutdown has already been requested, refuse the upgrade
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
        reg.gauge_inc("gigastt_ws_active_connections", &[], 1);
    }
    struct WsMetricsGuard(Arc<http::AppState>);
    impl Drop for WsMetricsGuard {
        fn drop(&mut self) {
            if let Some(ref reg) = self.0.metrics_registry {
                reg.gauge_inc("gigastt_ws_active_connections", &[], -1);
            }
        }
    }
    let _ws_guard = WsMetricsGuard(state.clone());
    // Snapshot the live engine once for the whole session; a concurrent
    // hot-reload swaps the `ArcSwap`, but this session keeps the engine (and
    // pool) it checked out from until it ends.
    let engine = state.engine.load_full();
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
            engine.pool.checkout(),
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
                    reg.counter_inc("gigastt_pool_timeouts_total", &[], 1);
                    reg.histogram_record("gigastt_pool_checkout_duration_seconds", &[], checkout_start.elapsed().as_secs_f64());
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
            &[],
            checkout_start.elapsed().as_secs_f64(),
        );
    }
    let reservation = guard.into_owned();

    let limits = state.limits.load();
    let result = handle_ws_inner(
        socket,
        peer,
        &engine,
        &limits,
        reservation,
        state.shutdown.clone(),
        state.metrics_registry.clone(),
    )
    .await;
    if let Err(e) = result {
        tracing::error!("WebSocket error from {peer}: {e}");
    }
}

/// Runs the WebSocket session loop. The reservation is consumed and returned
/// to the pool automatically when the function returns (or on panic unwind).
async fn handle_ws_inner(
    socket: WebSocket,
    peer: SocketAddr,
    engine: &Arc<Engine>,
    limits: &RuntimeLimits,
    reservation: gigastt_core::inference::OwnedReservation<SessionTriplet>,
    cancel: tokio_util::sync::CancellationToken,
    metrics: Option<Arc<super::metrics::MetricsRegistry>>,
) -> Result<()> {
    let (mut sink, source) = socket.split();
    let abort = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let _abort_guard = super::file_transcribe::AbortOnDrop(abort.clone());
    let mut incoming = incoming::Incoming::new(source, abort.clone());
    let partial = Arc::new(gigastt_core::inference::TranscriptSnapshot::default());
    tracing::info!("Client connected: {peer}");

    #[cfg(feature = "diarization")]
    let diarization_available = engine.has_speaker_encoder();
    #[cfg(not(feature = "diarization"))]
    let diarization_available = false;

    let ready = ServerMessage::Ready {
        model: engine.variant().model_id().into(),
        sample_rate: DEFAULT_SAMPLE_RATE,
        version: gigastt_core::protocol::PROTOCOL_VERSION.into(),
        supported_rates: SUPPORTED_RATES.to_vec(),
        diarization: diarization_available,
        min_protocol_version: Some(gigastt_core::protocol::MIN_PROTOCOL_VERSION.into()),
        max_session_secs: limits.max_session_secs,
        idle_timeout_secs: limits.idle_timeout_secs,
    };
    send_server_message(&mut sink, &ready).await?;

    let mut stream_state = engine.create_state(false);
    stream_state.abort = Some(abort.clone());
    stream_state.partial = Some(partial.clone());
    let mut state_opt = Some(stream_state);
    let mut reservation = Some(reservation);
    let mut client_sample_rate: u32 = DEFAULT_SAMPLE_RATE;
    let mut audio_received = false;
    let mut empty_frame_count: usize = 0;
    // Carries the trailing odd byte across PCM16 frames so clients
    // that split their streams on odd boundaries don't accumulate a
    // 1-sample phase shift in the decoded audio.
    let mut pending_byte: Option<u8> = None;
    let mut pcm_decode_buf: Vec<f32> = Vec::new();

    let idle_timeout = std::time::Duration::from_secs(limits.idle_timeout_secs);

    // Wall-clock deadline independent of `idle_timeout`. Setting
    // `max_session_secs = 0` disables the cap by parking the deadline far in
    // the future (u64::MAX / 2 ≈ 292 billion years) so `sleep_until` never
    // fires — callers who deliberately want unlimited sessions don't pay for
    // an additional branch in the select.
    let session_deadline = if limits.max_session_secs == 0 {
        tokio::time::Instant::now() + std::time::Duration::from_secs(u64::MAX / 2)
    } else {
        tokio::time::Instant::now() + std::time::Duration::from_secs(limits.max_session_secs)
    };

    // Server-initiated keepalive: ping every `WS_PING_INTERVAL_SECS`, close once
    // `WS_MAX_MISSED_PONGS` consecutive pings go unanswered. Any inbound frame
    // (Pong, Binary, Text) counts as liveness and resets the counter. The first
    // tick is one interval out so we don't ping at connect time.
    let ping_period = std::time::Duration::from_secs(WS_PING_INTERVAL_SECS);
    let mut ping_interval =
        tokio::time::interval_at(tokio::time::Instant::now() + ping_period, ping_period);
    ping_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut unanswered_pings: u32 = 0;

    let control = frames::StreamControl {
        abort: &abort,
        partial: &partial,
        shutdown: &cancel,
        disconnected: &incoming.closed,
        timeout_secs: limits.inference_timeout_secs,
        deadline: session_deadline,
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

            _ = ping_interval.tick() => {
                if keepalive_should_close(unanswered_pings) {
                    tracing::info!(
                        peer = %peer,
                        missed = unanswered_pings,
                        "WS peer unresponsive to pings — closing"
                    );
                    let _ = sink
                        .send(WsMessage::Close(Some(axum::extract::ws::CloseFrame {
                            code: 1001,
                            reason: "ping timeout".into(),
                        })))
                        .await;
                    break Ok(());
                }
                if sink.send(WsMessage::Ping(Vec::new().into())).await.is_err() {
                    break Ok(());
                }
                unanswered_pings += 1;
                continue;
            }

            maybe_msg = tokio::time::timeout(idle_timeout, incoming.rx.recv()) => {
                let msg = match maybe_msg {
                    Ok(Some(Ok(msg))) => msg,
                    Ok(Some(Err(e))) => break Err(e.into()),
                    Ok(None) => break Ok(()),
                    Err(_) => {
                        tracing::info!(
                            "Client {peer} idle timeout ({}s)",
                            limits.idle_timeout_secs
                        );
                        let err = ServerMessage::Error {
                            message: "Idle timeout".into(),
                            code: "idle_timeout".into(),
                            retry_after_ms: None,
                        };
                        let _ = sink.send(WsMessage::Text(json_text(&err).into())).await;
                        let _ = sink
                            .send(WsMessage::Close(Some(axum::extract::ws::CloseFrame {
                                code: 1001,
                                reason: "idle timeout".into(),
                            })))
                            .await;
                        break Ok(());
                    }
                };

                // Any inbound frame (Pong / Binary / Text) proves the peer is
                // alive and resets the keepalive counter.
                unanswered_pings = 0;

                let outcome = match msg {
                    WsMessage::Binary(data) => {
                        handle_binary_frame(
                            &mut sink,
                            engine,
                            &mut state_opt,
                            &mut reservation,
                            &mut audio_received,
                            &mut empty_frame_count,
                            client_sample_rate,
                            &mut pending_byte,
                            peer,
                            data,
                            &mut pcm_decode_buf,
                            metrics.as_ref(),
                            &control,
                        )
                        .await
                    }
                    WsMessage::Text(text) => match serde_json::from_str::<ClientMessage>(&text) {
                        Ok(ClientMessage::Configure {
                            sample_rate,
                            diarization,
                            protocol_version,
                            punctuation,
                            itn,
                            endpoint_mode,
                            min_silence_ms,
                            ..
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
                                punctuation,
                                itn,
                                endpoint_mode,
                                min_silence_ms,
                                peer,
                            )
                            .await
                        }
                        Ok(ClientMessage::Stop) => {
                            handle_stop_message(
                                &mut sink,
                                engine,
                                &mut state_opt,
                                &mut reservation,
                                peer,
                                &control,
                            )
                            .await
                        }
                        Ok(_) => Ok(FrameOutcome::Continue),
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
    result
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_keepalive_close_threshold_and_reset() {
        use super::{WS_MAX_MISSED_PONGS, keepalive_should_close};

        // Walk the tick sequence for a peer that never answers: each tick that
        // doesn't close sends a ping and bumps the counter. It must close after
        // exactly WS_MAX_MISSED_PONGS pings have gone unanswered, within a
        // bounded number of ticks.
        let mut unanswered = 0u32;
        let mut ticks = 0u32;
        while !keepalive_should_close(unanswered) {
            unanswered += 1; // ping sent this tick
            ticks += 1;
            assert!(
                ticks <= 16,
                "keepalive must close within a bounded tick count"
            );
        }
        assert_eq!(
            unanswered, WS_MAX_MISSED_PONGS,
            "socket closes exactly once the cap of unanswered pings is reached"
        );

        // Any inbound frame resets the counter to 0; the next tick must NOT close.
        assert!(
            !keepalive_should_close(0),
            "a reset (inbound frame) keeps the socket open"
        );
        assert!(
            !keepalive_should_close(WS_MAX_MISSED_PONGS - 1),
            "one below the cap must stay open"
        );
    }

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
