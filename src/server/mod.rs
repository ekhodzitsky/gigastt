//! HTTP + WebSocket server that accepts audio and streams transcripts.
//!
//! Single port serves both REST API (health, transcribe, SSE) and WebSocket.

pub mod http;

use crate::inference::{Engine, SessionTriplet};
use crate::protocol::{ClientMessage, ServerMessage};
use anyhow::{Context, Result};
use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::extract::State;
use axum::extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade};
use axum::response::Response;
use axum::routing::{get, post};
use futures_util::{SinkExt, StreamExt};
use std::net::SocketAddr;
use std::sync::Arc;

/// Serialize a server message to JSON with a safe fallback on error.
fn json_text(msg: &impl serde::Serialize) -> String {
    serde_json::to_string(msg).unwrap_or_else(|e| {
        tracing::error!("Failed to serialize server message: {e}");
        r#"{"type":"error","message":"Internal serialization error","code":"internal"}"#.into()
    })
}

/// Supported input sample rates (Hz). Default is 48000 for backward compatibility.
const SUPPORTED_RATES: &[u32] = &[8000, 16000, 24000, 44100, 48000];
const DEFAULT_SAMPLE_RATE: u32 = 48000;

/// Start the HTTP + WebSocket STT server on the given host and port.
///
/// Serves REST API endpoints and WebSocket on a single port:
/// - `GET /health` — health check
/// - `POST /v1/transcribe` — file transcription
/// - `POST /v1/transcribe/stream` — SSE streaming transcription
/// - `GET /ws` — WebSocket streaming protocol
///
/// Runs until `Ctrl-C` is received.
pub async fn run(engine: Engine, port: u16, host: &str) -> Result<()> {
    run_with_shutdown(engine, port, host, None).await
}

/// Start server with an optional programmatic shutdown signal.
///
/// When `shutdown` is `Some`, the server stops when the sender fires (or is dropped).
/// When `None`, the server stops on Ctrl-C. Used by tests for clean teardown.
pub async fn run_with_shutdown(
    engine: Engine,
    port: u16,
    host: &str,
    shutdown: Option<tokio::sync::oneshot::Receiver<()>>,
) -> Result<()> {
    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .context("Invalid host:port")?;

    let state = Arc::new(http::AppState {
        engine: Arc::new(engine),
    });

    let app = Router::new()
        .route("/health", get(http::health))
        .route("/v1/models", get(http::models))
        .route("/v1/transcribe", post(http::transcribe))
        .route("/v1/transcribe/stream", post(http::transcribe_stream))
        .route("/ws", get(ws_handler))
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024)) // 50MB
        .layer(axum::middleware::from_fn(cors_middleware))
        .with_state(state);

    tracing::info!("gigastt server listening on http://{addr}");
    tracing::info!("  WebSocket: ws://{addr}/ws");
    tracing::info!("  REST API:  http://{addr}/health, /v1/transcribe, /v1/transcribe/stream");

    let listener = tokio::net::TcpListener::bind(&addr).await?;

    let shutdown_fut = async {
        match shutdown {
            Some(rx) => {
                rx.await.ok();
            }
            None => {
                tokio::signal::ctrl_c().await.ok();
            }
        }
        tracing::info!("Shutting down server");
    };

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_fut)
    .await?;

    Ok(())
}

async fn cors_middleware(req: axum::extract::Request, next: axum::middleware::Next) -> Response {
    let mut response = next.run(req).await;
    let headers = response.headers_mut();
    headers.insert(
        axum::http::header::ACCESS_CONTROL_ALLOW_ORIGIN,
        axum::http::HeaderValue::from_static("*"),
    );
    headers.insert(
        axum::http::header::ACCESS_CONTROL_ALLOW_METHODS,
        axum::http::HeaderValue::from_static("GET, POST, OPTIONS"),
    );
    headers.insert(
        axum::http::header::ACCESS_CONTROL_ALLOW_HEADERS,
        axum::http::HeaderValue::from_static("*"),
    );
    response
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    axum::extract::ConnectInfo(peer): axum::extract::ConnectInfo<SocketAddr>,
    headers: axum::http::HeaderMap,
    State(state): State<Arc<http::AppState>>,
) -> Response {
    if let Some(origin) = headers
        .get("origin")
        .and_then(|v| v.to_str().ok())
        .filter(|o| !o.contains("127.0.0.1") && !o.contains("localhost"))
    {
        tracing::warn!("WebSocket connection from non-local origin: {origin} (peer: {peer})");
    }
    ws.max_message_size(512 * 1024)
        .max_frame_size(512 * 1024)
        .on_upgrade(move |socket| handle_ws(socket, peer, state))
}

async fn handle_ws(socket: WebSocket, peer: SocketAddr, state: Arc<http::AppState>) {
    let triplet = match tokio::time::timeout(
        std::time::Duration::from_secs(30),
        state.engine.pool.checkout(),
    )
    .await
    {
        Ok(triplet) => triplet,
        Err(_) => {
            tracing::warn!("WebSocket pool checkout timeout for {peer}");
            let (mut sink, _) = socket.split();
            let err = ServerMessage::Error {
                message: "Server busy, try again later".into(),
                code: "timeout".into(),
            };
            let _ = sink.send(WsMessage::Text(json_text(&err).into())).await;
            return;
        }
    };

    let (triplet_opt, result) = handle_ws_inner(socket, peer, &state.engine, triplet).await;
    if let Err(e) = result {
        tracing::error!("WebSocket error from {peer}: {e}");
    }

    if let Some(triplet) = triplet_opt {
        state.engine.pool.checkin(triplet).await;
    }
    // If triplet_opt is None, the triplet was lost due to a spawn_blocking panic.
    // The pool degrades gracefully with fewer available slots.
}

/// Runs the WebSocket session loop. Always tries to return the triplet so the
/// caller can check it back into the pool. Returns `None` only if the triplet
/// was lost due to a thread panic inside `spawn_blocking`.
async fn handle_ws_inner(
    socket: WebSocket,
    peer: SocketAddr,
    engine: &Arc<Engine>,
    triplet: SessionTriplet,
) -> (Option<SessionTriplet>, Result<()>) {
    let (mut sink, mut source) = socket.split();

    tracing::info!("Client connected: {peer}");

    // Send ready message
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
    };
    if let Err(e) = sink.send(WsMessage::Text(json_text(&ready).into())).await {
        return (Some(triplet), Err(e.into()));
    }

    let mut state_opt = Some(engine.create_state(
        #[cfg(feature = "diarization")]
        false,
    ));
    let mut triplet_opt = Some(triplet);
    let mut client_sample_rate: u32 = DEFAULT_SAMPLE_RATE;
    let mut audio_received = false;

    let result: Result<()> = 'outer: {
        loop {
            let msg = match tokio::time::timeout(std::time::Duration::from_secs(300), source.next())
                .await
            {
                Ok(Some(Ok(msg))) => msg,
                Ok(Some(Err(e))) => break 'outer Err(e.into()),
                Ok(None) => break,
                Err(_) => {
                    tracing::info!("Client {peer} idle timeout (300s)");
                    break;
                }
            };
            match msg {
                WsMessage::Binary(data) if data.is_empty() => {
                    tracing::debug!("Empty binary frame from {peer}, skipping");
                }
                WsMessage::Binary(data) => {
                    audio_received = true;
                    if data.len() % 2 != 0 {
                        tracing::warn!(
                            "Odd-length PCM frame ({} bytes) from {peer}, dropping last byte",
                            data.len()
                        );
                    }
                    let samples_f32: Vec<f32> = data
                        .chunks_exact(2)
                        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0)
                        .collect();

                    let samples_16k = if client_sample_rate == 16000 {
                        samples_f32
                    } else {
                        match crate::inference::audio::resample(
                            &samples_f32,
                            client_sample_rate,
                            16000,
                        ) {
                            Ok(s) => s,
                            Err(e) => break 'outer Err(e),
                        }
                    };

                    let mut state = match state_opt.take() {
                        Some(s) => s,
                        None => break 'outer Err(anyhow::anyhow!("Streaming state lost")),
                    };
                    let mut triplet = match triplet_opt.take() {
                        Some(t) => t,
                        None => {
                            tracing::error!("Triplet unexpectedly missing for {peer}");
                            break 'outer Err(anyhow::anyhow!("Triplet lost"));
                        }
                    };
                    let eng = engine.clone();
                    let join_result = tokio::task::spawn_blocking(move || {
                        let r = eng.process_chunk(&samples_16k, &mut state, &mut triplet);
                        (r, state, triplet)
                    })
                    .await;

                    match join_result {
                        Ok((result, state_back, triplet_back)) => {
                            state_opt = Some(state_back);
                            triplet_opt = Some(triplet_back);
                            match result {
                                Ok(segments) => {
                                    for seg in segments {
                                        let msg = if seg.is_final {
                                            ServerMessage::Final {
                                                text: seg.text,
                                                timestamp: seg.timestamp,
                                                words: seg.words,
                                            }
                                        } else {
                                            ServerMessage::Partial {
                                                text: seg.text,
                                                timestamp: seg.timestamp,
                                                words: seg.words,
                                            }
                                        };
                                        if let Err(e) =
                                            sink.send(WsMessage::Text(json_text(&msg).into())).await
                                        {
                                            break 'outer Err(e.into());
                                        }
                                    }
                                }
                                Err(e) => {
                                    tracing::error!("Inference error for {peer}: {e:#}");
                                    let err = ServerMessage::Error {
                                        message: "Inference failed. Please check audio format."
                                            .into(),
                                        code: "inference_error".into(),
                                    };
                                    if let Err(e) =
                                        sink.send(WsMessage::Text(json_text(&err).into())).await
                                    {
                                        break 'outer Err(e.into());
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            // spawn_blocking panicked — triplet is permanently lost.
                            // The pool degrades gracefully with one fewer slot.
                            // Recovering the triplet would require restructuring closure
                            // ownership (see SSE handler pattern in http.rs).
                            tracing::warn!(
                                "spawn_blocking panicked for {peer}: {e} — \
                                 triplet lost, pool capacity reduced"
                            );
                            break 'outer Err(anyhow::anyhow!("Inference thread panicked"));
                        }
                    }
                }
                WsMessage::Text(text) => {
                    match serde_json::from_str::<ClientMessage>(&text) {
                        Ok(ClientMessage::Configure {
                            sample_rate,
                            diarization,
                        }) => {
                            if audio_received {
                                let err = ServerMessage::Error {
                                    message: "Configure must be sent before first audio frame"
                                        .into(),
                                    code: "configure_too_late".into(),
                                };
                                if let Err(e) =
                                    sink.send(WsMessage::Text(json_text(&err).into())).await
                                {
                                    break 'outer Err(e.into());
                                }
                                continue;
                            }
                            if let Some(rate) = sample_rate {
                                if SUPPORTED_RATES.contains(&rate) {
                                    client_sample_rate = rate;
                                    tracing::info!(
                                        "Client {peer} configured sample rate: {rate}Hz"
                                    );
                                } else {
                                    let err = ServerMessage::Error {
                                        message: format!(
                                            "Unsupported sample rate: {rate}Hz. Supported: {SUPPORTED_RATES:?}"
                                        ),
                                        code: "invalid_sample_rate".into(),
                                    };
                                    if let Err(e) =
                                        sink.send(WsMessage::Text(json_text(&err).into())).await
                                    {
                                        break 'outer Err(e.into());
                                    }
                                }
                            }

                            // Re-create state if diarization preference changes
                            #[cfg(feature = "diarization")]
                            if let Some(enable_dia) = diarization {
                                tracing::info!(
                                    "Client {peer} configured diarization: {enable_dia}"
                                );
                                state_opt = Some(engine.create_state(enable_dia));
                            }
                            #[cfg(not(feature = "diarization"))]
                            let _ = diarization;
                        }
                        Ok(ClientMessage::Stop) => {
                            tracing::info!("Stop received from {peer}, finalizing");
                            let mut state = match state_opt.take() {
                                Some(s) => s,
                                None => break,
                            };
                            let flush_seg = engine.flush_state(&mut state);
                            drop(state);
                            let final_msg = if let Some(seg) = flush_seg {
                                ServerMessage::Final {
                                    text: seg.text,
                                    timestamp: seg.timestamp,
                                    words: seg.words,
                                }
                            } else {
                                ServerMessage::Final {
                                    text: String::new(),
                                    timestamp: crate::inference::now_timestamp(),
                                    words: vec![],
                                }
                            };
                            if let Err(e) = sink
                                .send(WsMessage::Text(json_text(&final_msg).into()))
                                .await
                            {
                                break 'outer Err(e.into());
                            }
                            break;
                        }
                        Err(_) => {
                            tracing::debug!(
                                "Unrecognized text message from {peer}: {}",
                                &text[..text.len().min(100)]
                            );
                        }
                    }
                }
                WsMessage::Close(_) => break,
                _ => {} // Ignore ping/pong
            }
        }
        Ok(())
    };

    tracing::info!("Client disconnected: {peer}");
    (triplet_opt, result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supported_rates_contains_common() {
        assert!(
            SUPPORTED_RATES.contains(&8000),
            "SUPPORTED_RATES must include 8000 Hz"
        );
        assert!(
            SUPPORTED_RATES.contains(&16000),
            "SUPPORTED_RATES must include 16000 Hz"
        );
        assert!(
            SUPPORTED_RATES.contains(&48000),
            "SUPPORTED_RATES must include 48000 Hz"
        );
    }

    #[test]
    fn test_default_sample_rate_in_supported() {
        assert!(
            SUPPORTED_RATES.contains(&DEFAULT_SAMPLE_RATE),
            "DEFAULT_SAMPLE_RATE ({DEFAULT_SAMPLE_RATE}) must be present in SUPPORTED_RATES"
        );
    }
}
