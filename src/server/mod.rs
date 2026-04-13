//! HTTP + WebSocket server that accepts audio and streams transcripts.
//!
//! Single port serves both REST API (health, transcribe, SSE) and WebSocket.

pub mod http;

use crate::inference::Engine;
use crate::protocol::{ClientMessage, ServerMessage};
use anyhow::{Context, Result};
use axum::extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::Response;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use axum::Router;
use futures_util::{SinkExt, StreamExt};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Semaphore;

const MAX_CONCURRENT_CONNECTIONS: usize = 4;

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
    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .context("Invalid host:port")?;

    let state = Arc::new(http::AppState {
        engine: Arc::new(engine),
        semaphore: Arc::new(Semaphore::new(MAX_CONCURRENT_CONNECTIONS)),
    });

    let app = Router::new()
        .route("/health", get(http::health))
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
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await?;

    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c().await.ok();
    tracing::info!("Shutting down server");
}

async fn cors_middleware(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
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
    if let Some(origin) = headers.get("origin").and_then(|v| v.to_str().ok()).filter(|o| !o.contains("127.0.0.1") && !o.contains("localhost")) {
        tracing::warn!("WebSocket connection from non-local origin: {origin} (peer: {peer})");
    }
    ws.max_message_size(512 * 1024)
        .max_frame_size(512 * 1024)
        .on_upgrade(move |socket| handle_ws(socket, peer, state))
}

async fn handle_ws(socket: WebSocket, peer: SocketAddr, state: Arc<http::AppState>) {
    let permit = match state.semaphore.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => return,
    };

    if let Err(e) = handle_ws_inner(socket, peer, &state.engine).await {
        tracing::error!("WebSocket error from {peer}: {e}");
    }

    drop(permit);
}

async fn handle_ws_inner(
    socket: WebSocket,
    peer: SocketAddr,
    engine: &Arc<Engine>,
) -> Result<()> {
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
    sink.send(WsMessage::Text(serde_json::to_string(&ready)?.into()))
        .await?;

    let mut state_opt = Some(engine.create_state(
        #[cfg(feature = "diarization")]
        false,
    ));
    let mut client_sample_rate: u32 = DEFAULT_SAMPLE_RATE;
    let mut audio_received = false;

    loop {
        let msg = match tokio::time::timeout(
            std::time::Duration::from_secs(300),
            source.next(),
        )
        .await
        {
            Ok(Some(msg)) => msg?,
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
                    crate::inference::audio::resample(
                        &samples_f32,
                        client_sample_rate,
                        16000,
                    )?
                };

                let mut state = state_opt.take().context("Streaming state lost")?;
                let eng = engine.clone();
                let (result, state_back) = tokio::task::spawn_blocking(move || {
                    let r = eng.process_chunk(&samples_16k, &mut state);
                    (r, state)
                })
                .await?;
                state_opt = Some(state_back);

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
                            sink.send(WsMessage::Text(serde_json::to_string(&msg)?.into()))
                                .await?;
                        }
                    }
                    Err(e) => {
                        tracing::error!("Inference error for {peer}: {e:#}");
                        let err = ServerMessage::Error {
                            message: "Inference failed. Please check audio format."
                                .into(),
                            code: "inference_error".into(),
                        };
                        sink.send(WsMessage::Text(serde_json::to_string(&err)?.into()))
                            .await?;
                    }
                }
            }
            WsMessage::Text(text) => {
                match serde_json::from_str::<ClientMessage>(&text) {
                    Ok(ClientMessage::Configure { sample_rate, diarization }) => {
                        if audio_received {
                            let err = ServerMessage::Error {
                                message: "Configure must be sent before first audio frame".into(),
                                code: "configure_too_late".into(),
                            };
                            sink.send(WsMessage::Text(serde_json::to_string(&err)?.into()))
                                .await?;
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
                                sink.send(WsMessage::Text(serde_json::to_string(&err)?.into()))
                                    .await?;
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
                        let mut state =
                            state_opt.take().context("Streaming state lost")?;
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
                        sink.send(WsMessage::Text(serde_json::to_string(&final_msg)?.into()))
                            .await?;
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

    tracing::info!("Client disconnected: {peer}");
    Ok(())
}
