//! WebSocket server that accepts PCM16 audio and streams transcripts.

use crate::inference::Engine;
use crate::protocol::{ClientMessage, ServerMessage};
use anyhow::{Context, Result};
use futures_util::{SinkExt, StreamExt};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use tokio_tungstenite::tungstenite::Message;

const MAX_CONCURRENT_CONNECTIONS: usize = 4;

pub async fn run(engine: Engine, port: u16, host: &str) -> Result<()> {
    let addr: SocketAddr = format!("{host}:{port}").parse()
        .context("Invalid host:port")?;
    let listener = TcpListener::bind(&addr).await?;
    let engine = Arc::new(engine);
    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_CONNECTIONS));

    tracing::info!("gigastt server listening on ws://{addr}");

    loop {
        tokio::select! {
            result = listener.accept() => {
                let (stream, peer) = match result {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::error!("Accept error: {e}");
                        continue;
                    }
                };
                let engine = engine.clone();
                let permit = semaphore.clone().acquire_owned().await?;
                tokio::spawn(async move {
                    if let Err(e) = handle_connection(stream, peer, engine).await {
                        tracing::error!("Connection error from {peer}: {e}");
                    }
                    drop(permit);
                });
            }
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("Shutting down server");
                break;
            }
        }
    }

    Ok(())
}

async fn handle_connection(
    stream: tokio::net::TcpStream,
    peer: SocketAddr,
    engine: Arc<Engine>,
) -> Result<()> {
    let ws_config = tokio_tungstenite::tungstenite::protocol::WebSocketConfig {
        max_message_size: Some(512 * 1024), // 512 KiB (~16s of 16kHz PCM16)
        max_frame_size: Some(512 * 1024),
        ..Default::default()
    };
    let ws_stream =
        tokio_tungstenite::accept_async_with_config(stream, Some(ws_config)).await?;
    let (mut sink, mut source) = ws_stream.split();

    tracing::info!("Client connected: {peer}");

    // Send ready message — accept 48kHz from clients, resample to 16kHz internally
    let ready = ServerMessage::Ready {
        model: "gigaam-v3-e2e-rnnt".into(),
        sample_rate: 48000,
        version: crate::protocol::PROTOCOL_VERSION.into(),
    };
    sink.send(Message::Text(serde_json::to_string(&ready)?)).await?;

    // Create per-connection streaming state (LSTM decoder + audio buffer)
    // Wrapped in Option so we can move it into/out of spawn_blocking
    let mut state_opt = Some(engine.create_state());

    // Process incoming audio frames
    while let Some(msg) = source.next().await {
        let msg = msg?;
        match msg {
            Message::Binary(data) => {
                // PCM16 mono at 48kHz — convert to f32 for resampling
                let samples_48k_f32: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0)
                    .collect();

                // Resample 48kHz → 16kHz using the same algorithm as file transcription
                let samples_16k = crate::inference::audio::resample(&samples_48k_f32, 48000, 16000);

                // Run inference in a blocking thread to avoid blocking the tokio runtime
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
                                }
                            } else {
                                ServerMessage::Partial {
                                    text: seg.text,
                                    timestamp: seg.timestamp,
                                }
                            };
                            sink.send(Message::Text(serde_json::to_string(&msg)?))
                                .await?;
                        }
                    }
                    Err(e) => {
                        tracing::error!("Inference error for {peer}: {e:#}");
                        let err = ServerMessage::Error {
                            message: "Inference failed. Please check audio format.".into(),
                            code: "inference_error".into(),
                        };
                        sink.send(Message::Text(serde_json::to_string(&err)?))
                            .await?;
                    }
                }
            }
            Message::Text(text) => {
                if let Ok(ClientMessage::Stop) = serde_json::from_str(&text) {
                    tracing::info!("Stop received from {peer}, finalizing");
                    let final_msg = ServerMessage::Final {
                        text: String::new(),
                        timestamp: crate::inference::now_timestamp(),
                    };
                    sink.send(Message::Text(serde_json::to_string(&final_msg)?))
                        .await?;
                    break;
                } else {
                    tracing::debug!("Unrecognized text message from {peer}: {}", &text[..text.len().min(100)]);
                }
            }
            Message::Close(_) => break,
            _ => {} // Ignore ping/pong
        }
    }

    tracing::info!("Client disconnected: {peer}");
    Ok(())
}
