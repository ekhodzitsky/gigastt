//! WebSocket server that accepts PCM16 audio and streams transcripts.

use crate::inference::Engine;
use crate::protocol::ServerMessage;
use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message;

pub async fn run(engine: Engine, port: u16) -> Result<()> {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = TcpListener::bind(&addr).await?;
    let engine = Arc::new(engine);

    tracing::info!("gigastt server listening on ws://{addr}");

    loop {
        let (stream, peer) = listener.accept().await?;
        let engine = engine.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, peer, engine).await {
                tracing::error!("Connection error from {peer}: {e}");
            }
        });
    }
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
    };
    sink.send(Message::Text(serde_json::to_string(&ready)?)).await?;

    // Create per-connection streaming state (LSTM decoder + audio buffer)
    let mut stream_state = engine.create_state();

    // Process incoming audio frames
    while let Some(msg) = source.next().await {
        let msg = msg?;
        match msg {
            Message::Binary(data) => {
                // PCM16 mono — 2 bytes per sample
                let samples_48k: Vec<i16> = data
                    .chunks_exact(2)
                    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();

                // Downsample 48kHz → 16kHz (3:1 ratio with averaging)
                let samples: Vec<i16> = samples_48k
                    .chunks(3)
                    .map(|c| {
                        let sum: i32 = c.iter().map(|&s| s as i32).sum();
                        (sum / c.len() as i32) as i16
                    })
                    .collect();

                match engine.process_chunk(&samples, &mut stream_state) {
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
                            sink.send(Message::Text(serde_json::to_string(&msg)?)).await?;
                        }
                    }
                    Err(e) => {
                        tracing::error!("Inference error for {peer}: {e:#}");
                        let err = ServerMessage::Error {
                            message: "Inference failed. Please check audio format.".into(),
                            code: "inference_error".into(),
                        };
                        sink.send(Message::Text(serde_json::to_string(&err)?)).await?;
                    }
                }
            }
            Message::Close(_) => break,
            _ => {} // Ignore text/ping/pong
        }
    }

    tracing::info!("Client disconnected: {peer}");
    Ok(())
}
