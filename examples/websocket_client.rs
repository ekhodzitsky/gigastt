//! WebSocket client that connects to a running gigastt server and streams audio.
//!
//! Usage:
//!   1. Start server: cargo run -- serve
//!   2. Run client:   cargo run --example websocket_client -- path/to/audio.wav

use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: cargo run --example websocket_client -- <audio-file>");
        std::process::exit(1);
    });

    let url = std::env::var("GIGASTT_URL")
        .unwrap_or_else(|_| "ws://127.0.0.1:9876/v1/ws".into());

    println!("Connecting to {url}...");
    let (ws, _) = tokio_tungstenite::connect_async(&url).await?;
    let (mut sink, mut stream) = ws.split();

    // Wait for Ready
    if let Some(Ok(msg)) = stream.next().await {
        let text = msg.into_text()?;
        let v: serde_json::Value = serde_json::from_str(&text)?;
        println!(
            "Server ready: model={}, sample_rate={}",
            v["model"], v["sample_rate"]
        );
    }

    // Read audio file as raw bytes and send in 32KB chunks
    let audio_data = std::fs::read(&path)?;
    println!("Sending {} bytes of audio...", audio_data.len());
    for chunk in audio_data.chunks(32 * 1024) {
        sink.send(Message::Binary(chunk.to_vec().into())).await?;
    }

    // Send stop
    sink.send(Message::Text(
        serde_json::to_string(&serde_json::json!({"type": "stop"}))
            .unwrap()
            .into(),
    ))
    .await?;

    // Read responses until Final
    while let Some(Ok(msg)) = stream.next().await {
        if let Ok(text) = msg.into_text() {
            let v: serde_json::Value = serde_json::from_str(&text)?;
            match v["type"].as_str() {
                Some("partial") => println!("  Partial: {}", v["text"]),
                Some("final") => {
                    println!("  Final:   {}", v["text"]);
                    break;
                }
                Some("error") => {
                    eprintln!("  Error: {}", v["message"]);
                    break;
                }
                _ => {}
            }
        }
    }

    sink.close().await?;
    Ok(())
}
