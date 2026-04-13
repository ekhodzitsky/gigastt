//! Integration tests for the WebSocket server protocol.
//!
//! These tests require the GigaAM model to be downloaded (~850MB).
//! Run with: `cargo test --test server_integration -- --ignored`

use futures_util::{SinkExt, StreamExt};
use std::path::PathBuf;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message;

fn home_dir() -> Option<PathBuf> {
    #[cfg(unix)]
    { std::env::var_os("HOME").map(PathBuf::from) }
    #[cfg(windows)]
    { std::env::var_os("USERPROFILE").map(PathBuf::from) }
}

/// Find a free port by binding to port 0.
async fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    listener.local_addr().unwrap().port()
}

/// Check if the model is available.
fn model_dir() -> Option<String> {
    let dir = home_dir()?.join(".gigastt").join("models");
    if dir.join("v3_e2e_rnnt_encoder.onnx").exists() {
        Some(dir.to_string_lossy().into_owned())
    } else {
        None
    }
}

#[tokio::test]
#[ignore] // Requires model download
async fn test_single_client_receives_ready() {
    let model_dir = model_dir().expect("Model not found. Run `cargo run -- download` first.");
    let port = free_port().await;

    let engine = gigastt::inference::Engine::load(&model_dir).unwrap();
    tokio::spawn(gigastt::server::run(engine, port, "127.0.0.1"));
    tokio::time::sleep(Duration::from_millis(200)).await;

    let (ws, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}/ws"))
        .await
        .unwrap();
    let (mut _sink, mut stream) = ws.split();

    // Should receive Ready message
    let msg = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("timeout waiting for Ready")
        .expect("stream ended")
        .expect("ws error");

    let text = msg.into_text().unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["type"], "ready");
    assert_eq!(v["version"], "1.0");
    assert_eq!(v["sample_rate"], 48000);
    assert!(v["model"].as_str().unwrap().contains("gigaam"));
    // Verify supported_rates is present and includes expected rates
    let rates = v["supported_rates"].as_array().expect("supported_rates missing");
    assert!(rates.len() >= 5);
    assert!(rates.contains(&serde_json::json!(8000)));
    assert!(rates.contains(&serde_json::json!(48000)));
}

#[tokio::test]
#[ignore] // Requires model download
async fn test_four_clients_connect_concurrently() {
    let model_dir = model_dir().expect("Model not found. Run `cargo run -- download` first.");
    let port = free_port().await;

    let engine = gigastt::inference::Engine::load(&model_dir).unwrap();
    tokio::spawn(gigastt::server::run(engine, port, "127.0.0.1"));
    tokio::time::sleep(Duration::from_millis(200)).await;

    let url = format!("ws://127.0.0.1:{port}/ws");

    // Connect 4 clients in parallel
    let mut handles = Vec::new();
    for i in 0..4 {
        let url = url.clone();
        handles.push(tokio::spawn(async move {
            let (ws, _) = tokio_tungstenite::connect_async(&url).await.unwrap();
            let (mut sink, mut stream) = ws.split();

            // Should receive Ready
            let msg = tokio::time::timeout(Duration::from_secs(5), stream.next())
                .await
                .expect("timeout")
                .expect("stream ended")
                .expect("ws error");

            let text = msg.into_text().unwrap();
            let v: serde_json::Value = serde_json::from_str(&text).unwrap();
            assert_eq!(v["type"], "ready", "Client {i} did not receive Ready");

            // Send Stop
            let stop = serde_json::json!({"type": "stop"});
            sink.send(Message::Text(serde_json::to_string(&stop).unwrap().into()))
                .await
                .unwrap();

            // Should receive Final (flush response)
            let msg = tokio::time::timeout(Duration::from_secs(5), stream.next())
                .await
                .expect("timeout waiting for Final")
                .expect("stream ended")
                .expect("ws error");

            let text = msg.into_text().unwrap();
            let v: serde_json::Value = serde_json::from_str(&text).unwrap();
            assert_eq!(v["type"], "final", "Client {i} did not receive Final after Stop");

            i
        }));
    }

    // All 4 should complete without panic
    for handle in handles {
        let client_id = handle.await.expect("client task panicked");
        assert!(client_id < 4);
    }
}

#[tokio::test]
#[ignore] // Requires model download
async fn test_stop_message_closes_gracefully() {
    let model_dir = model_dir().expect("Model not found. Run `cargo run -- download` first.");
    let port = free_port().await;

    let engine = gigastt::inference::Engine::load(&model_dir).unwrap();
    tokio::spawn(gigastt::server::run(engine, port, "127.0.0.1"));
    tokio::time::sleep(Duration::from_millis(200)).await;

    let (ws, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}/ws"))
        .await
        .unwrap();
    let (mut sink, mut stream) = ws.split();

    // Receive Ready
    let _ = stream.next().await;

    // Send some synthetic PCM16 audio (silence at 48kHz, 100ms = 4800 samples = 9600 bytes)
    let silence: Vec<u8> = vec![0u8; 9600];
    sink.send(Message::Binary(silence.into())).await.unwrap();

    // Send Stop
    let stop = serde_json::json!({"type": "stop"});
    sink.send(Message::Text(serde_json::to_string(&stop).unwrap().into()))
        .await
        .unwrap();

    // Should receive Final
    let msg = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("timeout waiting for Final")
        .expect("stream ended")
        .expect("ws error");

    let text = msg.into_text().unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["type"], "final");
}

#[tokio::test]
#[ignore] // Requires model download
async fn test_sse_ttfe_under_threshold() {
    let model_dir = model_dir().expect("Model not found. Run `cargo run -- download` first.");
    let port = free_port().await;

    let engine = gigastt::inference::Engine::load(&model_dir).unwrap();
    tokio::spawn(gigastt::server::run(engine, port, "127.0.0.1"));
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Generate 10s of synthetic 16kHz PCM16 silence as a WAV file in memory
    let sample_rate: u32 = 16000;
    let duration_s: u32 = 10;
    let num_samples = sample_rate * duration_s;
    let data_size = num_samples * 2; // 16-bit = 2 bytes per sample
    let file_size = 44 + data_size;

    let mut wav = Vec::with_capacity(file_size as usize);
    // RIFF header
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(file_size - 8).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    // fmt chunk
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    wav.extend_from_slice(&2u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    // data chunk
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());
    // Add a tiny sine wave instead of pure silence so the encoder has something
    for i in 0..num_samples {
        let sample = (440.0_f64 * 2.0 * std::f64::consts::PI * i as f64 / sample_rate as f64).sin() * 1000.0;
        wav.extend_from_slice(&(sample as i16).to_le_bytes());
    }

    let client = reqwest::Client::new();
    let start = std::time::Instant::now();
    let response = client
        .post(format!("http://127.0.0.1:{port}/v1/transcribe/stream"))
        .body(wav)
        .send()
        .await
        .expect("Failed to send SSE request");

    assert_eq!(response.status(), 200, "SSE endpoint returned non-200");

    // Read first SSE event
    let mut stream = response.bytes_stream();
    let first_chunk = tokio::time::timeout(Duration::from_secs(10), stream.next())
        .await
        .expect("Timeout waiting for first SSE event")
        .expect("Stream ended without events")
        .expect("Error reading SSE chunk");

    let ttfe = start.elapsed();
    let threshold_ms: u64 = std::option_env!("GIGASTT_TTFE_THRESHOLD_MS")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);

    eprintln!("TTFE: {}ms (threshold: {}ms)", ttfe.as_millis(), threshold_ms);
    eprintln!("First chunk ({} bytes): {:?}", first_chunk.len(), String::from_utf8_lossy(&first_chunk[..first_chunk.len().min(200)]));

    assert!(
        ttfe.as_millis() < threshold_ms as u128,
        "TTFE {ttfe:?} exceeded threshold {threshold_ms}ms"
    );
}

#[tokio::test]
#[ignore] // Requires model download
async fn test_four_concurrent_ws_with_audio() {
    let model_dir = model_dir().expect("Model not found. Run `cargo run -- download` first.");
    let port = free_port().await;

    let engine = gigastt::inference::Engine::load(&model_dir).unwrap();
    tokio::spawn(gigastt::server::run(engine, port, "127.0.0.1"));
    tokio::time::sleep(Duration::from_millis(200)).await;

    let url = format!("ws://127.0.0.1:{port}/ws");

    // Generate 2s of silence PCM16 at 48kHz (default rate)
    let silence: Vec<u8> = vec![0u8; 48000 * 2 * 2]; // 2 seconds * 2 bytes/sample

    let mut handles = Vec::new();
    for i in 0..4 {
        let url = url.clone();
        let audio = silence.clone();
        handles.push(tokio::spawn(async move {
            let (ws, _) = tokio_tungstenite::connect_async(&url).await.unwrap();
            let (mut sink, mut stream) = ws.split();

            // Receive Ready
            let msg = tokio::time::timeout(Duration::from_secs(10), stream.next())
                .await
                .expect("timeout")
                .expect("stream ended")
                .expect("ws error");
            let text = msg.into_text().unwrap();
            let v: serde_json::Value = serde_json::from_str(&text).unwrap();
            assert_eq!(v["type"], "ready", "Client {i} did not receive Ready");

            // Send audio in chunks
            for chunk in audio.chunks(9600) {
                sink.send(Message::Binary(chunk.to_vec().into()))
                    .await
                    .unwrap();
            }

            // Send Stop
            let stop = serde_json::json!({"type": "stop"});
            sink.send(Message::Text(serde_json::to_string(&stop).unwrap().into()))
                .await
                .unwrap();

            // Should receive Final
            let msg = tokio::time::timeout(Duration::from_secs(30), stream.next())
                .await
                .expect("timeout waiting for Final")
                .expect("stream ended")
                .expect("ws error");
            let text = msg.into_text().unwrap();
            let v: serde_json::Value = serde_json::from_str(&text).unwrap();
            assert_eq!(v["type"], "final", "Client {i} did not receive Final");

            i
        }));
    }

    // All 4 should complete without panic or deadlock
    for handle in handles {
        let client_id = tokio::time::timeout(Duration::from_secs(60), handle)
            .await
            .expect("Client task timed out (possible deadlock)")
            .expect("Client task panicked");
        assert!(client_id < 4);
    }
}

#[tokio::test]
#[ignore] // Requires model download
async fn test_configure_invalid_sample_rate() {
    let model_dir = model_dir().expect("Model not found. Run `cargo run -- download` first.");
    let port = free_port().await;

    let engine = gigastt::inference::Engine::load(&model_dir).unwrap();
    tokio::spawn(gigastt::server::run(engine, port, "127.0.0.1"));
    tokio::time::sleep(Duration::from_millis(200)).await;

    let (ws, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}/ws"))
        .await
        .unwrap();
    let (mut sink, mut stream) = ws.split();

    // Receive Ready
    let _ = stream.next().await;

    // Send Configure with invalid sample rate
    let configure = serde_json::json!({"type": "configure", "sample_rate": 7000});
    sink.send(Message::Text(serde_json::to_string(&configure).unwrap().into()))
        .await
        .unwrap();

    // Should receive Error with code "invalid_sample_rate"
    let msg = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("timeout waiting for Error")
        .expect("stream ended")
        .expect("ws error");

    let text = msg.into_text().unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["type"], "error");
    assert_eq!(v["code"], "invalid_sample_rate");
}
