//! Integration tests for the WebSocket server protocol.
//!
//! These tests require the GigaAM model to be downloaded (~850MB).
//! Run with: `cargo test --test server_integration -- --ignored`

use futures_util::{SinkExt, StreamExt};
use std::time::Duration;
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message;

/// Find a free port by binding to port 0.
async fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    listener.local_addr().unwrap().port()
}

/// Check if the model is available.
fn model_dir() -> Option<String> {
    let dir = dirs::home_dir()?.join(".gigastt").join("models");
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

    let (ws, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}"))
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

    let url = format!("ws://127.0.0.1:{port}");

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
            sink.send(Message::Text(serde_json::to_string(&stop).unwrap()))
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

    let (ws, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}"))
        .await
        .unwrap();
    let (mut sink, mut stream) = ws.split();

    // Receive Ready
    let _ = stream.next().await;

    // Send some synthetic PCM16 audio (silence at 48kHz, 100ms = 4800 samples = 9600 bytes)
    let silence: Vec<u8> = vec![0u8; 9600];
    sink.send(Message::Binary(silence)).await.unwrap();

    // Send Stop
    let stop = serde_json::json!({"type": "stop"});
    sink.send(Message::Text(serde_json::to_string(&stop).unwrap()))
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
async fn test_configure_invalid_sample_rate() {
    let model_dir = model_dir().expect("Model not found. Run `cargo run -- download` first.");
    let port = free_port().await;

    let engine = gigastt::inference::Engine::load(&model_dir).unwrap();
    tokio::spawn(gigastt::server::run(engine, port, "127.0.0.1"));
    tokio::time::sleep(Duration::from_millis(200)).await;

    let (ws, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}"))
        .await
        .unwrap();
    let (mut sink, mut stream) = ws.split();

    // Receive Ready
    let _ = stream.next().await;

    // Send Configure with invalid sample rate
    let configure = serde_json::json!({"type": "configure", "sample_rate": 7000});
    sink.send(Message::Text(serde_json::to_string(&configure).unwrap()))
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
