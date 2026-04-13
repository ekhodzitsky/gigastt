//! Load tests for the gigastt server.
//!
//! All tests require the GigaAM ONNX model to be downloaded (~850MB).
//! Run with: `cargo test --test load_test -- --ignored`

mod common;

use futures_util::{SinkExt, StreamExt};
use std::time::{Duration, Instant};
use tokio_tungstenite::tungstenite::Message;

// ---------------------------------------------------------------------------
// 1. 4 concurrent WebSocket clients streaming 5s of audio
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore] // Requires model download — run locally
async fn test_load_4_concurrent_ws_streaming() {
    let model_dir = common::model_dir();
    let (port, _shutdown) = common::start_server(&model_dir).await;

    let start = Instant::now();
    let mut handles = Vec::new();

    for i in 0..4usize {
        handles.push(tokio::spawn(async move {
            let (mut sink, mut stream, _ready) = common::ws_connect(port).await;

            // 5s of PCM16 silence at 48kHz = 480000 bytes, sent in 9600-byte chunks
            let silence = common::generate_pcm16_silence(5.0, 48000);
            for chunk in silence.chunks(9600) {
                sink.send(Message::Binary(chunk.to_vec()))
                    .await
                    .unwrap_or_else(|e| panic!("Client {i}: send audio chunk failed: {e}"));
            }

            // Send Stop
            sink.send(Message::Text(
                serde_json::to_string(&serde_json::json!({"type": "stop"})).unwrap(),
            ))
            .await
            .unwrap_or_else(|e| panic!("Client {i}: send stop failed: {e}"));

            // Drain Partials, wait for Final
            loop {
                let msg = tokio::time::timeout(Duration::from_secs(90), stream.next())
                    .await
                    .unwrap_or_else(|_| panic!("Client {i}: timed out waiting for Final"))
                    .unwrap_or_else(|| panic!("Client {i}: stream ended before Final"))
                    .unwrap_or_else(|e| panic!("Client {i}: ws error: {e}"));

                let text = msg
                    .into_text()
                    .unwrap_or_else(|_| panic!("Client {i}: expected text message"));
                let v: serde_json::Value = serde_json::from_str(&text)
                    .unwrap_or_else(|_| panic!("Client {i}: invalid JSON: {text}"));

                match v["type"].as_str().unwrap_or("") {
                    "partial" => continue,
                    "final" => break,
                    other => panic!("Client {i}: unexpected message type: {other}"),
                }
            }

            Instant::now()
        }));
    }

    let mut error_count = 0usize;
    for (i, handle) in handles.into_iter().enumerate() {
        match tokio::time::timeout(Duration::from_secs(120), handle).await {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => {
                eprintln!("Client {i} task panicked: {e}");
                error_count += 1;
            }
            Err(_) => {
                eprintln!("Client {i} timed out after 120s");
                error_count += 1;
            }
        }
    }

    let elapsed = start.elapsed();
    eprintln!("All 4 WS streaming clients completed in {elapsed:?}");
    assert_eq!(error_count, 0, "{error_count} client(s) failed");
    assert!(
        elapsed < Duration::from_secs(120),
        "Total elapsed {elapsed:?} exceeded 120s limit"
    );
}

// ---------------------------------------------------------------------------
// 2. 4 concurrent REST POST /v1/transcribe clients
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore] // Requires model download — run locally
async fn test_load_4_concurrent_rest_transcribe() {
    let model_dir = common::model_dir();
    let (port, _shutdown) = common::start_server(&model_dir).await;

    let start = Instant::now();
    let mut handles = Vec::new();

    for i in 0..4usize {
        // Generate 5s WAV at 16kHz per task
        let wav = common::generate_wav(5, 16000);

        handles.push(tokio::spawn(async move {
            let resp = reqwest::Client::new()
                .post(format!("http://127.0.0.1:{port}/v1/transcribe"))
                .body(wav)
                .send()
                .await
                .unwrap_or_else(|e| panic!("Client {i}: POST /v1/transcribe failed: {e}"));

            let status = resp.status();
            assert_eq!(
                status,
                200,
                "Client {i}: expected 200, got {status}"
            );

            i
        }));
    }

    let mut error_count = 0usize;
    for (i, handle) in handles.into_iter().enumerate() {
        match tokio::time::timeout(Duration::from_secs(120), handle).await {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => {
                eprintln!("Client {i} task panicked: {e}");
                error_count += 1;
            }
            Err(_) => {
                eprintln!("Client {i} timed out after 120s");
                error_count += 1;
            }
        }
    }

    let elapsed = start.elapsed();
    eprintln!("All 4 REST transcribe clients completed in {elapsed:?}");
    assert_eq!(error_count, 0, "{error_count} client(s) failed");
    assert!(
        elapsed < Duration::from_secs(120),
        "Total elapsed {elapsed:?} exceeded 120s limit"
    );
}

// ---------------------------------------------------------------------------
// 3. Burst of 20 WebSocket connections (pool is 4, rest will queue)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore] // Requires model download — run locally
async fn test_load_burst_20_connections() {
    let model_dir = common::model_dir();
    let (port, _shutdown) = common::start_server(&model_dir).await;

    let start = Instant::now();
    let mut handles = Vec::new();

    // Spawn all 20 tasks immediately with no delay — connections will queue
    // because the server pool is 4. That is expected behaviour.
    for i in 0..20usize {
        handles.push(tokio::spawn(async move {
            let (mut sink, mut stream, _ready) = common::ws_connect(port).await;

            // Send Stop immediately — no audio
            sink.send(Message::Text(
                serde_json::to_string(&serde_json::json!({"type": "stop"})).unwrap(),
            ))
            .await
            .unwrap_or_else(|e| panic!("Client {i}: send stop failed: {e}"));

            // Wait for Final
            loop {
                let msg = tokio::time::timeout(Duration::from_secs(25), stream.next())
                    .await
                    .unwrap_or_else(|_| panic!("Client {i}: timed out waiting for Final"))
                    .unwrap_or_else(|| panic!("Client {i}: stream ended before Final"))
                    .unwrap_or_else(|e| panic!("Client {i}: ws error: {e}"));

                let text = msg
                    .into_text()
                    .unwrap_or_else(|_| panic!("Client {i}: expected text message"));
                let v: serde_json::Value = serde_json::from_str(&text)
                    .unwrap_or_else(|_| panic!("Client {i}: invalid JSON: {text}"));

                match v["type"].as_str().unwrap_or("") {
                    "partial" => continue,
                    "final" => break,
                    other => panic!("Client {i}: unexpected message type: {other}"),
                }
            }

            i
        }));
    }

    let mut error_count = 0usize;
    for (i, handle) in handles.into_iter().enumerate() {
        match tokio::time::timeout(Duration::from_secs(30), handle).await {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => {
                eprintln!("Client {i} task panicked: {e}");
                error_count += 1;
            }
            Err(_) => {
                eprintln!("Client {i} timed out after 30s");
                error_count += 1;
            }
        }
    }

    let elapsed = start.elapsed();
    eprintln!("All 20 burst WS connections completed in {elapsed:?}");
    assert_eq!(error_count, 0, "{error_count} client(s) failed or panicked");
    assert!(
        elapsed < Duration::from_secs(30),
        "Total elapsed {elapsed:?} exceeded 30s limit"
    );
}
