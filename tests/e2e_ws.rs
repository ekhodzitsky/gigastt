//! End-to-end WebSocket protocol tests.
//!
//! All tests require the GigaAM ONNX model to be downloaded (~850MB).
//! Run with: `cargo test --test e2e_ws -- --ignored`

mod common;

use futures_util::{SinkExt, StreamExt};
use std::time::Duration;
use tokio_tungstenite::tungstenite::Message;

// ---------------------------------------------------------------------------
// 1. Ready message validation
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore] // Requires model download
async fn test_ws_connect_receives_ready() {
    let model_dir = common::model_dir();
    let (port, _shutdown) = common::start_server(&model_dir).await;

    let (_sink, _stream, ready) = common::ws_connect(port).await;

    assert_eq!(ready["type"], "ready");
    assert_eq!(ready["version"], "1.0");
    assert_eq!(ready["sample_rate"], 48000);
    assert!(
        ready["model"].as_str().unwrap().contains("gigaam"),
        "model field should contain 'gigaam', got: {:?}",
        ready["model"]
    );

    let rates = ready["supported_rates"]
        .as_array()
        .expect("supported_rates should be an array");
    assert!(
        rates.len() >= 5,
        "supported_rates should have >=5 entries, got {}",
        rates.len()
    );
    assert!(
        rates.contains(&serde_json::json!(8000)),
        "supported_rates should contain 8000"
    );
    assert!(
        rates.contains(&serde_json::json!(48000)),
        "supported_rates should contain 48000"
    );
}

// ---------------------------------------------------------------------------
// 2. Audio → Final
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore] // Requires model download
async fn test_ws_audio_produces_final() {
    let model_dir = common::model_dir();
    let (port, _shutdown) = common::start_server(&model_dir).await;

    let (mut sink, mut stream, _ready) = common::ws_connect(port).await;

    // 2 seconds of PCM16 silence at 48kHz = 192000 bytes
    let silence = common::generate_pcm16_silence(2.0, 48000);
    for chunk in silence.chunks(9600) {
        sink.send(Message::Binary(chunk.to_vec().into()))
            .await
            .unwrap();
    }

    // Send Stop
    sink.send(Message::Text(
        serde_json::to_string(&serde_json::json!({"type": "stop"}))
            .unwrap()
            .into(),
    ))
    .await
    .unwrap();

    // Drain any Partial messages; we only care about Final
    loop {
        let msg = tokio::time::timeout(Duration::from_secs(30), stream.next())
            .await
            .expect("timeout waiting for Final")
            .expect("stream ended")
            .expect("ws error");

        let text = msg.into_text().expect("expected text message");
        let v: serde_json::Value = serde_json::from_str(&text).expect("expected JSON");
        match v["type"].as_str().unwrap_or("") {
            "partial" => continue,
            "final" => {
                assert!(
                    v["text"].is_string(),
                    "Final message should have a text field"
                );
                break;
            }
            other => panic!("Unexpected message type: {other}, full: {text}"),
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Stop without audio → Final with empty text
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore] // Requires model download
async fn test_ws_stop_without_audio() {
    let model_dir = common::model_dir();
    let (port, _shutdown) = common::start_server(&model_dir).await;

    let (mut sink, mut stream, _ready) = common::ws_connect(port).await;

    sink.send(Message::Text(
        serde_json::to_string(&serde_json::json!({"type": "stop"}))
            .unwrap()
            .into(),
    ))
    .await
    .unwrap();

    let msg = tokio::time::timeout(Duration::from_secs(10), stream.next())
        .await
        .expect("timeout waiting for Final")
        .expect("stream ended")
        .expect("ws error");

    let v = common::assert_msg_type(msg, "final");
    assert_eq!(
        v["text"].as_str().unwrap_or(""),
        "",
        "Expected empty text for stop-without-audio"
    );
}

// ---------------------------------------------------------------------------
// 4. Configure with valid sample rate → Final (no error)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore] // Requires model download
async fn test_ws_configure_valid_sample_rate() {
    let model_dir = common::model_dir();
    let (port, _shutdown) = common::start_server(&model_dir).await;

    let (mut sink, mut stream, _ready) = common::ws_connect(port).await;

    // Configure to 16kHz
    sink.send(Message::Text(
        serde_json::to_string(&serde_json::json!({"type": "configure", "sample_rate": 16000}))
            .unwrap()
            .into(),
    ))
    .await
    .unwrap();

    // 1 second of PCM16 silence at 16kHz = 32000 bytes
    let silence = common::generate_pcm16_silence(1.0, 16000);
    sink.send(Message::Binary(silence.into())).await.unwrap();

    // Send Stop
    sink.send(Message::Text(
        serde_json::to_string(&serde_json::json!({"type": "stop"}))
            .unwrap()
            .into(),
    ))
    .await
    .unwrap();

    // Drain Partials, expect Final (not Error)
    loop {
        let msg = tokio::time::timeout(Duration::from_secs(20), stream.next())
            .await
            .expect("timeout waiting for Final")
            .expect("stream ended")
            .expect("ws error");

        let text = msg.into_text().expect("expected text message");
        let v: serde_json::Value = serde_json::from_str(&text).expect("expected JSON");
        match v["type"].as_str().unwrap_or("") {
            "partial" => continue,
            "final" => break,
            other => panic!("Unexpected message type: {other} (expected final, not error)"),
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Configure with invalid sample rate → Error
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore] // Requires model download
async fn test_ws_configure_invalid_sample_rate() {
    let model_dir = common::model_dir();
    let (port, _shutdown) = common::start_server(&model_dir).await;

    let (mut sink, mut stream, _ready) = common::ws_connect(port).await;

    sink.send(Message::Text(
        serde_json::to_string(&serde_json::json!({"type": "configure", "sample_rate": 7000}))
            .unwrap()
            .into(),
    ))
    .await
    .unwrap();

    let msg = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("timeout waiting for Error")
        .expect("stream ended")
        .expect("ws error");

    let v = common::assert_msg_type(msg, "error");
    assert_eq!(
        v["code"], "invalid_sample_rate",
        "Expected code=invalid_sample_rate, got: {:?}",
        v["code"]
    );
}

// ---------------------------------------------------------------------------
// 6. Configure after audio has been sent → Error
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore] // Requires model download
async fn test_ws_configure_after_audio() {
    let model_dir = common::model_dir();
    let (port, _shutdown) = common::start_server(&model_dir).await;

    let (mut sink, mut stream, _ready) = common::ws_connect(port).await;

    // Send some audio first
    let silence = common::generate_pcm16_silence(0.1, 48000);
    sink.send(Message::Binary(silence.into())).await.unwrap();

    // Now try to configure — should be rejected
    sink.send(Message::Text(
        serde_json::to_string(&serde_json::json!({"type": "configure", "sample_rate": 16000}))
            .unwrap()
            .into(),
    ))
    .await
    .unwrap();

    let msg = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("timeout waiting for Error")
        .expect("stream ended")
        .expect("ws error");

    let v = common::assert_msg_type(msg, "error");
    assert_eq!(
        v["code"], "configure_too_late",
        "Expected code=configure_too_late, got: {:?}",
        v["code"]
    );
}

// ---------------------------------------------------------------------------
// 7. Malformed JSON → connection stays alive, Stop still works
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore] // Requires model download
async fn test_ws_malformed_json() {
    let model_dir = common::model_dir();
    let (port, _shutdown) = common::start_server(&model_dir).await;

    let (mut sink, mut stream, _ready) = common::ws_connect(port).await;

    // Send garbage text that is not valid JSON
    sink.send(Message::Text("not json at all {{".to_string().into()))
        .await
        .unwrap();

    // Connection must NOT be closed; send Stop and expect Final
    sink.send(Message::Text(
        serde_json::to_string(&serde_json::json!({"type": "stop"}))
            .unwrap()
            .into(),
    ))
    .await
    .unwrap();

    // Drain until Final (server silently ignores malformed messages)
    loop {
        let msg = tokio::time::timeout(Duration::from_secs(10), stream.next())
            .await
            .expect("timeout — connection may have been closed by malformed JSON")
            .expect("stream ended unexpectedly after malformed JSON")
            .expect("ws error");

        let text = msg.into_text().expect("expected text message");
        let v: serde_json::Value = serde_json::from_str(&text).expect("expected JSON");
        match v["type"].as_str().unwrap_or("") {
            "partial" => continue,
            "final" => break,
            other => panic!("Unexpected message type after malformed JSON: {other}"),
        }
    }
}

// ---------------------------------------------------------------------------
// 8. Client disconnect mid-stream → server remains healthy
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore] // Requires model download
async fn test_ws_client_disconnect_midstream() {
    let model_dir = common::model_dir();
    let (port, _shutdown) = common::start_server(&model_dir).await;

    // First client: send audio then abruptly disconnect (drop sink + stream)
    {
        let (mut sink, _stream, _ready) = common::ws_connect(port).await;
        let silence = common::generate_pcm16_silence(0.5, 48000);
        sink.send(Message::Binary(silence.into())).await.unwrap();
        // Dropped here — abrupt disconnect without sending Close frame
    }

    // Give server a moment to detect the disconnect
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Verify server is still healthy: a new client should connect and receive Ready
    let (_sink2, _stream2, ready2) = common::ws_connect(port).await;
    assert_eq!(
        ready2["type"], "ready",
        "Server should still be healthy after abrupt client disconnect"
    );
}

// ---------------------------------------------------------------------------
// 9. Four concurrent clients — all receive Ready and Final
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore] // Requires model download
async fn test_ws_concurrent_4_clients() {
    let model_dir = common::model_dir();
    let (port, _shutdown) = common::start_server(&model_dir).await;

    let url = format!("ws://127.0.0.1:{port}/v1/ws");

    let mut handles = Vec::new();
    for i in 0..4usize {
        let url = url.clone();
        handles.push(tokio::spawn(async move {
            let (ws, _) = tokio_tungstenite::connect_async(&url)
                .await
                .unwrap_or_else(|e| panic!("Client {i} failed to connect: {e}"));
            let (mut sink, mut stream) = ws.split();

            // Should receive Ready
            let msg = tokio::time::timeout(Duration::from_secs(10), stream.next())
                .await
                .expect("timeout waiting for Ready")
                .expect("stream ended")
                .expect("ws error");
            let text = msg.into_text().unwrap();
            let v: serde_json::Value = serde_json::from_str(&text).unwrap();
            assert_eq!(v["type"], "ready", "Client {i} did not receive Ready");

            // Send Stop
            sink.send(Message::Text(
                serde_json::to_string(&serde_json::json!({"type": "stop"}))
                    .unwrap()
                    .into(),
            ))
            .await
            .unwrap();

            // Should receive Final
            let msg = tokio::time::timeout(Duration::from_secs(10), stream.next())
                .await
                .expect("timeout waiting for Final")
                .expect("stream ended")
                .expect("ws error");
            let text = msg.into_text().unwrap();
            let v: serde_json::Value = serde_json::from_str(&text).unwrap();
            assert_eq!(
                v["type"], "final",
                "Client {i} did not receive Final after Stop"
            );

            i
        }));
    }

    for handle in handles {
        let client_id = tokio::time::timeout(Duration::from_secs(30), handle)
            .await
            .expect("client task timed out")
            .expect("client task panicked");
        assert!(client_id < 4);
    }
}
