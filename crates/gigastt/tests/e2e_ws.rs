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

#[ignore]
#[tokio::test]
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

#[ignore]
#[tokio::test]
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

#[ignore]
#[tokio::test]
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

#[ignore]
#[tokio::test]
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

#[ignore]
#[tokio::test]
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

#[ignore]
#[tokio::test]
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

#[ignore]
#[tokio::test]
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

#[ignore]
#[tokio::test]
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

#[ignore]
#[tokio::test]
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

#[ignore]
#[tokio::test]
async fn test_ws_empty_frame_spam_closes_connection() {
    let model_dir = common::model_dir();
    let (port, _shutdown) = common::start_server(&model_dir).await;

    let (mut sink, mut stream) = {
        let (ws, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}/v1/ws"))
            .await
            .expect("WS connect failed");
        ws.split()
    };

    // Receive Ready
    let msg = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("timeout")
        .expect("stream ended")
        .expect("ws error");
    let text = msg.into_text().unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["type"], "ready");

    // Send many empty binary frames to trigger the spam limit.
    for _ in 0..1002 {
        sink.send(tokio_tungstenite::tungstenite::Message::Binary(
            vec![].into(),
        ))
        .await
        .expect("send empty frame");
    }

    // Server should close with an error or close frame.
    let msg = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("timeout waiting for error/close");

    match msg {
        Some(Ok(tokio_tungstenite::tungstenite::Message::Text(text))) => {
            let v: serde_json::Value = serde_json::from_str(&text).unwrap();
            assert_eq!(v["type"], "error");
            assert_eq!(v["code"], "policy_violation");
        }
        Some(Ok(tokio_tungstenite::tungstenite::Message::Close(_))) => {}
        other => panic!("Expected error or close, got: {other:?}"),
    }
}

#[ignore]
#[tokio::test]
async fn test_ws_configure_protocol_version_mismatch() {
    let model_dir = common::model_dir();
    let (port, _shutdown) = common::start_server(&model_dir).await;

    let (mut sink, mut stream) = {
        let (ws, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}/v1/ws"))
            .await
            .expect("WS connect failed");
        ws.split()
    };

    // Receive Ready
    let msg = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("timeout")
        .expect("stream ended")
        .expect("ws error");
    let text = msg.into_text().unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["type"], "ready");

    // Configure with an unsupported protocol version
    sink.send(Message::Text(
        serde_json::to_string(&serde_json::json!({
            "type": "configure",
            "protocol_version": "0.1",
        }))
        .unwrap()
        .into(),
    ))
    .await
    .expect("send configure");

    let msg = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("timeout waiting for protocol version error")
        .expect("stream ended")
        .expect("ws error");
    let text = msg.into_text().unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["type"], "error");
    assert_eq!(v["code"], "unsupported_protocol_version");
}

// ---------------------------------------------------------------------------
// 12. Unrecognized text message is ignored (covers the Ok(_) => Continue path)
// ---------------------------------------------------------------------------

#[ignore]
#[tokio::test]
async fn test_ws_unrecognized_text_message_ignored() {
    let model_dir = common::model_dir();
    let (port, _shutdown) = common::start_server(&model_dir).await;

    let (mut sink, mut stream) = {
        let (ws, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}/v1/ws"))
            .await
            .expect("WS connect failed");
        ws.split()
    };

    // Consume Ready
    let _ = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("timeout")
        .expect("stream ended")
        .expect("ws error");

    // Send a text message that is valid JSON but not a recognized ClientMessage variant
    sink.send(Message::Text(
        serde_json::json!({ "type": "unknown_command", "payload": 42 })
            .to_string()
            .into(),
    ))
    .await
    .expect("send text");

    // Send a minimal audio frame so the server has something to process,
    // then stop. This proves the unrecognized message did not kill the session.
    let chunk = common::generate_pcm16_silence(0.02, 48000);
    sink.send(Message::Binary(chunk.into())).await.unwrap();

    sink.send(Message::Text(
        serde_json::json!({ "type": "stop" }).to_string().into(),
    ))
    .await
    .unwrap();

    let msg = tokio::time::timeout(Duration::from_secs(10), stream.next())
        .await
        .expect("timeout waiting for final")
        .expect("stream ended")
        .expect("ws error");
    let text = msg.into_text().unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["type"], "final");
}

// ---------------------------------------------------------------------------
// 13. Client sends a Close frame — server should break cleanly
// ---------------------------------------------------------------------------

#[ignore]
#[tokio::test]
async fn test_ws_client_close_frame_ends_session() {
    let model_dir = common::model_dir();
    let (port, _shutdown) = common::start_server(&model_dir).await;

    let (mut sink, mut stream) = {
        let (ws, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}/v1/ws"))
            .await
            .expect("WS connect failed");
        ws.split()
    };

    // Consume Ready
    let _ = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("timeout")
        .expect("stream ended")
        .expect("ws error");

    // Send Close frame from client side
    sink.send(Message::Close(None)).await.expect("send close");

    // Server should end the stream; nothing more should arrive.
    let next = tokio::time::timeout(Duration::from_secs(3), stream.next()).await;
    match next {
        Ok(None) | Ok(Some(Ok(Message::Close(_)))) | Ok(Some(Err(_))) => {}
        other => panic!("Expected stream end after client Close, got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 15. Rapid frames during short max_session_secs hit the pre-check branch
// ---------------------------------------------------------------------------

#[ignore]
#[tokio::test]
async fn test_ws_max_session_precheck() {
    let model_dir = common::model_dir();
    let limits = gigastt::server::RuntimeLimits {
        max_session_secs: 1,
        idle_timeout_secs: 5,
        ..Default::default()
    };
    let (port, _shutdown) = common::start_server_with_limits(&model_dir, limits).await;

    let (mut sink, mut stream) = {
        let (ws, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}/v1/ws"))
            .await
            .expect("WS connect failed");
        ws.split()
    };

    // Consume Ready
    let _ = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("timeout")
        .expect("stream ended")
        .expect("ws error");

    // Stream frames rapidly (every 50 ms) so the pre-check at the top of the
    // loop fires before the select! sleep_until branch can time out.
    let chunk = common::generate_pcm16_silence(0.02, 48000);
    let stream_task = tokio::spawn(async move {
        for _ in 0..200 {
            if sink
                .send(Message::Binary(chunk.clone().into()))
                .await
                .is_err()
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    });

    // We expect either max_session_duration_exceeded error or a Close(1008).
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    let mut saw_cap_close = false;

    while tokio::time::Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        let next = tokio::time::timeout(remaining, stream.next()).await;
        match next {
            Ok(Some(Ok(Message::Text(text)))) => {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text)
                    && v["code"] == "max_session_duration_exceeded"
                {
                    saw_cap_close = true;
                }
            }
            Ok(Some(Ok(Message::Close(Some(
                tokio_tungstenite::tungstenite::protocol::frame::CloseFrame { code, .. },
            ))))) => {
                if u16::from(code) == 1008 {
                    saw_cap_close = true;
                }
                break;
            }
            Ok(None) | Ok(Some(Err(_))) => break,
            Err(_) => break,
            _ => continue,
        }
    }

    stream_task.abort();
    let _ = stream_task.await;

    assert!(
        saw_cap_close,
        "Must hit max session cap via pre-check or sleep_until"
    );
}
