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

    // Session caps are always advertised (server defaults: 3600s session,
    // 300s idle) so clients can plan reconnects before a close frame.
    assert_eq!(ready["max_session_secs"], 3600);
    assert_eq!(ready["idle_timeout_secs"], 300);
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
    // Explicit pool of 4 so all 4 concurrent clients get a slot — the default
    // pool size is 2 since v2.3, which would hang clients 3 and 4.
    let (port, _shutdown) = common::start_server_with_pool(&model_dir, 4).await;

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

// ---------------------------------------------------------------------------
// 16. Every supported sample rate configures cleanly and yields a Final
// ---------------------------------------------------------------------------

/// Drive one full configure→audio→stop cycle at the given client sample rate
/// and assert the session ends with a Final (never an Error). Exercises the
/// resample-vs-passthrough split inside `handle_binary_frame` for each rate.
async fn run_sample_rate_roundtrip(port: u16, rate: u32) {
    let (mut sink, mut stream, _ready) = common::ws_connect(port).await;

    sink.send(Message::Text(
        serde_json::to_string(&serde_json::json!({"type": "configure", "sample_rate": rate}))
            .unwrap()
            .into(),
    ))
    .await
    .unwrap();

    // ~0.3 s of silence at the configured rate.
    let silence = common::generate_pcm16_silence(0.3, rate);
    sink.send(Message::Binary(silence.into())).await.unwrap();

    sink.send(Message::Text(
        serde_json::to_string(&serde_json::json!({"type": "stop"}))
            .unwrap()
            .into(),
    ))
    .await
    .unwrap();

    loop {
        let msg = tokio::time::timeout(Duration::from_secs(20), stream.next())
            .await
            .unwrap_or_else(|_| panic!("timeout waiting for Final at {rate}Hz"))
            .expect("stream ended")
            .expect("ws error");
        let text = msg.into_text().expect("expected text message");
        let v: serde_json::Value = serde_json::from_str(&text).expect("expected JSON");
        match v["type"].as_str().unwrap_or("") {
            "partial" => continue,
            "final" => break,
            other => panic!("Unexpected message type {other} at {rate}Hz (expected final): {text}"),
        }
    }
}

#[ignore]
#[tokio::test]
async fn test_ws_configure_8khz_roundtrip() {
    let model_dir = common::model_dir();
    let (port, _shutdown) = common::start_server(&model_dir).await;
    run_sample_rate_roundtrip(port, 8000).await;
}

#[ignore]
#[tokio::test]
async fn test_ws_configure_24khz_roundtrip() {
    let model_dir = common::model_dir();
    let (port, _shutdown) = common::start_server(&model_dir).await;
    run_sample_rate_roundtrip(port, 24000).await;
}

#[ignore]
#[tokio::test]
async fn test_ws_configure_44100hz_roundtrip() {
    let model_dir = common::model_dir();
    let (port, _shutdown) = common::start_server(&model_dir).await;
    run_sample_rate_roundtrip(port, 44100).await;
}

#[ignore]
#[tokio::test]
async fn test_ws_configure_48khz_roundtrip() {
    let model_dir = common::model_dir();
    let (port, _shutdown) = common::start_server(&model_dir).await;
    run_sample_rate_roundtrip(port, 48000).await;
}

// ---------------------------------------------------------------------------
// 17. Binary audio sent before any Configure uses the 48kHz default and works
// ---------------------------------------------------------------------------

#[ignore]
#[tokio::test]
async fn test_ws_audio_before_configure_uses_default_rate() {
    let model_dir = common::model_dir();
    let (port, _shutdown) = common::start_server(&model_dir).await;

    let (mut sink, mut stream, _ready) = common::ws_connect(port).await;

    // No Configure at all — server must fall back to DEFAULT_SAMPLE_RATE (48k)
    // and resample. 0.3 s of silence at 48kHz.
    let silence = common::generate_pcm16_silence(0.3, 48000);
    sink.send(Message::Binary(silence.into())).await.unwrap();

    sink.send(Message::Text(
        serde_json::to_string(&serde_json::json!({"type": "stop"}))
            .unwrap()
            .into(),
    ))
    .await
    .unwrap();

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
            other => panic!("Unexpected message type {other} (expected final): {text}"),
        }
    }
}

// ---------------------------------------------------------------------------
// 18. Stop, then more audio — Stop ends the session, so the socket closes
// ---------------------------------------------------------------------------

#[ignore]
#[tokio::test]
async fn test_ws_stop_then_more_audio_session_ends() {
    let model_dir = common::model_dir();
    let (port, _shutdown) = common::start_server(&model_dir).await;

    let (mut sink, mut stream, _ready) = common::ws_connect(port).await;

    // Send a little audio, then Stop — Stop breaks the session loop.
    let silence = common::generate_pcm16_silence(0.1, 48000);
    sink.send(Message::Binary(silence.into())).await.unwrap();
    sink.send(Message::Text(
        serde_json::to_string(&serde_json::json!({"type": "stop"}))
            .unwrap()
            .into(),
    ))
    .await
    .unwrap();

    // Drain Partials until the Final that Stop produced.
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
            other => panic!("Unexpected message type {other} after stop: {text}"),
        }
    }

    // The server breaks the loop after Stop, returning the triplet to the pool.
    // Any further audio we push must NOT produce another Final/Partial — the
    // stream is being torn down. Sending may itself fail once the close
    // propagates; either way we must not observe more transcript frames.
    let more = common::generate_pcm16_silence(0.1, 48000);
    let _ = sink.send(Message::Binary(more.into())).await;

    let next = tokio::time::timeout(Duration::from_secs(3), stream.next()).await;
    match next {
        Ok(None) | Ok(Some(Ok(Message::Close(_)))) | Ok(Some(Err(_))) | Err(_) => {}
        Ok(Some(Ok(Message::Text(text)))) => {
            let v: serde_json::Value = serde_json::from_str(&text).expect("expected JSON");
            panic!("Did not expect a transcript frame after Stop, got: {v}");
        }
        Ok(Some(Ok(_other))) => {}
    }
}

// ---------------------------------------------------------------------------
// 19. A few empty binary frames (below the spam cap) are skipped, not fatal
// ---------------------------------------------------------------------------

#[ignore]
#[tokio::test]
async fn test_ws_empty_frames_below_cap_are_skipped() {
    let model_dir = common::model_dir();
    let (port, _shutdown) = common::start_server(&model_dir).await;

    let (mut sink, mut stream, _ready) = common::ws_connect(port).await;

    // A handful of empty binary frames — well under MAX_EMPTY_FRAMES_PER_SESSION.
    for _ in 0..5 {
        sink.send(Message::Binary(vec![].into())).await.unwrap();
    }

    // Real audio still flows afterwards, and Stop produces a Final.
    let silence = common::generate_pcm16_silence(0.2, 48000);
    sink.send(Message::Binary(silence.into())).await.unwrap();
    sink.send(Message::Text(
        serde_json::to_string(&serde_json::json!({"type": "stop"}))
            .unwrap()
            .into(),
    ))
    .await
    .unwrap();

    loop {
        let msg = tokio::time::timeout(Duration::from_secs(20), stream.next())
            .await
            .expect("timeout waiting for Final after empty frames")
            .expect("stream ended")
            .expect("ws error");
        let text = msg.into_text().expect("expected text message");
        let v: serde_json::Value = serde_json::from_str(&text).expect("expected JSON");
        match v["type"].as_str().unwrap_or("") {
            "partial" => continue,
            "final" => break,
            other => panic!("Unexpected message type {other} after empty frames: {text}"),
        }
    }
}

// ---------------------------------------------------------------------------
// 20. Client Ping → session stays usable (server ignores ping/pong frames)
// ---------------------------------------------------------------------------

#[ignore]
#[tokio::test]
async fn test_ws_client_ping_keeps_session_alive() {
    let model_dir = common::model_dir();
    let (port, _shutdown) = common::start_server(&model_dir).await;

    let (mut sink, mut stream, _ready) = common::ws_connect(port).await;

    // tokio-tungstenite auto-replies to Pings with Pongs at the protocol layer,
    // so we assert liveness functionally: after a Ping the session is still
    // usable (server ignores Ping/Pong in its match arm and continues).
    sink.send(Message::Ping(vec![1, 2, 3].into()))
        .await
        .unwrap();

    // Drain any Pong the client surfaces, then prove the session still works.
    let silence = common::generate_pcm16_silence(0.2, 48000);
    sink.send(Message::Binary(silence.into())).await.unwrap();
    sink.send(Message::Text(
        serde_json::to_string(&serde_json::json!({"type": "stop"}))
            .unwrap()
            .into(),
    ))
    .await
    .unwrap();

    loop {
        let msg = tokio::time::timeout(Duration::from_secs(20), stream.next())
            .await
            .expect("timeout waiting for Final after ping")
            .expect("stream ended")
            .expect("ws error");
        // Pong frames are non-text; skip them.
        if !msg.is_text() {
            continue;
        }
        let text = msg.into_text().expect("expected text message");
        let v: serde_json::Value = serde_json::from_str(&text).expect("expected JSON");
        match v["type"].as_str().unwrap_or("") {
            "partial" => continue,
            "final" => break,
            other => panic!("Unexpected message type {other} after ping: {text}"),
        }
    }
}

// ---------------------------------------------------------------------------
// 21. Multiple Configure messages before audio — last one wins, no errors
// ---------------------------------------------------------------------------

#[ignore]
#[tokio::test]
async fn test_ws_multiple_configure_last_wins() {
    let model_dir = common::model_dir();
    let (port, _shutdown) = common::start_server(&model_dir).await;

    let (mut sink, mut stream, _ready) = common::ws_connect(port).await;

    // First configure 8kHz, then re-configure to 16kHz before any audio.
    for rate in [8000u32, 16000u32] {
        sink.send(Message::Text(
            serde_json::to_string(&serde_json::json!({"type": "configure", "sample_rate": rate}))
                .unwrap()
                .into(),
        ))
        .await
        .unwrap();
    }

    // Audio at 16kHz (the last-configured rate); 16kHz is the passthrough path.
    let silence = common::generate_pcm16_silence(0.3, 16000);
    sink.send(Message::Binary(silence.into())).await.unwrap();
    sink.send(Message::Text(
        serde_json::to_string(&serde_json::json!({"type": "stop"}))
            .unwrap()
            .into(),
    ))
    .await
    .unwrap();

    loop {
        let msg = tokio::time::timeout(Duration::from_secs(20), stream.next())
            .await
            .expect("timeout waiting for Final after multiple configure")
            .expect("stream ended")
            .expect("ws error");
        let text = msg.into_text().expect("expected text message");
        let v: serde_json::Value = serde_json::from_str(&text).expect("expected JSON");
        match v["type"].as_str().unwrap_or("") {
            "partial" => continue,
            "final" => break,
            other => panic!("Unexpected message type {other} after configure x2: {text}"),
        }
    }
}

// ---------------------------------------------------------------------------
// 22. Configure with a diarization field is accepted (no-op without feature)
// ---------------------------------------------------------------------------

#[ignore]
#[tokio::test]
async fn test_ws_configure_with_diarization_field() {
    let model_dir = common::model_dir();
    let (port, _shutdown) = common::start_server(&model_dir).await;

    let (mut sink, mut stream, _ready) = common::ws_connect(port).await;

    // diarization=false is a no-op in the default build but must not error and
    // must leave the session fully usable.
    sink.send(Message::Text(
        serde_json::to_string(&serde_json::json!({
            "type": "configure",
            "sample_rate": 16000,
            "diarization": false,
        }))
        .unwrap()
        .into(),
    ))
    .await
    .unwrap();

    let silence = common::generate_pcm16_silence(0.3, 16000);
    sink.send(Message::Binary(silence.into())).await.unwrap();
    sink.send(Message::Text(
        serde_json::to_string(&serde_json::json!({"type": "stop"}))
            .unwrap()
            .into(),
    ))
    .await
    .unwrap();

    loop {
        let msg = tokio::time::timeout(Duration::from_secs(20), stream.next())
            .await
            .expect("timeout waiting for Final after diarization configure")
            .expect("stream ended")
            .expect("ws error");
        let text = msg.into_text().expect("expected text message");
        let v: serde_json::Value = serde_json::from_str(&text).expect("expected JSON");
        match v["type"].as_str().unwrap_or("") {
            "partial" => continue,
            "final" => break,
            other => panic!("Unexpected message type {other} with diarization field: {text}"),
        }
    }
}

// ---------------------------------------------------------------------------
// 23. Non-empty audio produces a Final whose text field is a string
// ---------------------------------------------------------------------------

#[ignore]
#[tokio::test]
async fn test_ws_tone_audio_produces_final_text() {
    let model_dir = common::model_dir();
    let (port, _shutdown) = common::start_server(&model_dir).await;

    let (mut sink, mut stream, _ready) = common::ws_connect(port).await;

    // A real (non-silent) tone at 16kHz so the encoder genuinely runs.
    sink.send(Message::Text(
        serde_json::to_string(&serde_json::json!({"type": "configure", "sample_rate": 16000}))
            .unwrap()
            .into(),
    ))
    .await
    .unwrap();

    let tone = common::generate_pcm16_tone(0.5, 16000, 220.0);
    for chunk in tone.chunks(3200) {
        sink.send(Message::Binary(chunk.to_vec().into()))
            .await
            .unwrap();
    }
    sink.send(Message::Text(
        serde_json::to_string(&serde_json::json!({"type": "stop"}))
            .unwrap()
            .into(),
    ))
    .await
    .unwrap();

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
                assert!(v["text"].is_string(), "Final must carry a text field");
                break;
            }
            other => panic!("Unexpected message type {other}: {text}"),
        }
    }
}

// ---------------------------------------------------------------------------
// 24. Streaming finals get ITN + punctuation; partials stay raw
//
// The engine enriches a segment's joined `text` at finalization time only
// (endpoint flush / Stop flush): ITN, then punctuation/casing restoration.
// `words[]` payloads always keep the raw decoder output, and `partial`
// messages are never rewritten. The enrichment-positive tests assume the
// bare `rnnt` head (the default download since v2.3) and self-skip on
// `e2e_rnnt`, which is already punctuated by the model itself.
// ---------------------------------------------------------------------------

const GOLOS_FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/golos_00.wav");

type WsSink = futures_util::stream::SplitSink<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
    tokio_tungstenite::tungstenite::Message,
>;
type WsStream = futures_util::stream::SplitStream<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
>;

/// Decode the golos fixture (real Russian speech, contains the spoken number
/// "шестьдесят тысяч") to PCM16 bytes at 16 kHz for WS streaming.
fn golos_pcm16() -> Vec<u8> {
    let samples =
        gigastt::inference::audio::decode_audio_file(GOLOS_FIXTURE).expect("decode golos fixture");
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for s in samples {
        let v = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// GET `/health` as JSON (used to read the loaded head / post-processing
/// capabilities when deciding whether a test is meaningful).
async fn get_health(port: u16) -> serde_json::Value {
    let text = reqwest::Client::new()
        .get(format!("http://127.0.0.1:{port}/health"))
        .send()
        .await
        .expect("GET /health failed")
        .text()
        .await
        .expect("health body");
    serde_json::from_str(&text).expect("health JSON")
}

/// Load the punct model from the default dir, or return None (test self-skips).
fn load_punctuator_or_skip() -> Option<gigastt::punctuation::Punctuator> {
    let punct_dir = common::home_dir()
        .expect("home")
        .join(".gigastt")
        .join("models")
        .join("punct");
    match gigastt::punctuation::Punctuator::load(&punct_dir) {
        Ok(p) => Some(p),
        Err(e) => {
            eprintln!("skipping: punct model unavailable ({e:#})");
            None
        }
    }
}

/// Stream the golos fixture over an established WS session (the caller must
/// have configured 16 kHz first), send Stop, and collect every partial plus
/// the first final as raw JSON payloads.
async fn stream_golos_and_collect(
    sink: &mut WsSink,
    stream: &mut WsStream,
) -> (Vec<serde_json::Value>, serde_json::Value) {
    let pcm = golos_pcm16();
    // ~100 ms per frame at 16 kHz PCM16 mono.
    for chunk in pcm.chunks(3200) {
        sink.send(Message::Binary(chunk.to_vec().into()))
            .await
            .unwrap();
    }
    sink.send(Message::Text(
        serde_json::to_string(&serde_json::json!({"type": "stop"}))
            .unwrap()
            .into(),
    ))
    .await
    .unwrap();

    let mut partials = Vec::new();
    loop {
        let msg = tokio::time::timeout(Duration::from_secs(30), stream.next())
            .await
            .expect("timeout waiting for Final")
            .expect("stream ended")
            .expect("ws error");
        let text = msg.into_text().expect("expected text message");
        let v: serde_json::Value = serde_json::from_str(&text).expect("expected JSON");
        match v["type"].as_str().unwrap_or("") {
            "partial" => partials.push(v),
            "final" => return (partials, v),
            other => panic!("Unexpected message type {other}: {text}"),
        }
    }
}

/// Rebuild the raw transcript from a segment's `words[]` payload (what the
/// `text` field looked like before any post-processing).
fn joined_words(segment: &serde_json::Value) -> String {
    segment["words"]
        .as_array()
        .map(|words| {
            words
                .iter()
                .filter_map(|w| w["word"].as_str())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_default()
}

/// With a punctuator attached, the WS final segment's text is enriched
/// (capitalized / punctuated) while its `words[]` stay raw decoder output.
#[ignore]
#[tokio::test]
async fn test_ws_final_punctuated_with_punctuator() {
    let Some(punctuator) = load_punctuator_or_skip() else {
        return;
    };
    let engine = gigastt::inference::Engine::load(&common::model_dir())
        .expect("engine")
        .with_punctuator(Some(punctuator));
    let (port, _shutdown) = common::start_server_with_engine(engine).await;
    if get_health(port).await["variant"].as_str().unwrap_or("") != "rnnt" {
        eprintln!("skipping: e2e_rnnt head is already punctuated by the model");
        return;
    }

    let (mut sink, mut stream, _ready) = common::ws_connect(port).await;
    sink.send(Message::Text(
        serde_json::to_string(&serde_json::json!({"type": "configure", "sample_rate": 16000}))
            .unwrap()
            .into(),
    ))
    .await
    .unwrap();

    let (_partials, final_seg) = stream_golos_and_collect(&mut sink, &mut stream).await;
    let text = final_seg["text"].as_str().unwrap_or_default();
    let raw = joined_words(&final_seg);
    eprintln!("final (enriched): {text}");
    eprintln!("words (raw join): {raw}");

    assert!(!raw.is_empty(), "speech fixture must decode to words");
    assert!(
        text != raw,
        "enriched final text must differ from the raw words join: {text}"
    );
    let first = text.chars().next().unwrap_or(' ');
    assert!(
        first.is_uppercase() || text.contains(['.', ',', '?', '!']),
        "final should carry restored casing/punctuation, got: {text}"
    );
    // Word payloads are never rewritten: joined words equal the bare decoder
    // hypothesis (all lowercase, no sentence punctuation).
    assert_eq!(raw, raw.to_lowercase(), "words stay raw: {raw}");
    assert!(
        !raw.contains(['.', ',', '?', '!']),
        "words carry no restored punctuation: {raw}"
    );
}

/// Partial messages must stay raw even when the punctuator is attached:
/// a partial's `text` is byte-identical to the join of its `words[]`.
#[ignore]
#[tokio::test]
async fn test_ws_partials_stay_raw_with_punctuator() {
    let Some(punctuator) = load_punctuator_or_skip() else {
        return;
    };
    let engine = gigastt::inference::Engine::load(&common::model_dir())
        .expect("engine")
        .with_punctuator(Some(punctuator));
    let (port, _shutdown) = common::start_server_with_engine(engine).await;
    if get_health(port).await["variant"].as_str().unwrap_or("") != "rnnt" {
        eprintln!("skipping: e2e_rnnt head is already punctuated by the model");
        return;
    }

    let (mut sink, mut stream, _ready) = common::ws_connect(port).await;
    sink.send(Message::Text(
        serde_json::to_string(&serde_json::json!({"type": "configure", "sample_rate": 16000}))
            .unwrap()
            .into(),
    ))
    .await
    .unwrap();

    let (partials, _final_seg) = stream_golos_and_collect(&mut sink, &mut stream).await;
    assert!(
        !partials.is_empty(),
        "streaming speech must produce at least one partial"
    );
    for p in &partials {
        let text = p["text"].as_str().unwrap_or_default();
        assert_eq!(
            text,
            joined_words(p),
            "partial text must equal the raw words join (never enriched): {text}"
        );
    }
}

/// `Configure{punctuation:false}` on a punctuator-equipped server disables the
/// pass for this session only: the final text is the raw words join.
#[ignore]
#[tokio::test]
async fn test_ws_configure_punctuation_false_disables_final_enrichment() {
    let Some(punctuator) = load_punctuator_or_skip() else {
        return;
    };
    let engine = gigastt::inference::Engine::load(&common::model_dir())
        .expect("engine")
        .with_punctuator(Some(punctuator));
    let (port, _shutdown) = common::start_server_with_engine(engine).await;
    if get_health(port).await["variant"].as_str().unwrap_or("") != "rnnt" {
        eprintln!("skipping: e2e_rnnt head is already punctuated by the model");
        return;
    }

    let (mut sink, mut stream, _ready) = common::ws_connect(port).await;
    sink.send(Message::Text(
        serde_json::to_string(
            &serde_json::json!({"type": "configure", "sample_rate": 16000, "punctuation": false}),
        )
        .unwrap()
        .into(),
    ))
    .await
    .unwrap();

    let (_partials, final_seg) = stream_golos_and_collect(&mut sink, &mut stream).await;
    let text = final_seg["text"].as_str().unwrap_or_default();
    let raw = joined_words(&final_seg);
    eprintln!("final with punctuation=false: {text}");

    assert!(!raw.is_empty(), "speech fixture must decode to words");
    assert_eq!(
        text, raw,
        "punctuation=false must leave the final text raw: {text}"
    );
    assert!(
        !text.contains(['.', ',', '?', '!']),
        "no restored punctuation expected: {text}"
    );
}

/// `Configure{itn:false}` on an ITN-equipped server leaves number-words as
/// words: no ASCII digits appear that ITN would have produced.
#[ignore]
#[tokio::test]
async fn test_ws_configure_itn_false_disables_final_enrichment() {
    let engine = gigastt::inference::Engine::load(&common::model_dir())
        .expect("engine")
        .with_itn(true);
    let (port, _shutdown) = common::start_server_with_engine(engine).await;
    if get_health(port).await["variant"].as_str().unwrap_or("") != "rnnt" {
        eprintln!("skipping: e2e_rnnt head has ITN baked into the model");
        return;
    }

    let (mut sink, mut stream, _ready) = common::ws_connect(port).await;
    sink.send(Message::Text(
        serde_json::to_string(
            &serde_json::json!({"type": "configure", "sample_rate": 16000, "itn": false}),
        )
        .unwrap()
        .into(),
    ))
    .await
    .unwrap();

    let (_partials, final_seg) = stream_golos_and_collect(&mut sink, &mut stream).await;
    let text = final_seg["text"].as_str().unwrap_or_default();
    eprintln!("final with itn=false: {text}");
    assert!(
        !text.chars().any(|c| c.is_ascii_digit()),
        "itn=false should leave number-words as words, got: {text}"
    );
}

/// A server WITHOUT a punctuator keeps the legacy behavior even when the
/// client asks for punctuation: `Configure{punctuation:true}` is a graceful
/// no-op — the session works and the final text is the raw words join.
#[ignore]
#[tokio::test]
async fn test_ws_configure_punctuation_true_without_punctuator_is_noop() {
    let (port, _shutdown) = common::start_server(&common::model_dir()).await;
    let health = get_health(port).await;
    if health["punctuation"].as_bool().unwrap_or(false) {
        eprintln!("skipping: default engine already has a punctuator loaded");
        return;
    }

    let (mut sink, mut stream, _ready) = common::ws_connect(port).await;
    sink.send(Message::Text(
        serde_json::to_string(
            &serde_json::json!({"type": "configure", "sample_rate": 16000, "punctuation": true}),
        )
        .unwrap()
        .into(),
    ))
    .await
    .unwrap();

    let (_partials, final_seg) = stream_golos_and_collect(&mut sink, &mut stream).await;
    let text = final_seg["text"].as_str().unwrap_or_default();
    let raw = joined_words(&final_seg);
    eprintln!("final without punctuator: {text}");

    assert!(!raw.is_empty(), "speech fixture must decode to words");
    assert_eq!(
        text, raw,
        "no punctuator attached → final text stays raw: {text}"
    );
}

/// Post-processing overrides survive mid-session state recreation: a
/// `configure` that sets `diarization` rebuilds the streaming state wholesale
/// (in diarization-enabled builds), and a previously sent `punctuation: false`
/// must carry over to the new state — configure-after-audio is rejected, so
/// the client has no way to re-send it.
#[ignore]
#[tokio::test]
async fn test_ws_punctuation_override_survives_diarization_reconfigure() {
    let Some(punctuator) = load_punctuator_or_skip() else {
        return;
    };
    let engine = gigastt::inference::Engine::load(&common::model_dir())
        .expect("engine")
        .with_punctuator(Some(punctuator));
    let (port, _shutdown) = common::start_server_with_engine(engine).await;
    if get_health(port).await["variant"].as_str().unwrap_or("") != "rnnt" {
        eprintln!("skipping: e2e_rnnt head is already punctuated by the model");
        return;
    }

    let (mut sink, mut stream, _ready) = common::ws_connect(port).await;
    sink.send(Message::Text(
        serde_json::to_string(
            &serde_json::json!({"type": "configure", "sample_rate": 16000, "punctuation": false}),
        )
        .unwrap()
        .into(),
    ))
    .await
    .unwrap();
    // This second configure recreates the streaming state (diarization
    // branch); the punctuation override above must survive the rebuild.
    sink.send(Message::Text(
        serde_json::to_string(&serde_json::json!({"type": "configure", "diarization": false}))
            .unwrap()
            .into(),
    ))
    .await
    .unwrap();

    let (_partials, final_seg) = stream_golos_and_collect(&mut sink, &mut stream).await;
    let text = final_seg["text"].as_str().unwrap_or_default();
    let raw = joined_words(&final_seg);
    eprintln!("final after diarization reconfigure: {text}");

    assert!(!raw.is_empty(), "speech fixture must decode to words");
    assert_eq!(
        text, raw,
        "punctuation=false must survive the diarization state recreation: {text}"
    );
}

// ---------------------------------------------------------------------------
// Session limits in `ready` + segment-level confidence
// ---------------------------------------------------------------------------

/// The `ready` message advertises the server's session caps verbatim so a
/// client can plan a reconnect before hitting `max_session_duration_exceeded`
/// or an idle close. Distinctive non-default values prove the payload is
/// wired to the live config, not hardcoded.
#[ignore]
#[tokio::test]
async fn test_ws_ready_reports_configured_session_limits() {
    let model_dir = common::model_dir();
    let limits = gigastt::server::RuntimeLimits {
        max_session_secs: 4567,
        idle_timeout_secs: 123,
        ..Default::default()
    };
    let (port, _shutdown) = common::start_server_with_limits(&model_dir, limits).await;

    let (_sink, _stream, ready) = common::ws_connect(port).await;

    assert_eq!(ready["max_session_secs"], 4567);
    assert_eq!(ready["idle_timeout_secs"], 123);
}

/// A final segment decoded from real speech carries a segment-level
/// `confidence` aggregate in [0, 1] alongside the per-word scores.
#[ignore]
#[tokio::test]
async fn test_ws_final_carries_segment_confidence() {
    let model_dir = common::model_dir();
    let (port, _shutdown) = common::start_server(&model_dir).await;

    let (mut sink, mut stream, _ready) = common::ws_connect(port).await;
    sink.send(Message::Text(
        serde_json::to_string(&serde_json::json!({"type": "configure", "sample_rate": 16000}))
            .unwrap()
            .into(),
    ))
    .await
    .unwrap();

    let (_partials, final_seg) = stream_golos_and_collect(&mut sink, &mut stream).await;
    assert!(
        !joined_words(&final_seg).is_empty(),
        "speech fixture must decode to words"
    );
    let confidence = final_seg["confidence"]
        .as_f64()
        .expect("final with words must carry a numeric confidence");
    assert!(
        (0.0..=1.0).contains(&confidence),
        "segment confidence must lie in [0, 1], got {confidence}"
    );
}
