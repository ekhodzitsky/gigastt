//! Graceful shutdown + session-cap tests for the gigastt server (V1-03 / V1-04).
//!
//! Verifies that in-flight WebSocket sessions and SSE streams terminate cleanly
//! when the server receives a shutdown signal, rather than hanging forever, and
//! that long-running WS sessions are capped by wall-clock duration.
//!
//! All tests require the GigaAM ONNX model to be downloaded (~850MB).
//! Run with: `cargo test --test e2e_shutdown -- --ignored --test-threads=1`

mod common;

use futures_util::{SinkExt, StreamExt};
use std::time::Duration;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::protocol::CloseFrame;

// ---------------------------------------------------------------------------
// 1. Shutdown during an active WebSocket session — liveness check
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore] // Requires model download
async fn test_shutdown_during_ws_session() {
    let model_dir = common::model_dir();
    let (port, shutdown) = common::start_server(&model_dir).await;

    // Connect and receive Ready
    let (mut sink, mut stream, _ready) = common::ws_connect(port).await;

    // Send 1 second of PCM16 silence at 48kHz to start a streaming session
    let silence = common::generate_pcm16_silence(1.0, 48000);
    sink.send(Message::Binary(silence.into())).await.unwrap();

    // Trigger server shutdown while the session is still open
    let _ = shutdown.send(());

    // The stream must terminate within 15 seconds — no hanging forever
    let result = tokio::time::timeout(Duration::from_secs(15), stream.next()).await;

    match result {
        Err(_elapsed) => {
            panic!("WebSocket stream did not terminate within 15s after server shutdown")
        }
        Ok(_) => {}
    }
}

// ---------------------------------------------------------------------------
// 2. Shutdown during an active SSE transcription stream — liveness check
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore] // Requires model download
async fn test_shutdown_during_sse_stream() {
    let model_dir = common::model_dir();
    let (port, shutdown) = common::start_server(&model_dir).await;

    // POST a long WAV so the SSE task is still running when we signal
    // shutdown. 440 Hz sine at 16 kHz does not produce transcript tokens —
    // so the stream may deliver no events at all. The assertion here is
    // purely about *clean termination*, not event delivery.
    let wav = common::generate_wav(60, 16000);

    let resp = tokio::time::timeout(Duration::from_secs(30), async {
        reqwest::Client::new()
            .post(format!("http://127.0.0.1:{port}/v1/transcribe/stream"))
            .body(wav)
            .send()
            .await
            .expect("POST /v1/transcribe/stream failed")
    })
    .await
    .expect("POST /v1/transcribe/stream timed out waiting for response headers");

    assert_eq!(
        resp.status(),
        200,
        "Expected 200 from /v1/transcribe/stream"
    );

    let mut bytes_stream = resp.bytes_stream();

    // Give the spawn_blocking task a beat to enter its inference loop so
    // the shutdown signal has something to cancel.
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Trigger server shutdown while the SSE stream is still open
    let _ = shutdown.send(());

    // The bytes stream must terminate within the default drain window (10 s)
    // plus a safety margin — no hanging forever.
    let start = tokio::time::Instant::now();
    loop {
        if start.elapsed() > Duration::from_secs(15) {
            panic!("SSE bytes_stream did not terminate within 15 s after server shutdown");
        }
        match tokio::time::timeout(Duration::from_secs(1), bytes_stream.next()).await {
            Ok(None) => break,
            Ok(Some(Err(_))) => break,
            Ok(Some(Ok(_))) => continue,
            Err(_) => continue,
        }
    }
}

// ---------------------------------------------------------------------------
// 3. V1-03: shutdown emits a Final frame + Close(1001) before socket EOF
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore] // Requires model download
async fn test_shutdown_ws_emits_final_and_close() {
    let model_dir = common::model_dir();
    // Use custom limits so the drain window is generous even on slow CI.
    let limits = gigastt::server::RuntimeLimits {
        shutdown_drain_secs: 10,
        ..Default::default()
    };
    let (port, shutdown) = common::start_server_with_limits(&model_dir, limits).await;

    let (mut sink, mut stream, _ready) = common::ws_connect(port).await;

    // Stream 0.5s of 440 Hz tone at 48 kHz so the encoder has something to process.
    let tone = common::generate_pcm16_tone(0.5, 48000, 440.0);
    sink.send(Message::Binary(tone.into())).await.unwrap();

    // Give the server a moment to start processing.
    tokio::time::sleep(Duration::from_millis(200)).await;

    let _ = shutdown.send(());

    // Drain the socket; assert we see a Final and a Close(1001).
    let mut saw_final = false;
    let mut saw_close_1001 = false;

    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    while tokio::time::Instant::now() < deadline {
        let next = tokio::time::timeout(Duration::from_secs(5), stream.next()).await;
        match next {
            Ok(Some(Ok(Message::Text(text)))) => {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text)
                    && v["type"] == "final"
                {
                    saw_final = true;
                }
            }
            Ok(Some(Ok(Message::Close(Some(CloseFrame { code, .. }))))) => {
                if u16::from(code) == 1001 {
                    saw_close_1001 = true;
                }
                break;
            }
            Ok(Some(Ok(Message::Close(None)))) => break,
            Ok(None) => break,
            Ok(Some(Err(_))) => break,
            Err(_) => break,
            _ => continue,
        }
    }

    assert!(
        saw_final,
        "Shutdown must emit a Final frame before closing the socket"
    );
    assert!(
        saw_close_1001,
        "Shutdown must close with status 1001 (Going Away)"
    );
}

// ---------------------------------------------------------------------------
// 4. V1-03: SSE stream terminates cleanly within the drain window
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore] // Requires model download
async fn test_shutdown_sse_stream_terminates_cleanly() {
    let model_dir = common::model_dir();
    let limits = gigastt::server::RuntimeLimits {
        shutdown_drain_secs: 10,
        ..Default::default()
    };
    let (port, shutdown) = common::start_server_with_limits(&model_dir, limits).await;

    // 60 s WAV — long enough that the SSE task is still processing chunks
    // when we signal shutdown. 440 Hz sine doesn't produce transcript
    // tokens, so we don't rely on any SSE event ever arriving; we only
    // assert that the HTTP stream terminates within the drain window.
    let wav = common::generate_wav(60, 16000);

    let resp = reqwest::Client::new()
        .post(format!("http://127.0.0.1:{port}/v1/transcribe/stream"))
        .body(wav)
        .send()
        .await
        .expect("POST /v1/transcribe/stream failed");
    assert_eq!(resp.status(), 200);

    let mut bytes_stream = resp.bytes_stream();

    // Let the spawn_blocking task enter the inference loop.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let shutdown_instant = tokio::time::Instant::now();
    let _ = shutdown.send(());

    // Must terminate well inside the drain window (10 s). Allow a small
    // slack (15 s total) to cover cold-start jitter on CI runners.
    let start = tokio::time::Instant::now();
    let mut saw_end = false;
    while start.elapsed() < Duration::from_secs(15) {
        match tokio::time::timeout(Duration::from_secs(1), bytes_stream.next()).await {
            Ok(None) | Ok(Some(Err(_))) => {
                saw_end = true;
                break;
            }
            Ok(Some(Ok(_))) | Err(_) => continue,
        }
    }

    assert!(
        saw_end,
        "SSE bytes_stream must terminate after shutdown (observed elapsed: {:?})",
        shutdown_instant.elapsed()
    );
}

// ---------------------------------------------------------------------------
// 5. V1-04: session duration cap fires even for a silence-streaming client
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore] // Requires model download
async fn test_max_session_duration_cap() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("gigastt=debug")
        .try_init();
    let model_dir = common::model_dir();
    let limits = gigastt::server::RuntimeLimits {
        max_session_secs: 3,
        ..Default::default()
    };
    let (port, _shutdown) = common::start_server_with_limits(&model_dir, limits).await;

    let (mut sink, mut stream, _ready) = common::ws_connect(port).await;

    // Silence streamer: 20 ms of silence every 250 ms for up to 15 s total.
    // Sending at 250 ms is slow enough that the server isn't buried in a
    // backlog of frames and the cap branch of the `select!` can preempt
    // between inferences. Each idle_timeout reset happens on every frame —
    // so *without* V1-04 the connection would live forever.
    let stream_task = tokio::spawn(async move {
        let chunk = common::generate_pcm16_silence(0.02, 48000);
        for _ in 0..60 {
            if sink.send(Message::Binary(chunk.clone().into())).await.is_err() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    });

    // Within cap (3 s) + generous slack we expect the error frame + Close(1008).
    // Slack covers test boot (~500 ms model checkout) + up to 1 s tail for the
    // in-flight spawn_blocking inference that was running when the cap expired.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    let mut saw_error_code = false;
    let mut saw_close_1008 = false;

    while tokio::time::Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        let next = tokio::time::timeout(remaining, stream.next()).await;
        match next {
            Ok(Some(Ok(Message::Text(text)))) => {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text)
                    && v["type"] == "error"
                    && v["code"] == "max_session_duration_exceeded"
                {
                    saw_error_code = true;
                }
            }
            Ok(Some(Ok(Message::Close(Some(CloseFrame { code, .. }))))) => {
                if u16::from(code) == 1008 {
                    saw_close_1008 = true;
                }
                break;
            }
            Ok(Some(Ok(Message::Close(None)))) => break,
            Ok(None) => break,
            Ok(Some(Err(_))) => break,
            Err(_) => break,
            _ => continue,
        }
    }

    stream_task.abort();
    let _ = stream_task.await;

    assert!(
        saw_error_code,
        "Session cap must emit Error with code=max_session_duration_exceeded"
    );
    assert!(
        saw_close_1008,
        "Session cap must close with status 1008 (Policy Violation)"
    );
}
