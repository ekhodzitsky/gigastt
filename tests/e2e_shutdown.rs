//! Graceful shutdown tests for the gigastt server.
//!
//! Verifies that in-flight WebSocket sessions and SSE streams terminate cleanly
//! when the server receives a shutdown signal, rather than hanging forever.
//!
//! All tests require the GigaAM ONNX model to be downloaded (~850MB).
//! Run with: `cargo test --test e2e_shutdown -- --ignored`

mod common;

use futures_util::{SinkExt, StreamExt};
use std::time::Duration;
use tokio_tungstenite::tungstenite::Message;

// ---------------------------------------------------------------------------
// 1. Shutdown during an active WebSocket session
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

    // The stream must terminate within 5 seconds — no hanging forever
    let result = tokio::time::timeout(Duration::from_secs(5), stream.next()).await;

    match result {
        // Timed out — connection hung, which is the failure we are guarding against
        Err(_elapsed) => {
            panic!("WebSocket stream did not terminate within 5s after server shutdown")
        }
        // Stream ended cleanly (None) or returned a Close frame or an error — all acceptable
        Ok(_) => {}
    }
}

// ---------------------------------------------------------------------------
// 2. Shutdown during an active SSE transcription stream
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore] // Requires model download
async fn test_shutdown_during_sse_stream() {
    let model_dir = common::model_dir();
    let (port, shutdown) = common::start_server(&model_dir).await;

    // POST a 10-second WAV to start a long-running SSE stream
    let wav = common::generate_wav(10, 16000);

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

    // Read the first SSE event to confirm the stream is live
    let first_event = tokio::time::timeout(Duration::from_secs(10), bytes_stream.next())
        .await
        .expect("Timed out waiting for first SSE event before shutdown")
        .expect("SSE stream ended before first event arrived");

    match first_event {
        Ok(bytes) => {
            let raw = String::from_utf8_lossy(&bytes);
            assert!(
                !raw.is_empty(),
                "First SSE chunk should contain data, got empty bytes"
            );
        }
        Err(e) => panic!("Error reading first SSE event: {e}"),
    }

    // Trigger server shutdown while the SSE stream is still open
    let _ = shutdown.send(());

    // The bytes stream must terminate within 5 seconds — no hanging forever
    let result = tokio::time::timeout(Duration::from_secs(5), bytes_stream.next()).await;

    match result {
        // Timed out — stream hung after shutdown signal, which is the failure we guard against
        Err(_elapsed) => {
            panic!("SSE bytes_stream did not terminate within 5s after server shutdown")
        }
        // Stream ended (None) or returned an error — both are acceptable termination signals
        Ok(_) => {}
    }
}
