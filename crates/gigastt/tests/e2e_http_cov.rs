//! End-to-end coverage tests for REST/SSE handler paths that the existing
//! e2e_rest suite does not exercise: the SSE inference loop driven by real
//! speech, the inference-timeout-disabled branch, and the pool-timeout metrics
//! paths for both the batch transcribe and SSE handlers.
//!
//! All model-gated (`#[ignore]`); run with:
//! `cargo test --test e2e_http_cov -- --ignored --test-threads=1`.

mod common;

use futures_util::SinkExt;
use std::time::Duration;
use tokio_tungstenite::tungstenite::Message;

const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/golos_00.wav");

/// SSE streaming of a real speech clip must drive the per-chunk inference loop
/// and emit `data:` events (covers the chunk-iterate / process_chunk / send-seg
/// / finish_stream path in `transcribe_stream`).
#[ignore = "requires the GigaAM model (~850MB)"]
#[tokio::test]
async fn test_sse_stream_real_speech_emits_segments() {
    let md = common::model_dir();
    let (port, shutdown) = common::start_server(&md).await;

    let wav = std::fs::read(FIXTURE).expect("read fixture");
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{port}/v1/transcribe/stream"))
        .body(wav)
        .send()
        .await
        .expect("sse request");
    assert!(resp.status().is_success(), "SSE status: {}", resp.status());

    // The handler transcribes the whole file then ends the stream, so reading the
    // full body collects every event.
    let text = resp.text().await.expect("sse body");
    assert!(
        text.contains("data:"),
        "expected SSE data events, got: {text:?}"
    );

    let _ = shutdown.send(());
}

/// With `inference_timeout_secs = 0` the REST transcribe path takes the
/// timeout-disabled branch (`handle.await` directly) instead of wrapping the
/// blocking run in a `tokio::time::timeout`.
#[ignore = "requires the GigaAM model (~850MB)"]
#[tokio::test]
async fn test_transcribe_inference_timeout_disabled() {
    let md = common::model_dir();
    let limits = gigastt::server::RuntimeLimits {
        inference_timeout_secs: 0,
        ..Default::default()
    };
    let (port, shutdown) = common::start_server_with_limits(&md, limits).await;

    let wav = std::fs::read(FIXTURE).expect("read fixture");
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{port}/v1/transcribe"))
        .body(wav)
        .send()
        .await
        .expect("transcribe request");
    assert!(
        resp.status().is_success(),
        "transcribe status: {}",
        resp.status()
    );

    let _ = shutdown.send(());
}

/// Saturating the pool while metrics are enabled forces the REST transcribe
/// checkout to time out and record the pool-timeout counter/histogram. Takes
/// ~30 s (the default pool-checkout timeout).
#[ignore = "requires the GigaAM model (~850MB); ~30s"]
#[tokio::test]
async fn test_transcribe_pool_timeout_records_metrics() {
    let md = common::model_dir();
    let (port, metrics_port, shutdown) = common::start_server_with_metrics(&md).await;

    // Saturate the default-size pool (2) with held WebSocket sessions.
    let mut held = Vec::new();
    for _ in 0..2 {
        let (sink, stream, _ready) = common::ws_connect(port).await;
        held.push((sink, stream));
    }

    let wav = common::generate_wav(1, 16000);
    let client = reqwest::Client::new();
    let resp = tokio::time::timeout(
        Duration::from_secs(40),
        client
            .post(format!("http://127.0.0.1:{port}/v1/transcribe"))
            .body(wav)
            .send(),
    )
    .await
    .expect("request returned before the test timeout")
    .expect("http request");
    assert_eq!(resp.status().as_u16(), 503, "expected pool-timeout 503");

    // The pool-timeout counter should now be exposed on the metrics port.
    let metrics = reqwest::get(format!("http://127.0.0.1:{metrics_port}/metrics"))
        .await
        .expect("metrics request")
        .text()
        .await
        .expect("metrics body");
    assert!(
        metrics.contains("gigastt_pool_timeouts_total"),
        "metrics should expose the pool-timeout counter"
    );

    // Release the held sessions.
    let stop = serde_json::to_string(&serde_json::json!({"type": "stop"})).unwrap();
    for (mut sink, mut stream) in held {
        let _ = sink.send(Message::Text(stop.clone().into())).await;
        let _ = tokio::time::timeout(Duration::from_secs(5), {
            use futures_util::StreamExt;
            stream.next()
        })
        .await;
    }
    let _ = shutdown.send(());
}

/// Same pool-timeout path for the SSE handler: a saturated pool makes
/// `transcribe_stream` time out at checkout and record the metric. ~30 s.
#[ignore = "requires the GigaAM model (~850MB); ~30s"]
#[tokio::test]
async fn test_sse_pool_timeout_records_metrics() {
    let md = common::model_dir();
    let (port, _metrics_port, shutdown) = common::start_server_with_metrics(&md).await;

    let mut held = Vec::new();
    for _ in 0..2 {
        let (sink, stream, _ready) = common::ws_connect(port).await;
        held.push((sink, stream));
    }

    let wav = common::generate_wav(1, 16000);
    let client = reqwest::Client::new();
    let resp = tokio::time::timeout(
        Duration::from_secs(40),
        client
            .post(format!("http://127.0.0.1:{port}/v1/transcribe/stream"))
            .body(wav)
            .send(),
    )
    .await
    .expect("request returned before the test timeout")
    .expect("http request");
    assert_eq!(resp.status().as_u16(), 503, "expected SSE pool-timeout 503");

    let stop = serde_json::to_string(&serde_json::json!({"type": "stop"})).unwrap();
    for (mut sink, mut stream) in held {
        let _ = sink.send(Message::Text(stop.clone().into())).await;
        let _ = tokio::time::timeout(Duration::from_secs(5), {
            use futures_util::StreamExt;
            stream.next()
        })
        .await;
    }
    let _ = shutdown.send(());
}
