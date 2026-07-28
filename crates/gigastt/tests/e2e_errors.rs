//! End-to-end error-path tests for the gigastt server.
//!
//! All tests except `test_ws_idle_timeout` require the ONNX model.
//! Run with: `cargo test --test e2e_errors -- --ignored`

mod common;

use futures_util::{SinkExt, StreamExt};
use std::time::Duration;
use tokio_tungstenite::tungstenite::Message;

// ─── 1. REST oversized body ─────────────────────────────────────────────────

/// POST /v1/transcribe with a body larger than the 50MB DefaultBodyLimit.
/// Expects a 413 Payload Too Large with machine-readable code
/// `payload_too_large` — the strict version of the previous `!= 200` assertion
/// that was too permissive to catch regressions in the body-limit guard.
#[ignore]
#[tokio::test]
async fn test_rest_oversized_body_rejected() {
    let model_dir = common::model_dir();
    let (port, shutdown) = common::start_server(&model_dir).await;

    // Build a reqwest client that does NOT enforce its own body limit.
    let client = reqwest::Client::builder()
        .build()
        .expect("Failed to build reqwest client");

    // 51 MB of zeros — just over the 50 MB server limit.
    let oversized_body: Vec<u8> = vec![0u8; 51 * 1024 * 1024];

    let response = client
        .post(format!("http://127.0.0.1:{port}/v1/transcribe"))
        .body(oversized_body)
        .send()
        .await
        .expect("Request should complete (connection not refused)");

    assert_eq!(
        response.status().as_u16(),
        413,
        "Expected 413 Payload Too Large for oversized body"
    );

    // Body format depends on which layer fired first:
    //   - axum's `DefaultBodyLimit` middleware returns plain text
    //     ("length limit exceeded") when Content-Length exceeds the cap
    //     before the handler runs.
    //   - Our handler's defence-in-depth `body.len() > limit` guard returns
    //     a JSON `{"code":"payload_too_large"}` body.
    // The contract is the 413 status; the JSON body is a bonus when
    // the handler-layer guard is the one that fires. Either is acceptable.
    let body_text = response
        .text()
        .await
        .expect("Response body should be readable");
    if let Ok(body) = serde_json::from_str::<serde_json::Value>(&body_text) {
        assert_eq!(
            body["code"], "payload_too_large",
            "Handler guard body must carry code='payload_too_large', got: {body}"
        );
    }

    let _ = shutdown.send(());
}

// ─── 2. WebSocket oversized frame ───────────────────────────────────────────

/// Send a binary frame larger than the 512 KB WS frame limit.
/// The server should close the connection. Verifies the server is still
/// healthy afterwards.
#[ignore]
#[tokio::test]
async fn test_ws_oversized_frame_rejected() {
    let model_dir = common::model_dir();
    let (port, shutdown) = common::start_server(&model_dir).await;

    // Use raw tokio_tungstenite so we can send an oversized frame without
    // the client library enforcing its own limit.
    let (mut ws, _) = tokio_tungstenite::connect_async_with_config(
        format!("ws://127.0.0.1:{port}/v1/ws"),
        Some({
            let mut cfg = tokio_tungstenite::tungstenite::protocol::WebSocketConfig::default();
            cfg.max_message_size = None;
            cfg.max_frame_size = None;
            cfg
        }),
        false,
    )
    .await
    .expect("WebSocket connection failed");

    // Consume the Ready message.
    let _ready = tokio::time::timeout(Duration::from_secs(5), ws.next())
        .await
        .expect("timeout waiting for Ready")
        .expect("stream ended")
        .expect("ws error");

    // Send a binary frame that exceeds the server's 512 KB limit.
    let oversized: Vec<u8> = vec![0u8; 600 * 1024];
    // The server may RST the connection as soon as it sees the oversized
    // frame header, so `send` can race with the close. Either outcome is
    // acceptable — what matters is that the connection is terminated and the
    // server stays healthy.
    let _ = ws.send(Message::Binary(oversized.into())).await;

    // Server should close the connection; the next read returns an error or
    // None (stream closed).
    let next = tokio::time::timeout(Duration::from_secs(5), ws.next()).await;
    match next {
        Ok(Some(Ok(Message::Close(_)))) | Ok(None) => {
            // Expected: clean close or stream ended.
        }
        Ok(Some(Err(_))) => {
            // Also expected: connection reset / protocol error from server.
        }
        Ok(Some(Ok(other))) => {
            panic!("Expected close after oversized frame, got: {other:?}");
        }
        Err(_) => {
            panic!("Timeout waiting for server to close connection after oversized frame");
        }
    }

    // Server must still be reachable.
    let health = reqwest::get(format!("http://127.0.0.1:{port}/health"))
        .await
        .expect("Health check failed after oversized frame test");
    assert!(health.status().is_success(), "Server unhealthy after test");

    let _ = shutdown.send(());
}

// ─── 3. Fifth WebSocket client is blocked ───────────────────────────────────

/// Saturate the pool with 4 WebSocket clients, then try a 5th.
/// The (pool+1)th client's TCP connection succeeds but pool.checkout() blocks,
/// so its Ready message never arrives within 3 seconds. Uses an explicit small
/// pool so the test does not depend on `DEFAULT_POOL_SIZE` (2 since v2.3) or the
/// RAM-aware cap.
#[ignore]
#[tokio::test]
async fn test_ws_client_hangs_when_pool_saturated() {
    const POOL: usize = 2;
    let model_dir = common::model_dir();
    let (port, shutdown) = common::start_server_with_pool(&model_dir, POOL).await;

    // Connect POOL clients and hold them open (saturating the pool).
    let mut clients = Vec::new();
    for _ in 0..POOL {
        let (sink, stream, _ready) = common::ws_connect(port).await;
        clients.push((sink, stream));
    }

    // Attempt one more client using raw connect_async (we don't want ws_connect
    // because that helper expects a Ready message).
    let (mut extra_ws, _) =
        tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}/v1/ws"))
            .await
            .expect("TCP connection for the extra client should succeed");

    // The pool is exhausted, so pool.checkout() blocks server-side.
    // The Ready message should NOT arrive within 3 seconds.
    let result = tokio::time::timeout(Duration::from_secs(3), extra_ws.next()).await;
    assert!(
        result.is_err(),
        "extra client should NOT receive Ready while pool is saturated, but got: {result:?}"
    );

    // Release all pool slots by sending Stop to each.
    let stop_json = serde_json::to_string(&serde_json::json!({"type": "stop"})).unwrap();
    for (mut sink, mut stream) in clients {
        sink.send(Message::Text(stop_json.clone().into()))
            .await
            .unwrap();
        // Drain until Final or stream ends.
        // Drain at most one message; we only need to confirm the Stop roundtrip.
        let _ = tokio::time::timeout(Duration::from_secs(5), stream.next()).await;
    }

    let _ = shutdown.send(());
}

// ─── 4. HTTP returns 503 when pool is saturated ─────────────────────────────

/// Hold all pool slots via WebSocket, then POST /v1/transcribe.
/// The HTTP handler has a 30-second pool.checkout() timeout and returns 503.
/// Uses an explicit small pool so the test does not depend on `DEFAULT_POOL_SIZE`.
///
/// This test takes ~30 seconds to complete (the HTTP timeout duration).
#[ignore]
#[tokio::test]
async fn test_rest_saturated_pool_returns_503() {
    const POOL: usize = 2;
    let model_dir = common::model_dir();
    let (port, shutdown) = common::start_server_with_pool(&model_dir, POOL).await;

    // Saturate the pool.
    let mut clients = Vec::new();
    for _ in 0..POOL {
        let (sink, stream, _ready) = common::ws_connect(port).await;
        clients.push((sink, stream));
    }

    let wav = common::generate_wav(1, 16000);
    let client = reqwest::Client::new();

    // Allow 35 seconds so the 30-second server timeout has room to expire.
    let response = tokio::time::timeout(
        Duration::from_secs(35),
        client
            .post(format!("http://127.0.0.1:{port}/v1/transcribe"))
            .body(wav)
            .send(),
    )
    .await
    .expect("Test timed out before server returned 503 — check pool timeout in http.rs")
    .expect("HTTP request failed");

    assert_eq!(
        response.status().as_u16(),
        503,
        "Expected 503 Service Unavailable when pool is saturated"
    );

    let body_text = response
        .text()
        .await
        .expect("Response body should be readable");
    let body: serde_json::Value =
        serde_json::from_str(&body_text).expect("Response body should be JSON");
    assert_eq!(
        body["code"], "timeout",
        "Expected code='timeout', got: {body}"
    );

    // Release pool slots.
    let stop_json = serde_json::to_string(&serde_json::json!({"type": "stop"})).unwrap();
    for (mut sink, mut stream) in clients {
        sink.send(Message::Text(stop_json.clone().into()))
            .await
            .unwrap();
        // Drain at most one message; we only need to confirm the Stop roundtrip.
        let _ = tokio::time::timeout(Duration::from_secs(5), stream.next()).await;
    }

    let _ = shutdown.send(());
}

// ─── 4b. SSE /v1/transcribe/stream reserves a pool slot BEFORE decoding ──────

/// Regression for the SSE memory-exhaustion ordering bug: `/v1/transcribe/stream`
/// must check out a pool slot *before* decoding the upload, exactly like the
/// synchronous `/v1/transcribe`. We prove the ordering behaviourally: with the
/// pool saturated, a stream request carrying *invalid* audio must block on the
/// pool checkout and time out with 503 `timeout` — NOT decode first and fail
/// fast with 422 `invalid_audio`. If decode ran before checkout (the old bug),
/// an unbounded number of concurrent decodes could each expand a compressed
/// upload into a full f32 PCM buffer and exhaust memory.
///
/// Takes ~30 s (the pool checkout timeout), like `test_rest_saturated_pool_returns_503`.
#[ignore]
#[tokio::test]
async fn test_sse_stream_reserves_pool_before_decode() {
    const POOL: usize = 2;
    let model_dir = common::model_dir();
    let (port, shutdown) = common::start_server_with_pool(&model_dir, POOL).await;

    // Saturate the pool with WebSocket sessions (the SSE handler draws from the
    // same batch pool, falling back to the interactive pool).
    let mut clients = Vec::new();
    for _ in 0..POOL {
        let (sink, stream, _ready) = common::ws_connect(port).await;
        clients.push((sink, stream));
    }

    // Deliberately invalid audio: if the handler decoded before reserving, this
    // would 422 immediately; reserving first makes it wait on the saturated pool.
    let not_audio = b"this is definitely not a valid audio container".to_vec();
    let client = reqwest::Client::new();

    // Allow 35 seconds so the 30-second server checkout timeout has room to expire.
    let response = tokio::time::timeout(
        Duration::from_secs(35),
        client
            .post(format!("http://127.0.0.1:{port}/v1/transcribe/stream"))
            .body(not_audio)
            .send(),
    )
    .await
    .expect(
        "Test timed out before server returned 503 — check checkout ordering in transcribe_stream",
    )
    .expect("HTTP request failed");

    assert_eq!(
        response.status().as_u16(),
        503,
        "SSE stream must reserve the pool before decoding: a saturated pool must \
         yield 503, not decode the (invalid) upload first and return 422"
    );

    let body_text = response
        .text()
        .await
        .expect("Response body should be readable");
    let body: serde_json::Value =
        serde_json::from_str(&body_text).expect("Response body should be JSON");
    assert_eq!(
        body["code"], "timeout",
        "Expected code='timeout' (blocked on pool checkout, upload never decoded), got: {body}"
    );

    // Release pool slots.
    let stop_json = serde_json::to_string(&serde_json::json!({"type": "stop"})).unwrap();
    for (mut sink, mut stream) in clients {
        sink.send(Message::Text(stop_json.clone().into()))
            .await
            .unwrap();
        let _ = tokio::time::timeout(Duration::from_secs(5), stream.next()).await;
    }

    let _ = shutdown.send(());
}

// ─── 5. WebSocket idle timeout ──────────────────────────────────────────────

/// Connect a WebSocket client, receive Ready, then send nothing.
/// The server closes the connection after the configured idle timeout.
/// Uses a short (3 s) idle timeout so the test finishes in under 10 s.
#[ignore]
#[tokio::test]
async fn test_ws_idle_timeout() {
    let model_dir = common::model_dir();
    let limits = gigastt::server::RuntimeLimits {
        idle_timeout_secs: 3,
        ..Default::default()
    };
    let (port, shutdown) = common::start_server_with_limits(&model_dir, limits).await;

    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}/v1/ws"))
        .await
        .expect("WebSocket connection failed");

    // Consume the Ready message.
    let _ready = tokio::time::timeout(Duration::from_secs(5), ws.next())
        .await
        .expect("timeout waiting for Ready")
        .expect("stream ended")
        .expect("ws error");

    // Wait up to 10 seconds for the server to close the idle connection (3 s timeout + margin).
    let result = tokio::time::timeout(Duration::from_secs(10), ws.next()).await;

    match result {
        Ok(None) => {
            // Stream ended cleanly — server closed the connection.
        }
        Ok(Some(Ok(Message::Close(_)))) => {
            // Server sent a Close frame.
        }
        Ok(Some(Err(_))) => {
            // Connection reset — also acceptable.
        }
        Ok(Some(Ok(Message::Text(text)))) => {
            // Server now sends an Error text message before the Close frame.
            let msg: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
            assert_eq!(
                msg.get("code").and_then(|v| v.as_str()),
                Some("idle_timeout")
            );
            // After the error text, the next message should be Close or stream end.
            let next = tokio::time::timeout(Duration::from_secs(5), ws.next()).await;
            match next {
                Ok(None) | Ok(Some(Ok(Message::Close(_)))) | Ok(Some(Err(_))) => {}
                other => panic!("Expected close after idle timeout text, got: {other:?}"),
            }
        }
        Ok(Some(Ok(other))) => {
            panic!("Expected idle-timeout close, got unexpected message: {other:?}");
        }
        Err(_) => {
            panic!(
                "Server did not close the idle connection within 10 seconds \
                 (expected 3-second idle timeout)"
            );
        }
    }

    let _ = shutdown.send(());
}

// ─── Cooperative cancellation / no-progress watchdog ────────────────────────

/// A run that makes no progress within `inference_timeout_secs` returns 504 AND
/// releases its pooled triplet within roughly one window — the whole point of
/// the change. With a single pool slot and a 1 s no-progress budget, a long
/// file trips the watchdog before its first ~24 s encoder window completes; the
/// follow-up request must then acquire the slot in far less time than the full
/// file would take to decode, proving the timed-out run was actually cancelled
/// rather than left wedged.
#[ignore = "requires the GigaAM model (~850MB)"]
#[tokio::test]
async fn test_inference_watchdog_frees_pool_slot_within_one_window() {
    let model_dir = common::model_dir();
    let limits = gigastt::server::RuntimeLimits {
        inference_timeout_secs: 1,
        ..Default::default()
    };
    let (port, shutdown) = common::start_server_with_pool_and_limits(&model_dir, 1, limits).await;

    // ~5 minutes of audio: a full decode is tens of seconds, so if the slot were
    // NOT freed on timeout the follow-up would block far past one window. 16 kHz
    // keeps the body ~9.6 MB, under the 50 MiB limit.
    let long = common::generate_wav(300, 16000);
    let short = common::generate_wav(1, 16000);
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("http://127.0.0.1:{port}/v1/transcribe"))
        .body(long)
        .send()
        .await
        .expect("long request");
    assert_eq!(
        resp.status().as_u16(),
        504,
        "a stalled run must trip the no-progress watchdog with 504"
    );

    // Measure how long the single freed slot takes to serve a fresh request.
    let reclaim_start = std::time::Instant::now();
    let resp2 = tokio::time::timeout(
        Duration::from_secs(20),
        client
            .post(format!("http://127.0.0.1:{port}/v1/transcribe"))
            .body(short)
            .send(),
    )
    .await
    .expect("follow-up returned before the test timeout")
    .expect("follow-up request");
    let reclaimed = reclaim_start.elapsed();
    assert!(
        resp2.status().is_success(),
        "follow-up must get the freed slot, got {}",
        resp2.status()
    );
    assert!(
        reclaimed < Duration::from_secs(15),
        "slot should free within ~one window; follow-up took {reclaimed:?}"
    );
    eprintln!("pool slot reclaimed + short decode in {reclaimed:?}");

    let _ = shutdown.send(());
}

/// `DELETE /v1/jobs/{id}` on a processing job frees its triplet in bounded time:
/// with a single job worker, a queued follow-up job can only run once the
/// cancelled one releases the slot. If cancellation reached the engine, the
/// follow-up completes in far less time than the cancelled job's full decode.
#[ignore = "requires the GigaAM model (~850MB)"]
#[tokio::test]
async fn test_delete_job_frees_triplet_in_bounded_time() {
    let model_dir = common::model_dir();
    let (port, shutdown) = common::start_server_with_jobs(&model_dir, 1).await;
    let client = reqwest::Client::new();

    // Submit a long job and wait until it is Processing (holding the only slot).
    let long = common::generate_wav(300, 16000);
    let submit_text = client
        .post(format!("http://127.0.0.1:{port}/v1/jobs"))
        .body(long)
        .send()
        .await
        .expect("submit long job")
        .text()
        .await
        .expect("submit response body");
    let submit: serde_json::Value =
        serde_json::from_str(&submit_text).expect("submit response json");
    let id = submit["job_id"].as_str().expect("job_id").to_string();

    let mut processing = false;
    for _ in 0..100 {
        let status_text = client
            .get(format!("http://127.0.0.1:{port}/v1/jobs/{id}"))
            .send()
            .await
            .expect("poll job")
            .text()
            .await
            .expect("status body");
        let status: serde_json::Value = serde_json::from_str(&status_text).expect("status json");
        if status["status"] == "processing" {
            processing = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(processing, "long job must reach processing");

    // Cancel it, then time how long a fresh short job takes to finish — it can
    // only start once the cancelled run releases its triplet.
    let del = client
        .delete(format!("http://127.0.0.1:{port}/v1/jobs/{id}"))
        .send()
        .await
        .expect("delete job");
    assert_eq!(del.status().as_u16(), 204, "cancel returns 204");

    let reclaim_start = std::time::Instant::now();
    let short = common::generate_wav(2, 16000);
    let submit2_text = client
        .post(format!("http://127.0.0.1:{port}/v1/jobs"))
        .body(short)
        .send()
        .await
        .expect("submit short job")
        .text()
        .await
        .expect("submit2 body");
    let submit2: serde_json::Value = serde_json::from_str(&submit2_text).expect("submit2 json");
    let id2 = submit2["job_id"].as_str().expect("job_id2").to_string();

    let mut done = false;
    for _ in 0..200 {
        let status_text = client
            .get(format!("http://127.0.0.1:{port}/v1/jobs/{id2}"))
            .send()
            .await
            .expect("poll job2")
            .text()
            .await
            .expect("status2 body");
        let status: serde_json::Value = serde_json::from_str(&status_text).expect("status2 json");
        match status["status"].as_str() {
            Some("done") => {
                done = true;
                break;
            }
            Some("failed") => panic!("follow-up job failed: {status:?}"),
            _ => {}
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let reclaimed = reclaim_start.elapsed();
    assert!(done, "follow-up job must complete once the slot is freed");
    assert!(
        reclaimed < Duration::from_secs(15),
        "DELETE must free the triplet promptly; follow-up took {reclaimed:?}"
    );
    eprintln!("triplet freed + follow-up job done in {reclaimed:?}");

    let _ = shutdown.send(());
}
