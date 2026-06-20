//! End-to-end REST API tests for the gigastt HTTP server.
//!
//! All tests require the GigaAM model to be downloaded (~850MB).
//! Run with: `cargo test --test e2e_rest -- --ignored`

mod common;

use futures_util::StreamExt;
use std::time::Duration;

// ---------------------------------------------------------------------------
// 1. Health endpoint
// ---------------------------------------------------------------------------

#[ignore]
#[tokio::test]
async fn test_health_returns_ok() {
    let (port, shutdown) = common::start_server(&common::model_dir()).await;

    let resp = tokio::time::timeout(Duration::from_secs(10), async {
        reqwest::Client::new()
            .get(format!("http://127.0.0.1:{port}/health"))
            .send()
            .await
            .expect("GET /health failed")
    })
    .await
    .expect("GET /health timed out");

    assert_eq!(resp.status(), 200);

    let text = resp.text().await.expect("Expected text body");
    let body: serde_json::Value = serde_json::from_str(&text).expect("Expected JSON body");
    assert_eq!(body["status"], "ok", "status field should be \"ok\"");
    assert!(
        body["model"]
            .as_str()
            .unwrap_or_default()
            .contains("gigaam"),
        "model field should contain \"gigaam\", got: {:?}",
        body["model"]
    );
    assert!(
        !body["version"].as_str().unwrap_or_default().is_empty(),
        "version field should be a non-empty string"
    );

    let _ = shutdown.send(());
}

// ---------------------------------------------------------------------------
// 2. POST /v1/transcribe — valid WAV
// ---------------------------------------------------------------------------

#[ignore]
#[tokio::test]
async fn test_transcribe_wav_returns_text() {
    let (port, shutdown) = common::start_server(&common::model_dir()).await;
    let wav = common::generate_wav(2, 16000);

    let resp = tokio::time::timeout(Duration::from_secs(30), async {
        reqwest::Client::new()
            .post(format!("http://127.0.0.1:{port}/v1/transcribe"))
            .body(wav)
            .send()
            .await
            .expect("POST /v1/transcribe failed")
    })
    .await
    .expect("POST /v1/transcribe timed out");

    assert_eq!(resp.status(), 200);

    let text = resp.text().await.expect("Expected text body");
    let body: serde_json::Value = serde_json::from_str(&text).expect("Expected JSON body");
    assert!(
        body["text"].is_string(),
        "\"text\" field should be a string, got: {:?}",
        body["text"]
    );
    assert!(
        body["words"].is_array(),
        "\"words\" field should be an array, got: {:?}",
        body["words"]
    );
    let duration = body["duration"]
        .as_f64()
        .expect("\"duration\" should be a number");
    assert!(duration > 0.0, "duration should be > 0, got {duration}");

    let _ = shutdown.send(());
}

// ---------------------------------------------------------------------------
// 3. POST /v1/transcribe — empty body → 400
// ---------------------------------------------------------------------------

#[ignore]
#[tokio::test]
async fn test_transcribe_empty_body_returns_400() {
    let (port, shutdown) = common::start_server(&common::model_dir()).await;

    let resp = tokio::time::timeout(Duration::from_secs(10), async {
        reqwest::Client::new()
            .post(format!("http://127.0.0.1:{port}/v1/transcribe"))
            .body(Vec::<u8>::new())
            .send()
            .await
            .expect("POST /v1/transcribe failed")
    })
    .await
    .expect("POST /v1/transcribe timed out");

    assert_eq!(resp.status(), 400);

    let text = resp.text().await.expect("Expected text body");
    let body: serde_json::Value = serde_json::from_str(&text).expect("Expected JSON body");
    assert_eq!(
        body["code"], "empty_body",
        "code field should be \"empty_body\", got: {:?}",
        body["code"]
    );

    let _ = shutdown.send(());
}

// ---------------------------------------------------------------------------
// 4. POST /v1/transcribe — invalid audio → 422
// ---------------------------------------------------------------------------

#[ignore]
#[tokio::test]
async fn test_transcribe_invalid_audio_returns_422() {
    let (port, shutdown) = common::start_server(&common::model_dir()).await;

    // 1000 random-ish bytes that are not a valid audio file
    let garbage: Vec<u8> = (0u8..=255).cycle().take(1000).collect();

    let resp = tokio::time::timeout(Duration::from_secs(30), async {
        reqwest::Client::new()
            .post(format!("http://127.0.0.1:{port}/v1/transcribe"))
            .body(garbage)
            .send()
            .await
            .expect("POST /v1/transcribe failed")
    })
    .await
    .expect("POST /v1/transcribe timed out");

    assert_eq!(resp.status(), 422);

    let text = resp.text().await.expect("Expected text body");
    let body: serde_json::Value = serde_json::from_str(&text).expect("Expected JSON body");
    let code = body["code"].as_str().unwrap_or_default();
    assert!(
        code == "invalid_audio" || code == "transcription_error",
        "code should be \"invalid_audio\" or \"transcription_error\", got: {code:?}"
    );

    let _ = shutdown.send(());
}

// ---------------------------------------------------------------------------
// 5. POST /v1/transcribe/stream — SSE stream completes without error
// ---------------------------------------------------------------------------

#[ignore]
#[tokio::test]
async fn test_transcribe_stream_sse_incremental() {
    let (port, shutdown) = common::start_server(&common::model_dir()).await;
    let wav = common::generate_wav(10, 16000);

    let resp = tokio::time::timeout(Duration::from_secs(60), async {
        reqwest::Client::new()
            .post(format!("http://127.0.0.1:{port}/v1/transcribe/stream"))
            .body(wav)
            .send()
            .await
            .expect("POST /v1/transcribe/stream failed")
    })
    .await
    .expect("POST /v1/transcribe/stream timed out");

    assert_eq!(resp.status(), 200);

    // Collect all SSE bytes — stream should terminate cleanly
    let mut stream = resp.bytes_stream();
    let mut all_bytes = Vec::new();

    tokio::time::timeout(Duration::from_secs(60), async {
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => all_bytes.extend_from_slice(&bytes),
                Err(e) => {
                    eprintln!("SSE stream error: {e}");
                    break;
                }
            }
        }
    })
    .await
    .expect("SSE stream did not complete within 60s");

    // Any data: lines present must be valid JSON with a type field
    let raw = String::from_utf8_lossy(&all_bytes);
    for line in raw.lines() {
        if let Some(json_str) = line.strip_prefix("data:") {
            let json_str = json_str.trim();
            if json_str.is_empty() {
                continue;
            }
            let v: serde_json::Value =
                serde_json::from_str(json_str).expect("SSE data should be valid JSON");
            assert!(
                v["type"].is_string(),
                "SSE event should have a \"type\" field, got: {:?}",
                v
            );
        }
    }

    let _ = shutdown.send(());
}

// ---------------------------------------------------------------------------
// 6. POST /v1/transcribe/stream — empty body → 400
// ---------------------------------------------------------------------------

#[ignore]
#[tokio::test]
async fn test_transcribe_stream_empty_body_returns_400() {
    let (port, shutdown) = common::start_server(&common::model_dir()).await;

    let resp = tokio::time::timeout(Duration::from_secs(10), async {
        reqwest::Client::new()
            .post(format!("http://127.0.0.1:{port}/v1/transcribe/stream"))
            .body(Vec::<u8>::new())
            .send()
            .await
            .expect("POST /v1/transcribe/stream failed")
    })
    .await
    .expect("POST /v1/transcribe/stream timed out");

    assert_eq!(resp.status(), 400);

    let _ = shutdown.send(());
}

// ---------------------------------------------------------------------------
// 7. SSE events well-formed: data: prefix + valid JSON with type field
// ---------------------------------------------------------------------------

#[ignore]
#[tokio::test]
async fn test_sse_events_well_formed() {
    let (port, shutdown) = common::start_server(&common::model_dir()).await;
    let wav = common::generate_wav(5, 16000);

    let resp = tokio::time::timeout(Duration::from_secs(60), async {
        reqwest::Client::new()
            .post(format!("http://127.0.0.1:{port}/v1/transcribe/stream"))
            .body(wav)
            .send()
            .await
            .expect("POST /v1/transcribe/stream failed")
    })
    .await
    .expect("POST /v1/transcribe/stream timed out");

    assert_eq!(resp.status(), 200);

    // Collect all SSE bytes
    let mut stream = resp.bytes_stream();
    let mut all_bytes = Vec::new();
    let collect_timeout = Duration::from_secs(30);

    tokio::time::timeout(collect_timeout, async {
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => all_bytes.extend_from_slice(&bytes),
                Err(e) => {
                    eprintln!("SSE stream error: {e}");
                    break;
                }
            }
        }
    })
    .await
    .ok(); // timeout is acceptable — stream may still be open

    let raw = String::from_utf8_lossy(&all_bytes);

    // Any data: lines present must be well-formed JSON with a type field.
    // Note: a pure sine wave may produce zero transcription events — that's OK.
    for line in raw.lines() {
        if let Some(json_str) = line.strip_prefix("data:") {
            let json_str = json_str.trim();
            if json_str.is_empty() {
                continue;
            }
            let v: serde_json::Value = serde_json::from_str(json_str)
                .unwrap_or_else(|_| panic!("SSE data line is not valid JSON: {json_str:?}"));
            let event_type = v["type"]
                .as_str()
                .unwrap_or_else(|| panic!("SSE event missing \"type\" field: {v:?}"));
            assert!(
                event_type == "partial" || event_type == "final",
                "SSE event type should be \"partial\" or \"final\", got: {event_type:?}"
            );
        }
    }

    let _ = shutdown.send(());
}

// ---------------------------------------------------------------------------
// 8. Midstream disconnect — server should not panic
// ---------------------------------------------------------------------------

#[ignore]
#[tokio::test]
async fn test_sse_midstream_disconnect() {
    let (port, shutdown) = common::start_server(&common::model_dir()).await;
    let wav = common::generate_wav(10, 16000);

    let resp = tokio::time::timeout(Duration::from_secs(60), async {
        reqwest::Client::new()
            .post(format!("http://127.0.0.1:{port}/v1/transcribe/stream"))
            .body(wav)
            .send()
            .await
            .expect("POST /v1/transcribe/stream failed")
    })
    .await
    .expect("POST /v1/transcribe/stream timed out");

    assert_eq!(resp.status(), 200);

    // Begin reading the stream, then drop it mid-flight to simulate a client
    // disconnect. The point of this test is the server's resilience to that
    // disconnect, not event timing: a 440 Hz sine is not speech, so the model
    // legitimately may emit no early `partial` and only a trailing segment —
    // hard-requiring a first event within 10 s made this test flaky on slow CI
    // runners. Poll once to start the response body flowing, tolerating either an
    // event or the window elapsing.
    let mut stream = resp.bytes_stream();
    let _ = tokio::time::timeout(Duration::from_secs(5), stream.next()).await;

    // Drop the stream, simulating client disconnect.
    drop(stream);

    // Give the server a moment to detect the disconnect and clean up
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Server should still be alive — verify with a /health check
    let health_resp = tokio::time::timeout(Duration::from_secs(10), async {
        reqwest::Client::new()
            .get(format!("http://127.0.0.1:{port}/health"))
            .send()
            .await
            .expect("GET /health after disconnect failed")
    })
    .await
    .expect("GET /health after disconnect timed out");

    assert_eq!(
        health_resp.status(),
        200,
        "Server should still be healthy after midstream disconnect"
    );

    let _ = shutdown.send(());
}

#[ignore]
#[tokio::test]
async fn test_metrics_not_on_primary_port_when_disabled() {
    // `/metrics` never rides the primary port — it lives on the separate
    // loopback listener. With metrics disabled there is no metrics listener at
    // all, so the primary port returns a bare 404 for `/metrics`.
    let (port, shutdown) = common::start_server(&common::model_dir()).await;
    let resp = reqwest::Client::new()
        .get(format!("http://127.0.0.1:{port}/metrics"))
        .send()
        .await
        .expect("GET /metrics failed");
    assert_eq!(
        resp.status(),
        404,
        "/metrics must not be served on the primary port"
    );
    let _ = shutdown.send(());
}

#[ignore]
#[tokio::test]
async fn test_metrics_enabled_returns_prometheus_on_separate_port() {
    let (port, metrics_port, shutdown) =
        common::start_server_with_metrics(&common::model_dir()).await;
    let client = reqwest::Client::new();

    // Metrics are served on the dedicated loopback listener, not the primary.
    let resp = client
        .get(format!("http://127.0.0.1:{metrics_port}/metrics"))
        .send()
        .await
        .expect("GET /metrics failed");
    assert_eq!(resp.status(), 200);
    let body = resp.text().await.unwrap();
    assert!(body.contains("# HELP gigastt_http_requests_total"));
    assert!(body.contains("# TYPE gigastt_http_requests_total counter"));

    // The primary port must NOT serve /metrics even when metrics are enabled —
    // locks in the separate-listener contract (telemetry off the CORS allowlist
    // / rate limiter).
    let primary = client
        .get(format!("http://127.0.0.1:{port}/metrics"))
        .send()
        .await
        .expect("GET primary /metrics failed");
    assert_eq!(
        primary.status(),
        404,
        "/metrics must stay off the primary port"
    );
    let _ = shutdown.send(());
}

#[ignore]
#[tokio::test]
async fn test_ready_returns_ok() {
    let (port, shutdown) = common::start_server(&common::model_dir()).await;
    let resp = reqwest::Client::new()
        .get(format!("http://127.0.0.1:{port}/ready"))
        .send()
        .await
        .expect("GET /ready failed");
    assert_eq!(resp.status(), 200);
    let _ = shutdown.send(());
}

#[ignore]
#[tokio::test]
async fn test_options_v1_models_returns_204() {
    let (port, shutdown) = common::start_server(&common::model_dir()).await;
    let resp = reqwest::Client::new()
        .request(
            reqwest::Method::OPTIONS,
            format!("http://127.0.0.1:{port}/v1/models"),
        )
        .send()
        .await
        .expect("OPTIONS /v1/models failed");
    assert_eq!(resp.status(), 204);
    let _ = shutdown.send(());
}

#[ignore]
#[tokio::test]
async fn test_server_list_models() {
    let (port, shutdown) = common::start_server(&common::model_dir()).await;
    let resp = reqwest::Client::new()
        .get(format!("http://127.0.0.1:{port}/v1/models"))
        .send()
        .await
        .expect("GET /v1/models failed");
    assert_eq!(resp.status(), 200);
    let text = resp.text().await.unwrap();
    let body: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(body["id"], "gigaam-v3-e2e-rnnt");
    assert_eq!(body["name"], "GigaAM v3 RNN-T");
    assert_eq!(body["sample_rate"], 16000);
    let enc = body["encoder"].as_str().unwrap();
    assert!(enc == "int8" || enc == "fp32");
    assert!(body["vocab_size"].as_u64().unwrap() > 0);
    let _ = shutdown.send(());
}

// ---------------------------------------------------------------------------
// Routing: OPTIONS preflight on every protected route returns 204
// ---------------------------------------------------------------------------

#[ignore]
#[tokio::test]
async fn test_options_protected_routes_return_204() {
    // Exercises the `options(...)` route wiring on the protected sub-router for
    // /v1/transcribe, /v1/transcribe/stream and /v1/ws (in addition to the
    // already-covered /v1/models). A loopback Origin keeps the request inside
    // the allowlist so the preflight reaches the route handler.
    let (port, shutdown) = common::start_server(&common::model_dir()).await;
    let client = reqwest::Client::new();

    for path in ["/v1/transcribe", "/v1/transcribe/stream", "/v1/ws"] {
        let resp = client
            .request(
                reqwest::Method::OPTIONS,
                format!("http://127.0.0.1:{port}{path}"),
            )
            .header("Origin", "http://localhost:3000")
            .send()
            .await
            .unwrap_or_else(|_| panic!("OPTIONS {path} failed"));
        assert_eq!(resp.status(), 204, "OPTIONS {path} should return 204");
    }

    let _ = shutdown.send(());
}

// ---------------------------------------------------------------------------
// CORS / origin handling on the wired router
// ---------------------------------------------------------------------------

#[ignore]
#[tokio::test]
async fn test_cross_origin_request_denied_on_v1() {
    // The wired router attaches `origin_layer` to /v1/*. A non-loopback Origin
    // must be denied with 403 + `origin_denied` before reaching the handler.
    let (port, shutdown) = common::start_server(&common::model_dir()).await;
    let resp = reqwest::Client::new()
        .get(format!("http://127.0.0.1:{port}/v1/models"))
        .header("Origin", "https://evil.example.com")
        .send()
        .await
        .expect("GET /v1/models with foreign Origin failed");
    assert_eq!(resp.status(), 403, "cross-origin /v1/* must be denied");
    let text = resp.text().await.expect("text body");
    let body: serde_json::Value = serde_json::from_str(&text).expect("JSON body");
    assert_eq!(body["code"], "origin_denied");
    let _ = shutdown.send(());
}

#[ignore]
#[tokio::test]
async fn test_loopback_origin_gets_cors_echo() {
    // A loopback Origin is allowed and the response echoes it back in
    // Access-Control-Allow-Origin (no wildcard by default).
    let (port, shutdown) = common::start_server(&common::model_dir()).await;
    let resp = reqwest::Client::new()
        .get(format!("http://127.0.0.1:{port}/v1/models"))
        .header("Origin", "http://localhost:3000")
        .send()
        .await
        .expect("GET /v1/models with loopback Origin failed");
    assert_eq!(resp.status(), 200, "loopback Origin must be allowed");
    assert_eq!(
        resp.headers()
            .get("access-control-allow-origin")
            .and_then(|v| v.to_str().ok()),
        Some("http://localhost:3000"),
        "CORS echo must mirror the loopback Origin"
    );
    let _ = shutdown.send(());
}

#[ignore]
#[tokio::test]
async fn test_health_skips_origin_guard_on_wired_router() {
    // /health is exempt from the origin guard even on the full router, so a
    // monitoring probe carrying a foreign Origin still gets 200.
    let (port, shutdown) = common::start_server(&common::model_dir()).await;
    let resp = reqwest::Client::new()
        .get(format!("http://127.0.0.1:{port}/health"))
        .header("Origin", "https://evil.example.com")
        .send()
        .await
        .expect("GET /health with foreign Origin failed");
    assert_eq!(resp.status(), 200, "/health must skip the origin guard");
    let _ = shutdown.send(());
}

// ---------------------------------------------------------------------------
// Request-id middleware is wired on the full router
// ---------------------------------------------------------------------------

#[ignore]
#[tokio::test]
async fn test_request_id_header_present_on_wired_router() {
    // The request_id_layer is attached to the full app, so every response
    // carries an X-Request-Id. With no client-supplied id the server mints a
    // UUIDv7.
    let (port, shutdown) = common::start_server(&common::model_dir()).await;
    let resp = reqwest::Client::new()
        .get(format!("http://127.0.0.1:{port}/health"))
        .send()
        .await
        .expect("GET /health failed");
    assert_eq!(resp.status(), 200);
    let rid = resp
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .expect("X-Request-Id header must be present");
    assert!(
        uuid::Uuid::parse_str(rid).is_ok(),
        "X-Request-Id should be a UUID, got {rid:?}"
    );
    let _ = shutdown.send(());
}

// ---------------------------------------------------------------------------
// /ready reports pool_exhausted when every triplet is checked out
// ---------------------------------------------------------------------------

#[ignore]
#[tokio::test]
async fn test_ready_returns_503_when_pool_exhausted() {
    // Single-triplet pool. Occupy the only slot with a long REST transcribe so
    // `/ready` observes `available == 0` and returns 503 + `pool_exhausted`.
    let (port, shutdown) = common::start_server_with_pool(&common::model_dir(), 1).await;
    let long_wav = common::generate_wav(60, 16000);

    let occupier_url = format!("http://127.0.0.1:{port}/v1/transcribe");
    let occupier = tokio::spawn(async move {
        let _ = reqwest::Client::new()
            .post(&occupier_url)
            .body(long_wav)
            .send()
            .await;
    });

    // Poll /ready until it flips to 503; the occupier needs a moment to
    // actually check out the triplet.
    let client = reqwest::Client::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    let mut saw_exhausted = false;
    while tokio::time::Instant::now() < deadline {
        let resp = client
            .get(format!("http://127.0.0.1:{port}/ready"))
            .send()
            .await
            .expect("GET /ready failed");
        if resp.status() == 503 {
            let text = resp.text().await.expect("text body");
            let body: serde_json::Value = serde_json::from_str(&text).expect("JSON body");
            assert_eq!(body["status"], "not_ready");
            assert_eq!(body["reason"], "pool_exhausted");
            assert_eq!(body["pool_total"], 1);
            saw_exhausted = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(
        saw_exhausted,
        "/ready must report pool_exhausted while the only triplet is in use"
    );

    let _ = shutdown.send(());
    occupier.abort();
    let _ = occupier.await;
}

// The v0.9.0-rc.1 zero-copy REST decode path used to carry a
// Linux-only VmRSS budget test here. It asserted that
// `RSS_after - RSS_before < wav.len() * 3 + 40 MiB` after POSTing a 300 s
// WAV to `/v1/transcribe`. In practice the full inference pass allocates
// 90+ MiB of encoder scratch alone for 5 minutes of 16 kHz audio, and
// ONNX Runtime keeps the INT8 session state resident — the delta was
// ~320 MiB in CI regardless of whether the upload path did 1× or 4× copies.
// The RSS signal from the upload path itself was drowned out by inference
// cost, so the test could neither catch the regression it was designed to
// catch nor pass reliably. The zero-copy contract is still enforced by the
// `BytesMediaSource` type in `src/inference/audio.rs`, which is covered by
// unit tests and is not exercised by this integration surface.
