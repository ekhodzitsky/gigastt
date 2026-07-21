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
    // id/name/variant reflect the head actually loaded on disk (rnnt or
    // e2e_rnnt), not a fixed literal — assert the consistent pair instead.
    let id = body["id"].as_str().unwrap();
    let variant = body["variant"].as_str().unwrap();
    assert!(
        (id == "gigaam-v3-rnnt" && variant == "rnnt")
            || (id == "gigaam-v3-e2e-rnnt" && variant == "e2e_rnnt"),
        "unexpected id/variant pair: {id} / {variant}"
    );
    assert_eq!(
        body["name"],
        if variant == "e2e_rnnt" {
            "GigaAM v3 E2E RNN-T"
        } else {
            "GigaAM v3 RNN-T"
        }
    );
    assert_eq!(body["sample_rate"], 16000);
    let enc = body["encoder"].as_str().unwrap();
    assert!(enc == "int8" || enc == "fp32");
    assert!(body["vocab_size"].as_u64().unwrap() > 0);
    // New capability fields are present and boolean.
    assert!(body["punctuation"].is_boolean());
    assert!(body["itn"].is_boolean());
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

// ---------------------------------------------------------------------------
// Per-request recognition-knob overrides on POST /v1/transcribe (roadmap #24).
//
// These exercise the ADDITIVE query params `punctuation` / `itn` / `vad` and
// the forward-compat `variant` guard. Absent params must reproduce the default
// response byte-for-byte; a knob turned on without its backing resource must
// 409 *before* the pool checkout. SSE and WebSocket are out of scope for the
// REST query-param surface (streaming enriches final segments by the engine
// boot policy / WS Configure instead — see e2e_ws.rs).
// ---------------------------------------------------------------------------

/// GET `/health` and parse it as JSON (the repo's `reqwest` build has no `json`
/// feature, so go through `.text()` + `serde_json` like the other tests).
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

/// Read the loaded recognition head from `/health` (`"rnnt"` or `"e2e_rnnt"`).
async fn loaded_variant(port: u16) -> String {
    get_health(port).await["variant"]
        .as_str()
        .unwrap_or_default()
        .to_string()
}

const GOLOS_FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/golos_00.wav");

async fn post_transcribe(port: u16, query: &str, body: Vec<u8>) -> (reqwest::StatusCode, String) {
    let sep = if query.is_empty() { "" } else { "?" };
    let resp = tokio::time::timeout(Duration::from_secs(30), async {
        reqwest::Client::new()
            .post(format!("http://127.0.0.1:{port}/v1/transcribe{sep}{query}"))
            .body(body)
            .send()
            .await
            .expect("POST /v1/transcribe failed")
    })
    .await
    .expect("POST /v1/transcribe timed out");
    let status = resp.status();
    let text = resp.text().await.expect("body text");
    (status, text)
}

/// Baseline: a request with NO override params must produce exactly the same
/// response as the pre-feature endpoint (the additive params are a no-op when
/// absent). We assert the transcript text is byte-identical across two calls,
/// one with an empty query and one with a genuinely-absent query string.
#[ignore]
#[tokio::test]
async fn test_transcribe_overrides_absent_is_baseline() {
    let (port, shutdown) = common::start_server(&common::model_dir()).await;
    let wav = std::fs::read(GOLOS_FIXTURE).expect("read golos fixture");

    let (s1, b1) = post_transcribe(port, "", wav.clone()).await;
    assert_eq!(s1, 200);
    // A no-op param (format defaults to json) must still be byte-identical.
    let (s2, b2) = post_transcribe(port, "download=", wav.clone()).await;
    assert_eq!(s2, 200);

    let v1: serde_json::Value = serde_json::from_str(&b1).unwrap();
    let v2: serde_json::Value = serde_json::from_str(&b2).unwrap();
    assert_eq!(
        v1["text"], v2["text"],
        "absent overrides must not change text"
    );
    assert!(v1["text"].is_string());

    let _ = shutdown.send(());
}

/// `?vad=true` against a server started WITHOUT a VAD must 409 `vad_not_loaded`
/// and must NOT consume a pool triplet (fail-fast before checkout).
#[ignore]
#[tokio::test]
async fn test_transcribe_vad_true_without_vad_returns_409() {
    let (port, shutdown) = common::start_server(&common::model_dir()).await;
    let wav = common::generate_wav(1, 16000);

    let (status, body) = post_transcribe(port, "vad=true", wav).await;
    assert_eq!(status, reqwest::StatusCode::CONFLICT);
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["code"], "vad_not_loaded");

    let _ = shutdown.send(());
}

/// `?vad=false` against a VAD-less server is a no-op (whole-buffer decode either
/// way) and must succeed with 200 — turning a knob OFF never needs the resource.
#[ignore]
#[tokio::test]
async fn test_transcribe_vad_false_without_vad_is_ok() {
    let (port, shutdown) = common::start_server(&common::model_dir()).await;
    let wav = common::generate_wav(1, 16000);

    let (status, _body) = post_transcribe(port, "vad=false", wav).await;
    assert_eq!(status, 200);

    let _ = shutdown.send(());
}

/// `?variant=<other head>` when a different head is loaded must 409
/// `variant_not_loaded`; requesting the loaded head must proceed (200).
#[ignore]
#[tokio::test]
async fn test_transcribe_variant_mismatch_returns_409() {
    let (port, shutdown) = common::start_server(&common::model_dir()).await;
    let wav = common::generate_wav(1, 16000);
    let loaded = loaded_variant(port).await;
    let other = if loaded == "rnnt" { "e2e_rnnt" } else { "rnnt" };

    // Mismatched variant → 409.
    let (status, body) = post_transcribe(port, &format!("variant={other}"), wav.clone()).await;
    assert_eq!(status, reqwest::StatusCode::CONFLICT);
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["code"], "variant_not_loaded");

    // Matching variant → proceeds (200).
    let (ok_status, _ok_body) = post_transcribe(port, &format!("variant={loaded}"), wav).await;
    assert_eq!(ok_status, 200);

    let _ = shutdown.send(());
}

/// `?punctuation=true` against a server with no punctuator loaded must 409
/// `punctuation_not_available`. Uses the default engine (no punctuator).
#[ignore]
#[tokio::test]
async fn test_transcribe_punctuation_true_without_punctuator_returns_409() {
    let (port, shutdown) = common::start_server(&common::model_dir()).await;
    // Only meaningful when the default engine has no punctuator attached.
    let health = get_health(port).await;
    if health["punctuation"].as_bool().unwrap_or(false) {
        eprintln!("skipping: default engine already has a punctuator loaded");
        let _ = shutdown.send(());
        return;
    }
    let wav = common::generate_wav(1, 16000);
    let (status, body) = post_transcribe(port, "punctuation=true", wav).await;
    assert_eq!(status, reqwest::StatusCode::CONFLICT);
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["code"], "punctuation_not_available");

    let _ = shutdown.send(());
}

/// Precedence: a server booted with punctuation ON, then `?punctuation=false`,
/// must yield UNPUNCTUATED text (the per-request override wins over the boot
/// policy). Compares against the default (punctuated) response on the same
/// audio. Requires the punct model to be present; otherwise the punctuator
/// won't attach and the test self-skips.
#[ignore]
#[tokio::test]
async fn test_transcribe_punctuation_false_overrides_boot_on() {
    let punct_dir = common::home_dir()
        .expect("home")
        .join(".gigastt")
        .join("models")
        .join("punct");
    let punctuator = match gigastt::punctuation::Punctuator::load(&punct_dir) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("skipping: punct model unavailable ({e:#})");
            return;
        }
    };
    let engine = gigastt::inference::Engine::load(&common::model_dir())
        .expect("engine")
        .with_punctuator(Some(punctuator));
    let (port, shutdown) = common::start_server_with_engine(engine).await;
    let wav = std::fs::read(GOLOS_FIXTURE).expect("read golos fixture");

    // Sanity: /health reports punctuation is active.
    let health = get_health(port).await;
    assert_eq!(health["punctuation"], true, "boot punctuation must be on");

    let (s_on, b_on) = post_transcribe(port, "", wav.clone()).await;
    assert_eq!(s_on, 200);
    let (s_off, b_off) = post_transcribe(port, "punctuation=false", wav).await;
    assert_eq!(s_off, 200);

    let on: serde_json::Value = serde_json::from_str(&b_on).unwrap();
    let off: serde_json::Value = serde_json::from_str(&b_off).unwrap();
    let on_text = on["text"].as_str().unwrap_or_default();
    let off_text = off["text"].as_str().unwrap_or_default();
    eprintln!("punctuation ON:  {on_text}");
    eprintln!("punctuation OFF: {off_text}");
    // The override-off text must not contain sentence punctuation the restorer
    // would have added. This is a weak-but-meaningful check that the pass was
    // actually skipped for the per-request call.
    assert!(
        !off_text.contains('.') && !off_text.contains(','),
        "punctuation=false should suppress restored punctuation, got: {off_text}"
    );

    let _ = shutdown.send(());
}

/// Precedence: a server booted with ITN ON, then `?itn=false`, must leave
/// number-words as words. ITN is pure code (no model), so this always runs.
/// Uses a golos fixture that contains spoken numbers if available; the check is
/// resilient — it only asserts the two responses differ *or* both are digit-free
/// when the audio has no numbers, and always asserts 200.
#[ignore]
#[tokio::test]
async fn test_transcribe_itn_false_overrides_boot_on() {
    let engine = gigastt::inference::Engine::load(&common::model_dir())
        .expect("engine")
        .with_itn(true);
    let (port, shutdown) = common::start_server_with_engine(engine).await;
    let wav = std::fs::read(GOLOS_FIXTURE).expect("read golos fixture");

    let health = get_health(port).await;
    assert_eq!(health["itn"], true, "boot ITN must be on");

    let (s_on, b_on) = post_transcribe(port, "", wav.clone()).await;
    assert_eq!(s_on, 200);
    let (s_off, b_off) = post_transcribe(port, "itn=false", wav).await;
    assert_eq!(s_off, 200);

    let on: serde_json::Value = serde_json::from_str(&b_on).unwrap();
    let off: serde_json::Value = serde_json::from_str(&b_off).unwrap();
    let off_text = off["text"].as_str().unwrap_or_default();
    eprintln!("ITN ON:  {}", on["text"].as_str().unwrap_or_default());
    eprintln!("ITN OFF: {off_text}");
    // With ITN off the number-words stay words: no ASCII digit should appear
    // that ITN would have produced from spoken numerals.
    assert!(
        !off_text.chars().any(|c| c.is_ascii_digit()),
        "itn=false should leave number-words as words, got: {off_text}"
    );

    let _ = shutdown.send(());
}

// ---------------------------------------------------------------------------
// 17. Channel-split transcription (`channels=split`)
// ---------------------------------------------------------------------------

#[ignore]
#[tokio::test]
async fn test_transcribe_channels_split_returns_speakers_and_ordered_words() {
    let (port, shutdown) = common::start_server(&common::model_dir()).await;
    let wav = common::generate_stereo_wav_split(
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/golos_00.wav"),
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/golos_01.wav"),
    );

    let resp = tokio::time::timeout(Duration::from_secs(60), async {
        reqwest::Client::new()
            .post(format!(
                "http://127.0.0.1:{port}/v1/transcribe?channels=split"
            ))
            .body(wav)
            .send()
            .await
            .expect("POST /v1/transcribe failed")
    })
    .await
    .expect("POST /v1/transcribe timed out");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("expected text body");
    let body: serde_json::Value = serde_json::from_str(&text).expect("expected JSON");
    let words = body["words"].as_array().expect("words array");
    assert!(!words.is_empty());

    let mut last_start = -1.0_f64;
    let mut saw_speaker_0 = false;
    let mut saw_speaker_1 = false;
    for w in words {
        let speaker = w["speaker"].as_u64().expect("speaker should be present");
        assert!(speaker == 0 || speaker == 1);
        let start = w["start"].as_f64().expect("start number");
        assert!(start >= last_start, "words must be ordered by start time");
        last_start = start;
        if speaker == 0 {
            saw_speaker_0 = true;
        } else {
            saw_speaker_1 = true;
        }
    }
    assert!(saw_speaker_0, "expected at least one speaker_0 word");
    assert!(saw_speaker_1, "expected at least one speaker_1 word");

    let _ = shutdown.send(());
}

#[ignore]
#[tokio::test]
async fn test_transcribe_channels_split_mono_fallback() {
    let (port, shutdown) = common::start_server(&common::model_dir()).await;
    let wav = common::generate_wav(1, 16000);

    let resp = tokio::time::timeout(Duration::from_secs(60), async {
        reqwest::Client::new()
            .post(format!(
                "http://127.0.0.1:{port}/v1/transcribe?channels=split"
            ))
            .body(wav)
            .send()
            .await
            .expect("POST /v1/transcribe failed")
    })
    .await
    .expect("POST /v1/transcribe timed out");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("expected text body");
    let body: serde_json::Value = serde_json::from_str(&text).expect("expected JSON");
    let words = body["words"].as_array().expect("words array");
    for w in words {
        assert!(
            w["speaker"].is_null(),
            "mono fallback must not emit speaker"
        );
    }

    let _ = shutdown.send(());
}

#[ignore]
#[tokio::test]
async fn test_transcribe_channels_split_dual_mono_fallback() {
    let (port, shutdown) = common::start_server(&common::model_dir()).await;
    let wav = common::generate_dual_mono_wav(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/golos_00.wav"
    ));

    let resp = tokio::time::timeout(Duration::from_secs(60), async {
        reqwest::Client::new()
            .post(format!(
                "http://127.0.0.1:{port}/v1/transcribe?channels=split"
            ))
            .body(wav)
            .send()
            .await
            .expect("POST /v1/transcribe failed")
    })
    .await
    .expect("POST /v1/transcribe timed out");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("expected text body");
    let body: serde_json::Value = serde_json::from_str(&text).expect("expected JSON");
    let words = body["words"].as_array().expect("words array");
    for w in words {
        assert!(
            w["speaker"].is_null(),
            "dual-mono fallback must not emit speaker"
        );
    }

    let _ = shutdown.send(());
}

// Offline diarization is opt-in per request: a plain `/v1/transcribe` must NOT
// label speakers, and only `?diarization=true` runs the speaker pass. This guards
// the 2.11.2 regression where the working fbank extractor made every offline
// transcript diarize unconditionally (the dual-mono fallback above then emitted
// speakers). Single-speaker fixture → the diarized pass labels every word speaker_0.
#[ignore]
#[tokio::test]
async fn test_transcribe_diarization_is_opt_in() {
    let (port, shutdown) = common::start_server(&common::model_dir()).await;
    let wav = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/golos_00.wav"
    ))
    .expect("read fixture");

    let words_for = |query: &str, wav: Vec<u8>| {
        let url = format!("http://127.0.0.1:{port}/v1/transcribe{query}");
        async move {
            let resp = reqwest::Client::new()
                .post(url)
                .body(wav)
                .send()
                .await
                .expect("POST /v1/transcribe failed");
            assert_eq!(resp.status(), 200);
            let body: serde_json::Value =
                serde_json::from_str(&resp.text().await.expect("text body")).expect("JSON");
            body["words"].as_array().cloned().expect("words array")
        }
    };

    // Default: no diarization requested → no speaker labels.
    let default_words = words_for("", wav.clone()).await;
    assert!(
        default_words.iter().all(|w| w["speaker"].is_null()),
        "plain /v1/transcribe must not emit speaker labels"
    );

    // Opt-in: ?diarization=true → the offline speaker pass labels words.
    let diarized_words = words_for("?diarization=true", wav).await;
    assert!(
        diarized_words.iter().any(|w| !w["speaker"].is_null()),
        "?diarization=true must label at least one word with a speaker"
    );

    let _ = shutdown.send(());
}

#[ignore]
#[tokio::test]
async fn test_transcribe_channels_split_with_diarization_returns_400() {
    let (port, shutdown) = common::start_server(&common::model_dir()).await;
    let wav = common::generate_wav(1, 16000);

    let resp = tokio::time::timeout(Duration::from_secs(10), async {
        reqwest::Client::new()
            .post(format!(
                "http://127.0.0.1:{port}/v1/transcribe?channels=split&diarization=true"
            ))
            .body(wav)
            .send()
            .await
            .expect("POST /v1/transcribe failed")
    })
    .await
    .expect("POST /v1/transcribe timed out");

    assert_eq!(resp.status(), 400);
    let text = resp.text().await.expect("expected text body");
    let body: serde_json::Value = serde_json::from_str(&text).expect("expected JSON");
    assert_eq!(body["code"], "conflicting_modes");

    let _ = shutdown.send(());
}

// ---------------------------------------------------------------------------
// 18. Segment-level output (`segments=true`)
// ---------------------------------------------------------------------------

#[ignore]
#[tokio::test]
async fn test_transcribe_segments_true_returns_words_and_segments() {
    let (port, shutdown) = common::start_server(&common::model_dir()).await;
    // Real speech fixture: a synthetic tone transcribes to zero words, which
    // would make the non-empty segments assertion below vacuously fail.
    let wav = std::fs::read(GOLOS_FIXTURE).expect("read golos fixture");

    let resp = tokio::time::timeout(Duration::from_secs(60), async {
        reqwest::Client::new()
            .post(format!(
                "http://127.0.0.1:{port}/v1/transcribe?segments=true&word_timestamps=true"
            ))
            .body(wav)
            .send()
            .await
            .expect("POST /v1/transcribe failed")
    })
    .await
    .expect("POST /v1/transcribe timed out");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("expected text body");
    let body: serde_json::Value = serde_json::from_str(&text).expect("expected JSON");

    assert!(body["words"].is_array(), "words array must be present");
    assert!(
        body["segments"].is_array(),
        "segments array must be present"
    );

    let words = body["words"].as_array().unwrap();
    let segments = body["segments"].as_array().unwrap();
    assert!(!segments.is_empty(), "segments must not be empty");

    // Every word appears in exactly one segment, in order.
    let segment_word_count: usize = segments
        .iter()
        .map(|s| s["words"].as_array().map(|w| w.len()).unwrap_or(0))
        .sum();
    assert_eq!(
        segment_word_count,
        words.len(),
        "segment words must cover all top-level words"
    );

    // Segment timestamps are monotonic and non-overlapping.
    let mut last_end = -1.0_f64;
    for seg in segments {
        let start = seg["start"].as_f64().expect("segment start number");
        let end = seg["end"].as_f64().expect("segment end number");
        assert!(
            start >= last_end,
            "segment start {start} < previous end {last_end}"
        );
        assert!(end >= start, "segment end {end} < start {start}");
        last_end = end;
    }

    let _ = shutdown.send(());
}

#[ignore]
#[tokio::test]
async fn test_transcribe_segments_true_with_channels_split_carries_speaker() {
    let (port, shutdown) = common::start_server(&common::model_dir()).await;
    let wav = common::generate_stereo_wav_split(
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/golos_00.wav"),
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/golos_01.wav"),
    );

    let resp = tokio::time::timeout(Duration::from_secs(60), async {
        reqwest::Client::new()
            .post(format!(
                "http://127.0.0.1:{port}/v1/transcribe?channels=split&segments=true"
            ))
            .body(wav)
            .send()
            .await
            .expect("POST /v1/transcribe failed")
    })
    .await
    .expect("POST /v1/transcribe timed out");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("expected text body");
    let body: serde_json::Value = serde_json::from_str(&text).expect("expected JSON");

    let segments = body["segments"].as_array().expect("segments array");
    assert!(!segments.is_empty());

    let mut saw_speaker = false;
    for seg in segments {
        if let Some(speaker) = seg["speaker"].as_u64() {
            assert!(speaker == 0 || speaker == 1);
            saw_speaker = true;
        }
    }
    assert!(
        saw_speaker,
        "at least one segment must carry a speaker label"
    );

    let _ = shutdown.send(());
}

#[ignore]
#[tokio::test]
async fn test_transcribe_md_segments_emits_headers() {
    let (port, shutdown) = common::start_server(&common::model_dir()).await;
    // Real speech fixture: a tone yields no words, hence no "### [" headers.
    let wav = std::fs::read(GOLOS_FIXTURE).expect("read golos fixture");

    let resp = tokio::time::timeout(Duration::from_secs(60), async {
        reqwest::Client::new()
            .post(format!(
                "http://127.0.0.1:{port}/v1/transcribe?format=md&segments=true"
            ))
            .body(wav)
            .send()
            .await
            .expect("POST /v1/transcribe failed")
    })
    .await
    .expect("POST /v1/transcribe timed out");

    assert_eq!(resp.status(), 200);
    let md = resp.text().await.expect("expected text body");
    assert!(
        md.contains("### ["),
        "segment-grouped Markdown must contain '### [' headers, got:\n{md}"
    );

    let _ = shutdown.send(());
}

// ---------------------------------------------------------------------------
// Telephony codecs: G.711 / G.722 in WAV containers and headerless raw streams
// ---------------------------------------------------------------------------

/// Directory with the telephony transcodes of the golos_00 speech fixture
/// ("шестьдесят тысяч тенге сколько будет стоить"), generated by
/// scripts/generate_telephony_fixtures.sh.
const TELEPHONY_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/telephony");

/// Keywords from the golos_00 reference transcript. Band-limited (8 kHz)
/// telephony transcodes may drop a word, so any keyword counts as a sensible
/// transcription.
const TELEPHONY_KEYWORDS: [&str; 3] = ["шестьдесят", "тенге", "стоить"];

/// Decode-level smoke test for the telephony fixtures (no model needed, runs
/// on every PR): every container fixture must decode through the generic
/// bytes path, and every raw stream through the explicit codec decode, to
/// ≈4 s of 16 kHz audio.
#[test]
fn test_telephony_fixtures_decode_without_model() {
    use gigastt::inference::audio::{TelephonyCodec, decode_telephony_raw};

    // 4 s fixture: G.711 WAVs resample 8k → 16k (FIR delay trims the tail);
    // G.722 WAV decodes to exactly 64_000 samples at its native 16 kHz.
    for (fixture, min_samples) in [
        ("speech_alaw.wav", 40_000),
        ("speech_mulaw.wav", 40_000),
        ("speech_g722.wav", 64_000),
    ] {
        let bytes = std::fs::read(format!("{TELEPHONY_DIR}/{fixture}")).expect("read fixture");
        let samples = gigastt::inference::audio::decode_audio_bytes(&bytes)
            .unwrap_or_else(|e| panic!("{fixture} must decode: {e:#}"));
        assert!(
            samples.len() >= min_samples,
            "{fixture}: expected ≥{min_samples} samples, got {}",
            samples.len()
        );
    }

    // Headerless streams via the explicit codec decode.
    for (fixture, codec, rate, min_samples) in [
        ("speech.ulaw", TelephonyCodec::Pcmu, 8000, 40_000),
        ("speech.alaw", TelephonyCodec::Pcma, 8000, 40_000),
        ("speech.g722", TelephonyCodec::G722, 8000, 64_000),
    ] {
        let bytes = std::fs::read(format!("{TELEPHONY_DIR}/{fixture}")).expect("read fixture");
        let samples = decode_telephony_raw(&bytes, codec, rate)
            .unwrap_or_else(|e| panic!("{fixture} must decode: {e:#}"));
        assert!(
            samples.len() >= min_samples,
            "{fixture}: expected ≥{min_samples} samples, got {}",
            samples.len()
        );
    }
}

/// Assert the transcript carries at least one keyword from the reference.
fn assert_sensible_transcript(body: &serde_json::Value, case: &str) {
    let text = body["text"].as_str().unwrap_or_default().to_lowercase();
    assert!(
        TELEPHONY_KEYWORDS.iter().any(|kw| text.contains(kw)),
        "{case}: transcript must contain a reference keyword, got: {text:?}"
    );
}

#[ignore]
#[tokio::test]
async fn test_transcribe_telephony_wav_codecs() {
    let (port, shutdown) = common::start_server(&common::model_dir()).await;

    // A-law and μ-law WAV decode via symphonia's PCM codec; G.722 WAV via the
    // dedicated fallback decoder. All must produce a sensible transcript.
    for (fixture, case) in [
        ("speech_alaw.wav", "G.711 A-law WAV"),
        ("speech_mulaw.wav", "G.711 μ-law WAV"),
        ("speech_g722.wav", "G.722 WAV"),
    ] {
        let bytes = std::fs::read(format!("{TELEPHONY_DIR}/{fixture}")).expect("read fixture");
        let (status, text) = post_transcribe(port, "", bytes).await;
        let body: serde_json::Value = serde_json::from_str(&text).expect("expected JSON body");
        assert_eq!(status, 200, "{case} must be accepted, got: {body}");
        assert_sensible_transcript(&body, case);
    }

    let _ = shutdown.send(());
}

#[ignore]
#[tokio::test]
async fn test_transcribe_telephony_raw_streams_with_codec_param() {
    let (port, shutdown) = common::start_server(&common::model_dir()).await;

    // Headerless streams (RTP-dump style) need the explicit codec + rate.
    // G.722 is exercised with the SDP clock-rate alias 8000; the decoder
    // always outputs its native 16 kHz.
    for (fixture, query, case) in [
        ("speech.ulaw", "codec=pcmu&sample_rate=8000", "raw μ-law"),
        ("speech.alaw", "codec=pcma&sample_rate=8000", "raw A-law"),
        ("speech.g722", "codec=g722&sample_rate=8000", "raw G.722"),
    ] {
        let bytes = std::fs::read(format!("{TELEPHONY_DIR}/{fixture}")).expect("read fixture");
        let (status, text) = post_transcribe(port, query, bytes).await;
        let body: serde_json::Value = serde_json::from_str(&text).expect("expected JSON body");
        assert_eq!(status, 200, "{case} must be accepted, got: {body}");
        assert_sensible_transcript(&body, case);
    }

    let _ = shutdown.send(());
}

#[ignore]
#[tokio::test]
async fn test_transcribe_raw_stream_without_codec_returns_422() {
    let (port, shutdown) = common::start_server(&common::model_dir()).await;

    // A headerless stream without `?codec=` keeps the previous behavior: the
    // container probe fails and the upload is rejected as undecodable.
    let bytes = std::fs::read(format!("{TELEPHONY_DIR}/speech.ulaw")).expect("read fixture");
    let (status, text) = post_transcribe(port, "", bytes).await;
    assert_eq!(status, 422, "raw stream without codec must be rejected");
    let body: serde_json::Value = serde_json::from_str(&text).expect("expected JSON body");
    let code = body["code"].as_str().unwrap_or_default();
    assert!(
        code == "invalid_audio" || code == "transcription_error",
        "expected invalid_audio/transcription_error, got: {code:?}"
    );

    let _ = shutdown.send(());
}

#[ignore]
#[tokio::test]
async fn test_transcribe_codec_query_validation_errors() {
    let (port, shutdown) = common::start_server(&common::model_dir()).await;
    let bytes = std::fs::read(format!("{TELEPHONY_DIR}/speech.ulaw")).expect("read fixture");

    // Unknown codec name → 400 unsupported_codec.
    let (status, text) = post_transcribe(port, "codec=g729&sample_rate=8000", bytes.clone()).await;
    assert_eq!(status, 400);
    let body: serde_json::Value = serde_json::from_str(&text).expect("expected JSON body");
    assert_eq!(body["code"], "unsupported_codec");

    // codec without the mandatory sample_rate → 400 invalid_sample_rate.
    let (status, text) = post_transcribe(port, "codec=pcmu", bytes.clone()).await;
    assert_eq!(status, 400);
    let body: serde_json::Value = serde_json::from_str(&text).expect("expected JSON body");
    assert_eq!(body["code"], "invalid_sample_rate");

    // G.722 accepts only 8000 (SDP alias) / 16000 → 400 invalid_sample_rate.
    let (status, text) = post_transcribe(port, "codec=g722&sample_rate=44100", bytes).await;
    assert_eq!(status, 400);
    let body: serde_json::Value = serde_json::from_str(&text).expect("expected JSON body");
    assert_eq!(body["code"], "invalid_sample_rate");

    let _ = shutdown.send(());
}
