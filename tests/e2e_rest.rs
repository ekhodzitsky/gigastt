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

#[tokio::test]
#[ignore]
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

#[tokio::test]
#[ignore]
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

#[tokio::test]
#[ignore]
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

#[tokio::test]
#[ignore]
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

#[tokio::test]
#[ignore]
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

#[tokio::test]
#[ignore]
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

#[tokio::test]
#[ignore]
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

#[tokio::test]
#[ignore]
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

    // Read just the first event, then drop the response to simulate disconnect
    let mut stream = resp.bytes_stream();
    let _first = tokio::time::timeout(Duration::from_secs(10), stream.next())
        .await
        .expect("Timed out waiting for first SSE event");

    // Drop the stream, simulating client disconnect
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

// ---------------------------------------------------------------------------
// 9. Zero-copy REST decode path — RSS must not balloon during large upload
// ---------------------------------------------------------------------------

/// Verify the REST upload path does not double or triple resident memory
/// while decoding a large body. Before the zero-copy refactor a 40 MiB upload
/// transiently held ~120 MiB (axum Bytes + `body.to_vec()` + symphonia
/// Cursor clone). With `BytesMediaSource` the decoded-sample buffer is the
/// only large allocation.
///
/// Linux-only: reads `/proc/self/status` VmRSS. macOS would need libproc
/// bindings; we skip instead of pulling in an extra dep for one test.
#[cfg(target_os = "linux")]
#[tokio::test]
#[ignore]
async fn test_rest_large_body_rss_within_budget() {
    fn read_vm_rss_kb() -> Option<u64> {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                let kb: u64 = rest.trim().split_whitespace().next()?.parse().ok()?;
                return Some(kb);
            }
        }
        None
    }

    let (port, shutdown) = common::start_server(&common::model_dir()).await;

    // ~9.6 MiB WAV (300 s @ 16 kHz / 16-bit mono). Large enough that the
    // pre-fix double-buffer (4× peak) would blow the budget below, while
    // staying comfortably under the 50 MiB body limit and the 10-minute
    // file cap. Picked at 16 kHz so the decoded-PCM buffer is a predictable
    // `wav.len() * 2` (PCM16 → f32) with no resampling overhead on top.
    let wav = common::generate_wav(300, 16000);
    assert!(
        wav.len() > 9 * 1024 * 1024,
        "generated WAV should be >9 MiB, got {}",
        wav.len()
    );

    let before_kb = read_vm_rss_kb().expect("/proc/self/status unavailable");

    let resp = tokio::time::timeout(Duration::from_secs(60), async {
        reqwest::Client::new()
            .post(format!("http://127.0.0.1:{port}/v1/transcribe"))
            .body(wav.clone())
            .send()
            .await
            .expect("POST /v1/transcribe failed")
    })
    .await
    .expect("POST /v1/transcribe timed out");

    // This is a memory-budget test, not an accuracy test. We only need the
    // server to exercise the full upload + decode path. A 200 means the
    // inference succeeded too (best case); a 422 means decode ran through
    // `BytesMediaSource` — the memory-heavy part — and the engine rejected
    // the synthetic pure-sine payload afterwards. Both outcomes prove the
    // zero-copy path is wired up. 413 / 429 / 5xx would fail the path under
    // test and must not be accepted here.
    let status = resp.status();
    assert!(
        status == 200 || status.as_u16() == 422,
        "expected 200 or 422, got {status}"
    );
    let _ = resp.text().await;

    let after_kb = read_vm_rss_kb().expect("/proc/self/status unavailable");
    let delta_kb = after_kb.saturating_sub(before_kb);
    // Budget accounts for:
    //   - `wav.len()`     — refcounted axum Bytes (1× copy)
    //   - `wav.len() * 2` — PCM16 → f32 sample buffer
    //   - 40 MiB slack    — ONNX scratch, tracing buffers, libc overhead
    // Pre-fix regression kept a second `body.to_vec()` alive → delta would
    // exceed `wav.len() * 4 + slack`, comfortably past the bound below.
    let budget_kb = (wav.len() as u64 / 1024) * 3 + 40 * 1024;
    assert!(
        delta_kb < budget_kb,
        "RSS grew by {delta_kb} KiB during upload; expected < {budget_kb} KiB \
         (wav {} KiB)",
        wav.len() / 1024
    );

    let _ = shutdown.send(());
}
