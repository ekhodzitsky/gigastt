//! E2E tests for the loopback-only model hot-reload endpoint
//! (`POST /v1/admin/reload`). Require the model on disk; run with:
//!
//! ```sh
//! cargo test -p gigastt --test e2e_admin_reload -- --ignored --test-threads=1
//! ```

mod common;

use common::{generate_wav, model_dir, start_server_reloadable, ws_connect};
use futures_util::{SinkExt, StreamExt};
use std::time::Duration;
use tokio_tungstenite::tungstenite::Message;

/// Decode a response body as JSON without depending on reqwest's `json` feature
/// (the crate enables only `stream`), matching the other e2e suites.
async fn json_body(resp: reqwest::Response) -> serde_json::Value {
    let text = resp.text().await.expect("Expected text body");
    serde_json::from_str(&text).expect("Expected JSON body")
}

/// Happy path: a reload swaps in a fresh engine and the server keeps serving.
#[tokio::test]
#[ignore = "requires model"]
async fn test_reload_happy_swap() {
    let dir = model_dir();
    let (port, _shutdown) = start_server_reloadable(&dir, &dir).await;
    let base = format!("http://127.0.0.1:{port}");
    let client = reqwest::Client::new();

    // Baseline: /v1/models serves a working model.
    let before = json_body(
        client
            .get(format!("{base}/v1/models"))
            .send()
            .await
            .unwrap(),
    )
    .await;
    let before_variant = before["variant"].as_str().unwrap().to_string();

    // Reload → 200 { reloaded: true, variant, encoder }.
    let resp = client
        .post(format!("{base}/v1/admin/reload"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "reload should succeed");
    let body = json_body(resp).await;
    assert_eq!(body["reloaded"], true);
    assert_eq!(body["variant"], before_variant);
    assert!(
        body["encoder"] == "int8" || body["encoder"] == "fp32",
        "encoder must be reported: {body:?}"
    );

    // /v1/models still serves the same model after the swap.
    let after = json_body(
        client
            .get(format!("{base}/v1/models"))
            .send()
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(after["variant"], before_variant);

    // A fresh transcription succeeds against the swapped-in engine.
    let wav = generate_wav(1, 16000);
    let resp = client
        .post(format!("{base}/v1/transcribe"))
        .body(wav)
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "transcribe must work after the model swap"
    );
    let v = json_body(resp).await;
    assert!(v.get("text").is_some(), "response must carry a text field");
}

/// Two simultaneous reloads: exactly one wins (200), the other is rejected with
/// 409 `reload_in_progress`.
#[tokio::test]
#[ignore = "requires model"]
async fn test_reload_concurrency_returns_409() {
    let dir = model_dir();
    let (port, _shutdown) = start_server_reloadable(&dir, &dir).await;
    let base = format!("http://127.0.0.1:{port}");
    let client = reqwest::Client::new();

    let a = client.post(format!("{base}/v1/admin/reload")).send();
    let b = client.post(format!("{base}/v1/admin/reload")).send();
    let (ra, rb) = tokio::join!(a, b);
    let (ra, rb) = (ra.unwrap(), rb.unwrap());

    let mut statuses = [ra.status().as_u16(), rb.status().as_u16()];
    statuses.sort_unstable();
    assert_eq!(
        statuses,
        [200, 409],
        "one reload must win (200) and the other must get 409, got {statuses:?}"
    );

    // The 409 body carries the machine-readable code.
    for r in [ra, rb] {
        if r.status() == 409 {
            let v = json_body(r).await;
            assert_eq!(v["code"], "reload_in_progress");
        }
    }
}

/// A build failure (garbage model dir) must keep the ORIGINAL engine serving —
/// the server is never left engineless.
#[tokio::test]
#[ignore = "requires model"]
async fn test_reload_bad_dir_keeps_old_engine() {
    let good = model_dir();
    let bad = tempfile::tempdir().unwrap();
    let bad_dir = bad.path().to_string_lossy().into_owned();

    let (port, _shutdown) = start_server_reloadable(&good, &bad_dir).await;
    let base = format!("http://127.0.0.1:{port}");
    let client = reqwest::Client::new();

    // Reload from the empty dir → 503 reload_failed.
    let resp = client
        .post(format!("{base}/v1/admin/reload"))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        503,
        "reload from a bad model dir must fail with 503"
    );
    let body = json_body(resp).await;
    assert_eq!(body["code"], "reload_failed");
    // Sanitized message — must not leak the bad path.
    let msg = body["error"].as_str().unwrap();
    assert!(
        !msg.contains(bad.path().to_string_lossy().as_ref()),
        "error message must not leak the model path: {msg}"
    );

    // The original engine still serves both /v1/models and /v1/transcribe.
    let resp = client
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "old engine must still serve /v1/models");

    let wav = generate_wav(1, 16000);
    let resp = client
        .post(format!("{base}/v1/transcribe"))
        .body(wav)
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "old engine must still transcribe after a failed reload"
    );
}

/// A WebSocket session started before the reload rides the old pool to a clean
/// `Final`, AND a fresh REST transcription after the swap succeeds. Proves the
/// swap drops no in-flight work.
#[tokio::test]
#[ignore = "requires model"]
async fn test_reload_does_not_drop_inflight_ws() {
    let dir = model_dir();
    let (port, _shutdown) = start_server_reloadable(&dir, &dir).await;
    let base = format!("http://127.0.0.1:{port}");
    let client = reqwest::Client::new();

    // Open a WS stream and feed some audio (48 kHz default sample rate).
    let (mut sink, mut stream, _ready) = ws_connect(port).await;
    let pcm = common::generate_pcm16_tone(0.5, 48000, 440.0);
    sink.send(Message::Binary(pcm.clone().into()))
        .await
        .unwrap();

    // Fire the reload mid-session.
    let reload_resp = client
        .post(format!("{base}/v1/admin/reload"))
        .send()
        .await
        .unwrap();
    assert_eq!(
        reload_resp.status(),
        200,
        "mid-session reload should succeed"
    );

    // Feed a little more, then Stop — the session must complete with a Final,
    // proving it kept its (old) pool across the swap.
    sink.send(Message::Binary(pcm.into())).await.unwrap();
    sink.send(Message::Text(r#"{"type":"stop"}"#.to_string().into()))
        .await
        .unwrap();

    let mut saw_final = false;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_secs(5), stream.next()).await {
            Ok(Some(Ok(msg))) => {
                if let Ok(text) = msg.into_text()
                    && let Ok(v) = serde_json::from_str::<serde_json::Value>(&text)
                    && v["type"] == "final"
                {
                    saw_final = true;
                    break;
                }
            }
            _ => break,
        }
    }
    assert!(
        saw_final,
        "in-flight WS session must still complete with a Final after a mid-session reload"
    );

    // A fresh REST transcription after the swap succeeds on the new engine.
    let wav = generate_wav(1, 16000);
    let resp = client
        .post(format!("{base}/v1/transcribe"))
        .body(wav)
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "a new request after the swap must hit a working engine"
    );
}
