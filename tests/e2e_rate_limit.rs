//! End-to-end rate-limiter test. Boots a real server with `--rate-limit-per-minute 30
//! --rate-limit-burst 1`, confirms the second request lands on 429 and that
//! waiting the refill window restores service.
//!
//! Requires the model (like the rest of the `e2e_*` suite).
//! Run with: `cargo test --test e2e_rate_limit -- --ignored`

mod common;

use std::time::Duration;

/// With `burst = 1`, the first request consumes the single token and every
/// subsequent request within the refill window must 429. After waiting the
/// refill interval (30 rpm → 1 token every 2 s) the bucket refills and the
/// follow-up succeeds.
#[tokio::test]
#[ignore]
async fn test_rate_limit_burst_then_refill() {
    let model_dir = common::model_dir();
    let limits = gigastt::server::RuntimeLimits {
        rate_limit_per_minute: 30,
        rate_limit_burst: 1,
        ..Default::default()
    };
    let (port, shutdown) = common::start_server_with_limits(&model_dir, limits).await;

    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{port}/v1/models");

    // First request consumes the single token.
    let r1 = client.get(&url).send().await.expect("first request");
    assert_eq!(r1.status().as_u16(), 200, "first request should succeed");

    // Second request immediately after must be rate-limited.
    let r2 = client.get(&url).send().await.expect("second request");
    assert_eq!(
        r2.status().as_u16(),
        429,
        "second request should be rate-limited, got {}",
        r2.status()
    );
    let retry_after = r2
        .headers()
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    assert_eq!(
        retry_after.as_deref(),
        Some("60"),
        "Retry-After header must be set to 60"
    );
    let body_text = r2.text().await.expect("429 body readable");
    let body: serde_json::Value =
        serde_json::from_str(&body_text).expect("429 body is JSON");
    assert_eq!(body["code"], "rate_limited");

    // Wait more than the refill interval (2 s for 30 rpm) and retry.
    tokio::time::sleep(Duration::from_millis(2_500)).await;
    let r3 = client.get(&url).send().await.expect("third request");
    assert_eq!(
        r3.status().as_u16(),
        200,
        "request should succeed after the refill window"
    );

    let _ = shutdown.send(());
}
