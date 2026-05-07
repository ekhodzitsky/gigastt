//! HTTP middleware: origin policy enforcement and metrics instrumentation.

use std::sync::Arc;
use axum::extract::State;
use axum::response::Response;
use super::config::{OriginPolicy, OriginVerdict};
use super::http;

/// Instrumentation middleware: records HTTP request counters and a duration
/// histogram under the `gigastt_http_*` namespace. Takes the whole
/// `AppState` so we can reach `metrics_registry` — when the operator did
/// not pass `--metrics` the registry is `None` and the middleware
/// degrades to a single `Instant::now()` per request.
pub(crate) async fn http_metrics_middleware(
    State(state): State<Arc<http::AppState>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let Some(registry) = state.metrics_registry.clone() else {
        return next.run(req).await;
    };
    let method = req.method().as_str().to_string();
    let path = req.uri().path().to_string();
    let start = std::time::Instant::now();
    // Sample pool availability on every request.
    registry.gauge_set(
        "gigastt_pool_available",
        vec![],
        state.engine.pool.available() as i64,
    );
    let response = next.run(req).await;
    let elapsed = start.elapsed().as_secs_f64();
    let status = response.status().as_u16().to_string();
    registry.counter_inc(
        "gigastt_http_requests_total",
        vec![
            ("method".into(), method.clone()),
            ("path".into(), path.clone()),
            ("status".into(), status),
        ],
        1,
    );
    registry.histogram_record(
        "gigastt_http_request_duration_seconds",
        vec![("method".into(), method), ("path".into(), path)],
        elapsed,
    );
    response
}

pub(crate) async fn origin_middleware(
    policy: Arc<OriginPolicy>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    use axum::http::{StatusCode, header};
    use axum::response::IntoResponse;

    // `/health` is a liveness probe for container orchestrators and monitoring
    // tools that don't send Origin — let it through unconditionally.
    if req.uri().path() == "/health" {
        return next.run(req).await;
    }

    let origin = req
        .headers()
        .get("origin")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);

    match policy.evaluate(origin.as_deref()) {
        OriginVerdict::AllowedNoEcho => next.run(req).await,
        OriginVerdict::Allowed(echo) => {
            let mut response = next.run(req).await;
            let headers = response.headers_mut();
            let value = if policy.allow_any { "*".into() } else { echo };
            if let Ok(v) = axum::http::HeaderValue::from_str(&value) {
                headers.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, v);
            }
            headers.insert(
                header::ACCESS_CONTROL_ALLOW_METHODS,
                axum::http::HeaderValue::from_static("GET, POST, OPTIONS"),
            );
            headers.insert(
                header::ACCESS_CONTROL_ALLOW_HEADERS,
                axum::http::HeaderValue::from_static("*"),
            );
            headers.insert(header::VARY, axum::http::HeaderValue::from_static("origin"));
            response
        }
        OriginVerdict::Denied => {
            let origin_str = origin.as_deref().unwrap_or("");
            let path = req.uri().path().to_string();
            tracing::warn!(
                origin = %origin_str,
                path = %path,
                "cross-origin request denied by default policy"
            );
            (
                StatusCode::FORBIDDEN,
                axum::response::Json(serde_json::json!({
                    "error": "Origin not allowed",
                    "code": "origin_denied",
                })),
            )
                .into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_origin_middleware_integration() {
        // End-to-end check of the origin_middleware layer on a minimal
        // router. Uses real axum::serve + reqwest to catch regressions that
        // unit tests on `OriginPolicy` alone would miss — e.g. the middleware
        // attaching to the wrong routes, or `/health` accidentally being
        // guarded.
        use axum::Router;
        use axum::routing::get;

        let policy = Arc::new(OriginPolicy::loopback_only());
        let origin_layer = {
            let policy = policy.clone();
            axum::middleware::from_fn(move |req, next| {
                let policy = policy.clone();
                async move { origin_middleware(policy, req, next).await }
            })
        };
        let app = Router::new()
            .route("/health", get(|| async { "ok" }))
            .route("/v1/transcribe", get(|| async { "ok" }))
            .layer(origin_layer);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        // Allow the server to bind.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let client = reqwest::Client::new();
        let base = format!("http://127.0.0.1:{port}");

        // /health is exempt — monitoring probes work even when Origin is set.
        let r = client
            .get(format!("{base}/health"))
            .header("Origin", "https://evil.example.com")
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "/health must skip the Origin guard");

        // Cross-origin request must be denied on /v1/*.
        let r = client
            .get(format!("{base}/v1/transcribe"))
            .header("Origin", "https://evil.example.com")
            .send()
            .await
            .unwrap();
        assert_eq!(
            r.status(),
            403,
            "non-loopback Origin must receive 403 Forbidden"
        );
        let text = r.text().await.unwrap();
        let body: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(body["code"], "origin_denied");

        // Loopback origin is always allowed.
        let r = client
            .get(format!("{base}/v1/transcribe"))
            .header("Origin", "http://localhost:3000")
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "loopback Origin must be allowed");
        assert_eq!(
            r.headers()
                .get("access-control-allow-origin")
                .and_then(|v| v.to_str().ok()),
            Some("http://localhost:3000"),
            "CORS echo must mirror the incoming Origin (no wildcard by default)",
        );

        // No Origin header (curl, CLI, server-to-server) — policy allows
        // through without a CORS echo.
        let r = client
            .get(format!("{base}/v1/transcribe"))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "requests without Origin must pass");

        // Attacker trying DNS continuation on the loopback prefix must be denied.
        let r = client
            .get(format!("{base}/v1/transcribe"))
            .header("Origin", "http://localhost.evil.example.com")
            .send()
            .await
            .unwrap();
        assert_eq!(
            r.status(),
            403,
            "localhost.* DNS continuation must not impersonate loopback"
        );
    }
}
