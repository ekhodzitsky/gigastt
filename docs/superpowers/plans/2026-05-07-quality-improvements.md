# Quality Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 8 quality improvements across 3 phases to bring the project from 82/100 to near-100 on code quality, observability, documentation, and testing.

**Architecture:** Three phases grouped by dependency order — refactoring first (clean codebase for subsequent work), then new server functionality, then quality/testing improvements. Each phase is a self-contained commit series.

**Tech Stack:** Rust 2024, axum, tracing, arc-swap, uuid, proptest, TOML config, GitHub Actions

**Spec:** `docs/superpowers/specs/2026-05-07-quality-improvements-design.md`

---

## Phase 1: Refactoring

### Task 1: Remove deprecated `/ws` path

**Files:**
- Modify: `src/server/mod.rs` (lines 376-380, 459, 672-717)
- Modify: `tests/e2e_ws.rs` (line 346)
- Modify: `tests/e2e_errors.rs` (lines 81, 153)
- Modify: `docs/asyncapi.yaml` (lines 13, 72)
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update E2E tests — change `/ws` to `/v1/ws`**

In `tests/e2e_ws.rs`, line 346:
```rust
// Before:
let url = format!("ws://127.0.0.1:{port}/ws");
// After:
let url = format!("ws://127.0.0.1:{port}/v1/ws");
```

In `tests/e2e_errors.rs`, line 81:
```rust
// Before:
format!("ws://127.0.0.1:{port}/ws"),
// After:
format!("ws://127.0.0.1:{port}/v1/ws"),
```

In `tests/e2e_errors.rs`, line 153:
```rust
// Before:
let (mut fifth_ws, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}/ws"))
// After:
let (mut fifth_ws, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}/v1/ws"))
```

- [ ] **Step 2: Delete `ws_handler_legacy` and remove route**

In `src/server/mod.rs`:

Delete the entire `ws_handler_legacy` function (lines 672-717):
```rust
// DELETE this entire function:
/// Deprecated WebSocket endpoint at `/ws`. Identical behaviour to `/v1/ws`
/// ...
async fn ws_handler_legacy( ... ) -> Response { ... }
```

Remove the legacy route from the protected router (line 380):
```rust
// DELETE this line:
.route("/ws", get(ws_handler_legacy))
```

Update the comment on line 376-378:
```rust
// Before:
// `/v1/ws` is the canonical path (versioned, aligned with REST); `/ws`
// remains as an alias for existing clients and logs a deprecation
// warning on each upgrade. Planned for removal in a future major version.
// After:
// `/v1/ws` is the canonical WebSocket path (versioned, aligned with REST).
```

- [ ] **Step 3: Update startup log**

In `src/server/mod.rs`, line 459:
```rust
// Before:
tracing::info!("  WebSocket: ws://{addr}/v1/ws (legacy alias: ws://{addr}/ws)");
// After:
tracing::info!("  WebSocket: ws://{addr}/v1/ws");
```

- [ ] **Step 4: Update docs**

In `docs/asyncapi.yaml`, line 13:
```yaml
# Before:
#    `/ws` remains as a deprecated alias — removal planned for v1.0.
# DELETE this line entirely.
```

In `docs/asyncapi.yaml`, lines 71-72:
```yaml
# Before:
#       `/ws` remains as a deprecated alias — removal planned for v1.0.
# DELETE this line entirely.
```

In `CLAUDE.md`, line 214 (the `run()` docstring mentions `/ws`):
```rust
// In src/server/mod.rs line 214, update the doc comment:
// Before:
/// - `GET /ws` — WebSocket streaming protocol
// After:
/// - `GET /v1/ws` — WebSocket streaming protocol
```

In `CLAUDE.md`, find and update references to deprecated `/ws`:
- Remove the line: `- Canonical WS path: `/v1/ws` (v0.7.0+). `/ws` remains as a deprecated alias...`
- Replace with: `- WebSocket path: `/v1/ws``

- [ ] **Step 5: Run tests and verify**

Run: `cargo test 2>&1 | tail -5`
Expected: all 142 unit tests pass, no compilation errors from removed function.

Run: `cargo clippy -- -D warnings -A dead_code`
Expected: no warnings.

- [ ] **Step 6: Commit**

```bash
git add src/server/mod.rs tests/e2e_ws.rs tests/e2e_errors.rs docs/asyncapi.yaml CLAUDE.md
git commit -m "refactor: remove deprecated /ws WebSocket alias"
```

---

### Task 2: Create `server/config.rs`

**Files:**
- Create: `src/server/config.rs`
- Modify: `src/server/mod.rs`

- [ ] **Step 1: Create `src/server/config.rs` with types moved from `mod.rs`**

Move the following from `src/server/mod.rs` into `src/server/config.rs`:
- `SUPPORTED_RATES` constant (line 34)
- `DEFAULT_SAMPLE_RATE` constant (line 35)
- `pool_retry_after_ms()` function (lines 40-45)
- `pool_retry_after_secs()` function (lines 46-48)
- `OriginPolicy` struct (lines 50-65)
- `OriginPolicy::loopback_only()` impl (lines 67-73)
- `OriginVerdict` enum (lines 75-86)
- `is_loopback_origin()` function (lines 88-106)
- `OriginPolicy::evaluate()` impl (lines 108-128)
- `RuntimeLimits` struct + Default impl (lines 130-173)
- `ServerConfig` struct + `local()` impl (lines 175-206)
- Related tests: `test_runtime_limits_*`, `test_supported_rates_*`, `test_default_sample_rate_*`, `test_loopback_origin_*`, `test_origin_policy_*` (lines 1331-1453)

The file header:
```rust
//! Server configuration types, origin policy, and runtime limits.

use std::sync::Arc;

/// Supported input sample rates (Hz). Default is 48000 for backward
/// compatibility.
pub(crate) const SUPPORTED_RATES: &[u32] = &[8000, 16000, 24000, 44100, 48000];
pub(crate) const DEFAULT_SAMPLE_RATE: u32 = 48000;
```

All types keep their existing visibility. `OriginVerdict` becomes `pub(crate)` (used by middleware).

- [ ] **Step 2: Update `src/server/mod.rs` — add module declaration and re-exports**

At the top of `mod.rs`, add:
```rust
pub mod config;
```

Add re-exports:
```rust
pub use config::{OriginPolicy, RuntimeLimits, ServerConfig};
pub(crate) use config::{
    DEFAULT_SAMPLE_RATE, SUPPORTED_RATES, pool_retry_after_ms, pool_retry_after_secs,
};
```

Remove all moved code from `mod.rs`. Update internal references to use the re-exports (most should work transparently due to `use config::*` style).

- [ ] **Step 3: Run tests**

Run: `cargo test 2>&1 | tail -5`
Expected: all tests pass.

Run: `cargo clippy -- -D warnings -A dead_code`
Expected: no warnings.

- [ ] **Step 4: Commit**

```bash
git add src/server/config.rs src/server/mod.rs
git commit -m "refactor: extract server/config.rs from server/mod.rs"
```

---

### Task 3: Create `server/middleware.rs`

**Files:**
- Create: `src/server/middleware.rs`
- Modify: `src/server/mod.rs`

- [ ] **Step 1: Create `src/server/middleware.rs`**

Move from `src/server/mod.rs`:
- `origin_middleware()` function (lines 570-628)
- `http_metrics_middleware()` function (lines 533-568)

```rust
//! HTTP middleware: origin policy enforcement and metrics instrumentation.

use std::sync::Arc;

use axum::extract::State;
use axum::response::Response;

use super::config::{OriginPolicy, OriginVerdict};
use super::http::AppState;
use super::metrics::MetricsRegistry;
```

Keep the `test_origin_middleware_integration` test — move it to `config.rs` tests (it tests `OriginPolicy` + middleware together, but the middleware function is now in `middleware.rs`; alternatively keep it in `middleware.rs` with its own `#[cfg(test)] mod tests`).

- [ ] **Step 2: Update `mod.rs` — add module, fix references**

```rust
pub(crate) mod middleware;
```

In `run_with_config_listener()`, update the `origin_layer` closure to reference `middleware::origin_middleware`:
```rust
let origin_layer = {
    let policy = policy.clone();
    axum::middleware::from_fn(move |req, next| {
        let policy = policy.clone();
        async move { middleware::origin_middleware(policy, req, next).await }
    })
};
```

Update the metrics middleware layer:
```rust
.layer(axum::middleware::from_fn_with_state(
    state.clone(),
    middleware::http_metrics_middleware,
))
```

- [ ] **Step 3: Run tests**

Run: `cargo test 2>&1 | tail -5`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/server/middleware.rs src/server/mod.rs
git commit -m "refactor: extract server/middleware.rs from server/mod.rs"
```

---

### Task 4: Create `server/ws.rs`

**Files:**
- Create: `src/server/ws.rs`
- Modify: `src/server/mod.rs`

- [ ] **Step 1: Create `src/server/ws.rs`**

Move from `src/server/mod.rs`:
- `FrameOutcome` enum (lines 819-824)
- `WsSink` type alias (line 826)
- `ws_handler()` function (lines 630-670)
- `handle_ws()` function (lines 719-815)
- `send_server_message()` function (lines 830-834)
- `handle_binary_frame()` function (lines 841-979)
- `handle_configure_message()` function (lines 986-1054)
- `handle_stop_message()` function (lines 1058-1082)
- `flush_and_final()` function (lines 1088-1106)
- `handle_ws_inner()` function (lines 1111-1325)
- Test: `test_catch_unwind_preserves_ownership_across_panic` (lines 1584-1607)

Header:
```rust
//! WebSocket handler: upgrade, session loop, PCM16 processing, and inference dispatch.

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Result;
use axum::extract::State;
use axum::extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade};
use axum::response::Response;
use futures_util::{SinkExt, StreamExt};

use super::config::{
    DEFAULT_SAMPLE_RATE, RuntimeLimits, SUPPORTED_RATES, pool_retry_after_ms,
};
use super::http::AppState;
use super::json_text;
use crate::inference::{Engine, SessionTriplet};
use crate::protocol::{ClientMessage, ServerMessage};
```

- [ ] **Step 2: Update `mod.rs` — add module, fix router reference**

```rust
mod ws;
```

In the protected router, change `get(ws_handler)` to `get(ws::ws_handler)`:
```rust
.route("/v1/ws", get(ws::ws_handler))
```

Make `ws_handler` `pub(super)` in `ws.rs` so `mod.rs` can reference it.

- [ ] **Step 3: Run tests**

Run: `cargo test 2>&1 | tail -5`
Expected: all tests pass.

Run: `cargo clippy -- -D warnings -A dead_code`
Expected: no warnings.

- [ ] **Step 4: Verify file sizes**

Run: `wc -l src/server/*.rs`
Expected: `config.rs` ~160, `middleware.rs` ~120, `ws.rs` ~700, `http.rs` ~507, `mod.rs` ~280 — all under 750 lines.

- [ ] **Step 5: Commit**

```bash
git add src/server/ws.rs src/server/mod.rs
git commit -m "refactor: extract server/ws.rs from server/mod.rs"
```

---

## Phase 2: Functionality

### Task 5: Add readiness endpoint `/ready`

**Files:**
- Modify: `src/server/http.rs`
- Modify: `src/server/mod.rs`
- Modify: `src/server/middleware.rs`

- [ ] **Step 1: Write the failing test**

Add to `src/server/http.rs` tests:
```rust
#[test]
fn test_readiness_response_ready_serialization() {
    let resp = ReadinessResponse {
        status: "ready".into(),
        pool_available: 3,
        pool_total: 4,
        reason: None,
    };
    let json = serde_json::to_value(&resp).unwrap();
    assert_eq!(json["status"], "ready");
    assert_eq!(json["pool_available"], 3);
    assert_eq!(json["pool_total"], 4);
    assert!(json.get("reason").is_none() || json["reason"].is_null());
}

#[test]
fn test_readiness_response_not_ready_serialization() {
    let resp = ReadinessResponse {
        status: "not_ready".into(),
        pool_available: 0,
        pool_total: 4,
        reason: Some("pool_exhausted".into()),
    };
    let json = serde_json::to_value(&resp).unwrap();
    assert_eq!(json["status"], "not_ready");
    assert_eq!(json["reason"], "pool_exhausted");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test test_readiness_response 2>&1 | tail -5`
Expected: FAIL — `ReadinessResponse` not defined.

- [ ] **Step 3: Implement readiness endpoint**

In `src/server/http.rs`, add struct and handler:
```rust
#[derive(Serialize)]
pub struct ReadinessResponse {
    pub status: String,
    pub pool_available: usize,
    pub pool_total: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// GET /ready — readiness probe for k8s and orchestrators.
/// Returns 200 when the server can accept new work, 503 otherwise.
pub async fn readiness(State(state): State<Arc<AppState>>) -> Response {
    if state.shutdown.is_cancelled() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ReadinessResponse {
                status: "not_ready".into(),
                pool_available: 0,
                pool_total: state.engine.pool.total(),
                reason: Some("shutting_down".into()),
            }),
        )
            .into_response();
    }
    let available = state.engine.pool.available();
    if available == 0 {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ReadinessResponse {
                status: "not_ready".into(),
                pool_available: 0,
                pool_total: state.engine.pool.total(),
                reason: Some("pool_exhausted".into()),
            }),
        )
            .into_response();
    }
    Json(ReadinessResponse {
        status: "ready".into(),
        pool_available: available,
        pool_total: state.engine.pool.total(),
        reason: None,
    })
    .into_response()
}
```

- [ ] **Step 4: Register the route**

In `src/server/mod.rs`, add `/ready` to the public router alongside `/health`:
```rust
let app = Router::new()
    .route("/health", get(http::health))
    .route("/ready", get(http::readiness))
    .merge(protected)
    // ...
```

- [ ] **Step 5: Exempt `/ready` from origin middleware**

In `src/server/middleware.rs`, update `origin_middleware` to skip `/ready`:
```rust
// Before:
if req.uri().path() == "/health" {
    return next.run(req).await;
}
// After:
let path = req.uri().path();
if path == "/health" || path == "/ready" {
    return next.run(req).await;
}
```

- [ ] **Step 6: Run tests**

Run: `cargo test test_readiness_response 2>&1 | tail -5`
Expected: both tests PASS.

Run: `cargo test 2>&1 | tail -5`
Expected: all tests pass.

- [ ] **Step 7: Update startup log**

In `src/server/mod.rs`, update the REST API log line:
```rust
// Before:
tracing::info!("  REST API:  http://{addr}/health, /v1/transcribe, /v1/transcribe/stream");
// After:
tracing::info!("  REST API:  http://{addr}/health, /ready, /v1/transcribe, /v1/transcribe/stream");
```

- [ ] **Step 8: Commit**

```bash
git add src/server/http.rs src/server/mod.rs src/server/middleware.rs
git commit -m "feat: add /ready readiness endpoint for k8s probes"
```

---

### Task 6: Add request-ID middleware and spans

**Files:**
- Modify: `Cargo.toml`
- Modify: `src/server/middleware.rs`
- Modify: `src/server/ws.rs`
- Modify: `src/server/mod.rs`

- [ ] **Step 1: Add `uuid` dependency**

In `Cargo.toml`, add under `[dependencies]`:
```toml
uuid = { version = "1", features = ["v7"] }
```

- [ ] **Step 2: Write failing test for request-id middleware**

In `src/server/middleware.rs`, add test:
```rust
#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn test_request_id_middleware_generates_id() {
        use axum::Router;
        use axum::routing::get;

        let app = Router::new()
            .route("/test", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn(super::request_id_middleware));

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let client = reqwest::Client::new();
        let resp = client
            .get(format!("http://127.0.0.1:{port}/test"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let rid = resp.headers().get("x-request-id").expect("missing X-Request-Id");
        let rid_str = rid.to_str().unwrap();
        assert!(uuid::Uuid::parse_str(rid_str).is_ok(), "X-Request-Id must be valid UUID");
    }

    #[tokio::test]
    async fn test_request_id_middleware_echoes_client_id() {
        use axum::Router;
        use axum::routing::get;

        let app = Router::new()
            .route("/test", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn(super::request_id_middleware));

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let client_id = "my-custom-request-id-123";
        let client = reqwest::Client::new();
        let resp = client
            .get(format!("http://127.0.0.1:{port}/test"))
            .header("X-Request-Id", client_id)
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.headers().get("x-request-id").unwrap().to_str().unwrap(),
            client_id
        );
    }
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test test_request_id_middleware 2>&1 | tail -5`
Expected: FAIL — `request_id_middleware` not defined.

- [ ] **Step 4: Implement `request_id_middleware`**

In `src/server/middleware.rs`:
```rust
use uuid::Uuid;

pub(crate) async fn request_id_middleware(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let request_id = req
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| Uuid::now_v7().to_string());

    let method = req.method().clone();
    let path = req.uri().path().to_string();

    use tracing::Instrument;

    let span = tracing::info_span!(
        "request",
        request_id = %request_id,
        method = %method,
        path = %path,
    );

    let mut response = next.run(req).instrument(span).await;

    if let Ok(v) = axum::http::HeaderValue::from_str(&request_id) {
        response.headers_mut().insert("x-request-id", v);
    }
    response
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test test_request_id_middleware 2>&1 | tail -5`
Expected: PASS.

- [ ] **Step 6: Wire middleware into the app router**

In `src/server/mod.rs`, add request-id layer to the full app (before origin layer, so all routes get it):
```rust
let request_id_layer = axum::middleware::from_fn(middleware::request_id_middleware);

let app = Router::new()
    .route("/health", get(http::health))
    .route("/ready", get(http::readiness))
    .merge(protected)
    .layer(DefaultBodyLimit::max(config.limits.body_limit_bytes))
    .layer(origin_layer)
    .layer(request_id_layer)
    .with_state(state);
```

- [ ] **Step 7: Add request-id span to WebSocket sessions**

In `src/server/ws.rs`, update `ws_handler` to extract the request-id from request extensions or header and create a session span:
```rust
pub(super) async fn ws_handler(
    ws: WebSocketUpgrade,
    axum::extract::ConnectInfo(peer): axum::extract::ConnectInfo<SocketAddr>,
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
) -> Response {
    let request_id = headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown")
        .to_string();

    if state.shutdown.is_cancelled() {
        // ... existing shutdown check ...
    }
    let max_bytes = state.limits.ws_frame_max_bytes;
    let state_cloned = state.clone();
    ws.max_message_size(max_bytes)
        .max_frame_size(max_bytes)
        .on_upgrade(move |socket| async move {
            let span = tracing::info_span!("ws_session", request_id = %request_id, peer = %peer);
            state_cloned
                .tracker
                .clone()
                .track_future(
                    handle_ws(socket, peer, state_cloned.clone())
                )
                .instrument(span)
                .await
        })
}
```

Add `use tracing::Instrument;` to the ws.rs imports.

- [ ] **Step 8: Run full test suite**

Run: `cargo test 2>&1 | tail -5`
Expected: all tests pass.

Run: `cargo clippy -- -D warnings -A dead_code`
Expected: no warnings.

- [ ] **Step 9: Commit**

```bash
git add Cargo.toml src/server/middleware.rs src/server/ws.rs src/server/mod.rs
git commit -m "feat: add X-Request-Id middleware and per-request tracing spans"
```

---

### Task 7: SIGHUP config reload

**Files:**
- Modify: `Cargo.toml`
- Modify: `src/server/config.rs`
- Modify: `src/server/http.rs`
- Modify: `src/server/mod.rs`
- Modify: `src/server/ws.rs`
- Modify: `src/main.rs`

- [ ] **Step 1: Add dependencies**

In `Cargo.toml`:
```toml
arc-swap = "1"
toml = "0.8"
```

- [ ] **Step 2: Write failing test for TOML config parsing**

In `src/server/config.rs`, add:
```rust
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct RuntimeLimitsConfig {
    pub idle_timeout_secs: u64,
    pub ws_frame_max_bytes: usize,
    pub body_limit_bytes: usize,
    pub rate_limit_per_minute: u32,
    pub rate_limit_burst: u32,
    pub max_session_secs: u64,
    pub shutdown_drain_secs: u64,
    pub pool_checkout_timeout_secs: u64,
}
```

Add test:
```rust
#[test]
fn test_runtime_limits_from_toml() {
    let toml_str = r#"
        idle_timeout_secs = 600
        rate_limit_per_minute = 120
    "#;
    let cfg: RuntimeLimitsConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(cfg.idle_timeout_secs, 600);
    assert_eq!(cfg.rate_limit_per_minute, 120);
    // Unspecified fields get defaults
    assert_eq!(cfg.max_session_secs, 3600);
}

#[test]
fn test_runtime_limits_config_to_limits() {
    let cfg = RuntimeLimitsConfig::default();
    let limits: RuntimeLimits = cfg.into();
    let defaults = RuntimeLimits::default();
    assert_eq!(limits.idle_timeout_secs, defaults.idle_timeout_secs);
    assert_eq!(limits.max_session_secs, defaults.max_session_secs);
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test test_runtime_limits_from_toml 2>&1 | tail -5`
Expected: FAIL.

- [ ] **Step 4: Implement `RuntimeLimitsConfig` and conversion**

In `src/server/config.rs`:

Add `Default` for `RuntimeLimitsConfig` that mirrors `RuntimeLimits::default()`:
```rust
impl Default for RuntimeLimitsConfig {
    fn default() -> Self {
        let d = RuntimeLimits::default();
        Self {
            idle_timeout_secs: d.idle_timeout_secs,
            ws_frame_max_bytes: d.ws_frame_max_bytes,
            body_limit_bytes: d.body_limit_bytes,
            rate_limit_per_minute: d.rate_limit_per_minute,
            rate_limit_burst: d.rate_limit_burst,
            max_session_secs: d.max_session_secs,
            shutdown_drain_secs: d.shutdown_drain_secs,
            pool_checkout_timeout_secs: d.pool_checkout_timeout_secs,
        }
    }
}

impl From<RuntimeLimitsConfig> for RuntimeLimits {
    fn from(cfg: RuntimeLimitsConfig) -> Self {
        Self {
            idle_timeout_secs: cfg.idle_timeout_secs,
            ws_frame_max_bytes: cfg.ws_frame_max_bytes,
            body_limit_bytes: cfg.body_limit_bytes,
            rate_limit_per_minute: cfg.rate_limit_per_minute,
            rate_limit_burst: cfg.rate_limit_burst,
            max_session_secs: cfg.max_session_secs,
            shutdown_drain_secs: cfg.shutdown_drain_secs,
            pool_checkout_timeout_secs: cfg.pool_checkout_timeout_secs,
        }
    }
}

pub fn load_config_file(path: &std::path::Path) -> anyhow::Result<RuntimeLimits> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config file: {}", path.display()))?;
    let cfg: RuntimeLimitsConfig = toml::from_str(&content)
        .with_context(|| format!("Failed to parse config file: {}", path.display()))?;
    Ok(cfg.into())
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test test_runtime_limits_from_toml test_runtime_limits_config_to_limits 2>&1 | tail -5`
Expected: PASS.

- [ ] **Step 6: Change `AppState.limits` to `ArcSwap`**

In `src/server/http.rs`, update `AppState`:
```rust
use arc_swap::ArcSwap;

pub struct AppState {
    pub engine: Arc<Engine>,
    pub limits: Arc<ArcSwap<RuntimeLimits>>,
    pub metrics_registry: Option<Arc<MetricsRegistry>>,
    pub shutdown: tokio_util::sync::CancellationToken,
    pub tracker: tokio_util::task::TaskTracker,
}
```

- [ ] **Step 7: Update all `state.limits` reads**

Every `&state.limits` becomes `state.limits.load()`. This affects:

In `src/server/http.rs` — handlers `transcribe`, `transcribe_stream`, `health`:
```rust
// Before: &state.limits
// After:  let limits = state.limits.load();
```

In `src/server/ws.rs` — `ws_handler`, `handle_ws`:
```rust
// Before: state.limits.ws_frame_max_bytes
// After:  let limits = state.limits.load(); limits.ws_frame_max_bytes
```

In `src/server/mod.rs` — `run_with_config_listener`, body limit setup.
Note: `DefaultBodyLimit` is set once at startup and cannot be reloaded dynamically — this is acceptable per spec.

- [ ] **Step 8: Update `AppState` construction in `run_with_config_listener`**

```rust
let state = Arc::new(http::AppState {
    engine: Arc::new(engine),
    limits: Arc::new(ArcSwap::from_pointee(config.limits.clone())),
    metrics_registry: metrics_registry.clone(),
    shutdown: shutdown_root.clone(),
    tracker: tracker.clone(),
});
```

- [ ] **Step 9: Add `--config` CLI flag**

In `src/main.rs`, add to `Commands::Serve`:
```rust
/// Path to TOML config file for RuntimeLimits. On SIGHUP, limits are
/// reloaded from this file. CLI args override config file at startup.
#[arg(long)]
config: Option<String>,
```

- [ ] **Step 10: Implement SIGHUP handler**

In `src/server/mod.rs`, add a `config_path: Option<PathBuf>` parameter to `ServerConfig`:
```rust
pub struct ServerConfig {
    // ... existing fields ...
    pub config_path: Option<std::path::PathBuf>,
}
```

In `run_with_config_listener`, after building `state`, spawn the SIGHUP task:
```rust
#[cfg(unix)]
{
    let reload_state = state.clone();
    let reload_path = config.config_path.clone();
    tokio::spawn(async move {
        let mut sig = tokio::signal::unix::signal(
            tokio::signal::unix::SignalKind::hangup()
        ).expect("failed to register SIGHUP handler");
        loop {
            sig.recv().await;
            let Some(ref path) = reload_path else {
                tracing::info!("No config file specified, ignoring SIGHUP");
                continue;
            };
            match config::load_config_file(path) {
                Ok(new_limits) => {
                    let old = reload_state.limits.load();
                    tracing::info!(
                        idle_timeout_secs = %format!("{} → {}", old.idle_timeout_secs, new_limits.idle_timeout_secs),
                        rate_limit_per_minute = %format!("{} → {}", old.rate_limit_per_minute, new_limits.rate_limit_per_minute),
                        "RuntimeLimits reloaded from {}",
                        path.display()
                    );
                    reload_state.limits.store(Arc::new(new_limits));
                }
                Err(e) => {
                    tracing::error!("Failed to reload config on SIGHUP: {e:#}");
                }
            }
        }
    });
}
```

- [ ] **Step 11: Run full test suite**

Run: `cargo test 2>&1 | tail -5`
Expected: all tests pass.

Run: `cargo clippy -- -D warnings -A dead_code`
Expected: no warnings.

- [ ] **Step 12: Commit**

```bash
git add Cargo.toml src/server/config.rs src/server/http.rs src/server/mod.rs src/server/ws.rs src/main.rs
git commit -m "feat: SIGHUP config reload for RuntimeLimits via arc-swap"
```

---

## Phase 3: Quality

### Task 8: Rustdoc coverage

**Files:**
- Modify: `src/lib.rs`
- Modify: `src/inference/mod.rs`
- Modify: `src/server/config.rs`
- Modify: `src/server/http.rs`
- Modify: `src/error.rs`

- [ ] **Step 1: Enhance crate-level doc in `lib.rs`**

The existing `lib.rs` already has a good `//!` block. Expand it with feature flags section:
```rust
//! ## Feature flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `server` | yes | HTTP + WebSocket server (`axum`) |
//! | `diarization` | yes | Speaker diarization (`polyvoice`) |
//! | `coreml` | no | CoreML execution provider (macOS ARM64) |
//! | `cuda` | no | CUDA execution provider (Linux x86_64) |
//! | `ffi` | no | C-ABI FFI for mobile integration |
```

- [ ] **Step 2: Add doc comments to undocumented pub items**

Scan each file for `pub` items missing `///`. Add one-line doc comments. Key items:

In `src/inference/mod.rs`:
- `Pool::new`, `Pool::checkout`, `Pool::close`, `Pool::total`, `Pool::available`
- `PoolGuard::into_owned`, `OwnedReservation::checkin`
- `DecoderState::new`
- `StreamingState` fields
- `FeatureExtractor` methods
- `TranscriptAssembler` methods
- `Engine` methods: `load`, `create_state`, `process_chunk`, `flush_state`, `transcribe_file`

In `src/error.rs`:
- All `GigasttError` variants

- [ ] **Step 3: Verify docs build**

Run: `cargo doc --no-deps 2>&1 | grep -c warning`
Expected: 0 warnings.

- [ ] **Step 4: Commit**

```bash
git add src/lib.rs src/inference/mod.rs src/server/config.rs src/server/http.rs src/error.rs
git commit -m "docs: add rustdoc coverage for all public API items"
```

---

### Task 9: Runnable examples

**Files:**
- Create: `examples/transcribe_file.rs`
- Create: `examples/websocket_client.rs`
- Modify: `Cargo.toml`

- [ ] **Step 1: Add dev-dependency for WebSocket example**

In `Cargo.toml`:
```toml
[dev-dependencies]
# ... existing ...
tokio-tungstenite = "0.26"
```

- [ ] **Step 2: Create `examples/transcribe_file.rs`**

```rust
//! Transcribe an audio file using the gigastt engine directly.
//!
//! Usage: cargo run --example transcribe_file -- path/to/audio.wav

use anyhow::Result;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| {
            eprintln!("Usage: cargo run --example transcribe_file -- <audio-file>");
            std::process::exit(1);
        });

    let model_dir = gigastt::model::default_model_dir();
    println!("Loading model from {model_dir}...");
    let engine = gigastt::inference::Engine::load(&model_dir)?;
    println!("Model loaded (INT8: {})", engine.is_int8());

    println!("Transcribing {path}...");
    let result = engine.transcribe_file(&path)?;
    println!("\nText: {}", result.text);
    println!("Duration: {:.1}s", result.duration);
    for word in &result.words {
        println!("  [{:.2}s] {}", word.start, word.word);
    }

    Ok(())
}
```

- [ ] **Step 3: Create `examples/websocket_client.rs`**

```rust
//! WebSocket client that connects to a running gigastt server and streams audio.
//!
//! Usage:
//!   1. Start server: cargo run -- serve
//!   2. Run client:   cargo run --example websocket_client -- path/to/audio.wav

use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| {
            eprintln!("Usage: cargo run --example websocket_client -- <audio-file>");
            std::process::exit(1);
        });

    let url = std::env::var("GIGASTT_URL")
        .unwrap_or_else(|_| "ws://127.0.0.1:9876/v1/ws".into());

    println!("Connecting to {url}...");
    let (ws, _) = tokio_tungstenite::connect_async(&url).await?;
    let (mut sink, mut stream) = ws.split();

    // Wait for Ready
    if let Some(Ok(msg)) = stream.next().await {
        let text = msg.into_text()?;
        let v: serde_json::Value = serde_json::from_str(&text)?;
        println!("Server ready: model={}, sample_rate={}", v["model"], v["sample_rate"]);
    }

    // Read audio file as raw PCM16 bytes
    let audio_data = std::fs::read(&path)?;
    println!("Sending {} bytes of audio...", audio_data.len());

    // Send in chunks of 32KB
    for chunk in audio_data.chunks(32 * 1024) {
        sink.send(Message::Binary(chunk.to_vec().into())).await?;
    }

    // Send stop
    sink.send(Message::Text(
        serde_json::json!({"type": "stop"}).to_string().into(),
    ))
    .await?;

    // Read responses until Final
    while let Some(Ok(msg)) = stream.next().await {
        if let Ok(text) = msg.into_text() {
            let v: serde_json::Value = serde_json::from_str(&text)?;
            match v["type"].as_str() {
                Some("partial") => println!("  Partial: {}", v["text"]),
                Some("final") => {
                    println!("  Final:   {}", v["text"]);
                    break;
                }
                Some("error") => {
                    eprintln!("  Error: {}", v["message"]);
                    break;
                }
                _ => {}
            }
        }
    }

    sink.close().await?;
    Ok(())
}
```

- [ ] **Step 4: Verify examples compile**

Run: `cargo build --examples 2>&1 | tail -5`
Expected: compiles without errors.

- [ ] **Step 5: Commit**

```bash
git add examples/transcribe_file.rs examples/websocket_client.rs Cargo.toml
git commit -m "docs: add transcribe_file and websocket_client examples"
```

---

### Task 10: Extract `parse_pcm16_with_carry` and add proptest

**Files:**
- Modify: `Cargo.toml`
- Modify: `src/inference/audio.rs`
- Modify: `src/server/ws.rs`

- [ ] **Step 1: Add proptest dev-dependency**

In `Cargo.toml`:
```toml
[dev-dependencies]
# ... existing ...
proptest = "1"
```

- [ ] **Step 2: Write failing test for `parse_pcm16_with_carry`**

In `src/inference/audio.rs`, add test:
```rust
#[test]
fn test_parse_pcm16_basic() {
    let data: &[u8] = &[0x00, 0x40, 0x00, 0xC0]; // two i16 samples: 16384, -16384
    let mut pending: Option<u8> = None;
    let samples = parse_pcm16_with_carry(data, &mut pending);
    assert_eq!(samples.len(), 2);
    assert!(pending.is_none());
    assert!((samples[0] - 0.5).abs() < 0.001);
    assert!((samples[1] + 0.5).abs() < 0.001);
}

#[test]
fn test_parse_pcm16_odd_length_carry() {
    let mut pending: Option<u8> = None;
    // 3 bytes → 1 sample + 1 carry byte
    let samples = parse_pcm16_with_carry(&[0x00, 0x00, 0xFF], &mut pending);
    assert_eq!(samples.len(), 1);
    assert_eq!(pending, Some(0xFF));

    // Next frame: carry + 1 byte → 1 sample
    let samples = parse_pcm16_with_carry(&[0x7F], &mut pending);
    assert_eq!(samples.len(), 1);
    assert!(pending.is_none());
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test test_parse_pcm16 2>&1 | tail -5`
Expected: FAIL — function not defined.

- [ ] **Step 4: Implement `parse_pcm16_with_carry`**

In `src/inference/audio.rs`:
```rust
/// Parse PCM16 LE bytes into f32 samples, carrying a trailing odd byte across calls.
///
/// WebSocket clients may split their audio stream on arbitrary byte boundaries.
/// This function maintains a carry byte across frames so that odd-length payloads
/// don't introduce a 1-sample phase shift in the decoded audio.
pub(crate) fn parse_pcm16_with_carry(data: &[u8], pending: &mut Option<u8>) -> Vec<f32> {
    let carry_prev = pending.take();
    let needs_combine = carry_prev.is_some() || !data.len().is_multiple_of(2);

    if needs_combine {
        let mut combined = Vec::with_capacity(data.len() + 1);
        if let Some(prev) = carry_prev {
            combined.push(prev);
        }
        combined.extend_from_slice(data);
        if !combined.len().is_multiple_of(2) {
            *pending = combined.pop();
        }
        combined
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0)
            .collect()
    } else {
        data.chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0)
            .collect()
    }
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test test_parse_pcm16 2>&1 | tail -5`
Expected: PASS.

- [ ] **Step 6: Update `handle_binary_frame` in `ws.rs` to use the extracted function**

In `src/server/ws.rs`, replace the inline PCM16 parsing in `handle_binary_frame` with:
```rust
let samples_f32 = crate::inference::audio::parse_pcm16_with_carry(&data, pending_byte);
```

Remove the now-duplicated inline parsing logic (the `let carry_prev = pending_byte.take(); ...` block).
Keep the `tracing::warn!` for odd-length streams — move it into `parse_pcm16_with_carry` or add a separate check:
```rust
if pending_byte.is_some() {
    tracing::warn!(
        "Odd-length PCM stream from {peer}: {} bytes, deferring 1 byte",
        data.len()
    );
}
```

- [ ] **Step 7: Add proptest tests**

In `src/inference/audio.rs`:
```rust
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn proptest_pcm16_carry_invariant(
            chunks in proptest::collection::vec(
                proptest::collection::vec(any::<u8>(), 0..1000),
                1..20
            )
        ) {
            let mut pending: Option<u8> = None;
            let mut total_samples = 0usize;
            let mut total_bytes = 0usize;

            for chunk in &chunks {
                total_bytes += chunk.len();
                let samples = parse_pcm16_with_carry(chunk, &mut pending);
                total_samples += samples.len();
            }

            // Invariant: total decoded samples == total bytes / 2 (integer division)
            let expected = total_bytes / 2;
            prop_assert_eq!(total_samples, expected,
                "samples ({total_samples}) must equal total_bytes/2 ({expected})");

            // At most 1 pending byte remains
            if total_bytes % 2 == 1 {
                prop_assert!(pending.is_some());
            } else {
                prop_assert!(pending.is_none());
            }
        }

        #[test]
        fn proptest_resample_no_panic(
            samples in proptest::collection::vec(-1.0f32..1.0f32, 0..10_000),
            rate_idx in 0..5usize,
        ) {
            let rates = [8000u32, 16000, 24000, 44100, 48000];
            let from_rate = SampleRate(rates[rate_idx]);
            if from_rate.0 == 16000 {
                // No-op resample, skip
                return Ok(());
            }
            let result = resample(&samples, from_rate, SampleRate(16000));
            prop_assert!(result.is_ok() || samples.is_empty());
        }
    }
}
```

- [ ] **Step 8: Run proptest**

Run: `cargo test proptests 2>&1 | tail -10`
Expected: PASS (proptest runs default 256 cases).

- [ ] **Step 9: Run full test suite**

Run: `cargo test 2>&1 | tail -5`
Expected: all tests pass.

- [ ] **Step 10: Commit**

```bash
git add Cargo.toml src/inference/audio.rs src/server/ws.rs
git commit -m "test: extract parse_pcm16_with_carry and add proptest coverage"
```

---

### Task 11: WER benchmark threshold in CI

**Files:**
- Create: `.github/wer-threshold.txt`
- Modify: `.github/workflows/ci.yml`
- Modify: `tests/benchmark.rs` (if output format needs adjustment)

- [ ] **Step 1: Verify benchmark output format**

Run: `grep -n 'WER:' tests/benchmark.rs`
Expected: line matching `"  WER: {:.1}% ..."` — the benchmark already prints `WER: <float>%`.

The existing `MAX_WER` constant in `tests/benchmark.rs` is `12.0` and the test already asserts `wer < MAX_WER`. The CI threshold file provides an additional external gate.

- [ ] **Step 2: Create threshold file**

Create `.github/wer-threshold.txt`:
```
12.0
```

Use `12.0` to match the existing `MAX_WER` in `tests/benchmark.rs`.

- [ ] **Step 3: Add benchmark job to CI**

In `.github/workflows/ci.yml`, add after the `e2e-tests` job:

```yaml
  benchmark:
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    needs: [e2e-tests]
    runs-on: ubuntu-latest
    name: WER Benchmark
    steps:
      - uses: actions/checkout@v6
      - uses: dtolnay/rust-toolchain@stable
      - uses: arduino/setup-protoc@v3
        with:
          version: '29.x'
          repo-token: ${{ secrets.GITHUB_TOKEN }}
      - uses: actions/cache@v5
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}

      - name: Cache model
        id: cache-model
        uses: actions/cache@v5
        with:
          path: ~/.gigastt/models
          key: gigastt-model-v3-onnx-${{ hashFiles('src/model/mod.rs') }}

      - name: Download model
        if: steps.cache-model.outputs.cache-hit != 'true'
        run: cargo run -- download

      - name: Run WER benchmark
        run: cargo test --test benchmark -- --ignored 2>&1 | tee benchmark_output.txt

      - name: Check WER threshold
        run: |
          THRESHOLD=$(cat .github/wer-threshold.txt | tr -d '[:space:]')
          WER=$(grep -oP 'WER: \K[0-9.]+' benchmark_output.txt | head -1)
          if [ -z "$WER" ]; then
            echo "ERROR: Could not extract WER from benchmark output"
            exit 1
          fi
          echo "WER: ${WER}%, Threshold: ${THRESHOLD}%"
          if awk "BEGIN {exit !($WER > $THRESHOLD)}"; then
            echo "FAIL: WER ${WER}% exceeds threshold ${THRESHOLD}%"
            exit 1
          fi
          echo "PASS: WER ${WER}% is within threshold ${THRESHOLD}%"

      - name: Upload benchmark results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: benchmark_output.txt
```

- [ ] **Step 4: Verify CI syntax**

Run: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" 2>&1`
Expected: no errors (valid YAML).

- [ ] **Step 5: Commit**

```bash
git add .github/wer-threshold.txt .github/workflows/ci.yml
git commit -m "ci: add WER benchmark with threshold gate on main push"
```

---

## Verification

### Task 12: Final verification

- [ ] **Step 1: Run full test suite**

Run: `cargo test 2>&1 | tail -10`
Expected: all tests pass (original 142 + new readiness tests + proptest + request-id tests).

- [ ] **Step 2: Run clippy**

Run: `cargo clippy -- -D warnings -A dead_code 2>&1 | tail -5`
Expected: no warnings.

- [ ] **Step 3: Check formatting**

Run: `cargo fmt --check 2>&1`
Expected: no formatting issues.

- [ ] **Step 4: Verify doc build**

Run: `cargo doc --no-deps 2>&1 | grep -c warning`
Expected: 0 warnings.

- [ ] **Step 5: Verify file sizes after decomposition**

Run: `wc -l src/server/*.rs`
Expected: all files under 750 lines.

- [ ] **Step 6: Verify no `/ws` references remain**

Run: `grep -rn '"/ws"' src/ tests/ 2>/dev/null`
Expected: no matches (only `/v1/ws` references).
