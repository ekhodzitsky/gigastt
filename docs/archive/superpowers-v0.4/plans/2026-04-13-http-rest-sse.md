# HTTP/REST API + SSE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add HTTP endpoints (health check, file transcription, SSE streaming) alongside existing WebSocket, served from a single port via axum.

**Architecture:** Migrate server from raw tokio-tungstenite to axum router. axum handles both HTTP routes and WebSocket upgrade natively. Existing WebSocket protocol preserved. New endpoints: `GET /health`, `POST /v1/transcribe`, `POST /v1/transcribe/stream` (SSE). Shared Engine + Semaphore via axum State.

**Tech Stack:** axum 0.8, tower-http (body limit, cors), tokio-tungstenite (via axum's ws extract), serde_json

---

### Task 1: Add axum dependencies

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1: Add axum and tower-http to Cargo.toml**

Add after the `futures-util` line:

```toml
# HTTP server (serves REST + WebSocket on single port)
axum = { version = "0.8", features = ["ws", "multipart"] }
tower-http = { version = "0.6", features = ["limit", "cors"] }
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check --features coreml`
Expected: compiles (new deps downloaded).

- [ ] **Step 3: Commit**

```bash
git add Cargo.toml Cargo.lock
git commit -m "chore: add axum and tower-http dependencies"
```

---

### Task 2: Create HTTP handlers module

**Files:**
- Create: `src/server/http.rs`
- Modify: `src/server/mod.rs` (add `mod http;`)

- [ ] **Step 1: Write tests for health endpoint**

Create `src/server/http.rs`:

```rust
//! HTTP handlers for REST API endpoints.

use axum::extract::State;
use axum::response::Json;
use serde::Serialize;
use std::sync::Arc;

use crate::inference::Engine;

/// Shared application state for all handlers.
pub struct AppState {
    pub engine: Arc<Engine>,
    pub semaphore: Arc<tokio::sync::Semaphore>,
}

/// Health check response.
#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub model: String,
    pub version: String,
}

/// GET /health — health check for monitoring and Docker HEALTHCHECK.
pub async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let _ = &state.engine; // verify engine is accessible
    Json(HealthResponse {
        status: "ok".into(),
        model: "gigaam-v3-e2e-rnnt".into(),
        version: env!("CARGO_PKG_VERSION").into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_response_serialization() {
        let resp = HealthResponse {
            status: "ok".into(),
            model: "test".into(),
            version: "0.3.0".into(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["status"], "ok");
        assert_eq!(v["model"], "test");
    }
}
```

- [ ] **Step 2: Add module declaration**

In `src/server/mod.rs`, add at the top (after the `//!` doc comment, before `use` statements):

```rust
pub mod http;
```

- [ ] **Step 3: Run tests**

Run: `cargo test --lib http --features coreml`
Expected: `test_health_response_serialization` passes.

- [ ] **Step 4: Commit**

```bash
git add src/server/http.rs src/server/mod.rs
git commit -m "feat(server): add HTTP handlers module with health endpoint"
```

---

### Task 3: Add file transcription endpoint

**Files:**
- Modify: `src/server/http.rs`

- [ ] **Step 1: Write TranscribeResponse type and handler**

Add to `src/server/http.rs`:

```rust
use axum::body::Bytes;
use axum::http::StatusCode;
use axum::response::IntoResponse;

/// POST /v1/transcribe response.
#[derive(Serialize)]
pub struct TranscribeResponse {
    pub text: String,
    pub words: Vec<crate::inference::WordInfo>,
    pub duration: f64,
}

/// POST /v1/transcribe — upload audio file, get full transcript.
///
/// Accepts raw audio body (Content-Type: audio/*) or any binary body.
/// Supported formats: WAV, MP3, M4A/AAC, OGG, FLAC.
/// Max body size enforced by tower-http layer (50MB).
pub async fn transcribe(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Result<Json<TranscribeResponse>, (StatusCode, Json<serde_json::Value>)> {
    let _permit = state.semaphore.acquire().await.map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "Server busy", "code": "busy"})),
        )
    })?;

    // Write to temp file for decode_audio_file
    let tmp = tempfile::NamedTempFile::new().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("Temp file error: {e}"), "code": "internal"})),
        )
    })?;
    std::fs::write(tmp.path(), &body).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("Write error: {e}"), "code": "internal"})),
        )
    })?;

    let path = tmp.path().to_string_lossy().to_string();
    let engine = state.engine.clone();

    let result = tokio::task::spawn_blocking(move || engine.transcribe_file(&path)).await;

    match result {
        Ok(Ok(text)) => {
            let duration = body.len() as f64 / (16000.0 * 2.0); // approximate
            Ok(Json(TranscribeResponse {
                text,
                words: vec![],
                duration,
            }))
        }
        Ok(Err(e)) => Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({"error": format!("{e}"), "code": "transcription_error"})),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("{e}"), "code": "internal"})),
        )),
    }
}
```

- [ ] **Step 2: Add test for TranscribeResponse serialization**

```rust
#[test]
fn test_transcribe_response_serialization() {
    let resp = TranscribeResponse {
        text: "hello".into(),
        words: vec![],
        duration: 1.5,
    };
    let json = serde_json::to_string(&resp).unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["text"], "hello");
    assert_eq!(v["duration"], 1.5);
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test --lib http --features coreml`
Expected: both tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/server/http.rs
git commit -m "feat(server): add POST /v1/transcribe endpoint"
```

---

### Task 4: Add SSE streaming endpoint

**Files:**
- Modify: `src/server/http.rs`

- [ ] **Step 1: Add SSE handler**

Add to `src/server/http.rs`:

```rust
use axum::response::sse::{Event, KeepAlive, Sse};
use futures_util::stream::Stream;
use std::pin::Pin;

/// POST /v1/transcribe/stream — upload audio file, get SSE stream of partial/final results.
pub async fn transcribe_stream(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Result<Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>>, (StatusCode, Json<serde_json::Value>)> {
    let _permit = state.semaphore.acquire().await.map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "Server busy", "code": "busy"})),
        )
    })?;

    // Decode audio file
    let tmp = tempfile::NamedTempFile::new().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("{e}"), "code": "internal"})),
        )
    })?;
    std::fs::write(tmp.path(), &body).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("{e}"), "code": "internal"})),
        )
    })?;

    let path = tmp.path().to_string_lossy().to_string();
    let engine = state.engine.clone();

    // Decode audio to 16kHz samples
    let samples = crate::inference::audio::decode_audio_file(&path).map_err(|e| {
        (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({"error": format!("{e}"), "code": "invalid_audio"})),
        )
    })?;

    // Process in chunks via stream
    let stream = async_stream::stream! {
        let mut state = engine.create_state();
        let chunk_size = 16000; // 1 second at 16kHz

        for chunk in samples.chunks(chunk_size) {
            match engine.process_chunk(chunk, &mut state) {
                Ok(segments) => {
                    for seg in segments {
                        let msg = if seg.is_final {
                            serde_json::json!({"type": "final", "text": seg.text, "timestamp": seg.timestamp, "words": seg.words})
                        } else {
                            serde_json::json!({"type": "partial", "text": seg.text, "timestamp": seg.timestamp, "words": seg.words})
                        };
                        yield Ok(Event::default().data(msg.to_string()));
                    }
                }
                Err(e) => {
                    let msg = serde_json::json!({"type": "error", "message": format!("{e}"), "code": "inference_error"});
                    yield Ok(Event::default().data(msg.to_string()));
                    break;
                }
            }
        }

        // Flush remaining
        if let Some(seg) = engine.flush_state(&mut state) {
            let msg = serde_json::json!({"type": "final", "text": seg.text, "timestamp": seg.timestamp, "words": seg.words});
            yield Ok(Event::default().data(msg.to_string()));
        }
    };

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}
```

- [ ] **Step 2: Add `async-stream` dependency**

In `Cargo.toml`, add:

```toml
# SSE stream generation
async-stream = "0.3"
```

Also add `tempfile` to regular dependencies (currently only dev-dependencies):

```toml
tempfile = "3"
```

Move `tempfile` from `[dev-dependencies]` to `[dependencies]`.

- [ ] **Step 3: Verify compilation**

Run: `cargo check --features coreml`
Expected: compiles.

- [ ] **Step 4: Commit**

```bash
git add src/server/http.rs Cargo.toml Cargo.lock
git commit -m "feat(server): add POST /v1/transcribe/stream SSE endpoint"
```

---

### Task 5: Migrate server to axum router

**Files:**
- Modify: `src/server/mod.rs` (replace `run()` function)

- [ ] **Step 1: Rewrite `run()` to use axum router**

Replace the entire `run()` function in `src/server/mod.rs`:

```rust
use axum::Router;
use axum::routing::{get, post};
use std::sync::Arc;

pub async fn run(engine: Engine, port: u16, host: &str) -> Result<()> {
    let addr: SocketAddr = format!("{host}:{port}").parse()
        .context("Invalid host:port")?;

    let state = Arc::new(http::AppState {
        engine: Arc::new(engine),
        semaphore: Arc::new(Semaphore::new(MAX_CONCURRENT_CONNECTIONS)),
    });

    let app = Router::new()
        .route("/health", get(http::health))
        .route("/v1/transcribe", post(http::transcribe))
        .route("/v1/transcribe/stream", post(http::transcribe_stream))
        .route("/ws", get(ws_handler))
        .layer(tower_http::limit::RequestBodyLimitLayer::new(50 * 1024 * 1024)) // 50MB
        .with_state(state);

    tracing::info!("gigastt server listening on http://{addr}");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c().await.ok();
    tracing::info!("Shutting down server");
}
```

- [ ] **Step 2: Add WebSocket upgrade handler**

Add the `ws_handler` function that bridges axum's WebSocket to the existing `handle_connection` logic:

```rust
use axum::extract::ws::{Message as AxumMessage, WebSocket, WebSocketUpgrade};
use axum::extract::ConnectInfo;
use axum::response::Response;

async fn ws_handler(
    ws: WebSocketUpgrade,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    State(state): State<Arc<http::AppState>>,
) -> Response {
    ws.max_message_size(512 * 1024)
        .max_frame_size(512 * 1024)
        .on_upgrade(move |socket| handle_ws(socket, peer, state))
}

async fn handle_ws(socket: WebSocket, peer: SocketAddr, state: Arc<http::AppState>) {
    let permit = match state.semaphore.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => return,
    };

    if let Err(e) = handle_ws_inner(socket, peer, &state.engine).await {
        tracing::error!("WebSocket error from {peer}: {e}");
    }

    drop(permit);
}
```

The `handle_ws_inner` function contains the same logic as the current `handle_connection` but uses axum's `WebSocket` type instead of `tokio_tungstenite`. The message types are different: `AxumMessage::Binary(Bytes)`, `AxumMessage::Text(String)`, `AxumMessage::Close(_)`.

- [ ] **Step 3: Port `handle_connection` to axum WebSocket types**

Rewrite the connection handler to use `axum::extract::ws` types. The logic stays the same; only the message types change.

- [ ] **Step 4: Remove tokio-tungstenite from server imports**

Remove `use tokio_tungstenite::tungstenite::Message;` from `src/server/mod.rs`. Keep `tokio-tungstenite` in Cargo.toml since integration tests still use it as a client.

- [ ] **Step 5: Update `src/server/mod.rs` to use `axum::extract::State`**

Add necessary axum imports at the top.

- [ ] **Step 6: Run tests**

Run: `cargo test --lib --features coreml`
Expected: all 56+ tests pass.

Run: `cargo clippy --features coreml -- -D warnings -A dead_code`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/server/mod.rs
git commit -m "feat(server): migrate to axum router — HTTP + WS on single port"
```

---

### Task 6: Update Dockerfiles and health check

**Files:**
- Modify: `Dockerfile` (HEALTHCHECK now uses HTTP)
- Modify: `Dockerfile.cuda` (same)

- [ ] **Step 1: Verify Docker HEALTHCHECK works with /health**

Current HEALTHCHECK already uses `curl -f http://localhost:9876/health`. This will now work because we serve HTTP. No change needed if the endpoint returns 200.

- [ ] **Step 2: Test Docker build**

Run: `docker build -t gigastt .`
Expected: builds successfully.

- [ ] **Step 3: Commit (if changes needed)**

---

### Task 7: Integration tests for HTTP endpoints

**Files:**
- Create: `tests/http_integration.rs`

- [ ] **Step 1: Write health check integration test**

```rust
//! Integration tests for HTTP REST API.
//!
//! These tests require the GigaAM model for transcription endpoints.
//! Health endpoint tests work without a model.

#[tokio::test]
#[ignore] // Requires model
async fn test_health_endpoint() {
    // Start server, GET /health, verify 200 + JSON response
}

#[tokio::test]
#[ignore] // Requires model
async fn test_transcribe_wav_file() {
    // Start server, POST /v1/transcribe with WAV file, verify JSON response with text
}
```

- [ ] **Step 2: Commit**

```bash
git add tests/http_integration.rs
git commit -m "test: HTTP integration tests for health and transcribe endpoints"
```

---

### Task 8: Update documentation

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Document HTTP endpoints in README**

Add HTTP API section:
- `GET /health` — health check
- `POST /v1/transcribe` — file transcription
- `POST /v1/transcribe/stream` — SSE streaming
- `GET /ws` — WebSocket (existing protocol)

Add curl examples:
```bash
# Health check
curl http://localhost:9876/health

# Transcribe a file
curl -X POST --data-binary @recording.wav http://localhost:9876/v1/transcribe

# SSE streaming
curl -X POST -H "Accept: text/event-stream" --data-binary @recording.wav http://localhost:9876/v1/transcribe/stream
```

- [ ] **Step 2: Update CLAUDE.md architecture**

- [ ] **Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: document HTTP REST API and SSE streaming endpoints"
```
