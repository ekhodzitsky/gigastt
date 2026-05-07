# Quality Improvements: +18 Points Design Spec

**Date:** 2026-05-07
**Scope:** 8 improvements across 3 phases, targeting code quality, observability, documentation, and testing.
**Excluded:** FFI workspace split (user decision).

## Implementation Strategy

3 phases grouped by dependency order. Each phase is a self-contained PR.

## Phase 1: Refactoring

### 1.1 Remove Deprecated `/ws` Path

Remove the legacy WebSocket alias at `/ws`. The canonical `/v1/ws` path has been available since v0.7.0; deprecation headers (`Deprecation: true`, `Link: </v1/ws>; rel="successor-version"`) have been served on every upgrade since then. Removal was planned for v1.0; the project is now at v1.0.1.

**Changes:**
- Delete `ws_handler_legacy()` function (~40 lines in `server/mod.rs`)
- Remove `.route("/ws", get(ws_handler_legacy))` from the protected router
- Update startup log: remove `"(legacy alias: ws://{addr}/ws)"` from the info line
- Update `CLAUDE.md`: remove references to deprecated `/ws` path
- Update `docs/asyncapi.yaml` and `docs/openapi.yaml` if they reference `/ws`
- E2E tests: remove any tests hitting `/ws`, verify all use `/v1/ws`

### 1.2 Decompose `server/mod.rs` (1608 → 4 files)

Split `server/mod.rs` into focused modules, each under 400 lines with a single responsibility.

**`server/config.rs`** (~160 lines):
- `OriginPolicy`, `OriginVerdict` (enum), `is_loopback_origin()`
- `OriginPolicy::evaluate()`, `OriginPolicy::loopback_only()`
- `RuntimeLimits`, `Default for RuntimeLimits`
- `ServerConfig`, `ServerConfig::local()`
- `SUPPORTED_RATES`, `DEFAULT_SAMPLE_RATE` constants
- `pool_retry_after_ms()`, `pool_retry_after_secs()` helpers
- Tests: `test_runtime_limits_*`, `test_origin_*`, `test_server_config_*`

**`server/middleware.rs`** (~100 lines):
- `origin_middleware()` — CORS / cross-origin deny
- `http_metrics_middleware()` — request duration + counter recording
- Re-uses types from `config.rs` via `super::config::{OriginPolicy, OriginVerdict}`

**`server/ws.rs`** (~700 lines):
- `FrameOutcome` enum, `WsSink` type alias
- `ws_handler()` — WebSocket upgrade handler (canonical `/v1/ws`)
- `handle_ws()` — pool checkout + metrics guard + delegation to `handle_ws_inner`
- `handle_ws_inner()` — main session loop (Ready, select!, frame dispatch)
- `handle_binary_frame()` — PCM16 processing + inference via `spawn_blocking`
- `handle_configure_message()` — sample rate / diarization / protocol version
- `handle_stop_message()` — flush + Final
- `flush_and_final()` — shared flush helper for cancel/cap paths
- `send_server_message()` — sink write helper
- `json_text()` — JSON serialization with fallback (shared with `http.rs` — keep in `mod.rs` as `pub(crate)`)

**`server/mod.rs`** (~250 lines):
- `pub mod config, http, metrics, middleware, rate_limit, ws;`
- Re-exports: `pub use config::{OriginPolicy, RuntimeLimits, ServerConfig};`
- `pub(crate) fn json_text()` — shared helper
- `run()`, `run_with_shutdown()`, `run_with_config()`, `run_with_config_listener()`
- Router assembly (protected + public routes, layers)
- Rate limiter setup + background eviction task
- Startup logs, graceful shutdown drain logic

**Visibility rules:**
- `pub` — only types in the crate's public API (`ServerConfig`, `OriginPolicy`, `RuntimeLimits`, `run*`)
- `pub(crate)` — inter-module types (`AppState`, `json_text`, `pool_retry_after_*`)
- Private — everything else

## Phase 2: Functionality

### 2.1 Health Liveness + Readiness Endpoints

**`GET /health`** (liveness, unchanged):
- Returns `{"status":"ok","model":"gigaam-v3-e2e-rnnt","version":"..."}`.
- Purpose: k8s `livenessProbe`, Docker HEALTHCHECK. Proves the process is alive.
- Exempt from rate limiting (as today).

**`GET /ready`** (new readiness endpoint):
- Checks: `pool.available()`, `shutdown.is_cancelled()`.
- Success (200): `{"status":"ready","pool_available":N,"pool_total":M}`.
- Not ready (503): `{"status":"not_ready","reason":"shutting_down"|"pool_exhausted"}`.
- Purpose: k8s `readinessProbe`. Indicates the server can accept new work.
- Exempt from rate limiting and origin policy (same as `/health`).
- Added to the public router (outside `protected`), alongside `/health`.

### 2.2 Request-ID Spans

**Dependency:** `uuid` crate (v1, feature `v7`) added to `[dependencies]`.

**Middleware** (in `server/middleware.rs`):
- Runs on all routes (protected + public).
- Checks for incoming `X-Request-Id` header; if absent or invalid, generates UUID v7.
- Creates `tracing::info_span!("request", request_id = %id, method = %method, path = %path)`.
- Attaches span to the request via `tracing::Instrument`.
- Inserts `X-Request-Id` header into the response.

**WebSocket sessions:**
- `ws_handler` creates a child span `tracing::info_span!("ws_session", request_id = %id, peer = %peer)`.
- Span lives for the entire WebSocket session lifetime.
- All logs within `handle_ws_inner` automatically inherit the span fields.

**Effect:** Every log line from a request/session includes `request_id`, enabling log correlation across concurrent sessions.

### 2.3 SIGHUP Config Reload

**Dependency:** `arc-swap` crate added to `[dependencies]`.

**Config file format:**
- Optional `--config <path>` CLI flag. TOML format.
- Contains only `RuntimeLimits` fields (flat key-value):
  ```toml
  idle_timeout_secs = 300
  ws_frame_max_bytes = 524288
  body_limit_bytes = 52428800
  rate_limit_per_minute = 0
  rate_limit_burst = 10
  max_session_secs = 3600
  shutdown_drain_secs = 10
  pool_checkout_timeout_secs = 30
  ```
- At startup: defaults ← config file ← CLI explicit args → initial `RuntimeLimits`.
- On SIGHUP: config file is re-read and applied directly. CLI overrides are NOT re-applied — they were a one-time startup override. If the user wants values to persist across reloads, they belong in the config file.

**Runtime storage:**
- `AppState.limits` changes from `RuntimeLimits` to `arc_swap::ArcSwap<RuntimeLimits>`.
- All handlers read via `state.limits.load()` — returns `arc_swap::Guard<Arc<RuntimeLimits>>` (lock-free, zero-copy).
- Write path: only SIGHUP handler calls `state.limits.store(Arc::new(new_limits))`.

**Signal handler:**
- `#[cfg(unix)]` — registers `SIGHUP` via `tokio::signal::unix::signal()`.
- On signal: re-reads config file, stores into `ArcSwap`.
- Logs changed fields at `info` level (old value → new value for each changed field).
- Without `--config`: SIGHUP logs `"No config file specified, ignoring SIGHUP"` at `info`.
- Rate limiter: if `rate_limit_per_minute` or `rate_limit_burst` changed, creates a new `RateLimiter` and swaps it (also via `ArcSwap`).

**Not reloadable** (by design): `port`, `host`, `pool_size`, `metrics_enabled`, `trust_proxy`, `origin_policy`. These require listener/pool recreation.

## Phase 3: Quality

### 3.1 Rustdoc + Examples

**Crate documentation (`lib.rs`):**
- `//!` block: project overview, feature flags, quick start, link to repo.
- Each `pub mod` gets a one-line `///` if missing.

**Public API doc coverage:**
- `///` on all `pub` items without doc comments across: `inference/mod.rs` (42 pub items), `server/mod.rs` + submodules (9), `server/http.rs` (9), `error.rs` (7).
- Key types (`Engine`, `ServerConfig`, `Pool`, `GigasttError`) get `# Examples` sections.

**`examples/` directory:**
- `examples/transcribe_file.rs` — loads engine, transcribes a WAV file, prints text. Demonstrates `Engine::new()` → `Engine::transcribe_file()`.
- `examples/websocket_client.rs` — connects to running server via `tokio-tungstenite`, sends PCM16 audio, prints partials/finals. Demonstrates the WS protocol.
- `[dev-dependencies]`: `tokio-tungstenite` (already transitively available via axum test utils, but made explicit).

### 3.2 Property-Based Tests (proptest)

**Dependency:** `proptest` in `[dev-dependencies]`.

**Test module in `inference/audio.rs`:**
- `proptest_resample_no_panic`: arbitrary `Vec<f32>` (length 0..50_000), sample rate from `SUPPORTED_RATES`, target 16kHz. Assert: no panic, output length > 0 for non-empty input.
- `proptest_resample_length_ratio`: for known rates, verify output length is within 1 sample of `input_len * 16000 / source_rate`.
- `proptest_buffer_frame_carry`: arbitrary `Vec<u8>` (length 0..10_000, simulating PCM16 data), verify carry byte is always 0 or 1, and `consumed + carry == input.len()`.

**PCM16 carry-byte extraction:**
- The carry-byte logic is currently inline in `handle_binary_frame()` (async, tightly coupled to WsSink/Engine). Extract a pure function `fn parse_pcm16_with_carry(data: &[u8], pending: &mut Option<u8>) -> Vec<f32>` in `inference/audio.rs`. This makes the logic testable independently.

**Test module in `inference/audio.rs` (additional):**
- `proptest_pcm16_carry_byte_invariant`: sequence of arbitrary-length byte slices simulating a stream of WS binary frames. After processing all frames via `parse_pcm16_with_carry`, verify: total samples decoded == total bytes / 2 (integer division), pending byte ∈ {None, Some(_)}.

### 3.3 WER Benchmark in CI

**Threshold file:** `.github/wer-threshold.txt` — single line: `20.0` (max acceptable WER %).

**CI integration** (`.github/workflows/ci.yml`, main push only):
- New job `benchmark` after `e2e` job, depends on cached model.
- Runs: `cargo test --test benchmark -- --ignored 2>&1 | tee benchmark_output.txt`.
- Parse step: extract WER value from output, compare against threshold.
- Fail CI if WER > threshold.
- Upload `benchmark_output.txt` as artifact for trend visibility.

**Benchmark output contract:** `tests/benchmark.rs` must print a line matching `WER: <float>%` for the CI parser to extract. If the current benchmark doesn't, add it.
