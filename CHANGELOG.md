# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.2] - 2026-04-21

### Fixed

- **CI: minisign signing step accepts password non-interactively** (`.github/workflows/release.yml`). v0.9.1 release job got through the build + SBOM + provenance steps but failed at `Sign tarballs + SHA256SUMS with minisign` because `rsign2 sign -W` interprets `-W` as "write signature" (not "password"), so the process still prompted for a passphrase and rejected the key with `Wrong password for that key`. Switched to the apt-installed `minisign` binary, which reads the passphrase on stdin when stdout is non-TTY — a well-supported CI pattern.

## [0.9.1] - 2026-04-21

### Fixed

- **CI: install `protoc` on every cargo-build job** (`.github/workflows/ci.yml`, `.github/workflows/release.yml`, `.github/workflows/soak.yml`). v0.9.0 rollout failed in the release workflow because `prost-build` shells out to `protoc` and the GitHub-hosted `macos-14` + `ubuntu-latest` runners don't carry it. Every cargo-build-facing job now runs `arduino/setup-protoc@v3` right after `rust-toolchain`. No source change — the v0.9.0 binaries would have been bit-identical if the CI had succeeded; v0.9.1 is purely a rebuild.

## [0.9.0] - 2026-04-21

_Stable release promoting `0.9.0-rc.2` + the follow-up supply-chain
lockdown (vendored ONNX protobuf, in-tree token-bucket rate limiter,
in-tree Prometheus encoder, CycloneDX SBOM, SLSA provenance, minisign
release signing). See the [Unreleased] rows moved in below for the
full rollup; no functional regressions since rc.2._


### Added

- **Vendored ONNX protobuf schema + native codegen** (V1-10 follow-up, `proto/onnx.proto`, `build.rs`, `src/onnx_proto.rs`). `proto/onnx.proto` is copied verbatim from github.com/onnx/onnx (MIT-licensed, 1 000 LoC) and regenerated on every build by `prost-build 0.13` — replacing the unmaintained `onnx-pb 0.1.4` crate (last published 2020, transitively pinned to `prost 0.6`). Requires `protoc` in `PATH` at build time (`brew install protobuf`, `apt install protobuf-compiler`); see `build.rs` for the friendly failure message. Closes `RUSTSEC-2021-0073`: the advisory targeted `prost-types 0.6`'s `From<Timestamp> for SystemTime` path, which no longer ships in our dependency graph — the ignore block is gone from `deny.toml` and `.cargo/audit.toml`.
- **Custom Prometheus text encoder** (`src/server/metrics.rs`, replaces `metrics-exporter-prometheus`). ~280-line `MetricsRegistry` that serialises counters + histograms in Prometheus 0.0.4 exposition format. We only expose two metrics (`gigastt_http_requests_total`, `gigastt_http_request_duration_seconds`), so a full Recorder-trait registry was overkill — the new `HttpMetricsMiddleware` calls `registry.counter_inc(...)` / `registry.histogram_record(...)` directly, no global recorder, no `metrics` / `metrics-util` / `indexmap` / `quanta` transitives. 125 lib tests pass (+ 7 new metrics tests covering counter increment, histogram bucket cumulativity, label ordering, label escaping, empty labels, sum tracking, and empty-registry rendering).
- **Nightly soak + load CI** (V1-09). New `.github/workflows/soak.yml` runs `cargo test --test soak_test -- --ignored` at 03:17 UTC daily (plus `workflow_dispatch` for on-demand checks), reusing the main-CI model cache so regressions in pool drift / descriptor leaks / RSS growth surface outside the fast-feedback envelope.
- **`docs/deployment.md`: rate-limiter & X-Forwarded-For section** (V1-11). The published nginx recipe used `$proxy_add_x_forwarded_for`, which appends client-supplied headers, and the Caddy recipe did not forward the real peer at all — operators who turned on `--rate-limit-per-minute` were running a defence attackers could trivially bypass. Both recipes now overwrite the header with `$remote_addr` / `{remote_host}`, and a new section explains why it's not optional.

### Changed

- **Custom per-IP token-bucket rate limiter (drops `tower_governor`)** (`src/server/rate_limit.rs`, `src/server/mod.rs`, `Cargo.toml`). Replaced the `tower_governor = "0.7"` dependency with a focused ~150-line implementation tailored to gigastt's single middleware hook. Drops `tower_governor`, `governor`, `forwarded-header-value`, and `nonzero_ext` from `Cargo.lock`; `dashmap` is promoted from transitive to direct so the lock-free shard map stays explicit. Refill math preserves the V1-06 formula (`refill_per_ms = rpm / 60_000`) — covered by both the existing `test_rate_limit_interval_formula` and the new `test_rate_limiter_refill_formula_matches_v1_06`. IP extraction still honours the V1-11 trust boundary (first hop of `X-Forwarded-For`, then `X-Real-IP`, then `ConnectInfo`). Memory is bounded by a `tokio::spawn` eviction task that tracks `shutdown_root`, not the old `std::thread::spawn` GC thread that leaked on shutdown.
- **`Engine::create_state` accepts `diarization_enabled` unconditionally** (V1-08, `src/inference/mod.rs`). The parameter used to be gated behind `#[cfg(feature = "diarization")]`, so the same public API mutated between feature builds — `src/lib.rs`'s doctest compiled only with `--features diarization`, and external consumers had to wrap every call site in their own gate. The bool is now always present; without the feature a `warn!` is emitted if the caller asked for diarization so the contract mismatch stays observable.
- **INT8 quantization is now always available and auto-invoked** (V1-10, `src/main.rs`, `src/quantize.rs`, `src/lib.rs`, `Cargo.toml`). The native Rust quantization pipeline no longer hides behind the `quantize` Cargo feature — `onnx-pb` and `prost` are now unconditional dependencies and `pub mod quantize` is always compiled in. `gigastt download` and `gigastt serve` both call `ensure_int8_encoder` after `model::ensure_model`, producing the `v3_e2e_rnnt_encoder_int8.onnx` artifact on first run (~2 min one-time). The `quantize` feature is retained as a documented no-op so existing `cargo install gigastt --features quantize` invocations keep working.
- **New `--skip-quantize` flag on `serve` and `download`** (V1-10, env `GIGASTT_SKIP_QUANTIZE=1`, default off). Opt out of the automatic quantization step when debugging against the FP32 encoder.

### Fixed

- **Model download TOCTOU** (V1-01, `src/model/mod.rs`). `download_file` used to stream each `v3_e2e_rnnt_*.onnx` blob directly into its final path, compute SHA-256 afterwards, and `remove_file` on mismatch. Between the last `write` and the hash comparison another process (or a second `ensure_model` call on restart) could observe an unverified file under the canonical name — and a crash in that window left a corrupt artefact that `model_files_exist()` would later accept, skipping re-download on next boot. Downloads now stream into `<filename>.partial`, SHA-256 is computed against the partial, and only after verification does `std::fs::rename` (atomic on the same filesystem) promote it to the final path. Mismatch or crash leaves nothing under the final name. Stale `.partial` files from previous crashed runs are deleted before the new download begins.
- **Speaker diarization model lacked SHA-256 verification** (V1-02, `src/model/mod.rs`, `--features diarization`). `ensure_speaker_model` streamed `wespeaker_resnet34.onnx` (26 535 549 bytes, from `onnx-community/wespeaker-voxceleb-resnet34-LM`) straight to its final path with no integrity check, so a tampered mirror or corrupted redirect was loaded into `ort::Session` without complaint — the same failure class as V1-01. The downloader now stages into `<name>.partial`, verifies SHA-256 against the new `SPEAKER_MODEL_SHA256 = "3955447b0499dc9e0a4541a895df08b03c69098eba4e56c02b5603e9f7f4fcbb"` constant (pinned to the 2026-04-20 HuggingFace copy), and only then atomically renames.
- **Odd-length PCM16 WebSocket frames corrupted subsequent frames** (V1-25, `src/server/mod.rs`). `handle_binary_frame` called `chunks_exact(2)` directly, silently dropping a trailing odd byte whenever a client split their PCM16 stream on an odd boundary. The dropped byte put the following frame 1 sample out of phase with the audio decoded so far — subtle in the waveform, measurable in the inference output, hard to diagnose. A per-connection `pending_byte: Option<u8>` now carries the remainder across frames (prepended before the next `chunks_exact`, re-stashed if the combined length is again odd).
- **`tests/e2e_rest.rs::test_rest_large_body_rss_within_budget` was mis-sized**. The helper call `generate_wav(150, 16000)` produced a 4.6 MiB WAV but the test asserted `> 30 MiB` and panicked before running. Regenerated at `generate_wav(300, 16000)` (9.6 MiB) with a budget that now accounts for the PCM16 → f32 expansion (2× wav.len() + 40 MiB slack).
- **`deny.toml` / `.cargo/audit.toml` justification for `RUSTSEC-2021-0073`** (V1-10 follow-up). `onnx-pb` has no newer release on crates.io (0.1.4 is the only published version) and the closest modern replacement (`onnx-protobuf` on the `protobuf` crate family) is broken at its current release. The advisory stays ignored with a refreshed rationale documenting that the affected `From<Timestamp> for SystemTime` code path is unreachable from our quantization pipeline.

## [0.9.0-rc.2] - 2026-04-20

### Fixed

- **`test_rest_oversized_body_rejected` e2e assertion** (`tests/e2e_errors.rs`). The rc.1 assertion insisted on a JSON body with `code="payload_too_large"`, but `axum::DefaultBodyLimit` returns a plain-text 413 when `Content-Length` exceeds the cap — the middleware layer fires before the handler's defence-in-depth guard. The V1-22 contract (strict 413 status) is unchanged; the JSON-body check is now conditional on the handler-layer guard being the one that fires. The rc.1 binaries are functionally correct.

## [0.9.0-rc.1] - 2026-04-20

_Release candidate for v0.9.0 — bundles five P0 fixes (V1-03, V1-04, V1-05, V1-06, V1-07) plus two supporting items (V1-21 `PoolGuard` Drop, V1-22 strict 413 assertion) from `specs/prod-readiness-v1.0.md`. RuntimeLimits gained two fields (`max_session_secs`, `shutdown_drain_secs`) — external callers constructing the struct literally must update their call sites. SessionPool checkout API replaced (`checkout() -> PoolGuard`)._

### Added

- **Graceful WebSocket / SSE drain on shutdown** (V1-03; closes `specs/prod-readiness-v1.0.md` P0). `axum::serve.with_graceful_shutdown` only tracks the HTTP router — WebSocket upgrades and SSE `spawn_blocking` tasks used to outlive the signal, so clients lost their `Final` frame on deploy. New `CancellationToken` + `TaskTracker` cascade through every handler; on SIGTERM each live session flushes, emits an empty-if-needed `Final`, and closes with `Close(1001 Going Away)`. After `axum::serve` returns, `run_with_config` waits up to `shutdown_drain_secs` for the tracker to drain.
- **Wall-clock max-session cap** (V1-04; closes `specs/prod-readiness-v1.0.md` P0). `idle_timeout` is reset on every frame, so a client that streams silence every 100 ms held a `SessionTriplet` forever. New `max_session_secs` limit closes the session with `Close(1008 Policy Violation)` + `Error { code: "max_session_duration_exceeded" }`. `0` disables the cap (not recommended).
- **CLI flags.**
  - `--max-session-secs` / `GIGASTT_MAX_SESSION_SECS` (default `3600`).
  - `--shutdown-drain-secs` / `GIGASTT_SHUTDOWN_DRAIN_SECS` (default `10`, clamped to `>= 1`).
- **`tests/e2e_shutdown.rs` re-enabled in CI** with four additional assertions: `test_shutdown_ws_emits_final_and_close`, `test_shutdown_sse_stream_terminates_cleanly`, `test_max_session_duration_cap`, and `test_shutdown_during_pool_saturation_returns_503_not_500`. The main-push e2e job now runs the full `--test e2e_rest --test e2e_ws --test e2e_errors --test e2e_shutdown` matrix.
- **`docs/runbook.md`** — rollback + on-call guidance for the new knobs.
- **`docs/deployment.md`** — `terminationGracePeriodSeconds` recommendation for k8s / docker-compose.

### Fixed

- **Per-IP rate-limiter math (V1-06, `src/server/mod.rs`).** `(rate_limit_per_minute / 60).max(1)` truncated every value below 60 rpm to a 1 rps refill (= 60 rpm), so a defender setting `--rate-limit-per-minute 10` actually allowed 60 rpm — 6× weaker than declared. Switched to `tower_governor`'s `per_millisecond(60_000 / rpm)`, which preserves sub-second precision down to 1 rpm and clamps the upper bound at 60 000 rpm with a `warn!`. The startup log now includes the resolved `interval_ms` alongside `rpm` for diagnostics.
- **Session pool panic + unfairness (V1-07, V1-21, `src/inference/mod.rs`).** Replaced the `tokio::sync::mpsc::Receiver` behind a `tokio::sync::Mutex` with a lock-free `async_channel`. The new `Pool<T>` (alias `SessionPool = Pool<SessionTriplet>`) is FIFO under contention, exposes `close()` so graceful shutdown wakes every waiter with `PoolError::Closed` instead of panicking via `.expect("Pool sender dropped")`, and returns a `PoolGuard` whose `Drop` impl auto-checks-in the triplet on panic unwind. Server shutdown now wires `engine.pool.close()` into the shutdown future, and the REST handlers translate `PoolError::Closed` into a distinct 503 `pool_closed` response (separate from the 503 `timeout` for the 30 s checkout deadline).
- **REST oversized-body rejection is now strict 413** (V1-22, `tests/e2e_errors.rs::test_rest_oversized_body_rejected`). Handlers in `src/server/http.rs` now add an explicit `body.len() > limits.body_limit_bytes → 413 payload_too_large` guard as defence-in-depth behind `DefaultBodyLimit`, and the e2e assertion upgrades from `!= 200` to a strict `== 413` with `code="payload_too_large"`.

### Changed

- `RuntimeLimits` gained `max_session_secs: u64` and `shutdown_drain_secs: u64`. External callers constructing `RuntimeLimits` literally will need to add the new fields (pre-1.0 minor bump — acceptable).
- `http::AppState` carries `shutdown: CancellationToken` and `tracker: TaskTracker`.
- `handle_ws_inner` switches from a bare `timeout(idle, source.next())` to a `biased;` `select!` with explicit cancel + deadline branches.
- `/v1/transcribe/stream` SSE task now runs on `TaskTracker::spawn_blocking` and polls the shutdown token between chunks so SIGTERM aborts long transcriptions instead of waiting them out.
- `SessionPool::{checkout, checkin, blocking_checkin}` replaced by `SessionPool::checkout() -> Result<PoolGuard, PoolError>`. The guard `Deref`s to `SessionTriplet` and auto-checks-in on drop. For `'static` consumers (`spawn_blocking`), call `guard.into_owned()` to get a `(SessionTriplet, OwnedReservation)` pair and return the triplet via `OwnedReservation::checkin(triplet)`.
- **Zero-copy REST upload decode path** (V1-05, `src/inference/audio.rs`, `src/inference/mod.rs`, `src/server/http.rs`). The `/v1/transcribe` and `/v1/transcribe/stream` handlers used to call `body.to_vec()` on the incoming `axum::body::Bytes`, then `decode_audio_bytes` cloned that `Vec<u8>` into a `std::io::Cursor`, and symphonia decoded the PCM into another `Vec<f32>` — four concurrent copies of the upload were in RAM at peak. A 4× concurrent upload of a 10-minute WAV held ~1 GiB transiently and could OOM on a 1 GiB container. New path: `bytes::Bytes` flows end-to-end via a crate-private `BytesMediaSource` that implements `Read + Seek + MediaSource` directly on the refcounted buffer; new `decode_audio_bytes_shared(Bytes)` and `Engine::transcribe_bytes_shared(Bytes, _)` entry points. The legacy `decode_audio_bytes(&[u8])` / `Engine::transcribe_bytes(&[u8], _)` functions remain as thin shims (one `Bytes::copy_from_slice` for non-REST callers), so no public API breakage.
- **Incremental 10-minute duration cap** inside the decode loop (V1-05). The check used to fire only after the full PCM buffer was assembled, so a malformed or hostile upload could still allocate hundreds of MiB before being rejected. Now each packet's samples are accumulated against a precomputed sample budget and the decoder bails out on the first packet that breaks the cap.

### Dependencies

- Promoted `tokio-util = { version = "0.7", features = ["rt"] }` from transitive to direct. Dev-deps gained `tracing-subscriber` so integration tests can surface server logs on failure.
- `async-channel = "2"` (transitive pieces — `concurrent-queue`, `event-listener` — were already in the graph).
- Added explicit `bytes = "1"` pin (previously transitive via `axum` / `tokio`) — makes the zero-copy contract between axum and symphonia visible in `Cargo.toml`.

## [0.8.1] - 2026-04-17

### Fixed

- **CoreML / CUDA startup crash on macOS 26+ (`Unable to serialize model as it contains compiled nodes`)** — `src/inference/mod.rs` previously called `.with_optimized_model_path(...)` after registering the CoreML / CUDA execution providers. Those EPs replace parts of the graph with compiled nodes that cannot be re-serialized as ONNX, so ORT aborted session creation before the server could bind. Regression introduced in v0.5.0. The optimized-ONNX cache path is removed from both EP paths; the CoreML block keeps its dedicated `coreml_cache/` (compiled-model cache) and the CUDA EP keeps its internal caches. Cost: ~1–2 s additional cold start. Benefit: `gigastt serve --features coreml` works again on macOS 14+.

## [0.8.0] - 2026-04-17

### Added

- **Prometheus `/metrics` endpoint** (closes `specs/todo.md` item 7). Enabled via `--metrics` (env `GIGASTT_METRICS=1`); off by default. Exposes
  - `gigastt_http_requests_total{method,path,status}` (counter)
  - `gigastt_http_request_duration_seconds{method,path}` (histogram).
  The endpoint sits behind the Origin allowlist and (when configured) the per-IP rate limiter. Recorder install is tolerant of double-install: emits a warning and keeps the server running instead of failing.
- **Per-IP rate limiting** (closes `specs/todo.md` item 17). `--rate-limit-per-minute N` (env `GIGASTT_RATE_LIMIT_PER_MINUTE`) + `--rate-limit-burst N` (env `GIGASTT_RATE_LIMIT_BURST`). Off by default. Applies to `/v1/*` and `/v1/ws`; `/health` is exempt. Implemented with `tower_governor` using `SmartIpKeyExtractor`. Returns 429 on violations. A background task evicts expired token buckets every 60 s.
- **`docs/deployment.md`** (closes `specs/todo.md` item 20). Reverse-proxy recipes for Caddy and nginx (certbot + `auth_basic`), Origin header behaviour, Docker binding strategy, health-check target, and a hardening checklist for remote deployments.

### Changed

- `ServerConfig` gained a `metrics_enabled: bool` field; `RuntimeLimits` gained `rate_limit_per_minute` + `rate_limit_burst`.
- `http::AppState` now carries `metrics_handle: Option<PrometheusHandle>`.
- The axum router splits into `/health` (public) and a `protected` sub-router for `/v1/*`, `/ws` alias, `/v1/ws`, and `/metrics` — rate limiter is layered on the protected branch only.

### Dependencies

- `tower_governor = "0.7"`
- `metrics = "0.24"`
- `metrics-exporter-prometheus = "0.17"` (default-features off)

## [0.7.2] - 2026-04-17

### Fixed

- **`cargo-deny` licenses + advisories** (`deny.toml`).
  - Added `CDLA-Permissive-2.0` to the license allowlist — `webpki-root-certs` (the Mozilla CA bundle) publishes under CDLA; behaves like MIT for our use.
  - Added `RUSTSEC-2021-0073` to `ignore` in `[advisories]`. `prost-types 0.6.1` is a build-time transitive of `onnx-pb` under the `quantize` feature; the affected `From<Timestamp> for SystemTime` path is not reached by our pipeline.

## [0.7.1] - 2026-04-17

### Fixed

- **`cargo-deny` CI job** (`.github/workflows/ci.yml`) — removed the trailing `arguments: licenses advisories bans sources`. The installed `cargo-deny` on stable-musl interpreted them as subcommands and failed with `unrecognized subcommand 'licenses'`. Default `check` already covers licenses, advisories, bans, and sources.

## [0.7.0] - 2026-04-17

### Added

- **Configurable runtime limits** (`gigastt::server::RuntimeLimits`, closes `specs/todo.md` item 6). Three knobs exposed via CLI + environment variables:
  - `--idle-timeout-secs` / `GIGASTT_IDLE_TIMEOUT_SECS` — WebSocket idle timeout (default 300).
  - `--ws-frame-max-bytes` / `GIGASTT_WS_FRAME_MAX_BYTES` — max WS frame / message (default 512 KiB).
  - `--body-limit-bytes` / `GIGASTT_BODY_LIMIT_BYTES` — max REST body (default 50 MiB).
  Delivered via a new `RuntimeLimits` field on `ServerConfig` and `http::AppState`; TOML config file support stays for a follow-up.
- **Canonical WebSocket path `/v1/ws`** (closes `specs/todo.md` item 11). Versioned path aligned with REST; legacy `/ws` remains as an alias with a warn-level deprecation log on every upgrade. Removal planned for v1.0.
- **`diarization` capability in `GET /v1/models`** (closes `specs/todo.md` item 12). Mirrors the WebSocket `Ready` field so clients can probe capabilities without opening a WS.
- **Docker `GIGASTT_BAKE_MODEL=1` build-arg** (closes `specs/todo.md` item 10). When set, a dedicated `model-fetcher` stage runs `gigastt download` during image build and the runtime stage copies the model into `/home/gigastt/.gigastt/models/`. Default (`0`) preserves the slim image.
- **`cargo deny check` in CI + `deny.toml`** (closes first half of `specs/todo.md` item 14 — SBOM stays for later). Enforces license allowlist + advisory scan + crates.io-only source + wildcard ban on every PR via `EmbarkStudios/cargo-deny-action@v2`.

### Changed

- `http::AppState` now carries `limits: RuntimeLimits` alongside `engine`; `handle_ws_inner` takes `&RuntimeLimits` so the idle timeout is no longer hard-coded.
- `DefaultBodyLimit::max` in the Axum router reads from `config.limits.body_limit_bytes` instead of the old `50 * 1024 * 1024` literal.
- `ws_handler` reads `ws_frame_max_bytes` from `AppState` instead of baking 512 KiB into the code path.

### Notes

- `RuntimeLimits`, `ServerConfig::local`, and `run_with_config` are public — downstream embedders can construct a fully customised server without CLI.
- Murmur remains on v0.6.0: no wire protocol break, new limits are additive + opt-in.

## [0.6.1] - 2026-04-17

### Changed

- **`handle_ws_inner` refactor** (`src/server/mod.rs`) — extracted three frame handlers (`handle_binary_frame`, `handle_configure_message`, `handle_stop_message`) and a `send_server_message` helper; the session loop is now ~60 lines with a single `FrameOutcome` dispatch. Behavior unchanged (same tests + clippy clean); reduces future risk when touching the hot path.

### Added

- **Integration test for origin middleware** (`src/server/mod.rs::tests::test_origin_middleware_integration`) — spins a minimal axum router with `origin_middleware` on a real port and verifies: `/health` is exempt, cross-origin `/v1/*` returns 403 `origin_denied`, loopback Origin is echoed into `Access-Control-Allow-Origin`, no-Origin requests pass through, and `localhost.evil.example.com` DNS-continuation attempts are denied.

## [0.6.0] - 2026-04-17

### Added

- **Origin allowlist middleware.** Cross-origin requests from non-loopback pages are denied by default across `/v1/*` and `/ws`; loopback origins (`localhost`, `127.0.0.1`, `[::1]`) always pass. New CLI flags:
  - `--allow-origin <URL>` (repeatable) — exact-match, case-insensitive Origin allowlist.
  - `--cors-allow-any` — legacy `Access-Control-Allow-Origin: *` behaviour, opt-in.
  `/health` remains free of Origin checks for monitoring / Docker `HEALTHCHECK`.
- **`--bind-all` guard.** `gigastt serve` refuses non-loopback `--host` values (`0.0.0.0`, LAN IPs, …) unless `--bind-all` is passed or `GIGASTT_ALLOW_BIND_ANY=1` is set. Both `Dockerfile` and `Dockerfile.cuda` now pass `--bind-all` in their `CMD` line.
- **`Retry-After` on pool saturation.**
  - REST `/v1/transcribe` and `/v1/transcribe/stream` return HTTP 503 with a `Retry-After: 30` header (RFC 9110 §10.2.3) and `retry_after_ms: 30000` in the JSON body.
  - WebSocket `ServerMessage::Error` gained an optional `retry_after_ms` field (omitted from JSON when absent, preserving backward compatibility); the pool-timeout path at connect emits `retry_after_ms: 30000`.
- **`gigastt::server::{ServerConfig, OriginPolicy, run_with_config}`** — new public API for programmatic startup with explicit origin policy.

### Changed

- **Default cross-origin posture is now deny.** Previous behaviour (wildcard CORS + non-local Origin only warned about) is preserved behind `--cors-allow-any`. Browser integrations hitting the server from a non-loopback page must either add their origin via `--allow-origin` or run the server with `--cors-allow-any`.

### Security

- Closes `specs/todo.md` P1 items 4, 5, 8, 9. Reduces the risk that a malicious webpage can drive-by-connect to the local transcription server and exfiltrate microphone audio.

## [0.5.3] - 2026-04-17

### Security

- **`rustls-webpki` 0.103.10 → 0.103.12** (`Cargo.lock`) — resolves RUSTSEC-2026-0098 (name constraints for URI names incorrectly accepted) and RUSTSEC-2026-0099 (name constraints accepted for certificates asserting a wildcard name). Pulled in transitively via `reqwest → hyper-rustls → rustls`.

## [0.5.2] - 2026-04-17

### Fixed

- **CI clippy** (`src/model/mod.rs:29`) — replaced manual `if self.total > 0` division guard with `checked_div`, satisfying Rust 1.95's new `clippy::manual_checked_ops` lint that broke CI on v0.5.1.
- **Release workflow** (`.github/workflows/release.yml`) — removed the `linux-x86_64-cuda` matrix entry: `Jimver/cuda-toolkit@v0.2.19` cannot resolve the `cuda-nvcc-12-4` / `cuda-cudart-12-4` packages on `ubuntu-latest`. Tracked for re-enabling in `specs/todo.md`. Until then CUDA users build from source.

## [0.5.1] - 2026-04-17

### Added

- **Release automation** (`.github/workflows/release.yml`) — tag-triggered matrix workflow that produces `gigastt-<ver>-aarch64-apple-darwin.tar.gz` (coreml), `gigastt-<ver>-x86_64-unknown-linux-gnu.tar.gz` (cpu), `gigastt-<ver>-x86_64-unknown-linux-gnu-cuda.tar.gz`, per-asset `.sha256` files, and aggregated `SHA256SUMS.txt`. Replaces ad-hoc manual uploads that previously broke SHA-pinned downstream clients.
- **`CONTRIBUTING.md`** — release checklist and contribution guidelines, including an explicit prohibition on manual `gh release upload` of binary assets.
- **`examples/bun_client.ts`, `examples/go_client.go`, `examples/KotlinClient.kt`** — WebSocket client samples in Go, Kotlin (OkHttp), and Bun-native TypeScript.
- **`specs/todo.md` + `specs/plan.md`** — 20-item follow-up list from the v0.5.0 critique, ranked P0/P1/P2 and sequenced into six phases through v1.0.0.

### Fixed

- **WebSocket pool recovery after inference panic** (`src/server/mod.rs`) — a panic inside `process_chunk` used to leak the `SessionTriplet` and permanently shrink the pool. Now the blocking task owns the state and triplet, wraps the inner call in `catch_unwind(AssertUnwindSafe(_))`, and returns both unconditionally. On panic the WS session sends an `inference_panic` error, resets its streaming state, and continues instead of tearing down.
- **`clippy::never_loop`** in `tests/e2e_errors.rs` (two occurrences) — replaced the single-iteration `while let` drains with a `tokio::time::timeout(_).await` call, unblocking stricter lint levels.

### Removed

- **`scripts/quantize.py`** — superseded by native Rust quantization (`gigastt quantize --features quantize`).
- **`examples/js_client.mjs`** — replaced by `examples/bun_client.ts`.

## [0.5.0] - 2026-04-13

### Added

- **Native Rust INT8 quantization** (`--features quantize`) — `gigastt quantize` command replaces `scripts/quantize.py`. Per-channel symmetric QDQ format, hardened against shared weights and malformed tensors.
- **Auto-quantize on download/serve** — automatically creates INT8 encoder when built with `--features quantize`. Prints hint otherwise.
- **`GET /v1/models` endpoint** — returns model info: encoder type (int8/fp32), vocab size, pool status, supported formats and sample rates.
- **`--log-level` CLI option** — global flag for all commands (`gigastt --log-level debug serve`), replaces `RUST_LOG`-only config.
- **`--pool-size` CLI option** — configurable concurrent inference sessions for `serve` command.
- **`Engine::is_int8()`** method exposes encoder quantization status.
- **PrepackedWeights** — shared ONNX Runtime weight memory across session pool (reduced memory footprint).
- **Inference instrumentation** — encoder/decoder timing logged at info level.
- **Russian README** (`README_RU.md`) with language switcher.
- **CI `cargo fmt --check`** job for format enforcement.

### Changed

- **WER benchmark** verified on 993 Golos samples (4991 words): FP32 10.5%, INT8 10.4% — 0% degradation confirmed.
- **README** updated with verified metrics: WER 10.4%, latency ~700ms, memory ~560MB. Expanded comparison table.
- **Decoder optimization** — cached decoder output during blank runs (86% decoder call reduction).
- **Optimized model cache** directory for pre-compiled ONNX models.

### Fixed

- **Server hardening** — WS pool checkout timeout (30s), REST `catch_unwind` for panic recovery, removed `unwrap`/`expect` in handlers.
- **Security** — upgraded `tokio-tungstenite` 0.24→0.28, resolving RUSTSEC-2026-0097 (`rand` 0.8.5 unsoundness).
- **CI stability** — e2e tests serialized with `--test-threads=1` (prevents OOM), shutdown tests excluded (require graceful connection termination), SSE tests resilient to non-speech audio.
- **Benchmark overflow** — `number_to_words` handles numbers > 999,999.
- **Dockerfiles** updated to Rust 1.85+ for edition 2024 support.
- **Audio decode refactor** — extracted shared inner function, eliminated ~80 line duplication.

## [0.4.3] - 2026-04-13

### Added

- **Comprehensive e2e test infrastructure** — 28 new tests across 7 files:
  - `tests/e2e_rest.rs` (8 tests): health, transcribe, SSE streaming, error paths
  - `tests/e2e_ws.rs` (9 tests): WebSocket protocol — ready, audio, stop, configure, malformed JSON, disconnect, concurrency
  - `tests/e2e_errors.rs` (5 tests): oversized body/frame rejection, pool saturation (503), idle timeout
  - `tests/e2e_shutdown.rs` (2 tests): graceful shutdown during active WS/SSE sessions
  - `tests/load_test.rs` (3 tests): 4 concurrent WS/REST, burst 20 connections
  - `tests/soak_test.rs` (1 test): continuous WS cycling (configurable via `GIGASTT_SOAK_DURATION_SECS`)
- **Shared test helpers** (`tests/common/mod.rs`): `start_server` with clean shutdown, `wait_for_ready` with exponential backoff, WAV generation, WebSocket connect helpers.
- **`server::run_with_shutdown()`** — accepts optional `oneshot::Receiver<()>` for programmatic server shutdown (used by tests; `run()` unchanged).
- **CI feature matrix** — split into 7 jobs: clippy, unit tests, build-coreml, build-cuda, build-diarization, e2e tests (main push only with cached model), security audit.

### Changed

- CI workflow restructured: PRs get fast feedback (unit + clippy + feature builds), main push adds full e2e suite with ~850MB cached model (OS-independent cache key).

## [0.4.2] - 2026-04-13

### Removed

- **`dirs` dependency** — replaced with `env::var("HOME")` / `USERPROFILE` (~10 lines).
- **`indicatif` dependency** — replaced with simple stderr progress output (~50 transitive deps removed).
- **`tempfile` from production deps** — HTTP handlers decode audio from memory via `Cursor<Vec<u8>>` (faster, no disk I/O). Kept in dev-dependencies for tests.
- **`async-stream` dependency** — replaced with `futures_util::stream::unfold`.
- **`tower-http` dependency** — replaced with axum's built-in `DefaultBodyLimit`.

### Added

- `decode_audio_bytes()` — decode audio from in-memory bytes without temp files.
- `Engine::transcribe_bytes()` — transcribe from byte buffer directly.
- Security audit job in CI workflow (`cargo audit`).
- Non-root user in Dockerfiles (hardened containers).

## [0.4.1] - 2026-04-13

### Changed

- Diarization module no longer depends on internal `ort_err()` helper — uses `anyhow::Context` instead. Module is now self-contained and ready for future crate extraction.

### Fixed

- Centroid re-normalization after running average update (prevents speaker clustering drift).
- Semaphore timeout (30s) on HTTP endpoints prevents DoS via hanging requests.
- SSE semaphore permit held for stream lifetime (was dropped before stream consumed).
- SSE inference wrapped in `spawn_blocking` (no longer blocks async runtime).
- Error messages sanitized at HTTP API boundary (no internal path/model leakage).
- Speaker count capped at 64 (`MAX_SPEAKERS`) with graceful fallback.
- Cosine similarity zero-norm check uses epsilon (1e-8) instead of exact float equality.
- Request body dropped after temp file write (reduces peak memory ~2x for large files).
- Configure message rejected after first audio frame (`configure_too_late` error).
- WebSocket idle timeout (300s) disconnects silent clients.
- Unnecessary `samples_16k_copy` allocation skipped when diarization disabled at runtime.
- Async `tokio::fs::write` replaces blocking `std::fs::write` in HTTP handlers.
- `tokio-tungstenite` moved to dev-dependencies (unused in production code).
- `hound` dependency removed (unused).
- CLAUDE.md and README.md updated: test counts, architecture tree, WebSocket URL `/ws`, REST API docs, version references.

## [0.4.0] - 2026-04-13

### Added

- **Cross-platform support** via compile-time Cargo feature flags:
  - `--features coreml`: macOS ARM64 (CoreML + Neural Engine) — existing behavior
  - `--features cuda`: Linux x86_64 (NVIDIA CUDA 12+)
  - Default (no features): CPU-only, compiles on any platform
  - `compile_error!` guard prevents enabling both `coreml` and `cuda`
- **Flexible sample rate**: `ClientMessage::Configure { sample_rate }` lets clients declare input rate (8kHz, 16kHz, 24kHz, 44.1kHz, 48kHz). Default 48kHz for backward compatibility.
- **Polyphase FIR resampler** (rubato `SincFixedIn`) replaces linear interpolation — significantly better audio quality.
- **`ServerMessage::Ready`** extended with `supported_rates` field (list of accepted sample rates).
- **HTTP REST API** via axum (single port serves HTTP + WebSocket):
  - `GET /health` — health check for monitoring and Docker HEALTHCHECK
  - `POST /v1/transcribe` — upload audio file, receive full JSON transcript
  - `POST /v1/transcribe/stream` — upload audio file, receive SSE stream of partial/final results
  - `GET /ws` — WebSocket streaming (existing protocol, new path)
- **Speaker diarization** (optional, `--features diarization`):
  - WeSpeaker ResNet34 ONNX model (26.5MB, 256-dim embeddings, 16kHz)
  - Online incremental clustering (cosine similarity, configurable threshold)
  - `WordInfo.speaker: Option<u32>` field identifies speakers per word
  - `Configure { diarization: true }` enables per-session
  - `gigastt download --diarization` fetches speaker model separately
  - MAX_SPEAKERS=64 cap with graceful fallback to closest match
- **`Dockerfile.cuda`** — multi-stage CUDA build with `nvidia/cuda:12.6.3-cudnn-runtime`
- **GitHub Actions CI** matrix: macOS (CoreML) + Linux (CPU) in parallel
- **Semaphore timeout** (30s) on HTTP endpoints prevents DoS via hanging requests
- **WebSocket idle timeout** (300s) disconnects silent clients
- **Configure guard** — server rejects `Configure` after first audio frame

### Changed

- **Server migrated from raw tokio-tungstenite to axum** — single port serves HTTP routes + WebSocket upgrade
- **WebSocket endpoint moved to `/ws`** (was root `/`). Clients must connect to `ws://host:port/ws`.
- **`ClientMessage::Configure.sample_rate`** changed from `u32` to `Option<u32>` to support partial configuration (sample rate only, diarization only, or both).
- **Dockerfile** CPU build no longer uses `--no-default-features` (default features are now empty = CPU).
- **SSE inference** runs in `spawn_blocking` (no longer blocks async runtime).
- **Error responses** in HTTP handlers sanitized — generic messages to clients, details logged server-side.
- `tokio-tungstenite` moved from production to dev-dependencies (only used in integration tests).
- `hound` dependency removed (unused; all audio decoding via symphonia).

### Fixed

- **Centroid drift in speaker clustering** — centroids re-normalized after running average update.
- **Cosine similarity** zero-norm check uses epsilon (1e-8) instead of exact float equality.
- **SSE semaphore permit** held for stream lifetime (was dropped before stream consumed).
- **HTTP body memory** — request body dropped after temp file write, reducing peak memory usage.
- **Async file I/O** — `tokio::fs::write` replaces blocking `std::fs::write` in HTTP handlers.

### Breaking Changes

- WebSocket path changed: `/` → `/ws`. Update client connection URLs.
- `Configure.sample_rate` type changed: `u32` → `Option<u32>`. Existing JSON `{"type":"configure","sample_rate":8000}` still works via `#[serde(default)]`.
- Default `cargo build` (no features) now produces CPU-only binary. macOS users must explicitly add `--features coreml`.

## [0.3.0] - 2026-04-12

### Added

- `GigasttError` enum (`error` module) with variants: `ModelLoad`, `Inference`, `InvalidAudio`, `Io` — enables `match`-based error handling.
- `#[non_exhaustive]` on all public structs and enums — future additions are non-breaking.
- Comprehensive `///` rustdoc on all public types, functions, fields, and constants.
- Crate-level documentation with quick-start examples in `lib.rs`.
- Stress tests for NaN/infinity audio samples, empty inputs, and buffer boundary conditions.

### Fixed

- Potential panic on odd-length WebSocket binary frames (`chunks_exact(2)` now drops trailing byte with warning).
- Non-finite audio samples (NaN, infinity) in `resample()` replaced with zeros instead of propagating.

### Breaking Changes

- `Engine::load()`, `Engine::process_chunk()`, and `Engine::transcribe_file()` return `Result<T, GigasttError>` instead of `anyhow::Result<T>`.
- All public structs/enums are `#[non_exhaustive]` — external struct literal construction requires constructor methods.

## [0.2.0] - 2026-04-06

### Added

- Partial transcripts with real-time streaming via WebSocket.
- Endpointing detection (~600ms silence triggers finalization).
- Per-word timestamps (`WordInfo.start`, `WordInfo.end`) relative to stream start.
- Per-word confidence scores (`WordInfo.confidence`) averaged over BPE tokens.
- CoreML execution provider for macOS ARM64 (Neural Engine + CPU).
- INT8 quantized encoder support (`v3_e2e_rnnt_encoder_int8.onnx`, ~4x smaller, ~43% faster).
- CoreML model cache directory (`~/.gigastt/models/coreml_cache/`).
- Docker multi-stage build (`Dockerfile`).
- Python quantization script (`scripts/quantize.py`).

### Changed

- Audio pipeline: accept 48kHz from WebSocket clients, resample to 16kHz internally.
- Encoder output shape handling: channels-first `[1, 768, T]` format.

## [0.1.2] - 2026-04-01

### Added

- GigaAM v3 e2e_rnnt inference engine with ONNX Runtime.
- WebSocket server (tokio + tungstenite) for streaming audio.
- CLI: `serve`, `download`, `transcribe` commands.
- HuggingFace model auto-download (`istupakov/gigaam-v3-onnx`).
- BPE tokenizer (1025 tokens).
- Mel spectrogram (64 bins, FFT=320, hop=160, HTK).
- RNN-T greedy decode loop.
- Multi-format audio support: WAV, MP3, M4A/AAC, OGG/Vorbis, FLAC (via symphonia).
- 39 unit tests (tokenizer, features, decode, inference, protocol).

[Unreleased]: https://github.com/ekhodzitsky/gigastt/compare/v0.9.2...HEAD
[0.9.2]: https://github.com/ekhodzitsky/gigastt/compare/v0.9.1...v0.9.2
[0.9.1]: https://github.com/ekhodzitsky/gigastt/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/ekhodzitsky/gigastt/compare/v0.9.0-rc.2...v0.9.0
[0.9.0-rc.2]: https://github.com/ekhodzitsky/gigastt/compare/v0.9.0-rc.1...v0.9.0-rc.2
[0.9.0-rc.1]: https://github.com/ekhodzitsky/gigastt/compare/v0.8.1...v0.9.0-rc.1
[0.8.1]: https://github.com/ekhodzitsky/gigastt/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/ekhodzitsky/gigastt/compare/v0.7.2...v0.8.0
[0.7.2]: https://github.com/ekhodzitsky/gigastt/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/ekhodzitsky/gigastt/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/ekhodzitsky/gigastt/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/ekhodzitsky/gigastt/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/ekhodzitsky/gigastt/compare/v0.5.3...v0.6.0
[0.5.3]: https://github.com/ekhodzitsky/gigastt/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/ekhodzitsky/gigastt/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/ekhodzitsky/gigastt/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/ekhodzitsky/gigastt/compare/v0.4.3...v0.5.0
[0.4.3]: https://github.com/ekhodzitsky/gigastt/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/ekhodzitsky/gigastt/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/ekhodzitsky/gigastt/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/ekhodzitsky/gigastt/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ekhodzitsky/gigastt/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ekhodzitsky/gigastt/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/ekhodzitsky/gigastt/releases/tag/v0.1.2
