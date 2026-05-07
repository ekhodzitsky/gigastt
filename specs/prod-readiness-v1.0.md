# gigastt v1.0 — production-readiness TODO

Consolidated findings from the 4-critic review (2026-04-18) covering:
architecture, prod-readiness, test coverage, security, and gaps.
Supersedes the residual items in `specs/todo.md`; see `specs/plan.md`
for the execution phases.

Each item carries: **file:line** anchor, **symptom**, **fix direction**,
**effort** (S ≤ 0.5 day, M ≤ 2 days, L > 2 days), **acceptance** (how we
know it's done). Convergent findings (flagged by ≥ 2 critics independently)
are marked **[C]** — higher confidence, prioritise.

Legend: ✅ done · 🔨 in progress · ⏳ open · 💭 discussion needed

## Ranking methodology

Rows are ordered by execution priority, not by numeric ID (IDs are stable
anchors and do not imply order). Priority factors, in order of weight:

1. **Blast radius** — exploit / data loss / process crash > perf hit > polish.
2. **Convergence** — flagged by ≥ 2 critics independently **[C]** gets a boost.
3. **User visibility** — user-facing outage > internal debt.
4. **Effort vs impact** — small S-effort fixes with high impact jump ahead.
5. **Blocks others** — items that unblock future work go first within a tier.

Within each tier the order is prescriptive: start at the top, finish the
tier before moving on unless a lower item can be parallelised with zero
coupling.

## v0.9.0 rollup (2026-04-21)

The following IDs shipped in v0.9.0 (plus the 0.9.0-rc.1/rc.2 + 0.9.1/0.9.2
patch releases that cleaned the release pipeline around it). The detail
rows further down still carry their original `⏳ open` status for the
historical audit trail; trust this rollup over the table cells for
"what's already fixed":

- **P0 lane (closed):** V1-01 (model download TOCTOU + `.partial` atomic
  rename), V1-02 (speaker-model SHA-256), V1-03 (graceful WS / SSE drain
  on SIGTERM), V1-04 (max-session-secs cap), V1-05 (REST zero-copy
  `BytesMediaSource`), V1-06 (rate-limiter per-ms math + clamp), V1-07
  (pool `async-channel` + `PoolGuard`), V1-08 (`Engine::create_state`
  unconditional), V1-09 (`soak.yml` nightly CI), V1-10 (vendored ONNX
  proto + in-tree quantizer, drops `onnx-pb`/`prost 0.6`).
- **P1 (closed):** V1-11 (X-Forwarded-For trust boundary + recipe
  correctness), V1-14 (Deprecation + Link headers on `/ws` —
  shipped 2026-04-21), V1-22 (strict 413 assertion), V1-23
  (`server_integration.rs` deletion — shipped 2026-04-21), V1-25
  (odd-length PCM frame carry).
- **Sustainability (closed):** SUS-02 (CycloneDX SBOM in `release.yml`),
  SUS-03 (minisign signatures + public key published), SUS-05 (SLSA
  provenance attestations).

## Progress snapshot (2026-04-20) — priority-sorted

### P0 — blockers (ordered most → least critical)

| Rank | # | Area | Status |
|------|---|------|--------|
| P0.1 | V1-03 | Graceful WS drain on SIGTERM **[C]** (data loss on every deploy; `e2e_shutdown` disabled in CI) | ✅ done (v0.9.0) |
| P0.2 | V1-05 | REST `body.to_vec()` double-buffer **[C]** (515 MiB peak per upload → OOM) | ✅ done (v0.9.0) |
| P0.3 | V1-04 | Max session duration cap (silence-stream DoS; pool starvation in 5 connections) | ✅ done (v0.9.0) |
| P0.4 | V1-06 | Rate-limiter `/60` int-division (documented protection 60× weaker than declared) | ✅ done (v0.9.0) |
| P0.5 | V1-07 | Pool `Mutex<mpsc::Receiver>` + `.expect("Pool sender dropped")` (unfair + panic cascade) | ✅ done (v0.9.0) |
| P0.6 | V1-01 | Model download TOCTOU (SHA256 verified after write → RCE-class via tampered model) | ✅ done (v0.9.0) |
| P0.7 | V1-02 | Speaker model has no SHA256 verify at all (same class as V1-01, under feature) | ✅ done (v0.9.0) |
| P0.8 | V1-08 | `Engine::create_state` signature toggles on feature (breaks crates.io doctest, SemVer) | ✅ done (v0.9.0) |
| P0.9 | V1-10 | `prost 0.6` + `onnx-pb 0.1.4` supply-chain (3+ yr unmaintained, `cargo install` UX) | ✅ done (v0.9.0) |
| P0.10 | V1-09 | Nightly soak/load in CI (infra, not a bug — but without it P0 regressions ship) | ✅ done (v0.9.0) |

### P1 — ship-before-v1.0 (ordered most → least critical)

| Rank | # | Area | Status |
|------|---|------|--------|
| P1.1 | V1-11 | `X-Forwarded-For` spoofing in published nginx/Caddy recipes (per-IP RL bypass) | ✅ done (v0.9.0) |
| P1.2 | V1-12 | `/metrics` on CORS-allowlisted router + rate-limited (info leak + Prom throttle) | ⏳ open |
| P1.3 | V1-16 | `thread::scope` panic on pool-load aborts the whole process (OOM on 4 GB box) | ⏳ open |
| P1.4 | V1-21 | `SessionPool` lacks Drop-guard (per-call `catch_unwind` is opt-in, bugs pool) | ⏳ open |
| P1.5 | V1-25 | Odd-length PCM frame drops last byte silently (phase-shift corruption) | ✅ done (v0.9.0) |
| P1.6 | V1-27 | `/health` does not probe encoder (k8s liveness false-positive) | ⏳ open |
| P1.7 | V1-24 | Batch REST pool starves WS streaming (9 min jobs → 30 s WS timeouts) | ⏳ open |
| P1.8 | V1-17 | Global `PrometheusBuilder` forces `--test-threads=1` (CI slowdown + stale `/metrics`) | ⏳ open |
| P1.9 | V1-30 | Missing pool / inference / WS Prometheus metrics (unalertable SRE posture) | ⏳ open |
| P1.10 | V1-15 | Rate-limiter retain-recent `std::thread` never exits (leak + slow SIGTERM) | ⏳ open |
| P1.11 | V1-28 | Pool checkout timeout hardcoded in two places (operational rigidity) | ⏳ open |
| P1.12 | V1-13 | WebSocket protocol version is declarative, no negotiation (future breakage) | ⏳ open |
| P1.13 | V1-14 | `/ws` deprecation has no `Deprecation` / `Sunset` hint for clients | ✅ done (v0.9.0) |
| P1.14 | V1-18 | Decoder loop allocations in hot path (millions of `.to_vec()` per 40 s clip) | ⏳ open |
| P1.15 | V1-19 | `SincFixedIn::new(chunk_size = samples.len())` rebuilds FIR per streaming chunk | ⏳ open |
| P1.16 | V1-20 | `quantize.rs` hardcodes `axis=0` for all ops (silent WER regression on MatMul) | ⏳ open |
| P1.17 | V1-22 | `test_rest_oversized_body_rejected` asserts `!= 200` (fake test) | ✅ done (v0.9.0) |
| P1.18 | V1-29 | Idle timeout test burns 310 s wall-clock in CI main push | ⏳ open |
| P1.19 | V1-23 | Remove `tests/server_integration.rs` (6 `sleep(200 ms)` duplicates of `e2e_ws`) | ✅ done (v0.9.0) |
| P1.20 | V1-26 | Engine god-object (`transcribe_file` / `process_chunk` / `tokens_to_words`) | ⏳ open |

### P2 — polish for v1.x (ordered most → least critical)

| Rank | # | Area | Status |
|------|---|------|--------|
| P2.1 | V1-33 | AsyncAPI spec diverged from code (paths, schema, error enum, sample math) | ⏳ open |
| P2.2 | V1-47 | No inference timeout around `spawn_blocking` (ORT hang → pool dry) | ⏳ open |
| P2.3 | V1-41 | Benchmark has no baseline or regression gate (WER drift unnoticed) | ⏳ open |
| P2.4 | V1-36 | Prometheus `path` label = raw URI (cardinality explosion via scanners) | ⏳ open |
| P2.5 | V1-37 | No server-side WebSocket ping timer (proxies drop idle connections silently) | ⏳ open |
| P2.6 | V1-42 | Only 15 Golos fixtures (not statistically significant for WER gating) | ⏳ open |
| P2.7 | V1-46 | `Engine::warmup()` missing (first-request cold start, CoreML compile) | ⏳ open |
| P2.8 | V1-50 | Multi-model support via `manifest.toml` (blocks GigaAM v4 without code change) | ⏳ open |
| P2.9 | V1-35 | SSE error-code parity with WebSocket (single `inference_error` hides variants) | ⏳ open |
| P2.10 | V1-48 | VAD endpointing (noise breaks blank-run endpointing today) | ⏳ open |
| P2.11 | V1-40 | Pin `tokio`/`serde` minor versions + dry-run check (supply-chain hygiene) | ⏳ open |
| P2.12 | V1-31 | `consecutive_blanks++` on MAX_TOKENS overrun (semantic mixing) | ⏳ open |
| P2.13 | V1-43 | `now_timestamp()` uses `unwrap_or_default()` (clock-skew edge case) | ⏳ open |
| P2.14 | V1-38 | `.cargo/audit.toml` + `deny.toml` duplicate RUSTSEC-2021-0073 | ⏳ open |
| P2.15 | V1-32 | Word-boundary `\u{2581}` as literal vs escape (hidden coupling) | ⏳ open |
| P2.16 | V1-34 | `supported_rates` redeclared in two modules | ⏳ open |
| P2.17 | V1-49 | Shutdown oneshot swallows `Err(_)` silently (debug UX) | ⏳ open |
| P2.18 | V1-45 | `health` handler touches state for no reason | ⏳ open |
| P2.19 | V1-44 | `MelSpectrogram::new()` without `Default` (clippy complaint) | ⏳ open |
| P2.20 | V1-39 | Empty `scripts/` directory — either populate or delete | ⏳ open |

### Sustainability (ordered most → least critical)

| Rank | # | Work | Status |
|------|---|------|--------|
| S.1 | SUS-01 | `SECURITY.md` (disclosure contact, supported versions) — mandatory for crates.io | ⏳ open |
| S.2 | SUS-04 | Dependabot / Renovate weekly (enables the `prost` / `ort-rc` upgrades of V1-10) | ⏳ open |
| S.3 | SUS-03 | Signed releases (GPG / minisign) + publish `SHA256SUMS.asc` (MITM vector today) | ✅ done (v0.9.0) |
| S.4 | SUS-14 | `cargo-semver-checks` in CI (protects against accidental breaking changes) | ⏳ open |
| S.5 | SUS-13 | `docs/privacy.md` — make the privacy-first claim auditable | ⏳ open |
| S.6 | SUS-02 | CycloneDX SBOM in release workflow | ✅ done (v0.9.0) |
| S.7 | SUS-05 | SLSA build provenance attestations in release workflow | ✅ done (v0.9.0) |
| S.8 | SUS-10 | `docs/runbook.md` (pool exhaustion, model-download failure, OOM) | ⏳ open |
| S.9 | SUS-11 | `terminationGracePeriodSeconds` + shutdown guide in `deployment.md` | ⏳ open |
| S.10 | SUS-09 | Grafana dashboard JSON + `alerts.yml` examples | ⏳ open |
| S.11 | SUS-06 | `cargo-fuzz` harness for WAV header, WS binary frame, ONNX protobuf | ⏳ open |
| S.12 | SUS-08 | Coverage gate (`tarpaulin` or `grcov`) | ⏳ open |
| S.13 | SUS-07 | Miri + ASAN / TSAN nightly | ⏳ open |
| S.14 | SUS-12 | OpenAPI spec for REST (complements AsyncAPI) | ⏳ open |

### Documentation

| Rank | # | Work | Status |
|------|---|------|--------|
| D.1 | DOC-01 | Sync `specs/todo.md` ↔ `CHANGELOG.md` (addressed alongside this plan) | 🔨 in progress |

---

## P0 — blockers before any external exposure

### V1-01 — Model download: SHA256 verify after full-file write (TOCTOU)
- **File:** `src/model/mod.rs:186-239`
- **Symptom:** File is streamed to its final path, then SHA256 is computed;
  on mismatch `remove_file` is called. A parallel process can open the
  unverified file, a crash between write and verify leaves a file that
  `model_files_exist()` (`src/model/mod.rs:182-184`) later accepts.
- **Fix:** download to `<target>.partial`, finalise SHA256 on the partial
  file, then `rename` to final path (atomic on same fs). Re-verify every
  file on `Engine::load`. Record absolute path + hash in a small
  `.model-manifest.json` so restart cannot be fooled by partial writes.
- **Effort:** M
- **Acceptance:** new unit test: `download` writes `<file>.partial`,
  simulated crash between write and rename leaves no `v3_e2e_rnnt_*.onnx`
  visible to `model_files_exist`; tampered partial file fails SHA
  verification without leaving a final artefact.

### V1-02 — Speaker model has no SHA256 verification
- **File:** `src/model/mod.rs:136-180` (under `feature = "diarization"`)
- **Symptom:** `download_speaker_model` streams the file and returns
  without any integrity check; no hash constant in `MODEL_CHECKSUMS`.
- **Fix:** add `SPEAKER_MODEL_SHA256` constant; verify with the same
  tmp-file pattern as V1-01. Extend integration tests in the
  `diarization` CI job.
- **Effort:** S
- **Acceptance:** corrupted download is rejected; constant documented in
  `README`/`CHANGELOG`.

### V1-03 — Graceful shutdown does not drain WebSocket sessions [C]
- **File:** `src/server/mod.rs:331-350` (shutdown path),
  `src/server/mod.rs:745-789` (ws loop),
  `tests/e2e_shutdown.rs` (currently CI-excluded — smoking gun)
- **Symptom:** axum's `with_graceful_shutdown` tracks HTTP task
  completion, but `handle_ws_inner` runs as an untracked background task
  that only terminates on client disconnect. On SIGTERM → the k8s
  `terminationGracePeriodSeconds` (default 30s) → SIGKILL path loses the
  final `Final` message for every in-flight session.
- **Fix:** thread a `tokio_util::sync::CancellationToken` into
  `handle_ws_inner`; use `tokio::select!` on
  `cancel_token.cancelled()` vs. `source.next()`. On cancel: flush
  buffered audio, emit `Final`, send `Close(CloseFrame{code:1001,
  reason:"server shutdown"})`, return triplet to pool, exit within a
  configurable `shutdown_drain_secs` (default 10). Re-enable
  `tests/e2e_shutdown.rs` in CI. Add HTTP handler cancel handling where
  streaming (SSE) is involved.
- **Effort:** M
- **Acceptance:** `e2e_shutdown` passes in CI; integration test that
  fires a client stream, sends SIGTERM, asserts `Final` arrives before
  close and pool capacity is preserved.

### V1-04 — Idle timeout resets on every frame; no max-session cap
- **File:** `src/server/mod.rs:780-789`
- **Symptom:** `timeout(idle_timeout, source.next())` resets on any WS
  message, including 20 ms silence frames. A looped sender keeps a
  triplet pinned forever; 5 such clients starve the pool.
- **Fix:** track `session_started_at: Instant` inside the handler; add
  `max_session_secs` (default 3600) and reject/close once exceeded
  with error code `max_session_duration_exceeded`. Additionally, use
  the already-computed `audio_received` flag to cap sessions that
  never produce non-trivial audio.
- **Effort:** S
- **Acceptance:** e2e test that streams silence-only frames and is
  closed after the cap; idle timeout behaviour for mixed traffic is
  unchanged.

### V1-05 — REST `body.to_vec()` doubles memory + full-file decode [C]
- **File:** `src/server/http.rs:163-184` (buffer clone),
  `src/inference/audio.rs:58, 91` (symphonia decode)
- **Symptom:** `body: Bytes` (50 MiB cap) is cloned into `Vec<u8>` then
  decoded to `Vec<f32>`. Peak: `body_limit × 2 + decoded_pcm` per
  request; 4 concurrent uploads ≈ 515 MiB transient. OOM on 4 GB boxes.
- **Fix:** (a) drop the `to_vec()` — pass `Bytes` directly into
  `transcribe_bytes`; symphonia already receives a `Cursor<Vec<u8>>`,
  wrap `Bytes` in a `std::io::Read` adapter; (b) add an
  `axum::body::Body` streaming path for `/v1/transcribe/stream` that
  feeds `MediaSourceStream` chunk-by-chunk; (c) reject when
  `Content-Length > body_limit_bytes` before reading.
- **Effort:** M
- **Acceptance:** memory benchmark shows ≤ 1.2× body size peak per
  concurrent request; new unit test that feeds a streaming body without
  OOM on a small heap.

### V1-06 — Rate-limiter per-second math truncates sub-60 rpm to 1 rps
- **File:** `src/server/mod.rs:278`
  (`let per_second = (config.limits.rate_limit_per_minute / 60).max(1);`)
- **Symptom:** any `rate_limit_per_minute ∈ [1,59]` produces `per_second = 1`
  (60 rpm). Documented DoS protection is 60× weaker than declared; a
  defender setting `--rate-limit-per-minute 10` actually gets 60 rpm.
- **Fix:** switch to
  `GovernorConfigBuilder::per_millisecond(60_000 / rpm.max(1))` so that
  any rate resolves precisely. Add `assert!(rpm >= 1)` at config parse.
  Update `docs/deployment.md` and the `--help` text.
- **Effort:** S
- **Acceptance:** unit test constructs a layer for rpm ∈ {1, 10, 30,
  60, 600} and verifies observed rate within ±5 %.

### V1-07 — Pool `Mutex<mpsc::Receiver>` + `.expect("Pool sender dropped")`
- **File:** `src/inference/mod.rs:89, 96-106, 109-115`
- **Symptom:** unfair under saturation (lock ordering not FIFO) and a
  `sender.drop()` during shutdown panics every waiting client, turning
  503s into 500 cascades.
- **Fix:** replace with a semaphore-plus-vec pattern *or*
  `async_channel` (lock-free, fair, `close()` returns `Err`). On
  `recv()` failure return `503 pool_closed`. Emit a `pool_closed`
  error in the WS error path.
- **Effort:** M
- **Acceptance:** stress test with `--ignored` that holds 10 × pool_size
  concurrent transcriptions and shows wait ordering within FIFO
  tolerance; shutdown during checkout returns 503 without panic logs.

### V1-08 — `Engine::create_state` changes signature on feature toggle
- **File:** `src/inference/mod.rs:477-480`, `src/lib.rs:17` (doctest)
- **Symptom:** `#[cfg(feature = "diarization")] diarization_enabled: bool`
  makes the same API break between feature sets. `src/lib.rs` doctest
  fails to compile without the `diarization` feature.
- **Fix:** always accept `diarization_enabled: bool`. Ignore when the
  feature is disabled (or return `Err(GigasttError::FeatureDisabled)`).
  Mark this explicitly in the doc comment.
- **Effort:** S
- **Acceptance:** `cargo test --doc` passes with and without
  `--features diarization`.

### V1-09 — Nightly soak / load tests missing from CI
- **File:** `.github/workflows/ci.yml`, `tests/soak_test.rs`,
  `tests/load_test.rs`
- **Symptom:** both suites are `#[ignore]` and never run in CI. Known
  classes of defects (descriptor leaks, pool drift, memory growth)
  would only surface in production.
- **Fix:** add `.github/workflows/soak.yml` that runs on `schedule`
  (nightly) and on-demand. Caches the model, runs `soak_test` with
  `GIGASTT_SOAK_DURATION_SECS=300`, uploads metrics artefacts. Use a
  self-hosted runner if GitHub runners cannot keep up; until then,
  run on `ubuntu-latest` with reduced duration.
- **Effort:** M
- **Acceptance:** workflow produces an artefact; `soak_test` passes;
  failures open an issue automatically via `peter-evans/create-issue-from-file`.

### V1-10 — `prost 0.6` + `onnx-pb 0.1.4` supply-chain debt
- **File:** `Cargo.toml:67-68`, `src/quantize.rs:30`
- **Symptom:** `prost 0.6` is 3+ years unmaintained; combined with
  `RUSTSEC-2021-0073` (currently ignored in `deny.toml`), parsing
  untrusted ONNX protobuf becomes a risk. `onnx-pb 0.1.4` transitively
  pulls a build-time `protoc` requirement that degrades
  `cargo install gigastt --features quantize`.
- **Fix (preferred):** replace `src/quantize.rs` with ORT's native
  quantization API (QDQ) when available, or emit a one-line warning
  plus a `scripts/quantize.py` reference. Alternative: upgrade
  `onnx-pb` → `0.9`+ (requires `prost 0.11+`), bound `num_elements`
  casts in `src/quantize.rs:57-58` to avoid integer overflow.
- **Effort:** L (if rewriting) / M (if upgrading)
- **Acceptance:** `cargo deny check advisories` passes without
  `RUSTSEC-2021-0073` ignore, OR the ignore block in `deny.toml:4-10`
  contains a unit test that asserts the vulnerable path is never
  reachable.

---

## P1 — ship-before-v1.0

### V1-11 — Proxy docs allow `X-Forwarded-For` spoofing
- **File:** `docs/deployment.md:19-90`
- **Symptom:** nginx recipe uses `$proxy_add_x_forwarded_for` (appends
  client-supplied header), Caddy recipe omits the header entirely.
  `tower_governor::SmartIpKeyExtractor` trusts the header when present,
  so per-IP rate-limiting is bypassable.
- **Fix:** nginx → `proxy_set_header X-Forwarded-For $remote_addr;`
  (overwrite). Caddy → `header_up X-Forwarded-For {remote_host}`.
  Add a CLI flag `--trust-forwarded=false` (default) so only operators
  who explicitly enable it trust the header. Document the threat model
  in `docs/deployment.md`.
- **Effort:** S
- **Acceptance:** docs PR reviewed; unit test that when
  `trust_forwarded=false`, the key extractor ignores
  `X-Forwarded-For` and uses `ConnectInfo`.

### V1-12 — `/metrics` sits behind CORS/rate-limit and leaks telemetry
- **File:** `src/server/http.rs:30-50`, `src/server/mod.rs:264-275`
- **Symptom:** (a) a Prometheus scraper at 15s interval with default
  60 rpm is throttled; (b) any browser origin on the allowlist can
  `fetch('/metrics')` and read request histograms.
- **Fix:** add `--metrics-listen 127.0.0.1:9100` (secondary
  listener) matching Prometheus convention. Exclude `/metrics` from
  the primary rate-limiter and origin allowlist; keep it
  loopback-only by default.
- **Effort:** M
- **Acceptance:** two-port integration test; docs updated; e2e test
  that a non-allowlisted origin cannot fetch `/metrics` on the
  primary port.

### V1-13 — WebSocket protocol version is declarative, not negotiated
- **File:** `src/protocol/mod.rs:6`, `src/server/mod.rs:760-766`
- **Symptom:** `ServerMessage::Ready { version: "1.0" }` is informational.
  `ClientMessage::Configure` has no `client_supported_versions`; a
  future `2.0` breaks silently.
- **Fix:** add optional `supported_protocol_versions: Vec<String>` to
  `Configure`; server replies with `Ready { version, negotiated_version }`
  (or error `unsupported_protocol_version`). Alternative or additional:
  use `Sec-WebSocket-Protocol` subprotocol negotiation at HTTP upgrade.
- **Effort:** M
- **Acceptance:** e2e negotiation test for matching and mismatching
  versions; AsyncAPI schema updated.

### V1-14 — `/ws` deprecation lacks client-visible hint
- **File:** `src/server/mod.rs:460-473`
- **Symptom:** legacy handler merely logs `warn`. Clients that still
  hit `/ws` after the 1.0 drop receive a silent 404 with no migration
  guidance.
- **Fix:** during upgrade, include `Deprecation: true` and `Sunset: <RFC
  8594 date>` response headers. Send a `ServerMessage::Ready` that
  includes `deprecation: { path: "/ws", replacement: "/v1/ws",
  sunset: "2026-XX-XX" }`. Log once per minute instead of on every
  connect.
- **Effort:** S
- **Acceptance:** e2e test asserts the `Deprecation` header and
  `Ready.deprecation` payload.

### V1-15 — Rate-limiter `retain-recent` thread leaks on shutdown
- **File:** `src/server/mod.rs:290-295`
- **Symptom:** `std::thread::spawn` with `loop { sleep(60); ... }`
  never exits. In tests that restart the server threads accumulate;
  in production, SIGTERM waits up to 60 s on the thread.
- **Fix:** replace with a `tokio::spawn` that `tokio::select!`s on
  either a shutdown `oneshot::Receiver` or `tokio::time::interval`.
  Tie the shutdown receiver to the top-level cancellation token used
  in V1-03.
- **Effort:** S
- **Acceptance:** restart benchmark shows no thread growth across
  10 cycles; SIGTERM drops the reaper within 1 s.

### V1-16 — `thread::scope` panic during pool load aborts the process
- **File:** `src/inference/mod.rs:412-434`
- **Symptom:** `.expect("Thread panicked during model loading")` on
  `pool_size = 4` with a 4 GB RAM box — OOM in one thread kills all.
- **Fix:** collect per-thread `Result<SessionTriplet, anyhow::Error>`;
  if fewer than `pool_size` succeed but at least one did, emit a
  `warn!("degraded pool: loaded N/M triplets")` and continue. If all
  fail, return an error. Add CLI flag
  `--pool-min-size=1` to require a minimum; default to the lower
  bound.
- **Effort:** S
- **Acceptance:** unit test using a mocked loader that fails on the
  second invocation; server boots with one triplet.

### V1-17 — `PrometheusBuilder::install_recorder` is a global
- **File:** `src/server/mod.rs:232`
- **Symptom:** in-process tests force `--test-threads=1`; second
  install returns a bare warn but leaves `/metrics` serving stale
  data.
- **Fix:** switch to
  `PrometheusBuilder::build_recorder()` → `LocalRecorderGuard` scoped
  to the server handle (using `metrics::with_local_recorder`). Each
  `run_with_config` invocation owns its own recorder. Drop
  `--test-threads=1` from `ci.yml`.
- **Effort:** M
- **Acceptance:** parallel e2e run passes; CI wall-clock halved on
  `ubuntu-latest`.

### V1-18 — Decoder-loop allocations in hot path
- **File:** `src/inference/decode.rs:107-109, 133`;
  `src/inference/mod.rs:747`
- **Symptom:** per encoder frame: `dec_data.to_vec()`,
  `new_h_data.to_vec()`, `new_c_data.to_vec()`, joiner `logits.to_vec()`.
  For a 40 s audio: ≈ 1000 × 3 × Vec allocs (decoder) + ≈ 1000–3000
  joiner Vec<f32, len=1025>.
- **Fix:** introduce `DecoderBuffers { dec: Vec<f32>, h: Vec<f32>, c:
  Vec<f32>, logits: Vec<f32> }` owned by `DecoderState`; replace
  `to_vec()` with `copy_from_slice` into the pre-allocated buffers.
- **Effort:** M
- **Acceptance:** criterion micro-bench shows ≥ 20 % fewer allocations
  in `greedy_decode`; no regression in `tests/benchmark.rs`.

### V1-19 — `SincFixedIn::new(chunk_size = samples.len())` per call
- **File:** `src/inference/audio.rs:185-192`
- **Symptom:** for streaming WS, each chunk rebuilds the FIR kernel;
  for a 10-minute file the resampler allocates for the full buffer.
- **Fix:** introduce `StreamingResampler` that is created once in
  `StreamingState` when `sample_rate != 16000`; use `FftFixedIn` or
  `FastFixedIn` with a fixed chunk size and drain remainder across
  frames. For file path, keep `SincFixedIn` but chunk the input.
- **Effort:** M
- **Acceptance:** unit test verifies bit-identical output between
  one-shot and streaming resample on a 1-minute file; latency micro-
  bench shows ≥ 3× lower per-frame cost.

### V1-20 — `quantize.rs` forces `axis = 0` on every op
- **File:** `src/quantize.rs:90, 166`
- **Symptom:** for MatMul / Gemm with transposed weights the correct
  per-channel axis is 1; using 0 produces silent WER regressions.
- **Fix:** inspect `op_type` and `attribute` (e.g. `transB`) to pick
  the axis. Fall back to `onnxruntime.quantization` via
  `scripts/quantize.py` when the op isn't a simple Conv.
- **Effort:** M
- **Acceptance:** WER on Golos before / after quantization matches
  within 1 % absolute; regression fixture committed under
  `tests/fixtures/quantize/`.

### V1-21 — `SessionPool` lacks a Drop-guard
- **File:** `src/inference/mod.rs:70-134`
- **Symptom:** if a caller panics between `checkout()` and
  `checkin()`, the triplet is lost forever. `spawn_blocking` closures
  recover via `catch_unwind`, but that's opt-in per call site.
- **Fix:** return `PoolGuard<'a>(Option<SessionTriplet>)`. `Deref` to
  the triplet; `Drop` returns it on panic unwind. Collapse the
  existing `checkin` / `blocking_checkin` into the guard.
- **Effort:** M
- **Acceptance:** unit test where a handler panics inside the guard's
  scope and asserts pool capacity is preserved.

### V1-22 — `test_rest_oversized_body_rejected` asserts `!= 200`
- **File:** `tests/e2e_errors.rs:37-41`
- **Symptom:** accepts 500 as success; masks regressions.
- **Fix:** `assert_eq!(response.status().as_u16(), 413);` and assert
  the body contains `{"code":"payload_too_large"}`.
- **Effort:** S
- **Acceptance:** intentional-regression branch (temporarily return
  500) fails the test.

### V1-23 — Remove `tests/server_integration.rs`
- **File:** `tests/server_integration.rs` (6 tests)
- **Symptom:** legacy duplicates of `e2e_ws.rs`; uses `sleep(200ms)`
  races; not run in CI; drifts from production behaviour.
- **Fix:** delete the file; port any unique assertions into `e2e_*.rs`;
  remove the entry from `CLAUDE.md` testing docs.
- **Effort:** S
- **Acceptance:** `cargo test` still covers every scenario; CI matrix
  unchanged.

### V1-24 — Separate batch-REST pool from streaming WS pool
- **File:** `src/inference/mod.rs:70-134` (design),
  `src/server/http.rs:188-237` (long-running batch path)
- **Symptom:** a 9-minute REST job holds a triplet for its entire
  duration; pool_size=4 → four simultaneous batch jobs starve every
  real-time WS client for 30 s until timeout.
- **Fix:** introduce two sub-pools (`batch_pool_size=1`,
  `stream_pool_size=pool_size-1`) or a priority channel. Route REST
  `/v1/transcribe` → batch; WS + SSE → stream. CLI flags:
  `--batch-pool-size`, `--stream-pool-size`.
- **Effort:** M
- **Acceptance:** integration test that dispatches a 60 s REST job
  and a WS session in parallel; WS session receives its first Partial
  within 1 s.

### V1-25 — Odd-length PCM frame silently drops last byte
- **File:** `src/server/mod.rs:548-557`
- **Symptom:** `chunks_exact(2)` discards the trailing byte; the next
  frame begins 1-sample-shifted from the intended stream, creating a
  silent phase artefact.
- **Fix:** carry a `pending_byte: Option<u8>` in `StreamingState`;
  prepend to the next frame. Alternative: reject with
  `{"code":"invalid_audio_alignment"}`.
- **Effort:** S
- **Acceptance:** unit test that feeds `[3, 5, 7]` + `[11, 13]` twice
  and verifies sample sequence matches `[3, 5], [7, 11], [13]`.

### V1-26 — `Engine` mixes file/streaming/formatting concerns
- **File:** `src/inference/mod.rs:233-855`
- **Symptom:** `transcribe_file`, `process_chunk`, `tokens_to_words`
  belong to different layers; cannot be unit-tested without model.
- **Fix:** extract `TokenFormatter::tokens_to_words`; split
  `FileTranscriber { engine: &Engine }` from `Engine` core (pool +
  single-inference primitive). Streaming remains on `Engine` for now.
- **Effort:** M
- **Acceptance:** `TokenFormatter` gains unit tests without model;
  public API unchanged (old methods re-export as shims for the 0.9
  cycle, then removed in 1.0).

### V1-27 — `/health` does not probe the encoder
- **File:** `src/server/http.rs:118-125`
- **Symptom:** returns 200 even after encoder session has failed.
  k8s livenessProbe gives a false negative.
- **Fix:** run a cached no-op inference (empty 1-sample buffer) on
  `/health/ready`; keep `/health` as liveness-only. Differentiate:
  `/healthz` (liveness, always 200 while process alive),
  `/readyz` (readiness, 503 if pool exhausted or session unhealthy).
- **Effort:** M
- **Acceptance:** shutdown test asserts `/readyz` → 503 before the
  process exits; liveness stays green until process exit.

### V1-28 — Pool checkout timeout hardcoded in two places
- **File:** `src/server/mod.rs:477-478, 39-40`,
  `src/server/http.rs:176-181, 276-281`
- **Symptom:** 30 s literal + comment "matches Retry-After" — but
  Retry-After is read from a different constant.
- **Fix:** add `pool_checkout_timeout_secs` to `RuntimeLimits`; thread
  into both REST and WS paths; source `Retry-After` from the same
  field. Env: `GIGASTT_POOL_CHECKOUT_TIMEOUT_SECS`.
- **Effort:** S
- **Acceptance:** integration test sets the value to 1 s and observes
  the corresponding 429/WS error within 1.5 s.

### V1-29 — `idle_timeout` test uses 310 s wall-clock
- **File:** `tests/e2e_errors.rs:249`
- **Symptom:** runs for 5+ minutes in CI main-push path.
- **Fix:** depend on V1-28 (or a sibling limit) and run the test with
  a 3 s idle timeout; move the long-duration test into a separate
  `#[ignore]` soak scenario.
- **Effort:** S
- **Acceptance:** `e2e_errors` wall-clock < 30 s in CI.

### V1-30 — Missing pool / inference / WS Prometheus metrics
- **File:** `src/server/mod.rs` (metrics registry),
  `src/inference/mod.rs:70-134` (pool counters)
- **Symptom:** `/metrics` exposes RED (requests/errors/duration) but
  not pool depth, inference stage timings, WS session counts, or
  rate-limit rejections.
- **Fix:** introduce gauges & histograms:
  - `gigastt_pool_available{pool="batch|stream"}`
  - `gigastt_pool_checkout_seconds` (histogram)
  - `gigastt_pool_timeout_total`
  - `gigastt_inference_seconds{stage="encoder|decoder|joiner"}`
  - `gigastt_inference_panics_total`
  - `gigastt_ws_active_connections`
  - `gigastt_rate_limit_rejections_total{path}`
  - `gigastt_audio_buffer_bytes` (histogram)
- **Effort:** M
- **Acceptance:** each metric visible in a scrape; alert-rule
  examples shipped under `docs/observability/alerts.yml`.

---

## P2 — polish for v1.x

### V1-31 — `consecutive_blanks` incremented on MAX_TOKENS overrun
- **File:** `src/inference/decode.rs:213-222`
- **Fix:** introduce `stuck_frames` counter for the overrun case; keep
  `consecutive_blanks` for actual blank emissions.
- **Effort:** S

### V1-32 — Word-boundary char written two ways
- **File:** `src/inference/mod.rs:804, 826`; `src/inference/tokenizer.rs:81`
- **Fix:** `const WORD_BOUNDARY: char = '\u{2581}';` re-exported from
  `tokenizer.rs`.
- **Effort:** S

### V1-33 — AsyncAPI spec out of sync
- **File:** `docs/asyncapi.yaml`
- **Symptoms:** `/ws` still canonical; "512 KiB ≈ 16 s" math is wrong
  (≈ 5.5 s at 48 kHz); `WordInfo` misses `confidence` / `speaker`;
  error enum misses `timeout`, `inference_panic`; rates enum misses
  24 kHz / 44.1 kHz where applicable.
- **Fix:** single-file update + AsyncAPI linter in CI
  (`asyncapi/github-action-for-cli`).
- **Effort:** S

### V1-34 — `supported_rates` defined twice
- **File:** `src/server/http.rs:145` vs. `src/server/mod.rs:32`
- **Fix:** reference the `SUPPORTED_RATES` constant.
- **Effort:** S

### V1-35 — SSE emits a single error code
- **File:** `src/server/http.rs:343`
- **Fix:** translate `GigasttError` variants into distinct
  `event: error / data: {code, message}` payloads, matching the WS
  contract.
- **Effort:** S

### V1-36 — Prometheus `path` label = raw URI
- **File:** `src/server/mod.rs:364`
- **Symptom:** scanners (e.g. `/wp-login.php`) create new label
  cardinality.
- **Fix:** match against a known-path set; label with
  `"other"` for unmatched; or use a bounded label set.
- **Effort:** S

### V1-37 — No server-side WebSocket ping
- **File:** `src/server/mod.rs` (no ping timer)
- **Fix:** `tokio::time::interval(Duration::from_secs(30))` sending
  `Message::Ping(Vec::new())`; disconnect if no `Pong` in
  `2 * interval`.
- **Effort:** S

### V1-38 — `audit.toml` and `deny.toml` duplicate `RUSTSEC-2021-0073`
- **File:** `.cargo/audit.toml`, `deny.toml:10`
- **Fix:** keep the ignore in `deny.toml` only; delete
  `.cargo/audit.toml` (CI uses `cargo deny`).
- **Effort:** S

### V1-39 — Empty `scripts/` directory
- **File:** `scripts/`
- **Fix:** either delete or populate with documented helpers (e.g.
  `scripts/dev-setup.sh`, `scripts/release.sh`).
- **Effort:** S

### V1-40 — Pin tokio/serde etc. to minor versions
- **File:** `Cargo.toml`
- **Fix:** bump to `1.40` / latest minor, add `cargo update --dry-run`
  check in CI.
- **Effort:** S

### V1-41 — Benchmark without a baseline gate
- **File:** `tests/benchmark.rs:282-371`
- **Fix:** commit `tests/benchmark_baseline.json`; assert WER does
  not regress more than 0.5 % vs. baseline. Print a diff table for
  PRs.
- **Effort:** S

### V1-42 — Only 15 Golos fixtures
- **File:** `tests/fixtures/manifest.json`
- **Fix:** commit 100+ samples (Git LFS), documented provenance &
  licensing.
- **Effort:** M (licensing dominates)

### V1-43 — `now_timestamp()` uses `unwrap_or_default()`
- **File:** `src/inference/mod.rs:43-48`
- **Fix:** compute durations relative to a process-start `Instant`
  (monotonic) for all wire-visible timestamps.
- **Effort:** S

### V1-44 — `MelSpectrogram::new()` without `Default`
- **File:** `src/inference/features.rs:17`
- **Fix:** derive / implement `Default`.
- **Effort:** S

### V1-45 — `health` touches state for no reason
- **File:** `src/server/http.rs:118-125`
- **Fix:** remove the `let _ = &state.engine;` line once V1-27 lands.
- **Effort:** S

### V1-46 — No Engine::warmup
- **File:** `src/inference/mod.rs`
- **Fix:** expose `warmup()` that runs a dummy 1-second inference;
  call it from `run_with_config` before `axum::serve`. Removes the
  first-request cold start.
- **Effort:** S

### V1-47 — No ORT inference timeout
- **File:** `src/inference/mod.rs` (run calls)
- **Fix:** wrap `spawn_blocking` with `tokio::time::timeout`
  (configurable, default 60 s). Return a typed error on expiry.
- **Effort:** S

### V1-48 — VAD-based endpointing
- **File:** `src/inference/decode.rs` (endpoint logic)
- **Fix:** optional feature using WebRTC VAD or Silero VAD to drive
  endpointing when blanks don't arrive (noisy input).
- **Effort:** L

### V1-49 — Shutdown oneshot silently swallows errors
- **File:** `src/server/mod.rs:333-343`
- **Fix:** `match rx.await { Ok(()) => ..., Err(_) => warn!(...) }`.
- **Effort:** S

### V1-50 — Multi-model support via manifest
- **File:** `src/inference/mod.rs:281-360`
- **Fix:** read model filenames from `<model_dir>/manifest.toml`
  (with sane defaults). Enables GigaAM v4 without a code change.
- **Effort:** M

---

## Sustainability (not feature work; ship under 1.0 umbrella)

| ID | Work | Effort | Notes |
|----|------|--------|-------|
| SUS-01 | `SECURITY.md` (contact, supported versions, disclosure policy) | S | Required for crates.io hygiene |
| SUS-02 | CycloneDX SBOM generation in release workflow | S | `cargo-cyclonedx` / `anchore/sbom-action` |
| SUS-03 | Sign releases (minisign or gpg); publish `SHA256SUMS.asc` | S | MITM vector today |
| SUS-04 | `.github/dependabot.yml` weekly for `cargo` + `github-actions` | S | `ort=rc.12`, `prost 0.6` make this urgent |
| SUS-05 | `actions/attest-build-provenance@v2` SLSA attestations | S | `softprops/action-gh-release` supports it |
| SUS-06 | `cargo-fuzz` harness for WAV header, WS binary frame, ONNX | M | Initial `cargo fuzz run` targets |
| SUS-07 | Miri + ASAN/TSAN nightly job | M | Only for `cargo test` subset |
| SUS-08 | `tarpaulin` or `grcov` coverage gate | S | Upload to codecov |
| SUS-09 | Grafana dashboard JSON + `alerts.yml` | S | Exports live under `docs/observability/` |
| SUS-10 | `docs/runbook.md` (pool exhaustion, model-download failure, OOM) | S | |
| SUS-11 | Document `terminationGracePeriodSeconds` in k8s section of `deployment.md` | S | |
| SUS-12 | `docs/openapi.yaml` for REST | M | |
| SUS-13 | `docs/privacy.md` (no audio log, no transcript persistence) | S | |
| SUS-14 | `cargo-semver-checks` action in CI | S | |

---

## Doc sync

### DOC-01 — `specs/todo.md` is out of sync with `CHANGELOG.md`
- Items 6, 7, 10, 11, 12, 14, 17, 20 are marked `⏳ open` in
  `specs/todo.md` but shipped in v0.7.0–v0.8.0 (see CHANGELOG).
- **Fix:** flip those rows to `✅`; add a pointer row to this file
  for outstanding work. Already addressed in the concurrent edit of
  `specs/todo.md`.
- **Effort:** S

---

## Acceptance for v1.0

A tag is only cut when:

1. Every P0 item above is closed (verified by linked test evidence).
2. At least 80 % of P1 items are closed; remainder recorded as known
   issues in `CHANGELOG.md` and carried into v1.1.
3. `SECURITY.md`, `docs/runbook.md`, `docs/privacy.md`,
   `docs/observability/` exist.
4. Nightly soak has run ≥ 14 consecutive days without a regression.
5. Benchmark WER regression gate is green on main.
6. `cargo deny check advisories` has no ignored entries (V1-10) OR
   every ignore carries a gated unit test proving the vulnerable
   path is unreachable.

## Parallelisation

Within P0:

- V1-01, V1-02 share `src/model/mod.rs` → do sequentially.
- V1-03, V1-04 share `handle_ws_inner` → do in the same PR.
- V1-05 is independent (REST path).
- V1-06, V1-07 both touch `src/server/mod.rs` but different sections;
  can parallelise across two executors with a merge gate.
- V1-08 is a signature change; independent.
- V1-09 is CI-only; independent.
- V1-10 is supply-chain; independent from runtime work.

Within P1: see per-item anchors; most items touch a single file.
