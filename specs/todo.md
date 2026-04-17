# gigastt — critique follow-ups (TODO)

Outstanding issues from the v0.5.0 critique. Items already resolved
(native Rust quantization, Python script removal, client examples
trimmed to Go/Python/Kotlin/Bun) are intentionally excluded.

Each item: **P0/P1/P2** priority, a short problem statement, the
concrete symptom, and the proposed direction. Full rollout sequence
lives in `specs/plan.md`.

## Progress snapshot (2026-04-17)

| Item | Priority | Status |
|------|----------|--------|
| 1. Release pipeline | P0 | ✅ v0.5.2 (`release.yml` matrix, SHA256SUMS) |
| 2. Semver violation | P0 | ✅ v0.5.2 (CONTRIBUTING release checklist) |
| 3. Pool depletion on panic | P0 | ✅ v0.5.1 (`catch_unwind` in WS handler) |
| 4. CORS `*` + weak Origin check | P1 | ✅ v0.6.0 (origin_middleware) |
| 5. Pool timeout without Retry-After | P1 | ✅ v0.6.0 (header + retry_after_ms) |
| 6. Hard-coded runtime limits | P1 | ⏳ open |
| 7. `/metrics` / observability | P1 | ⏳ open |
| 8. Origin-check covers REST | P1 | ✅ v0.6.0 (middleware before routing) |
| 9. `--bind-all` guard | P1 | ✅ v0.6.0 (CLI + Dockerfiles) |
| 10. Docker bake-model option | P1 | ⏳ open |
| 11. `/v1/ws` canonical path | P2 | ⏳ open |
| 12. `/v1/models.capabilities` | P2 | ⏳ open |
| 13. `handle_ws_inner` split | P2 | ✅ v0.6.1 (three frame handlers + e2e test) |
| 14. `cargo deny` + SBOM | P2 | ⏳ open |
| 15. WER histogram breakdown | P2 | ⏳ open |
| 16. Self-hosted nightly soak | P2 | ⏳ open |
| 17. Per-IP rate limiting | P2 | ⏳ open |
| 18. `ort_err()` wrapper audit | P2 | ⏳ open |
| 19. Hot-reload model | P2 | ⏳ open |
| 20. TLS/auth deployment docs | P2 | ⏳ open |
| CUDA in release matrix | P0 addendum | ⏳ open (removed from matrix v0.5.2+) |

Also shipped alongside (2026-04-14 advisory): `rustls-webpki` 0.103.10→0.103.12 closing RUSTSEC-2026-0098/99 in v0.5.3.

---

## P0 — production-correctness blockers

### 1. Release pipeline absent (`.github/workflows/release.yml`)
- Only `ci.yml` exists. Tags are cut manually; assets don't build.
- Already bit us: `v0.5.0` tag had no tarball → Murmur SHA-pinned
  download returned 404. Temporary fix: manual `gh release upload`.
- Fix: tag-triggered matrix workflow producing
  `gigastt-<ver>-aarch64-apple-darwin.tar.gz`,
  `gigastt-<ver>-x86_64-unknown-linux-gnu.tar.gz` (cpu + cuda),
  plus `SHA256SUMS.txt`. Upload with `softprops/action-gh-release`.

### 2. Version-to-artifact semver violation
- `v0.5.0` exists on crates.io AND as a tag, but for ~weeks the
  tag had no binaries. Two artifacts under one version name.
- Fix: release workflow (item 1) must run BEFORE `cargo publish`.
  Add a release checklist to `CONTRIBUTING.md`.

### 3. WebSocket pool silently depletes on `spawn_blocking` panic
- `src/server/mod.rs:315` — on blocking-task panic, triplet is lost,
  `pool capacity reduced`. No auto-refill, no alert.
- Under repeated inference panics the pool drifts to 0 → all new
  clients hit `checkout` timeout → generic `Server busy`.
- Fix: either (a) restructure closure ownership so the triplet is
  recoverable (pattern already used by SSE handler in `http.rs`),
  or (b) detect depletion and rebuild a fresh `SessionTriplet` in
  a background supervisor task. Option (a) is strictly better.

---

## P1 — ship-before-v1

### 4. CORS `*` + permissive WebSocket origin check
- `src/server/mod.rs:101-117` always emits
  `Access-Control-Allow-Origin: *`.
- `ws_handler` only *warns* on non-local Origin
  (`src/server/mod.rs:125-131`) — does not deny.
- Exposure: any webpage a user visits can open
  `ws://127.0.0.1:9876/ws` and stream microphone audio from a
  compromised client. Privacy-first product claim is undermined.
- Fix: default deny Origin ∉ {`null`, `http(s)://localhost`,
  `http(s)://127.0.0.1`, any explicit `--allow-origin=…` entry
  from CLI/env). CORS `*` becomes opt-in via `--cors-allow-any`.

### 5. Pool checkout timeout → generic 503, no `Retry-After`
- Same location as (3). Client sees `{"type":"error","code":"timeout"}`
  but has no hint when to retry. REST/SSE handlers behave the same.
- Fix: on REST send HTTP 429 + `Retry-After: <seconds>`. On WS include
  `retry_after_ms: <u32>` in the error payload.

### 6. Hard-coded runtime limits (only `--pool-size` is configurable)
- `IDLE_TIMEOUT_SECS = 300`, audio buffer cap 5 s, file cap 10 min,
  WS frame limit 512 KB — all baked in.
- Fix: expose via CLI flags AND env (`GIGASTT_IDLE_TIMEOUT_SECS`,
  `GIGASTT_WS_FRAME_MAX_BYTES`, `GIGASTT_AUDIO_BUFFER_SECS`,
  `GIGASTT_FILE_MAX_MINUTES`). Also accept a TOML config file
  (`--config /etc/gigastt/config.toml`) for systemd/launchd.

### 7. No `/metrics` or structured observability
- `tracing` exists but no Prometheus exporter; no per-stage timings
  (mel, encoder, decoder, joiner); no queue depth gauge.
- First production regression will be debugged blind.
- Fix: add optional `metrics` feature (uses `metrics-exporter-prometheus`);
  expose `GET /metrics`. Instrument RED per endpoint + audio-specific
  histograms. Gate behind `--metrics` flag so single-user default
  install does not open an extra port.

### 8. Origin-check deny does not extend to REST
- `/v1/transcribe` and `/v1/transcribe/stream` also accept
  cross-origin if CORS allows (it does, `*`). Once a browser page has
  the WAV bytes it can upload them for transcription even though
  the user never granted that page server access.
- Fix: covered by (4) — Origin check at middleware level before
  route dispatch.

### 9. Default Docker binds `0.0.0.0` with no auth
- `Dockerfile`/`Dockerfile.cuda` use `--host 0.0.0.0`. Documented,
  but one stray port-forward = public transcription service.
- Fix: require explicit `--bind-all` flag (or env
  `GIGASTT_ALLOW_BIND_ANY=1`) before the server will listen on
  non-loopback addresses. Update Dockerfiles to set that env.

### 10. Docker image bakes no model, no runtime progress UX
- First `docker run` blocks for ~850 MB download with only tracing
  lines. For `compose up` this is a silent 2–5 min hang.
- Fix: optional build arg `GIGASTT_BAKE_MODEL=1` that triggers
  `gigastt download` during image build (produces a ~1.1 GB image
  but zero cold-start cost). Document both modes.

---

## P2 — v1.x hardening

### 11. `/ws` path is unversioned while REST is under `/v1/*`
- A breaking WS protocol change will have to negotiate via the
  `Ready` message (soft break) or add `/v2/ws` (hard break, two
  routers). Today neither is wired.
- Fix: introduce `/v1/ws` as the canonical path, keep `/ws` as an
  alias (deprecation log) for two minor versions.

### 12. `/v1/models` does not declare capabilities before WS handshake
- Client must connect WS to learn whether diarization is available
  (via `Ready.diarization`). Thin REST probe would be enough.
- Fix: extend `/v1/models` payload with
  `{"capabilities":{"diarization":false,"supported_rates":[…]}}`.
  Mirrors the WS `Ready` fields exactly.

### 13. `handle_ws_inner` is 240 lines of state+match
- `src/server/mod.rs:172-420` — `mut state_opt`, `mut triplet_opt`,
  `mut audio_received`, three control-flow layers.
- Fix: extract `handle_binary_frame`, `handle_configure`, `handle_stop`
  into free functions. Keeps the connection loop to ~60 lines and
  exposes the frame handlers to unit tests.

### 14. Supply-chain hygiene gaps
- `cargo audit --locked` is in CI; `cargo deny` is not.
- No SBOM (`cyclonedx` or `spdx`) in release artifacts.
- No `cargo-license` policy report.
- Fix: add `cargo deny check` to PR CI (licenses + advisories + bans).
  Generate `bom.cdx.json` in the release workflow (item 1).

### 15. Benchmark reports single WER number, no histograms
- README lists `10.4%` on 993 Golos samples. No distribution by
  utterance length, no noise-bucket breakdown, no per-speaker
  variance.
- Fix: emit `tests/benchmark.rs` output as JSON + markdown table with
  percentile slices. Commit the JSON; diff in PRs.

### 16. Load/soak tests are local-only
- `load_test.rs` and `soak_test.rs` are `#[ignore]` and never run
  in CI. Perf regressions only caught by manual runs.
- Fix: self-hosted runner (one M2 mini is enough) running the
  model-cached soak every night. Report to a small dashboard.

### 17. Rate-limiting is purely semaphore-based
- `MAX_CONCURRENT_CONNECTIONS = 4` defends against resource
  exhaustion but not against rapid reconnect storms.
- Fix: token-bucket per remote IP (default 10 conn/min) —
  `tower_governor` or hand-rolled; gated behind `--rate-limit` flag.

### 18. `ort_err()` wrapper is a leaky abstraction
- Keeps `ort::Error` Send/Sync via `anyhow::Error`. If upstream
  fixes Send, the wrapper becomes dead weight.
- Fix: track `ort` release notes; when Send is implemented, delete
  the helper and let `?` propagate natively.

### 19. Model reload requires restart
- No hot-swap of the INT8 encoder if it is created after server
  start. Not critical, but surfaces in the auto-quantize path on
  low-memory machines.
- Fix: `POST /v1/admin/reload` (loopback-only, no auth since local)
  re-creates the session pool.

### 20. No TLS / auth for remote deployments
- Docker/remote use is deferred to reverse proxy. Fine for now;
  document it prominently; add a `docs/deployment.md` covering
  Caddy/nginx + `auth_basic` recipe.

---

## Trace of what IS resolved (for completeness)

- **Native Rust INT8 quantization** — `src/quantize.rs`, CLI
  `gigastt quantize`, auto-quantize on `serve`/`download`
  via `--features quantize`. Python script removed.
- **Client examples** trimmed to Go / Python / Kotlin / Bun (TS).
