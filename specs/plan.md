# gigastt — fix-rollout plan

Plan of attack for the follow-ups in `specs/todo.md`. Ordered so
that each milestone unblocks the next and keeps `main` shippable
at every boundary. Every phase ends with a git tag + release-notes
bullet.

## Milestones delivered (2026-04-20)

| Phase | Tag | Highlights |
|-------|-----|-----------|
| 0 | v0.5.2 / v0.5.3 | release.yml + CONTRIBUTING + `rustls-webpki` advisory fix |
| 1 | v0.6.0 | origin allowlist, Retry-After, `--bind-all` guard, pool recovery via `catch_unwind` |
| 1.5 | v0.6.1 | `handle_ws_inner` split + origin middleware integration test |
| 2 | v0.7.x | configurable runtime limits, bake-model, `/v1/ws` canonical, capabilities |
| 3 | v0.8.x | `/metrics`, per-IP rate limit, deployment docs, CoreML/CUDA startup fix |

**Carry-over not in v1.0 critique:** item 15 (WER histograms), item 16
(nightly soak), item 18 (`ort_err()` audit), item 19 (hot-reload model),
CUDA release asset. Re-sequenced into Phase 6 / 7 below.

## Phase 0 — stop the bleeding (1 day)

**Goal:** prevent the class of problems we already hit (Murmur
SHA-pinned download 404).

- **Item 1** — `release.yml` matrix workflow:
  jobs `macos-arm64`, `linux-x86_64-cpu`, `linux-x86_64-cuda`.
  Produces `gigastt-<ver>-<triple>.tar.gz` + `SHA256SUMS.txt`.
  Triggered on `v*` tag push. Uses `softprops/action-gh-release`.
- **Item 2** — `CONTRIBUTING.md` gets a release checklist:
  (a) bump `Cargo.toml`, (b) update `CHANGELOG`, (c) `git tag -s`,
  (d) push tag → wait for release workflow green,
  (e) `cargo publish --dry-run`, (f) `cargo publish`.
- **Deliverable:** `v0.5.1` tag cut through the new pipeline. CI
  green. `SHA256SUMS.txt` published. Murmur can revert its
  manual-upload workaround.

## Phase 1 — safety & stability (≈1 week)

**Goal:** close the real security and reliability gaps before we
invite broader adoption.

- **Item 3** — pool depletion fix. Restructure `handle_ws_inner`
  closure ownership so the triplet is recoverable after
  `spawn_blocking` panic (mirror SSE handler pattern in
  `src/server/http.rs`). Add a unit test that panics inside the
  blocking task and asserts pool capacity is preserved.
- **Item 4 + 8** — Origin-deny middleware. Single `Layer` that
  enforces allowlist for both `/ws` and `/v1/*`. CORS `*` becomes
  opt-in (`--cors-allow-any`). Integration test with a fake
  non-local Origin header.
- **Item 5** — `Retry-After` wiring in 503/WS-error payloads.
- **Item 9** — `--bind-all` / `GIGASTT_ALLOW_BIND_ANY=1` guard.
  Default: refuse non-loopback bind without the flag. Update
  Dockerfiles to set the env.
- **Deliverable:** `v0.6.0`. README gains a short "Security"
  section referencing the new knobs.

## Phase 2 — configurability & observability (≈1 week)

**Goal:** make the server deployable without a fork.

- **Item 6** — CLI + env + TOML config parsing. One struct,
  three layers (`clap` → `envy` → `toml`). Precedence:
  flag > env > file > default. Document in `docs/config.md`.
- **Item 7** — `metrics` feature flag. `GET /metrics` behind
  `--metrics` (bind on same port, disabled by default). Standard
  RED metrics + per-stage timings + pool depth gauge.
- **Item 10** — `GIGASTT_BAKE_MODEL=1` build arg for Docker.
  Publish both a slim and a baked-model image tag
  (`gigastt:0.7.0`, `gigastt:0.7.0-model`).
- **Item 19** — `POST /v1/admin/reload` (loopback-only).
- **Deliverable:** `v0.7.0`. Sample systemd unit + Caddy config
  land under `docs/deployment/`.

## Phase 3 — API surface polish (≈3 days)

**Goal:** make the public HTTP/WS contract something we can live
with for v1.0 without deprecation cycles immediately after.

- **Item 11** — `/v1/ws` canonical, `/ws` alias with warn log.
- **Item 12** — extend `/v1/models` with `capabilities`.
- **Item 13** — split `handle_ws_inner` into three frame
  handlers + one orchestration loop. Adds ~4 small unit tests.
- **Item 20** — `docs/deployment.md` TLS/auth recipe via reverse
  proxy. No server-side TLS yet (scope creep).
- **Deliverable:** `v0.8.0`. `docs/asyncapi.yaml` updated to
  reflect `/v1/ws` and `capabilities` field.

## Phase 4 — supply chain & benchmarks (≈3 days)

**Goal:** auditability for privacy-conscious adopters.

- **Item 14** — `cargo deny check` in PR CI;
  `cyclonedx-bom` generation in release workflow.
- **Item 15** — benchmark harness emits JSON (`benchmark.json`) +
  markdown summary with length/SNR buckets. Commit the JSON so
  diffs are visible in PRs.
- **Item 17** — optional token-bucket rate limit behind
  `--rate-limit` (per remote IP).
- **Deliverable:** `v0.9.0`. Every release tarball accompanied by
  `bom.cdx.json`, `SHA256SUMS.txt`, and `benchmark.json`.

## Phase 5 — observability + security hardening (≈1 week, post-critique)

**Goal:** close the critical gaps flagged by the 2026-04-18 review
before inviting broader adoption. All items reference IDs in
[`specs/prod-readiness-v1.0.md`](prod-readiness-v1.0.md).

### P0 lane (lockdown)

- **V1-01** — model download TOCTOU: `.partial` tmp file + verify +
  `rename` pattern. Re-verify on `Engine::load`.
- **V1-02** — add SHA256 for speaker model (diarization feature).
- **V1-03** — graceful WS drain on SIGTERM. Thread
  `CancellationToken` into `handle_ws_inner`; re-enable
  `tests/e2e_shutdown.rs` in CI.
- **V1-04** — max session duration cap; prevent silence-stream DoS.
- **V1-05** — drop REST `body.to_vec()`; streaming body path for
  `/v1/transcribe/stream`.
- **V1-06** — rate-limiter per-millisecond math; fix sub-60 rpm bug.
- **V1-07** — replace `Mutex<mpsc::Receiver>` pool primitive; remove
  `expect("Pool sender dropped")`.
- **V1-08** — `Engine::create_state` unconditional signature.
- **V1-09** — nightly soak / load CI workflow.
- **V1-10** — `prost 0.6` / `onnx-pb 0.1.4` decision: upgrade or
  remove. Audit `deny.toml` ignores.

### P1 lane (ship-before-v1)

- **V1-11 … V1-14** — proxy docs, separate `/metrics` listener, WS
  protocol negotiation, `/ws` deprecation headers.
- **V1-15 … V1-17** — background-thread cancellation, degraded pool
  load, per-server Prometheus recorder.
- **V1-18 … V1-20** — decode-loop allocation fixes, streaming
  resampler in `StreamingState`, quantizer `axis` correctness.
- **V1-21 … V1-25** — pool Drop-guard, assertion tightening, purge
  `server_integration.rs`, batch/stream pool split, odd-PCM frame
  carry.
- **V1-26 … V1-30** — Engine decomposition, `/healthz` vs `/readyz`,
  configurable checkout timeout, faster idle test, pool/inference/WS
  metrics.

**Deliverable:** `v0.9.0`. Every P0 item above is closed with a
regression test; `e2e_shutdown` is back in CI.

**Parallelisation:** V1-01 → V1-02 sequential (same file); V1-03 +
V1-04 in one PR (same handler); V1-05 independent; V1-06 + V1-07
parallel with merge gate; V1-08, V1-09, V1-10 independent.

## Phase 6 — v1.0 GA (≈1 week)

**Goal:** declare the public contract stable.

- **V1-31 … V1-50** — P2 polish: endpointing semantics, AsyncAPI
  sync, supported-rates dedup, SSE error parity, Prometheus label
  cardinality, ping timer, `audit.toml`/`deny.toml` dedup, version
  pins, benchmark baseline gate, Golos fixture expansion, monotonic
  timestamps, warmup method, inference timeout, VAD endpointing,
  multi-model manifest, etc.
- **Sustainability batch (SUS-01 … SUS-14):** `SECURITY.md`,
  CycloneDX SBOM, release signing, Dependabot, SLSA attestations,
  `cargo-fuzz`, Miri/ASAN/TSAN job, coverage gate, Grafana
  dashboards + alert rules, runbook, `terminationGracePeriodSeconds`
  docs, OpenAPI, privacy doc, `cargo-semver-checks`.
- Legacy carry-over: item 15 (WER histogram → V1-41), item 18
  (`ort_err()` audit), item 19 (`POST /v1/admin/reload`), CUDA in
  release matrix.

**Deliverable:** `v1.0.0`. WS protocol (`1.0`) and REST surface
declared stable. Deprecation policy documented under
`docs/compatibility.md`. Tag requires:

1. All P0 (V1-01 … V1-10) closed with linked test evidence.
2. ≥ 80 % of P1 items closed; remainder listed as known issues in
   `CHANGELOG.md`.
3. Nightly soak has been green for 14 consecutive days.
4. Benchmark WER regression gate is active and green on main.
5. `cargo deny check advisories` has no unexplained ignores.

---

## Parallelization hints

Within a phase, items are independent unless listed with a `+`.
When delegating to executors:

- Phase 1: items 3, 4+8, 5, 9 can all run in parallel.
- Phase 2: items 6 and 7 share the CLI struct; do 6 first, then 7.
  Item 10 is independent. Item 19 depends on 6.
- Phase 3: items 11, 12, 13 are independent.
- Phase 4: all independent.
- Phase 5: see P0/P1 lane notes above; keep the two lanes on
  separate PR chains to avoid merge thrash.
- Phase 6: polish items parallelise freely; sustainability items
  pair naturally (SUS-02 ↔ SUS-05, SUS-09 ↔ SUS-10).

## Skipped / explicitly not-now

- Server-side TLS: reverse-proxy recipe covers it. Reopen only if
  embedded-device users ask.
- gRPC / protobuf transcription API: no demand signal.
- Streaming diarization: architectural work, separate plan.

## Ownership

This plan is a starting point, not a contract. Items move between
phases as signal arrives (user reports, regressions, adoption
milestones). Re-evaluate after each phase closes.
