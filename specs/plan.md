# gigastt — fix-rollout plan

Plan of attack for the follow-ups in `specs/todo.md`. Ordered so
that each milestone unblocks the next and keeps `main` shippable
at every boundary. Every phase ends with a git tag + release-notes
bullet.

## Milestones delivered (2026-04-17)

| Phase | Tag | Highlights |
|-------|-----|-----------|
| 0 | v0.5.2 / v0.5.3 | release.yml + CONTRIBUTING + `rustls-webpki` advisory fix |
| 1 | v0.6.0 | origin allowlist, Retry-After, `--bind-all` guard, pool recovery via `catch_unwind` |
| 1.5 | v0.6.1 | `handle_ws_inner` split + origin middleware integration test |

Still open for Phase 1: item 6 (hard-coded limits) + item 7 (`/metrics`).
Phase 2 onward untouched. CUDA Linux asset temporarily removed from
release matrix (P0 addendum in `specs/todo.md`).

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

## Phase 5 — v1.0 readiness (≈2 days)

- **Item 16** — self-hosted runner (one M2 mini is enough) running
  `cargo test --test soak_test` nightly; emits summary to a GitHub
  Discussions thread or a tiny static page.
- **Item 18** — audit `ort_err()` helper against latest `ort`
  release; delete if Send/Sync is no longer required.
- Polish: ensure every README claim has a command that validates
  it ("WER 10.4%" → reproducible via `cargo test --test benchmark`).
- **Deliverable:** `v1.0.0`. Declare the WS protocol (`1.0`) and
  REST surface stable. Document deprecation policy.

---

## Parallelization hints

Within a phase, items are independent unless listed with a `+`.
When delegating to executors:

- Phase 1: items 3, 4+8, 5, 9 can all run in parallel.
- Phase 2: items 6 and 7 share the CLI struct; do 6 first, then 7.
  Item 10 is independent. Item 19 depends on 6.
- Phase 3: items 11, 12, 13 are independent.
- Phase 4: all independent.

## Skipped / explicitly not-now

- Server-side TLS: reverse-proxy recipe covers it. Reopen only if
  embedded-device users ask.
- gRPC / protobuf transcription API: no demand signal.
- Streaming diarization: architectural work, separate plan.

## Ownership

This plan is a starting point, not a contract. Items move between
phases as signal arrives (user reports, regressions, adoption
milestones). Re-evaluate after each phase closes.
