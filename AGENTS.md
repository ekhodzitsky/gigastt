# gigastt — agent guide

Read this guide before changing the repository. The product entry point is
[README.md](README.md); [docs/README.md](docs/README.md) indexes the maintained
references. Documentation follows the checked-out source, not necessarily the
latest package in every registry.

## Working rules

- Preserve unrelated user changes. Use an isolated worktree when the current
  checkout is dirty; do not delete untracked research or local model/data caches
  as repository cleanup.
- Read local `agents/backlog.md` and `agents/coding-principles.md` when present.
  They are ignored workstation instructions, not files required in a public clone.
- This project uses Backlog.md for local tracking. If configured, run
  `backlog instructions overview`, read the matching creation/execution/finalization
  guide, and mutate records through the CLI. Keep one logical task per PR.
- Keep internal planning identifiers and local tool state out of tracked files,
  branches, commits, tags and PR metadata. Describe what changed and why in English.
- Use the installed `rust-skills` instructions when writing, reviewing or
  refactoring Rust. Local skill installations are optional checkout tooling;
  load only relevant rules.
- Prefer a minimal change with a reproducible failure and regression coverage.
  Avoid new dependencies when existing code or the standard library suffices.
- Never bypass Git hooks or weaken required checks to publish a change.

## Build and validation

Rust 2024, minimum Rust version in [Cargo.toml](Cargo.toml), stable toolchain.
Native build dependencies and OpenSSL setup are in
[Contributing](.github/CONTRIBUTING.md#development). Default builds fetch ONNX
Runtime at build time; offline inference does not imply an offline build.

```sh
git config core.hooksPath .githooks
cargo fmt --all --check
cargo test --workspace --lib --bins
cargo clippy --workspace --all-targets -- -D warnings
python3 scripts/check-docs-drift.py
python3 scripts/test_docs_drift.py
python3 scripts/check-publication.py
```

Always run the workspace unit/bin tests and Clippy before finishing a change.
Do not run bare `cargo test` or `cargo test --workspace`: the custom WER target
has `harness = false` and can run for hours even when `--ignored` is supplied.
Use explicit integration targets. Model-backed, load/soak, long-form and feature
checks are documented in [testing](docs/testing.md).

All GitHub workflows use manual `workflow_dispatch`; do not add automatic push,
PR, schedule or chained triggers. Dispatch only when requested. Required checks
still apply to merging and publishing; see
[manual CI operation](docs/performance-regressions.md#manual-ci-operation).

## Source map

| Area | Location |
|---|---|
| Core engine, decoding, audio and session pool | `crates/gigastt-core/src/inference/` |
| Model catalog, pinned downloads, caches | `crates/gigastt-core/src/model/` |
| Runtime traits and tensors | `crates/gigastt-core/src/runtime/` |
| Backend/provider selection | `crates/gigastt-core/src/runtime/ort/factory.rs` |
| Protocol types | `crates/gigastt-core/src/protocol/` |
| CLI, boot policy, HTTP/WS server | `crates/gigastt/src/` |
| Packaging-only ONNX quantizer | `crates/gigastt-quantize/` |
| C ABI / Node / UniFFI | `crates/gigastt-ffi/`, `crates/gigastt-node/`, `crates/gigastt-uniffi/` |
| Server client SDKs | `sdks/go/`, `sdks/js/` |
| Model-free and model-backed integration tests | `crates/gigastt/tests/` |
| Benchmark inputs, harnesses and evidence | `benchmark/` |

## Rust and API conventions

- Use `anyhow` for internal errors and typed `GigasttError` at the public API.
  Convert ORT errors to `RuntimeError` at the runtime seam.
- No `unwrap()` / `expect()` in production paths. Use `?`, context and explicit
  recovery. Use `tracing`, not `println!`, in library code.
- Keep execution-provider selection in `runtime/ort/factory.rs`, not inference.
  `runtime/factory.rs` defines the backend traits.
- Shared inference constants belong in `inference/mod.rs`; do not duplicate them.
- Test names use `test_<behavior>`. Write a failing regression test before fixing
  an existing behavior; do not silently replace its acceptance baseline.
- Preserve additive protocol compatibility. Do not remove or rename existing
  fields. A breaking wire change requires explicit versioning and integration
  coverage, not an implicit reinterpretation of the old message.
- Streaming blank-run duration: `serve --endpoint-silence-ms` /
  `GIGASTT_ENDPOINT_SILENCE_MS` (40–2000 ms, default 600), overridden by
  `configure.endpoint_silence_ms`; applies only to auto mode without VAD.
- Keep CLI flags/env vars synchronized with [CLI](docs/cli.md), and endpoint
  behavior with [API](docs/api.md), [OpenAPI](docs/openapi.yaml) and
  [AsyncAPI](docs/asyncapi.yaml). Run the documentation drift check.
- Backend, model, codec and thread-policy changes require measured quality,
  speed and resources under the [acceptance policy](docs/performance-regressions.md).
  Dependency compatibility alone does not establish performance compatibility.

## Documentation and cleanup

Use [Models](docs/models.md) for model files and formatting policy,
[Architecture](docs/architecture.md) for component relationships and
[Deployment](docs/deployment.md) / [Runbook](docs/runbook.md) for operations.
Keep this guide focused on contributor rules rather than duplicating those
references' flag tables, benchmark numbers or dependency versions.

Historical measurements must retain provenance, raw evidence and limitations.
Never relabel a measured model revision as the current one, mix file and live
streaming metrics, or treat RSS and platform-specific physical footprint as the
same measure. Preserve data licenses and fixtures during cleanup.
