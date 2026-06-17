# gigastt — Agent Guide

> Local speech-to-text server powered by GigaAM v3 e2e_rnnt. On-device Russian
> speech recognition via ONNX Runtime. No cloud APIs, no API keys, full privacy.
>
> Repository: https://github.com/ekhodzitsky/gigastt  
> crates.io: https://crates.io/crates/gigastt  
> License: MIT

## Project Overview

**gigastt** is a single-binary Rust server that turns any machine into a
real-time Russian speech-to-text endpoint. It loads the GigaAM v3 RNN-T model
(Conformer encoder + LSTM decoder + joiner, 240M params) via ONNX Runtime and
exposes:

- **WebSocket** (`/v1/ws`) — streaming transcription with partial/final results
- **REST** (`/v1/transcribe`) — file upload, full JSON response
- **SSE** (`/v1/transcribe/stream`) — file upload, streaming Server-Sent Events
- **CLI** — `serve`, `download`, `transcribe`, `quantize` commands

The model (~850 MB FP32, ~225 MB INT8) auto-downloads from HuggingFace on first
run. INT8 quantization is native Rust (no Python), always compiled since v0.9.0,
and auto-invoked on first `serve`/`download` unless `--skip-quantize` is passed.

### Key metrics

| Property | Value |
|---|---|
| WER (Russian) | **11.4%** (9 994 Golos crowd samples, 50 394 words, 95% CI [10.9%, 11.9%]) |
| Latency (16s audio, M1) | ~700 ms |
| Memory (RSS) | ~560 MB |
| Concurrent sessions | 4 (configurable via `--pool-size`) |

## Technology Stack

- **Language**: Rust 2024 edition, stable toolchain (1.88+)
- **ONNX Runtime**: `ort` 2.0.0-rc.12
- **Async runtime**: tokio (full features)
- **HTTP + WebSocket server**: axum 0.8 (`ws`, `multipart`)
- **CLI**: clap 4 (derive, env)
- **Serialization**: serde + serde_json
- **Logging**: tracing + tracing-subscriber (env-filter)
- **Error handling**: anyhow (internal), `GigasttError` (public API)
- **Audio decoding**: symphonia (AAC, MP3, OGG, FLAC, WAV, PCM)
- **Audio resampling**: rubato 0.16
- **FFT**: rustfft 6
- **Protobuf**: prost 0.14 + prost-build 0.14 (build-time)
- **Rate limiting**: in-tree token-bucket (dashmap-backed)
- **Metrics**: in-tree Prometheus text encoder (optional `--metrics` flag)

### Execution providers (compile-time features)

| Platform | Feature | Provider |
|---|---|---|
| macOS ARM64 | `--features coreml` | CoreML + Neural Engine |
| Linux x86_64 + NVIDIA | `--features cuda` | CUDA 12+ |
| Android / ARM64 | `--features nnapi` | NNAPI (via `ort/nnapi`) |
| Any | _(default)_ | CPU |

Features `coreml` and `cuda` are **mutually exclusive**. `nnapi` is not mutually exclusive with either.

## Build Requirements

- Rust 1.88+ (stable)
- `protoc` (Protocol Buffers compiler) on `PATH` — required by `build.rs` which
  regenerates ONNX protobuf types via `prost-build`
  - macOS: `brew install protobuf`
  - Debian/Ubuntu: `apt install protobuf-compiler`
- **Build-time network fetch:** `ort`'s default `download-binaries` feature downloads a
  prebuilt onnxruntime native library over the network at build time (outside `Cargo.lock`,
  verified by an embedded checksum). The "no cloud" guarantee is runtime-only. For
  air-gapped builds, use `ort` with `default-features = false` + `load-dynamic` and pin the
  native library via `ORT_*` env vars / `.cargo/config.toml`.

## Build Commands

```sh
# Debug build (CPU only, any platform)
cargo build

# Release build (LTO, stripped, single codegen unit)
cargo build --release

# macOS ARM64 with CoreML / Neural Engine
cargo build --release --features coreml

# Linux x86_64 with NVIDIA CUDA 12+
cargo build --release --features cuda

# Android with NNAPI
cargo build --release --features nnapi

# With speaker diarization support
cargo build --release --features diarization
```

## Test Commands

The project uses a three-tier test architecture:

### Unit tests (no model required, run in CI on every PR)

```sh
cargo test --workspace               # unit tests across the workspace
cargo clippy                         # Lint (zero warnings expected)
cargo fmt --check                    # Format check
```

Unit tests live in `#[cfg(test)] mod tests` at the bottom of each source file.
They use synthetic data. Test naming convention: `test_<what>_<expected_behavior>`.

### E2E tests (require model ~850 MB, run in CI on main push only)

```sh
# Download model first
cargo run -- download

# Run all e2e tests serially (single-threaded to avoid OOM)
cargo test -p gigastt --test e2e_rest --test e2e_ws --test e2e_errors --test e2e_shutdown --test e2e_rate_limit -- --ignored --test-threads=1
```

| Test file | Coverage |
|---|---|
| `tests/e2e_rest.rs` | REST API: health, transcribe, SSE streaming, error paths |
| `tests/e2e_ws.rs` | WebSocket: ready, audio, stop, configure, errors, concurrent |
| `tests/e2e_errors.rs` | Error paths: oversized body/frame, pool saturation, idle timeout |
| `tests/e2e_shutdown.rs` | Graceful shutdown: WS final + close, SSE termination, max-session cap |
| `tests/e2e_rate_limit.rs` | Per-IP rate limiter 429 behavior |

Shared helpers are in `tests/common/mod.rs` (server startup with shutdown handle,
WAV generation, WebSocket connect, readiness polling).

### Load & soak tests (require model, run locally + nightly CI)

```sh
cargo test -p gigastt --test load_test -- --ignored           # 3 load tests
cargo test -p gigastt --test soak_test -- --ignored           # Continuous WS cycling
```

Soak duration is configurable via `GIGASTT_SOAK_DURATION_SECS` (default 300s).

### Benchmark suite

```sh
cargo test -p gigastt --test benchmark -- --ignored            # WER on Golos fixtures
```

Custom harness (`harness = false` in `Cargo.toml`).

## Code Organization

```
crates/
  gigastt-core/src/       # Core library (inference engine, no server deps)
    lib.rs                # Public module exports
    error.rs              # Typed error types (GigasttError)
    quantize.rs           # Native Rust INT8 quantization pipeline
    onnx_proto.rs         # prost-generated ONNX types (included from OUT_DIR)
    inference/
      mod.rs              # Engine: ONNX session management, SessionPool, StreamingState
      features.rs         # Mel spectrogram (64 bins, FFT=320, hop=160, HTK)
      tokenizer.rs        # BPE tokenizer (1025 tokens)
      decode.rs           # RNN-T greedy decode loop
      audio.rs            # Audio loading, resampling, channel mixing
    protocol/mod.rs       # WebSocket JSON message types (Ready, Partial, Final, Error)
    model/mod.rs          # HuggingFace model download (streaming + SHA256 + atomic rename)
  gigastt-core/proto/
    onnx.proto            # Vendored ONNX protobuf schema
  gigastt-ffi/src/        # C-ABI FFI layer (cdylib for Android/mobile)
    lib.rs                # Exported C functions: engine_new, transcribe_file, stream_*, etc.
  gigastt/src/            # Server binary + CLI
    lib.rs                # Re-exports gigastt-core::* for backward compat
    main.rs               # CLI (clap): serve, download, transcribe, quantize
    server/
      mod.rs              # axum router, origin middleware, graceful shutdown
      http.rs             # REST handlers: /health, /v1/models, /v1/transcribe, SSE
      rate_limit.rs       # In-tree per-IP token-bucket rate limiter
      metrics.rs          # In-tree Prometheus text encoder
  gigastt/tests/
    common/mod.rs         # Shared e2e helpers
    benchmark.rs          # WER evaluation (custom harness)
    e2e_*.rs              # E2E test suites
    load_test.rs          # Load tests
    soak_test.rs          # Soak test
```

## Key Constants

Defined in `crates/gigastt-core/src/inference/mod.rs`:

| Constant | Value | Meaning |
|---|---|---|
| `N_MELS` | 64 | Mel frequency bins |
| `N_FFT` | 320 | FFT window size (20ms @ 16kHz) |
| `HOP_LENGTH` | 160 | Hop length (10ms @ 16kHz) |
| `PRED_HIDDEN` | 320 | Decoder LSTM hidden dim |
| `DEFAULT_POOL_SIZE` | 4 | Concurrent inference sessions |

## Model Files

Downloaded to `~/.gigastt/models/` from `istupakov/gigaam-v3-onnx`:

| File | Size | Purpose |
|---|---|---|
| `v3_e2e_rnnt_encoder.onnx` | 844 MB | Conformer encoder (FP32) |
| `v3_e2e_rnnt_encoder_int8.onnx` | ~225 MB | Quantized encoder (auto-generated) |
| `v3_e2e_rnnt_decoder.onnx` | 4.4 MB | LSTM decoder |
| `v3_e2e_rnnt_joint.onnx` | 2.6 MB | RNN-T joiner |
| `v3_e2e_rnnt_vocab.txt` | small | BPE vocabulary (1025 tokens) |

## Development Conventions

### Code style

- Rust 2024 edition
- `anyhow` for internal error handling, `GigasttError` for public API
- `tracing` for logging (never `println!` in library code)
- **No `unwrap()` in production paths** — use `?`, `.context()`, or `unwrap_or_else`
- Shared constants live in `inference/mod.rs`, referenced by sub-modules
- `ort` errors wrapped via `ort_err()` helper (Send/Sync workaround for ort 2.0-rc)
- Execution provider selection uses `#[cfg(feature = "coreml")]` / `#[cfg(feature = "cuda")]` blocks
- **No internal task-tracker IDs in code/docs.** Never write tracker indices (`V1-NN`, `SUS-NN`, `TODO-NN`, etc.) into source comments, `CHANGELOG.md`, `docs/`, CI files, or any artifact — they mean nothing to a reader without the tracker. Comments should say *what* and *why*; the link from a fix to a tracked item lives only in `specs/prod-readiness-v1.0.md`, not in the code.

### TDD workflow

1. Write failing test first
2. Implement minimal code to pass
3. Refactor, verify tests still pass
4. `cargo test && cargo clippy` before every commit (or `make check`)
5. Enable the pre-commit hook so step 4 is enforced automatically:
   ```sh
   git config core.hooksPath .githooks
   ```

### API versioning & backward compatibility

- WebSocket protocol version: `PROTOCOL_VERSION = "1.0"` (in `protocol/mod.rs`)
- Canonical WS path: `/v1/ws` (v0.7.0+). The `/ws` alias was removed in v0.8.0.
- New fields are **additive only** — never remove or rename existing fields
- Fields like `supported_rates`, `diarization`, `retry_after_ms` use
  `skip_serializing_if` to keep older clients happy
- Breaking changes require a new message type, not modification of existing
- Deprecation: add `deprecated: true` field, support old format for 2 minor versions

### Audio format support

- **File transcription**: WAV, M4A/AAC, MP3, OGG/Vorbis, FLAC (via symphonia)
- **WebSocket streaming**: raw PCM16 binary frames at configurable sample rate
  (8/16/24/44.1/48 kHz, default 48kHz); resampled to 16kHz server-side via rubato
- Auto mono mix for multi-channel files

## CI / CD

### Workflows

| Workflow | Trigger | What it does |
|---|---|---|
| `.github/workflows/ci.yml` | PR + main push | fmt, clippy, unit tests, feature compile checks (coreml, cuda, diarization), `cargo audit`, `cargo deny` |
| `.github/workflows/soak.yml` | Nightly 03:17 UTC + manual | soak_test + load_test with cached model |
| `.github/workflows/release.yml` | Tag push `v*` + manual | Multi-arch build, tarball + SHA256, CycloneDX SBOM, SLSA provenance, minisign signatures |
| `.github/workflows/homebrew.yml` | Release published | Update Homebrew tap Formula |

### E2E test strategy

- E2E tests run **only on main push**, not on PRs, to keep PR feedback fast
- Model is cached via `actions/cache` with key derived from `crates/gigastt-core/src/model/mod.rs`
- E2E tests run with `--test-threads=1` because each loads the full ONNX model
  into memory; concurrent runs OOM on CI runners

### Branch protection (repository settings)

The following rules must be enabled in **GitHub Settings → Branches** for `main`:

- **Require a pull request before merging** — direct push to `main` is blocked.
- **Require status checks to pass** — at minimum `fmt`, `clippy`, and `unit-tests` jobs from `ci.yml` must be green.
- **Include administrators** — rules apply to everyone, no exceptions.

This guarantees that a regression like a broken `cargo test` cannot reach `main` even if the local pre-commit hook is bypassed.

## Security Considerations

- **Loopback bind by default.** Server refuses non-loopback addresses unless
  `--bind-all` or `GIGASTT_ALLOW_BIND_ANY=1` is set. Prevents accidental public
  exposure.
- **Origin allowlist.** Cross-origin requests denied by default. Loopback origins
  always allowed. Extra origins via `--allow-origin` (repeatable). Wildcard CORS
  is opt-in via `--cors-allow-any`.
- **Runtime limits** (all configurable via CLI flags and env vars):
  - `--idle-timeout-secs` (default 300) — WebSocket idle timeout
  - `--ws-frame-max-bytes` (default 512 KiB) — max WS frame size
  - `--body-limit-bytes` (default 50 MiB) — max REST body size
  - `--pool-size` (default 4) — concurrent inference sessions
  - `--pool-min-size` (default 1) — minimum triplets required to boot (degraded-pool floor)
  - `--batch-pool-size` (default 0) — triplets reserved for batch REST jobs (0 = shared pool)
  - `--inference-timeout-secs` (default 600) — per-request inference timeout; 0 disables
  - `--max-session-secs` (default 3600) — wall-clock session cap
  - `--shutdown-drain-secs` (default 10) — graceful shutdown drain window
- **Per-IP rate limiting** (opt-in, off by default): `--rate-limit-per-minute N`
  enables token-bucket limiter on `/v1/*`; `/health` is exempt. Returns HTTP 429
  + `Retry-After` when exhausted.
- **Pool saturation backpressure.** REST returns 503 + `Retry-After: 30`;
  WebSocket error includes `retry_after_ms: 30000`.
- **SHA-256 verification + atomic rename** on model files. Download stages to
  `.partial`, verifies hash, then atomically renames. Corrupt downloads are
  removed, not promoted.
- **Internal errors sanitized** — no path or model leakage to clients.
- **Prometheus `/metrics`** (opt-in via `--metrics`): exposes
  `gigastt_http_requests_total` and `gigastt_http_request_duration_seconds`.
  Served on a separate loopback listener (default `127.0.0.1:9090`, override
  via `--metrics-listen` / `GIGASTT_METRICS_LISTEN`) — not the main API port,
  and therefore off the CORS allowlist and the per-IP rate limiter.

## Docker

```sh
# CPU (any platform)
docker build -t gigastt .
docker run -p 9876:9876 gigastt

# CUDA (Linux, requires NVIDIA Container Toolkit)
docker build -f Dockerfile.cuda -t gigastt-cuda .
docker run --gpus all -p 9876:9876 gigastt-cuda

# Baked image (model included at build time, ~1.1 GB)
docker build --build-arg GIGASTT_BAKE_MODEL=1 -t gigastt:baked .
```

Docker images run with `--bind-all --host 0.0.0.0` because container networking
requires listening on all interfaces. The non-Docker default is `127.0.0.1`.

## Environment Variables

All CLI flags have corresponding env vars:

| Env var | CLI flag | Default |
|---|---|---|
| `GIGASTT_ALLOW_BIND_ANY` | `--bind-all` | — |
| `GIGASTT_IDLE_TIMEOUT_SECS` | `--idle-timeout-secs` | 300 |
| `GIGASTT_WS_FRAME_MAX_BYTES` | `--ws-frame-max-bytes` | 524288 |
| `GIGASTT_BODY_LIMIT_BYTES` | `--body-limit-bytes` | 52428800 |
| `GIGASTT_RATE_LIMIT_PER_MINUTE` | `--rate-limit-per-minute` | 0 |
| `GIGASTT_RATE_LIMIT_BURST` | `--rate-limit-burst` | 10 |
| `GIGASTT_POOL_MIN_SIZE` | `--pool-min-size` | 1 |
| `GIGASTT_BATCH_POOL_SIZE` | `--batch-pool-size` | 0 |
| `GIGASTT_INFERENCE_TIMEOUT_SECS` | `--inference-timeout-secs` | 600 |
| `GIGASTT_MAX_SESSION_SECS` | `--max-session-secs` | 3600 |
| `GIGASTT_SHUTDOWN_DRAIN_SECS` | `--shutdown-drain-secs` | 10 |
| `GIGASTT_SKIP_QUANTIZE` | `--skip-quantize` | false |
| `GIGASTT_METRICS` | `--metrics` | false |
| `GIGASTT_METRICS_LISTEN` | `--metrics-listen` | 127.0.0.1:9090 |
| `RUST_LOG` | — | `gigastt=info` |

## Useful Commands for Agents

```sh
# Quick iteration cycle
cargo test && cargo clippy

# Run with model (after `cargo run -- download`)
cargo run --release -- serve
cargo run --release -- transcribe recording.wav

# Check all feature combinations compile
cargo check --features coreml
cargo check --features cuda
cargo check --features diarization

# Security audit
cargo audit
cargo deny check

# Run a specific e2e test
cargo test -p gigastt --test e2e_ws -- --ignored test_ws_ready_message

# Run with tracing at debug level
RUST_LOG=gigastt=debug cargo run -- serve
```

## Notes for AI Agents

- **Always run `cargo test && cargo clippy` before finishing any change.**
- When modifying the WebSocket protocol, update `PROTOCOL_VERSION` in
  `protocol/mod.rs` and add tests in `tests/e2e_ws.rs`.
- When adding new CLI flags, add the corresponding env var and document it in
  both `main.rs` and this file.
- The `quantize` Cargo feature is a no-op retained for backward compatibility;
  do not gate new code behind it.
- Model download logic is in `crates/gigastt-core/src/model/mod.rs`. If you change HF repo or file
  names, update `MODEL_CHECKSUMS` and the cache key in `.github/workflows/ci.yml`.
- The project uses English for all code comments, documentation, and commit
  messages.
