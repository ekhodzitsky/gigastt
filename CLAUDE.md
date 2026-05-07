# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**gigastt** — local speech-to-text server powered by GigaAM v3 e2e_rnnt. On-device Russian speech recognition via ONNX Runtime. No cloud APIs, no API keys, full privacy.

- **Repository**: https://github.com/ekhodzitsky/gigastt
- **crates.io**: https://crates.io/crates/gigastt
- **License**: MIT

## Build & Test

```sh
cargo build                          # CPU-only debug build (default, any platform)
cargo build --features coreml        # macOS ARM64 (CoreML / Neural Engine)
cargo build --features cuda          # Linux x86_64 (CUDA 12+)
cargo build --release                # Release build (LTO, stripped)
cargo test                           # Run all 125 unit tests, CPU (no model required)
cargo test --features coreml         # Same tests with CoreML EP enabled (macOS)
cargo test --test e2e_rest --test e2e_ws --test e2e_errors --test e2e_shutdown --test e2e_rate_limit -- --ignored --test-threads=1  # E2E tests (requires model)
cargo test --test load_test -- --ignored           # Load tests (requires model, local only)
cargo test --test soak_test -- --ignored           # Soak test (requires model, local only)
cargo clippy             # Lint (no expected warnings)
```

Note: `cargo build` requires `protoc` in `PATH` for the in-tree ONNX quantization pipeline (see `build.rs`). Install via `brew install protobuf` (macOS) or `apt install protobuf-compiler` (Debian/Ubuntu).

Model download (required for E2E testing and file transcription, ~850MB):
```sh
cargo run -- download                    # Downloads to ~/.gigastt/models/ and auto-generates INT8 encoder
cargo run -- download --skip-quantize    # Skip auto-quantization (FP32 only)
cargo run -- quantize                    # Regenerate INT8 encoder manually (~210MB)
```

## Docker

Multi-stage production build:
```sh
# CPU / macOS (default Dockerfile)
docker build -t gigastt .
docker run -p 9876:9876 gigastt
# Model auto-downloads on first run, binds to 0.0.0.0:9876

# CUDA (Linux, requires NVIDIA Container Toolkit)
docker build -f Dockerfile.cuda -t gigastt-cuda .
docker run --gpus all -p 9876:9876 gigastt-cuda

# Bake the model into the image (zero cold-start, ~1.1 GB image):
docker build --build-arg GIGASTT_BAKE_MODEL=1 -t gigastt:baked .
```

The Dockerfile passes `--bind-all` so the server listens on `0.0.0.0` inside the container. Local deployments use `127.0.0.1` by default; `--bind-all` (or `GIGASTT_ALLOW_BIND_ANY=1`) is required to listen on non-loopback addresses.

## Architecture

```
src/
  lib.rs                  # Public module exports
  main.rs                 # CLI (clap): serve, download, transcribe, quantize
  model/mod.rs            # HuggingFace model download (streaming + SHA256 + atomic rename)
  inference/
    mod.rs                # Engine: ONNX session management, SessionPool, StreamingState, DecoderState
    features.rs           # Mel spectrogram (64 bins, FFT=320, hop=160, HTK)
    tokenizer.rs          # BPE tokenizer (1025 tokens)
    decode.rs             # RNN-T greedy decode loop
    audio.rs              # Audio loading, resampling, channel mixing (symphonia + rubato)
  error.rs                # Typed error types (GigasttError)
  quantize.rs             # Native Rust INT8 quantizer (always compiled since v0.9.0)
  onnx_proto.rs           # prost-generated ONNX types from proto/onnx.proto
  server/
    mod.rs                # axum router: HTTP + WebSocket on single port, origin middleware, graceful drain
    http.rs               # REST handlers: /health, /v1/models, /v1/transcribe, /v1/transcribe/stream (SSE)
    rate_limit.rs         # In-tree per-IP token-bucket rate limiter (dashmap-backed)
    metrics.rs            # In-tree Prometheus text encoder (counters + histograms)
  protocol/mod.rs         # JSON message types (Ready, Partial, Final, Error + retry_after_ms)
```

### Performance optimizations (v0.9)
- **CoreML execution provider** (`--features coreml`, macOS ARM64): MLProgram format + Neural Engine
  - Automatically loads quantized encoder if available (~4x smaller, ~43% faster)
  - Caches compiled models in `~/.gigastt/models/coreml_cache/`
- **CUDA execution provider** (`--features cuda`, Linux x86_64 CUDA 12+): GPU inference via ONNX Runtime CUDA EP
  - Features are compile-time and mutually exclusive; default build uses CPU EP on all platforms
- **INT8 quantization** (always compiled, auto-invoked since v0.9.0): encoder_int8.onnx (~210MB vs 844MB)
  - Rust-native quantization in `src/quantize.rs` (no Cargo feature required; `quantize` feature kept as no-op for backward compat)
  - Auto-invoked by `gigastt download` and `gigastt serve` on first run (~2 min one-time)
  - Opt out with `--skip-quantize` or `GIGASTT_SKIP_QUANTIZE=1`
  - Auto-detection: Engine uses INT8 encoder if present, falls back to FP32
- **Zero-copy REST upload path** (v0.9.0): `bytes::Bytes` flows end-to-end from axum into symphonia via a crate-private `BytesMediaSource`, eliminating the 4× upload copy that used to OOM small containers on concurrent 10-minute uploads.

### Key constants (defined in `inference/mod.rs`)
- `N_MELS = 64`, `N_FFT = 320`, `HOP_LENGTH = 160`, `PRED_HIDDEN = 320`
- Encoder dim: 768, Vocab: 1025 tokens, Blank: 1024

### Data flow
```
Audio (PCM16) → Mel Spectrogram → Conformer Encoder (ONNX)
  → RNN-T Decoder+Joiner loop → BPE tokens → Text
```

### Streaming
- `StreamingState` persists LSTM h/c and audio buffer across WebSocket chunks
- `DecoderState` holds decoder hidden state (h, c, prev_token)
- Server accepts configurable sample rates (8kHz, 16kHz, 24kHz, 44.1kHz, 48kHz) via `Configure` message
- Default 48kHz for backward compatibility; resamples to 16kHz via rubato (polyphase FIR)
- Odd-length PCM16 frames are carried across to the next frame (v0.9.0 V1-25) to avoid single-byte phase drift.

### Graceful shutdown (v0.9.0)
- A single `CancellationToken` + `TaskTracker` cascades through every WebSocket / SSE handler.
- On SIGTERM each live session flushes, emits an empty-if-needed `Final`, and closes with `Close(1001 Going Away)`.
- After `axum::serve` returns, `run_with_config` waits up to `--shutdown-drain-secs` (default 10) for the tracker to drain.
- A wall-clock `--max-session-secs` cap (default 3600) closes silence-stream DoS attempts with `Close(1008) + code=max_session_duration_exceeded`.

## Development guidelines

### TDD workflow
1. Write failing test first
2. Implement minimal code to pass
3. Refactor, verify tests still pass
4. `cargo test && cargo clippy` before every commit

### API versioning & backward compatibility
- WebSocket protocol version: `PROTOCOL_VERSION = "1.0"` (in `protocol/mod.rs`)
- `ServerMessage::Ready` includes `version` field sent on connection
- WebSocket path: `/v1/ws`
- WebSocket protocol messages are versioned via `type` field
- New fields are additive only (never remove or rename existing fields). `supported_rates`, `diarization`, and `retry_after_ms` are serialized with `skip_serializing_if` to keep older clients happy.
- Breaking changes require new message type, not modification of existing
- Deprecation: add `deprecated: true` field, support old format for 2 minor versions

### Testing

Three-tier test architecture:

**Unit tests** (no model required, run in CI on every PR):
- Live in `#[cfg(test)] mod tests` at bottom of each file
- Use synthetic data, test names: `test_<what>_<expected_behavior>`
- 125 unit tests across 15 modules (as of v0.9.2)
- `cargo test` — runs all unit tests

**E2E tests** (require model ~850MB, run in CI on main push only):
- `tests/e2e_rest.rs` — REST API tests (health, transcribe, SSE streaming, error paths)
- `tests/e2e_ws.rs` — WebSocket protocol tests (ready, audio, stop, configure, errors, concurrent)
- `tests/e2e_errors.rs` — error path tests (oversized body/frame, pool saturation, idle timeout)
- `tests/e2e_shutdown.rs` — graceful shutdown tests (WS final + close, SSE termination, max-session cap, shutdown under pool saturation)
- `tests/e2e_rate_limit.rs` — per-IP rate limiter 429 behavior (v0.8.0+)
- `tests/common/mod.rs` — shared helpers (start_server with shutdown handle, WAV generation, WS connect)
- `cargo test --test e2e_rest --test e2e_ws --test e2e_errors --test e2e_shutdown --test e2e_rate_limit -- --ignored --test-threads=1` — all e2e tests

**Load/soak tests** (require model, run locally + nightly CI via `.github/workflows/soak.yml`):
- `tests/load_test.rs` — 3 load tests (concurrent WS, concurrent REST, burst connections)
- `tests/soak_test.rs` — 1 soak test (continuous WS cycling, configurable via `GIGASTT_SOAK_DURATION_SECS`)
- `cargo test --test load_test -- --ignored` / `cargo test --test soak_test -- --ignored`

**Benchmark suite:**
- `tests/benchmark.rs` — WER evaluation on Golos fixtures (custom harness, `harness = false`)

### CI structure
- **PR CI** (`.github/workflows/ci.yml`, fast): fmt, clippy, unit tests, feature compile checks (CoreML, CUDA, diarization), `cargo audit`, `cargo deny`
- **Main push CI**: all PR checks + e2e tests with cached model (~850MB, OS-independent cache key)
- **Nightly soak** (`.github/workflows/soak.yml`): `cargo test --test soak_test` at 03:17 UTC, reuses the main CI model cache
- **Release** (`.github/workflows/release.yml`, tag-triggered): multi-arch tarballs, per-asset `.sha256` + `SHA256SUMS.txt`, CycloneDX SBOM, SLSA provenance, minisign signatures
- Load tests are local-only, not in CI

### Code style
- Rust 2024 edition
- `anyhow` for error handling, `tracing` for logging
- No `unwrap()` in production paths (use `?`, `context()`, or `unwrap_or_else`)
- Shared constants in `inference/mod.rs`, referenced by sub-modules
- `ort` errors wrapped via `ort_err()` helper (Send/Sync workaround)
- Execution provider selection uses `#[cfg(feature = "coreml")]` / `#[cfg(feature = "cuda")]` blocks in `inference/mod.rs`; default falls through to CPU EP

### Audio format support
- File transcription: WAV, M4A/AAC, MP3, OGG/Vorbis, FLAC (via symphonia)
- WebSocket: raw PCM16 binary frames at configurable sample rate (8kHz/16kHz/24kHz/44.1kHz/48kHz, default 48kHz); resampled to 16kHz server-side via rubato
- Auto mono mix for multi-channel files

### Security
- **Loopback bind by default.** `127.0.0.1` only; `--bind-all` / `GIGASTT_ALLOW_BIND_ANY=1` required for non-loopback.
- **Origin allowlist.** Cross-origin callers denied by default; loopback origins always allowed. `--allow-origin` (repeatable) for explicit additions; `--cors-allow-any` for wildcard.
- **Runtime limits configurable via CLI / env** (v0.7.0): `--idle-timeout-secs` (default 300), `--ws-frame-max-bytes` (512 KiB), `--body-limit-bytes` (50 MiB), `--pool-size` (4), `--max-session-secs` (3600), `--shutdown-drain-secs` (10).
- **Per-IP rate limiting** (v0.8.0, opt-in): `--rate-limit-per-minute N` + `--rate-limit-burst` on `/v1/*` (`/health` exempt); HTTP 429 + `Retry-After` when exhausted.
- **Pool saturation backpressure.** REST returns 503 + `Retry-After: 30`; WebSocket error includes `retry_after_ms: 30000`.
- **SHA-256 verification + atomic rename** on both encoder/decoder/joiner model files and the optional speaker diarization model.
- **Internal errors sanitized** — no path or model leakage to clients.
- **Prometheus `/metrics`** (v0.8.0, opt-in via `--metrics`): `gigastt_http_requests_total`, `gigastt_http_request_duration_seconds`.

## Model

GigaAM v3 e2e_rnnt from `istupakov/gigaam-v3-onnx` on HuggingFace:
- Files: `v3_e2e_rnnt_{encoder,decoder,joint}.onnx` + `v3_e2e_rnnt_vocab.txt`
- Encoder: 844MB (FP32) or 210MB (INT8 quantized), Decoder: 4.4MB, Joiner: 2.6MB
- Sample rate: 16kHz, Features: 64 mel bins
- ONNX tensors: encoder out `[1, 768, T]` (channels-first), decoder state `[1, 1, 320]`

### Quantization

Rust-native quantization via `src/quantize.rs` (always compiled since v0.9.0):
```sh
cargo run -- quantize --model-dir ~/.gigastt/models
# Produces: v3_e2e_rnnt_encoder_int8.onnx (~210MB, ~4x smaller, ~43% faster)
```

Engine auto-detects and prefers INT8 if available; falls back to FP32.
`gigastt serve` and `gigastt download` invoke the same pipeline automatically on first run unless `--skip-quantize` / `GIGASTT_SKIP_QUANTIZE=1` is set.

## Known limitations (v0.9)
- CPU EP runs on any platform; CoreML EP requires macOS ARM64; CUDA EP requires Linux x86_64 with CUDA 12+
- `protoc` must be on `PATH` at build time (in-tree ONNX quantization pipeline regenerates types via `prost-build`)
- Hot-reload of the INT8 encoder after `serve` boot still requires a restart (tracked as open item `19` in `specs/todo.md`).
