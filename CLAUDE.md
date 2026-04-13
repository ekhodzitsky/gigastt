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
cargo test                           # Run all 72 unit tests, CPU (no model required)
cargo test --features coreml         # Same tests with CoreML EP enabled (macOS)
cargo test --test e2e_rest --test e2e_ws --test e2e_errors --test e2e_shutdown -- --ignored --test-threads=1  # E2E tests (requires model)
cargo test --test load_test -- --ignored           # Load tests (requires model, local only)
cargo test --test soak_test -- --ignored           # Soak test (requires model, local only)
cargo test --test server_integration -- --ignored  # Legacy integration tests (requires model)
cargo clippy             # Lint (no expected warnings)
```

Model download (required for E2E testing and file transcription, ~850MB):
```sh
cargo run -- download                    # Downloads to ~/.gigastt/models/
python scripts/quantize.py               # Optional: generate INT8 encoder (~210MB)
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
```

The Dockerfile uses `--host 0.0.0.0` to allow container networking. Local deployments should use `--host 127.0.0.1` (default).

## Architecture

```
src/
  lib.rs                  # Public module exports
  main.rs                 # CLI (clap): serve, download, transcribe, quantize
  model/mod.rs            # HuggingFace model download (streaming to disk)
  inference/
    mod.rs                # Engine: ONNX session management, StreamingState, DecoderState
    features.rs           # Mel spectrogram (64 bins, FFT=320, hop=160, HTK)
    tokenizer.rs          # BPE tokenizer (1025 tokens)
    decode.rs             # RNN-T greedy decode loop
    audio.rs              # Audio loading, resampling, channel mixing (symphonia + rubato)
    diarization.rs        # Speaker diarization (optional feature)
  error.rs                # Typed error types (GigasttError)
  server/
    mod.rs                # axum router: HTTP + WebSocket on single port
    http.rs               # REST handlers: /health, /v1/transcribe, /v1/transcribe/stream (SSE)
  protocol/mod.rs         # JSON message types (Ready, Partial, Final, Error)
```

### Performance optimizations (v0.4)
- **CoreML execution provider** (`--features coreml`, macOS ARM64): MLProgram format + Neural Engine + model cache directory
  - Automatically loads quantized encoder if available (~4x smaller, ~43% faster)
  - Caches compiled models in `~/.gigastt/models/coreml_cache/`
- **CUDA execution provider** (`--features cuda`, Linux x86_64 CUDA 12+): GPU inference via ONNX Runtime CUDA EP
  - Features are compile-time and mutually exclusive; default build uses CPU EP on all platforms
- **INT8 quantization** (optional): encoder_int8.onnx (~210MB vs 844MB)
  - Run `python scripts/quantize.py` to generate (requires onnxruntime)
  - Auto-detection: Engine uses INT8 encoder if present, falls back to FP32

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

## Development guidelines

### TDD workflow
1. Write failing test first
2. Implement minimal code to pass
3. Refactor, verify tests still pass
4. `cargo test && cargo clippy` before every commit

### API versioning & backward compatibility
- WebSocket protocol version: `PROTOCOL_VERSION = "1.0"` (in `protocol/mod.rs`)
- `ServerMessage::Ready` includes `version` field sent on connection
- WebSocket protocol messages are versioned via `type` field
- New fields are additive only (never remove or rename existing fields)
- Breaking changes require new message type, not modification of existing
- Deprecation: add `deprecated: true` field, support old format for 2 minor versions

### Testing

Three-tier test architecture:

**Unit tests** (no model required, run in CI on every PR):
- Live in `#[cfg(test)] mod tests` at bottom of each file
- Use synthetic data, test names: `test_<what>_<expected_behavior>`
- 72 unit tests across 11 modules
- `cargo test` — runs all unit tests

**E2E tests** (require model ~850MB, run in CI on main push only):
- `tests/e2e_rest.rs` — 8 REST API tests (health, transcribe, SSE streaming, error paths)
- `tests/e2e_ws.rs` — 9 WebSocket protocol tests (ready, audio, stop, configure, errors, concurrent)
- `tests/e2e_errors.rs` — 5 error path tests (oversized body/frame, pool saturation, idle timeout)
- `tests/e2e_shutdown.rs` — 2 graceful shutdown tests
- `tests/common/mod.rs` — shared helpers (start_server with shutdown handle, WAV generation, WS connect)
- `cargo test --test e2e_rest --test e2e_ws --test e2e_errors --test e2e_shutdown -- --ignored --test-threads=1` — all e2e tests

**Load/soak tests** (require model, run locally only):
- `tests/load_test.rs` — 3 load tests (concurrent WS, concurrent REST, burst connections)
- `tests/soak_test.rs` — 1 soak test (continuous WS cycling, configurable via `GIGASTT_SOAK_DURATION_SECS`)
- `cargo test --test load_test -- --ignored` / `cargo test --test soak_test -- --ignored`

**Legacy integration tests** (kept for backward compat):
- `tests/server_integration.rs` — 6 tests, `#[ignore]`, `cargo test --test server_integration -- --ignored`

**Benchmark suite:**
- `tests/benchmark.rs` — WER evaluation on 15 Golos fixtures (custom harness, `harness = false`)

### CI structure
- **PR CI** (fast): clippy, unit tests, feature compile checks (CoreML, CUDA, diarization), audit
- **Main push CI** (full): all PR checks + e2e tests with cached model (~850MB, OS-independent cache key)
- Load/soak tests are local-only, not in CI

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
- Server binds 127.0.0.1 only (local)
- WebSocket frame limit: 512KB
- Connection semaphore: max 4 concurrent
- Audio buffer cap: 5 seconds (OOM protection)
- File transcription cap: 10 minutes (OOM protection)
- Internal errors hidden from clients (generic message sent)

## Model

GigaAM v3 e2e_rnnt from `istupakov/gigaam-v3-onnx` on HuggingFace:
- Files: `v3_e2e_rnnt_{encoder,decoder,joint}.onnx` + `v3_e2e_rnnt_vocab.txt`
- Encoder: 844MB (FP32) or 210MB (INT8 quantized), Decoder: 4.4MB, Joiner: 2.6MB
- Sample rate: 16kHz, Features: 64 mel bins
- ONNX tensors: encoder out `[1, 768, T]` (channels-first), decoder state `[1, 1, 320]`

### Quantization (optional)

`scripts/quantize.py` generates INT8 quantized encoder (QInt8, per-channel):
```sh
pip install onnxruntime
python scripts/quantize.py --model-dir ~/.gigastt/models
# Produces: v3_e2e_rnnt_encoder_int8.onnx (~210MB, ~4x smaller, ~43% faster)
```

Engine auto-detects and prefers INT8 if available; falls back to FP32.

## Known limitations (v0.4)
- CPU EP runs on any platform; CoreML EP requires macOS ARM64; CUDA EP requires Linux x86_64 with CUDA 12+
