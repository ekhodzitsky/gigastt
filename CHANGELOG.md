# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/ekhodzitsky/gigastt/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/ekhodzitsky/gigastt/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ekhodzitsky/gigastt/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ekhodzitsky/gigastt/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/ekhodzitsky/gigastt/releases/tag/v0.1.2
