# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/ekhodzitsky/gigastt/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/ekhodzitsky/gigastt/compare/v0.4.3...v0.5.0
[0.4.3]: https://github.com/ekhodzitsky/gigastt/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/ekhodzitsky/gigastt/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/ekhodzitsky/gigastt/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/ekhodzitsky/gigastt/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ekhodzitsky/gigastt/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ekhodzitsky/gigastt/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/ekhodzitsky/gigastt/releases/tag/v0.1.2
