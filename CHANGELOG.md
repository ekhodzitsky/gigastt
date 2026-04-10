# Changelog

## 0.3.0

### Breaking Changes

- `Engine::load()`, `Engine::process_chunk()`, and `Engine::transcribe_file()` now return `Result<T, GigasttError>` instead of `anyhow::Result<T>`.
  - **No change needed** if you use `?` in `-> anyhow::Result<T>` contexts (blanket `From<E: Error>` covers it).
  - If you call `.context()` on Engine results, replace with `.map_err(|e| anyhow::anyhow!(e).context("..."))`.
- All public structs and enums are now `#[non_exhaustive]`. External code that constructs these types via struct literals will need to use the provided constructor methods instead.

### Added

- `GigasttError` enum (`error` module) with variants: `ModelLoad`, `Inference`, `InvalidAudio`, `Io` — enables `match`-based error handling for library consumers.
- `#[non_exhaustive]` on `DecoderState`, `StreamingState`, `WordInfo`, `TranscriptSegment`, `ServerMessage`, `ClientMessage`, `GigasttError` — future field/variant additions are non-breaking.
- Comprehensive `///` rustdoc on all public types, functions, fields, and constants.
- Crate-level documentation with quick-start examples in `lib.rs`.
- Stress tests for NaN/infinity audio samples, empty inputs, and buffer boundary conditions.

### Fixed

- Potential panic on odd-length WebSocket binary frames (`chunks_exact(2)` now drops trailing byte with warning).
- Non-finite audio samples (NaN, infinity) in `resample()` are now replaced with zeros instead of propagating.

### Unchanged

- `server::run()` and `model::ensure_model()` continue to return `anyhow::Result` (application-level functions).
- All 39 original unit tests pass without modification.

## 0.2.0

- Partial transcripts, endpointing, word timestamps, confidence scores.
- CoreML execution provider (macOS ARM64).
- INT8 quantized encoder support.

## 0.1.2

- Initial release.
