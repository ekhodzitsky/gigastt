# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**gigastt** — local speech-to-text server powered by GigaAM v3 e2e_rnnt. On-device Russian speech recognition via ONNX Runtime. No cloud APIs, no API keys, full privacy.

- **Repository**: https://github.com/ekhodzitsky/gigastt
- **crates.io**: https://crates.io/crates/gigastt
- **License**: MIT

## Build & Test

```sh
cargo build              # Debug build
cargo build --release    # Release build (LTO, stripped)
cargo test               # Run all tests
cargo clippy             # Lint (only `ClientMessage` dead_code warning is expected)
```

Model download (required for E2E testing, ~850MB):
```sh
cargo run -- download    # Downloads to ~/.gigastt/models/
```

## Architecture

```
src/
  main.rs                 # CLI (clap): serve, download, transcribe
  model/mod.rs            # HuggingFace model download (streaming to disk)
  inference/
    mod.rs                # Engine: ONNX session management, StreamingState, DecoderState
    features.rs           # Mel spectrogram (64 bins, FFT=320, hop=160, HTK)
    tokenizer.rs          # BPE tokenizer (1025 tokens)
    decode.rs             # RNN-T greedy decode loop
  server/mod.rs           # WebSocket server (tokio + tungstenite)
  protocol/mod.rs         # JSON message types (Ready, Partial, Final, Error)
```

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
- Server accepts 48kHz, resamples to 16kHz internally

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
- Unit tests live in `#[cfg(test)] mod tests` at bottom of each file
- Tests use synthetic data (no model download required)
- Test names: `test_<what>_<expected_behavior>`
- Current: 39 tests (tokenizer: 5, features: 4, decode: 9, inference: 16, protocol: 6). All modules covered

### Code style
- Rust 2024 edition
- `anyhow` for error handling, `tracing` for logging
- No `unwrap()` in production paths (use `?`, `context()`, or `unwrap_or_else`)
- Shared constants in `inference/mod.rs`, referenced by sub-modules
- `ort` errors wrapped via `ort_err()` helper (Send/Sync workaround)

### Audio format support
- File transcription: WAV, M4A/AAC, MP3, OGG/Vorbis, FLAC (via symphonia)
- WebSocket: raw PCM16 binary frames at 48kHz (resampled server-side)
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
- Encoder: 844MB, Decoder: 4.4MB, Joiner: 2.6MB
- Sample rate: 16kHz, Features: 64 mel bins
- ONNX tensors: encoder out `[1, 768, T]` (channels-first), decoder state `[1, 1, 320]`

## Known limitations
- macOS ARM64 only (CoreML feature in ort)
- Linear interpolation resampler (upgrade to polyphase FIR for better quality)
