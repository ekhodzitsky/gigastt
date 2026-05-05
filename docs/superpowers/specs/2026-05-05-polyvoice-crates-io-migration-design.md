# Polyvoice: path -> crates.io migration + default feature

**Date:** 2026-05-05
**Status:** Approved

## Goal

Replace the local path dependency on polyvoice (`path = "../polyvoice"`) with the published crates.io version (`0.4.3`), fix API mismatches introduced by polyvoice's validated wrapper types, and make `diarization` a default feature.

## Scope

- Cargo.toml dependency and feature changes
- API adaptation in `src/inference/mod.rs`
- CLI flag changes in `src/main.rs` (replace `--diarization` with `--skip-diarization`)
- No new tests, no protocol changes, no cfg-gate removal

## Changes

### 1. Cargo.toml

**Dependency:**
```toml
# Before
polyvoice = { path = "../polyvoice", features = ["onnx"], optional = true }

# After
polyvoice = { version = "0.4.3", features = ["onnx"], optional = true }
```

**Default features:**
```toml
# Before
default = ["server"]

# After
default = ["server", "diarization"]
```

The `diarization = ["dep:polyvoice"]` feature line stays unchanged. polyvoice remains `optional = true` so `--no-default-features --features server` still compiles without it.

### 2. API fixes in src/inference/mod.rs

**Import:**
```rust
// Before
use polyvoice::{DiarizationConfig as DiaConfig, OfflineDiarizer, OnlineDiarizer, OnnxEmbeddingExtractor};

// After
use polyvoice::{DiarizationConfig as DiaConfig, OfflineDiarizer, OnlineDiarizer, OnnxEmbeddingExtractor, SampleRate};
```

**OnlineDiarizer config:**
```rust
// Before
OnlineDiarizer::new(DiaConfig {
    threshold: 0.5,
    max_speakers: 64,
    window_secs: 1.5,
    hop_secs: 0.75,
    min_speech_secs: 0.25,
    sample_rate: 16000,
})

// After
OnlineDiarizer::new(DiaConfig {
    threshold: 0.5,
    max_speakers: 64,
    window_secs: 1.5,
    hop_secs: 0.75,
    min_speech_secs: 0.25,
    max_gap_secs: 0.5,
    sample_rate: SampleRate::new(16000).expect("16kHz is valid"),
})
```

**OfflineDiarizer config:** no changes needed — uses `DiaConfig::default()` (line 1029), which already returns correct `SampleRate` and `max_gap_secs` values.

### 3. CLI changes in src/main.rs

**Model download:**
- Remove `--diarization` flag from `download` subcommand
- Speaker model downloads by default (when `diarization` feature is enabled)
- Add `--skip-diarization` flag to opt out of downloading the speaker model

**Server:**
- No changes — `gigastt serve` already auto-loads speaker encoder if model file exists, graceful degradation otherwise

### 4. cfg-gates

All `#[cfg(feature = "diarization")]` blocks remain unchanged. This preserves the ability to build without diarization via `--no-default-features --features server`.

### 5. Testing

- `cargo test` — now runs diarization unit tests by default
- `cargo test --no-default-features --features server` — verify cfg-gates still compile
- `cargo clippy` — zero warnings
- CI workflows: check for explicit `--features diarization` flags that become redundant (harmless but noisy)

## Out of scope

- Adopting additional polyvoice types (SpeakerId, Confidence, SpeakerTurn) in gigastt's protocol
- Removing cfg-gates
- New tests
- Protocol/server response changes

## Risks

- **Low:** crates.io 0.4.3 API may have subtle differences beyond sample_rate/max_gap_secs. Mitigation: `cargo check --features diarization` will catch at compile time.
- **Low:** increased default binary size (~26.5 MB model + polyvoice code). Mitigation: `--no-default-features` escape hatch exists.
