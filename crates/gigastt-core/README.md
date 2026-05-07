# gigastt-core

Core inference engine for [gigastt](https://github.com/ekhodzitsky/gigastt) — Russian speech recognition powered by GigaAM v3 via ONNX Runtime. No server dependencies, no tokio runtime requirement for inference — embed directly into any Rust application.

## Usage

```toml
[dependencies]
gigastt-core = "2.0"
```

```rust,ignore
use gigastt_core::inference::Engine;
use gigastt_core::model;

// Download model on first run (~850 MB)
let model_dir = model::default_model_dir();
model::ensure_model(&model_dir, false, |p| {
    println!("Downloading: {:.0}%", p.percent());
}).await?;

// Load engine (pool_size controls concurrent sessions)
let engine = Engine::load(&model_dir, 1)?;

// Transcribe a file
let mut guard = engine.pool.checkout().await?;
let text = engine.transcribe_file("recording.wav", &mut guard)?;
println!("{text}");
// guard is returned to the pool on drop
```

### Streaming recognition

```rust,ignore
use gigastt_core::inference::Engine;

let engine = Engine::load(&model_dir, 1)?;
let mut guard = engine.pool.checkout().await?;
let mut state = engine.create_state(&mut guard, false)?;

// Feed PCM16 chunks (16 kHz mono)
let segments = engine.process_chunk(&mut guard, &mut state, &pcm16_bytes, 16000)?;
for seg in &segments {
    println!("[partial] {}", seg.text);
}

// Flush remaining audio
let final_segments = engine.flush_state(&mut guard, &mut state)?;
```

## Features

| Feature | Description |
|---|---|
| `diarization` | Speaker identification via polyvoice (default: enabled) |
| `coreml` | CoreML + Neural Engine on macOS ARM64 |
| `cuda` | CUDA 12+ on Linux x86_64 |
| `nnapi` | Android NNAPI for NPU/DSP acceleration |

Features are compile-time and mutually exclusive (`coreml` / `cuda`).

## What's included

- **Inference engine** — ONNX Runtime session pool, Conformer encoder, RNN-T decoder + joiner
- **Mel spectrogram** — 64 bins, FFT=320, hop=160, HTK scale
- **BPE tokenizer** — 1025 tokens with automatic punctuation
- **Audio loading** — WAV, M4A, MP3, OGG, FLAC via symphonia; resampling via rubato
- **Model download** — streaming from HuggingFace with SHA-256 verification + atomic rename
- **INT8 quantization** — native Rust quantizer, auto-detected at runtime
- **Protocol types** — `ClientMessage`, `ServerMessage`, `TranscriptSegment` for WebSocket/REST

## Requirements

- Rust 1.85+ (edition 2024)
- `protoc` on PATH (`brew install protobuf` / `apt install protobuf-compiler`)

## License

MIT
