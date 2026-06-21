# gigastt-core

Core inference engine for [gigastt](https://github.com/ekhodzitsky/gigastt) — Russian speech recognition powered by GigaAM v3 via ONNX Runtime. No server dependencies, no tokio runtime requirement for inference — embed directly into any Rust application.

## Usage

```toml
[dependencies]
gigastt-core = "2.3"
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

Defaults (`diarization`, `net`, `async-pool`, `file-decode`) make the engine work out of the box. For a lean embedded build that side-loads models and feeds raw PCM, disable defaults:

```toml
gigastt-core = { version = "2.3", default-features = false }
```

That drops `tokio`, `reqwest`/HTTP, and `symphonia` from the dependency graph. Opt features back in as needed.

| Feature | Default | Description |
|---|---|---|
| `net` | on | HTTP model download (`reqwest` + async fs); off → side-loaded models only |
| `async-pool` | on | async `Pool::checkout`; off → synchronous `checkout_blocking` only (no tokio runtime) |
| `file-decode` | on | file transcription via `symphonia` (WAV/MP3/M4A/OGG/FLAC); off → raw-PCM streaming only |
| `diarization` | on | speaker identification via polyvoice |
| `ort-load-dynamic` | off | link a system/vendored onnxruntime instead of the build-time download |
| `coreml` / `cuda` / `nnapi` | off | hardware acceleration (`coreml` / `cuda` are mutually exclusive) |

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
