# gigastt-core

Core inference engine for [gigastt](https://github.com/ekhodzitsky/gigastt) — Russian (and optional multilingual) speech recognition powered by GigaAM v3 via ONNX Runtime. No server dependencies; no tokio runtime required for inference itself — embed it in any Rust application.

Runtime is **INT8 only**. Default `rnnt` / `e2e_rnnt` files come from GitHub Releases; `ml_ctc` / `ml_ctc_large` come from HuggingFace, as do the optional speaker-embedding and punctuation sidecars — the Silero VAD sidecar comes from GitHub (`snakers4/silero-vad`).

## Usage

```toml
[dependencies]
gigastt-core = "2.22"
```

```rust,ignore
use gigastt_core::inference::Engine;
use gigastt_core::model;

// Download the lean INT8 bundle on first run (~310 MiB, GitHub Releases)
let model_dir = model::default_model_dir();
model::ensure_model(&model_dir).await?;

// Default pool size is 2 (1 on Android); use load_with_pool_size to override
let engine = Engine::load(&model_dir)?;
// let engine = Engine::load_with_pool_size(&model_dir, 1)?;

let mut guard = engine.pool.checkout().await?;
let result = engine.transcribe_file("recording.wav", &mut guard)?;
println!("{}", result.text);
// guard is returned to the pool on drop
```

### Streaming recognition

`process_chunk` takes **16 kHz mono `f32` samples**, not PCM16 bytes. Convert
or resample before the call (the server does that on the WebSocket path).

```rust,ignore
use gigastt_core::inference::Engine;

let engine = Engine::load(&model_dir)?;
let mut guard = engine.pool.checkout().await?;
let mut state = engine.create_state(false);

// samples: &[f32], 16 kHz mono, any chunk length
let segments = engine.process_chunk(&samples, &mut state, &mut guard)?;
for seg in &segments {
    println!("[{}] {}", if seg.is_final { "final" } else { "partial" }, seg.text);
}

if let Some(tail) = engine.flush_state(&mut state) {
    println!("[final] {}", tail.text);
}
```

Live WebSocket WER is **not** batch-equal — buffered offline RNN-T, not a
native streaming AM. Numbers and protocol:
[docs/benchmarks.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/benchmarks.md#streaming-measurement-protocol).

## Features

Defaults (`diarization`, `net`, `async-pool`, `file-decode`, `quantize`) make
the engine work out of the box. For a lean embedded build that side-loads
INT8 models and feeds raw PCM, disable defaults:

```toml
gigastt-core = { version = "2.22", default-features = false }
```

That drops `tokio`, `reqwest`/HTTP, `symphonia`, `ryf`, and the quantizer (`protoc`)
from the dependency graph. Opt features back in as needed.

| Feature | Default | Description |
|---|---|---|
| `net` | on | HTTP model download (`reqwest` + async fs); off → side-loaded models only |
| `async-pool` | on | async `Pool::checkout`; off → synchronous `checkout_blocking` only (no tokio runtime) |
| `file-decode` | on | file transcription via `ryf` (WAVE family: PCM/IEEE, G.711, G.722, GSM 06.10, ADPCM, RF64) + `symphonia` (MP3/M4A/OGG/FLAC/Opus, WebM/MKV); off → raw-PCM streaming only |
| `diarization` | on | speaker identification via polyvoice |
| `quantize` | on | packaging-only INT8 rebuild from a local FP32 ONNX (`protoc` required) |
| `ort-load-dynamic` | off | link a system/vendored onnxruntime instead of the build-time download |
| `coreml` / `cuda` / `nnapi` | off | ORT execution providers (`coreml` / `cuda` are mutually exclusive) |
| `ane` / `candle` | off | non-ORT backends (Apple Neural Engine `.mlpackage` / Candle+Metal) |

## What's included

- **Inference engine** — ONNX Runtime session pool, Conformer encoder, RNN-T decoder + joiner (or greedy CTC on the multilingual heads)
- **Mel spectrogram** — 64 bins, FFT=320, hop=160, HTK scale
- **Tokenizer** — char vocab 34 (`rnnt`), BPE 1025 (`e2e_rnnt`), multilingual char 71 (`ml_ctc` / `ml_ctc_large`)
- **Audio loading** — WAVE family via `ryf` (PCM/IEEE, G.711, G.722, GSM 06.10, MS/IMA ADPCM, RF64/RIFX/Wave64); M4A, MP3, OGG, FLAC, Opus, WebM/MKV via symphonia; resampling via rubato
- **Model download** — streaming fetch with SHA-256 verification + atomic rename (Releases for default INT8; HuggingFace for CTC / sidecars)
- **Protocol types** — `ClientMessage`, `ServerMessage`, `TranscriptSegment` for WebSocket/REST

## Requirements

- Rust 1.94+ (edition 2024)
- `protoc` on PATH only when the `quantize` feature is on (`brew install protobuf` / `apt install protobuf-compiler`)

## License

MIT
