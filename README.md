<p align="center">
  <h1 align="center">gigastt</h1>
  <p align="center">Local speech-to-text server powered by GigaAM v3 — on-device Russian speech recognition</p>
  <p align="center">
    <a href="https://github.com/ekhodzitsky/gigastt/actions"><img src="https://github.com/ekhodzitsky/gigastt/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="https://crates.io/crates/gigastt"><img src="https://img.shields.io/crates/v/gigastt.svg" alt="crates.io"></a>
    <a href="https://github.com/ekhodzitsky/gigastt/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  </p>
</p>

## Features

- **Real-time streaming** — partial transcription results via WebSocket as you speak
- **On-device inference** — no cloud APIs, no API keys, zero cost, full privacy
- **5.3% WER on Russian** — GigaAM v3 e2e_rnnt, 3-4× better accuracy than Whisper-large-v3 on Russian benchmarks
- **CoreML & Neural Engine** — Conformer encoder optimized for Apple Silicon via CoreML acceleration
- **Multi-format audio** — WAV, M4A/AAC, MP3, OGG/Vorbis, FLAC support for file transcription
- **INT8 quantization** — reduced memory footprint and faster inference
- **Automatic punctuation** — end-to-end model includes text normalization
- **Docker ready** — containerized deployment with configurable host/port binding
- **Auto-download** — model fetched from HuggingFace on first run (~850MB)

## Quick Start

### Cargo

```sh
cargo install gigastt
gigastt serve
# Listening on ws://127.0.0.1:9876
```

### Docker

```sh
docker build -t gigastt .
docker run -p 9876:9876 gigastt serve --host 0.0.0.0
# Model auto-downloaded on first run (~850MB)
```

## CLI Usage

### Start STT Server

```sh
gigastt serve
# Options:
#   --port 9876              (default: 9876)
#   --host 127.0.0.1         (default: 127.0.0.1, use 0.0.0.0 for Docker)
#   --model-dir ~/.gigastt/models
```

Server binds to local address only by default (127.0.0.1). Use `--host 0.0.0.0` in Docker to accept external connections.

### Transcribe Audio File (Offline)

```sh
gigastt transcribe recording.wav
# Outputs transcribed Russian text to stdout
# Supported: WAV, M4A, MP3, OGG, FLAC (mono or auto-mixed to mono)
```

### Download Model Only

```sh
gigastt download
# Downloads to ~/.gigastt/models/ (~850MB)
```

## WebSocket API

### Connection & Message Flow

Connect to `ws://127.0.0.1:9876` and send PCM16 mono audio frames at 48kHz. Server auto-resamples to 16kHz internally.

```
Client                          Server
  │                               │
  ├──────── connect ────────────→ │
  │                               │
  │ ←────── Ready message ─────── │
  │ {type:"ready", version:"1.0"} │
  │                               │
  ├────── binary frames ────────→ │
  │ (PCM16, 48kHz)                │
  │                               │
  │ ←────── Partial results ────── │
  │ {type:"partial", text:"что"}  │
  │                               │
  │ ←─────── Final result ──────── │
  │ {type:"final", text:"Что?"}   │
  │                               │
  └───────── close ──────────────→ │
```

### Message Types

Full protocol documentation in [`docs/asyncapi.yaml`](docs/asyncapi.yaml).

| Direction | Type | Fields | Notes |
|-----------|------|--------|-------|
| **Server** | `ready` | `model`, `sample_rate`, `version` | Sent on connection. Includes protocol v1.0. |
| **Server** | `partial` | `text`, `timestamp`, `words` | Interim transcription (may change with more audio) |
| **Server** | `final` | `text`, `timestamp`, `words` | Complete utterance with punctuation |
| **Server** | `error` | `message`, `code` | Error occurred; connection may close |
| **Client** | `stop` | — | Request finalization of buffered audio |

### Example Session

```json
{"type": "ready", "model": "gigaam-v3-e2e-rnnt", "sample_rate": 48000, "version": "1.0"}
{"type": "partial", "text": "что такое", "timestamp": 0.5}
{"type": "partial", "text": "что такое Node", "timestamp": 1.2}
{"type": "final", "text": "Что такое Node.js?", "timestamp": 2.1}
```

## Client Examples

See [`examples/`](examples/) for ready-to-use WebSocket clients:

- **Python**: `python examples/python_client.py recording.wav`
- **JavaScript**: `node examples/js_client.mjs recording.wav`

## Performance

### Benchmarks

| Metric | v0.2 |
|--------|------|
| **WER (Russian)** | 5.3% |
| **vs Whisper-large-v3** | 3-4× better |
| **Latency (16s audio)** | ~800ms (M1) |
| **Memory** | ~500MB |

### Acceleration

- **CoreML** — Conformer encoder optimized via ONNX Runtime's CoreML execution provider
- **Neural Engine** — INT8 quantization leverages Apple Neural Engine for 2-3× speedup
- **Streaming** — stateful decoder persists across chunks; no full-audio re-inference needed

## Architecture

```
┌─────────────────────────────────────┐
│ Audio Input (PCM16, 48/16kHz)       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Mel Spectrogram (64 bins)           │
│ FFT=320, hop=160, HTK               │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Conformer Encoder (ONNX)            │
│ 16 layers, d=768, 240M params       │
│ ┌─ CoreML execution (M1/M2/M3/M4)   │
│ └─ INT8 quantized                   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ RNN-T Decoder + Joiner (ONNX)       │
│ ┌─ Stateful: h/c persisted          │
│ └─ Per-chunk processing             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ BPE Tokenizer (1025 tokens)         │
│ + Automatic Punctuation             │
└──────────────┬──────────────────────┘
               │
               ▼
      Final Russian Text
```

## Model

[**GigaAM v3 e2e_rnnt**](https://huggingface.co/istupakov/gigaam-v3-onnx) — Conformer-based RNN-T ASR by [SberDevices](https://github.com/salute-developers/GigaAM):

| Property | Value |
|----------|-------|
| **Architecture** | RNN-T (encoder + decoder + joiner) |
| **Encoder** | 16-layer Conformer, 768-dim, 240M params |
| **Training Data** | 700K+ hours of Russian speech |
| **Vocabulary** | 1025 BPE tokens |
| **Input** | 16kHz mono PCM16 |
| **Quantization** | INT8 (v0.2+) |
| **License** | MIT |
| **Download Size** | ~850MB (encoder 844MB, decoder 4.4MB, joiner 2.6MB) |

## Requirements

- **OS**: macOS 14+ (Sonoma or newer)
- **CPU**: Apple Silicon (M1, M2, M3, M4)
- **Disk**: ~1.5GB (model + binary)
- **RAM**: ~500MB during inference
- **Rust**: 1.85+ (edition 2024, for building from source)

## Installation

### From crates.io

```sh
cargo install gigastt
```

### From source

```sh
git clone https://github.com/ekhodzitsky/gigastt
cd gigastt
cargo install --path .
```

## Build & Development

```sh
cargo build              # Debug build
cargo build --release   # Release (LTO, stripped)
cargo test              # Run tests
cargo clippy            # Lint

# Download model (required for integration tests, ~850MB)
cargo run -- download
```

## License

MIT — see [LICENSE](LICENSE)

## Acknowledgments

- [**GigaAM**](https://github.com/salute-developers/GigaAM) by [SberDevices](https://github.com/salute-developers) — the speech recognition model
- [**onnx-asr**](https://github.com/istupakov/onnx-asr) by [@istupakov](https://github.com/istupakov) — ONNX model export and reference implementation
- [**ONNX Runtime**](https://github.com/microsoft/onnxruntime) — inference engine with CoreML & Neural Engine support
- [**ort**](https://github.com/pykeio/ort) — Rust bindings for ONNX Runtime
