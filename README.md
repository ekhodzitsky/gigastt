<p align="center">
  <h1 align="center">gigastt</h1>
  <p align="center">Local speech-to-text server powered by <a href="https://github.com/salute-developers/GigaAM">GigaAM v3</a> — on-device Russian speech recognition via ONNX Runtime</p>
  <p align="center">
    <a href="https://github.com/ekhodzitsky/gigastt/actions"><img src="https://github.com/ekhodzitsky/gigastt/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="https://crates.io/crates/gigastt"><img src="https://img.shields.io/crates/v/gigastt.svg" alt="crates.io"></a>
    <a href="https://github.com/ekhodzitsky/gigastt/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  </p>
</p>

<!-- TODO: Record GIF demo with `vhs` or `asciinema` showing real-time streaming transcription -->
<!-- ![demo](assets/demo.gif) -->

## Features

- **Real-time streaming** — partial transcription results appear as you speak via WebSocket
- **On-device** — no cloud APIs, no API keys, zero cost, full privacy
- **Punctuation** — automatic punctuation and text normalization (e2e model)
- **Russian-first** — GigaAM v3 e2e_rnnt, 50% better than Whisper-large-v3 on Russian benchmarks
- **Fast** — Conformer encoder (240M params) optimized for Apple Silicon via CoreML
- **Auto-download** — model fetched from HuggingFace on first run

## Quick Start

```sh
cargo install gigastt
gigastt serve
# Listening on ws://127.0.0.1:9876
# Model auto-downloaded (~850MB) on first run
```

## Usage

### Start STT server

```sh
gigastt serve
# Options: --port 9876 --model-dir ~/.gigastt/models
```

### Transcribe a file

```sh
gigastt transcribe recording.wav
# Outputs transcribed Russian text to stdout
# Requires: mono PCM16 WAV at 16kHz
```

### Download model only

```sh
gigastt download
# Downloads to ~/.gigastt/models/
```

## WebSocket Protocol

Connect to `ws://127.0.0.1:9876` and send PCM16 mono 48kHz binary frames (server resamples to 16kHz internally).

### Messages

| Direction | Type | Fields | Description |
|-----------|------|--------|-------------|
| Server | `ready` | `model`, `sample_rate` | Server is ready to accept audio |
| Server | `partial` | `text`, `timestamp` | Interim transcription (may change) |
| Server | `final` | `text`, `timestamp` | Complete utterance with punctuation |
| Server | `error` | `message`, `code` | Error occurred |

### Example

```json
{"type": "ready", "model": "gigaam-v3-e2e-rnnt", "sample_rate": 16000}
{"type": "partial", "text": "что такое"}
{"type": "final", "text": "Что такое Node.js?"}
```

## Client Examples

See [`examples/`](examples/) for ready-to-use WebSocket clients:

- **Python**: `python examples/python_client.py recording.wav`
- **JavaScript**: `node examples/js_client.mjs recording.wav`

## Model

[GigaAM v3 e2e_rnnt](https://huggingface.co/istupakov/gigaam-v3-onnx) — Conformer-based ASR model by SberDevices:

| Property | Value |
|----------|-------|
| Parameters | 240M (Conformer encoder) |
| Architecture | RNN-T (encoder + decoder + joiner) |
| Training data | 700K+ hours of Russian speech |
| Vocabulary | 1025 BPE tokens |
| Input | 16kHz mono PCM16 |
| License | MIT |

## Requirements

- macOS 14+ on Apple Silicon (M1/M2/M3/M4)
- ~1.5GB disk space (model + binary)
- ~500MB RAM during inference
- Rust 1.75+ (for building from source)

## Architecture

```
Audio (PCM16 16kHz)
  │
  ▼
┌──────────────────┐
│  Mel Spectrogram  │  64 bins, FFT=320, hop=160
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Conformer       │  16 layers, d=768
│  Encoder (ONNX)  │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  RNN-T Decoder   │  LSTM h/c state persisted
│  + Joiner (ONNX) │  across audio chunks
└────────┬─────────┘
         ▼
  Text (with punctuation)
```

## License

MIT

## Acknowledgments

- [GigaAM](https://github.com/salute-developers/GigaAM) by SberDevices — the speech recognition model
- [onnx-asr](https://github.com/istupakov/onnx-asr) by @istupakov — ONNX model export and reference implementation
- [ort](https://github.com/pykeio/ort) — Rust bindings for ONNX Runtime
