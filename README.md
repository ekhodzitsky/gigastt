# gigastt

Local speech-to-text server powered by [GigaAM v3](https://github.com/salute-developers/GigaAM) — on-device Russian speech recognition via ONNX Runtime.

## Features

- **On-device** — no cloud APIs, no API keys, zero cost
- **Streaming** — real-time transcription via WebSocket
- **Punctuation** — automatic punctuation and text normalization (e2e model)
- **Russian** — built specifically for Russian speech (GigaAM v3 e2e_rnnt)
- **Auto-download** — model downloaded from HuggingFace on first run
- **macOS ARM64** — optimized for Apple Silicon via CoreML

## Install

```sh
cargo install gigastt
```

## Usage

### Start STT server

```sh
gigastt serve
# Listening on ws://127.0.0.1:9876
# Model auto-downloaded to ~/.gigastt/models/ on first run
```

### Download model only

```sh
gigastt download
```

### Transcribe a file

```sh
gigastt transcribe recording.wav
```

## WebSocket Protocol

Connect to `ws://127.0.0.1:9876` and send PCM16 mono 48kHz binary frames.

### Server messages

```json
{"type": "ready", "model": "gigaam-v3-e2e-rnnt", "sample_rate": 48000}
{"type": "partial", "text": "Что такое", "timestamp": 1234567890.123}
{"type": "final", "text": "Что такое Node.js?", "timestamp": 1234567890.456}
{"type": "error", "message": "...", "code": "inference_error"}
```

## Model

Uses [GigaAM v3 e2e_rnnt](https://huggingface.co/ai-sage/GigaAM-v3) (~1GB ONNX):
- 240M parameters, Conformer architecture
- Pre-trained on 700K hours of Russian speech
- 50% better than Whisper-large-v3 on Russian benchmarks
- MIT license

## Requirements

- macOS 14+ on Apple Silicon (M1/M2/M3/M4)
- ~1.5GB disk space (model)
- ~500MB RAM during inference

## License

MIT
