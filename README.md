<p align="center">
  <h1 align="center">gigastt</h1>
  <p align="center"><strong>On-device Russian speech recognition with 5.3% WER</strong></p>
  <p align="center">Local STT server powered by GigaAM v3 — no cloud, no API keys, full privacy</p>
  <p align="center">
    <a href="https://github.com/ekhodzitsky/gigastt/actions"><img src="https://github.com/ekhodzitsky/gigastt/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="https://crates.io/crates/gigastt"><img src="https://img.shields.io/crates/v/gigastt.svg" alt="crates.io"></a>
    <a href="https://github.com/ekhodzitsky/gigastt/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
    <a href="https://github.com/ekhodzitsky/gigastt/blob/main/CHANGELOG.md"><img src="https://img.shields.io/badge/changelog-Keep%20a%20Changelog-orange" alt="Changelog"></a>
  </p>
</p>

---

**gigastt** turns any machine into a real-time Russian speech recognition server. One binary, one command, state-of-the-art accuracy — everything runs locally.

```sh
cargo install gigastt && gigastt serve
# WebSocket: ws://127.0.0.1:9876/ws
# REST API:  http://127.0.0.1:9876/v1/transcribe
```

## Why gigastt?

| | gigastt | Whisper large-v3 | Cloud APIs |
|---|:---:|:---:|:---:|
| **WER (Russian)** | **10.4%** | ~18% | 5-10% |
| **Latency (16s audio, M1)** | **~700ms** | ~4s | network-dependent |
| **Streaming** | real-time WebSocket | batch only | varies |
| **Privacy** | 100% local | 100% local | data leaves device |
| **Cost** | free forever | free | $0.006/min+ |
| **Setup** | `cargo install` | Python + deps | API key + billing |
| **Binary size** | single static binary | Python runtime | N/A |
| **INT8 quantization** | auto, 0% WER loss | manual | N/A |
| **Concurrent sessions** | 4 (configurable) | 1 | unlimited |

> GigaAM v3 was trained on **700K+ hours** of Russian speech. It delivers better accuracy than Whisper-large-v3 on Russian benchmarks while running faster on Apple Silicon and NVIDIA GPUs. WER measured on 993 Golos crowd-sourced samples (4991 words).

## Features

- **Real-time streaming** — partial transcription via WebSocket as you speak
- **REST API + SSE** — file transcription with instant or streaming response
- **Hardware acceleration** — CoreML + Neural Engine (macOS), CUDA 12+ (Linux), CPU everywhere
- **INT8 quantization** — 4x smaller model, 43% faster inference
- **Multi-format audio** — WAV, M4A/AAC, MP3, OGG/Vorbis, FLAC
- **Speaker diarization** — identify who said what (optional feature)
- **Automatic punctuation** — GigaAM v3 model produces punctuated, normalized text
- **Auto-download** — model fetched from HuggingFace on first run (~850 MB)
- **Docker ready** — CPU and CUDA images with multi-stage builds
- **Hardened** — connection limits, frame caps, idle timeouts, sanitized errors

## Quick Start

### Install & Run

```sh
# From crates.io
cargo install gigastt
gigastt serve

# From source
git clone https://github.com/ekhodzitsky/gigastt
cd gigastt
cargo run --release -- serve
```

The model (~850 MB) downloads automatically on first run.

### Docker

```sh
# CPU (any platform)
docker build -t gigastt .
docker run -p 9876:9876 gigastt

# CUDA (Linux, requires NVIDIA Container Toolkit)
docker build -f Dockerfile.cuda -t gigastt-cuda .
docker run --gpus all -p 9876:9876 gigastt-cuda

# Model auto-downloads on first run (~850 MB)
```

### Transcribe a File

```sh
# CLI
gigastt transcribe recording.wav

# REST API
curl -X POST http://127.0.0.1:9876/v1/transcribe \
  -H "Content-Type: application/octet-stream" \
  --data-binary @recording.wav
# {"text":"Привет, как дела?","words":[],"duration":3.5}
```

## API

### WebSocket — Real-time Streaming

Connect to `ws://127.0.0.1:9876/ws`, send PCM16 audio frames, receive transcription in real time.

```
Client                            Server
  |                                 |
  |-------- connect --------------> |
  |                                 |
  | <------- ready ----------------- |
  | {type:"ready", version:"1.0"}  |
  |                                 |
  |------- configure (optional) --> |
  | {type:"configure",              |
  |  sample_rate:16000}             |
  |                                 |
  |-------- binary PCM16 --------> |
  |                                 |
  | <------- partial --------------- |
  | {type:"partial", text:"привет"} |
  |                                 |
  | <------- final ----------------- |
  | {type:"final",                  |
  |  text:"Привет, как дела?"}      |
```

**Supported sample rates:** 8, 16, 24, 44.1, 48 kHz (default 48 kHz, resampled to 16 kHz internally).

### REST API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check (`{"status":"ok"}`) |
| `/v1/transcribe` | POST | File transcription, full JSON response |
| `/v1/transcribe/stream` | POST | File transcription with SSE streaming |
| `/ws` | GET | WebSocket upgrade for real-time streaming |

**SSE streaming example:**

```sh
curl -X POST http://127.0.0.1:9876/v1/transcribe/stream \
  -H "Content-Type: application/octet-stream" \
  --data-binary @recording.wav
# data: {"type":"partial","text":"привет как"}
# data: {"type":"partial","text":"привет как дела"}
# data: {"type":"final","text":"Привет, как дела?"}
```

Full protocol spec: [`docs/asyncapi.yaml`](docs/asyncapi.yaml)

### Client Libraries

Ready-to-use WebSocket clients in [`examples/`](examples/):

```sh
# Python
pip install websockets
python examples/python_client.py recording.wav

# JavaScript (Node.js)
npm install ws
node examples/js_client.mjs recording.wav
```

## Performance

| Metric | Value |
|---|---|
| **WER (Russian)** | 10.4% (993 Golos crowd samples, 4991 words) |
| **INT8 vs FP32** | 0% WER degradation (10.4% vs 10.5% on 993 samples) |
| **Latency (16s audio, M1)** | ~700 ms (encoder 667 ms + decode 31 ms) |
| **Memory (RSS)** | ~560 MB |
| **Model size** | 851 MB (FP32) / 222 MB (INT8) |
| **Concurrent sessions** | up to 4 (configurable via `--pool-size`) |

### Hardware Acceleration

| Platform | Feature flag | Execution Provider |
|---|---|---|
| macOS ARM64 (M1-M4) | `--features coreml` | CoreML + Neural Engine |
| Linux x86_64 + NVIDIA | `--features cuda` | CUDA 12+ |
| Any platform | _(default)_ | CPU |

```sh
cargo build --release --features coreml   # macOS: CoreML + Neural Engine
cargo build --release --features cuda     # Linux: NVIDIA CUDA 12+
cargo build --release                     # CPU (any platform)
```

Features are compile-time and mutually exclusive.

### INT8 Quantization

Optional quantized encoder: 4x smaller, ~43% faster, 0% WER degradation (verified on 993 Golos samples / 4991 words). Auto-detected at runtime.

When built with `--features quantize`, INT8 encoder is created automatically on first `download` or `serve` — no manual steps needed.

```sh
# Automatic (recommended)
cargo install gigastt --features quantize
gigastt serve   # downloads model + auto-quantizes on first run

# Manual
gigastt quantize                     # native Rust quantization
python scripts/quantize.py           # legacy Python alternative
```

## Architecture

```
                    Audio Input
                   (PCM16, multi-rate)
                        |
                        v
               +-----------------+
               | Mel Spectrogram |  64 bins, FFT=320, hop=160
               +-----------------+
                        |
                        v
            +------------------------+
            |   Conformer Encoder    |  16 layers, 768-dim, 240M params
            |  (ONNX Runtime)        |  CoreML | CUDA | CPU
            +------------------------+
                        |
                        v
            +------------------------+
            | RNN-T Decoder + Joiner |  Stateful: h/c persisted
            |  (ONNX Runtime)        |  across streaming chunks
            +------------------------+
                        |
                        v
            +------------------------+
            |   BPE Tokenizer        |  1025 tokens
            |   + Auto-punctuation   |
            +------------------------+
                        |
                        v
                  Russian Text
```

## CLI Reference

```
gigastt <COMMAND>

Commands:
  serve        Start STT server
  download     Download model (~850 MB)
  transcribe   Transcribe audio file (offline)

gigastt serve [OPTIONS]
  --port <PORT>          Listen port [default: 9876]
  --host <HOST>          Bind address [default: 127.0.0.1]
  --model-dir <DIR>      Model directory [default: ~/.gigastt/models]
  --pool-size <N>        Concurrent inference sessions [default: 4]

gigastt transcribe <FILE>
  Supports: WAV, M4A, MP3, OGG, FLAC (mono or auto-mixed)

gigastt quantize [OPTIONS]          # requires --features quantize
  --model-dir <DIR>      Model directory [default: ~/.gigastt/models]
  --force                Re-quantize even if INT8 model exists
```

## Model

[**GigaAM v3 e2e_rnnt**](https://huggingface.co/istupakov/gigaam-v3-onnx) by [SberDevices](https://github.com/salute-developers/GigaAM):

| Property | Value |
|---|---|
| Architecture | RNN-T (Conformer encoder + LSTM decoder + joiner) |
| Encoder | 16-layer Conformer, 768-dim, 240M params |
| Training data | 700K+ hours of Russian speech |
| Vocabulary | 1025 BPE tokens |
| Input | 16 kHz mono PCM16 |
| Quantization | INT8 available (v0.2+) |
| License | MIT |
| Download | ~850 MB (encoder 844 MB, decoder 4.4 MB, joiner 2.6 MB) |

## Requirements

| | macOS ARM64 | Linux x86_64 |
|---|---|---|
| **OS** | macOS 14+ (Sonoma) | Any modern distro |
| **CPU** | Apple Silicon (M1-M4) | x86_64 |
| **GPU** | _(integrated, via CoreML)_ | NVIDIA + CUDA 12+ (optional) |
| **Disk** | ~1.5 GB | ~1.5 GB |
| **RAM** | ~560 MB | ~560 MB |
| **Rust** | 1.85+ | 1.85+ |

## Security

- Binds to `127.0.0.1` only by default (localhost)
- WebSocket frame limit: 512 KB
- Connection semaphore: max 4 concurrent sessions
- Audio buffer cap: 5 s (streaming) / 10 min (file upload)
- Internal errors sanitized — no path or model leakage to clients
- Idle connection timeout: 300 s

## Testing

72 unit tests + 28 e2e tests + load & soak tests:

```sh
cargo test                           # 72 unit tests (no model needed)
cargo clippy                         # Lint (zero warnings)

# E2E tests (require model, serial to avoid OOM)
cargo run -- download
cargo test --test e2e_rest --test e2e_ws --test e2e_errors --test e2e_shutdown -- --ignored --test-threads=1

# Load & soak (local only)
cargo test --test load_test -- --ignored
cargo test --test soak_test -- --ignored
```

## License

MIT — see [LICENSE](LICENSE)

## Acknowledgments

- [**GigaAM**](https://github.com/salute-developers/GigaAM) by [SberDevices](https://github.com/salute-developers) — the speech recognition model
- [**onnx-asr**](https://github.com/istupakov/onnx-asr) by [@istupakov](https://github.com/istupakov) — ONNX model export and reference
- [**ONNX Runtime**](https://github.com/microsoft/onnxruntime) — inference engine
- [**ort**](https://github.com/pykeio/ort) — Rust bindings for ONNX Runtime
