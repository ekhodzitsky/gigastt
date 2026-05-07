<p align="center">
  <h1 align="center">gigastt</h1>
  <p align="center"><strong>On-device Russian speech recognition with 10.4% WER</strong></p>
  <p align="center">Local STT server powered by GigaAM v3 — no cloud, no API keys, full privacy</p>
  <p align="center">
    <a href="https://github.com/ekhodzitsky/gigastt/actions"><img src="https://github.com/ekhodzitsky/gigastt/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="https://crates.io/crates/gigastt"><img src="https://img.shields.io/crates/v/gigastt.svg" alt="crates.io"></a>
    <a href="https://github.com/ekhodzitsky/gigastt/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
    <a href="https://github.com/ekhodzitsky/gigastt/blob/main/CHANGELOG.md"><img src="https://img.shields.io/badge/changelog-Keep%20a%20Changelog-orange" alt="Changelog"></a>
  <p align="center"><b>English</b> | <a href="README_RU.md">Русский</a></p>
</p>

<p align="center">
  <sub>Latest: <b>v2.0.1</b> — see <a href="CHANGELOG.md">CHANGELOG</a>.</sub>
</p>

---

**gigastt** turns any machine into a real-time Russian speech recognition server. One binary, one command, state-of-the-art accuracy — everything runs locally.

```sh
cargo install gigastt && gigastt serve
# WebSocket: ws://127.0.0.1:9876/v1/ws
# REST API:  http://127.0.0.1:9876/v1/transcribe
```

### Demo

```sh
$ gigastt transcribe recording.wav
Привет, как дела?

$ curl -X POST http://127.0.0.1:9876/v1/transcribe \
    -H "Content-Type: application/octet-stream" \
    --data-binary @recording.wav
{"text":"Привет, как дела?","words":[],"duration":3.5}
```

## Why gigastt?

| | gigastt | whisper.cpp | faster-whisper | Vosk | sherpa-onnx | Cloud APIs |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Model** | GigaAM v3 | Whisper large-v3 | Whisper large-v3 | Vosk models | varies | vendor |
| **WER (Russian)** | **10.4%** | ~18% | ~18% | ~20%+ | model-dependent | 5–10% |
| **Languages** | Russian | 99 | 99 | 20+ | 10+ | 100+ |
| **Streaming** | real-time WebSocket | — | — | WebSocket + gRPC | WebSocket + TCP | varies |
| **Latency (16s, M1)** | **~700ms** | ~4s | ~2s | ~3s | ~1.5s | network |
| **Privacy** | 100% local | 100% local | 100% local | 100% local | 100% local | data leaves device |
| **Setup** | `cargo install` | cmake + make | `pip install` | `pip install` | cmake or pip | API key + billing |
| **Implementation** | Rust | C/C++ | Python/C++ | C++/Java | C++ | N/A |
| **Bindings** | Rust, C FFI | C, Python, Go, JS… | Python | Python, Java, JS, Go… | C, Python, Java, Swift… | SDK per vendor |
| **INT8 quantization** | auto, 0% WER loss | GGML quant | CTranslate2 quant | — | — | N/A |
| **Concurrent sessions** | configurable pool | 1 | 1 | 1 | 1 | provider limits |
| **Cost** | free | free | free | free | free | $0.006/min+ |

> **Trade-off:** gigastt supports Russian only. If you need multilingual recognition, consider whisper.cpp or sherpa-onnx. If you need the best Russian accuracy running locally — gigastt is the only Rust-native option built on GigaAM v3, the current SOTA for Russian ASR. Trained on **700K+ hours** of Russian speech. WER measured on 993 Golos crowd-sourced samples (4991 words).

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
# Homebrew (macOS ARM64 / Linux x86_64)
brew tap ekhodzitsky/gigastt https://github.com/ekhodzitsky/gigastt
brew install gigastt
gigastt serve

# From crates.io (requires `protoc` on PATH: `brew install protobuf` / `apt install protobuf-compiler`)
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
# CPU — model auto-downloads on first run (~850 MB)
docker build -t gigastt .
docker run -p 9876:9876 gigastt

# CUDA (Linux, requires NVIDIA Container Toolkit)
docker build -f Dockerfile.cuda -t gigastt-cuda .
docker run --gpus all -p 9876:9876 gigastt-cuda

# Baked image — model included at build time, zero cold-start (~1.1 GB)
docker build --build-arg GIGASTT_BAKE_MODEL=1 -t gigastt:baked .
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

Connect to `ws://127.0.0.1:9876/v1/ws`, send PCM16 audio frames, receive transcription in real time.

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
| `/ready` | GET | Readiness probe (200 when engine pool is ready) |
| `/v1/models` | GET | Model info (encoder type, pool size, capabilities) |
| `/v1/transcribe` | POST | File transcription, full JSON response |
| `/v1/transcribe/stream` | POST | File transcription with SSE streaming |
| `/v1/ws` | GET | WebSocket upgrade for real-time streaming |
| `/metrics` | GET | Prometheus metrics (enabled with `--metrics`) |

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

#### Error Responses

| HTTP | Code | When |
|---|---|---|
| 400 | `bad_request` | Invalid audio format or malformed request |
| 413 | `payload_too_large` | File exceeds `--body-limit-bytes` (default 50 MiB) |
| 429 | `rate_limit_exceeded` | Per-IP token bucket exhausted; `Retry-After` header included |
| 503 | `pool_saturated` | All inference sessions busy; `Retry-After: 30` |
| 503 | `pool_closed` | Server is shutting down, pool closed to new checkouts |

```json
// Example: pool saturation
HTTP/1.1 503 Service Unavailable
Retry-After: 30

{"code":"pool_saturated","message":"All inference sessions are busy"}
```

### Client Libraries

Ready-to-use WebSocket clients in [`examples/`](examples/):

#### Python
```sh
pip install websockets
python examples/python_client.py recording.wav
```

#### Bun (TypeScript)
```sh
bun examples/bun_client.ts recording.wav
```

#### Go
```sh
# go mod init gigastt-client && go get github.com/gorilla/websocket
go run examples/go_client.go recording.wav
```

#### Kotlin
```sh
# See header in KotlinClient.kt for Gradle/Maven deps
kotlinc examples/KotlinClient.kt -include-runtime -d client.jar
java -jar client.jar recording.wav
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

Quantized encoder: 4x smaller, ~43% faster, 0% WER degradation (verified on 993 Golos samples / 4991 words). Auto-detected at runtime.

Since v0.9.0 quantization is always compiled in and auto-invoked on first `download` or `serve` — no feature flag and no manual steps needed. The `quantize` Cargo feature is retained as a no-op for backward compat.

```sh
# Automatic (recommended)
cargo install gigastt
gigastt serve           # downloads model + auto-quantizes on first run

# Opt out of auto-quantization (FP32 only)
gigastt serve --skip-quantize
# or: GIGASTT_SKIP_QUANTIZE=1 gigastt serve

# Manual re-quantization
gigastt quantize                     # native Rust quantization
gigastt quantize --force             # re-quantize even if INT8 model exists
```

## Project Structure

gigastt is organized as a 3-crate Cargo workspace:

| Crate | Type | Purpose |
|---|---|---|
| [`gigastt-core`](crates/gigastt-core) | lib (rlib) | Inference engine, model download, quantization, protocol types |
| [`gigastt-ffi`](crates/gigastt-ffi) | lib (cdylib) | C-ABI FFI layer for Android / mobile embedding |
| [`gigastt`](crates/gigastt) | bin | Server binary (axum HTTP/WS) + CLI |

`gigastt-core` has no server dependencies — embed inference in any Rust project with `gigastt-core = "2.0"`.

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

## Android / FFI

gigastt can be embedded into Android applications via a C-ABI FFI layer (no HTTP server, no JNI boilerplate required).

```sh
# Build libgigastt_ffi.so for Android (arm64)
cargo ndk -t arm64-v8a -o ./jniLibs build --release -p gigastt-ffi
```

| Function | Purpose |
|---|---|
| `gigastt_engine_new(model_dir)` | Load engine (default pool_size = 4) |
| `gigastt_engine_new_with_pool_size(model_dir, pool_size)` | Load engine with custom RAM budget |
| `gigastt_transcribe_file(engine, wav_path)` | Synchronous file transcription |
| `gigastt_stream_new(engine)` | Start a real-time streaming session |
| `gigastt_stream_process_chunk(...)` | Feed PCM16 audio, get JSON segments |
| `gigastt_stream_flush(...)` | Finalize stream |

The `nnapi` feature on `gigastt-ffi` pulls in `ort/nnapi` for NPU/DSP acceleration on Android: `cargo ndk ... build -p gigastt-ffi --features nnapi`.
For pool sizing on mobile: use `pool_size = 1` to stay within ~350 MB RAM.

Full integration guide: [`ANDROID.md`](ANDROID.md)  
Kotlin bridge: [`ffi/android/GigasttBridge.kt`](ffi/android/GigasttBridge.kt)

## CLI Reference

Key flags for the most common commands. Every flag also has an environment variable — see the [full CLI reference](docs/cli.md).

```sh
# Start server
gigastt serve --port 9876 --bind-all --metrics

# Transcribe a file
gigastt transcribe recording.wav

# Re-quantize encoder (native Rust, ~2 min one-time)
gigastt quantize --force
```

| Flag | Default | Description |
|---|---|---|
| `--port` | 9876 | Listen port |
| `--host` | 127.0.0.1 | Bind address (loopback-only by default) |
| `--bind-all` | — | Allow non-loopback bind |
| `--pool-size` | 4 | Concurrent inference sessions |
| `--metrics` | — | Expose Prometheus at `/metrics` |
| `--idle-timeout-secs` | 300 | WebSocket idle timeout |
| `--max-session-secs` | 3600 | Wall-clock session cap |
| `--rate-limit-per-minute` | 0 | Per-IP rate limit (0 = off) |
| `--skip-quantize` | — | Skip INT8 quantization on first run |

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

- **Loopback-only bind.** The server refuses to listen on anything other than
  `127.0.0.1` / `::1` / `localhost` unless the operator explicitly passes
  `--bind-all` (or sets `GIGASTT_ALLOW_BIND_ANY=1`). Prevents accidental public
  exposure behind a reverse proxy or stray port forward.
- **Cross-origin requests denied by default.** A browser page at
  `https://evil.example.com` cannot drive-by connect to the local WebSocket /
  REST API. Loopback origins are always allowed; extra origins must be added
  via `--allow-origin https://app.example.com` (repeatable). Legacy
  `Access-Control-Allow-Origin: *` behaviour is opt-in via
  `--cors-allow-any`.
- **Retry-After on backpressure.** Pool saturation returns HTTP 503 with a
  `Retry-After: 30` header; WebSocket `error` payloads include
  `retry_after_ms: 30000` so clients can back off without guessing.
- **WebSocket frame limit:** 512 KB.
- **Session pool:** max 4 concurrent sessions (configurable via `--pool-size`).
- **Audio buffer cap:** 5 s (streaming) / 10 min (file upload).
- **Internal errors sanitized** — no path or model leakage to clients.
- **Idle connection timeout:** 300 s.
- **Per-IP rate limiting** (optional, off by default): `--rate-limit-per-minute N`
  enables a token-bucket limiter on all `/v1/*` endpoints; `/health` is exempt.
  Returns HTTP 429 when the bucket is exhausted. Privacy-first default: disabled.

Remote deployment (TLS + reverse proxy): see [`docs/deployment.md`](docs/deployment.md).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `protoc` not found during build | Missing Protocol Buffers compiler | `brew install protobuf` (macOS) or `apt install protobuf-compiler` (Debian/Ubuntu) |
| Model download hangs or fails | Network / HuggingFace availability | Retry `gigastt download`; check `~/.gigastt/models/` permissions |
| `Cannot quantize: FP32 encoder not found` | Partial download | Delete `~/.gigastt/models/` and re-run `gigastt download` |
| OOM on startup | Pool size too large for available RAM | Lower `--pool-size` (default 4); each session loads the full encoder |
| CoreML not used on macOS | Built without `--features coreml` | Re-build: `cargo build --release --features coreml` |
| CUDA not available on Linux | Built without `--features cuda` or missing CUDA 12+ | Re-build: `cargo build --release --features cuda`; verify `nvidia-smi` |
| WebSocket closes with 1008 | Session exceeded `--max-session-secs` | Increase `--max-session-secs` or send shorter streams |
| 429 Too Many Requests | Rate limiter enabled and bucket exhausted | Wait for `Retry-After` interval, or disable with `--rate-limit-per-minute 0` |
| Empty transcription for noisy audio | Input too quiet or wrong format | Ensure 16-bit PCM; normalize audio level; check supported formats |

## Testing

163 unit tests (including property-based via proptest) + 35 e2e/load/soak tests + WER benchmark:

```sh
cargo test --workspace               # 163 unit tests (no model needed)
cargo clippy --workspace             # Lint (zero warnings)

# E2E tests (require model, serial to avoid OOM)
cargo run -p gigastt -- download
cargo test -p gigastt --test e2e_rest --test e2e_ws --test e2e_errors --test e2e_shutdown --test e2e_rate_limit -- --ignored --test-threads=1

# Load & soak (local only)
cargo test -p gigastt --test load_test -- --ignored
cargo test -p gigastt --test soak_test -- --ignored
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) — development setup, PR guidelines, and release checklist.

## License

MIT — see [LICENSE](LICENSE)

## Acknowledgments

- [**GigaAM**](https://github.com/salute-developers/GigaAM) by [SberDevices](https://github.com/salute-developers) — the speech recognition model
- [**onnx-asr**](https://github.com/istupakov/onnx-asr) by [@istupakov](https://github.com/istupakov) — ONNX model export and reference
- [**ONNX Runtime**](https://github.com/microsoft/onnxruntime) — inference engine
- [**ort**](https://github.com/pykeio/ort) — Rust bindings for ONNX Runtime
