<p align="center">
  <h1 align="center">gigastt</h1>
  <p align="center"><strong>Embeddable on-device Russian speech-to-text — one Rust binary, no cloud, MIT-clean weights.</strong></p>
  <p align="center">
    <a href="https://github.com/ekhodzitsky/gigastt/actions"><img src="https://github.com/ekhodzitsky/gigastt/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="https://crates.io/crates/gigastt"><img src="https://img.shields.io/crates/v/gigastt.svg" alt="crates.io"></a>
    <a href="https://docs.rs/gigastt-core"><img src="https://docs.rs/gigastt-core/badge.svg" alt="docs.rs"></a>
    <a href="https://github.com/ekhodzitsky/gigastt/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT"></a>
  </p>
  <p align="center"><b>English</b> | <a href="README_RU.md">Русский</a></p>
</p>

---

gigastt turns any machine into a private Russian speech-recognition server — or embeds the same engine into a Rust app or an Android binary. It runs the open **GigaAM v3** model fully on-device via ONNX Runtime: no cloud, no API keys.

```sh
cargo install gigastt && gigastt serve
# WebSocket  ws://127.0.0.1:9876/v1/ws
# REST       http://127.0.0.1:9876/v1/transcribe
```

```sh
$ gigastt transcribe recording.wav
Привет, как дела?
```

## Highlights

- **Real-time streaming** — incremental partials over WebSocket; REST + SSE for files
- **Embeddable** — a single static binary, a C-ABI FFI `cdylib` for Android/mobile, or the `gigastt-core` crate
- **Accurate & small** — **2.6% WER** on Golos crowd (rnnt head), INT8 model ~225 MB, real-time on CPU (RTF ~0.11); CoreML / CUDA / NNAPI acceleration
- **Hardened server** — loopback-only by default, origin allowlist, per-IP rate limiting, graceful drain, Prometheus metrics
- **MIT-clean** — gigastt (MIT) on GigaAM v3 weights (MIT) — usable in commercial on-device products

## Where it fits

gigastt is **Russian-only** and built for **embedding**. Its `rnnt` head (the v2.3 default) reaches **2.6% WER on Golos crowd** — competitive with the strongest Russian engines on clean read — and that error is genuinely acoustic, not normalization-inflated. For multilingual use see whisper.cpp / sherpa-onnx / NVIDIA Parakeet. gigastt's niche is the **smallest Russian model with no language-model trade-off**, wrapped in an **embeddable single-binary / FFI / streaming** server with **MIT-clean weights**, and competitive on spontaneous and telephony speech. Full honest comparison vs Vosk 0.54, T-one and Whisper → **[Benchmarks](docs/benchmarks.md)**.

## Documentation

| Guide | Contents |
|---|---|
| **[API](docs/api.md)** | WebSocket protocol, REST + SSE, error codes, client examples (Python/Bun/Go/Kotlin) |
| **[Benchmarks](docs/benchmarks.md)** | WER / RTF / footprint vs 6 engines across 4 Russian domains, with caveats |
| **[Architecture](docs/architecture.md)** | Pipeline, model, hardware acceleration, INT8 quantization, project layout |
| **[Android / FFI](ANDROID.md)** | Embedding via the C-ABI on Android |
| **[CLI](docs/cli.md)** · **[Deployment](docs/deployment.md)** · **[Security](SECURITY.md)** · **[Troubleshooting](docs/troubleshooting.md)** | Reference & ops |

## Install

```sh
# Homebrew (macOS arm64 / Linux x86_64)
brew tap ekhodzitsky/gigastt https://github.com/ekhodzitsky/gigastt && brew install gigastt

# crates.io — needs protoc on PATH (brew install protobuf / apt install protobuf-compiler)
cargo install gigastt

# Docker (CUDA: Dockerfile.cuda; bake the model with --build-arg GIGASTT_BAKE_MODEL=1)
docker build -t gigastt . && docker run -p 9876:9876 gigastt
```

The GigaAM v3 model (~850 MB) auto-downloads on first run and is INT8-quantized to ~225 MB.

> Building also fetches a prebuilt onnxruntime over the network (ort's default `download-binaries`); the on-device / no-cloud guarantee covers **runtime inference**, not the build. See [Architecture](docs/architecture.md) for air-gapped builds.

## Requirements

Rust **1.88+**, `protoc` on `PATH`. macOS 14+ (Apple Silicon, CoreML) or Linux x86_64 (optional NVIDIA CUDA 12+). ~1.5 GB disk, ~790 MB RAM at the default `--pool-size 2` (~400 MB single-session). The `gigastt-core` crate has no server dependencies — embed it directly: `gigastt-core = "2.0"`.

## License

MIT — see [LICENSE](LICENSE).

> **Benchmark data** under `benchmark/` is **not** MIT: OpenSTT (`openstt_*`, CC BY-NC 4.0) and Golos (`golos_*`, Sber Public License) transcripts keep their non-commercial licenses. See [`NOTICE`](NOTICE) and [`benchmark/DATA_LICENSE`](benchmark/DATA_LICENSE).

## Acknowledgments

- [**GigaAM**](https://github.com/salute-developers/GigaAM) by [SberDevices](https://github.com/salute-developers) — the speech recognition model
- [**onnx-asr**](https://github.com/istupakov/onnx-asr) by [@istupakov](https://github.com/istupakov) — ONNX export & reference
- [**ONNX Runtime**](https://github.com/microsoft/onnxruntime) · [**ort**](https://github.com/pykeio/ort) — inference engine & Rust bindings
