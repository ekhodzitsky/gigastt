# gigastt

Local speech recognition for files, live audio and embedded applications.

[![crates.io](https://img.shields.io/crates/v/gigastt.svg)](https://crates.io/crates/gigastt)
[![docs.rs](https://docs.rs/gigastt-core/badge.svg)](https://docs.rs/gigastt-core)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

gigastt is a Rust CLI and HTTP/WebSocket server powered by
[GigaAM](https://github.com/salute-developers/GigaAM) and ONNX Runtime.
Russian recognition is the default; multilingual models also support English,
Kazakh, Kyrgyz and Uzbek. Inference runs on your machine, with no cloud API,
account or API key. Models are downloaded once and can be provisioned offline.

This documentation follows **main**. For an installed release, use the
[documentation at its tag](https://github.com/ekhodzitsky/gigastt/tags).
Model bundles and language-binding releases have their own versions.

## Install

Choose one installation method:

- **Prebuilt binary:** download the archive for your platform from
  [Releases](https://github.com/ekhodzitsky/gigastt/releases), unpack it and put
  `gigastt` (`gigastt.exe` on Windows) on `PATH`. Builds cover macOS Apple Silicon,
  Linux x86_64/aarch64 and Windows x86_64. See [release verification](docs/verifying-releases.md).
- **Homebrew:**
  ```sh
  brew tap ekhodzitsky/gigastt https://github.com/ekhodzitsky/gigastt
  brew install gigastt
  ```
- **Build the published crate:**
  ```sh
  cargo install gigastt --locked
  ```
  Requires Rust 1.94+, a native toolchain and `protoc`; Linux also needs OpenSSL
  development files. [Build prerequisites](.github/CONTRIBUTING.md#development).

Docker and offline/systemd packages: [deployment](docs/deployment.md).

## Transcribe a file

```sh
gigastt transcribe recording.wav
gigastt transcribe recording.wav --format srt --output recording.srt
```

The first run downloads missing models. `gigastt download` preloads the ASR
bundle and, in default builds, the speaker model; punctuation is fetched when
enabled by `serve` or `transcribe`. See [model setup and sizes](docs/models.md).

Batch processing and directory watching:

```sh
gigastt transcribe-batch recordings/ transcripts/
gigastt watch inbox/ transcripts/
```

Files include WAV, AAC/M4A, MP3, FLAC, OGG/Vorbis, OGG/Opus and WebM/Opus.
WAV telephony codecs and raw G.711/G.722 are supported; see the
[format and codec reference](docs/api.md#audio-formats-and-telephony-codecs).

## Run the server

```sh
gigastt serve
```

From another terminal, check readiness and upload a file:

```sh
curl --fail http://127.0.0.1:9876/ready
curl --fail http://127.0.0.1:9876/v1/transcribe \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @recording.wav
```

Wait for `/ready` to return HTTP 200 before sending audio. `/health` also
answers during model loading. The default bind is loopback; remote access needs
explicit configuration and a proxy for TLS/authentication. See [deployment](docs/deployment.md).

| Interface | Purpose |
|---|---|
| CLI | Files, batch and watch; text, JSON, SRT, VTT and Markdown exports |
| `POST /v1/transcribe` | File upload with word timings and optional speaker labels |
| `POST /v1/transcribe/stream` | File upload with SSE results |
| `GET /v1/ws` | Live mono PCM16 at 8, 16, 24, 44.1 or 48 kHz; partial/final transcripts |
| `POST /v1/audio/transcriptions` | OpenAI-compatible multipart upload and output formats |
| `/v1/jobs` | Optional in-memory job queue, progress and cancellation (`--enable-jobs`) |

[API reference](docs/api.md) · [CLI reference](docs/cli.md) ·
[Go client](sdks/go/README.md) · [TypeScript client](sdks/js/README.md)

## Choose a model

| Model | Languages | Text output |
|---|---|---|
| `rnnt` (default) | Russian | Bare recognition; CLI/server can add punctuation and number normalization |
| `e2e_rnnt` | Russian | Built-in punctuation, casing and text normalization |
| `ml_ctc`, `ml_ctc_large` | Russian, English, Kazakh, Kyrgyz, Uzbek | Bare multilingual text |

Select with `--model-variant`. [Models](docs/models.md) explains downloads,
formatting policy and upgrades; [Languages](docs/languages.md) reports measured
multilingual quality. Choose using recognition quality, speed and memory on your
workload, as well as punctuation.

## Performance

The current main-branch RNN-T ASR bundle is about **310 MiB on disk**. It uses
INT8 matrix weights with floating-point convolutions; side models and optimized
caches add space. Disk size is not a RAM requirement. The older v2.22.0 release
pins a different bundle; see [the model transition](docs/rnnt-quantization.md).

[Benchmarks](docs/benchmarks.md) records corpus, model and hardware provenance.
Historical M1 figures do not describe the new bundle. Live WebSocket recognition
uses an offline model with buffered incremental decoding, so file WER and speed
must not be presented as live-stream measurements.

## Embed the engine

Use `gigastt-core = "2.22"` in Rust, `npm install gigastt` for the Node native
binding, or `pip install gigastt` for Python. These packages load side-provisioned
models. Swift, Kotlin and C ABI options are covered in
[in-process quickstarts](docs/quickstarts.md) and [packaging](docs/embedding-packaging.md).
Package availability and versions differ across ecosystems.

CPU is the default backend. CUDA, CoreML, NNAPI and experimental ANE/Candle
builds are described in [architecture](docs/architecture.md).

## Documentation and development

[Documentation index](docs/README.md) · [Troubleshooting](docs/troubleshooting.md) ·
[Contributing](.github/CONTRIBUTING.md) · [Changelog](CHANGELOG.md) ·
[Privacy](docs/privacy.md) · [Security](.github/SECURITY.md)

GitHub workflows run only on manual dispatch. Local validation and the manual
review/release process are documented in [Contributing](.github/CONTRIBUTING.md).

MIT. ASR weights are MIT; the optional WeSpeaker model is CC BY 4.0.
See [LICENSE](LICENSE), [NOTICE](NOTICE) and [benchmark data licenses](benchmark/DATA_LICENSE).
