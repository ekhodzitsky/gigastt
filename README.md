# gigastt

Russian speech to text, locally.

[![crates.io](https://img.shields.io/crates/v/gigastt.svg)](https://crates.io/crates/gigastt)
[![docs.rs](https://docs.rs/gigastt-core/badge.svg)](https://docs.rs/gigastt-core)
[![CI](https://github.com/ekhodzitsky/gigastt/actions/workflows/ci.yml/badge.svg)](https://github.com/ekhodzitsky/gigastt/actions/workflows/ci.yml)
[![coverage](https://codecov.io/gh/ekhodzitsky/gigastt/branch/main/graph/badge.svg)](https://codecov.io/gh/ekhodzitsky/gigastt)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A speech recognizer in Rust, powered by [GigaAM v3](https://github.com/salute-developers/GigaAM)
and ONNX Runtime. Transcribe files, serve HTTP and WebSocket clients, or embed
the engine. CPU by default. The default INT8 model is about 225 MB; inference
runs on the device after the initial downloads. No cloud API or API key.

## Examples

Transcribe a recording or write subtitles:

```sh
gigastt transcribe recording.wav
gigastt transcribe recording.wav --format srt --output recording.srt
```

Process a folder, or watch for new recordings:

```sh
gigastt transcribe-batch recordings/ transcripts/
gigastt watch inbox/ transcripts/
```

Start the server:

```sh
gigastt serve
```

From another terminal, upload a file:

```sh
curl http://127.0.0.1:9876/v1/transcribe \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @recording.wav
```

The server binds to loopback by default. [Deployment](docs/deployment.md)
covers Docker, remote clients and reverse proxies.

## Interfaces

| Interface | Use |
|---|---|
| CLI | Files, batch folders, watched folders; TXT, JSON, SRT, VTT, Markdown |
| `/v1/transcribe` | File upload; word timings, segments and optional speaker labels |
| `/v1/transcribe/stream` | File upload with SSE results |
| `/v1/ws` | Live PCM16 audio with partial and final transcripts |
| `/v1/audio/transcriptions` | OpenAI-compatible multipart upload; JSON, text, subtitles, verbose JSON |
| `/v1/jobs` | Opt-in queue for file transcription; progress, polling and cancellation |
| In process | Rust, C ABI, Node and Python; Swift/Kotlin packaging guides |

[API reference](docs/api.md) · [Go client](sdks/go) ·
[TypeScript client](sdks/js) · [Embedding quickstarts](docs/quickstarts.md)

## Audio and models

Files: WAV (PCM, IEEE float, G.711, G.722, GSM, ADPCM, RF64), AAC/M4A, MP3,
FLAC, OGG/Vorbis, OGG/Opus and WebM/Opus. Raw telephony input is available
through the CLI and native REST API. WebSocket takes PCM16 at
8, 16, 24, 44.1 or 48 kHz.

The default `rnnt` head recognizes Russian. `e2e_rnnt` includes punctuation
and casing; `ml_ctc` / `ml_ctc_large` support Russian, English, Kazakh,
Kyrgyz and Uzbek. Optional punctuation, Russian text normalization,
hotwords and speaker diarization are described in the [API](docs/api.md).

CPU works out of the box. CUDA, CoreML and NNAPI builds, plus experimental
ANE and Candle backends: [architecture](docs/architecture.md).

## Performance

Reported file-transcription WER (%), lower is better. Apple M1 CPU,
Golos/OpenSTT slices, 1,000 samples per domain (992 clean references).

| Engine | Clean read | Far-field | Phone | YouTube |
|---|---:|---:|---:|---:|
| gigastt (`rnnt`, INT8) | 3.55 | 4.08 | 18.50 | 10.91 |
| Vosk 0.54 | 2.97 | 6.29 | 22.74 | 17.24 |
| faster-whisper (Large v3) | 15.53 | 17.34 | 24.93 | 15.45 |

Clean-read confidence intervals overlap. These datasets are close to
GigaAM's training distribution; results on held-out sets differ. Full
comparisons, confidence intervals and artifact provenance:
[benchmarks](docs/benchmarks.md).

File RTF is about 0.10 on M1 CPU. At the default two-session pool, measured
resident memory is about 66 MB on M1 Pro; RSS is about 510 MB because it
also counts the shared mapped model. [Measurement details](docs/benchmarks.md#footprint).

Live WebSocket recognition uses a buffered offline model with incremental
partials. Its accuracy and latency differ from file transcription;
see the [streaming measurements](docs/benchmarks.md#streaming-measurement-protocol).

## Install

Homebrew:

```sh
brew tap ekhodzitsky/gigastt https://github.com/ekhodzitsky/gigastt
brew install gigastt
```

Or build from crates.io:

```sh
cargo install gigastt
```

Building requires Rust 1.94+ and `protoc`; ONNX Runtime is downloaded at
build time by default. [Prebuilt releases](https://github.com/ekhodzitsky/gigastt/releases)
cover macOS Apple Silicon, Linux x86_64/aarch64 and Windows x86_64.
[Docker instructions](docs/deployment.md#docker) use the published GHCR images.

For Rust embedding: `gigastt-core = "2.21"`. Node: `npm install gigastt`.
Python: `pip install gigastt`. Model setup and platform packaging:
[quickstarts](docs/quickstarts.md).

[Who uses gigastt](docs/who-uses.md) · [Documentation](docs/README.md) ·
[CLI](docs/cli.md) · [API](docs/api.md) · [Benchmarks](docs/benchmarks.md) ·
[Changelog](CHANGELOG.md)

MIT. Default ASR weights are MIT; the optional WeSpeaker model is
CC BY 4.0. Benchmark datasets retain their own licenses. See
[LICENSE](LICENSE), [NOTICE](NOTICE) and [data licenses](benchmark/DATA_LICENSE).
