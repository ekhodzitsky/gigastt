<p align="center">
  <h1 align="center">gigastt</h1>
  <p align="center"><strong>Embeddable on-device Russian speech-to-text — one Rust binary, no cloud, MIT-clean weights.</strong></p>
  <p align="center">
    <a href="https://github.com/ekhodzitsky/gigastt/actions"><img src="https://github.com/ekhodzitsky/gigastt/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="https://codecov.io/gh/ekhodzitsky/gigastt"><img src="https://codecov.io/gh/ekhodzitsky/gigastt/branch/main/graph/badge.svg" alt="codecov"></a>
    <a href="https://crates.io/crates/gigastt"><img src="https://img.shields.io/crates/v/gigastt.svg" alt="crates.io"></a>
    <a href="https://docs.rs/gigastt-core"><img src="https://docs.rs/gigastt-core/badge.svg" alt="docs.rs"></a>
    <a href="https://github.com/ekhodzitsky/gigastt/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT"></a>
  </p>
  <p align="center"><b>English</b> | <a href="README_RU.md">Русский</a></p>
</p>

---

gigastt turns any machine into a private Russian speech-recognition server — or embeds the same engine into a Rust app or an Android binary. It runs the open **GigaAM v3** model fully on-device via ONNX Runtime: no cloud, no API keys.

## At a glance

| Private, on-device | Embeddable + streaming | Accurate Russian | Tiny &amp; real-time |
|---|---|---|---|
| No cloud, no keys — runtime is 100% local. MIT engine on MIT weights, commercial-ready. | One static binary, a C-ABI FFI for mobile, or the `gigastt-core` crate — with incremental WebSocket partials, no Python. | Most accurate on 3 of 4 Russian domains: far-field 4.08%, phone 18.50%, YouTube 10.91%; statistical tie on clean read. | ~225 MB INT8 model, RTF ~0.10 (~10× real-time on CPU), 0.94 s cold-start. |

**WER** clean 3.55% / far-field 4.08% / phone 18.50% / YouTube 10.91%  ·  **RTF** ~0.10  ·  **Model** ~225 MB INT8  ·  **Cold-start** 0.94 s  ·  **RAM** ~400 MB single / 790 MB pool-2  ·  **Streaming** first partial ~0.78 s

> GigaAM v3 `rnnt` head, INT8, Apple M1 CPU, 1000 samples/domain, failures = 100% WER, 95% bootstrap CIs. Every competitor is measured like-for-like through the [same harness](docs/benchmarks.md), manifests, and normalization.

## How it compares

WER (%) on four Russian domains, lower is better — plus every axis that decides a deployment. gigastt is the `rnnt` head, INT8.

| Engine | Clean | Far-field | Phone | YouTube | RTF | Disk | Peak RAM | Cold-start | Streaming | Punct. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|---|
| **gigastt** (GigaAM v3 `rnnt`) | 3.55 | **4.08** | **18.50** | **10.91** | 0.10 | ~225 MB | 790 / ~400 MB | **0.94 s** | **Yes** — incremental WS | **Yes** |
| Vosk 0.54 (Zipformer2) | **2.97** | 6.29 | 22.74 | 17.24 | ~0.03 | 966 MB | 560 MB | 1.16 s | Yes (server) | Add-on |
| T-one (beam + LM) | 6.61 | 14.62 | 21.73 | 23.23 | 0.065 | 138 MB + 5.5 GB LM | — | — | Yes (300 ms) | No |
| T-one (greedy, no LM) | 7.85 | 17.22 | 22.37 | 26.54 | 0.065 | 138 MB | 672 MB | 1.87 s | Yes (300 ms) | No |
| whisper.cpp (Large v3) | 15.26 | 17.91 | 32.73 | 22.61 | 0.36–0.77 | 2.9 GB | — | — | No | Yes |
| faster-whisper (Large v3) | 15.53 | 17.34 | 24.93 | 15.45 | &gt;1.0 | 2.9 GB | 2619 MB | 8.2 s | No | Yes |
| faster-whisper-turbo | 14.45 | 18.30 | 26.58 | 15.45 | &gt;1.0 | 1.6 GB | 2154 MB | 6.8 s | No | Yes |

Conditions: Apple M1, CPU EP, INT8/greedy, 1000 samples/domain (clean read 992; turbo = 300-sample slice), 95% bootstrap CIs. Clean read 3.55 (2.9–4.2) overlaps Vosk 0.54 2.97 (2.4–3.6) — a statistical tie; far-field / phone / YouTube wins are CI-separated. RTF &gt; 1.0 = slower than real-time on CPU. gigastt RAM is at the default `--pool-size 2` (single-session ~400 MB). "—" = not measured. Full methodology and caveats: [Benchmarks](docs/benchmarks.md).

**Streaming:** the Whisper engines are offline-only — no partials while you speak. gigastt streams genuine incremental WebSocket partials (~0.78 s to first partial on CPU) from one self-contained binary with no Python; Vosk-server and T-one (300 ms chunks) also stream. So streaming is gigastt's clear win over the Whisper family; over Vosk / T-one the edge is packaging — incremental partials plus a C-ABI FFI in a single binary — not lower latency.

**Punctuation &amp; casing:** gigastt outputs readable Russian out of the box — native on the `e2e_rnnt` head, or via a small bundled RuPunct + ITN pass on the default `rnnt` head (`--punctuation` / `--itn`, auto-downloaded). That matches the Whisper engines (punctuated natively) and beats the Russian specialists — Vosk needs a separate 1.6 GB `recasepunc` add-on and T-one emits none.

## Scope &amp; honest caveats

Where rivals win, and when not to reach for gigastt:

- **Clean read is a tie, not a win** — gigastt 3.55% (2.9–4.2) vs Vosk 0.54 2.97% (2.4–3.6); the CIs overlap and Vosk's point estimate is slightly ahead.
- **Russian-first, narrowly multilingual** — the default `rnnt` / `e2e_rnnt` heads are Russian-only; the opt-in `ml_ctc` / `ml_ctc_large` heads add just ru/en/kk/ky/uz. For real breadth use Vosk (20+ languages) or whisper.cpp / faster-whisper / sherpa-onnx (~99). gigastt is a specialist.
- **Not the speed leader** — Vosk (RTF ~0.03) and T-one (~0.06) are faster; gigastt (~0.10) is comfortably real-time, not the fastest.
- **Peak RAM at the default `--pool-size 2` (790 MB) loses** to Vosk 0.54 (560 MB) and T-one greedy (672 MB); single-session (~400 MB) is competitive — drop to `--pool-size 1` for the lean profile.
- **Streaming is buffered/chunked** over an offline RNN-T, not a natively streaming acoustic model; ~0.78 s TTFP is not a lowest-latency claim.
- **Training-data overlap** — GigaAM v3 is trained heavily on Golos; the Golos / OpenSTT benchmark slices likely overlap its training distribution, so these are best-case in-distribution upper bounds, not WER on unseen data.

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

## Quickstart

```sh
$ gigastt transcribe recording.wav
Привет, как дела?

# Or run the server — WebSocket + REST + SSE on one port (loopback only):
$ gigastt serve
# WebSocket  ws://127.0.0.1:9876/v1/ws
# REST       http://127.0.0.1:9876/v1/transcribe
```

## Capabilities

| Capability | Support |
|---|---|
| Heads | `rnnt` (34-token char, default — lowest WER) · `e2e_rnnt` (1025-token BPE, punctuation / casing / ITN baked in) · `ml_ctc` / `ml_ctc_large` (GigaAM Multilingual charwise CTC, 220M / 600M, 71-token multilingual char — ru/en/kk/ky/uz) |
| Post-processing | optional punctuation, casing &amp; Russian ITN — native on `e2e_rnnt`, or a bundled RuPunct + ITN pass on `rnnt` (auto-downloaded; `--punctuation` / `--itn`), overridable per request (`?punctuation=` / `?itn=` / `?vad=`) |
| Delivery | static binary · C-ABI FFI `cdylib` (Android / mobile) · `gigastt-core` crate (no server deps) |
| Execution providers | CPU (any platform) · CoreML EP (macOS ARM64) · CUDA 12+ (Linux x86_64) · NNAPI (Android) |
| Streaming | incremental WebSocket partials · REST + SSE for files · single port 9876 |
| Audio in | WAV · M4A/AAC · MP3 · OGG/Vorbis · FLAC (auto mono mix for multi-channel) |
| Stereo telephony recordings | Optional channel-speaker mode (`--stereo-speakers` CLI / `channels=split` REST) labels the left/right channels as `speaker_0` and `speaker_1` |
| Async jobs | Long-file / batch transcription queue via `/v1/jobs` (opt-in with `--enable-jobs`): submit, poll, cancel, SSE progress, retry, and TTL eviction |
| Export | JSON · TXT · SRT · VTT · Markdown — per-word timings + confidence, or segment-level (`?segments=true` JSON, `### [mm:ss]` Markdown) |
| Server hardening | loopback-only by default · origin allowlist · per-IP rate limiting · graceful drain · Prometheus `/metrics` on a separate port · loopback-only model hot-reload (`POST /v1/admin/reload`) |

## Documentation

| Guide | Contents |
|---|---|
| **[API](docs/api.md)** | WebSocket protocol, REST + SSE, error codes, client examples (Python/Bun/Go/Kotlin) |
| **[Benchmarks](docs/benchmarks.md)** | WER / RTF / footprint vs 6 engines across 4 Russian domains, with caveats |
| **[Architecture](docs/architecture.md)** | Pipeline, model, hardware acceleration, INT8 quantization, project layout |
| **[Android / FFI](ANDROID.md)** | Embedding via the C-ABI on Android |
| **[CLI](docs/cli.md)** · **[Deployment](docs/deployment.md)** · **[Security](SECURITY.md)** · **[Troubleshooting](docs/troubleshooting.md)** | Reference & ops |

## Requirements

Rust **1.88+**, `protoc` on `PATH`. macOS 14+ (Apple Silicon, CoreML) or Linux x86_64 (optional NVIDIA CUDA 12+). ~1.5 GB disk, ~790 MB RAM at the default `--pool-size 2` (~400 MB single-session). The `gigastt-core` crate has no server dependencies — embed it directly: `gigastt-core = "2.11"`.

## License

MIT — see [LICENSE](LICENSE).

> **Benchmark data** under `benchmark/` is **not** MIT: OpenSTT (`openstt_*`, CC BY-NC 4.0) and Golos (`golos_*`, Sber Public License) transcripts keep their non-commercial licenses. See [`NOTICE`](NOTICE) and [`benchmark/DATA_LICENSE`](benchmark/DATA_LICENSE).

## Acknowledgments

- [**GigaAM**](https://github.com/salute-developers/GigaAM) by [SberDevices](https://github.com/salute-developers) — the speech recognition model
- [**onnx-asr**](https://github.com/istupakov/onnx-asr) by [@istupakov](https://github.com/istupakov) — ONNX export & reference
- [**ONNX Runtime**](https://github.com/microsoft/onnxruntime) · [**ort**](https://github.com/pykeio/ort) — inference engine & Rust bindings
