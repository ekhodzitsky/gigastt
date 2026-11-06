# Architecture

```
        Audio (PCM16 multi-rate · file containers · raw telephony codecs)
                  |
          Mel Spectrogram            64 bins, FFT=320, hop=160 @ 16 kHz
                  |
        Conformer Encoder (ONNX)     model-dependent geometry · CPU / accelerated backends
                  |
     ┌────────────┴────────────┐
     │                         │
  RNN-T (rnnt / e2e_rnnt)   CTC (ml_ctc / ml_ctc_large)
  decoder + joiner loop     greedy decode, encoder-only
     │                         │
     └────────────┬────────────┘
                  |
        Tokenizer (char 34 / BPE 1025 / multilingual 71)
                  |
     optional text postprocessing: ITN → punctuation
                  |
             transcript (+ optional word times / speaker labels)
```

VAD selects speech regions before ASR; hotword bias affects decoding. Optional
request paths include:

- **Speaker diarization** — WeSpeaker ResNet34 fbank embeddings + polyvoice clustering; opt-in per REST request (`?diarization=true`) or WS `Configure`. Offline matches word midpoints to turns; streaming assigns the latest turn to the current word tail.
- **Stereo channel-speakers** — `channels=split` / `--stereo-speakers` runs ASR per channel and labels `speaker_0` / `speaker_1` (mutually exclusive with ML diarization).
- **Jobs queue** — opt-in `--enable-jobs` async store for long-file / batch REST work.
- **Hot reload** — loopback-only `POST /v1/admin/reload` rebuilds the engine from the boot recipe, warms it, then swaps atomically (keeps the old engine on failure). The mapped encoder is shared; extra peak is unmeasured after mmap — keep headroom or restart on edge.

## Crates

gigastt is a **6-crate** Cargo workspace:

| Crate | Type | Purpose |
|---|---|---|
| [`gigastt-core`](../crates/gigastt-core) | lib (rlib) | Inference engine, model download, protocol types — **no server deps** |
| [`gigastt`](../crates/gigastt) | bin + lib | Server (axum HTTP/WS/SSE/jobs) + CLI |
| [`gigastt-quantize`](../crates/gigastt-quantize) | lib | Native Rust INT8 quantizer (packaging; optional feature on core) |
| [`gigastt-ffi`](../crates/gigastt-ffi) | lib (cdylib) | C-ABI FFI for Android / mobile embedding |
| [`gigastt-uniffi`](../crates/gigastt-uniffi) | lib (cdylib) | UniFFI bindings (Python wheels / Swift / Kotlin path) |
| [`gigastt-node`](../crates/gigastt-node) | lib (cdylib) | napi-rs Node.js / Electron binding |

Embed inference in any Rust project with `gigastt-core = "2.22"`. For a lean embedded build, disable defaults (`default-features = false`) to drop `tokio` / `reqwest` / `symphonia` / `ryf` / polyvoice; opt capabilities back in via the `net`, `async-pool`, `file-decode`, `diarization`, and `quantize` features (`quantize` is packaging-only — the INT8 quantizer, not a runtime path).

The optional `extended-aac` feature enables syom and implies `file-decode`. It is
not enabled by default; see [the AAC audit](aac-audit.md) before choosing it.
Symphonia retains AAC and MP4 support for compatibility in both configurations.

Server surfaces (single process, one primary port unless metrics is enabled):

| Surface | Path | Notes |
|---|---|---|
| Liveness / readiness | `GET /health`, `GET /ready` | Bootstrap answers while the model loads |
| REST / SSE | `POST /v1/transcribe`, `/v1/transcribe/stream` | File upload |
| Models | `GET /v1/models` | Available model variants |
| OpenAI-compatible | `POST /v1/audio/transcriptions` | multipart `file` + `model` → `{"text":…}` |
| WebSocket | `GET /v1/ws` | Streaming partials/finals |
| Jobs | `/v1/jobs…` | Only when `--enable-jobs` |
| Admin | `POST /v1/admin/reload` | **Loopback peers only** |
| Metrics | `GET /metrics` | Separate `--metrics-listen` (default `127.0.0.1:9090`) |

## Models

The supported catalog, exact file sets, formatting policy and current bundle
sizes are documented in [Models](models.md). The standard ONNX path loads
prequantized encoders; RNN-T convolutions and decoder/joiner remain floating
point. Multilingual CTC is encoder-only and has different output geometry and
vocabulary. No per-request language field changes the loaded head.

Use [Languages](languages.md) and [Benchmarks](benchmarks.md) for measured
quality. Larger models and built-in formatting do not establish a universal
accuracy or speed advantage.

## Long recordings

Files over 30 seconds use independent overlapping windows: 24 seconds on CPU,
CoreML EP and CUDA, or 30 seconds on ANE, with a 2-second overlap. Each window
starts with a fresh decoder state. Overlap words are aligned by text, order and
time before merging; unmatched regions use the midpoint cut. VAD removes longer
pauses before windowing and maps word times back to the original recording.
Short files retain the single-pass path.

Changing window context can still change recognition. See
[long-form stitching](longform-stitching.md) for the policy, measurements and
reproduction commands.

## Hardware acceleration

| Platform | Feature flag | Execution Provider |
|---|---|---|
| macOS ARM64 (M1–M4) | `--features coreml` | CoreML + Neural Engine |
| Linux x86_64 + NVIDIA | `--features cuda` | CUDA 12+ |
| Android / ARM64 | `--features nnapi` | NNAPI (NPU/DSP) |
| macOS ARM64 | `--features ane` | Apple Neural Engine `.mlpackage` (bucketed file/live decoding; see [ane-backend.md](ane-backend.md)) |
| macOS ARM64 | `--features candle` | Candle/Metal, experimental parity (see [candle-backend.md](candle-backend.md)) |
| Any platform | _(default)_ | CPU |

`coreml` and `cuda` are mutually exclusive. `ane` and `candle` are each
mutually exclusive with the other and with `coreml`, `cuda`, and `nnapi`.
They select a separate backend rather than an ORT execution provider. `nnapi`
can be combined with the ORT `coreml` or `cuda` build features.

**CoreML path.** Dynamic-shape partitions can fail during prediction, so the
provider is restricted to static-shape subgraphs. Startup warmup tests actual
execution. With `--execution-provider auto`, provider failure can rebuild the
pool on CPU; an explicitly requested provider fails boot instead. Confirm the
provider returned by `/v1/models` before attributing a result to acceleration.
Historical speedups on older weights are not guarantees for current bundles.

## INT8 quantization

The optional `quantize` feature is a packaging tool, enabled by default. It
requires local FP32 ONNX sources and `protoc`; normal installation downloads
already prepared models. The accepted RNN-T recipe quantizes matrix weights
while retaining floating convolutions. The old all-ConvInteger recipe could
be both slower and less accurate. See the [quantization report](rnnt-quantization.md)
for the acceptance measurements and [quantizer crate](../crates/gigastt-quantize/README.md)
for the API. Quantization quality and resource costs must be measured per model.

## Air-gapped / offline builds

`ort`'s default `download-binaries` feature fetches a prebuilt onnxruntime over the
network at build time (verified by an embedded checksum) — outside `Cargo.lock`. The
"no cloud / full privacy" guarantee covers **runtime inference**, not the build. For
air-gapped builds, use the documented [`ort-load-dynamic` configuration](embedding-packaging.md)
and a compatible locally provisioned native library. Cargo dependencies and
other build inputs must also be available locally.
`protoc` must also be on `PATH` (the in-tree ONNX quantization pipeline regenerates
types via `prost-build`).
