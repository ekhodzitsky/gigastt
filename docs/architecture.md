# Architecture

```
        Audio (PCM16 multi-rate · file containers · raw telephony codecs)
                  |
          Mel Spectrogram            64 bins, FFT=320, hop=160 @ 16 kHz
                  |
        Conformer Encoder (ONNX)     16 layers, 768-dim · ort CPU | CoreML | CUDA | ANE | Candle
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
     optional: punctuation · ITN · hotword bias · VAD endpointing
                  |
             transcript (+ optional word times / speaker labels)
```

Optional parallel paths (not on the ASR critical path by default):

- **Speaker diarization** — WeSpeaker ResNet34 fbank embeddings + polyvoice clustering; opt-in per REST request (`?diarization=true`) or WS `Configure`. Offline matches word midpoints to turns; streaming assigns the latest turn to the current word tail.
- **Stereo channel-speakers** — `channels=split` / `--stereo-speakers` runs ASR per channel and labels `speaker_0` / `speaker_1` (mutually exclusive with ML diarization).
- **Jobs queue** — opt-in `--enable-jobs` async store for long-file / batch REST work.
- **Hot reload** — loopback-only `POST /v1/admin/reload` rebuilds the engine from the boot recipe, warms it, then swaps atomically (keeps the old engine on failure).

## Crates

gigastt is a **5-crate** Cargo workspace:

| Crate | Type | Purpose |
|---|---|---|
| [`gigastt-core`](../crates/gigastt-core) | lib (rlib) | Inference engine, model download, quantization, protocol types — **no server deps** |
| [`gigastt`](../crates/gigastt) | bin + lib | Server (axum HTTP/WS/SSE/jobs) + CLI |
| [`gigastt-ffi`](../crates/gigastt-ffi) | lib (cdylib) | C-ABI FFI for Android / mobile embedding |
| [`gigastt-uniffi`](../crates/gigastt-uniffi) | lib (cdylib) | UniFFI bindings (Python wheels / Swift / Kotlin path) |
| [`gigastt-node`](../crates/gigastt-node) | lib (cdylib) | napi-rs Node.js / Electron binding |

Embed inference in any Rust project with `gigastt-core = "2.14"`. For a lean embedded build, disable defaults (`default-features = false`) to drop `tokio` / `reqwest` / `symphonia` / polyvoice; opt capabilities back in via the `net`, `async-pool`, `file-decode`, and `diarization` features.

Server surfaces (single process, one primary port unless metrics is enabled):

| Surface | Path | Notes |
|---|---|---|
| Liveness / readiness | `GET /health`, `GET /ready` | Bootstrap answers while the model loads |
| REST / SSE | `POST /v1/transcribe`, `/v1/transcribe/stream` | File upload |
| WebSocket | `GET /v1/ws` | Streaming partials/finals |
| Jobs | `/v1/jobs…` | Only when `--enable-jobs` |
| Admin | `POST /v1/admin/reload` | **Loopback peers only** |
| Metrics | `GET /metrics` | Separate `--metrics-listen` (default `127.0.0.1:9090`) |

## Model

[**GigaAM v3**](https://huggingface.co/istupakov/gigaam-v3-onnx) by
[SberDevices](https://github.com/salute-developers/GigaAM) — RNN-T (Conformer encoder +
LSTM decoder + joiner), 16-layer 768-dim encoder (240M params); the vocab depends on the
head (`rnnt` 34-token char — the v2.3 default — or `e2e_rnnt` 1025-token BPE), 16 kHz
mono input, MIT licensed. Download ~850 MB (encoder 844 MB, decoder 4.4 MB, joiner
2.6 MB); INT8 encoder ~215 MB. Trained on 700K+ hours of Russian speech.

Two opt-in heads (`--model-variant ml_ctc` / `ml_ctc_large`) use
[**GigaAM Multilingual**](https://huggingface.co/istupakov/gigaam-multilingual-ctc-onnx)
instead: an encoder-only charwise-CTC model (no LSTM decoder / joiner, greedy CTC
decoding), MIT licensed, sharing the 64-mel / FFT 320 / hop 160 / 16 kHz frontend above.
`ml_ctc` has a 220M-param encoder, `ml_ctc_large`
([`istupakov/gigaam-multilingual-large-ctc-onnx`](https://huggingface.co/istupakov/gigaam-multilingual-large-ctc-onnx))
600M; both use a shared 71-class multilingual character vocabulary (blank id 70) and
emit bare lowercase text. Trained across 70+ languages, best-in-class on Russian,
Kazakh, Kyrgyz, and Uzbek. Both download istupakov's pre-quantized INT8 encoder
directly — no FP32 download, no on-device quantization (`ml_ctc` ~225 MB, `ml_ctc_large`
~592 MB).

## Hardware acceleration

| Platform | Feature flag | Execution Provider |
|---|---|---|
| macOS ARM64 (M1–M4) | `--features coreml` | CoreML + Neural Engine |
| Linux x86_64 + NVIDIA | `--features cuda` | CUDA 12+ |
| Android / ARM64 | `--features nnapi` | NNAPI (NPU/DSP) |
| Any platform | _(default)_ | CPU |

`coreml` and `cuda` are mutually exclusive; `nnapi` can be combined with either.

**CoreML path.** The Conformer encoder has a dynamic time axis, and CoreML cannot
reliably execute dynamic-shape partitions (they fail at prediction time, issue #42).
gigastt compiles the model as `MLProgram` and restricts CoreML to statically-shaped
subgraphs — heavy conv/matmul on the Neural Engine, dynamic-shape ops on CPU. On an
M1 Pro (INT8, release, median of 5): **~3× faster encoder** on a 4 s clip (~210 ms vs
~690 ms) and **~5.6×** on a 2-minute file vs the pure-CPU build. On startup a ~1 s
silent warmup probe verifies CoreML; on failure the engine logs `falling back to CPU
execution provider` and transparently rebuilds sessions on CPU — it degrades, never
crashes.

## INT8 quantization

Native-Rust quantization (always compiled). The encoder shrinks ~3.9× and runs as true
INT8 integer compute (`DynamicQuantizeLinear` + `MatMulInteger`/`ConvInteger`), so the CPU
EP executes fast integer kernels instead of dequantizing the weights back to float — RTF
well below 1.0 on CPU — with negligible WER change. Auto-detected and auto-invoked on
first `download` / `serve`; opt out with `--skip-quantize` (or `GIGASTT_SKIP_QUANTIZE=1`).
Re-quantize manually with `gigastt quantize [--force]`.

## Air-gapped / offline builds

`ort`'s default `download-binaries` feature fetches a prebuilt onnxruntime over the
network at build time (verified by an embedded checksum) — outside `Cargo.lock`. The
"no cloud / full privacy" guarantee covers **runtime inference**, not the build. For
air-gapped builds, use `ort` with `default-features = false` + `load-dynamic` (or a
vendored onnxruntime) and pin the native library via `ORT_*` env vars / `.cargo/config.toml`.
`protoc` must also be on `PATH` (the in-tree ONNX quantization pipeline regenerates
types via `prost-build`).
