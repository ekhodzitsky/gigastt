# Architecture

```
        Audio (PCM16, multi-rate)
                  |
          Mel Spectrogram            64 bins, FFT=320, hop=160
                  |
        Conformer Encoder (ONNX)     16 layers, 768-dim, 240M params · CoreML | CUDA | CPU
                  |
        RNN-T Decoder + Joiner       stateful h/c persisted across streaming chunks
                  |
        Tokenizer (char 34 / BPE 1025) -> optional punctuation / ITN
                  |
             Russian text
```

## Crates

gigastt is a 3-crate Cargo workspace:

| Crate | Type | Purpose |
|---|---|---|
| [`gigastt-core`](../crates/gigastt-core) | lib (rlib) | Inference engine, model download, quantization, protocol types — **no server deps** |
| [`gigastt-ffi`](../crates/gigastt-ffi) | lib (cdylib) | C-ABI FFI for Android / mobile embedding |
| [`gigastt`](../crates/gigastt) | bin | Server (axum HTTP/WS/SSE) + CLI |

Embed inference in any Rust project with `gigastt-core = "2.3"`. For a lean embedded build, disable defaults (`default-features = false`) to drop `tokio` / `reqwest` / `symphonia`; opt capabilities back in via the `net`, `async-pool`, and `file-decode` features.

## Model

[**GigaAM v3**](https://huggingface.co/istupakov/gigaam-v3-onnx) by
[SberDevices](https://github.com/salute-developers/GigaAM) — RNN-T (Conformer encoder +
LSTM decoder + joiner), 16-layer 768-dim encoder (240M params); the vocab depends on the
head (`rnnt` 34-token char — the v2.3 default — or `e2e_rnnt` 1025-token BPE), 16 kHz
mono input, MIT licensed. Download ~850 MB (encoder 844 MB, decoder 4.4 MB, joiner
2.6 MB); INT8 encoder ~215 MB. Trained on 700K+ hours of Russian speech.

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
