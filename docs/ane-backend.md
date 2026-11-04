# ANE (CoreML) backend — user guide

> Optional `--features ane`: run the GigaAM v3 **encoder** on the Apple **Neural
> Engine** via native Core ML `.mlpackage`s. Additive and opt-in; the default `ort`
> path uses ONNX Runtime. This guide describes the implemented backend;
> conversion scripts and parity tests are the reproducible references.

gigastt's default inference backend is ONNX Runtime (`ort`). An **optional**
native Apple **Neural Engine** backend is available behind the `ane` Cargo
feature: it runs the GigaAM v3 **encoder** on the ANE via per-bucket fixed-shape
Core ML `.mlpackage`s, while the decoder/joiner (and any encoder window outside
the bucket fill floor) stay on the `ort` CPU path.

It is **additive and opt-in** — the default build is unchanged and still uses `ort`.

## Status

- **macOS ARM64 (Apple Silicon) only.** The backend links Apple's Core ML
  framework; on every other target the `ane` feature degrades to the `ort` path.
- Targets the default **`rnnt`** head (char vocab). `e2e_rnnt` and the
  multilingual `ml_ctc` / `ml_ctc_large` models transparently fall back to
  the `ort` encoder (the ANE backend is rnnt-only, mirroring `candle`).
- **Bucketed pad-up** backend: the encoder window is padded up to the nearest
  fixed bucket and run on the ANE — in file mode (fill floor 0.5) **and in
  streaming** (zero fill floor, so the ≤ 2.5 s streaming window pads into
  bucket 512 and also runs on the ANE). The CPU/`ort` fallback remains only
  for windows outside the bucket ladder (see [Behavior](#behavior)).
- `ane` is **mutually exclusive** with `coreml` (the ort CoreML EP), `cuda`,
  `nnapi`, and `candle` (a `compile_error!` fires if combined). Auxiliary models
  (VAD, punctuation) continue to run on the CPU `ort` path.

## 1. Build with the feature

```sh
# server binary (macOS ARM64 only)
cargo build --release --features ane

# or just the core library
cargo build -p gigastt-core --release --features ane
```

Do **not** combine with `--features coreml`, `--features cuda`,
`--features nnapi`, or `--features candle` (mutually exclusive). The default
`ort` build remains `cargo build --release` (unchanged).

## 2. Obtain the bucket packages

The ANE backend reads per-bucket `.mlpackage`s from
`~/.gigastt/models/ane/` (a sibling of the ONNX encoder), one fixed-shape
package per bucket in the ladder `[512, 768, 1536, 3000]` mel frames (≈5 s / 8 s /
15 s / 30 s windows):

```
~/.gigastt/models/ane/gigaam_v3_encoder_512.mlpackage
~/.gigastt/models/ane/gigaam_v3_encoder_768.mlpackage
~/.gigastt/models/ane/gigaam_v3_encoder_1536.mlpackage
~/.gigastt/models/ane/gigaam_v3_encoder_3000.mlpackage
```

**Pre-built packages (default):** the per-bucket packages are published on the
[`ane-v3-2026-06-24` GitHub release](https://github.com/ekhodzitsky/gigastt/releases/tag/ane-v3-2026-06-24);
`gigastt download --ane` fetches them (SHA-256-verified) into the same
directory:

```sh
gigastt download --ane
```

**Convert locally instead:** to rebuild the packages from the PyTorch model
(e.g. with a different bucket ladder), run on macOS ARM64:

```sh
uv run --python 3.12 \
    --with torch --with coremltools --with gigaam --with soundfile --with scikit-learn \
    --with "numpy<2" \
    python scripts/convert_gigaam_ane.py
```

This writes the per-bucket `.mlpackage`s into `~/.gigastt/models/ane/`.

The bucket-768 package alone is enough to engage the ANE path for short files;
the engine logs which buckets it found and compiles each present package once
(shared across the session pool).

## 3. Run

The server and CLI use the ANE encoder automatically when built with
`--features ane` and a `rnnt` model is loaded — `production_factory` routes the
encoder through the composite ANE factory. Usage is otherwise identical to the
default build:

```sh
gigastt serve                      # ANE encoder + ort decoder/joiner
gigastt transcribe audio.wav       # file-mode transcription on the ANE
```

If no bucket package is present, the encoder load fails with a clear message
pointing at the conversion / `gigastt download --ane` step.

## Behavior

- **Pad-up to fixed buckets.** Each file-mode encoder call pads its mel window up
  to the smallest bucket `N ≥ frames`, runs it on the ANE (Float16), then trims
  the output back to the real frame count. The frame count emitted matches the
  `ort` encoder exactly.
- **Fill floor (`FILL_FLOOR = 0.5`).** A window must fill **≥ 50%** of its bucket
  for the ANE path to be trusted. Below that, the mask-free zero-pad output
  diverges enough that a borderline token could flip, so the window falls back to
  the variable-length `ort` encoder. The smallest bucket (512) therefore covers
  real frame counts in `[256, 512]` — the typical 3–5 s clip range — at higher
  fill (less pad-up waste / lower divergence) than routing those clips up to 768;
  768 now covers `(512, 768]`. All buckets clear the ~288-mel ANE-residency floor,
  so each stays resident on the Neural Engine.
- **Streaming runs on the ANE too.** Since v2.14.2 the streaming encoder call
  uses a zero fill floor (`STREAMING_FILL_FLOOR = 0.0`, vs the file-mode 0.5),
  so the ≤ 2.5 s streaming window (≤ 250 mel frames) pads up to bucket 512 and
  executes on the ANE — trading pad-up waste for lower latency. The `ort`
  fallback remains only for windows outside the bucket ladder.
- **Over-max windows.** Files longer than the largest bucket use gigastt's
  windowed chunking — 30 s windows on the ANE backend (each full chunk nearly
  fills bucket 3000, recovering pad-up waste), 24 s on `ort`; any window
  outside the bucket range falls back to `ort` (no silent truncation).

## Performance & accuracy (honest numbers)

Measured on an Apple M1 Pro at the v2.5.0 ship gate (`rnnt` head, Golos clips).
The v2.5.0 CHANGELOG entry carries the rounded figures (≈ 10× warm e2e, encoder
~15×, WER ≈ 1.11%); the precise numbers quoted below (112 RTFx warm,
23.6 ms / 369 ms per window) come from local ship-gate measurement notes,
which are not published artifacts. An earlier revision of this section
quoted a pre-ship measurement round (≈ 3.7× e2e, ~230× encoder); it is
superseded by the shipped figures below.

- **Warm end-to-end ≈ 10× over the `ort` CPU build** (≈ 112 RTFx warm),
  **decode-bound**: the ANE cuts the encoder to ≈ 23.6 ms from ≈ 369 ms per
  window (≈ 15.6×, 99.8% ANE residency), but the CPU RNN-T greedy decode loop
  and feature extraction dominate the full pipeline once the encoder is
  offloaded, so the e2e win is far smaller than the raw encoder win.
- **WER vs `ort` ≈ 1.11%** on the 15-clip Golos set: transcripts are
  byte-identical except for a single borderline FP16-pad-up token flip on one
  clip. The ANE encoder is FP16 (not byte-exact), so parity is "near-lossless",
  not bit-exact (unlike the `candle` backend, which is byte-for-byte identical).
- **Cold-start:** a compiled-model disk cache cuts the ~20 s first load to
  ~86 ms on later starts.

## Confirming the ANE path is engaged

- **Startup log:** in a `--features ane` build, `gigastt serve` on a `rnnt`
  model logs
  `ANE encoder backend active (Core ML / Apple Neural Engine, macOS ARM64): …`.
  On an `e2e_rnnt` model it instead logs that the head is not `rnnt` and the ort
  encoder is used.
- **Per-window debug log:** at `--log-level debug` the encoder logs
  `ANE encoder path (bucketed pad-up)` (with the chosen bucket) for windows that
  pad into a bucket — file-mode and streaming alike — and
  `ANE encoder path (ort fallback: no bucket within fill-floor)` for windows
  outside the bucket ladder / below the file-mode fill floor.
- **RTFx:** a file transcription completing well above real time (with the
  decode loop, not the encoder, as the bottleneck) confirms the ANE path.
