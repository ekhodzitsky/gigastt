# Native ANE (CoreML) backend for GigaAM v3 — design + implementation plan

> Optional `--features ane`: run the GigaAM v3 **encoder** on the Apple **Neural Engine** via a native Core ML `.mlpackage`, behind the PR #115 runtime seam. Additive/opt-in; default `ort` path unchanged.

> **Note — this first half is the ORIGINAL pre-implementation plan, superseded by the shipped design in the [Quickstart](#quickstart-user-guide) below.** Where the two disagree, the Quickstart is authoritative. Known divergences: (1) the shipped bucket ladder is `[512, 768, 1536, 3000]`, not the plan's `[384, 768, 1536, 3000]` — 384 was a **fill floor** (50% of the 768 bucket), not a real bucket; 512 is the real smallest bucket, covering 3–5 s clips at higher fill; (2) `EnumeratedShapes` was **rejected** in favor of per-bucket FIXED-shape `.mlpackage`s. The plan is kept for historical context only.

## Why (validated by spike, 2026-06-23, Apple M1 Pro)
A native coremltools conversion of the GigaAM v3 conformer encoder (NOT the ORT CoreML EP) lands **99.9% of compute on the ANE** from a straight torch.jit.trace (no attention rewrite needed), at **339× warm RTFx** for the encoder on a 15 s window (vs 126× CPU), with **near-lossless accuracy**: on 15 Golos clips, FP16-ANE vs fp32 baseline = 14/15 byte-identical, WER delta **+1.33%** (one near-homophone slip). This refutes the prior "ANE not worth it / ORT CoreML EP can't run the conformer" conclusion (issue #42 was an ORT-EP limitation, not a CoreML one). Win over the existing Candle/Metal path is power/thermals + raw encoder throughput; tradeoff is FP16 (not byte-exact) + a CoreML build/distribution step + an objc2 bridge.

## Decisions (maintainer)
- **New `ane` Cargo feature** (distinct from `coreml` = ort CoreML EP, which stays). Mutually exclusive with `coreml`/`cuda`/`nnapi`/`candle`.
- **Streaming from the start** — cover gigastt's full encoder call range via bucketed fixed shapes.
- **`.mlpackage` distributed from a HuggingFace/GitHub release** (+ SHA256 verify), mirroring the pre-quantized model pipeline.
- **rnnt head only** (like candle); e2e_rnnt → fall back to ort (variant-gate, per the candle audit fix).
- **Encoder only on ANE**; decoder/joiner stay on ort/CPU (tiny; state is external).
- **FP16 first**; Argmax-style palettization (~215 MB) is a later option.

## The bucketing problem (core design point)
The encoder is called with a CONTINUOUS range of `num_frames` (constants in `inference/mod.rs`):
- streaming: growing windows up to `STREAM_MAX_WINDOW_SAMPLES`=2.5 s → ≤ 250 mel frames;
- file mode ≤30 s: single pass, 1..3000 mel frames; >30 s: 24 s windows = 2400 mel frames.

ANE requires fixed shapes. **Empirical findings (Phase 1a + threshold spike, M1 Pro — these REVISED the original plan):**
- **`EnumeratedShapes` is OUT.** Measured: an enumerated single-`.mlpackage` lands **0% ANE / 100% CPU** (flexible shapes evict), RTFx @1024 = 33× vs fixed 394× (12× slower). So: **per-bucket FIXED-shape `.mlpackage`s**, one per bucket.
- **Sharp size threshold:** encoder output `T' ≤ 64` (≤256 mel ≈2.5 s) → **0% ANE** (and CPU_AND_NE is *slower* than CPU_ONLY there — ANE-dispatch overhead); `T' ≥ 72` (≥288 mel ≈2.9 s) → **99.8% ANE**, 9–18 ms, 3.1–3.8× over CPU. CoreML keeps small workloads on CPU.
- **Streaming IS rescued by pad-up.** A 2.5 s streaming window (250 mel) padded up to the **384-mel floor** runs **99.8% ANE @ 9.7 ms** (vs the real small window on CPU ~27 ms). (Batching 4 small windows also flips to ANE @ 5.5 ms/window — bonus for concurrent streams.) So streaming benefits from ANE after all.
- **Bucket ladder (ANE-resident sizes only):** `[384, 768, 1536, 3000]` mel (≈4/8/15/30 s; all 99.8% ANE). Pad any input up to the nearest bucket ≥ real frames; windows < 384 use the **384 floor**; > 3000 → gigastt's existing 24 s chunking. Single mel input (drop `length`; pad → run → trim output to `ceil(real_frames/4)`). Coverage logged; over-max windows fall back to the ort encoder (no silent truncation).
- **Distribution:** N× FIXED `.mlpackage` ≈ 442 MB each (FP16) → palettization is REQUIRED to keep the download sane (see Distribution; pending the palettization spike — target ≤~215 MB/bucket while holding ANE residency + WER).
- **Padding correctness:** zero-padded mel frames are attended to (att_mask=None at batch=1). The spike showed this is near-lossless in practice (14/15 identical), but Phase 3 validates WER per bucket; if padding shifts outputs, add a key-padding mask (the encoder supports att_mask) for padded buckets.

## Architecture (mirrors `runtime/candle/`)
New module `crates/gigastt-core/src/runtime/coreml/` (feature `ane`), implementing the #115 traits:
- `factory.rs` — `AneFactory: RuntimeFactory` (creates the runtime; `cpu_fallback` → ort cpu).
- `runtime.rs` — `AneRuntime: Runtime`; `load_session(path, is_encoder)`: encoder → load the `.mlpackage` via objc2-core-ml, build `EncoderSession`; decoder/joiner → return an error (handled by the ort fallback, see routing).
- `session.rs` — `EncoderSession: RuntimeSession`: `run([mel,len]) -> [encoded]`: pick bucket ≥ frames, pad mel to bucket, marshal to `MLMultiArray` Float16, `predictionFromFeatures`, read Float16 output, trim to `ceil(frames/4)`, return `[1,768,T']`.
- `bridge.rs` — objc2-core-ml glue: load/compile `.mlpackage`→`.mlmodelc`, `MLDictionaryFeatureProvider`, Float16 MLMultiArray pack/unpack, `MLModelConfiguration{computeUnits=.cpuAndNeuralEngine}`. macOS-only (`#[cfg(target_os="macos")]`); `// SAFETY:` on every unsafe.
- Routing: encoder → `AneFactory` (ANE); decoder/joiner + aux (VAD/punct) → ort cpu. So `production_factory` under `ane` returns a composite: ANE encoder + ort decoder/joiner. (Simplest: `AneFactory::create` returns an `AneRuntime` whose `load_session(is_encoder=false)` delegates to an inner ort runtime.)
- Variant-gate: only `ModelVariant::Rnnt` → ANE; else ort (mirror the candle audit fix).
- Dep: `objc2-core-ml` optional, macOS-target-gated. Isolation guard extended (objc2_core_ml only under runtime/coreml/).

## Distribution
- `scripts/convert_gigaam_ane.py` — the spike's proven `convert.py`, producing one **FIXED-shape** `.mlpackage` per bucket (single mel input, mlprogram, `CPU_AND_NE`, `macOS15` target), then **palettizing** each with `ct.optimize.coreml` (kmeans, **6-bit, per_grouped_channel group_size 32**). Palettization spike result (M1 Pro): 6-bit = **167 MB** (vs 442 FP16), cos **0.997**, RTFx **374×**, **WER 1.33% = FP16** (Golos 15-clip gate). ANE residency is PRESERVED — palettization adds `constexpr_lut_to_dense` (load-time LUT expansion, not runtime CPU); RTFx holding ~340–377× confirms the compute stays on ANE (unlike ort's INT8 DequantizeLinear which evicted). 4-bit (112 MB) holds WER on the gate but cos drops to 0.958 → too risky for ship without a full-Golos check; 8-bit (225 MB) is the conservative fallback. Run on macOS: `uv --python 3.13 --with torch --with coremltools --with gigaam --with soundfile --with scikit-learn`.
- **Bucket-count is the size lever:** each FIXED bucket `.mlpackage` carries a FULL copy of the (palettized) weights (~167 MB), so N buckets ≈ N×167 MB. Ship the MINIMUM ladder: e.g. `{768, 3000}` (≈8 s + 30 s) = ~334 MB total — 768 covers streaming-padded + short files, 3000 covers long; (optionally add 1536). EnumeratedShapes can't share weights (it evicts to CPU), so few-fixed-buckets is the tradeoff.
- A `release-ane.yml` workflow (workflow_dispatch, macos-14) converts + palettizes + zips + publishes the per-bucket `.mlpackage`s to a GitHub release `ane-v3-<date>`, prints per-file SHA256.
- `model/mod.rs`: add `ANE_RELEASE_BASE` + per-bucket checksums + an `ensure_ane_packages(dir)` download path (mirror `ensure_prequantized_model_variant`). `gigastt download --ane` fetches them. `AneRuntime::load_session` reads `<model_dir>/ane/gigaam_v3_encoder_<bucket>.mlpackage`, errors clearly (with the download hint) if absent.

## Gates
- **Per-bucket numeric parity** vs ort encoder (FP16 tolerance, cos ≥ 0.99).
- **WER parity** on the FULL Golos benchmark: ANE-encoder + ort decode vs the ort baseline; target WER delta ≤ ~1-2% (spike: +1.33% on 15 clips). End-to-end transcript diff reported.
- **e2e RTFx** measured (note: gated by the RNN-T greedy loop on CPU; encoder is ~free on ANE).
- Default `ort` build + suite unchanged; `compile_error!` on `ane`+{coreml,cuda,nnapi,candle}.

---

## Implementation plan (phased, subagent-driven; TDD where testable)

### Phase 0 — scaffold (default build unchanged)
- Add optional `objc2-core-ml` dep, macOS-target-gated; `ane` Cargo feature on gigastt-core + gigastt passthrough.
- `compile_error!` guard: `ane` ⊕ `coreml`/`cuda`/`nnapi`/`candle`.
- `runtime/coreml/{mod,factory,runtime}.rs` scaffold behind `feature="ane"`, stub `load_session` (clear "not implemented" error); `default_factory`/`production_factory` route to `AneFactory` only for `ModelVariant::Rnnt` under the feature, else ort.
- Extend the Runtime Isolation guard (objc2_core_ml only under runtime/coreml/); add a macos-14 `Build (ANE)` CI lane (`cargo build/clippy -p gigastt-core --features ane`).
- Verify: default `cargo build`/clippy unchanged; `--features ane` compiles; isolation grep clean.

### Phase 1 — conversion pipeline (the .mlpackage)
- Generalize the spike `convert.py` → `scripts/convert_gigaam_ane.py`: EnumeratedShapes over the bucket ladder, single mel input, FP16 mlprogram, `compute_units=CPU_AND_NE`. Produce `gigaam_v3_encoder_ane.mlpackage`.
- Verify per-bucket numeric parity vs the PyTorch/ONNX encoder (cos ≥ 0.99 each bucket; reuse the spike's parity harness).
- `release-ane.yml` (workflow_dispatch, macos-14): convert + publish + SHA256.
- `model/mod.rs`: `ANE_RELEASE_BASE` + checksum + `ensure_ane_package` download path; `gigastt download --ane`.

### Phase 2 — objc2-core-ml adapter + EncoderSession
- `bridge.rs`: load/compile `.mlpackage`, Float16 MLMultiArray pack/unpack, predict, computeUnits=cpuAndNeuralEngine. `// SAFETY:` on each unsafe; macOS-gated.
- `EncoderSession::run`: bucket select + pad + predict + trim + `[1,768,T']`. Unit-test bucket-selection + pad/trim logic with a stub (no model).
- `runtime.rs`: encoder → EncoderSession from `<dir>/ane/...mlpackage`; decoder/joiner → inner ort runtime (composite). Error + download hint if `.mlpackage` missing.

### Phase 3 — end-to-end parity + WER gate
- Model-gated `#[ignore]` tests: ANE-encoder transcription vs ort baseline on Golos fixtures → assert WER delta within tolerance; print per-clip diff.
- Full-Golos WER run (benchmark harness with the ANE encoder) → record the real delta.
- Measure warm e2e RTFx (file + streaming) vs ort/candle; log.

### Phase 4 — streaming buckets + CI/docs
- Confirm streaming path picks the right small buckets per growing window; padding-mask if WER per small bucket regresses.
- Wire `gigastt` server passthrough; quickstart `docs/`; (optional) palettized ~215 MB variant.
- CI: run `cargo test -p gigastt-core --features ane --lib` on macos-14; model-gated parity stays manual/nightly.

## Open risks
- objc2-core-ml is low-level unsafe ObjC interop — Phase 2 is the riskiest lane.
- Padding-attention may need a key-padding mask for small buckets (validate in Phase 3).
- e2e RTFx gated by the CPU RNN-T loop — ANE speeds the encoder, not the decode loop.
- Distribution size: one FP16 `.mlpackage` ≈ 421 MB (palettize later to ~215 MB).

---

# Quickstart (user guide)

gigastt's default inference backend is ONNX Runtime (`ort`). An **optional**
native Apple **Neural Engine** backend is available behind the `ane` Cargo
feature: it runs the GigaAM v3 **encoder** on the ANE via per-bucket fixed-shape
Core ML `.mlpackage`s, while the decoder/joiner (and any encoder window outside
the bucket fill floor) stay on the `ort` CPU path.

It is **additive and opt-in** — the default build is unchanged and still uses `ort`.

## Status

- **macOS ARM64 (Apple Silicon) only.** The backend links Apple's Core ML
  framework; on every other target the `ane` feature degrades to the `ort` path.
- Targets the default **`rnnt`** head (char vocab). An `e2e_rnnt` model
  transparently falls back to the `ort` encoder (the ANE backend is rnnt-only,
  mirroring `candle`).
- **File-mode** backend: the encoder window is padded up to the nearest fixed
  bucket and run on the ANE. **Streaming and short windows below the fill floor
  fall back to the CPU/`ort` encoder** — they work without crashing but get no
  ANE speedup (this is intentional; see [Behavior](#behavior)).
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

**Today (no published ANE release):** convert + palettize them locally from the
PyTorch model. Run on macOS ARM64:

```sh
uv run --python 3.13 \
    --with torch --with coremltools --with gigaam --with soundfile --with scikit-learn \
    python scripts/convert_gigaam_ane.py
```

This writes the per-bucket `.mlpackage`s into `~/.gigastt/models/ane/`.

**Once an ANE release is published** (maintainer-gated; deferred for now):

```sh
gigastt download --ane
```

fetches the per-bucket packages (SHA-256-verified) into the same directory.
Until then `gigastt download --ane` has no release to pull from — use the local
conversion above.

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
- **Streaming falls back to CPU.** The streaming window is capped at 2.5 s
  (≤ 250 mel frames), which is below the 256-frame floor of the smallest (512)
  bucket, so **every streaming window takes the `ort` fallback**. Streaming works exactly as
  on the default build — no crash, no ANE benefit. ANE is a file-mode
  accelerator.
- **Over-max windows.** Files longer than the largest bucket use gigastt's
  existing 24 s windowed chunking; any window outside the bucket range falls back
  to `ort` (no silent truncation).

## Performance & accuracy (honest numbers)

- **End-to-end RTFx ≈ 3.7× over the `ort` CPU build** (≈ 8.7 → 32 RTFx on the
  measured Golos clips), **decode-bound**: the ANE makes the encoder almost free
  (~230× encoder speedup), but the CPU RNN-T greedy decode loop dominates the
  full pipeline, so the e2e win is far smaller than the raw encoder win.
- **WER vs `ort` ≈ 1.11%** on the 15-clip Golos set: transcripts are
  byte-identical except for a single borderline FP16-pad-up token flip on one
  clip. The ANE encoder is FP16 (not byte-exact), so parity is "near-lossless",
  not bit-exact (unlike the `candle` backend, which is byte-for-byte identical).

## Confirming the ANE path is engaged

- **Startup log:** `gigastt serve --features ane` on a `rnnt` model logs
  `ANE encoder backend active (Core ML / Apple Neural Engine, macOS ARM64): …`.
  On an `e2e_rnnt` model it instead logs that the head is not `rnnt` and the ort
  encoder is used.
- **Per-window debug log:** at `--log-level debug` the encoder logs
  `ANE encoder path (bucketed pad-up)` (with the chosen bucket) for file-mode
  windows, and `ANE encoder path (ort fallback: no bucket within fill-floor)`
  for streaming / sub-floor windows.
- **RTFx:** a file transcription completing well above real time (with the
  decode loop, not the encoder, as the bottleneck) confirms the ANE path.
