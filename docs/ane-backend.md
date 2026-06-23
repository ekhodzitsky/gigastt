# Native ANE (CoreML) backend for GigaAM v3 — design + implementation plan

> Optional `--features ane`: run the GigaAM v3 **encoder** on the Apple **Neural Engine** via a native Core ML `.mlpackage`, behind the PR #115 runtime seam. Additive/opt-in; default `ort` path unchanged.

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

ANE requires fixed shapes. Solution: **one `.mlpackage` using coremltools `EnumeratedShapes`** over a bucket ladder of mel-frame counts, with a **single mel input** (drop the `length` tensor — pad input with zeros up to the chosen bucket, run, then trim the output to `ceil(real_frames/4)`). EnumeratedShapes share weights → one ~421 MB FP16 file covers all buckets (NOT N× files). One variable input avoids the E5RT multi-input EnumeratedShapes limitation noted in the recon.
- **Bucket ladder (mel frames):** `[64, 128, 256, 512, 1024, 1536, 2400, 3000]` (≈0.6/1.3/2.6/5/10/15/24/30 s). Streaming (≤250) uses the first 2-3 buckets; file mode the larger ones. Pick the smallest bucket ≥ real frames; clamp >3000 to chunked 24 s windows (gigastt already chunks >30 s). Coverage logged; if a window exceeds the max bucket it falls back to the ort encoder (no silent truncation).
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
- `scripts/convert_gigaam_ane.py` — the spike's proven `convert.py` generalized to `EnumeratedShapes` over the bucket ladder, single mel input, FP16 mlprogram; outputs `gigaam_v3_encoder_ane.mlpackage`. Run on macOS with `uv --python 3.13 --with torch --with coremltools --with gigaam`.
- A `release-ane.yml` workflow (workflow_dispatch, macos-14) converts + zips + publishes the `.mlpackage` to a GitHub release `ane-v3-<date>`, prints SHA256.
- `model/mod.rs`: add `ANE_RELEASE_BASE` + checksum + an `ensure_ane_package(dir)` download path (mirror `ensure_prequantized_model_variant`). `gigastt download --ane` fetches it. `AneRuntime::load_session` reads `<model_dir>/ane/gigaam_v3_encoder_ane.mlpackage`, errors clearly (with the download hint) if absent.

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
