# Candle/Metal Inference Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional `--features candle` GigaAM v3 inference backend (Candle + Metal) behind the PR #115 runtime seam, without changing the default ONNX Runtime path.

**Architecture:** A new `crates/gigastt-core/src/runtime/candle/` module implements the `RuntimeFactory`/`Runtime`/`RuntimeSession` traits. The conformer encoder is vendored from RustASR's `model-gigaam` (MIT/Apache, RoPE, candle-core/nn); the RNN-T decoder (LSTM prediction net) and joiner are hand-written to honor the exact decode-loop tensor contract. Weights come from PyTorch→safetensors conversion. The feature is mutually exclusive with `coreml`/`cuda`; default builds never compile Candle.

**Tech Stack:** Rust 2024, `candle-core`/`candle-nn` (Metal/CPU), `safetensors`, existing `gigastt-core` runtime traits, `cargo` features.

---

## Conventions used in this plan

- All paths are relative to the repo root. The backend crate is `gigastt-core`.
- Run unit tests with `cargo test -p gigastt-core --lib <filter>` (NEVER bare `cargo test --workspace` — it pulls the ~2.5h WER benchmark).
- Build/test the new backend with the feature on:
  `cargo build -p gigastt-core --no-default-features --features candle` and
  `cargo test -p gigastt-core --features candle --lib <filter>`.
- The default (ort) path is validated with plain `cargo build` / `cargo test -p gigastt-core --lib`.
- "Parity" Python scripts run under `uv` + `python3.13` (local Python 3.14 is broken).

## File Structure (created/modified)

**Created (all gated behind `feature = "candle"`):**
- `crates/gigastt-core/src/runtime/candle/mod.rs` — module root + re-exports + isolation doc.
- `crates/gigastt-core/src/runtime/candle/factory.rs` — `CandleFactory: RuntimeFactory`, device selection (`Metal`/`Cpu`).
- `crates/gigastt-core/src/runtime/candle/runtime.rs` — `CandleRuntime: Runtime`, `load_session` dispatch on filename.
- `crates/gigastt-core/src/runtime/candle/session.rs` — `EncoderSession`/`DecoderSession`/`JoinerSession: RuntimeSession`, Tensor↔candle bridge.
- `crates/gigastt-core/src/runtime/candle/tensor.rs` — convert `runtime::tensor::Tensor` ↔ `candle_core::Tensor`.
- `crates/gigastt-core/src/runtime/candle/config.rs` — vendored `EncoderConfig` (+`v3_rnnt()` constructor).
- `crates/gigastt-core/src/runtime/candle/conformer.rs` — VENDORED encoder (RustASR), license header.
- `crates/gigastt-core/src/runtime/candle/rnnt.rs` — hand-written prediction LSTM + joiner.
- `scripts/convert_gigaam_candle.py` — PyTorch→safetensors conversion (rnnt head).
- `.github/workflows/` change: a `candle` build/test lane (macOS-14 Metal).

**Modified:**
- `crates/gigastt-core/Cargo.toml` — optional `candle-core`/`candle-nn`/`safetensors` deps + `candle` feature + `compile_error!` exclusivity.
- `crates/gigastt-core/src/runtime/mod.rs` — `#[cfg(feature="candle")] pub mod candle;`.
- `crates/gigastt-core/src/runtime/ort/factory.rs` — `default_factory()` gains a `#[cfg(feature="candle")]` arm returning `CandleFactory`.
- The "Runtime Isolation" guard config (the regex check) — allow `candle` types only under `runtime/candle/`.

---

## Phase 0 — Scaffold (default build stays byte-identical)

### Task 0.1: Add the `candle` feature and optional deps

**Files:**
- Modify: `crates/gigastt-core/Cargo.toml`

- [ ] **Step 1: Add optional deps + feature**

In `[dependencies]` add (pin to the candle version RustASR uses; check
`askidmobile/RustASR` root `Cargo.toml` `[workspace.dependencies]` and match it):

```toml
candle-core = { version = "0.9", optional = true }
candle-nn   = { version = "0.9", optional = true }
safetensors = { version = "0.4", optional = true }
```

In `[features]` add:

```toml
candle = ["dep:candle-core", "dep:candle-nn", "dep:safetensors"]
```

- [ ] **Step 2: Add the mutual-exclusivity guard**

In `crates/gigastt-core/src/lib.rs` (top level, next to any existing
coreml/cuda guard), add:

```rust
#[cfg(all(feature = "candle", any(feature = "coreml", feature = "cuda")))]
compile_error!("feature `candle` is mutually exclusive with `coreml`/`cuda`");
```

- [ ] **Step 3: Verify the default build is unchanged**

Run: `cargo build -p gigastt-core` then `cargo tree -p gigastt-core -e features | grep -c candle`
Expected: build OK; grep prints `0` (candle not in the default graph).

- [ ] **Step 4: Verify the feature compiles the deps**

Run: `cargo build -p gigastt-core --no-default-features --features candle`
Expected: compiles (no candle code yet, just deps resolve). On macOS this pulls the Metal backend.

- [ ] **Step 5: Commit**

```bash
git add crates/gigastt-core/Cargo.toml crates/gigastt-core/src/lib.rs Cargo.lock
git commit -m "feat(candle): add optional candle feature + deps (no code yet)"
```

### Task 0.2: Empty backend module behind the seam

**Files:**
- Create: `crates/gigastt-core/src/runtime/candle/mod.rs`
- Create: `crates/gigastt-core/src/runtime/candle/factory.rs`
- Create: `crates/gigastt-core/src/runtime/candle/runtime.rs`
- Create: `crates/gigastt-core/src/runtime/candle/session.rs`
- Modify: `crates/gigastt-core/src/runtime/mod.rs`

- [ ] **Step 1: Wire the module**

In `crates/gigastt-core/src/runtime/mod.rs` add:

```rust
#[cfg(feature = "candle")]
pub mod candle;
```

- [ ] **Step 2: Write `factory.rs` returning `unimplemented`-but-typed stubs**

```rust
//! Candle/Metal runtime factory. Gated behind `feature = "candle"`.
use std::path::Path;
use crate::runtime::{error::RuntimeError, factory::{Runtime, RuntimeFactory}, session::RuntimeSession, tensor::Tensor};

/// Device selection for the Candle backend.
#[derive(Clone, Copy, Debug)]
pub enum CandleDevice { Cpu, Metal }

pub struct CandleFactory { device: CandleDevice }

impl CandleFactory {
    pub fn new() -> Self {
        // Prefer Metal on Apple Silicon; fall back to CPU elsewhere.
        let device = if candle_core::Device::new_metal(0).is_ok() { CandleDevice::Metal } else { CandleDevice::Cpu };
        Self { device }
    }
    pub fn cpu() -> Self { Self { device: CandleDevice::Cpu } }
}

impl RuntimeFactory for CandleFactory {
    fn create(&self, _intra_threads: usize) -> Result<Box<dyn Runtime>, RuntimeError> {
        Ok(Box::new(super::runtime::CandleRuntime::new(self.device)?))
    }
    fn cpu_fallback(&self) -> Box<dyn RuntimeFactory> { Box::new(CandleFactory::cpu()) }
}
```

- [ ] **Step 3: Write `runtime.rs` + `session.rs` stubs that return a clear error**

`runtime.rs`:

```rust
use std::path::Path;
use candle_core::Device;
use crate::runtime::{error::RuntimeError, factory::Runtime, session::RuntimeSession};
use super::factory::CandleDevice;

pub struct CandleRuntime { device: Device }

impl CandleRuntime {
    pub fn new(dev: CandleDevice) -> Result<Self, RuntimeError> {
        let device = match dev {
            CandleDevice::Metal => Device::new_metal(0).map_err(|e| RuntimeError::Backend(e.to_string()))?,
            CandleDevice::Cpu => Device::Cpu,
        };
        Ok(Self { device })
    }
}

impl Runtime for CandleRuntime {
    fn load_session(&self, model_path: &Path, _is_encoder: bool)
        -> Result<Box<dyn RuntimeSession>, RuntimeError> {
        Err(RuntimeError::Backend(format!(
            "candle backend not yet implemented for {}", model_path.display()
        )))
    }
}
```

(Use whatever `RuntimeError` variant exists for backend errors; check
`runtime/error.rs` and reuse the existing constructor rather than inventing
`Backend` if it differs.)

- [ ] **Step 4: Verify it compiles with the feature**

Run: `cargo build -p gigastt-core --no-default-features --features candle`
Expected: compiles. Default build (`cargo build -p gigastt-core`) still excludes all of this.

- [ ] **Step 5: Commit**

```bash
git add crates/gigastt-core/src/runtime
git commit -m "feat(candle): scaffold runtime/candle backend module (stub load_session)"
```

### Task 0.3: Select CandleFactory in `default_factory()` under the feature

**Files:**
- Modify: `crates/gigastt-core/src/runtime/ort/factory.rs`

- [ ] **Step 1: Add the feature arm**

In `default_factory()`, add the candle arm FIRST (it owns its own device):

```rust
pub fn default_factory() -> Box<dyn RuntimeFactory> {
    #[cfg(feature = "candle")]
    { return Box::new(crate::runtime::candle::factory::CandleFactory::new()); }
    #[cfg(not(feature = "candle"))]
    {
        if cfg!(feature = "coreml") { Box::new(OrtFactory::coreml()) }
        else if cfg!(feature = "cuda") { Box::new(OrtFactory::cuda()) }
        else if cfg!(feature = "nnapi") { Box::new(OrtFactory::nnapi()) }
        else { Box::new(OrtFactory::cpu()) }
    }
}
```

Note: `production_factory()`/`cpu_factory()` stay ort-based; auxiliary models
(VAD, punct) keep running through ort even when `candle` is on — the Candle
backend targets the main encoder/decoder/joiner triplet first. (Revisit in a
later phase if desired.)

- [ ] **Step 2: Verify both builds**

Run: `cargo build -p gigastt-core` (default, ort path intact) and
`cargo build -p gigastt-core --no-default-features --features candle`
Expected: both compile.

- [ ] **Step 3: Commit**

```bash
git add crates/gigastt-core/src/runtime/ort/factory.rs
git commit -m "feat(candle): route default_factory to CandleFactory under feature"
```

### Task 0.4: Extend the Runtime Isolation guard + CI lane

**Files:**
- Modify: the isolation-guard config (find it: `rg -n 'OrtRuntime|Runtime Isolation' .github crates scripts`)
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Find and read the existing guard**

Run: `rg -n 'Runtime Isolation|\\bOrtRuntime\\b|runtime/ort' .github`
Read it; it asserts ort-only types stay in `runtime/ort/`. Add an analogous
rule: `candle_core`/`candle_nn` may appear only under `runtime/candle/`.

- [ ] **Step 2: Add a `candle` CI job** (macos-14, Metal available)

```yaml
  candle-build:
    name: Build (Candle/Metal)
    runs-on: macos-14
    steps:
      - uses: actions/checkout@v7
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo build -p gigastt-core --no-default-features --features candle
      - run: cargo clippy -p gigastt-core --no-default-features --features candle -- -D warnings
```

- [ ] **Step 3: Verify locally**

Run: `cargo clippy -p gigastt-core --no-default-features --features candle -- -D warnings`
Expected: no warnings.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows
git commit -m "ci(candle): add Candle/Metal build lane + isolation guard rule"
```

---

## Phase 1 — Encoder (vendor + weights + numeric parity)

### Task 1.1: Vendor the conformer encoder + config

**Files:**
- Create: `crates/gigastt-core/src/runtime/candle/config.rs`
- Create: `crates/gigastt-core/src/runtime/candle/conformer.rs`

- [ ] **Step 1: Copy the source files**

Copy `askidmobile/RustASR:crates/model-gigaam/src/config.rs` →
`runtime/candle/config.rs` and `:src/conformer.rs` → `runtime/candle/conformer.rs`.

- [ ] **Step 2: Add the attribution/license header to each vendored file**

Prepend:

```rust
// Vendored from askidmobile/RustASR (crates/model-gigaam), dual-licensed
// MIT OR Apache-2.0. Copyright (c) the RustASR authors. Adapted for gigastt:
// import paths changed; CTC head removed; only the v3 conformer encoder kept.
```

- [ ] **Step 3: Fix imports + trim**

- `conformer.rs`: change `use crate::config::EncoderConfig;` →
  `use super::config::EncoderConfig;`. It otherwise imports only
  `candle_core` / `candle_nn`. Remove nothing else.
- `config.rs`: keep `EncoderConfig`/`PreprocessorConfig`; drop `HeadConfig`
  (CTC) and `GigaAmConfig` fields we don't need, OR keep them — but add a
  `v3_rnnt()` encoder config constructor mirroring `v3_e2e_ctc()`'s
  `EncoderConfig` (same encoder: 16 layers, d_model 768, 16 heads, conv1d ×4,
  conv_kernel 5, rotary). Confirm against the converted checkpoint in Task 1.2.

- [ ] **Step 4: Verify it compiles under the feature**

Run: `cargo build -p gigastt-core --no-default-features --features candle`
Expected: compiles.

- [ ] **Step 5: Commit**

```bash
git add crates/gigastt-core/src/runtime/candle/{config.rs,conformer.rs}
git commit -m "feat(candle): vendor GigaAM v3 conformer encoder from RustASR (MIT/Apache)"
```

### Task 1.2: Weight conversion script (encoder first)

**Files:**
- Create: `scripts/convert_gigaam_candle.py`

- [ ] **Step 1: Adapt RustASR's converter**

Base it on `askidmobile/RustASR:scripts/convert_gigaam.py`. Load the
`salute-developers/GigaAM` v3 **rnnt** checkpoint, export encoder weights to
`encoder.safetensors` with the SAME tensor keys the vendored `conformer.rs`
`VarBuilder` expects (`pre_encode.conv.*`, `layers.{i}.norm_*`,
`layers.{i}.self_attn.linear_{q,k,v,out}`, `layers.{i}.conv.*`,
`layers.{i}.feed_forward{1,2}.linear{1,2}`, `layers.{i}.norm_out`). Run under
`uv run --python 3.13 python scripts/convert_gigaam_candle.py`.

- [ ] **Step 2: Inspect the produced keys vs the VarBuilder**

Run (reuse RustASR's helper): `uv run --python 3.13 python -c "from safetensors import safe_open; f=safe_open('encoder.safetensors','pt'); print('\n'.join(sorted(f.keys()))[:4000])"`
Expected: keys line up with `conformer.rs` `vb.pp(...)` paths. Fix the converter's
renaming until they match.

- [ ] **Step 3: Commit**

```bash
git add scripts/convert_gigaam_candle.py
git commit -m "feat(candle): PyTorch->safetensors converter for GigaAM v3 rnnt encoder"
```

### Task 1.3: `tensor.rs` bridge

**Files:**
- Create: `crates/gigastt-core/src/runtime/candle/tensor.rs`
- Test: same file, `#[cfg(test)] mod tests`

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::tensor::{Shape, Tensor as RTensor, TensorData};
    use candle_core::Device;

    #[test]
    fn test_f32_roundtrip_preserves_shape_and_data() {
        let r = RTensor::new(Shape::new(vec![1, 2, 3]),
            TensorData::F32(vec![1.0,2.0,3.0,4.0,5.0,6.0])).unwrap();
        let c = to_candle(&r, &Device::Cpu).unwrap();
        assert_eq!(c.dims(), &[1, 2, 3]);
        let back = from_candle(&c).unwrap();
        assert_eq!(back.view().data().as_f32().unwrap(), &[1.0,2.0,3.0,4.0,5.0,6.0]);
        assert_eq!(back.view().shape().dims(), &[1, 2, 3]);
    }
}
```

- [ ] **Step 2: Run it, verify it fails**

Run: `cargo test -p gigastt-core --no-default-features --features candle --lib runtime::candle::tensor`
Expected: FAIL (`to_candle`/`from_candle` not found).

- [ ] **Step 3: Implement the bridge**

```rust
use candle_core::{Device, Tensor as CTensor};
use crate::runtime::{error::RuntimeError, tensor::{Shape, Tensor, TensorData}};

pub fn to_candle(t: &Tensor, dev: &Device) -> Result<CTensor, RuntimeError> {
    let dims = t.view().shape().dims().to_vec();
    let map = |e: candle_core::Error| RuntimeError::Backend(e.to_string());
    match t.view().data() {
        crate::runtime::tensor::TensorDataView::F32(d) => CTensor::from_slice(d, dims, dev).map_err(map),
        crate::runtime::tensor::TensorDataView::I64(d) => CTensor::from_slice(d, dims, dev).map_err(map),
        crate::runtime::tensor::TensorDataView::I32(d) => CTensor::from_slice(d, dims, dev).map_err(map),
    }
}

pub fn from_candle(c: &CTensor) -> Result<Tensor, RuntimeError> {
    let dims = c.dims().to_vec();
    let map = |e: candle_core::Error| RuntimeError::Backend(e.to_string());
    let flat = c.flatten_all().map_err(map)?;
    let data = TensorData::F32(flat.to_vec1::<f32>().map_err(map)?);
    Tensor::new(Shape::new(dims), data).map_err(|e| RuntimeError::Backend(e.to_string()))
}
```

(Adjust to the real `Tensor`/`TensorView` API in `runtime/tensor.rs` — use the
exact accessor names. `from_candle` returns F32; decoder/joiner outputs are F32.)

- [ ] **Step 4: Run the test, verify it passes**

Run: `cargo test -p gigastt-core --no-default-features --features candle --lib runtime::candle::tensor`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/gigastt-core/src/runtime/candle/tensor.rs
git commit -m "feat(candle): Tensor<->candle bridge with roundtrip test"
```

### Task 1.4: `EncoderSession` + parity test vs ort

**Files:**
- Modify: `crates/gigastt-core/src/runtime/candle/session.rs`
- Modify: `crates/gigastt-core/src/runtime/candle/runtime.rs` (load encoder)
- Test: `crates/gigastt-core/tests/candle_encoder_parity.rs` (model-gated, `#[ignore]`)

- [ ] **Step 1: Implement encoder load + run**

In `runtime.rs` `load_session`, when `is_encoder`: build `ConformerEncoder`
from `EncoderConfig::v3_rnnt()` and a `VarBuilder` over the sibling
`encoder.safetensors` (in `model_path`'s parent dir), store in an
`EncoderSession`. In `session.rs`:

```rust
pub struct EncoderSession { enc: super::conformer::ConformerEncoder, device: candle_core::Device }

impl RuntimeSession for EncoderSession {
    fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, RuntimeError> {
        // inputs[0] = mel [1,64,T] f32; inputs[1] = length [1] (ignored: batch=1).
        let mel = super::tensor::to_candle(&inputs[0], &self.device)?;
        let out = self.enc.forward(&mel).map_err(|e| RuntimeError::Backend(e.to_string()))?; // [1,768,T/4]
        Ok(vec![super::tensor::from_candle(&out)?])
    }
}
```

- [ ] **Step 2: Write the parity test (model-gated)**

```rust
// tests/candle_encoder_parity.rs
#[test]
#[ignore = "requires GigaAM v3 rnnt model + converted encoder.safetensors"]
fn test_candle_encoder_matches_ort_within_tolerance() {
    // 1. Build mel features for tests/fixtures/golos_00.wav via the shared
    //    feature pipeline (crate::inference::features).
    // 2. Run the ort encoder (default factory) and the candle EncoderSession.
    // 3. assert max_abs_diff(ort_out, candle_out) < 1e-2.
    // (Fill with the real helpers from tests/common + inference::features.)
}
```

- [ ] **Step 3: Run parity (local, model present)**

Run: `cargo test -p gigastt-core --no-default-features --features candle --test candle_encoder_parity -- --ignored`
Expected: PASS (max abs diff < 1e-2). If it fails, debug against
RustASR's `scripts/full_encoder_comparison.py` reference dumps.

- [ ] **Step 4: Commit**

```bash
git add crates/gigastt-core/src/runtime/candle tests/candle_encoder_parity.rs
git commit -m "feat(candle): encoder session + numeric parity vs ort"
```

**GATE:** Do not start Phase 2 until encoder parity passes. This proves the
vendored encoder + weight conversion are correct on our checkpoint.

---

## Phase 2 — RNN-T head (decoder + joiner) + WER parity

### Task 2.1: Inspect the rnnt decoder/joiner PyTorch structure

**Files:**
- Modify: `scripts/convert_gigaam_candle.py`

- [ ] **Step 1: Dump the decoder/joiner submodule shapes**

Extend the converter to print the rnnt `decoder` (prediction LSTM: embedding +
LSTM(input=embed_dim, hidden=320) ) and `joint` (encoder_proj 768→D,
decoder_proj 320→D, then 257/34-class output) parameter names and shapes. Use
RustASR's `scripts/analyze_joint.py` / `compare_lstm_joint.py` as the reference
shape probes.

- [ ] **Step 2: Export decoder.safetensors + joiner.safetensors** with keys
  chosen to match the modules written in Task 2.2.

- [ ] **Step 3: Commit**

```bash
git add scripts/convert_gigaam_candle.py
git commit -m "feat(candle): export rnnt decoder+joiner weights to safetensors"
```

### Task 2.2: `DecoderSession` (prediction LSTM) honoring the contract

**Files:**
- Create: `crates/gigastt-core/src/runtime/candle/rnnt.rs`
- Modify: `session.rs`, `runtime.rs`
- Test: `rnnt.rs` `#[cfg(test)]` (shape test, no model)

- [ ] **Step 1: Write the failing shape test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    // The decoder session must accept [prev_token[1,1] i64, h[1,1,320], c[1,1,320]]
    // and return three outputs of length 320 each. Drive it with a tiny random-init
    // PredictionNet on Device::Cpu (no real weights) to assert shapes/contract.
    #[test]
    fn test_decoder_contract_shapes() {
        let dev = candle_core::Device::Cpu;
        let net = PredictionNet::new_for_test(&dev); // small init, vocab=34, hidden=320
        let (dec, h, c) = net.step(0i64, &vec![0.0f32;320], &vec![0.0f32;320]).unwrap();
        assert_eq!(dec.len(), 320);
        assert_eq!(h.len(), 320);
        assert_eq!(c.len(), 320);
    }
}
```

- [ ] **Step 2: Run, verify it fails**

Run: `cargo test -p gigastt-core --no-default-features --features candle --lib runtime::candle::rnnt`
Expected: FAIL (`PredictionNet` not found).

- [ ] **Step 3: Implement `PredictionNet`** (embedding + `candle_nn::LSTM`,
  hidden 320) and a `DecoderSession: RuntimeSession` that maps the contract:
  read `prev_token` (i64 scalar), `h`/`c` `[1,1,320]`; call `step`; return
  `dec`,`new_h`,`new_c` as `[320]` F32 tensors (matching `decode.rs`'s
  `decoder_outputs[0..3]` which are flattened to len-320 slices). Reference:
  RustASR `model-parakeet` prediction net.

- [ ] **Step 4: Run, verify it passes**

Run: `cargo test -p gigastt-core --no-default-features --features candle --lib runtime::candle::rnnt`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/gigastt-core/src/runtime/candle/rnnt.rs crates/gigastt-core/src/runtime/candle/session.rs
git commit -m "feat(candle): rnnt prediction LSTM decoder session"
```

### Task 2.3: `JoinerSession` honoring the contract

**Files:**
- Modify: `crates/gigastt-core/src/runtime/candle/rnnt.rs`, `session.rs`, `runtime.rs`
- Test: `rnnt.rs` `#[cfg(test)]`

- [ ] **Step 1: Write the failing shape test**

```rust
#[test]
fn test_joiner_contract_shapes() {
    let dev = candle_core::Device::Cpu;
    let j = Joiner::new_for_test(&dev); // enc 768, dec 320, vocab 34
    let logits = j.join(&vec![0.0f32;768], &vec![0.0f32;320]).unwrap();
    assert_eq!(logits.len(), 34);
}
```

- [ ] **Step 2: Run, verify it fails.**
Run: `cargo test -p gigastt-core --no-default-features --features candle --lib runtime::candle::rnnt`
Expected: FAIL (`Joiner` not found).

- [ ] **Step 3: Implement `Joiner`** (encoder_proj + decoder_proj, add, tanh,
  output linear → `[vocab]`) and `JoinerSession: RuntimeSession` reading
  `enc_frame [1,768,1]` + `dec_data [1,320,1]`, returning `logits [vocab]`.

- [ ] **Step 4: Run, verify it passes.** Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/gigastt-core/src/runtime/candle/rnnt.rs crates/gigastt-core/src/runtime/candle/session.rs crates/gigastt-core/src/runtime/candle/runtime.rs
git commit -m "feat(candle): rnnt joiner session"
```

### Task 2.4: End-to-end file transcription + WER parity

**Files:**
- Test: `crates/gigastt-core/tests/candle_wer_parity.rs` (model-gated)

- [ ] **Step 1: Write the parity test (model-gated)**

```rust
#[test]
#[ignore = "requires GigaAM v3 rnnt model + converted safetensors"]
fn test_candle_wer_matches_ort_rnnt_within_delta() {
    // Transcribe the Golos benchmark fixtures via Engine with the candle factory
    // and via the ort factory; assert WER_candle <= WER_ort + 0.1 (abs %).
    // Reuse the harness in tests/benchmark.rs.
}
```

- [ ] **Step 2: Run it (local, model present)**

Run: `cargo test -p gigastt-core --no-default-features --features candle --test candle_wer_parity -- --ignored`
Expected: PASS. If WER diverges, bisect with the per-stage parity dumps from
RustASR's `scripts/full_pipeline_compare.py`.

- [ ] **Step 3: Measure RTF** (log it; expect a large speedup vs CPU ort on Metal).

- [ ] **Step 4: Commit**

```bash
git add crates/gigastt-core/tests/candle_wer_parity.rs
git commit -m "test(candle): end-to-end WER parity vs ort rnnt baseline"
```

**GATE:** WER parity within delta is the acceptance criterion for "C works".

---

## Phase 3 — Streaming + pool

### Task 3.1: Streaming decode through `StreamingState`

**Files:**
- Test: `crates/gigastt-core/tests/candle_streaming.rs` (model-gated)

- [ ] **Step 1: Write a model-gated streaming test** that feeds a WAV in chunks
  through `Engine::process_chunk` (candle factory) and asserts the final
  transcript matches the whole-file candle transcript (state threading correct).
  The decoder LSTM state is already external (`DecoderState`), so no new state
  plumbing is needed — this test proves that.

- [ ] **Step 2: Run it (local).** Expected: PASS.

- [ ] **Step 3: Verify pooling + warmup** work with the candle factory: run an
  existing e2e/load test with `--features candle` locally; confirm
  `Engine::warmup()` runs the candle sessions without error.

- [ ] **Step 4: Commit**

```bash
git add crates/gigastt-core/tests/candle_streaming.rs
git commit -m "test(candle): streaming parity + pool/warmup smoke"
```

---

## Phase 4 — CI/docs (and optional quantization follow-up)

### Task 4.1: Quickstart docs

**Files:**
- Modify: `docs/quickstarts.md` (and `README.md` build matrix if it lists features)

- [ ] **Step 1: Document the candle backend**: how to convert weights
  (`scripts/convert_gigaam_candle.py`), how to build
  (`cargo build --release --no-default-features --features candle`), the
  mutual-exclusivity with coreml/cuda, and that it is opt-in / experimental.

- [ ] **Step 2: Commit**

```bash
git add docs/quickstarts.md README.md
git commit -m "docs(candle): quickstart for the opt-in Candle/Metal backend"
```

### Task 4.2: (Optional, separate) Candle-native quantization

- [ ] Investigate GGUF Q8/Q6 weight loading in Candle for the encoder; gate
  behind a sub-option; re-run WER parity. Out of scope for the first merge.

---

## Self-Review

**Spec coverage:**
- "optional, doesn't break default" → Phase 0 (optional deps, feature gate, mutual exclusion, default-build verification steps). ✓
- "behind #115 seam" → factory/runtime/session implement the three traits; contracts table drives Tasks 1.4/2.2/2.3. ✓
- "reuse encoder" → Task 1.1 vendor + attribution. ✓
- "hand-write rnnt head" → Tasks 2.2/2.3. ✓
- "rnnt head first, FP32, Metal" → config `v3_rnnt()`, FP32 bridge, CandleDevice::Metal. ✓
- "weight conversion" → Tasks 1.2/2.1. ✓
- "numeric + WER parity gates" → Tasks 1.4/2.4. ✓
- "CI + isolation guard" → Task 0.4. ✓

**Placeholder scan:** Model-port internals (exact PyTorch→Candle weight-key
renames, decoder/joiner submodule layout) are intentionally derived in Tasks 1.2
and 2.1 by inspecting the real checkpoint, because they are facts that live in
the model, not assumptions — each such task has a concrete inspect-and-match
step and command, not a "TODO". Vendored encoder code is referenced by exact
source path rather than re-pasted (600 LOC). These are deliberate, not gaps.

**Type consistency:** `CandleFactory`/`CandleRuntime`/`{Encoder,Decoder,Joiner}Session`,
`to_candle`/`from_candle`, `PredictionNet`/`Joiner`, `EncoderConfig::v3_rnnt()`
are used consistently across tasks. Tensor contracts (PRED_HIDDEN=320,
ENC_DIM=768, N_MELS=64, vocab=34) match `decode.rs`/`inference/mod.rs` on `main`.

**Risk note (no silent caps):** the plan assumes our v3 rnnt encoder is the same
RoPE conformer as RustASR's v3 CTC encoder (shared v3 backbone). Task 1.2 Step 2
and Task 1.4 verify this empirically before any head work; if the encoder differs,
the converter's key-mapping (not the vendored module) is where it surfaces.
