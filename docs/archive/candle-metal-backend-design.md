# Candle/Metal Inference Backend — Design

**Status:** approved (conversational brainstorming, 2026-06-23)
**Owner:** ekhodzitsky

## Goal

Add an **optional** pure-Rust inference backend (Candle, with the Metal GPU
backend on Apple Silicon) for GigaAM v3, behind the runtime-abstraction seam
introduced in PR #115. It must give a large speedup on Apple Silicon (the ort
CoreML EP does not accelerate the conformer; reference: RustASR reports
RTF ≈ 0.017 for GigaAM v3 on Candle/Metal) **without changing or risking the
default ONNX Runtime path**.

## Non-goals

- Removing or replacing `ort` (it stays the default backend).
- WASM (roadmap 85, separate).
- The `e2e_rnnt` head (target the default `rnnt` head first).
- INT8 / quantization in the first iteration (FP32 on Metal first; Candle-native
  GGUF quantization is a later, separate step).

## Why this is safe ("C as an option, doesn't break current")

The backend is gated behind a new compile-time Cargo feature `candle`:

- Candle deps (`candle-core`, `candle-nn`, `safetensors`) are `optional = true`
  and only pulled in by the `candle` feature. **Default builds do not compile
  Candle at all** — zero new default dependencies, zero change to the ort path.
- All Candle code lives in a new module `crates/gigastt-core/src/runtime/candle/`,
  parallel to `runtime/ort/`. Shared code (Engine, decode loop, streaming,
  server, punct, VAD) talks only to the `#115` traits and is untouched.
- `candle` is **mutually exclusive** with `coreml`/`cuda` (compile-time
  `compile_error!` guard, like the existing coreml⊕cuda guard). Default (CPU
  ort) is unchanged.
- The existing "Runtime Isolation" CI guard is extended so Candle types cannot
  leak outside `runtime/candle/`.

The only integration risk is honoring the **exact tensor I/O contract** the
decode loop expects (below); all adaptation lives inside the backend, never in
shared code.

## The #115 seam (contract a backend must implement)

From `crates/gigastt-core/src/runtime/{factory,session,tensor}.rs` on `main`:

```rust
pub trait RuntimeFactory: Send + Sync + 'static {
    fn create(&self, intra_threads: usize) -> Result<Box<dyn Runtime>, RuntimeError>;
    fn cpu_fallback(&self) -> Box<dyn RuntimeFactory>;
}
pub trait Runtime: Send + Sync + 'static {
    fn load_session(&self, model_path: &Path, is_encoder: bool)
        -> Result<Box<dyn RuntimeSession>, RuntimeError>;
}
pub trait RuntimeSession: Send + Sync + 'static {
    fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, RuntimeError>;
}
// Tensor = Shape{dims:Vec<usize>} + TensorData::{F32(Vec<f32>),I32,I64}
```

Recurrent state is **external** (gigastt threads decoder LSTM h/c in/out as
tensors), so each `RuntimeSession` is stateless: tensors in, tensors out.

### Exact per-session tensor contract (rnnt head)

Established from `inference/decode.rs` and `inference/mod.rs` on `main`.
Constants: `N_MELS=64`, `ENC_DIM=768`, `PRED_HIDDEN=320`, `ENCODER_SUBSAMPLING=4`.

| Session | `load_session` selects on | Inputs (ordered) | Outputs (ordered) |
|---|---|---|---|
| encoder | `is_encoder == true` | `audio_signal` `[1,64,T]` F32; `length` `[1]` I32/I64 | `[1,768,T/4]` F32 (channels-first) |
| decoder | filename contains `decoder` | `prev_token` `[1,1]` I64; `h` `[1,1,320]` F32; `c` `[1,1,320]` F32 | `dec` `[320]` F32; `new_h` `[320]` F32; `new_c` `[320]` F32 |
| joiner | filename contains `joint`/`joiner` | `enc_frame` `[1,768,1]` F32; `dec_data` `[1,320,1]` F32 | `logits` `[V]` F32 (V=34 for rnnt) |

`load_session` only gets `is_encoder: bool`; decoder vs joiner is resolved by
inspecting the model_path filename (`v3_rnnt_decoder.onnx` vs `v3_rnnt_joint.onnx`).
For the Candle backend the path identifies which sub-model to build; weights are
loaded from a sibling safetensors file (see Weights).

## Reuse decision (research-before-building)

Inspected `askidmobile/RustASR` (license **MIT OR Apache-2.0**):

- `crates/model-gigaam/src/conformer.rs` (~600 LOC) is a clean, **standalone**
  Candle implementation of the **GigaAM v3 conformer encoder** (RoPE attention,
  conv1d ×4 subsampling, macaron FFN, GLU conv module, SiLU). Its only non-Candle
  import is `crate::config::EncoderConfig`. Output layout is
  `(batch, d_model, seq/4)` = **`[1,768,T/4]`, matching our encoder contract**.
  It already contains hard-won Metal correctness fixes (per-4-layer
  `device.synchronize()` to avoid an AGXMetalG16X crash on M4/macOS 26.x; the
  "no xscale" v3 detail). Weight tensor keys match the official PyTorch
  `salute-developers/GigaAM`.
- The RNN-T head (LSTM prediction net + joiner) is **not** in `model-gigaam`
  (which is CTC), but `crates/model-parakeet` implements a TDT transducer
  (LSTM prediction + joint) in Candle — a structural reference for our head.

**Decision:** vendor the conformer encoder (+ its config) with attribution;
hand-write the RNN-T decoder/joiner referencing model-parakeet. Build only the
genuinely-new part.

## Weights

GigaAM v3 PyTorch checkpoints (salute-developers/GigaAM, rnnt head) →
safetensors via an adapted `scripts/convert_gigaam.py`. Encoder keys are reused
as-is (they match the vendored conformer's `VarBuilder` paths); add decoder
(LSTM prediction net) and joiner keys. Converted safetensors are hosted
alongside the model (a HuggingFace repo, like the existing rupunct artifact) and
fetched into the model dir; the Candle `Runtime::load_session` reads them
instead of the `.onnx` files. The `.onnx` files remain for the ort backend.

## Validation gates

- **Encoder numeric parity:** Candle encoder output vs ort encoder output on a
  fixed clip, max abs diff under tolerance (target ≤ 1e-2 FP32 on Metal/CPU).
- **WER parity:** the existing benchmark harness (`tests/benchmark.rs`,
  Golos fixtures) run with the `candle` backend must match the ort `rnnt`
  baseline within a small WER delta (target ≤ +0.1 abs WER%).
- **Default path untouched:** default-feature build + unit/e2e suites unchanged;
  `compile_error!` on `candle`+`coreml`/`cuda`.

## Phasing

0. Scaffold: `candle` feature, optional deps, empty `runtime/candle/` impl
   behind the seam, isolation-guard + CI lane. Default build unchanged.
1. Encoder: vendor conformer, weight conversion, `CandleSession` for encoder,
   numeric parity vs ort.
2. RNN-T head: decoder (LSTM) + joiner sessions honoring the contract; full
   file transcription; WER parity on Golos.
3. Streaming + pool: per-chunk decode through `StreamingState`, warmup, pooling.
4. CI/docs: dedicated `--features candle` macOS-Metal lane, quickstart docs,
   (optional) Candle-native quantization follow-up.
