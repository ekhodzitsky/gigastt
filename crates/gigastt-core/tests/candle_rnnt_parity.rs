//! Per-stage numeric parity gate for the Candle RNN-T decoder + joiner vs ort.
//!
//! Model-gated (`#[ignore]`): requires the GigaAM v3 rnnt decoder/joiner ONNX
//! (`~/.gigastt/models/v3_rnnt_{decoder,joint}.onnx`) and the converted Candle
//! weights (`~/.gigastt/models/candle/{decoder,joiner}.safetensors`).
//!
//! Run with:
//! `cargo test -p gigastt-core --features candle --test candle_rnnt_parity -- --ignored --nocapture`
//!
//! Both backends are driven through the same `RuntimeSession` seam with identical
//! inputs; outputs are compared element-wise. The tolerance is intentionally
//! tight (max abs diff < 1e-4): the point of this test is to PROVE the pure-Rust
//! decoder/joiner reproduce the ONNX graphs bit-for-bit (within f32 op ordering),
//! not to rubber-stamp them.
#![cfg(feature = "candle")]

use std::path::Path;

use gigastt_core::runtime_api::{
    Runtime, RuntimeSession, Shape, Tensor, TensorData, TensorDataView, candle_factory, cpu_factory,
};

const PRED_HIDDEN: usize = 320;
const ENC_DIM: usize = 768;
const VOCAB: usize = 34;

fn model_dir() -> std::path::PathBuf {
    Path::new(&gigastt_core::model::default_model_dir()).to_path_buf()
}

fn as_f32(t: &Tensor) -> Vec<f32> {
    match t.view().data() {
        TensorDataView::F32(v) => v.to_vec(),
        other => panic!("expected f32 tensor, got {other:?}"),
    }
}

/// Max-abs and mean-abs diff between two equal-length f32 slices.
fn diffs(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    let mut max_abs = 0.0_f64;
    let mut sum_abs = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x as f64 - *y as f64).abs();
        max_abs = max_abs.max(d);
        sum_abs += d;
    }
    (max_abs, sum_abs / a.len() as f64)
}

#[test]
#[ignore = "requires v3_rnnt decoder/joint ONNX + candle/{decoder,joiner}.safetensors"]
fn candle_decoder_matches_ort() {
    let dir = model_dir();
    let dec_onnx = dir.join("v3_rnnt_decoder.onnx");
    assert!(
        dec_onnx.exists(),
        "decoder ONNX not found at {}",
        dec_onnx.display()
    );
    assert!(
        dir.join("candle/decoder.safetensors").exists(),
        "candle/decoder.safetensors not found under {}",
        dir.display()
    );

    // Use the raw ort factory directly: `production_factory` auto-selects the
    // Candle backend for an rnnt model dir, which would make this a candle-vs-candle
    // comparison instead of the intended ort-vs-candle parity gate.
    let ort = cpu_factory().create(1).expect("ort runtime");
    let ort_sess = ort
        .load_session(&dec_onnx, false)
        .expect("ort load decoder");
    let candle = candle_factory().create(1).expect("candle runtime");
    let candle_sess = candle_factory_load(candle.as_ref(), &dec_onnx);

    // 3-step sequence: tokens [5, 12, 0]; start h=c=zeros[1,1,320]; each step
    // feeds prev_token + the PREVIOUS step's h/c (carried independently per
    // backend, but they should stay in lockstep if parity holds).
    let tokens: [i64; 3] = [5, 12, 0];
    let mut ort_h = vec![0.0_f32; PRED_HIDDEN];
    let mut ort_c = vec![0.0_f32; PRED_HIDDEN];
    let mut cd_h = vec![0.0_f32; PRED_HIDDEN];
    let mut cd_c = vec![0.0_f32; PRED_HIDDEN];

    let mut worst = 0.0_f64;
    for (step, &tok) in tokens.iter().enumerate() {
        let ort_out = run_decoder(ort_sess.as_ref(), tok, &ort_h, &ort_c);
        let cd_out = run_decoder(candle_sess.as_ref(), tok, &cd_h, &cd_c);

        let (dec_max, dec_mean) = diffs(&ort_out.0, &cd_out.0);
        let (h_max, h_mean) = diffs(&ort_out.1, &cd_out.1);
        let (c_max, c_mean) = diffs(&ort_out.2, &cd_out.2);

        eprintln!(
            "DECODER step {step} tok={tok}: dec max={dec_max:.3e} mean={dec_mean:.3e} | \
             h max={h_max:.3e} mean={h_mean:.3e} | c max={c_max:.3e} mean={c_mean:.3e}"
        );

        worst = worst.max(dec_max).max(h_max).max(c_max);

        assert!(
            dec_max < 1e-4,
            "decoder `dec` diverges at step {step}: max_abs={dec_max:.6e}"
        );
        assert!(
            h_max < 1e-4,
            "decoder `h` diverges at step {step}: max_abs={h_max:.6e}"
        );
        assert!(
            c_max < 1e-4,
            "decoder `c` diverges at step {step}: max_abs={c_max:.6e}"
        );

        // Advance both backends' states with their own outputs.
        ort_h = ort_out.1;
        ort_c = ort_out.2;
        cd_h = cd_out.1;
        cd_c = cd_out.2;
    }

    eprintln!("DECODER PARITY worst max_abs_diff across all steps = {worst:.6e}");
}

#[test]
#[ignore = "requires v3_rnnt decoder/joint ONNX + candle/{decoder,joiner}.safetensors"]
fn candle_joiner_matches_ort() {
    let dir = model_dir();
    let joint_onnx = dir.join("v3_rnnt_joint.onnx");
    assert!(
        joint_onnx.exists(),
        "joiner ONNX not found at {}",
        joint_onnx.display()
    );
    assert!(
        dir.join("candle/joiner.safetensors").exists(),
        "candle/joiner.safetensors not found under {}",
        dir.display()
    );

    // Use the raw ort factory directly: `production_factory` auto-selects the
    // Candle backend for an rnnt model dir, which would make this a candle-vs-candle
    // comparison instead of the intended ort-vs-candle parity gate.
    let ort = cpu_factory().create(1).expect("ort runtime");
    let ort_sess = ort
        .load_session(&joint_onnx, false)
        .expect("ort load joiner");
    let candle = candle_factory().create(1).expect("candle runtime");
    let candle_sess = candle_factory_load(candle.as_ref(), &joint_onnx);

    // 2 deterministic non-zero input pairs: a ramp and a sin fill.
    let enc_ramp: Vec<f32> = (0..ENC_DIM).map(|i| (i as f32) * 0.001 - 0.3).collect();
    let dec_ramp: Vec<f32> = (0..PRED_HIDDEN).map(|i| (i as f32) * 0.002 - 0.2).collect();
    let enc_sin: Vec<f32> = (0..ENC_DIM).map(|i| (i as f32 * 0.05).sin()).collect();
    let dec_sin: Vec<f32> = (0..PRED_HIDDEN).map(|i| (i as f32 * 0.07).cos()).collect();

    let mut worst = 0.0_f64;
    for (pair, (enc, dec)) in [(enc_ramp, dec_ramp), (enc_sin, dec_sin)]
        .iter()
        .enumerate()
    {
        let ort_logits = run_joiner(ort_sess.as_ref(), enc, dec);
        let cd_logits = run_joiner(candle_sess.as_ref(), enc, dec);

        assert_eq!(
            ort_logits.len(),
            VOCAB,
            "ort joiner output not length {VOCAB}"
        );
        assert_eq!(
            cd_logits.len(),
            VOCAB,
            "candle joiner output not length {VOCAB}"
        );

        let (max, mean) = diffs(&ort_logits, &cd_logits);
        eprintln!("JOINER pair {pair}: logits max={max:.3e} mean={mean:.3e} (n={VOCAB})");
        worst = worst.max(max);

        assert!(
            max < 1e-4,
            "joiner logits diverge for pair {pair}: max_abs={max:.6e}"
        );
    }

    eprintln!("JOINER PARITY worst max_abs_diff = {worst:.6e}");
}

/// Load a candle session for `onnx_path` through the candle runtime's
/// filename-dispatched `load_session` (decoder/joiner read sibling safetensors).
fn candle_factory_load(runtime: &dyn Runtime, onnx_path: &Path) -> Box<dyn RuntimeSession> {
    runtime
        .load_session(onnx_path, false)
        .expect("candle load session")
}

/// Run one decoder step; returns (dec, new_h, new_c), each length PRED_HIDDEN.
fn run_decoder(
    sess: &dyn RuntimeSession,
    token: i64,
    h: &[f32],
    c: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let prev = Tensor::new(Shape::new(vec![1, 1]), TensorData::I64(vec![token])).unwrap();
    let h_t = Tensor::new(
        Shape::new(vec![1, 1, PRED_HIDDEN]),
        TensorData::F32(h.to_vec()),
    )
    .unwrap();
    let c_t = Tensor::new(
        Shape::new(vec![1, 1, PRED_HIDDEN]),
        TensorData::F32(c.to_vec()),
    )
    .unwrap();
    let out = sess.run(&[prev, h_t, c_t]).expect("decoder run");
    assert_eq!(out.len(), 3, "decoder must return [dec, h, c]");
    (as_f32(&out[0]), as_f32(&out[1]), as_f32(&out[2]))
}

/// Run the joiner on one (enc, dec) pair; returns the flattened logits.
fn run_joiner(sess: &dyn RuntimeSession, enc: &[f32], dec: &[f32]) -> Vec<f32> {
    let enc_t = Tensor::new(
        Shape::new(vec![1, ENC_DIM, 1]),
        TensorData::F32(enc.to_vec()),
    )
    .unwrap();
    let dec_t = Tensor::new(
        Shape::new(vec![1, PRED_HIDDEN, 1]),
        TensorData::F32(dec.to_vec()),
    )
    .unwrap();
    let out = sess.run(&[enc_t, dec_t]).expect("joiner run");
    assert_eq!(out.len(), 1, "joiner must return [logits]");
    as_f32(&out[0])
}
