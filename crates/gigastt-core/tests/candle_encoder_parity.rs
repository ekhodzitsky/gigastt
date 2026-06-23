//! Numeric parity gate for the Candle Conformer encoder vs the ort encoder.
//!
//! Model-gated (`#[ignore]`): requires the GigaAM v3 rnnt FP32 encoder
//! (`~/.gigastt/models/v3_rnnt_encoder.onnx`) and the converted Candle weights
//! (`~/.gigastt/models/candle/encoder.safetensors`).
//!
//! Run with:
//! `cargo test -p gigastt-core --features candle --test candle_encoder_parity -- --ignored --nocapture`
//!
//! Both backends are driven through the same `RuntimeSession` seam with an
//! identical mel input (computed by the engine's own `FeatureExtractor`), and
//! the encoder outputs are compared element-wise. The tolerance is intentionally
//! tight (max abs diff < 1e-2): the point of this test is to PROVE the
//! pure-Rust encoder reproduces the ONNX encoder, not to rubber-stamp it.
#![cfg(feature = "candle")]

use std::path::Path;

use gigastt_core::inference::FeatureExtractor;
use gigastt_core::inference::N_MELS;
use gigastt_core::inference::audio::decode_audio_file;
use gigastt_core::model::default_model_dir;
use gigastt_core::runtime_api::{
    Shape, Tensor, TensorData, TensorDataView, candle_factory, cpu_factory,
};

#[test]
#[ignore = "requires v3_rnnt model + candle/encoder.safetensors"]
fn candle_encoder_matches_ort() {
    let model_dir = default_model_dir();
    let model_dir = Path::new(&model_dir);

    // FP32 ONNX encoder: parity must be against the same precision the Candle
    // safetensors were converted from (NOT the INT8 encoder).
    let enc_onnx = model_dir.join("v3_rnnt_encoder.onnx");
    assert!(
        enc_onnx.exists(),
        "FP32 encoder not found at {}",
        enc_onnx.display()
    );
    assert!(
        model_dir.join("candle/encoder.safetensors").exists(),
        "candle/encoder.safetensors not found under {}",
        model_dir.display()
    );

    // Mel input from the SAME feature path the engine uses.
    let fixture = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../gigastt/tests/fixtures/golos_00.wav"
    );
    let samples = decode_audio_file(fixture).expect("decode fixture wav");
    let fe = FeatureExtractor::new();
    let (mel_flat, num_frames) = fe.compute(&samples);
    assert!(num_frames > 0, "no mel frames extracted");
    assert_eq!(
        mel_flat.len(),
        N_MELS * num_frames,
        "mel buffer length mismatch"
    );
    eprintln!(
        "fixture={fixture}  samples={}  mel_frames={num_frames}",
        samples.len()
    );

    // Encoder contract: [audio_signal [1, 64, T] F32, length [1] I64].
    let mel = Tensor::new(
        Shape::new(vec![1, N_MELS, num_frames]),
        TensorData::F32(mel_flat),
    )
    .expect("build mel tensor");
    let len = Tensor::new(
        Shape::new(vec![1]),
        TensorData::I64(vec![num_frames as i64]),
    )
    .expect("build length tensor");

    // ort encoder session. Use the ort CPU factory explicitly: under
    // `--features candle`, `production_factory` now returns the candle backend,
    // so the cross-backend parity check must pin the reference side to ort.
    let ort_runtime = cpu_factory().create(1).expect("ort runtime");
    let ort_sess = ort_runtime
        .load_session(&enc_onnx, true)
        .expect("ort load encoder");

    // candle encoder session (reads the sibling candle/encoder.safetensors).
    let candle_runtime = candle_factory().create(1).expect("candle runtime");
    let candle_sess = candle_runtime
        .load_session(&enc_onnx, true)
        .expect("candle load encoder");

    let o = ort_sess
        .run(&[mel.clone(), len.clone()])
        .expect("ort encoder run");
    let c = candle_sess.run(&[mel, len]).expect("candle encoder run");

    let o0 = &o[0];
    let c0 = &c[0];

    eprintln!("ort   output shape: {:?}", o0.shape().dims());
    eprintln!("candle output shape: {:?}", c0.shape().dims());
    assert_eq!(
        o0.shape().dims(),
        c0.shape().dims(),
        "encoder output shapes differ (ort {:?} vs candle {:?})",
        o0.shape().dims(),
        c0.shape().dims()
    );

    let od = match o0.view().data() {
        TensorDataView::F32(v) => v.to_vec(),
        other => panic!("ort encoder output not f32: {other:?}"),
    };
    let cd = match c0.view().data() {
        TensorDataView::F32(v) => v.to_vec(),
        other => panic!("candle encoder output not f32: {other:?}"),
    };
    assert_eq!(od.len(), cd.len(), "element count differs");

    let mut max_abs = 0.0_f64;
    let mut sum_abs = 0.0_f64;
    for (a, b) in od.iter().zip(cd.iter()) {
        let d = (*a as f64 - *b as f64).abs();
        max_abs = max_abs.max(d);
        sum_abs += d;
    }
    let mean_abs = sum_abs / od.len() as f64;

    eprintln!(
        "PARITY max_abs_diff = {max_abs:.6e}  mean_abs_diff = {mean_abs:.6e}  n = {}",
        od.len()
    );

    assert!(
        max_abs < 1e-2,
        "candle encoder diverges from ort: max_abs_diff = {max_abs:.6e} (>= 1e-2), mean_abs_diff = {mean_abs:.6e}"
    );
}
