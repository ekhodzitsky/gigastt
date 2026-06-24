//! ANE encoder session: runs the GigaAM v3 Conformer encoder on the Apple
//! Neural Engine via a per-bucket fixed-shape `.mlpackage`, with a pad-up +
//! fill-floor selection policy and an ort encoder fallback.
//!
//! ISOLATION: all `objc2_core_ml` usage stays in `runtime/coreml/` (the model
//! handle here is produced by [`super::bridge`]; this file only holds it and
//! calls `bridge::predict_f32`). Gated `#[cfg(all(feature = "ane",
//! target_os = "macos"))]`.

use std::sync::Arc;

use objc2::rc::Retained;
use objc2_core_ml::MLModel;

use crate::runtime::{
    error::RuntimeError,
    session::RuntimeSession,
    tensor::{Shape, Tensor, TensorData, TensorView},
};

use super::bridge;

/// Encoder output channel dim (`[1, ENC_DIM, T']`, channels-first).
const ENC_DIM: usize = 768;
/// Mel feature bins (`[1, N_MELS, T]`, channels-first).
const N_MELS: usize = 64;

/// Minimum fraction of a bucket the real mel must fill for the pad-up path to be
/// trusted. Below this, the mask-free zero-padded encoder output diverges enough
/// from the unpadded baseline that a borderline token can flip; clips that don't
/// reach the floor fall back to the variable-length ort encoder.
///
/// Calibrated by the pad-parity experiment (mask-free pad-up): at >= 50% fill the
/// padded ANE output stays at cosine >= 0.94 vs the ort baseline and WER tracks
/// the baseline; below 50% the raw output diverges and the transcript is no
/// longer trustworthy.
///
/// The cos >= 0.94 @ >= 50%-fill calibration was measured against the REAL FP16
/// `.mlpackage` (Float16 mel input, see [`super::bridge::predict_f32`]), so the
/// floor already absorbs the Float16 quantization of the mel input — not just the
/// zero-pad effect. A future reader must NOT "fix" this floor assuming the
/// encoder sees f32 input; the f16 round-trip is baked into the threshold.
const FILL_FLOOR: f64 = 0.5;

/// `Retained<MLModel>` is not auto-`Send`/`Sync` (it wraps an Objective-C
/// pointer), but Apple documents `-[MLModel predictionFromFeatures:error:]` as
/// thread-safe: a single loaded model may be used for prediction concurrently
/// from multiple threads. We only ever call prediction (never mutate the model)
/// once it is loaded, so sharing an `Arc<SharedModel>` across pooled encoder
/// sessions is sound.
///
/// SAFETY: the wrapped `MLModel` is immutable after load and Apple guarantees
/// `predictionFromFeatures:error:` is safe to call concurrently; no other
/// Objective-C method is invoked on it from this crate.
pub struct SharedModel(pub Retained<MLModel>);

// SAFETY: see the type-level doc above — prediction on a loaded MLModel is
// documented thread-safe and we never mutate it after load.
unsafe impl Send for SharedModel {}
// SAFETY: same justification — concurrent `&MLModel` prediction is supported.
unsafe impl Sync for SharedModel {}

/// One compiled bucket model paired with its mel-frame window size.
pub struct BucketModel {
    /// Padded mel window length `N` this package expects (`[1, 64, N]`).
    pub size: usize,
    /// Shared compiled model (loaded once, shared across all pool slots).
    pub model: Arc<SharedModel>,
}

/// Encoder time subsampling: two stride-2 conv layers (×4 total), each
/// `out = (in - 1) / 2 + 1` (integer division). Equivalent to `in.div_ceil(2)`
/// applied twice, which is exactly what the Candle `StridingSubsampling` and the
/// ONNX-derived bucket package compute, so the trim below stays aligned with the
/// frames the model actually emits.
pub fn calc_output_length(t: usize) -> usize {
    fn stride2(x: usize) -> usize {
        (x - 1) / 2 + 1
    }
    stride2(stride2(t))
}

/// Pick the smallest available bucket `N >= t` whose fill ratio `t / N` meets
/// `floor`. Returns `None` when no bucket satisfies both (caller falls back to
/// the variable-length ort encoder). `buckets` need not be sorted.
pub fn select_bucket(t: usize, buckets: &[usize], floor: f64) -> Option<usize> {
    buckets
        .iter()
        .copied()
        .filter(|&n| n >= t && t as f64 / n as f64 >= floor)
        .min()
}

/// Zero-pad a channels-first `[1, channels, t]` row-major mel buffer up to
/// `[1, channels, n]` (append `n - t` zero frames to each channel row). `t <= n`
/// is required; the layout is channel-major with the time axis contiguous-last,
/// so each of the `channels` rows is copied then zero-filled to length `n`.
pub fn pad_time(mel: &[f32], channels: usize, t: usize, n: usize) -> Vec<f32> {
    debug_assert!(t <= n, "pad target {n} must be >= source {t}");
    debug_assert_eq!(mel.len(), channels * t, "mel len mismatch in pad_time");
    let mut out = vec![0.0f32; channels * n];
    for c in 0..channels {
        let src = &mel[c * t..c * t + t];
        out[c * n..c * n + t].copy_from_slice(src);
    }
    out
}

/// Trim a channels-first `[1, channels, t_padded]` row-major buffer down to
/// `[1, channels, t_keep]` (keep the first `t_keep` frames of each channel row).
/// `t_keep <= t_padded` is required.
pub fn trim_time(out: &[f32], channels: usize, t_padded: usize, t_keep: usize) -> Vec<f32> {
    debug_assert!(t_keep <= t_padded, "trim {t_keep} must be <= {t_padded}");
    debug_assert_eq!(
        out.len(),
        channels * t_padded,
        "out len mismatch in trim_time"
    );
    let mut trimmed = Vec::with_capacity(channels * t_keep);
    for c in 0..channels {
        let row = &out[c * t_padded..c * t_padded + t_keep];
        trimmed.extend_from_slice(row);
    }
    trimmed
}

/// ANE-backed encoder session. Runs the encoder on the Neural Engine via a
/// per-bucket fixed-shape package when a clip pads-up into a bucket at >=
/// [`FILL_FLOOR`]; otherwise delegates to the ort encoder fallback.
pub struct AneEncoderSession {
    /// Available compiled bucket models, sorted ascending by `size`.
    buckets: Vec<BucketModel>,
    /// Variable-length ort encoder for clips outside the fill-floor / bucket range.
    ort_fallback: Box<dyn RuntimeSession>,
}

impl AneEncoderSession {
    pub fn new(mut buckets: Vec<BucketModel>, ort_fallback: Box<dyn RuntimeSession>) -> Self {
        buckets.sort_by_key(|b| b.size);
        Self {
            buckets,
            ort_fallback,
        }
    }

    fn bucket_sizes(&self) -> Vec<usize> {
        self.buckets.iter().map(|b| b.size).collect()
    }

    fn model_for(&self, size: usize) -> Option<&Arc<SharedModel>> {
        self.buckets
            .iter()
            .find(|b| b.size == size)
            .map(|b| &b.model)
    }
}

impl RuntimeSession for AneEncoderSession {
    /// Encoder contract (mirrors the ort ONNX encoder, which emits two outputs):
    /// `inputs[0] = mel [1, 64, T] F32`, `inputs[1] = length [1]` (ignored; T is
    /// read from the mel shape, which is authoritative on this path). Returns
    /// `[encoded [1, 768, T'] F32, encoded_len [1] I64]` so the engine's decode
    /// loop reads `encoder_outputs[1]` for the frame count exactly as for ort.
    fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, RuntimeError> {
        if inputs.is_empty() {
            return Err(RuntimeError::InvalidInputCount {
                expected: 2,
                got: inputs.len(),
            });
        }

        let mel_view: TensorView<'_> = inputs[0].view();
        let mel = mel_view.data().as_f32().ok_or_else(|| {
            RuntimeError::InferenceFailed("ANE encoder mel input is not f32".to_string())
        })?;
        let dims = mel_view.shape().dims();
        if dims.len() != 3 || dims[0] != 1 || dims[1] != N_MELS {
            return Err(RuntimeError::InferenceFailed(format!(
                "ANE encoder expects mel shape [1, {N_MELS}, T], got {dims:?}"
            )));
        }
        let t = dims[2];

        let sizes = self.bucket_sizes();
        match select_bucket(t, &sizes, FILL_FLOOR) {
            Some(n) => {
                let model = self.model_for(n).ok_or_else(|| {
                    RuntimeError::InferenceFailed(format!("no compiled model for bucket {n}"))
                })?;
                let padded = pad_time(mel, N_MELS, t, n);
                let (out, out_shape) =
                    bridge::predict_f32(&model.0, "mel", &padded, &[1, N_MELS, n], "encoded")?;
                // The package emits [1, 768, T'_N] for the padded window N.
                if out_shape.len() != 3 || out_shape[0] != 1 || out_shape[1] != ENC_DIM {
                    return Err(RuntimeError::InferenceFailed(format!(
                        "ANE encoder output shape {out_shape:?} != [1, {ENC_DIM}, T']"
                    )));
                }
                let t_padded_prime = out_shape[2];
                let t_real_prime = calc_output_length(t);
                if t_real_prime > t_padded_prime {
                    return Err(RuntimeError::InferenceFailed(format!(
                        "computed encoder output length {t_real_prime} exceeds bucket output {t_padded_prime}"
                    )));
                }
                tracing::debug!(
                    t,
                    bucket = n,
                    t_real_prime,
                    t_padded_prime,
                    "ANE encoder path (bucketed pad-up)"
                );
                let trimmed = trim_time(&out, ENC_DIM, t_padded_prime, t_real_prime);
                Ok(vec![
                    Tensor::new(
                        Shape::new(vec![1, ENC_DIM, t_real_prime]),
                        TensorData::F32(trimmed),
                    )?,
                    Tensor::new(
                        Shape::new(vec![1]),
                        TensorData::I64(vec![t_real_prime as i64]),
                    )?,
                ])
            }
            None => {
                tracing::debug!(
                    t,
                    "ANE encoder path (ort fallback: no bucket within fill-floor)"
                );
                self.ort_fallback.run(inputs)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calc_output_length_known_points() {
        // ×4 subsampling: matches T/4 for multiples of 4.
        assert_eq!(calc_output_length(768), 192);
        assert_eq!(calc_output_length(1500), 375);
        assert_eq!(calc_output_length(3000), 750);
        // Odd / non-multiple-of-4 values: ceil-div twice.
        assert_eq!(calc_output_length(593), 149);
        assert_eq!(calc_output_length(250), 63);
        assert_eq!(calc_output_length(769), 193);
        assert_eq!(calc_output_length(400), 100);
    }

    #[test]
    fn test_select_bucket_fill_floor_cases() {
        let buckets = [768usize, 1536, 3000];
        // 400/768 = 52% -> smallest N>=400 meeting floor.
        assert_eq!(select_bucket(400, &buckets, 0.5), Some(768));
        // 200/768 = 26% < floor -> no bucket, fallback.
        assert_eq!(select_bucket(200, &buckets, 0.5), None);
        // 800 > 768 so 768 fails N>=T; smallest N>=800 is 1536 (800/1536=52%).
        assert_eq!(select_bucket(800, &buckets, 0.5), Some(1536));
        // 769 -> 1536 (769/1536 = 50.06% >= floor).
        assert_eq!(select_bucket(769, &buckets, 0.5), Some(1536));
        // 4000 > max bucket -> fallback.
        assert_eq!(select_bucket(4000, &buckets, 0.5), None);
        // Exact fit.
        assert_eq!(select_bucket(768, &buckets, 0.5), Some(768));
    }

    #[test]
    fn test_select_bucket_fill_equal_to_floor_is_selected() {
        // 384/768 = exactly 0.5 == floor -> the `>=` comparison must include the
        // boundary, so the bucket is selected (not rejected to the ort fallback).
        assert_eq!(select_bucket(384, &[768], 0.5), Some(768));
    }

    #[test]
    fn test_select_bucket_unsorted_input() {
        let buckets = [3000usize, 768, 1536];
        assert_eq!(select_bucket(400, &buckets, 0.5), Some(768));
        assert_eq!(select_bucket(800, &buckets, 0.5), Some(1536));
    }

    #[test]
    fn test_pad_time_appends_zeros_per_channel() {
        // 2 channels, t=2, n=4: each row [a,b] -> [a,b,0,0].
        let mel = vec![1.0, 2.0, 3.0, 4.0];
        let padded = pad_time(&mel, 2, 2, 4);
        assert_eq!(padded, vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0]);
        assert_eq!(padded.len(), 2 * 4);
    }

    #[test]
    fn test_pad_time_noop_when_t_equals_n() {
        let mel = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let padded = pad_time(&mel, 2, 3, 3);
        assert_eq!(padded, mel);
    }

    #[test]
    fn test_trim_time_keeps_leading_frames_per_channel() {
        // 2 channels, t_padded=4, t_keep=2: each row [a,b,c,d] -> [a,b].
        let out = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let trimmed = trim_time(&out, 2, 4, 2);
        assert_eq!(trimmed, vec![1.0, 2.0, 5.0, 6.0]);
        assert_eq!(trimmed.len(), 2 * 2);
    }

    #[test]
    fn test_pad_then_trim_roundtrip_recovers_prefix() {
        // pad up then trim back to the original length recovers the real frames.
        let mel = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]; // 2 ch x 3
        let padded = pad_time(&mel, 2, 3, 5);
        let back = trim_time(&padded, 2, 5, 3);
        assert_eq!(back, mel);
    }

    /// Streaming smoke (no model / no ANE hardware): an `AneEncoderSession` with
    /// the production bucket ladder but NO compiled bucket models routes a
    /// streaming-sized window through the ort fallback rather than the ANE path.
    ///
    /// The streaming path (`Engine::decode_window`) caps its window at
    /// `STREAM_MAX_WINDOW_SAMPLES` = 2.5 s ⇒ ≤ 250 mel frames, which is below the
    /// 50%-fill floor of the smallest shipped bucket (768 ⇒ floor 384). So
    /// `select_bucket` returns `None` for every streaming window and `run`
    /// delegates to the ort fallback session: streaming works on CPU with no
    /// crash and no ANE benefit (the intended "ANE = file-mode" behavior). This
    /// pins that routing without needing a real `.mlpackage` or the Neural Engine
    /// by using a mock fallback session and asserting it was invoked.
    #[test]
    fn test_streaming_window_routes_to_ort_fallback() {
        use crate::runtime::mock::MockSession;

        // Production buckets; none has a compiled model loaded here.
        const SHIPPED_BUCKETS: &[usize] = &[768, 1536, 3000];
        // A streaming-sized window: 250 mel frames (the 2.5 s window cap), well
        // below the 384-frame fill floor of the 768 bucket.
        const T: usize = 250;

        // Sanity: confirm no shipped bucket accepts this window at the floor, so
        // `run` MUST take the fallback branch.
        assert_eq!(
            select_bucket(T, SHIPPED_BUCKETS, FILL_FLOOR),
            None,
            "a 250-frame streaming window must not select any shipped bucket"
        );

        // Mock ort encoder fallback: accepts the [1,64,T] mel + [1] length pair
        // and returns the encoder's two-output contract ([encoded], [encoded_len]).
        let t_prime = calc_output_length(T);
        let fallback = MockSession::new(
            vec![Shape::new(vec![1, N_MELS, T]), Shape::new(vec![1])],
            vec![
                Tensor::new(
                    Shape::new(vec![1, ENC_DIM, t_prime]),
                    TensorData::F32(vec![0.0; ENC_DIM * t_prime]),
                )
                .unwrap(),
                Tensor::new(Shape::new(vec![1]), TensorData::I64(vec![t_prime as i64])).unwrap(),
            ],
        );

        // No bucket models -> every window falls back.
        let session = AneEncoderSession::new(Vec::new(), Box::new(fallback));

        let mel = Tensor::new(
            Shape::new(vec![1, N_MELS, T]),
            TensorData::F32(vec![0.0; N_MELS * T]),
        )
        .unwrap();
        let len = Tensor::new(Shape::new(vec![1]), TensorData::I64(vec![T as i64])).unwrap();
        let out = session.run(&[mel, len]).expect("fallback run succeeds");

        // The fallback's recorded contract flows straight through.
        assert_eq!(out.len(), 2, "encoder emits [encoded, encoded_len]");
        assert_eq!(out[0].shape().dims(), &[1, ENC_DIM, t_prime]);
        match out[1].view().data() {
            crate::runtime::tensor::TensorDataView::I64(v) => {
                assert_eq!(v[0], t_prime as i64, "fallback encoded_len passes through")
            }
            other => panic!("expected I64 encoded_len, got {other:?}"),
        }
    }
}
