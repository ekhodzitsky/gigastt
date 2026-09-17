//! Predictor-state probe for long-form window seams (model-gated, `#[ignore]`).
//!
//! Decodes the *same* encoded window three ways so the RNN-T prediction
//! network's contribution is isolated from encoder context:
//!
//! - `cold`: the production path (decode every frame from a fresh state);
//! - `cold_from_cut`: fresh state, decode only frames at or after the seam;
//! - `warm_from_cut`: state warmed by the preceding window's labels before the
//!   seam, decode only frames at or after the seam (no label replay).
//!
//! ```sh
//! GIGASTT_PROBE_AUDIO=/path/long_example.wav cargo test --release -p gigastt-core --lib \
//!   predictor_probe -- --ignored --nocapture --test-threads=1
//! ```

use super::*;
use crate::inference::decode::{self, TokenInfo};
use crate::inference::token_format::TokenFormatter;
use crate::inference::{PRED_HIDDEN, SECONDS_PER_FRAME};
use crate::model::ModelVariant;
use crate::runtime::tensor::{Shape, Tensor, TensorData, TensorDataView};

const ENC_DIM: usize = 768;
const FRAME_SAMPLES: usize = HOP_LENGTH * ENCODER_SUBSAMPLING;

struct Encoded {
    data: Vec<f32>,
    len: usize,
}

fn encode(engine: &Engine, triplet: &mut SessionTriplet, samples: &[f32]) -> Encoded {
    let (features, num_frames) = engine.features.compute(samples);
    triplet.encoder_inputs[0].resize_to(Shape::new(vec![1, N_MELS, num_frames]));
    triplet.encoder_inputs[0]
        .as_f32_mut()
        .unwrap()
        .copy_from_slice(&features);
    triplet.encoder_inputs[1].as_i64_mut().unwrap()[0] = num_frames as i64;
    let outputs = triplet.encoder.run(&triplet.encoder_inputs).unwrap();
    let len = match outputs[1].view().data() {
        TensorDataView::I32(v) => v[0] as usize,
        TensorDataView::I64(v) => v[0] as usize,
        _ => panic!("unexpected encoder length tensor"),
    };
    let data = outputs[0].view().data().as_f32().unwrap().to_vec();
    Encoded { data, len }
}

/// Greedy-decode frames `[f0, len)` of `enc` from `state`; word times are
/// absolute (`frame_offset` is the window's first frame).
fn decode_from(
    engine: &Engine,
    triplet: &SessionTriplet,
    enc: &Encoded,
    f0: usize,
    state: &mut DecoderState,
    frame_offset: usize,
) -> (Vec<TokenInfo>, Vec<WordInfo>) {
    let new_len = enc.len - f0;
    let mut sliced = vec![0f32; ENC_DIM * new_len];
    for ch in 0..ENC_DIM {
        sliced[ch * new_len..(ch + 1) * new_len]
            .copy_from_slice(&enc.data[ch * enc.len + f0..ch * enc.len + enc.len]);
    }
    let tensor = Tensor::new_checked(
        Shape::new(vec![1, ENC_DIM, new_len]),
        TensorData::F32(sliced),
    );
    let result = decode::greedy_decode(
        triplet.decoder.as_deref().unwrap(),
        triplet.joiner.as_deref().unwrap(),
        &tensor.view(),
        new_len,
        engine.tokenizer.blank_id(),
        state,
        None,
        None,
    )
    .unwrap();
    let tokens: Vec<TokenInfo> = result
        .tokens
        .iter()
        .map(|t| TokenInfo {
            token_id: t.token_id,
            frame_index: t.frame_index + f0,
            confidence: t.confidence,
        })
        .collect();
    let words = TokenFormatter::tokens_to_words(&engine.tokenizer, &tokens, frame_offset);
    (tokens, words)
}

/// The predictor state after emitting exactly `labels`, as the greedy loop
/// would leave it: `h`/`c` have consumed every label but the last, which is
/// `prev_token`.
fn warm_state(engine: &Engine, triplet: &SessionTriplet, labels: &[usize]) -> DecoderState {
    let decoder = triplet.decoder.as_deref().unwrap();
    let mut state = DecoderState::new(engine.tokenizer.blank_id());
    for &label in labels {
        let inputs = vec![
            Tensor::new_checked(
                Shape::new(vec![1, 1]),
                TensorData::I64(vec![state.prev_token]),
            ),
            Tensor::new_checked(
                Shape::new(vec![1, 1, PRED_HIDDEN]),
                TensorData::F32(state.h.clone()),
            ),
            Tensor::new_checked(
                Shape::new(vec![1, 1, PRED_HIDDEN]),
                TensorData::F32(state.c.clone()),
            ),
        ];
        let out = decoder.run(&inputs).unwrap();
        state
            .h
            .copy_from_slice(out[1].view().data().as_f32().unwrap());
        state
            .c
            .copy_from_slice(out[2].view().data().as_f32().unwrap());
        state.prev_token = label as i64;
    }
    state
}

struct Case {
    name: &'static str,
    shift_s: f64,
    prev: (f64, f64),
    next: (f64, f64),
    cut: f64,
    /// Target span in original (unshifted) time.
    target: (f64, f64),
}

fn hits(words: &[WordInfo], shift: f64, target: (f64, f64)) -> String {
    words
        .iter()
        .filter(|w| w.start - shift < target.1 + 0.05 && w.end - shift > target.0 - 0.05)
        .map(|w| {
            format!(
                "{}@{:.2}-{:.2}({:.2})",
                w.word,
                w.start - shift,
                w.end - shift,
                w.confidence
            )
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
#[ignore = "requires the GigaAM model and GIGASTT_PROBE_AUDIO"]
fn predictor_probe() {
    let audio = std::env::var("GIGASTT_PROBE_AUDIO").expect("GIGASTT_PROBE_AUDIO");
    let source = crate::inference::audio::decode_audio_file(&audio).unwrap();
    let cases = [
        Case {
            name: "temnu@3",
            shift_s: 3.0,
            prev: (0.0, 24.0),
            next: (22.0, 46.0),
            cut: 23.0,
            target: (21.52, 21.84),
        },
        Case {
            name: "pozlash@0",
            shift_s: 0.0,
            prev: (0.0, 24.0),
            next: (22.0, 46.0),
            cut: 23.0,
            target: (23.44, 24.16),
        },
        Case {
            name: "pozlash@3",
            shift_s: 3.0,
            prev: (0.0, 24.0),
            next: (22.0, 46.0),
            cut: 23.0,
            target: (23.44, 24.16),
        },
        Case {
            name: "raskayanya@0",
            shift_s: 0.0,
            prev: (22.0, 46.0),
            next: (44.0, 68.0),
            cut: 45.0,
            target: (64.08, 65.0),
        },
        Case {
            name: "raskayanya@3",
            shift_s: 3.0,
            prev: (44.0, 68.0),
            next: (66.0, 74.25),
            cut: 67.0,
            target: (64.08, 65.0),
        },
        Case {
            name: "slozhi@0",
            shift_s: 0.0,
            prev: (44.0, 68.0),
            next: (66.0, 71.25),
            cut: 67.0,
            target: (68.36, 68.8),
        },
        Case {
            name: "slozhi@3",
            shift_s: 3.0,
            prev: (44.0, 68.0),
            next: (66.0, 74.25),
            cut: 67.0,
            target: (68.36, 68.8),
        },
    ];
    for variant in [ModelVariant::Rnnt, ModelVariant::E2eRnnt] {
        let engine = Engine::load_with_pools_threads_variant(
            &crate::model::default_model_dir(),
            Some(variant),
            1,
            1,
            0,
            4,
        )
        .unwrap();
        let mut guard = engine.pool.checkout_blocking().unwrap();
        let triplet: &mut SessionTriplet = &mut guard;
        for case in &cases {
            let shift = (case.shift_s * 16000.0) as usize;
            let mut buf = vec![0f32; shift];
            buf.extend_from_slice(&source);
            let win = |a: f64, b: f64| {
                let s = (a * 16000.0).round() as usize;
                let e = ((b * 16000.0).round() as usize).min(buf.len());
                (s, &buf[s..e])
            };
            let (ps, prev_samples) = win(case.prev.0, case.prev.1);
            let (ns, next_samples) = win(case.next.0, case.next.1);
            let prev_enc = encode(&engine, triplet, prev_samples);
            let mut cold = DecoderState::new(engine.tokenizer.blank_id());
            let (prev_tokens, prev_words) = decode_from(
                &engine,
                triplet,
                &prev_enc,
                0,
                &mut cold,
                ps / FRAME_SAMPLES,
            );
            let cut_frame_abs = (case.cut / SECONDS_PER_FRAME).round() as usize;
            let labels: Vec<usize> = prev_tokens
                .iter()
                .filter(|t| t.frame_index + ps / FRAME_SAMPLES < cut_frame_abs)
                .map(|t| t.token_id)
                .collect();
            let next_enc = encode(&engine, triplet, next_samples);
            let f0 = cut_frame_abs - ns / FRAME_SAMPLES;
            let mut s_a = DecoderState::new(engine.tokenizer.blank_id());
            let (_, a) = decode_from(&engine, triplet, &next_enc, 0, &mut s_a, ns / FRAME_SAMPLES);
            let mut s_b = DecoderState::new(engine.tokenizer.blank_id());
            let (_, b) = decode_from(
                &engine,
                triplet,
                &next_enc,
                f0,
                &mut s_b,
                ns / FRAME_SAMPLES,
            );
            let mut s_c = warm_state(&engine, triplet, &labels);
            let (_, c) = decode_from(
                &engine,
                triplet,
                &next_enc,
                f0,
                &mut s_c,
                ns / FRAME_SAMPLES,
            );
            let mut s_d = warm_state(&engine, triplet, &labels);
            let (_, d) = decode_from(&engine, triplet, &next_enc, 0, &mut s_d, ns / FRAME_SAMPLES);
            let prefix_text = engine.tokenizer.decode(&labels);
            println!(
                "== {:?} {} prev={:?} next={:?} cut={} labels={} prefix_tail={:?}",
                variant,
                case.name,
                case.prev,
                case.next,
                case.cut,
                labels.len(),
                prefix_text
                    .chars()
                    .rev()
                    .take(40)
                    .collect::<String>()
                    .chars()
                    .rev()
                    .collect::<String>()
            );
            println!(
                "  prev_window   : {}",
                hits(&prev_words, case.shift_s, case.target)
            );
            println!("  cold          : {}", hits(&a, case.shift_s, case.target));
            println!("  cold_from_cut : {}", hits(&b, case.shift_s, case.target));
            println!("  warm_from_cut : {}", hits(&c, case.shift_s, case.target));
            println!("  warm_replay   : {}", hits(&d, case.shift_s, case.target));
            let text = |w: &[WordInfo]| {
                w.iter()
                    .map(|x| x.word.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            println!("  text cold          : {}", text(&a));
            println!("  text cold_from_cut : {}", text(&b));
            println!("  text warm_from_cut : {}", text(&c));
        }
    }
}

/// Full-context control: one encoder run over the whole (shifted) recording,
/// no windows at all, so the same spans are read with every second of context
/// the file has.
#[test]
#[ignore = "requires the GigaAM model and GIGASTT_PROBE_AUDIO"]
fn fullpass_probe() {
    let audio = std::env::var("GIGASTT_PROBE_AUDIO").expect("GIGASTT_PROBE_AUDIO");
    let source = crate::inference::audio::decode_audio_file(&audio).unwrap();
    let targets = [
        ("temnu", (21.52, 21.84)),
        ("pozlash", (23.44, 24.16)),
        ("raskayanya", (64.08, 65.0)),
        ("slozhi", (68.36, 68.8)),
    ];
    for variant in [
        ModelVariant::Rnnt,
        ModelVariant::E2eRnnt,
        ModelVariant::MlCtc,
    ] {
        let engine = Engine::load_with_pools_threads_variant(
            &crate::model::default_model_dir(),
            Some(variant),
            1,
            1,
            0,
            4,
        )
        .unwrap();
        let mut guard = engine.pool.checkout_blocking().unwrap();
        for shift_s in [0.0, 3.0, 22.0] {
            let mut buf = vec![0f32; (shift_s * 16000.0) as usize];
            buf.extend_from_slice(&source);
            let (features, num_frames) = engine.features.compute(&buf);
            let mut state = DecoderState::new(engine.tokenizer.blank_id());
            let (words, _) = engine
                .run_inference(
                    &mut guard, &features, num_frames, &mut state, 0, false, None, None,
                )
                .unwrap();
            println!(
                "== {:?} shift={} fullpass words={}",
                variant,
                shift_s,
                words.len()
            );
            for (name, target) in targets {
                println!("  {name:11}: {}", hits(&words, shift_s, target));
            }
        }
    }
}
