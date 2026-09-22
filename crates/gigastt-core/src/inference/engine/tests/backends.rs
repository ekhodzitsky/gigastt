#![allow(unused_imports)] // candle / ANE tests are cfg-gated

use super::*;
use std::path::Path;

/// End-to-end proof that the Candle backend transcribes IDENTICALLY to the
/// ort backend through the full engine pipeline (mel + encoder + RNN-T
/// decode), not just per-stage tensors.
///
/// Two engines are built on the SAME model dir — one forced onto the ort CPU
/// backend, one forced onto the Candle backend (which reads the sibling
/// `candle/*.safetensors`) — and the same fixture wav is transcribed by both.
/// Per-stage parity is bit-exact, so the decoded text MUST be byte-identical.
///
/// Run with:
/// `cargo test -p gigastt-core --features candle --lib -- --ignored --nocapture candle_ort_transcription_parity`
#[cfg(feature = "candle")]
#[test]
#[ignore = "requires v3_rnnt model + candle/*.safetensors"]
fn candle_ort_transcription_parity() {
    let model_dir = crate::model::default_model_dir();
    let model_path = Path::new(&model_dir);

    let ort_engine = Engine::load_with_factory(
        model_path,
        None,
        1,
        1,
        0,
        Box::new(crate::runtime::ort::factory::OrtFactory::cpu()),
        1,
        true,
        "cpu",
    )
    .expect("ort engine should load");
    let candle_engine = Engine::load_with_factory(
        model_path,
        None,
        1,
        1,
        0,
        Box::new(crate::runtime::candle::factory::CandleFactory::new()),
        1,
        true,
        "candle",
    )
    .expect("candle engine should load");

    let fixtures = [
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../gigastt/tests/fixtures/golos_00.wav"
        ),
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../gigastt/tests/fixtures/golos_01.wav"
        ),
    ];

    for fixture in fixtures {
        let mut ort_guard = ort_engine.pool.checkout_blocking().expect("ort checkout");
        let ort_text = ort_engine
            .transcribe_file(fixture, &mut ort_guard)
            .expect("ort transcription")
            .text;

        let mut candle_guard = candle_engine
            .pool
            .checkout_blocking()
            .expect("candle checkout");
        let candle_text = candle_engine
            .transcribe_file(fixture, &mut candle_guard)
            .expect("candle transcription")
            .text;

        eprintln!("fixture = {fixture}");
        eprintln!("ort    = {ort_text:?}");
        eprintln!("candle = {candle_text:?}");

        assert_eq!(
            ort_text, candle_text,
            "candle transcript diverges from ort for {fixture}:\n  ort    = {ort_text:?}\n  candle = {candle_text:?}"
        );
    }
}

/// End-to-end proof that the Candle backend produces IDENTICAL output to ort
/// through the STREAMING path (sliding-window `process_chunk` + `finish_stream`),
/// not just through the whole-file `transcribe_file` path.
///
/// Both engines receive the SAME 8 000-sample (0.5 s) chunks fed in order;
/// all returned segment texts are concatenated and compared byte-for-byte.
/// Per-stage tensor parity is bit-exact, so the streamed transcripts must match.
///
/// Run with:
/// `cargo test -p gigastt-core --features candle --lib -- --ignored --nocapture candle_ort_streaming_parity`
#[cfg(feature = "candle")]
#[test]
#[ignore = "requires v3_rnnt model + candle/*.safetensors"]
fn candle_ort_streaming_parity() {
    const CHUNK_SAMPLES: usize = 8_000; // 0.5 s at 16 kHz

    let model_dir = crate::model::default_model_dir();
    let model_path = Path::new(&model_dir);

    let ort_engine = Engine::load_with_factory(
        model_path,
        None,
        1,
        1,
        0,
        Box::new(crate::runtime::ort::factory::OrtFactory::cpu()),
        1,
        true,
        "cpu",
    )
    .expect("ort engine should load");
    let candle_engine = Engine::load_with_factory(
        model_path,
        None,
        1,
        1,
        0,
        Box::new(crate::runtime::candle::factory::CandleFactory::new()),
        1,
        true,
        "candle",
    )
    .expect("candle engine should load");

    let fixtures = [
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../gigastt/tests/fixtures/golos_00.wav"
        ),
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../gigastt/tests/fixtures/golos_01.wav"
        ),
    ];

    for fixture in fixtures {
        let samples = audio::decode_audio_file(fixture)
            .unwrap_or_else(|e| panic!("failed to decode {fixture}: {e:#}"));

        // --- ort streaming ---
        let mut ort_guard = ort_engine.pool.checkout_blocking().expect("ort checkout");
        let mut ort_state = ort_engine.create_state(false);
        let mut ort_text = String::new();
        for chunk in samples.chunks(CHUNK_SAMPLES) {
            let segs = ort_engine
                .process_chunk(chunk, &mut ort_state, &mut ort_guard)
                .expect("ort process_chunk must not error");
            for seg in segs {
                if !ort_text.is_empty() {
                    ort_text.push(' ');
                }
                ort_text.push_str(seg.text.trim());
            }
        }
        if let Some(seg) = ort_engine.finish_stream(&mut ort_state, &mut ort_guard) {
            if !ort_text.is_empty() {
                ort_text.push(' ');
            }
            ort_text.push_str(seg.text.trim());
        }

        // --- candle streaming ---
        let mut candle_guard = candle_engine
            .pool
            .checkout_blocking()
            .expect("candle checkout");
        let mut candle_state = candle_engine.create_state(false);
        let mut candle_text = String::new();
        for chunk in samples.chunks(CHUNK_SAMPLES) {
            let segs = candle_engine
                .process_chunk(chunk, &mut candle_state, &mut candle_guard)
                .expect("candle process_chunk must not error");
            for seg in segs {
                if !candle_text.is_empty() {
                    candle_text.push(' ');
                }
                candle_text.push_str(seg.text.trim());
            }
        }
        if let Some(seg) = candle_engine.finish_stream(&mut candle_state, &mut candle_guard) {
            if !candle_text.is_empty() {
                candle_text.push(' ');
            }
            candle_text.push_str(seg.text.trim());
        }

        eprintln!("fixture = {fixture}");
        eprintln!("ort    (streamed) = {ort_text:?}");
        eprintln!("candle (streamed) = {candle_text:?}");

        assert_eq!(
            ort_text, candle_text,
            "candle streamed transcript diverges from ort for {fixture}:\n  ort    = {ort_text:?}\n  candle = {candle_text:?}"
        );
    }
}

/// Word-level edit distance / WER between a reference and a hypothesis
/// transcript (Levenshtein over whitespace tokens, normalized by reference
/// word count). Used by the ANE measurement harness below.
#[cfg(all(feature = "ane", target_os = "macos"))]
fn word_error_rate(reference: &str, hypothesis: &str) -> f64 {
    let r: Vec<&str> = reference.split_whitespace().collect();
    let h: Vec<&str> = hypothesis.split_whitespace().collect();
    if r.is_empty() {
        return if h.is_empty() { 0.0 } else { 1.0 };
    }
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    let mut cur = vec![0usize; h.len() + 1];
    for (i, rw) in r.iter().enumerate() {
        cur[0] = i + 1;
        for (j, hw) in h.iter().enumerate() {
            let cost = if rw == hw { 0 } else { 1 };
            cur[j + 1] = (prev[j] + cost).min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[h.len()] as f64 / r.len() as f64
}

/// The 15 Golos fixtures with ground-truth references (from
/// `crates/gigastt/tests/fixtures/manifest.json`). `(path, reference)`.
#[cfg(all(feature = "ane", target_os = "macos"))]
fn golos_fixtures() -> Vec<(String, &'static str)> {
    const REFS: &[(&str, &str)] = &[
        (
            "golos_00.wav",
            "шестьдесят тысяч тенге сколько будет стоить",
        ),
        (
            "golos_01.wav",
            "покажи мне на смотрешке телеканал синергия тв",
        ),
        ("golos_02.wav", "заказать яблоки зеленые"),
        (
            "golos_03.wav",
            "алиса закажи килограммовый торт графские развалины",
        ),
        ("golos_04.wav", "ищи телеканал про бизнес на тиви"),
        ("golos_05.wav", "михаила мурадяна"),
        (
            "golos_06.wav",
            "любовницы две тысячи тринадцать пятнадцатый сезон",
        ),
        ("golos_07.wav", "найди боевики"),
        ("golos_08.wav", "гетто сезон три"),
        ("golos_09.wav", "хочу посмотреть ростов папа на телевизоре"),
        ("golos_10.wav", "сбер какое твое самое ненавистное занятие"),
        ("golos_11.wav", "афина чем платят у китайцев"),
        (
            "golos_12.wav",
            "джой как работает досрочное погашение кредита",
        ),
        ("golos_13.wav", "у тебя найдется люк кейдж"),
        ("golos_14.wav", "у тебя будет лучшая часть пинк"),
    ];
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../gigastt/tests/fixtures/");
    REFS.iter()
        .map(|(f, r)| (format!("{dir}{f}"), *r))
        .collect()
}

/// Build the two engines (ort baseline + composite ANE) on the rnnt model in
/// the default model dir. Returns `None` (with a printed SKIP) when the
/// rnnt model or the bucket-768 `.mlpackage` is absent so the `#[ignore]`d
/// measurement tests degrade cleanly on machines without the assets.
#[cfg(all(feature = "ane", target_os = "macos"))]
fn ane_measurement_engines() -> Option<(Engine, Engine)> {
    let model_dir = crate::model::default_model_dir();
    let model_path = Path::new(&model_dir);

    let ane_dir = model_path.join("ane");
    let bucket_768 = ane_dir.join(crate::model::ane_package_dir_name(768));
    if !crate::model::ane_package_complete(&bucket_768) {
        eprintln!(
            "SKIP: ANE bucket-768 package missing in {} (run `gigastt download --ane` or convert locally)",
            ane_dir.display()
        );
        return None;
    }
    if ModelVariant::detect_in_dir(model_path).is_none() {
        eprintln!("SKIP: no model files in {model_dir} (run `gigastt download`)");
        return None;
    }

    let ort_engine = Engine::load_with_factory(
        model_path,
        None,
        1,
        1,
        0,
        Box::new(crate::runtime::ort::factory::OrtFactory::cpu()),
        1,
        true,
        "cpu",
    )
    .expect("ort engine should load");
    let ane_engine = Engine::load_with_factory(
        model_path,
        None,
        1,
        1,
        0,
        Box::new(crate::runtime::coreml::factory::AneFactory::new()),
        1,
        true,
        "ane",
    )
    .expect("ANE engine should load");
    Some((ort_engine, ane_engine))
}

/// Run one encoder pass directly through a checked-out triplet (mirrors
/// [`Engine::run_inference`]'s encoder setup) and return the emitted
/// `encoded_len`. Used to compare the ANE and ort encoders' length tensors
/// for the SAME mel input, pinning [`calc_output_length`] against ONNX drift.
#[cfg(all(feature = "ane", target_os = "macos"))]
fn encoder_emitted_len(engine: &Engine, features: &[f32], num_frames: usize) -> usize {
    let mut guard = engine.pool.checkout_blocking().expect("checkout");
    let triplet = &mut *guard;
    triplet.encoder_inputs[0].resize_to(Shape::new(vec![1, N_MELS, num_frames]));
    triplet.encoder_inputs[0]
        .as_f32_mut()
        .expect("encoder signal tensor is f32")
        .copy_from_slice(features);
    triplet.encoder_inputs[1]
        .as_i64_mut()
        .expect("encoder length tensor is i64")[0] = num_frames as i64;
    let outputs = triplet
        .encoder
        .run(&triplet.encoder_inputs)
        .expect("encoder run");
    match outputs[1].view().data() {
        TensorDataView::I32(v) => usize::try_from(v[0]).expect("non-negative len"),
        TensorDataView::I64(v) => usize::try_from(v[0]).expect("non-negative len"),
        _ => panic!("unexpected encoder length tensor type"),
    }
}

/// FULL-GOLOS WER + frame-count-equality measurement (Part 2a + frame pin).
///
/// For every Golos fixture, transcribes through BOTH the composite ANE engine
/// and the pure-ort baseline, records mel length `T`, the bucket fill % and
/// which PATH the ANE encoder took (ANE bucket vs ort fallback), and emits a
/// per-clip table plus aggregate WER(ANE vs ort), WER(ANE vs truth) and
/// WER(ort vs truth). Additionally asserts the ANE and ort encoders emit the
/// SAME `encoded_len` for the same mel input across all fixtures' real `T`
/// (pins [`calc_output_length`] == the ort ONNX length op against drift).
///
/// ANE parity is SOFT (mask-free FP16 pad-up is not byte-exact: cosine >= 0.94
/// at >= 50% fill, a borderline token can flip), so the per-clip ANE-vs-ort
/// gate is a small WER threshold rather than byte equality.
///
/// Run with:
/// `cargo test -p gigastt-core --features ane --lib -- --ignored --nocapture ane_ort_transcription_parity`
#[cfg(all(feature = "ane", target_os = "macos"))]
#[test]
#[ignore = "requires v3_rnnt model + ~/.gigastt/models/ane/*.mlpackage + ANE hardware"]
fn ane_ort_transcription_parity() {
    let Some((ort_engine, ane_engine)) = ane_measurement_engines() else {
        return;
    };

    // Mirror the encoder-session selection policy (select_bucket over the
    // shipped ladder) so the table can label the bucket the ANE engine took
    // without instrumenting the session itself.
    use crate::model::ANE_BUCKETS;
    use crate::runtime::coreml::encoder_session::select_bucket;
    const FILL_FLOOR: f64 = 0.5;
    // Aggregate gate: the mean WER(ANE vs ort) across all 15 clips must stay
    // small. Per-clip gate: at most ONE word may differ from ort — the
    // documented FP16-pad-up borderline-token flip (see FILL_FLOOR) is allowed
    // on a single word, but a multi-word divergence is a real regression.
    const MAX_MEAN_WER: f64 = 0.05;
    const MAX_WORD_DIFF_PER_CLIP: usize = 1;

    let fixtures = golos_fixtures();
    let mut sum_wer_ane_ort = 0.0;
    let mut sum_wer_ane_truth = 0.0;
    let mut sum_wer_ort_truth = 0.0;
    let mut frame_eq_checked: Vec<usize> = Vec::new();
    // (clip, word-diff vs ort) for the post-table per-clip gate.
    let mut clip_word_diffs: Vec<(String, usize)> = Vec::new();

    eprintln!(
        "\n{:<12} {:>5} {:>6} {:>9} {:>6} texts",
        "clip", "T", "fill%", "path", "ident"
    );
    for (path, reference) in &fixtures {
        let samples = audio::decode_audio_file(path).expect("decode fixture");
        let (features, num_frames) = ane_engine.features.compute(&samples);
        let bucket = select_bucket(num_frames, ANE_BUCKETS, FILL_FLOOR);
        let on_ane = bucket.is_some();
        let fill = match bucket {
            Some(n) => num_frames as f64 / n as f64,
            None => 0.0,
        };
        let path_label = match bucket {
            Some(n) => format!("ANE-{n}"),
            None => "ort-fb".to_string(),
        };

        // Frame-count equality: the ANE and ort encoders must emit the SAME
        // encoded_len for the same mel input. (On the ort-fallback path this
        // is trivially the same session class, but we still assert it; on the
        // ANE path it pins calc_output_length against the ONNX length op.)
        let ane_len = encoder_emitted_len(&ane_engine, &features, num_frames);
        let ort_len = encoder_emitted_len(&ort_engine, &features, num_frames);
        assert_eq!(
            ane_len, ort_len,
            "encoded_len mismatch ANE={ane_len} ort={ort_len} for T={num_frames} ({path})"
        );
        if on_ane {
            let formula = crate::runtime::coreml::encoder_session::calc_output_length(num_frames);
            assert_eq!(
                formula, ort_len,
                "calc_output_length({num_frames})={formula} != ort encoder emitted {ort_len}"
            );
            frame_eq_checked.push(num_frames);
        }

        let mut ort_guard = ort_engine.pool.checkout_blocking().expect("ort checkout");
        let ort_text = ort_engine
            .transcribe_file(path, &mut ort_guard)
            .expect("ort transcription")
            .text;
        drop(ort_guard);

        let mut ane_guard = ane_engine.pool.checkout_blocking().expect("ANE checkout");
        let ane_text = ane_engine
            .transcribe_file(path, &mut ane_guard)
            .expect("ANE transcription")
            .text;
        drop(ane_guard);

        let wer_ane_ort = word_error_rate(&ort_text, &ane_text);
        let wer_ane_truth = word_error_rate(reference, &ane_text);
        let wer_ort_truth = word_error_rate(reference, &ort_text);
        sum_wer_ane_ort += wer_ane_ort;
        sum_wer_ane_truth += wer_ane_truth;
        sum_wer_ort_truth += wer_ort_truth;

        // Absolute word-edit distance vs ort (WER * ort word count, rounded).
        let ort_words = ort_text.split_whitespace().count().max(1);
        let word_diff = (wer_ane_ort * ort_words as f64).round() as usize;

        let clip = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(path);
        clip_word_diffs.push((clip.to_string(), word_diff));
        eprintln!(
            "{:<12} {:>5} {:>5.1}% {:>9} {:>6} ort={:?}",
            clip,
            num_frames,
            fill * 100.0,
            path_label,
            ort_text == ane_text,
            ort_text
        );
        eprintln!(
            "{:<12} {:>5} {:>6} {:>9} {:>6} ane={:?}  truth={:?}",
            "", "", "", "", "", ane_text, reference
        );
        eprintln!(
            "{:<12} WER ANE-vs-ort={wer_ane_ort:.4} (word_diff={word_diff}) ANE-vs-truth={wer_ane_truth:.4} ort-vs-truth={wer_ort_truth:.4}",
            ""
        );
    }

    let n = fixtures.len() as f64;
    let mean_ane_ort = sum_wer_ane_ort / n;
    eprintln!("\n=== AGGREGATE (n={}) ===", fixtures.len());
    eprintln!("mean WER(ANE vs ort)   = {mean_ane_ort:.4}");
    eprintln!("mean WER(ANE vs truth) = {:.4}", sum_wer_ane_truth / n);
    eprintln!("mean WER(ort vs truth) = {:.4}", sum_wer_ort_truth / n);
    eprintln!(
        "frame-count equality (ANE==ort encoded_len & ==calc_output_length) verified for T = {:?}",
        frame_eq_checked
    );

    // Gate AFTER measuring every clip (a measurement harness must not abort
    // mid-run). Aggregate parity must be tight; per clip, at most one word may
    // differ from ort (the documented single FP16-pad-up borderline flip).
    assert!(
        mean_ane_ort <= MAX_MEAN_WER,
        "mean WER(ANE vs ort) {mean_ane_ort:.4} > {MAX_MEAN_WER}"
    );
    for (clip, diff) in &clip_word_diffs {
        assert!(
            *diff <= MAX_WORD_DIFF_PER_CLIP,
            "clip {clip}: {diff} words differ from ort (> {MAX_WORD_DIFF_PER_CLIP}) — multi-word ANE divergence is a regression, not a borderline flip"
        );
    }
}

/// END-TO-END RTFx measurement (Part 2b).
///
/// For the fixtures that take the ANE path (>= 384 mel frames, >= 50% fill),
/// measures FULL-PIPELINE wall time (audio decode -> mel -> encoder -> RNN-T
/// greedy decode -> text) through the ANE engine and the ort baseline, warm
/// (first run discarded), median of >= 5. Reports RTFx (audio_secs / median_s)
/// for each engine and the speedup ratio ANE/ort. This quantifies how little
/// of the encoder-only ~230x ANE speedup survives the CPU-bound RNN-T decode
/// loop: the encoder is nearly free on the ANE, but end-to-end the pipeline is
/// decode-bound, so the realized full-pipeline speedup is only ~3.7x.
///
/// Run with:
/// `cargo test -p gigastt-core --features ane --lib -- --ignored --nocapture ane_e2e_rtfx`
#[cfg(all(feature = "ane", target_os = "macos"))]
#[test]
#[ignore = "requires v3_rnnt model + ~/.gigastt/models/ane/*.mlpackage + ANE hardware"]
fn ane_e2e_rtfx() {
    let Some((ort_engine, ane_engine)) = ane_measurement_engines() else {
        return;
    };

    use crate::model::ANE_BUCKETS;
    use crate::runtime::coreml::encoder_session::select_bucket;
    const FILL_FLOOR: f64 = 0.5;
    const WARM: usize = 1;
    const TIMED: usize = 6;

    fn median_secs(engine: &Engine, path: &str) -> f64 {
        // Warmup (discarded) + timed full-pipeline runs.
        for _ in 0..WARM {
            let mut g = engine.pool.checkout_blocking().expect("checkout");
            let _ = engine.transcribe_file(path, &mut g).expect("transcribe");
        }
        let mut times: Vec<f64> = Vec::with_capacity(TIMED);
        for _ in 0..TIMED {
            let mut g = engine.pool.checkout_blocking().expect("checkout");
            let t = std::time::Instant::now();
            let _ = engine.transcribe_file(path, &mut g).expect("transcribe");
            times.push(t.elapsed().as_secs_f64());
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times[times.len() / 2]
    }

    eprintln!(
        "\n{:<12} {:>5} {:>6} {:>6} {:>9} {:>9} {:>9} {:>9} {:>8}",
        "clip",
        "T",
        "bucket",
        "audio_s",
        "ort_med_s",
        "ane_med_s",
        "ort_RTFx",
        "ane_RTFx",
        "speedup"
    );
    let mut any_ane = false;
    for (path, _ref) in golos_fixtures() {
        let samples = audio::decode_audio_file(&path).expect("decode fixture");
        let (_features, num_frames) = ane_engine.features.compute(&samples);
        let Some(bucket) = select_bucket(num_frames, ANE_BUCKETS, FILL_FLOOR) else {
            continue; // only clips that exercise the ANE encoder path
        };
        any_ane = true;
        let audio_s = samples.len() as f64 / 16000.0;
        let ort_med = median_secs(&ort_engine, &path);
        let ane_med = median_secs(&ane_engine, &path);
        let ort_rtfx = audio_s / ort_med;
        let ane_rtfx = audio_s / ane_med;
        let speedup = ort_med / ane_med;
        let clip = std::path::Path::new(&path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(&path)
            .to_string();
        eprintln!(
            "{:<12} {:>5} {:>6} {:>6.2} {:>9.4} {:>9.4} {:>9.1} {:>9.1} {:>7.2}x",
            clip, num_frames, bucket, audio_s, ort_med, ane_med, ort_rtfx, ane_rtfx, speedup
        );
    }
    assert!(
        any_ane,
        "no Golos fixture took the ANE path (>= 256 mel frames at >= 50% fill); cannot measure e2e RTFx"
    );
}

/// CONCURRENT-PREDICTION test (Part 1 item 2).
///
/// Builds ONE `AneEncoderSession` backed by a single shared `Arc<SharedModel>`
/// and fires concurrent `run` calls from N >= 4 threads on the SAME model,
/// asserting no crash/panic and that every thread gets the SAME output for the
/// same input (deterministic). Exercises the `unsafe impl Send/Sync` under
/// real `CPUAndNeuralEngine` multi-thread load.
///
/// Run with:
/// `cargo test -p gigastt-core --features ane --lib -- --ignored --nocapture ane_concurrent_prediction_deterministic`
#[cfg(all(feature = "ane", target_os = "macos"))]
#[test]
#[ignore = "requires ~/.gigastt/models/ane/gigaam_v3_encoder_768.mlpackage + ANE hardware"]
fn ane_concurrent_prediction_deterministic() {
    use crate::runtime::coreml::bridge;
    use crate::runtime::coreml::encoder_session::{SharedModel, pad_time};
    use std::sync::Arc;

    let model_dir = crate::model::default_model_dir();
    let pkg = Path::new(&model_dir)
        .join("ane")
        .join(crate::model::ane_package_dir_name(768));
    if !crate::model::ane_package_complete(&pkg) {
        eprintln!("SKIP: ANE bucket-768 package missing at {}", pkg.display());
        return;
    }

    // Compile + load ONCE; share across threads (the production sharing model).
    let model = Arc::new(SharedModel(
        bridge::compile_and_load(&pkg, true).expect("compile_and_load bucket-768"),
    ));

    // Deterministic-but-non-trivial mel input padded to the 768 window.
    const T: usize = 600;
    const N: usize = 768;
    let mut mel = vec![0.0f32; N_MELS * T];
    for (i, v) in mel.iter_mut().enumerate() {
        *v = ((i % 97) as f32 * 0.013 - 0.5).sin();
    }
    let padded: Arc<Vec<f32>> = Arc::new(pad_time(&mel, N_MELS, T, N));

    // Single-threaded reference output.
    let (reference, ref_shape) =
        bridge::predict_f32(&model.0, "mel", &padded, &[1, N_MELS, N], "encoded")
            .expect("reference predict");

    const THREADS: usize = 4;
    const PER_THREAD: usize = 5;
    let mut handles = Vec::with_capacity(THREADS);
    for _ in 0..THREADS {
        let model = Arc::clone(&model);
        let padded = Arc::clone(&padded);
        handles.push(std::thread::spawn(move || {
            let mut outs = Vec::with_capacity(PER_THREAD);
            for _ in 0..PER_THREAD {
                let (out, shape) =
                    bridge::predict_f32(&model.0, "mel", &padded, &[1, N_MELS, N], "encoded")
                        .expect("concurrent predict");
                outs.push((out, shape));
            }
            outs
        }));
    }

    let mut total = 0usize;
    for h in handles {
        let outs = h.join().expect("thread did not panic");
        for (out, shape) in outs {
            assert_eq!(shape, ref_shape, "concurrent output shape diverged");
            assert_eq!(
                out.len(),
                reference.len(),
                "concurrent output length diverged"
            );
            assert!(
                out.iter().all(|v| v.is_finite()),
                "concurrent output has non-finite values"
            );
            // Bit-for-bit determinism: same model + same input -> same output.
            assert_eq!(
                out, reference,
                "concurrent prediction diverged from the single-threaded reference"
            );
            total += 1;
        }
    }
    eprintln!(
        "concurrent OK: {THREADS} threads x {PER_THREAD} predicts = {total} runs, all deterministic & finite"
    );
}
