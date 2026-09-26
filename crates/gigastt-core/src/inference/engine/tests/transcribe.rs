use crate::inference::windows::CHUNK_THRESHOLD_SAMPLES;
use crate::test_support;

#[test]
fn test_transcribe_samples_silence_yields_empty_text() {
    // The single-pass file path on pure silence: the encoder runs, decode
    // produces no words, and the result text is empty with a correct
    // duration.
    let (engine, _tmp) = test_support::rnnt_engine();
    let mut guard = engine.pool.checkout_blocking().expect("checkout");
    // A quarter-second still runs the encoder and the empty decode. Two
    // seconds of mel FFT does not finish inside the nightly Miri budget.
    let (samples, secs) = if cfg!(miri) {
        (4_000, 0.25)
    } else {
        (16_000 * 2, 2.0)
    };
    let silence = vec![0.0f32; samples];
    let result = engine
        .transcribe_samples(&silence, &mut guard)
        .expect("silence transcription must not error");
    assert!(result.text.trim().is_empty(), "silence yields no text");
    assert!(result.words.is_empty());
    assert!((result.duration_s - secs).abs() < 1e-6);
}

#[test]
fn test_transcribe_samples_short_sub_frame_audio() {
    // Audio shorter than a single FFT frame: the mel extractor pads to one
    // zero frame and the encoder still runs — exercising the short-input
    // single-pass branch without panicking. The decode output is whatever
    // the model emits on a lone padded frame; we only assert it doesn't
    // error and the reported duration matches the (tiny) input.
    let (engine, _tmp) = test_support::rnnt_engine();
    let mut guard = engine.pool.checkout_blocking().expect("checkout");
    let tiny = vec![0.0f32; 100]; // < N_FFT (320)
    let result = engine
        .transcribe_samples(&tiny, &mut guard)
        .expect("sub-frame audio must not error");
    assert!((result.duration_s - 100.0 / 16000.0).abs() < 1e-9);
}

#[test]
#[cfg_attr(
    miri,
    ignore = "single-pass decode of the 30 s chunk threshold does not finish under Miri"
)]
fn test_transcribe_samples_below_chunk_threshold_single_pass() {
    // Just under the long-form chunk threshold (30s) takes the single-pass
    // path; pure silence still yields no words but must not error.
    let (engine, _tmp) = test_support::rnnt_engine();
    let mut guard = engine.pool.checkout_blocking().expect("checkout");
    let samples = vec![0.0f32; CHUNK_THRESHOLD_SAMPLES]; // exactly at threshold → single pass
    let result = engine
        .transcribe_samples(&samples, &mut guard)
        .expect("at-threshold audio must not error");
    assert!(result.words.is_empty(), "silence decodes to no words");
}
