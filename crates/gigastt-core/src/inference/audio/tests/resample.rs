use super::*;
use rubato::Resampler;

// --- resample tests ---

#[test]
#[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
fn test_narrowband_resampling_matches_author_pcm16_precision() {
    // GigaAM's file loader emits s16le after resampling. Retaining sub-LSB
    // float precision suppresses out-of-band energy and can make multilingual
    // INT8 CTC return blanks on otherwise intelligible telephone speech.
    let input: Vec<f32> = (0..8000).map(|i| 0.4 * (i as f32 * 0.317).sin()).collect();
    let whole = resample(&input, SampleRate(8000), SampleRate(16000)).unwrap();
    let mut cache = None;
    let mut streamed = Vec::new();
    let mut chunk = Vec::new();
    for part in input.chunks(997) {
        resample_with_cache(
            part.to_vec(),
            SampleRate(8000),
            SampleRate(16000),
            &mut cache,
            &mut chunk,
        )
        .unwrap();
        streamed.extend_from_slice(&chunk);
    }
    assert_eq!(
        whole, streamed,
        "PCM conversion must not depend on chunk boundaries"
    );
    assert!(whole.iter().any(|s| s.abs() > 0.3));
    for sample in whole {
        let pcm = sample * 32768.0;
        assert_eq!(
            pcm,
            pcm.round(),
            "narrowband output must have PCM16 precision"
        );
        assert!((-32768.0..=32767.0).contains(&pcm));
    }
}

#[test]
#[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
fn test_narrowband_pcm16_preserves_silence_and_saturates_overload() {
    let silence = resample(&vec![0.0; 800], SampleRate(8000), SampleRate(16000)).unwrap();
    assert!(
        silence.iter().all(|&s| s == 0.0),
        "no artificial dither on silence"
    );
    for (input, expected) in [(2.0, 32767.0 / 32768.0), (-2.0, -1.0)] {
        let output = resample(&vec![input; 8000], SampleRate(8000), SampleRate(16000)).unwrap();
        assert_eq!(
            output[output.len() / 2],
            expected,
            "saturate without wrapping"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
fn test_resample_downsample_length() {
    let input: Vec<f32> = (0..4800).map(|i| (i as f32).sin()).collect();
    let output = resample(&input, SampleRate(48000), SampleRate(16000)).unwrap();
    // Rubato FIR filter has sinc_len/2 delay; output is shorter than ideal ratio.
    // For 4800 samples at 3:1 ratio, expect ~1556 (not exact 1600).
    assert!(!output.is_empty());
    assert!(
        output.len() > 1400 && output.len() < 1700,
        "Unexpected output length: {}",
        output.len()
    );
}

#[test]
#[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
fn test_resample_upsample_length() {
    let input: Vec<f32> = (0..800).map(|i| (i as f32).sin()).collect();
    let output = resample(&input, SampleRate(8000), SampleRate(16000)).unwrap();
    // Rubato FIR delay reduces output; expect ~1340 (not exact 1600).
    assert!(!output.is_empty());
    assert!(
        output.len() > 1200 && output.len() < 1700,
        "Unexpected output length: {}",
        output.len()
    );
}

#[test]
#[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
fn test_resample_preserves_dc() {
    // Constant signal should remain approximately constant after resampling.
    // Rubato FIR filter may cause transients at edges; check the middle 80%.
    let input = vec![0.5_f32; 4800];
    let output = resample(&input, SampleRate(48000), SampleRate(16000)).unwrap();
    let start = output.len() / 10;
    let end = output.len() - start;
    for &sample in &output[start..end] {
        assert!(
            (sample - 0.5).abs() < 0.05,
            "DC signal not preserved: {sample}"
        );
    }
}

#[test]
fn test_resample_empty() {
    let output = resample(&[], SampleRate(48000), SampleRate(16000)).unwrap();
    assert!(output.is_empty());
}

#[test]
fn test_resample_zero_rate_returns_empty() {
    let input = vec![1.0, 2.0, 3.0];
    assert!(
        resample(&input, SampleRate(0), SampleRate(16000))
            .unwrap()
            .is_empty()
    );
    assert!(
        resample(&input, SampleRate(16000), SampleRate(0))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn test_resample_same_rate() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let output = resample(&input, SampleRate(16000), SampleRate(16000)).unwrap();
    assert_eq!(output.len(), input.len());
    for (a, b) in input.iter().zip(output.iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}

// --- stress tests: robustness edge cases ---

#[test]
#[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
fn test_resample_nan_input() {
    let input = vec![f32::NAN; 1000];
    let output = resample(&input, SampleRate(48000), SampleRate(16000)).unwrap();
    // NaN should be replaced with zeros
    assert!(!output.is_empty());
    for &s in &output {
        assert!(s.is_finite(), "NaN should be sanitized to zero, got {s}");
    }
}

#[test]
#[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
fn test_resample_infinity_input() {
    let input = vec![f32::INFINITY; 500];
    let output = resample(&input, SampleRate(48000), SampleRate(16000)).unwrap();
    assert!(!output.is_empty());
    for &s in &output {
        assert!(
            s.is_finite(),
            "Infinity should be sanitized to zero, got {s}"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
fn test_resample_mixed_nan_normal() {
    let mut input = vec![0.5_f32; 480];
    input[100] = f32::NAN;
    input[200] = f32::NEG_INFINITY;
    let output = resample(&input, SampleRate(48000), SampleRate(16000)).unwrap();
    assert!(!output.is_empty());
    for &s in &output {
        assert!(s.is_finite(), "Non-finite values should be sanitized");
    }
}

// --- SampleRate tests ---

#[test]
fn test_sample_rate_new_zero_errors() {
    let result = SampleRate::new(0);
    assert!(result.is_err(), "zero sample rate must error");
}

#[test]
fn test_sample_rate_new_positive_ok() {
    let sr = SampleRate::new(16000).unwrap();
    assert_eq!(sr.get(), 16000);
    assert_eq!(sr.0, 16000);
}

// --- resample_with_cache tests ---

#[test]
fn test_resample_with_cache_empty_clears_buffer() {
    let mut cache: Option<rubato::Async<f32>> = None;
    let mut out = vec![1.0, 2.0, 3.0];
    resample_with_cache(
        Vec::new(),
        SampleRate(48000),
        SampleRate(16000),
        &mut cache,
        &mut out,
    )
    .unwrap();
    assert!(out.is_empty(), "empty input must clear the output buffer");
    assert!(cache.is_none(), "no resampler created for empty input");
}

#[test]
fn test_resample_with_cache_zero_rate_clears_buffer() {
    let mut cache: Option<rubato::Async<f32>> = None;
    let mut out = vec![9.0];
    resample_with_cache(
        vec![1.0, 2.0],
        SampleRate(0),
        SampleRate(16000),
        &mut cache,
        &mut out,
    )
    .unwrap();
    assert!(out.is_empty());
    let mut out2 = vec![9.0];
    resample_with_cache(
        vec![1.0, 2.0],
        SampleRate(16000),
        SampleRate(0),
        &mut cache,
        &mut out2,
    )
    .unwrap();
    assert!(out2.is_empty());
}

#[test]
fn test_resample_with_cache_same_rate_passthrough() {
    let mut cache: Option<rubato::Async<f32>> = None;
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let mut out = Vec::new();
    resample_with_cache(
        input.clone(),
        SampleRate(16000),
        SampleRate(16000),
        &mut cache,
        &mut out,
    )
    .unwrap();
    assert_eq!(out, input, "same rate must pass through unchanged");
    assert!(
        cache.is_none(),
        "no resampler created for same-rate passthrough"
    );
}

#[test]
#[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
fn test_resample_with_cache_sanitizes_non_finite() {
    let mut cache: Option<rubato::Async<f32>> = None;
    let mut input = vec![0.5_f32; 480];
    input[10] = f32::NAN;
    input[20] = f32::INFINITY;
    input[30] = f32::NEG_INFINITY;
    let mut out = Vec::new();
    resample_with_cache(
        input,
        SampleRate(48000),
        SampleRate(16000),
        &mut cache,
        &mut out,
    )
    .unwrap();
    assert!(!out.is_empty());
    assert!(
        cache.is_some(),
        "resampler should be cached after first use"
    );
    for &s in &out {
        assert!(
            s.is_finite(),
            "non-finite values must be sanitized, got {s}"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
fn test_resample_with_cache_growing_chunks_match_one_shot() {
    use std::f32::consts::PI;

    // 1 s of a continuous two-tone signal at 48 kHz, continuous across the
    // whole stream so any seam glitch shows up against the reference.
    let n = 48_000usize;
    let signal: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / 48_000.0;
            0.5 * (2.0 * PI * 440.0 * t).sin() + 0.3 * (2.0 * PI * 1_200.0 * t).sin()
        })
        .collect();

    // Reference: one-shot resample of the whole signal in a single call.
    let reference = resample(&signal, SampleRate(48_000), SampleRate(16_000)).unwrap();

    // Stream the same signal in strictly growing frames (10 ms @ 48 kHz,
    // +10 ms per frame). Every growth step used to recreate the resampler,
    // resetting its FIR history and fractional phase at each seam.
    let mut cache: Option<rubato::Async<f32>> = None;
    let mut out = Vec::new();
    let mut streamed = Vec::new();
    let mut pos = 0usize;
    let mut chunk = 480usize;
    while pos < signal.len() {
        let end = (pos + chunk).min(signal.len());
        resample_with_cache(
            signal[pos..end].to_vec(),
            SampleRate(48_000),
            SampleRate(16_000),
            &mut cache,
            &mut out,
        )
        .unwrap();
        streamed.extend_from_slice(&out);
        pos = end;
        chunk += 480;
    }
    assert!(streamed.iter().all(|s| s.is_finite()));

    // A recreated resampler drops the output-delay tail (~85 samples at
    // 3:1) per recreation, so the streamed length collapses vs one-shot.
    let len_diff = reference.len().abs_diff(streamed.len());
    assert!(
        len_diff <= 2,
        "chunked stream diverged from one-shot reference: {} vs {} samples",
        streamed.len(),
        reference.len()
    );

    // Beyond the initial sinc transient (~sinc_len/2 * 1/3 ≈ 43 samples)
    // the streamed output must track the one-shot reference closely; a
    // seam discontinuity (FIR reset fade-in) shows up as a large spike.
    let skip = 128;
    let cmp_len = reference.len().min(streamed.len());
    assert!(cmp_len > skip + 1_000, "not enough overlap to compare");
    let mut max_diff = 0.0f32;
    let mut max_at = 0usize;
    for i in skip..cmp_len {
        let d = (reference[i] - streamed[i]).abs();
        if d > max_diff {
            max_diff = d;
            max_at = i;
        }
    }
    assert!(
        max_diff < 1e-3,
        "seam discontinuity: max |streamed - reference| = {max_diff} at sample {max_at}"
    );
}

#[test]
#[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
fn test_resample_with_cache_growth_keeps_instance() {
    let mut cache: Option<rubato::Async<f32>> = None;
    let mut out = Vec::new();
    let feed = |cache: &mut Option<rubato::Async<f32>>, out: &mut Vec<f32>, n: usize, seed: f32| {
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * seed).sin()).collect();
        resample_with_cache(input, SampleRate(48_000), SampleRate(16_000), cache, out).unwrap();
    };

    // First frame fixes the resampler capacity.
    feed(&mut cache, &mut out, 480, 0.01);
    let capacity = cache.as_ref().unwrap().input_frames_max();
    assert!(capacity >= 480);

    // Growing frames must NOT change the capacity: a change means the
    // resampler was recreated and its FIR state was lost.
    feed(&mut cache, &mut out, 960, 0.02);
    assert_eq!(
        cache.as_ref().unwrap().input_frames_max(),
        capacity,
        "resampler recreated on frame growth"
    );
    feed(&mut cache, &mut out, 2_000, 0.03);
    assert_eq!(cache.as_ref().unwrap().input_frames_max(), capacity);

    // A frame larger than the initial capacity must also survive without
    // recreation (fed through in capacity-sized pieces).
    feed(&mut cache, &mut out, capacity + 1_001, 0.01);
    assert_eq!(
        cache.as_ref().unwrap().input_frames_max(),
        capacity,
        "oversized frame must be split, not trigger recreation"
    );
    assert!(out.iter().all(|s| s.is_finite()));

    // A frame one sample over capacity splits into a full piece plus a
    // 1-sample remainder (which defers its output via the fractional
    // phase); this must succeed and keep the instance.
    feed(&mut cache, &mut out, capacity + 1, 0.02);
    assert_eq!(cache.as_ref().unwrap().input_frames_max(), capacity);
    assert!(out.iter().all(|s| s.is_finite()));
}

#[test]
#[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
fn test_resample_with_cache_reuses_across_chunk_sizes() {
    let mut cache: Option<rubato::Async<f32>> = None;
    let mut out = Vec::new();
    // First call creates the resampler.
    let input1: Vec<f32> = (0..480).map(|i| (i as f32 * 0.01).sin()).collect();
    resample_with_cache(
        input1,
        SampleRate(48000),
        SampleRate(16000),
        &mut cache,
        &mut out,
    )
    .unwrap();
    assert!(cache.is_some());
    let len_first = out.len();
    assert!(len_first > 0);

    // Second call with the SAME chunk size reuses the cached resampler.
    let input2: Vec<f32> = (0..480).map(|i| (i as f32 * 0.02).cos()).collect();
    resample_with_cache(
        input2,
        SampleRate(48000),
        SampleRate(16000),
        &mut cache,
        &mut out,
    )
    .unwrap();
    assert!(cache.is_some());
    assert!(!out.is_empty());

    // Third call with a DIFFERENT chunk size resizes in place — the
    // resampler is never recreated, so its FIR state survives.
    let input3: Vec<f32> = (0..960).map(|i| (i as f32 * 0.01).sin()).collect();
    resample_with_cache(
        input3,
        SampleRate(48000),
        SampleRate(16000),
        &mut cache,
        &mut out,
    )
    .unwrap();
    assert!(cache.is_some());
    assert!(!out.is_empty());
    for &s in &out {
        assert!(s.is_finite());
    }
}
