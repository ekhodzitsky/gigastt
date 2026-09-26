use super::*;
use bytes::Bytes;

// --- streaming resample equivalence (whole-buffer reference) ---

/// PCM16 samples of a committed 16 kHz mono WAV fixture, used as a real-signal
/// input for the streaming-resample equivalence tests.
fn fixture_tone_pcm() -> Vec<i16> {
    let wav = include_bytes!("../../../../tests/fixtures/telephony/tone_src.wav");
    let data = crate::inference::audio::telephony::find_riff_chunk(wav, b"data")
        .expect("fixture data chunk");
    data.as_chunks::<2>()
        .0
        .iter()
        .map(|b| i16::from_le_bytes(*b))
        .collect()
}

/// Linear sine sweep, PCM16. Sweeping across the whole band is the adversarial
/// case for a resampler seam: any FIR-history reset shows up as a spike.
fn sweep_pcm(rate: u32, seconds: f32) -> Vec<i16> {
    let n = (rate as f32 * seconds) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / rate as f32;
            let f = 50.0 + (0.45 * rate as f32 - 50.0) * (t / seconds);
            (0.8 * (std::f32::consts::PI * f * t).sin() * 32000.0) as i16
        })
        .collect()
}

/// Pure tone, PCM16, with the phase accumulated in f64 so the *input* carries
/// no drift of its own. Where `sweep_pcm` probes chunk seams, a single tone
/// probes long-run phase accumulation: it has an analytic ground truth, so each
/// path can be scored against the truth instead of only against the other one.
/// `freq` is chosen to complete a whole number of cycles per 16 kHz analysis
/// window, which makes the phase estimate below leakage-free.
fn tone_pcm(rate: u32, seconds: f64, freq: f64) -> Vec<i16> {
    let n = (f64::from(rate) * seconds) as usize;
    (0..n)
        .map(|i| {
            let t = i as f64 / f64::from(rate);
            (0.8 * (std::f64::consts::TAU * freq * t).sin() * 32000.0) as i16
        })
        .collect()
}

/// Largest deviation, across one-second windows, of the measured phase of
/// `freq` from the phase of the first window. A resampler whose fractional read
/// position accumulates error stretches time slightly, which shows up here as a
/// phase that walks away from where it started.
fn max_phase_drift_16k(samples: &[f32], freq: f64) -> f64 {
    const WINDOW: usize = 16_000;
    let phase_of = |start: usize| {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (i, &v) in samples[start..start + WINDOW].iter().enumerate() {
            let w = std::f64::consts::TAU * freq * ((start + i) as f64 / 16_000.0);
            re += f64::from(v) * w.cos();
            im += f64::from(v) * w.sin();
        }
        im.atan2(re)
    };
    let first = phase_of(0);
    let mut worst = 0.0f64;
    for w in 0..samples.len() / WINDOW {
        let mut d = phase_of(w * WINDOW) - first;
        // Wrap into (-pi, pi] so a drift that crosses a cycle stays comparable.
        d -= std::f64::consts::TAU * (d / std::f64::consts::TAU).round();
        worst = worst.max(d.abs());
    }
    worst
}

/// Signal-to-error ratio, in dB, of `candidate` measured against `reference`.
fn signal_to_error_db(reference: &[f32], candidate: &[f32]) -> f64 {
    let mut err = 0.0f64;
    let mut sig = 0.0f64;
    for (&r, &c) in reference.iter().zip(candidate) {
        let d = f64::from(r) - f64::from(c);
        err += d * d;
        sig += f64::from(r) * f64::from(r);
    }
    if err == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (sig / err).log10()
}

/// Assert the streaming path agrees with a whole-buffer `resample()` of the
/// same decoded signal: same length within one sample, max per-sample delta
/// below 1e-4.
fn assert_matches_whole_buffer(streamed: &[f32], reference: &[f32], what: &str) {
    let len_diff = streamed.len().abs_diff(reference.len());
    assert!(
        len_diff <= 1,
        "{what}: length diverged, streaming {} vs whole-buffer {}",
        streamed.len(),
        reference.len()
    );
    let cmp = streamed.len().min(reference.len());
    assert!(cmp > 0, "{what}: nothing to compare");
    let mut max_diff = 0.0f32;
    let mut max_at = 0usize;
    for i in 0..cmp {
        let d = (streamed[i] - reference[i]).abs();
        if d > max_diff {
            max_diff = d;
            max_at = i;
        }
    }
    assert!(
        max_diff <= 1e-4,
        "{what}: max |streaming - whole-buffer| = {max_diff} at sample {max_at}"
    );
}

/// Decoding the same PCM twice — once with a 16 kHz header (passthrough, so
/// the result is exactly what the decoder produced) and once with a `rate`
/// header (the streaming resample path) — must agree with running the
/// passthrough result through the whole-buffer `resample()`.
fn check_mono_equivalence(pcm: &[i16], rate: u32, what: &str) {
    let at_source = decode_audio_bytes(&make_wav_bytes(pcm, 16000)).unwrap();
    let reference = resample(&at_source, SampleRate(rate), SampleRate(16000)).unwrap();
    let streamed = decode_audio_bytes(&make_wav_bytes(pcm, rate)).unwrap();
    assert_matches_whole_buffer(&streamed, &reference, what);
}

fn check_stereo_equivalence(left: &[i16], right: &[i16], rate: u32, what: &str) {
    // Mono-mix path.
    let mixed_at_source = decode_audio_bytes(&make_stereo_wav_bytes(left, right, 16000)).unwrap();
    let mixed_reference = resample(&mixed_at_source, SampleRate(rate), SampleRate(16000)).unwrap();
    let mixed_streamed = decode_audio_bytes(&make_stereo_wav_bytes(left, right, rate)).unwrap();
    assert_matches_whole_buffer(&mixed_streamed, &mixed_reference, &format!("{what} mixed"));

    // Split-channel path.
    let split_at_source =
        decode_audio_bytes_shared_channels(Bytes::from(make_stereo_wav_bytes(left, right, 16000)))
            .unwrap();
    let split_streamed =
        decode_audio_bytes_shared_channels(Bytes::from(make_stereo_wav_bytes(left, right, rate)))
            .unwrap();
    assert_eq!(split_streamed.len(), split_at_source.len());
    for (c, (streamed, source)) in split_streamed.iter().zip(&split_at_source).enumerate() {
        let reference = resample(source, SampleRate(rate), SampleRate(16000)).unwrap();
        assert_matches_whole_buffer(streamed, &reference, &format!("{what} channel {c}"));
    }
}

#[test]
#[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
fn test_streaming_decode_matches_whole_buffer_resample_48k() {
    let pcm = sweep_pcm(48_000, 2.5);
    check_mono_equivalence(&pcm, 48_000, "48k sweep mono");
}

#[test]
#[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
fn test_streaming_decode_matches_whole_buffer_resample_44k1() {
    let pcm = sweep_pcm(44_100, 2.5);
    check_mono_equivalence(&pcm, 44_100, "44.1k sweep mono");
}

#[test]
#[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
fn test_streaming_decode_matches_whole_buffer_resample_stereo_48k() {
    let left = sweep_pcm(48_000, 2.5);
    let right: Vec<i16> = left.iter().rev().copied().collect();
    check_stereo_equivalence(&left, &right, 48_000, "48k sweep stereo");
}

#[test]
#[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
fn test_streaming_decode_matches_whole_buffer_resample_stereo_44k1() {
    let left = sweep_pcm(44_100, 2.5);
    let right: Vec<i16> = left.iter().rev().copied().collect();
    check_stereo_equivalence(&left, &right, 44_100, "44.1k sweep stereo");
}

/// Long-input gate for the staged resample at a NON-INTEGER ratio.
///
/// The fixtures above hold both paths to 1e-4 per sample, but they are 2.5 s
/// long and that tolerance is only reachable at that scale. rubato carries its
/// fractional read position in a single f64 that runs monotonically for the
/// whole of one `process` call (`idx += 1/ratio` per output sample) and takes
/// the sub-sample offset as `idx * 256 - floor(idx * 256)`, so the resolution
/// of that offset halves every time `idx` doubles. One whole-buffer call over
/// 300 s of 44.1 kHz audio drives `idx` to 13.2 million, where the offset is
/// quantised to ~1e-6; the staged path restarts `idx` near zero on every flush
/// and holds ~1e-9. The two therefore separate as the input grows, and past
/// ~300 s the gap is over the 1e-4 per-sample bound the short fixtures use —
/// this case measures 1.2e-4, which is why it is scored on the error-to-signal
/// ratio instead. Sweeping the duration over the same comparison gives max
/// per-sample deltas of 6.7e-7 at 30 s, 1.2e-5 at 120 s, 1.3e-4 at 300 s and
/// 4.1e-4 at 600 s.
///
/// What separates is the *reference*, not the path under test. Against the
/// analytic tone the staged path's phase is flat with duration (9.2e-5 rad at
/// 30 s, 9.3e-5 rad at 600 s) while the whole-buffer path's walks (9.3e-5 rad
/// at 30 s, 4.2e-4 rad at 600 s); this case measures 8.4e-5 rad staged against
/// 2.4e-4 rad whole-buffer. Asserting the ordering pins that direction: a plain
/// delta cannot say which side moved, but this fails if the staged path ever
/// becomes the one that drifts. Integer ratios are exempt from all of it —
/// `1/ratio` is exact in f64 at 48/32/8 kHz, so those stay bit-identical at any
/// length and keep the strict per-sample gate.
///
/// Costs ~50 s in a debug build, which is why one duration is covered rather
/// than a sweep; 300 s is the shortest that reaches the regime. That cost is
/// also why it is `#[ignore]`d: PR CI runs `cargo test --workspace --lib` and
/// stays fast, while the main-push lane runs this by name.
#[test]
#[ignore = "~50 s in debug; long-duration numeric gate, run on main push"]
#[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
fn test_streaming_decode_long_44k1_input_holds_phase_better_than_whole_buffer() {
    // A whole number of cycles per one-second analysis window, so the phase
    // estimate sees no spectral leakage.
    const FREQ: f64 = 1_000.0;
    let pcm = tone_pcm(44_100, 300.0, FREQ);

    let at_source = decode_audio_bytes(&make_wav_bytes(&pcm, 16000)).unwrap();
    let reference = resample(&at_source, SampleRate(44_100), SampleRate(16000)).unwrap();
    let streamed = decode_audio_bytes(&make_wav_bytes(&pcm, 44_100)).unwrap();

    // Length stays exact: the divergence is sub-sample phase, never a dropped
    // or duplicated frame at a flush boundary.
    assert_eq!(
        streamed.len(),
        reference.len(),
        "long 44.1k input: length diverged"
    );

    // Error-to-signal floor for non-integer ratios at this length, in place of
    // the per-sample tolerance the short fixtures use.
    let snr = signal_to_error_db(&reference, &streamed);
    assert!(
        snr >= 70.0,
        "long 44.1k input: streaming vs whole-buffer SNR {snr:.1} dB below the 70 dB floor"
    );

    let streamed_drift = max_phase_drift_16k(&streamed, FREQ);
    let reference_drift = max_phase_drift_16k(&reference, FREQ);
    assert!(
        streamed_drift <= reference_drift,
        "long 44.1k input: staged path drifted {streamed_drift:.3e} rad, \
         more than the whole-buffer reference's {reference_drift:.3e} rad"
    );
    // Absolute floor as well, so the ordering assertion cannot be satisfied by
    // both paths degrading together.
    assert!(
        streamed_drift <= 2e-4,
        "long 44.1k input: staged path phase drift {streamed_drift:.3e} rad exceeds 2e-4"
    );
}

#[test]
#[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
fn test_streaming_decode_matches_whole_buffer_resample_fixture() {
    // The fixture is exactly one staging chunk long, so on its own it drains
    // once and never crosses a chunk boundary — the seam this test exists to
    // guard would go unobserved. Doubling it puts a full flush on each side of
    // a boundary plus a short tail, so a resampler that dropped its FIR history
    // between flushes shows up here.
    let pcm = [fixture_tone_pcm(), fixture_tone_pcm()].concat();
    assert!(
        pcm.len() > crate::inference::audio::resample::RESAMPLE_STAGING_FRAMES,
        "fixture must span more than one staging flush, got {} samples",
        pcm.len()
    );
    check_mono_equivalence(&pcm, 48_000, "fixture 48k");
    check_mono_equivalence(&pcm, 44_100, "fixture 44.1k");
}

#[test]
fn test_streaming_decode_16k_input_is_bit_identical() {
    // A 16 kHz source must never reach the resampler: every sample stays the
    // raw PCM16 conversion and the frame count is preserved exactly.
    // The fixture is a few seconds; Miri checks the same identity on a short
    // tone so the nightly job still finishes.
    let pcm = if cfg!(miri) {
        (0..2_000)
            .map(|i| ((i as f32 * 0.02).sin() * 8000.0) as i16)
            .collect()
    } else {
        fixture_tone_pcm()
    };
    let mono = decode_audio_bytes(&make_wav_bytes(&pcm, 16000)).unwrap();
    assert_eq!(mono.len(), pcm.len());
    for (i, (&raw, &got)) in pcm.iter().zip(&mono).enumerate() {
        let expected = f32::from(raw) / 32768.0;
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "sample {i} was filtered: {got} vs {expected}"
        );
    }

    let right: Vec<i16> = pcm.iter().rev().copied().collect();
    let channels =
        decode_audio_bytes_shared_channels(Bytes::from(make_stereo_wav_bytes(&pcm, &right, 16000)))
            .unwrap();
    assert_eq!(channels.len(), 2);
    for (c, raw) in [&pcm, &right].iter().enumerate() {
        assert_eq!(channels[c].len(), raw.len());
        for (i, (&r, &got)) in raw.iter().zip(&channels[c]).enumerate() {
            let expected = f32::from(r) / 32768.0;
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "channel {c} sample {i} was filtered: {got} vs {expected}"
            );
        }
    }
}

#[test]
#[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
fn test_telephony_raw_streaming_matches_whole_buffer_resample() {
    // 8 kHz G.711 upsamples to 16 kHz through the same staged path. Long
    // enough to fill the staging buffer more than once — at 2.5 s the whole
    // clip fits in a single flush and no chunk boundary is ever crossed.
    let pcm = sweep_pcm(8_000, 12.5);
    assert!(
        pcm.len() > crate::inference::audio::resample::RESAMPLE_STAGING_FRAMES,
        "clip must span more than one staging flush, got {} samples",
        pcm.len()
    );
    let mut encoder = audio_codec::pcmu::PcmuEncoder::new();
    let encoded = encode_telephony(&mut encoder, &pcm);
    let mut decoder = audio_codec::pcmu::PcmuDecoder::new();
    let round_tripped = decode_telephony_pcm(&mut decoder, &encoded);
    let at_source: Vec<f32> = round_tripped
        .iter()
        .map(|&s| f32::from(s) / 32768.0)
        .collect();
    let reference = resample(&at_source, SampleRate(8_000), SampleRate(16_000)).unwrap();
    let streamed = decode_telephony_raw(&encoded, TelephonyCodec::Pcmu, 8_000).unwrap();
    assert_matches_whole_buffer(&streamed, &reference, "pcmu 8k");
}

// --- AudioChunks (fixed-size streaming decode) ---

/// The chunk sequence must be exactly `slice.chunks(n)` over the flat decode —
/// same samples, same boundaries — because the streaming recognizer's state
/// depends on the chunk cadence, so any drift here would change a transcript.
#[test]
fn test_audio_chunks_match_flat_decode_chunked() {
    // 48k–120k samples of encode-plus-decode do not finish under Miri.
    // The short lengths still cross a chunk boundary.
    let lengths: &[usize] = if cfg!(miri) {
        &[1, 999, 2_000]
    } else {
        &[1, 999, 16_000, 16_001, 48_000, 120_000]
    };
    for &n in lengths {
        for &chunk in &[16_000usize, 640, 7_000] {
            let src: Vec<f32> = (0..n)
                .map(|i| 0.4 * ((i as f32) * 0.017).sin() + 0.2 * ((i as f32) * 0.0031).sin())
                .collect();
            let wav = bytes::Bytes::from(encode_wav_pcm16(&src, 16000));
            let flat = decode_audio_bytes(&wav).expect("flat decode");

            let mut chunks = AudioChunks::from_bytes(wav, chunk, None).expect("open chunks");
            let mut got: Vec<Vec<f32>> = Vec::new();
            while let Some(c) = chunks.next_chunk().expect("chunk") {
                got.push(c.to_vec());
            }
            let want: Vec<Vec<f32>> = flat.chunks(chunk).map(<[f32]>::to_vec).collect();
            assert_eq!(got, want, "n={n} chunk={chunk}");
            assert_eq!(
                chunks.total_16k_samples(),
                flat.len(),
                "n={n} chunk={chunk}"
            );
        }
    }
}

/// An operator length limit still trips, and as the typed `AudioTooLong` so the
/// HTTP layer can answer 413 rather than a generic decode failure.
#[test]
fn test_audio_chunks_honour_max_audio_secs() {
    // Five seconds natively. One second still exceeds the cap below, and it
    // finishes under Miri.
    let secs = if cfg!(miri) { 1 } else { 5 };
    let src = vec![0.1f32; 16_000 * secs];
    let wav = bytes::Bytes::from(encode_wav_pcm16(&src, 16000));
    // Unbounded: the whole clip streams.
    let mut ok = AudioChunks::from_bytes(wav.clone(), 16_000, None).expect("open");
    let mut total = 0;
    while let Some(c) = ok.next_chunk().expect("chunk") {
        total += c.len();
    }
    assert_eq!(total, src.len());

    // Bounded below the clip length: the budget trips during the pull.
    let cap = if cfg!(miri) { 0.25 } else { 1.0 };
    let mut capped = AudioChunks::from_bytes(wav, 16_000, Some(cap)).expect("open");
    let err = loop {
        match capped.next_chunk() {
            Ok(Some(_)) => continue,
            Ok(None) => panic!("clip must not drain under a limit shorter than itself"),
            Err(e) => break e,
        }
    };
    assert!(
        matches!(
            err.downcast_ref::<crate::error::GigasttError>(),
            Some(crate::error::GigasttError::AudioTooLong { .. })
        ),
        "expected a typed AudioTooLong, got: {err:#}"
    );
}

// --- Streaming dual-mono detection ---

/// The streaming detector must agree with the batch pair
/// (`normalized_correlation` behind `is_dual_mono`) on both the value and, more
/// importantly, the *verdict*: it decides whether `channels=split` transcribes
/// two speakers or falls back to a mono mix, so a flip would change output.
#[test]
fn test_dual_mono_detector_matches_batch_correlation() {
    // Deterministic, no rand: a base signal plus independent-ish perturbations.
    // 40_000 samples with one-sample pushes is the native recurrence. Miri
    // checks the same recurrence on a shorter buffer.
    let n = if cfg!(miri) { 4_000 } else { 40_000 };
    let base: Vec<f32> = (0..n)
        .map(|i| 0.5 * ((i as f32) * 0.013).sin() + 0.2 * ((i as f32) * 0.0007).cos())
        .collect();
    let other: Vec<f32> = (0..n)
        .map(|i| 0.5 * ((i as f32) * 0.031 + 1.7).sin() - 0.3 * ((i as f32) * 0.0021).cos())
        .collect();

    let cases: Vec<(&str, Vec<f32>, Vec<f32>)> = vec![
        // Identical: the PBX-recorded-the-mix case the check exists for.
        ("identical", base.clone(), base.clone()),
        // Same content, slightly attenuated — still dual-mono.
        (
            "attenuated",
            base.clone(),
            base.iter().map(|v| v * 0.85).collect(),
        ),
        // Same content plus a small amount of the other — near the threshold.
        (
            "mostly-same",
            base.clone(),
            base.iter()
                .zip(&other)
                .map(|(l, r)| l * 0.9 + r * 0.1)
                .collect(),
        ),
        // Genuine stereo: two different sources.
        ("independent", base.clone(), other.clone()),
        // Phase-inverted: strongly anti-correlated, must not read as dual-mono.
        ("inverted", base.clone(), base.iter().map(|v| -v).collect()),
        // One channel silent.
        ("right-silent", base.clone(), vec![0.0; n]),
        // Both silent.
        ("both-silent", vec![0.0; n], vec![0.0; n]),
        // DC offset on both: the case naive power sums would lose.
        (
            "dc-offset",
            base.iter().map(|v| v + 0.7).collect(),
            base.iter().map(|v| v + 0.7).collect(),
        ),
    ];

    for (label, left, right) in cases {
        let batch = crate::inference::audio::decode::normalized_correlation_for_test(&left, &right);
        // Feed in irregular blocks so the recurrence is exercised across pushes.
        let mut det = DualMonoDetector::new();
        let mut i = 0;
        let mut step = 1;
        while i < left.len() {
            let end = (i + step).min(left.len());
            det.push(&left[i..end], &right[i..end]);
            i = end;
            step = step % 1_237 + 1;
        }
        assert!(
            (det.correlation() - batch).abs() < 1e-6,
            "{label}: streaming {} vs batch {batch}",
            det.correlation()
        );
        let want = is_dual_mono(&[left, right]);
        assert_eq!(det.is_dual_mono(), want, "{label}: verdict diverged");
    }
}

#[test]
fn test_dual_mono_detector_empty_is_not_dual_mono() {
    let det = DualMonoDetector::new();
    assert!(!det.is_dual_mono());
    assert_eq!(det.correlation(), 0.0);
    // An empty push keeps it empty.
    let mut det = DualMonoDetector::new();
    det.push(&[], &[]);
    assert!(!det.is_dual_mono());
}

/// Mismatched block lengths count only the overlap, mirroring `is_dual_mono`'s
/// `min(len)` truncation.
#[test]
fn test_dual_mono_detector_uses_the_overlap_only() {
    let a = vec![0.3f32, -0.4, 0.5, 0.9, -0.1];
    let b = vec![0.3f32, -0.4, 0.5];
    let mut det = DualMonoDetector::new();
    det.push(&a, &b);
    let batch = crate::inference::audio::decode::normalized_correlation_for_test(&a[..3], &b);
    assert!((det.correlation() - batch).abs() < 1e-9);
}

// --- One-pass channel scan (channels=split decision) ---

/// Two-channel PCM16 WAV with the given channels.
#[cfg(feature = "file-decode")]
fn stereo_wav(left: &[f32], right: &[f32], rate: u32) -> bytes::Bytes {
    let frames = left.len().min(right.len());
    let data_bytes = (frames * 4) as u32;
    let mut w = Vec::with_capacity(44 + data_bytes as usize);
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + data_bytes).to_le_bytes());
    w.extend_from_slice(b"WAVE");
    w.extend_from_slice(b"fmt ");
    w.extend_from_slice(&16u32.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes());
    w.extend_from_slice(&2u16.to_le_bytes());
    w.extend_from_slice(&rate.to_le_bytes());
    w.extend_from_slice(&(rate * 4).to_le_bytes());
    w.extend_from_slice(&4u16.to_le_bytes());
    w.extend_from_slice(&16u16.to_le_bytes());
    w.extend_from_slice(b"data");
    w.extend_from_slice(&data_bytes.to_le_bytes());
    let q = |s: f32| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
    for i in 0..frames {
        w.extend_from_slice(&q(left[i]).to_le_bytes());
        w.extend_from_slice(&q(right[i]).to_le_bytes());
    }
    bytes::Bytes::from(w)
}

/// The scan replaces "decode every channel of the whole file, then correlate".
/// Its verdict picks between transcribing two speakers and mixing to mono, so
/// it must match the batch answer exactly — checked at both a passthrough and a
/// resampling rate, on dual-mono and on genuine stereo.
#[cfg(feature = "file-decode")]
#[test]
fn test_scan_channels_matches_batch_dual_mono_verdict() {
    // 48 kHz resamples both channels. That sinc does not finish under Miri;
    // 16 kHz is the passthrough.
    let rates: &[u32] = if cfg!(miri) {
        &[16_000]
    } else {
        &[16_000, 48_000]
    };
    for &rate in rates {
        let n = if cfg!(miri) { 1_600 } else { rate as usize * 2 };
        let a: Vec<f32> = (0..n)
            .map(|i| 0.5 * ((i as f32) * 0.011).sin() + 0.15 * ((i as f32) * 0.0009).cos())
            .collect();
        let b: Vec<f32> = (0..n)
            .map(|i| 0.45 * ((i as f32) * 0.029 + 0.9).sin())
            .collect();

        for (label, left, right) in [
            ("identical", a.clone(), a.clone()),
            ("attenuated", a.clone(), a.iter().map(|v| v * 0.8).collect()),
            ("stereo", a.clone(), b.clone()),
        ] {
            let wav = stereo_wav(&left, &right, rate);
            let batch = decode_audio_bytes_shared_channels(wav.clone()).expect("batch");
            let want = is_dual_mono(&batch);
            let scan = scan_channels(wav, None).expect("scan");
            assert_eq!(scan.channels, 2, "{label} @{rate}");
            assert_eq!(
                scan.dual_mono, want,
                "{label} @{rate}: scan verdict diverged from batch"
            );
            assert_eq!(
                scan.mono_fallback_reason().is_some(),
                want,
                "{label} @{rate}"
            );
        }
    }
}

/// Anything that is not exactly two channels is decided from the header, so the
/// scan must not need to decode — and must report the same fallback the
/// whole-buffer path chose.
#[cfg(feature = "file-decode")]
#[test]
fn test_scan_channels_non_stereo_is_header_only() {
    let mono = encode_wav_pcm16(&vec![0.2f32; 16_000], 16000);
    let scan = scan_channels(bytes::Bytes::from(mono), None).expect("scan mono");
    assert_eq!(scan.channels, 1);
    assert!(!scan.dual_mono);
    assert_eq!(scan.mono_fallback_reason(), Some("mono audio"));
}

#[cfg(feature = "file-decode")]
#[test]
fn test_channel_scan_fallback_reasons() {
    let r = |channels, dual_mono| {
        ChannelScan {
            channels,
            dual_mono,
        }
        .mono_fallback_reason()
    };
    assert_eq!(r(0, false), Some("no channels"));
    assert_eq!(r(1, false), Some("mono audio"));
    assert_eq!(r(2, true), Some("dual-mono audio"));
    assert_eq!(r(2, false), None);
    assert_eq!(r(6, false), Some("more than two channels"));
}
