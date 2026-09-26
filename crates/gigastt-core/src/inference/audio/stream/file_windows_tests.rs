use super::*;
use crate::inference::audio::encode_wav_pcm16;
use bytes::Bytes;

/// The ort long-form geometry, matching `Engine::window_spec` (CPU backend).
fn ort_spec() -> WindowSpec {
    WindowSpec::new(16000 * 30, 16000 * 24, 16000 * 2)
}

/// Deterministic, PCM16-quantization-exercising signal in [-1, 1).
fn signal(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32;
            0.4 * ((t * 0.017 + seed).sin() + 0.5 * (t * 0.0031 + seed).sin())
        })
        .collect()
}

/// Minimal interleaved-stereo PCM16 WAV (symphonia decodes this to two
/// channels, exercising the mono-mix branch of the streaming decode).
fn stereo_wav_pcm16(left: &[f32], right: &[f32], rate: u32) -> Vec<u8> {
    let frames = left.len().min(right.len());
    let data_bytes = (frames * 2 * 2) as u32;
    let byte_rate = rate * 2 * 2;
    let mut w = Vec::with_capacity(44 + data_bytes as usize);
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + data_bytes).to_le_bytes());
    w.extend_from_slice(b"WAVE");
    w.extend_from_slice(b"fmt ");
    w.extend_from_slice(&16u32.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes()); // PCM
    w.extend_from_slice(&2u16.to_le_bytes()); // channels
    w.extend_from_slice(&rate.to_le_bytes());
    w.extend_from_slice(&byte_rate.to_le_bytes());
    w.extend_from_slice(&4u16.to_le_bytes()); // block align
    w.extend_from_slice(&16u16.to_le_bytes()); // bits
    w.extend_from_slice(b"data");
    w.extend_from_slice(&data_bytes.to_le_bytes());
    let q = |s: f32| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
    for i in 0..frames {
        w.extend_from_slice(&q(left[i]).to_le_bytes());
        w.extend_from_slice(&q(right[i]).to_le_bytes());
    }
    w
}

/// Drain the whole windowed source into owned `(start, samples)` pairs.
fn window_seq(mut fw: FileWindows) -> Vec<(usize, Vec<f32>)> {
    let mut out = Vec::new();
    while let Some(w) = fw.next_window().expect("window") {
        out.push((w.start_sample, w.samples.to_vec()));
    }
    out
}

/// The [`SliceWindows`] sequence over a materialized buffer — the oracle for
/// the windowed (`> single_pass_max`) regime.
fn slice_seq(buf: &[f32], spec: WindowSpec) -> Vec<(usize, Vec<f32>)> {
    let mut sw = SliceWindows::new(buf, spec);
    let mut out = Vec::new();
    while let Some(w) = sw.next_window().expect("slice window") {
        out.push((w.start_sample, w.samples.to_vec()));
    }
    out
}

/// What `Engine::decode_words` would feed for a fully-decoded buffer: one
/// whole-buffer window at/under the single-pass ceiling, else the standard
/// overlapping geometry.
fn expected_seq(flat: &[f32], spec: WindowSpec) -> Vec<(usize, Vec<f32>)> {
    if flat.len() <= spec.single_pass_max() {
        vec![(0, flat.to_vec())]
    } else {
        slice_seq(flat, spec)
    }
}

fn assert_16k_geometry(spec: WindowSpec, lengths: &[usize]) {
    for &n in lengths {
        let src = signal(n, 1.0);
        let wav = encode_wav_pcm16(&src, 16000);
        let flat = FileWindows::from_bytes(Bytes::copy_from_slice(&wav), spec, None)
            .expect("open flat")
            .drain_to_vec()
            .expect("drain");
        // 16 kHz is the passthrough path: no resampler, so the decoded length
        // is exact.
        assert_eq!(flat.len(), n, "passthrough length changed at n={n}");
        let got = window_seq(
            FileWindows::from_bytes(Bytes::copy_from_slice(&wav), spec, None)
                .expect("open windows"),
        );
        assert_eq!(got, expected_seq(&flat, spec), "geometry mismatch at n={n}");
    }
}

#[test]
fn test_file_windows_16k_geometry_matches_decode_words() {
    // A few thousand samples still cross a single-pass ceiling, so Miri checks
    // the overlapping geometry. The production 30 s lengths below are the same
    // check at the real ceiling; hundreds of thousands of samples do not finish
    // inside the nightly Miri budget (the job stalls on this test).
    let small = WindowSpec::new(2_000, 1_600, 320);
    assert_16k_geometry(small, &[1, 800, 2_000, 2_001, 4_000]);
    #[cfg(not(miri))]
    assert_16k_geometry(ort_spec(), &[1, 8_000, 480_000, 480_001, 560_000, 900_000]);
}

#[test]
#[cfg_attr(
    miri,
    ignore = "40 s of 48 kHz stereo plus the sinc resampler does not finish under Miri"
)]
fn test_file_windows_48k_stereo_matches_slice_over_drain() {
    let spec = ort_spec();
    // 40 s @48 kHz stereo → ~40 s @16 kHz mono, above the single-pass ceiling,
    // so the windowed geometry (not a single window) is exercised, through the
    // resampler and the mono-mix branch.
    let n = 48_000 * 40;
    let left = signal(n, 0.3);
    let right = signal(n, 2.1);
    let wav = stereo_wav_pcm16(&left, &right, 48_000);
    let flat = FileWindows::from_bytes(Bytes::copy_from_slice(&wav), spec, None)
        .expect("open flat")
        .drain_to_vec()
        .expect("drain");
    assert!(
        flat.len() > spec.single_pass_max(),
        "expected the chunked regime, got {} samples",
        flat.len()
    );
    let got = window_seq(
        FileWindows::from_bytes(Bytes::copy_from_slice(&wav), spec, None).expect("open windows"),
    );
    // Incremental resample + windowing is byte-identical to whole-buffer
    // resample + SliceWindows: the staged chunk sequence is the same either
    // way, so every resampled sample matches.
    assert_eq!(got, slice_seq(&flat, spec));
}

/// The load-bearing claim for streaming `channels=split`: pulling channel
/// `k` through the packet loop yields the *same samples* the whole-buffer
/// per-channel decode produced. Checked at 16 kHz (resampler passthrough)
/// and 48 kHz (real resampling), for both channels of a genuine stereo
/// stream whose channels differ.
#[test]
fn test_file_windows_channel_select_matches_batch_per_channel_decode() {
    // 48 kHz resamples. A 3 s clip of that does not finish under Miri; the
    // 16 kHz arm is the passthrough the interpreter can still check.
    let rates: &[u32] = if cfg!(miri) {
        &[16_000]
    } else {
        &[16_000, 48_000]
    };
    for &rate in rates {
        let n = if cfg!(miri) { 1_600 } else { rate as usize * 3 };
        let left = signal(n, 0.3);
        let right = signal(n, 2.1);
        let wav = stereo_wav_pcm16(&left, &right, rate);
        let batch = crate::inference::audio::decode_audio_bytes_shared_channels(
            Bytes::copy_from_slice(&wav),
        )
        .expect("batch per-channel decode");
        assert_eq!(batch.len(), 2, "expected a stereo decode at {rate}Hz");
        // The channels must actually differ, or the test proves nothing.
        assert_ne!(batch[0], batch[1]);

        for (k, want) in batch.iter().enumerate() {
            let streamed = FileWindows::from_bytes_channel(
                Bytes::copy_from_slice(&wav),
                WindowSpec::flat(),
                None,
                k,
            )
            .expect("open channel")
            .drain_to_vec()
            .expect("drain channel");
            assert_eq!(&streamed, want, "rate={rate} channel={k}");
        }
    }
}

/// Selecting a channel the stream does not have yields nothing rather than
/// silently falling back to another channel's audio.
#[test]
fn test_file_windows_channel_select_out_of_range_is_empty() {
    let src = signal(16_000, 1.0);
    let wav = encode_wav_pcm16(&src, 16000); // mono
    let streamed =
        FileWindows::from_bytes_channel(Bytes::copy_from_slice(&wav), WindowSpec::flat(), None, 5)
            .expect("open")
            .drain_to_vec()
            .expect("drain");
    assert!(streamed.is_empty(), "got {} samples", streamed.len());
}

#[test]
fn test_file_windows_single_pass_yields_one_window() {
    let spec = ort_spec();
    let src = signal(10_000, 0.7);
    let wav = encode_wav_pcm16(&src, 16000);
    let got = window_seq(
        FileWindows::from_bytes(Bytes::copy_from_slice(&wav), spec, None).expect("open"),
    );
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].0, 0);
    assert_eq!(got[0].1.len(), 10_000);
}

#[test]
fn test_file_windows_total_16k_samples_is_exact_at_16k() {
    // 700_000 samples is the native chunked case. Miri uses a small spec that
    // is still past its own single-pass ceiling.
    let (spec, n) = if cfg!(miri) {
        (WindowSpec::new(2_000, 1_600, 320), 4_000)
    } else {
        (ort_spec(), 700_000)
    };
    let src = signal(n, 1.3);
    let wav = encode_wav_pcm16(&src, 16000);
    let mut fw = FileWindows::from_bytes(Bytes::copy_from_slice(&wav), spec, None).expect("open");
    while fw.next_window().expect("window").is_some() {}
    assert_eq!(fw.total_16k_samples(), n);
}

/// Peak-RSS instrument (decode-only, no model). Writes an `N`-second 48 kHz
/// stereo WAV to a temp file, streams every window discarding the samples,
/// and asserts the total is right. Run it under a peak-RSS meter at several
/// `GIGASTT_PEAK_SECONDS` values — the slope of peak RSS per audio-second is
/// the memory claim:
///
/// ```sh
/// for s in 60 300 1200; do GIGASTT_PEAK_SECONDS=$s /usr/bin/time -l \
///   cargo test -p gigastt-core --lib \
///   file_windows_tests::zzz_streaming_decode_peak_instrument \
///   -- --ignored --exact --nocapture 2>&1 | grep -E 'maximum resident'; done
/// ```
#[test]
#[ignore = "decode-only peak-RSS instrument; drive with GIGASTT_PEAK_SECONDS under /usr/bin/time"]
fn zzz_streaming_decode_peak_instrument() {
    let secs: usize = std::env::var("GIGASTT_PEAK_SECONDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let path = std::env::temp_dir().join(format!("gigastt_peak_{secs}s.wav"));

    // Stream the synthetic WAV to disk one second at a time so generating it
    // never holds more than a second of audio in RAM — the decode is what we
    // are measuring, not the fixture.
    {
        use std::io::Write;
        let rate = 48_000u32;
        let frames = secs * rate as usize;
        let data_bytes = (frames * 2 * 2) as u32;
        let f = std::fs::File::create(&path).expect("create temp wav");
        let mut w = std::io::BufWriter::new(f);
        w.write_all(b"RIFF").unwrap();
        w.write_all(&(36 + data_bytes).to_le_bytes()).unwrap();
        w.write_all(b"WAVE").unwrap();
        w.write_all(b"fmt ").unwrap();
        w.write_all(&16u32.to_le_bytes()).unwrap();
        w.write_all(&1u16.to_le_bytes()).unwrap();
        w.write_all(&2u16.to_le_bytes()).unwrap();
        w.write_all(&rate.to_le_bytes()).unwrap();
        w.write_all(&(rate * 4).to_le_bytes()).unwrap();
        w.write_all(&4u16.to_le_bytes()).unwrap();
        w.write_all(&16u16.to_le_bytes()).unwrap();
        w.write_all(b"data").unwrap();
        w.write_all(&data_bytes.to_le_bytes()).unwrap();
        for sec in 0..secs {
            let base = (sec * rate as usize) as f32;
            for i in 0..rate as usize {
                let t = base + i as f32;
                let s = (0.4 * (t * 0.02).sin() * i16::MAX as f32) as i16;
                w.write_all(&s.to_le_bytes()).unwrap();
                w.write_all(&s.to_le_bytes()).unwrap();
            }
        }
        w.flush().unwrap();
    }

    let p = path.to_str().unwrap();
    // `GIGASTT_PEAK_MODE=drain` measures the old whole-buffer decode (peak
    // grows with duration) for a same-build A/B against the default windowed
    // path (peak bounded by one window).
    let total = if std::env::var("GIGASTT_PEAK_MODE").as_deref() == Ok("drain") {
        FileWindows::decode_file(p, None).expect("drain").len()
    } else {
        let mut fw = FileWindows::open(p, ort_spec(), None).expect("open temp wav");
        let mut counted = 0usize;
        while let Some(win) = fw.next_window().expect("window") {
            counted += win.samples.len();
        }
        // Overlapping windows re-count their overlap, so the summed window
        // length exceeds the (bounded) true total.
        assert!(counted >= fw.total_16k_samples());
        fw.total_16k_samples()
    };
    let _ = std::fs::remove_file(&path);

    let expected = secs * 16_000;
    assert!(
        (total as i64 - expected as i64).unsigned_abs() < 16_000,
        "total {total} not within 1 s of {expected}"
    );
}
