use super::*;

#[test]
fn test_default_delegates_to_new() {
    // `Default` forwards to `new()`; assert both produce an equivalent
    // extractor by checking they yield identical features on silence.
    let silence = vec![0.0_f32; 3200];
    let (a, fa) = MelSpectrogram::default().compute(&silence);
    let (b, fb) = MelSpectrogram::new().compute(&silence);
    assert_eq!(fa, fb);
    assert_eq!(a, b);
}

#[test]
fn test_silence() {
    let mel = MelSpectrogram::new();
    let silence = vec![0.0_f32; 16000]; // 1 second of silence
    let (features, num_frames) = mel.compute(&silence);
    assert!(num_frames > 0);
    assert_eq!(features.len(), 64 * num_frames);
    // All mel energies should be at the floor value ln(1e-10)
    let floor = (1e-10_f32).ln();
    for &v in &features {
        assert!((v - floor).abs() < 0.01, "Expected ~{floor}, got {v}");
    }
}

#[test]
fn test_output_shape() {
    let mel = MelSpectrogram::new();
    let samples = vec![0.0_f32; 3200]; // 200ms at 16kHz
    let (features, num_frames) = mel.compute(&samples);
    // center=false: (3200 - 320) / 160 + 1 = 19 frames
    assert_eq!(num_frames, 19);
    assert_eq!(features.len(), 64 * 19);
}

#[test]
fn test_too_short() {
    let mel = MelSpectrogram::new();
    let samples = vec![0.0_f32; 100]; // Less than n_fft=320
    let (features, num_frames) = mel.compute(&samples);
    assert_eq!(num_frames, 1);
    assert_eq!(features.len(), 64);
}

#[test]
fn test_sine_wave() {
    let mel = MelSpectrogram::new();
    // 440Hz sine wave, 1 second at 16kHz
    let samples: Vec<f32> = (0..16000)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
        .collect();
    let (features, num_frames) = mel.compute(&samples);
    assert!(num_frames > 0);
    // Sine wave should produce non-floor values in some mel bins
    let floor = (1e-10_f32).ln();
    let non_floor = features
        .iter()
        .filter(|&&v| (v - floor).abs() > 1.0)
        .count();
    assert!(
        non_floor > 0,
        "Expected some non-floor values for sine wave"
    );
}

#[test]
fn test_sparsify_mel_filterbank_matches_dense_dot() {
    let n_fft = crate::inference::N_FFT;
    let n_mels = crate::inference::N_MELS;
    let n_freqs = n_fft / 2 + 1;
    let dense = MelSpectrogram::create_mel_filterbank(n_fft, n_mels, 16000.0, 0.0, 8000.0);
    let bands = MelSpectrogram::sparsify_mel_filterbank(&dense, n_mels, n_freqs);

    // Synthetic power spectrum with energy across the band.
    let power: Vec<f32> = (0..n_freqs)
        .map(|k| ((k as f32) * 0.01).sin().abs() + 0.1)
        .collect();

    for m in 0..n_mels {
        let mut dense_e = 0.0_f32;
        let row = &dense[m * n_freqs..(m + 1) * n_freqs];
        for (k, &p) in power.iter().enumerate() {
            dense_e += row[k] * p;
        }
        let band = &bands[m];
        let mut sparse_e = 0.0_f32;
        for (i, &w) in band.weights.iter().enumerate() {
            sparse_e += w * power[band.start + i];
        }
        assert!(
            (dense_e - sparse_e).abs() <= 1e-5 * dense_e.max(1.0),
            "band {m}: dense={dense_e} sparse={sparse_e}"
        );
    }

    // Sparsity: mean non-zeros should be far below n_freqs (triangular HTK).
    let mean_nz = bands.iter().map(|b| b.weights.len()).sum::<usize>() as f32 / n_mels as f32;
    assert!(
        mean_nz < (n_freqs as f32) * 0.25,
        "expected sparse bands, mean nz={mean_nz} of {n_freqs}"
    );
}

#[test]
fn test_sparse_compute_matches_dense_reference() {
    // Reconstruct a dense-apply reference and compare against the product
    // sparse path on a short sine — features must match within float noise.
    let n_fft = crate::inference::N_FFT;
    let n_mels = crate::inference::N_MELS;
    let hop = crate::inference::HOP_LENGTH;
    let n_freqs = n_fft / 2 + 1;
    let dense = MelSpectrogram::create_mel_filterbank(n_fft, n_mels, 16000.0, 0.0, 8000.0);
    let mel = MelSpectrogram::new();
    let samples: Vec<f32> = (0..3200)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
        .collect();
    let (sparse_feats, n_frames) = mel.compute(&samples);

    // Dense reference: same window + FFT plan, dense filterbank rows.
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let window: Vec<f32> = (0..n_fft)
        .map(|n| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / (n_fft - 1) as f32).cos()))
        .collect();
    let mut dense_feats = vec![0.0_f32; n_mels * n_frames];
    let mut fft_input = vec![Complex::new(0.0_f32, 0.0); n_fft];
    let mut power = vec![0.0_f32; n_freqs];
    for frame_idx in 0..n_frames {
        let start = frame_idx * hop;
        for i in 0..n_fft {
            let sample = samples.get(start + i).copied().unwrap_or(0.0);
            fft_input[i] = Complex::new(sample * window[i], 0.0);
        }
        fft.process(&mut fft_input[..n_fft]);
        for k in 0..n_freqs {
            power[k] = fft_input[k].norm_sqr();
        }
        for m in 0..n_mels {
            let mut e = 0.0_f32;
            let row = &dense[m * n_freqs..(m + 1) * n_freqs];
            for (k, &p) in power.iter().enumerate() {
                e += row[k] * p;
            }
            dense_feats[m * n_frames + frame_idx] = e.max(1e-10).ln();
        }
    }

    assert_eq!(sparse_feats.len(), dense_feats.len());
    for (i, (s, d)) in sparse_feats.iter().zip(dense_feats.iter()).enumerate() {
        let tol = 1e-4_f32 * d.abs().max(1.0);
        assert!(
            (s - d).abs() <= tol,
            "feat[{i}]: sparse={s} dense={d} tol={tol}"
        );
    }
}

/// Log-mel of `golos_00.wav` against the author package preprocessor
/// (`center=false`, HTK, hop 160, n_fft 320). The dump is float32, shape
/// `[64, 399]`, time fastest. The author clamps energy at `1e-9` before
/// `log`; this frontend uses `1e-10`, so silent bins differ by `ln(10)` and
/// are excluded from the active-bin check.
#[cfg(feature = "file-decode")]
#[test]
fn test_golos_00_mel_matches_author_preprocessor() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let wav = root.join("../gigastt/tests/fixtures/golos_00.wav");
    let author_path = root.join("../../benchmark/oracle/golos_00_author_mel.f32");
    let samples =
        crate::inference::audio::decode_audio_file(wav.to_str().expect("fixture path is utf-8"))
            .expect("decode golos_00");
    assert_eq!(samples.len(), 64_000, "golos_00 is 4 seconds at 16 kHz");
    let (features, frames) = MelSpectrogram::new().compute(&samples);
    assert_eq!(frames, 399);
    let bytes = std::fs::read(&author_path).expect("author mel dump");
    let (chunks, rest) = bytes.as_chunks::<4>();
    assert!(
        rest.is_empty(),
        "author mel dump is not a whole number of f32s"
    );
    assert_eq!(chunks.len(), features.len());
    let author: Vec<f32> = chunks.iter().copied().map(f32::from_le_bytes).collect();
    let author_floor = (1e-9_f32).ln();
    let mut max_abs = 0.0_f32;
    let mut active_max = 0.0_f32;
    let mut active_sum = 0.0_f64;
    let mut active = 0_usize;
    for (&ours, &theirs) in features.iter().zip(author.iter()) {
        let delta = (ours - theirs).abs();
        max_abs = max_abs.max(delta);
        if (theirs - author_floor).abs() > 0.05 {
            active_max = active_max.max(delta);
            active_sum += f64::from(delta);
            active += 1;
        }
    }
    assert!(
        active > 1_000,
        "expected speech bins above the author floor, got {active}"
    );
    let active_mean = active_sum / active as f64;
    // Silent bins sit on different clamps (author 1e-9, here 1e-10), so the
    // worst bin is ln(10) ≈ 2.303. Measured active-bin gap: mean 0.0088, max 0.307.
    assert!(
        max_abs < 2.31,
        "mel max abs {max_abs} (floor bins may differ by ln(10) ≈ 2.30)"
    );
    assert!(
        active_max < 0.35,
        "active-bin mel max abs {active_max} over {active} bins (mean {active_mean})"
    );
    assert!(active_mean < 0.02, "active-bin mel mean abs {active_mean}");
}
