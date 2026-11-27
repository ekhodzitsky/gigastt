use super::*;

#[test]
fn test_v4_mkl_frame_geometry_rejects_invalid_inputs_before_native_call() {
    for n in [0, 1, 319, 480001] {
        assert!(validate_pcm(&vec![0.0; n], n).is_err());
    }
    assert!(validate_pcm(&vec![f32::NAN; 320], 320).is_err());
    assert!(validate_pcm(&[0.0; 320], 321).is_err());
    let grid = validate_pcm(&[0.0; 960], 641).unwrap();
    assert_eq!((grid.mel_frames(), grid.valid_mel_frames()), (5, 3));
}

#[test]
fn test_v4_mkl_aligned_buffer_and_exact_lengths() {
    let mut buffer = AlignedFloats::new(322 * 3, f32::NAN).unwrap();
    assert_eq!(buffer.as_mut_slice().len(), 322 * 3);
    assert_eq!(buffer.as_mut_slice().as_ptr().align_offset(64), 0);
    assert!(buffer.as_mut_slice().iter().all(|x| x.is_nan()));
    assert!(AlignedFloats::new(usize::MAX, 0.0).is_err());
}

#[test]
fn test_v4_mkl_runtime_invalid_paths_are_rejected_without_load() {
    let dir = tempfile::tempdir().unwrap();
    assert!(mkl::validate_runtime_dir(&dir.path().join("absent")).is_err());
    assert!(mkl::validate_runtime_dir(dir.path()).is_err());
    std::fs::create_dir(dir.path().join("glibc-hwcaps")).unwrap();
    assert!(mkl::validate_runtime_dir(dir.path()).is_err());
}

#[test]
fn test_v4_mkl_log_clamp_has_author_bounds() {
    for value in [0.0, 1e-12, 1e-9] {
        assert_eq!(log_energy(value), 1e-9_f32.ln());
    }
    assert_eq!(log_energy(1.0), 0.0);
    assert_eq!(log_energy(1e9), 1e9_f32.ln());
    assert_eq!(log_energy(1e10), 1e9_f32.ln());
}

/// Exact original author PCM, features and three archived spectra. This
/// ignored test runs the actual public API without loading any ASR sessions.
#[test]
#[ignore = "requires exact local v4 model pack, author fixtures and MKL side-load directory; no ASR"]
fn test_v4_mkl_pinned_author_frontend_parity() {
    use crate::sha256::{Sha256, hex_lower};
    use serde_json::{Value, json};
    use std::fs;
    fn digest(bytes: &[u8]) -> String {
        let mut hash = Sha256::new();
        hash.update(bytes);
        hex_lower(&hash.finalize())
    }
    fn verified(path: &Path, expected: &str) -> Vec<u8> {
        let raw = fs::read(path).unwrap();
        assert_eq!(digest(&raw), expected);
        raw
    }
    fn floats(raw: &[u8]) -> Vec<f32> {
        assert_eq!(raw.len() % 4, 0);
        raw.as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }
    fn bits(x: &[f32]) -> Vec<u8> {
        x.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
    fn number(v: &Value) -> usize {
        usize::try_from(v.as_u64().unwrap()).unwrap()
    }
    let root = std::env::var_os("GIGASTT_V4_REFERENCE_DIR").expect("local oracle root");
    let root = Path::new(&root);
    let output_dir =
        std::env::var_os("GIGASTT_V4_MKL_OUTPUT_DIR").expect("new private output directory");
    let output_dir = Path::new(&output_dir);
    fs::create_dir(output_dir).unwrap();
    let pack_dir = std::env::var_os("GIGASTT_QUALIFIED_V4_PACK_DIR").expect("local verified pack");
    let runtime_dir =
        std::env::var_os("GIGASTT_V4_MKL_RUNTIME_DIR").expect("exact local MKL runtime");
    let fake_path =
        std::env::var_os("GIGASTT_V4_MKL_FAKE_LIB").expect("local same-symbol test control");
    assert_eq!(
        digest(&fs::read(&fake_path).unwrap()),
        "6395c0220d8da9e9b7772331d528f851e8140e1665fe42c32e7553ab61f737bc"
    );
    let fake = mkl::FakeSymbolControl::open(Path::new(&fake_path));
    fake.check(1);
    let pack = QualifiedV4Pack::load(Path::new(&pack_dir)).unwrap();
    let frontend = V4Frontend::from_pack(&pack, Path::new(&runtime_dir)).unwrap();
    let feature_manifest: Value = serde_json::from_slice(&verified(
        &root.join("reference/feature-fixtures.json"),
        "4a6fd78a5a2d89aab8a2dbf3524947e564101a831c67ba7fe8159396bb6021d6",
    ))
    .unwrap();
    let loader_protocol: Value = serde_json::from_slice(&verified(
        &root.join("isolated-mkl-loader-vml-closure/protocol.json"),
        "e27b02134abb75bddf441f927df27e0c8dad9ff0a733e0da7d02307d22818338",
    ))
    .unwrap();
    assert_eq!(
        digest(COEFFICIENTS),
        "15c9da5c673fcfe5f6fa5dbfd28b8e19b658c14f15fd4c7ff656405764a417c0"
    );
    let cases = feature_manifest["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 21);
    let mut results = Vec::new();
    let mut controls = 0;
    let mut transforms = 0;
    for case in cases {
        let id = case["id"].as_str().unwrap();
        let samples = floats(&verified(
            Path::new(case["pcm_path"].as_str().unwrap()),
            case["pcm_sha256"].as_str().unwrap(),
        ));
        assert_eq!(samples.len(), number(&case["samples"]));
        let actual = frontend.compute_padded(&samples, number(&case["logical_samples"]));
        if case["status"] == "expected_author_error" {
            results.push(json!({"id":id,"expected_error":true,"pass":actual.is_err()}));
            continue;
        }
        let actual = actual.unwrap();
        let case_dir = output_dir.join(id);
        fs::create_dir(&case_dir).unwrap();
        for (name, values) in [
            ("windowed", &actual.trace.windowed),
            ("spectrum", &actual.trace.spectrum),
            ("power", &actual.trace.power),
            ("mel", &actual.trace.mel),
            ("log", &actual.values),
        ] {
            use std::io::Write;
            let mut f = fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(case_dir.join(format!("{name}.f32le")))
                .unwrap();
            f.write_all(&bits(values)).unwrap();
        }
        let grid = actual.grid;
        transforms += grid.mel_frames();
        let reference = floats(&verified(
            Path::new(case["features_path"].as_str().unwrap()),
            case["features_sha256"].as_str().unwrap(),
        ));
        let geometry = actual.values.len() == reference.len()
            && grid.mel_frames() == number(&case["mel_frames"])
            && grid.valid_mel_frames() == number(&case["valid_mel_frames"])
            && grid.encoder_frames() == number(&case["physical_encoded_frames_source_formula"])
            && grid.valid_encoder_frames() == number(&case["encoded_frames"])
            && (grid.frame_shift_seconds() - case["frame_shift_seconds"].as_f64().unwrap()).abs()
                <= 1e-9
            && (0..grid.mel_frames()).all(|k| grid.mel_frame_start_sample(k) == Some(k * 160));
        let floor = f32::from_bits(u32::try_from(number(&case["floor_bits"])).unwrap());
        let mut max_abs = 0.0_f64;
        let mut active_max = 0.0_f64;
        let mut active_sum = 0.0_f64;
        let mut active = 0_usize;
        for (&a, &r) in actual.values.iter().zip(&reference) {
            let diff = (f64::from(a) - f64::from(r)).abs();
            max_abs = max_abs.max(diff);
            if (f64::from(r) - f64::from(floor)).abs() > 0.05 {
                active += 1;
                active_max = active_max.max(diff);
                active_sum += diff;
            }
        }
        let active_mean = if active == 0 {
            0.0
        } else {
            active_sum / active as f64
        };
        let numeric = geometry
            && active == number(&case["active_bins"])
            && max_abs < 1e-3
            && active_max < 1e-3
            && active_mean < 1e-5;
        let old_spectrum = loader_protocol["standalone_spectra"][id].as_str().unwrap();
        let old_path = Path::new(old_spectrum);
        let old_hash = loader_protocol["assets"][old_spectrum].as_str().unwrap();
        let spectrum_equal = bits(&actual.trace.spectrum) == verified(old_path, old_hash);
        // The product's windowed input after DFTI must match its exact expected
        // f32 multiplication, including silent/padded rows; no in-place change.
        let expected_windowed: Vec<f32> = (0..grid.mel_frames())
            .flat_map(|f| (0..WINDOW).map(move |i| (f, i)))
            .map(|(f, i)| samples[f * HOP + i] * frontend.window[i])
            .collect();
        let input_unchanged = bits(&actual.trace.windowed) == bits(&expected_windowed);
        let archived = if let Some(control) = loader_protocol["archived_spectra"].get(id) {
            controls += 1;
            let mut matches = Vec::new();
            for (name, data) in [
                ("spectrum", &actual.trace.spectrum),
                ("power", &actual.trace.power),
                ("mel", &actual.trace.mel),
                ("log", &actual.values),
            ] {
                let path = control[name].as_str().unwrap();
                let expected = verified(
                    Path::new(path),
                    loader_protocol["assets"][path].as_str().unwrap(),
                );
                matches.push(bits(data) == expected);
            }
            Some(matches.into_iter().all(|x| x))
        } else {
            None
        };
        results.push(json!({"id":id,"geometry":geometry,"max_abs":max_abs,"active_max":active_max,"active_mean":active_mean,"active_bins":active,"archived_spectrum_post_bits_equal":archived,"standalone_spectrum_bits_equal_descriptive":spectrum_equal,"out_of_place_input_unchanged":input_unchanged,"pass":numeric&&input_unchanged&&archived!=Some(false)}));
    }
    assert_eq!(controls, 3);
    assert_eq!(transforms, 6865);
    println!(
        "V4_MKL_FRONTEND_PARITY {}",
        serde_json::to_string(&results).unwrap()
    );
    assert!(results.iter().all(|r| r["pass"] == true));
    // Reopening the same trusted directory reuses the already-loaded group.
    let second = V4Frontend::from_pack(&pack, Path::new(&runtime_dir)).unwrap();
    assert!(std::ptr::eq(frontend.runtime, second.runtime));
    let alternate = std::env::var_os("GIGASTT_V4_MKL_ALTERNATE_RUNTIME_DIR")
        .expect("distinct verified runtime directory");
    let error = V4Frontend::from_pack(&pack, Path::new(&alternate))
        .err()
        .expect("alternate runtime must fail");
    assert!(format!("{error:#}").contains("different MKL runtime"));
    // Two fixed concurrent one-frame synthetic silence calls exercise the
    // same process-lifetime group on different OS threads, serialized by the
    // runtime mutex. These are separate from the 18 oracle batches.
    let (a, b) = std::thread::scope(|s| {
        let a = s.spawn(|| frontend.compute(&[0.0; 320]).unwrap().values);
        let b = s.spawn(|| second.compute(&[0.0; 320]).unwrap().values);
        (a.join().unwrap(), b.join().unwrap())
    });
    assert_eq!(a, b);
    assert!(a.iter().all(|&x| x == 1e-9_f32.ln()));
    println!("V4_MKL_REUSE_AND_TWO_CONCURRENT_SILENCE_CALLS PASS");
    fake.check(2);
    println!("V4_MKL_BASE_FAKE_UNCHANGED PASS");
}
