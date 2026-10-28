// Installed unchanged as a test-only child of the production Core ML bridge.
use super::*;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::time::Instant;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[test]
#[ignore = "run with scripts/compare-objc-revisions.py on macOS"]
#[allow(clippy::assertions_on_constants)]
fn test_objc_revision_probe() {
    assert!(!cfg!(debug_assertions), "release probe required");
    let package = PathBuf::from(std::env::var_os("GIGASTT_OBJC_PROBE_MODEL").unwrap());
    assert!(package.is_dir(), "verified model package missing");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../gigastt/tests/fixtures");
    let features = crate::inference::FeatureExtractor::new();
    let mut fixtures = Vec::new();
    let mut inputs = Vec::new();
    for name in ["golos_00.wav", "golos_01.wav", "golos_02.wav"] {
        let file = root.join(name);
        fixtures.push(hex(&Sha256::digest(std::fs::read(&file).unwrap())));
        let samples = crate::inference::audio::decode_audio_file(file.to_str().unwrap()).unwrap();
        let (mel, frames) = features.compute(&samples[..samples.len().min(48_000)]);
        assert!((1..=768).contains(&frames));
        let mut padded = vec![0.0; 64 * 768];
        for band in 0..64 {
            padded[band * 768..band * 768 + frames]
                .copy_from_slice(&mel[band * frames..(band + 1) * frames]);
        }
        inputs.push(padded);
    }
    let fingerprint = |values: &[f32], shape: &[usize]| {
        assert!(!values.is_empty());
        assert!(values.iter().all(|v| v.is_finite()));
        assert_eq!(shape.iter().product::<usize>(), values.len());
        let mut hash = Sha256::new();
        for value in values {
            hash.update(value.to_bits().to_le_bytes());
        }
        serde_json::json!({"sha256": hex(&hash.finalize()), "shape": shape})
    };
    let mut modes = serde_json::Map::new();
    for (name, cpu_and_ne) in [("cpu", false), ("cpu_and_ne", true)] {
        let cached = super::cache::cached_model_path(&package).is_dir();
        let started = Instant::now();
        let model = compile_and_load(&package, cpu_and_ne).unwrap();
        let load_seconds = started.elapsed().as_secs_f64();
        let mut outputs = Vec::new();
        for input in &inputs {
            let (values, shape) =
                predict_f32(&model, "mel", input, &[1, 64, 768], "encoded").unwrap();
            outputs.push(fingerprint(&values, &shape));
        }
        let mut seconds = 0.0;
        for _ in 0..3 {
            for (input, expected) in inputs.iter().zip(&outputs) {
                let started = Instant::now();
                let (values, shape) =
                    predict_f32(&model, "mel", input, &[1, 64, 768], "encoded").unwrap();
                seconds += started.elapsed().as_secs_f64();
                assert_eq!(&fingerprint(&values, &shape), expected, "unstable encoder");
            }
        }
        modes.insert(
            name.to_owned(),
            serde_json::json!({
                "cached": cached, "load_seconds": load_seconds, "seconds": seconds,
                "outputs": outputs,
            }),
        );
    }
    println!(
        "objc revision {}",
        serde_json::json!({"fixtures": fixtures, "modes": modes})
    );
}
