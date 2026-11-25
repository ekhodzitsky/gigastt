use super::Sample;
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::Path;

pub fn read_baseline(path: &Path, set: &str) -> Result<Value> {
    let baseline: Value =
        serde_json::from_slice(&std::fs::read(path).context("read benchmark baseline")?)
            .context("parse benchmark baseline")?;
    validate_baseline(&baseline, set)?;
    Ok(baseline)
}

fn number(value: &Value, name: &str, min: f64) -> Result<()> {
    ensure!(
        value.as_f64().is_some_and(|n| n.is_finite() && n >= min),
        "invalid baseline {name}"
    );
    Ok(())
}

fn integer(value: &Value, name: &str, min: u64) -> Result<()> {
    ensure!(
        value
            .as_u64()
            .is_some_and(|n| n >= min && n <= i64::MAX as u64),
        "invalid baseline {name}"
    );
    Ok(())
}

fn validate_baseline(b: &Value, set: &str) -> Result<()> {
    number(&b["tolerance_pp"], "tolerance_pp", 0.0)?;
    for k in ["max_wer", "max_rtf"] {
        number(&b["ceilings"][k], k, f64::MIN_POSITIVE)?;
    }
    ensure!(
        b["ceilings"]["max_wer"]
            .as_f64()
            .is_some_and(|v| v <= super::MAX_WER),
        "baseline exceeds fixed WER ceiling"
    );
    for k in [
        "max_rss_after_load_mb",
        "max_rss_after_decode_mb",
        "max_cold_start_ms",
    ] {
        integer(&b["ceilings"][k], k, 1)?;
    }
    number(&b["tolerances"]["rtf_abs"], "rtf_abs", 0.0)?;
    for k in ["rss_mb", "cold_start_ms"] {
        integer(&b["tolerances"][k], k, 0)?;
    }
    let selected = &b["sets"][set];
    integer(&selected["samples"], "samples", 1)?;
    number(&selected["wer"], "wer", 0.0)?;
    for k in [
        "rtf",
        "rss_after_load_mb",
        "rss_after_decode_mb",
        "cold_start_ms",
    ] {
        let v = selected
            .get(k)
            .with_context(|| format!("missing baseline {k}"))?;
        if !v.is_null() {
            if k == "rtf" {
                number(v, k, 0.0)?;
            } else {
                integer(v, k, 0)?;
            }
        }
    }
    Ok(())
}

/// Hash the ordered filename, reference and audio bytes, each length-prefixed
/// with a little-endian u64. JSON formatting and filesystem paths are irrelevant.
fn corpus_hash(samples: &[Sample], directory: &Path) -> Result<String> {
    let mut hash = Sha256::new();
    for sample in samples {
        let audio = std::fs::read(directory.join(&sample.filename))
            .with_context(|| format!("read audio {}", sample.filename))?;
        for bytes in [
            sample.filename.as_bytes(),
            sample.reference.as_bytes(),
            &audio,
        ] {
            hash.update((bytes.len() as u64).to_le_bytes());
            hash.update(bytes);
        }
    }
    Ok(hash.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

pub fn validate_corpus(
    samples: &[Sample],
    directory: &Path,
    baseline: &Value,
    set: &str,
    full: bool,
) -> Result<()> {
    ensure!(!samples.is_empty(), "empty corpus");
    let mut names = std::collections::HashSet::new();
    for sample in samples {
        ensure!(
            names.insert(&sample.filename),
            "duplicate corpus filename {}",
            sample.filename
        );
        ensure!(
            directory.join(&sample.filename).is_file(),
            "missing corpus audio {}",
            sample.filename
        );
    }
    ensure!(
        samples
            .iter()
            .any(|sample| !super::normalize_for_wer(&sample.reference).is_empty()),
        "no reference words in corpus"
    );
    let selected = &baseline["sets"][set];
    if full {
        ensure!(
            selected["samples"].as_u64() == Some(samples.len() as u64),
            "corpus sample count does not match baseline"
        );
    }
    if set == "bundled" && full {
        let expected = selected["corpus_sha256"]
            .as_str()
            .context("missing bundled corpus pin")?;
        ensure!(
            corpus_hash(samples, directory)? == expected,
            "bundled corpus identity mismatch"
        );
    }
    Ok(())
}

/// Required CI measures the current built-in RNNT INT8 model pack, not an
/// alternate configured or historically cached set of weights.
pub fn verify_required_model(directory: &Path) -> Result<()> {
    use gigastt::model::ModelVariant;
    use std::io::Read;
    let variant = ModelVariant::Rnnt;
    ensure!(
        !directory.join("manifest.toml").exists(),
        "required mode needs the built-in RNNT model layout"
    );
    for filename in variant.prequantized_files() {
        let expected = variant
            .prequantized_checksum(filename)
            .context("missing pinned model checksum")?;
        let mut file = std::fs::File::open(directory.join(filename))
            .with_context(|| format!("missing required model file {filename}"))?;
        let mut hash = Sha256::new();
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            hash.update(&buffer[..count]);
        }
        let actual: String = hash.finalize().iter().map(|b| format!("{b:02x}")).collect();
        ensure!(
            actual == expected,
            "required model checksum mismatch: {filename}"
        );
    }
    Ok(())
}

pub fn self_test() {
    let baseline: Value =
        serde_json::from_str(include_str!("../benchmark_baseline.json")).expect("test baseline");
    assert!(validate_baseline(&baseline, "bundled").is_ok());
    assert!(validate_baseline(&baseline, "external").is_ok());
    assert!(validate_baseline(&baseline, "missing").is_err());
    assert!(read_baseline(Path::new("/nonexistent/benchmark_baseline.json"), "bundled").is_err());
    let mut invalid = baseline.clone();
    invalid["sets"]["bundled"]["wer"] = Value::Null;
    assert!(validate_baseline(&invalid, "bundled").is_err());
    invalid = baseline.clone();
    invalid["ceilings"]["max_wer"] = serde_json::json!(16);
    assert!(validate_baseline(&invalid, "bundled").is_err());
    let directory = tempfile::tempdir().expect("test directory");
    let malformed = directory.path().join("baseline.json");
    std::fs::write(&malformed, b"{").expect("malformed baseline fixture");
    assert!(read_baseline(&malformed, "bundled").is_err());
    assert!(verify_required_model(directory.path()).is_err());
    std::fs::write(
        directory.path().join("v3_rnnt_encoder_int8.onnx"),
        b"foreign",
    )
    .expect("test model");
    assert!(verify_required_model(directory.path()).is_err());
    let mut samples = vec![Sample {
        filename: "sample.wav".into(),
        reference: "тест".into(),
    }];
    std::fs::write(directory.path().join("sample.wav"), b"fixture audio").expect("test fixture");
    let mut baseline = baseline;
    baseline["sets"]["bundled"]["samples"] = serde_json::json!(1);
    baseline["sets"]["bundled"]["corpus_sha256"] =
        serde_json::json!(corpus_hash(&samples, directory.path()).expect("test hash"));
    assert!(validate_corpus(&samples, directory.path(), &baseline, "bundled", true).is_ok());
    samples[0].reference.push_str(" изменить");
    assert!(validate_corpus(&samples, directory.path(), &baseline, "bundled", true).is_err());
    samples[0].reference = "тест".into();
    std::fs::write(directory.path().join("sample.wav"), b"changed audio").expect("test fixture");
    assert!(validate_corpus(&samples, directory.path(), &baseline, "bundled", true).is_err());
    baseline["sets"]["bundled"]["samples"] = serde_json::json!(2);
    assert!(validate_corpus(&samples, directory.path(), &baseline, "bundled", true).is_err());
    assert!(validate_corpus(&[], directory.path(), &baseline, "bundled", false).is_err());
    samples[0].reference = String::new();
    assert!(validate_corpus(&samples, directory.path(), &baseline, "external", false).is_err());
}
