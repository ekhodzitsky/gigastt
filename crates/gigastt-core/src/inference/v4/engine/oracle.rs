//! Opt-in local model-backed differential test with externally frozen author data.
//! Inputs and outputs are private operator-supplied files, not test fixtures in git.
use super::*;
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};

fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value[key]
        .as_str()
        .with_context(|| format!("missing string {key}"))
}
fn number(value: &Value, key: &str) -> Result<usize> {
    Ok(usize::try_from(
        value[key]
            .as_u64()
            .with_context(|| format!("missing integer {key}"))?,
    )?)
}
fn digest(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut result = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        write!(&mut result, "{byte:02x}").expect("string write");
    }
    result
}
fn read_floats(path: &Path, sha: &str) -> Result<Vec<f32>> {
    let bytes = fs::read(path)?;
    ensure!(digest(&bytes) == sha, "input SHA-256 mismatch");
    let (chunks, remainder) = bytes.as_chunks::<4>();
    ensure!(remainder.is_empty(), "unaligned float input");
    let data: Vec<f32> = chunks.iter().map(|b| f32::from_le_bytes(*b)).collect();
    ensure!(data.iter().all(|x| x.is_finite()), "nonfinite reference");
    Ok(data)
}
fn asset(value: &Value) -> Result<Vec<f32>> {
    read_floats(Path::new(string(value, "path")?), string(value, "sha256")?)
}
fn write_floats(path: &Path, data: &[f32]) -> Result<Value> {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(path, &bytes)?;
    Ok(json!({"path":path,"sha256":digest(&bytes),"bytes":bytes.len()}))
}
fn numeric(data: &[f32], reference: &Value) -> Result<Value> {
    let expected = asset(reference)?;
    if data.len() != expected.len() || data.is_empty() {
        return Ok(
            json!({"pass":false,"actual_elements":data.len(),"expected_elements":expected.len()}),
        );
    }
    let mut max = 0.0_f64;
    let mut sum = 0.0_f64;
    let mut within = true;
    for (&a, &b) in data.iter().zip(&expected) {
        let error = (f64::from(a) - f64::from(b)).abs();
        max = max.max(error);
        sum += error;
        within &= error <= 0.002 + 0.002 * f64::from(b).abs();
    }
    let mean = sum / data.len() as f64;
    Ok(json!({"max_abs":max,"mean_abs":mean,"pass":within&&mean<=0.001}))
}
fn feature_numeric(data: &[f32], reference: &[f32]) -> Result<Value> {
    ensure!(
        data.len() == reference.len() && !data.is_empty(),
        "feature shape mismatch"
    );
    let floor = f64::from(1.0e-9_f32.ln());
    let mut max = 0.0_f64;
    let mut active_max = 0.0_f64;
    let mut active_sum = 0.0_f64;
    let mut active = 0;
    for (&a, &b) in data.iter().zip(reference) {
        let error = (f64::from(a) - f64::from(b)).abs();
        max = max.max(error);
        if (f64::from(b) - floor).abs() > 0.05 {
            active += 1;
            active_max = active_max.max(error);
            active_sum += error;
        }
    }
    let mean = if active == 0 {
        0.0
    } else {
        active_sum / active as f64
    };
    Ok(
        json!({"max_abs":max,"active_max_abs":active_max,"active_mean_abs":mean,"active_count":active,"pass":max<0.001&&active_max<0.001&&mean<0.00001}),
    )
}
fn number_array_equal(actual: &[f64], expected: &Value, tolerance: f64) -> bool {
    expected.as_array().is_some_and(|arr| {
        arr.len() == actual.len()
            && actual
                .iter()
                .zip(arr)
                .all(|(a, b)| b.as_f64().is_some_and(|b| (a - b).abs() <= tolerance))
    })
}
fn head_output(result: &V4Transcription, reference: &Value, dir: &Path) -> Result<Value> {
    let ids: Vec<_> = result.tokens.iter().map(|t| t.id).collect();
    let frames: Vec<_> = result.tokens.iter().map(|t| t.frame).collect();
    let confidences: Vec<_> = result.tokens.iter().map(|t| t.confidence).collect();
    let groups:Vec<_>=result.words.iter().zip(&result.word_spans).map(|(w,s)|json!({"text":w.word,"start":w.start,"end":w.end,"confidence":w.confidence,"speaker":w.speaker,"token_indices":s.clone().collect::<Vec<_>>() })).collect();
    let mut arrays = serde_json::Map::new();
    let mut checks = serde_json::Map::new();
    let mut pass = true;
    for (name, data) in [
        ("prediction", &result.trace.prediction),
        ("h", &result.trace.h),
        ("c", &result.trace.c),
        ("log_probs", &result.trace.log_probs),
    ] {
        arrays.insert(
            name.into(),
            write_floats(&dir.join(format!("{name}.f32le")), data)?,
        );
        for backend in ["author", "split"] {
            let metrics = numeric(data, &reference[format!("{backend}_trace")][name])?;
            pass &= metrics["pass"] == true;
            checks.insert(format!("{backend}_{name}"), metrics);
        }
    }
    let metadata_ok = result.duration_s.to_bits()
        == (result.grid.valid_samples() as f64 / 16_000.0).to_bits()
        && result.words.iter().all(|w| {
            w.speaker.is_none()
                && w.start >= 0.0
                && w.end >= w.start
                && w.end <= result.duration_s + 1e-9
        });
    let reference_weighted = {
        let refs = reference["author_word_confidence_groups"]
            .as_array()
            .context("missing confidence groups")?;
        if refs.is_empty() {
            None
        } else {
            let mut weighted = 0.0;
            let mut weights = 0.0;
            for word in refs {
                let weight = word["end"].as_f64().context("word end")?
                    - word["start"].as_f64().context("word start")?;
                weighted += weight * word["confidence"].as_f64().context("word confidence")?;
                weights += weight;
            }
            ensure!(weights > 0.0, "zero reference word durations");
            Some(weighted / weights)
        }
    };
    let confidence_aggregate_ok = match (result.confidence, reference_weighted) {
        (None, None) => true,
        (Some(a), Some(b)) => (f64::from(a) - b).abs() <= 0.001,
        _ => false,
    };
    let official = &reference["official"];
    let exact = official["text"] == result.text
        && official["tokens"] == json!(ids)
        && official["token_frames"] == json!(frames);
    let official_groups = official["words"]
        .as_array()
        .context("missing official words")?;
    let official_words = official_groups.len() == groups.len()
        && official_groups.iter().zip(&groups).all(|(e, a)| {
            e["text"] == a["text"]
                && ["start", "end"].iter().all(|&k| {
                    e[k].as_f64()
                        .zip(a[k].as_f64())
                        .is_some_and(|(e, a)| (e - a).abs() <= 1e-9)
                })
        });
    let official_counts = official["counts"]["predictor_calls"]
        == json!(result.trace.predictor_inputs.len())
        && official["counts"]["joiner_calls"] == json!(result.trace.joint_inputs.len())
        && official["counts"]["joint_scored_rows"] == json!(result.trace.joint_inputs.len() * 8);
    pass &= exact && official_words && official_counts && metadata_ok && confidence_aggregate_ok;
    for (backend, trajectory) in [
        ("author", &reference["author_instrumented"]),
        ("split", &reference["split_ort"]),
    ] {
        let shape_ok = trajectory["text"] == result.text
            && trajectory["tokens"] == json!(ids)
            && trajectory["token_frames"] == json!(frames)
            && trajectory["predictor_inputs"] == json!(result.trace.predictor_inputs)
            && trajectory["joint_inputs"] == json!(result.trace.joint_inputs)
            && trajectory["predictor_calls"] == json!(result.trace.predictor_inputs.len())
            && trajectory["joiner_calls"] == json!(result.trace.joint_inputs.len())
            && trajectory["joint_scored_rows"] == json!(result.trace.joint_inputs.len() * 8)
            && trajectory["valid_joint_rows"]
                == json!(
                    result
                        .trace
                        .joint_inputs
                        .iter()
                        .map(|x| x[3])
                        .sum::<usize>()
                );
        let confidence_ok = number_array_equal(&confidences, &trajectory["confidence"], 0.001);
        let expected_groups = reference[format!("{backend}_word_confidence_groups")]
            .as_array()
            .context("missing word groups")?;
        let group_ok = expected_groups.len() == groups.len()
            && expected_groups.iter().zip(&groups).all(|(e, a)| {
                e["text"] == a["text"]
                    && e["token_indices"] == a["token_indices"]
                    && ["start", "end", "confidence"].iter().all(|&k| {
                        let tolerance = if k == "confidence" { 0.001 } else { 1.0e-9 };
                        e[k].as_f64()
                            .zip(a[k].as_f64())
                            .is_some_and(|(e, a)| (e - a).abs() <= tolerance)
                    })
            });
        pass &= shape_ok && confidence_ok && group_ok;
        checks.insert(
            format!("{backend}_trajectory"),
            json!({"exact":shape_ok,"confidence":confidence_ok,"groups":group_ok}),
        );
    }
    Ok(
        json!({"pass":pass,"exact_official":exact,"metadata_ok":metadata_ok,"aggregate_confidence_ok":confidence_aggregate_ok,"duration_s":result.duration_s,"style":match result.style { TranscriptStyle::Raw => "raw", TranscriptStyle::E2e => "e2e" },"aggregate_confidence":result.confidence,"official_words":official_words,"official_counts":official_counts,"text":result.text,"tokens":ids,"token_frames":frames,"confidence":confidences,"words":groups,"predictor_inputs":result.trace.predictor_inputs,"joint_inputs":result.trace.joint_inputs,"predictor_calls":result.trace.predictor_inputs.len(),"joiner_calls":result.trace.joint_inputs.len(),"joint_scored_rows":result.trace.joint_inputs.len()*8,"valid_joint_rows":result.trace.joint_inputs.iter().map(|x|x[3]).sum::<usize>(),"arrays":arrays,"checks":checks}),
    )
}
fn evaluate(
    result: &V4DualTranscription,
    case: &Value,
    reference_features: &[f32],
    lane: &str,
    dir: &Path,
) -> Result<Value> {
    let grid = result.raw.grid;
    ensure!(grid == result.e2e.grid, "different head geometry");
    let geometry_ok = grid.physical_samples() == number(case, "samples")?
        && grid.valid_samples() == number(case, "logical_samples")?
        && grid.mel_frames() == number(case, "mel_frames")?
        && grid.valid_mel_frames() == number(case, "valid_mel_frames")?
        && grid.encoder_frames() == number(case, "encoded_physical_frames")?
        && grid.valid_encoder_frames() == number(case, "encoded_frames")?
        && (grid.frame_shift_seconds()
            - case["frame_shift_seconds"]
                .as_f64()
                .context("frame shift")?)
        .abs()
            <= 1e-9;
    let features = feature_numeric(&result.features, reference_features)?;
    let feature_exact = result
        .features
        .iter()
        .map(|x| x.to_bits())
        .eq(reference_features.iter().map(|x| x.to_bits()));
    let encoder_author = numeric(&result.encoded, &case["acoustic"]["author_encoded"])?;
    let encoder_split = numeric(&result.encoded, &case["acoustic"]["split_encoded"])?;
    let raw_dir = dir.join("raw");
    let e2e_dir = dir.join("e2e");
    fs::create_dir(&raw_dir)?;
    fs::create_dir(&e2e_dir)?;
    let raw = head_output(&result.raw, &case["heads"]["raw"], &raw_dir)?;
    let e2e = head_output(&result.e2e, &case["heads"]["e2e"], &e2e_dir)?;
    let style_ok =
        result.raw.style == TranscriptStyle::Raw && result.e2e.style == TranscriptStyle::E2e;
    let pass = style_ok
        && geometry_ok
        && features["pass"] == true
        && (lane != "author_features" || feature_exact)
        && encoder_author["pass"] == true
        && encoder_split["pass"] == true
        && raw["pass"] == true
        && e2e["pass"] == true;
    Ok(
        json!({"pass":pass,"case":case["id"],"lane":lane,"geometry_ok":geometry_ok,"style_ok":style_ok,"geometry":{"physical_samples":grid.physical_samples(),"valid_samples":grid.valid_samples(),"physical_mel":grid.mel_frames(),"valid_mel":grid.valid_mel_frames(),"physical_encoder":grid.encoder_frames(),"valid_encoder":grid.valid_encoder_frames(),"frame_shift":grid.frame_shift_seconds()},"features":write_floats(&dir.join("features.f32le"),&result.features)?,"encoded":write_floats(&dir.join("encoded.f32le"),&result.encoded)?,"feature_metrics":features,"feature_exact":feature_exact,"encoder_author":encoder_author,"encoder_split":encoder_split,"raw":raw,"e2e":e2e}),
    )
}

#[test]
#[ignore = "requires exact qualified local v4 artifacts and frozen author oracle"]
fn test_v4_native_engine_against_frozen_author_oracle() -> Result<()> {
    let manifest = PathBuf::from(std::env::var("GIGASTT_V4_ENGINE_ORACLE")?);
    let output = PathBuf::from(std::env::var("GIGASTT_V4_ENGINE_ORACLE_OUTPUT")?);
    let pack = PathBuf::from(std::env::var("GIGASTT_V4_ENGINE_ORACLE_PACK")?);
    let mkl = PathBuf::from(std::env::var("GIGASTT_V4_ENGINE_ORACLE_MKL")?);
    let bytes = fs::read(manifest)?;
    let expected_sha = std::env::var("GIGASTT_V4_ENGINE_ORACLE_SHA256")?;
    ensure!(
        digest(&bytes) == expected_sha,
        "oracle manifest SHA mismatch"
    );
    let data: Value = serde_json::from_slice(&bytes)?;
    let cases = data["cases"].as_array().context("missing cases")?;
    ensure!(cases.len() == 11, "expected fixed eleven-case suite");
    fs::create_dir(&output)?;
    let engine = V4Engine::load(&pack, &mkl)?;
    // Model-free validation of the actual public API; neither request may
    // reach a native stage. No partial dual transcript is returned.
    let cancelled = AtomicBool::new(true);
    ensure!(
        matches!(
            engine.transcribe_both(V4Pcm::full(&[0.0; 320]), Some(&cancelled)),
            Err(GigasttError::Cancelled)
        ),
        "pre-cancelled dual request"
    );
    ensure!(
        matches!(
            engine.transcribe(V4Pcm::full(&[0.0; 319]), TranscriptStyle::Raw, None),
            Err(GigasttError::InvalidAudio { .. })
        ),
        "invalid audio request"
    );
    let mut summaries = Vec::new();
    for lane in ["author_features", "native_pcm"] {
        for case in cases {
            let key = string(case, "id")?;
            ensure!(
                key.chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
                "invalid case ID"
            );
            let dir = output.join(format!("{lane}-{key}"));
            fs::create_dir(&dir)?;
            let samples = number(case, "samples")?;
            let logical = number(case, "logical_samples")?;
            let grid = V4FrameGrid::new(samples, logical)?;
            let reference_features = asset(&case["features"])?;
            let result = if lane == "author_features" {
                let _slot = engine.acquire(None)?;
                engine.decode_both(reference_features.clone(), grid, None)?
            } else {
                let pcm = read_floats(
                    Path::new(string(case, "pcm_path")?),
                    string(case, "pcm_sha256")?,
                )?;
                ensure!(pcm.len() == samples, "PCM sample count mismatch");
                engine.transcribe_both(V4Pcm::padded(&pcm, logical), None)?
            };
            let summary = evaluate(&result, case, &reference_features, lane, &dir)?;
            fs::write(
                dir.join("result.json"),
                serde_json::to_vec_pretty(&summary)?,
            )?;
            summaries.push(summary);
        }
    }
    let pass = summaries.iter().all(|s| s["pass"] == true);
    let calls = |key: &str| {
        summaries
            .iter()
            .flat_map(|s| [&s["raw"], &s["e2e"]])
            .map(|h| h[key].as_u64().unwrap_or_default())
            .sum::<u64>()
    };
    fs::write(
        output.join("completion.json"),
        serde_json::to_vec_pretty(
            &json!({"pass":pass,"pack_identity":engine.pack_identity(),"public_api_pre_cancelled":true,"public_api_invalid_pcm":true,"case_count":cases.len(),"encoder_calls":summaries.len(),"head_trajectories":summaries.len()*2,"predictor_calls":calls("predictor_calls"),"joiner_calls":calls("joiner_calls"),"cases":summaries}),
        )?,
    )?;
    ensure!(pass, "one or more frozen v4 parity gates failed");
    Ok(())
}

fn exact_full_context_archive(summary: &Value, archived: &Value) -> bool {
    let mut exact = ["features", "encoded"]
        .iter()
        .all(|key| summary[key]["sha256"] == archived[key]["sha256"]);
    for name in ["raw", "e2e"] {
        for key in [
            "text",
            "tokens",
            "token_frames",
            "confidence",
            "words",
            "style",
            "duration_s",
            "aggregate_confidence",
            "predictor_inputs",
            "joint_inputs",
            "predictor_calls",
            "joiner_calls",
            "joint_scored_rows",
            "valid_joint_rows",
        ] {
            exact &= summary[name][key] == archived[name][key];
        }
        for key in ["prediction", "h", "c", "log_probs"] {
            exact &=
                summary[name]["arrays"][key]["sha256"] == archived[name]["arrays"][key]["sha256"];
        }
    }
    exact
}

#[test]
fn test_v4_saved_regression_roundtrips_archive_and_rejects_one_ulp() -> Result<()> {
    let make_summary = |end: f64, path: &str, metric: f64| {
        let asset = json!({"sha256":"same-contents", "path":path});
        let head = json!({"text":"known word", "words":[{"end":end}], "checks":{"diagnostic":metric},
            "arrays":{"prediction":asset,"h":asset,"c":asset,"log_probs":asset}});
        json!({"features":asset, "encoded":asset, "raw":head, "e2e":head})
    };
    let expected = 2.8799999999999994_f64;
    let current = make_summary(expected, "actual-output.f32le", 1e-7);
    let stored = serde_json::to_string(&make_summary(expected, "archived-output.f32le", 1e-5))?;
    let archived: Value = serde_json::from_str(&stored)?;
    // The test-only float_roundtrip feature preserves the exact stored f64.
    // Artifact paths and descriptive numeric metrics are not in the gate.
    assert_eq!(
        archived["raw"]["words"][0]["end"]
            .as_f64()
            .map(f64::to_bits),
        Some(expected.to_bits())
    );
    assert!(exact_full_context_archive(&current, &archived));
    let changed = make_summary(
        f64::from_bits(expected.to_bits() - 1),
        "actual-output.f32le",
        1e-7,
    );
    assert!(!exact_full_context_archive(&changed, &archived));
    Ok(())
}

#[test]
#[ignore = "one exact local full-context regression after native streaming decoder refactor"]
fn test_v4_full_context_ru_short_streaming_refactor_regression() -> Result<()> {
    let manifest = fs::read(std::env::var("GIGASTT_V4_ENGINE_ORACLE")?)?;
    ensure!(
        digest(&manifest) == std::env::var("GIGASTT_V4_ENGINE_ORACLE_SHA256")?,
        "oracle manifest SHA mismatch"
    );
    let oracle: Value = serde_json::from_slice(&manifest)?;
    let case = oracle["cases"]
        .as_array()
        .context("missing cases")?
        .iter()
        .find(|c| c["id"] == "ru_short")
        .context("missing ru_short")?;
    let archived = fs::read(std::env::var("GIGASTT_V4_ENGINE_ARCHIVE")?)?;
    ensure!(
        digest(&archived) == std::env::var("GIGASTT_V4_ENGINE_ARCHIVE_SHA256")?,
        "archived native result SHA mismatch"
    );
    let archived: Value = serde_json::from_slice(&archived)?;
    ensure!(
        archived["case"] == "ru_short"
            && archived["lane"] == "native_pcm"
            && archived["pass"] == true,
        "archived native lane mismatch"
    );
    let output = PathBuf::from(std::env::var("GIGASTT_V4_ENGINE_ORACLE_OUTPUT")?);
    fs::create_dir(&output)?;
    let pcm = read_floats(
        Path::new(string(case, "pcm_path")?),
        string(case, "pcm_sha256")?,
    )?;
    ensure!(
        pcm.len() == number(case, "samples")? && pcm.len() == number(case, "logical_samples")?,
        "unexpected ru_short geometry"
    );
    let native = V4Engine::load(
        Path::new(&std::env::var("GIGASTT_V4_ENGINE_ORACLE_PACK")?),
        Path::new(&std::env::var("GIGASTT_V4_ENGINE_ORACLE_MKL")?),
    )?;
    let result = native.transcribe_both(V4Pcm::full(&pcm), None)?;
    let summary = evaluate(
        &result,
        case,
        &asset(&case["features"])?,
        "native_pcm",
        &output,
    )?;
    let exact = exact_full_context_archive(&summary, &archived);
    let pass = exact && summary["pass"] == true;
    fs::write(
        output.join("completion.json"),
        serde_json::to_vec_pretty(
            &json!({"pass":pass,"archived_bits_metadata_exact":exact,"encoder_calls":1,"head_trajectories":2,"summary":summary}),
        )?,
    )?;
    ensure!(
        pass,
        "full-context regression was not exactly equal to archived native output"
    );
    Ok(())
}
