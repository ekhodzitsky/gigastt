use super::*;

// ── ANE packages ────────────────────────────────────────────────────────

#[cfg(feature = "ane")]
#[test]
fn test_ane_buckets_ladder_pinned() {
    // Must match the convert script's --buckets default.
    assert_eq!(ANE_BUCKETS, &[512, 768, 1536, 3000]);
}

/// Every shipped bucket must clear the ANE-residency floor (~288 mel frames):
/// below it the fixed-shape graph falls off the Neural Engine onto the CPU EP
/// (measured in the conversion spike), so a too-small bucket would silently
/// regress to CPU. 512 (the smallest) clears 288; this guards future ladder
/// edits from adding a bucket below the residency floor.
#[cfg(feature = "ane")]
#[test]
fn test_ane_buckets_above_residency_floor() {
    const ANE_RESIDENCY_FLOOR: usize = 288;
    for &b in ANE_BUCKETS {
        assert!(
            b >= ANE_RESIDENCY_FLOOR,
            "ANE bucket {b} is below the {ANE_RESIDENCY_FLOOR}-mel residency floor — it would evict to CPU"
        );
    }
}

#[cfg(all(feature = "net", feature = "ane"))]
#[test]
fn test_ane_tar_checksums_shape() {
    // Exactly one entry per bucket; each entry is either the empty
    // (unreleased) sentinel or a valid 64-char lowercase-hex digest.
    assert_eq!(ANE_TAR_CHECKSUMS.len(), ANE_BUCKETS.len());
    for &b in ANE_BUCKETS {
        let entries: Vec<_> = ANE_TAR_CHECKSUMS
            .iter()
            .filter(|(bucket, _)| *bucket == b)
            .collect();
        assert_eq!(entries.len(), 1, "exactly one ANE checksum entry for {b}");
        let sum = entries[0].1;
        if sum.is_empty() {
            continue; // genuine unreleased state
        }
        assert_eq!(
            sum.len(),
            64,
            "ANE {b} checksum must be 64 hex chars: {sum}"
        );
        assert!(
            sum.chars()
                .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
            "ANE {b} checksum must be lowercase hex: {sum}"
        );
    }
}

#[cfg(feature = "ane")]
#[test]
fn test_ane_filename_helpers() {
    assert_eq!(ane_package_dir_name(768), "gigaam_v3_encoder_768.mlpackage");
    assert_eq!(ane_tar_name(768), "gigaam_v3_encoder_768.mlpackage.tar");
}

#[cfg(feature = "ane")]
#[test]
fn test_default_ane_model_dir_is_model_sibling() {
    let ane = default_ane_model_dir();
    assert!(
        ane.contains(".gigastt") && ane.ends_with("ane"),
        "ane dir should be under .gigastt and end with 'ane', got: {ane}"
    );
}

/// Stage the FULL structurally-required file set Core ML writes into a
/// `.mlpackage` (manifest + model spec + weights blob) under a bucket dir.
#[cfg(feature = "ane")]
fn stage_complete_ane_package(pkg: &Path) {
    let coreml = pkg.join("Data").join("com.apple.CoreML");
    std::fs::create_dir_all(coreml.join("weights")).unwrap();
    std::fs::write(pkg.join("Manifest.json"), b"{}").unwrap();
    std::fs::write(coreml.join("model.mlmodel"), b"spec").unwrap();
    std::fs::write(coreml.join("weights").join("weight.bin"), b"w").unwrap();
}

#[cfg(feature = "ane")]
#[test]
fn test_is_ane_present_false_on_empty_then_true_when_staged() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();
    assert!(!is_ane_present(dir), "empty dir has no ANE packages");

    for &b in ANE_BUCKETS {
        stage_complete_ane_package(&dir.join(ane_package_dir_name(b)));
    }
    assert!(is_ane_present(dir), "all buckets fully staged → present");
}

/// A torn package (only `Manifest.json`, no model spec / weights) must NOT
/// be reported complete — otherwise the download path wedges forever.
#[cfg(feature = "ane")]
#[test]
fn test_ane_package_complete_false_when_torn() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();
    let pkg = dir.join(ane_package_dir_name(768));
    std::fs::create_dir_all(&pkg).unwrap();
    std::fs::write(pkg.join("Manifest.json"), b"{}").unwrap();

    assert!(
        !ane_package_complete(&pkg),
        "manifest-only package is torn, not complete"
    );

    // Stage the other buckets fully; the torn 768 bucket must still drag
    // the whole-dir check to false.
    for &b in &ANE_BUCKETS[1..] {
        stage_complete_ane_package(&dir.join(ane_package_dir_name(b)));
    }
    assert!(!is_ane_present(dir), "torn bucket → not present");
}

/// Build a deterministic `.tar` (a `<pkg_name>/` dir whose arcnames are
/// prefixed with the package name) holding the full required file set,
/// written at `tar_path`. Mirrors what `release-ane.yml` publishes.
#[cfg(feature = "ane")]
fn build_ane_tar(tar_path: &Path, pkg_name: &str) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let pkg = tmp.path().join(pkg_name);
    stage_complete_ane_package(&pkg);

    let file = std::fs::File::create(tar_path).unwrap();
    let mut builder = tar::Builder::new(file);
    builder.append_dir_all(pkg_name, &pkg).unwrap();
    builder.finish().unwrap();
}

/// Building a deterministic tar (a `gigaam_v3_encoder_768.mlpackage/` dir
/// with the full file set) and unpacking it with `tar::Archive` reconstructs
/// the directory + files — proves the extract step end-to-end, no network.
#[cfg(feature = "ane")]
#[test]
fn test_ane_tar_roundtrip_extract() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let tar_path = tmp.path().join("pkg.tar");
    build_ane_tar(&tar_path, "gigaam_v3_encoder_768.mlpackage");

    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let file = std::fs::File::open(&tar_path).unwrap();
    tar::Archive::new(file).unpack(&out).unwrap();

    let extracted = out.join("gigaam_v3_encoder_768.mlpackage");
    assert!(
        ane_package_complete(&extracted),
        "extracted .mlpackage must be complete"
    );
}

/// `extract_ane_tar_atomic` reconstructs the package at its final path and
/// leaves no `.extract.*` staging dir behind on success.
#[cfg(all(feature = "net", feature = "ane"))]
#[test]
fn test_extract_ane_tar_atomic_no_staging_leak() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();
    let pkg_name = ane_package_dir_name(768);
    let tar_dest = dir.join(ane_tar_name(768));
    build_ane_tar(&tar_dest, &pkg_name);

    extract_ane_tar_atomic(&tar_dest, dir, &pkg_name).expect("atomic extract");

    assert!(
        ane_package_complete(&dir.join(&pkg_name)),
        "package must land complete at its final path"
    );
    let leaked = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .any(|e| e.file_name().to_string_lossy().starts_with(".extract."));
    assert!(
        !leaked,
        "no .extract.* staging dir may remain after success"
    );
}

/// Every shipped bucket resolves to its pinned `.tar` checksum; a bucket with
/// no pin (an empty sentinel, or one outside the ladder) surfaces the
/// actionable "not yet published" bail rather than downloading unverified.
#[cfg(all(feature = "net", feature = "ane"))]
#[test]
fn test_require_ane_tar_checksum_resolves_pinned_and_bails_unpinned() {
    // Each ladder bucket is pinned to its release `.tar` SHA-256.
    for &b in ANE_BUCKETS {
        let sum = require_ane_tar_checksum(b).expect("shipped bucket must be pinned");
        assert_eq!(sum.len(), 64, "checksum must be 64 hex chars, got: {sum}");
    }
    // A bucket with no pin (here: outside the ladder) takes the bail path.
    let err = require_ane_tar_checksum(99_999).expect_err("unpinned bucket must bail");
    assert!(
        format!("{err}").contains("not yet published"),
        "unexpected error: {err}"
    );
}

/// The offline-bundle fetch script duplicates the crate's SHA-256 pins
/// (it must run on a machine without gigastt installed). Silent drift —
/// e.g. re-quantizing the encoder and bumping only the crate constant —
/// would break release builds at tag time; this pins the two sources of
/// truth together in PR CI instead.
#[cfg(feature = "net")]
#[test]
fn test_fetch_offline_models_script_pins_match_crate_constants() {
    let script_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/fetch_offline_models.sh");
    let script = std::fs::read_to_string(&script_path)
        .expect("read fetch_offline_models.sh")
        // Windows checkouts may use CRLF; join-continuations below match on `\n`.
        .replace("\r\n", "\n")
        .replace('\r', "\n");

    assert!(
        script.contains(PREQUANT_RELEASE_BASE),
        "script must fetch the model bundle from the release the crate pins ({PREQUANT_RELEASE_BASE})"
    );
    assert!(
        script.contains(PUNCT_HF_REPO),
        "script must fetch the punctuation model from the repo the crate pins ({PUNCT_HF_REPO})"
    );

    // Join backslash-continued lines so every `fetch "URL" "DEST" "SHA"`
    // call is a single parseable line.
    let joined = script.replace("\\\n", " ");
    let mut checked = 0usize;
    for line in joined.lines() {
        let line = line.trim();
        if !line.starts_with("fetch ") {
            continue;
        }
        let parts: Vec<&str> = line.split('"').collect();
        assert!(parts.len() >= 6, "unparseable fetch line: {line}");
        let (dest, sha) = (parts[3], parts[5]);
        let file = dest.rsplit('/').next().expect("dest basename");
        let expected = if file == ModelVariant::Rnnt.encoder_int8_file() {
            ModelVariant::Rnnt.encoder_int8_checksum()
        } else if let Some(c) = ModelVariant::Rnnt.checksum(file) {
            c
        } else if let Some((_, c)) = PUNCT_FILES.iter().find(|(f, _)| *f == file) {
            c
        } else {
            panic!("script fetches {file}, which the crate has no pin for");
        };
        assert_eq!(
            sha, expected,
            "SHA-256 pin drift for {file}: script says {sha}, crate says {expected}"
        );
        checked += 1;
    }
    assert!(
        checked >= 7,
        "expected the script to pin at least 7 files, parsed {checked}"
    );
}
