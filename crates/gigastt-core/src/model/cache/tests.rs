use super::*;
use std::io::Write;

fn write_file(path: &Path, bytes: &[u8]) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(bytes).unwrap();
}

fn content_cache_path(cache: &Path, encoder: &Path) -> PathBuf {
    cache.join(optimized_content_basename(
        &optimized_source_hash(encoder).unwrap().unwrap(),
        &"a".repeat(64),
    ))
}

#[test]
fn test_optimized_cache_basename_from_int8_encoder() {
    let p = Path::new("/m/v3_rnnt_encoder_int8.onnx");
    assert_eq!(
        optimized_cache_basename(p).as_deref(),
        Some("v3_rnnt_encoder_int8_optimized.ort")
    );
}

#[test]
fn test_prune_optimized_cache_keeps_active_int8_drops_zombies() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    // Active rnnt INT8 install (lean set stub).
    for f in ModelVariant::Rnnt.prequantized_files() {
        write_file(&dir.join(f), f.as_bytes());
    }
    let cache = dir.join("optimized_cache");
    let keep = content_cache_path(&cache, &dir.join("v3_rnnt_encoder_int8.onnx"));
    let legacy_active = cache.join("v3_rnnt_encoder_int8_optimized.onnx");
    let fp32 = cache.join("v3_rnnt_encoder_optimized.onnx");
    let e2e = cache.join("v3_e2e_rnnt_encoder_int8_optimized.ort");
    write_file(&keep, &[1u8; 100]);
    write_file(&legacy_active, &[2u8; 200]);
    write_file(&fp32, &[3u8; 300]);
    write_file(&e2e, &[4u8; 400]);
    // Unrelated file must stay.
    write_file(&cache.join("notes.txt"), b"keep me");

    let report = prune_optimized_cache(dir, false).unwrap();
    assert_eq!(report.kept, vec![keep.clone()]);
    assert_eq!(report.removed.len(), 3);
    assert!(keep.exists());
    assert!(!legacy_active.exists());
    assert!(!fp32.exists());
    assert!(!e2e.exists());
    assert!(cache.join("notes.txt").exists());
    assert_eq!(report.freed_bytes, 200 + 300 + 400);
}

#[test]
fn test_prune_optimized_cache_keeps_every_installed_head() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    // Two heads installed (rnnt + e2e_rnnt); ml_ctc is not.
    for f in ModelVariant::Rnnt.prequantized_files() {
        write_file(&dir.join(f), f.as_bytes());
    }
    for f in ModelVariant::E2eRnnt.prequantized_files() {
        write_file(&dir.join(f), f.as_bytes());
    }
    let cache = dir.join("optimized_cache");
    let rnnt = content_cache_path(&cache, &dir.join("v3_rnnt_encoder_int8.onnx"));
    let e2e = content_cache_path(&cache, &dir.join("v3_e2e_rnnt_encoder_int8.onnx"));
    // Graph for a head whose weights are absent: zombie.
    let ml = cache.join("multilingual_ctc.int8_optimized.ort");
    // Legacy pre-flatbuffer graphs: zombies even for installed heads.
    let legacy = cache.join("v3_rnnt_encoder_int8_optimized.onnx");
    write_file(&rnnt, &[1u8; 100]);
    write_file(&e2e, &[2u8; 200]);
    write_file(&ml, &[3u8; 300]);
    write_file(&legacy, &[4u8; 400]);

    let report = prune_optimized_cache(dir, false).unwrap();
    assert_eq!(report.kept.len(), 2);
    assert!(report.kept.contains(&rnnt));
    assert!(report.kept.contains(&e2e));
    assert_eq!(report.removed.len(), 2);
    assert!(rnnt.exists());
    assert!(e2e.exists(), "installed head's graph must survive cache-gc");
    assert!(!ml.exists());
    assert!(!legacy.exists());
    assert_eq!(report.freed_bytes, 300 + 400);
}

#[test]
fn test_prune_optimized_cache_keeps_ctc_head_dotted_stem() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    // ml_ctc-only install: dotted stem (`multilingual_ctc.int8`) must
    // survive `file_stem` handling on the keep path.
    for f in ModelVariant::MlCtc.prequantized_files() {
        write_file(&dir.join(f), f.as_bytes());
    }
    let cache = dir.join("optimized_cache");
    let keep = content_cache_path(&cache, &dir.join("multilingual_ctc.int8.onnx"));
    let zombie = cache.join("v3_rnnt_encoder_int8_optimized.ort");
    write_file(&keep, &[1u8; 100]);
    write_file(&zombie, &[2u8; 200]);

    let report = prune_optimized_cache(dir, false).unwrap();
    assert_eq!(report.kept, vec![keep.clone()]);
    assert_eq!(report.removed, vec![zombie.clone()]);
    assert!(keep.exists());
    assert!(!zombie.exists());
    assert_eq!(report.freed_bytes, 200);
}

#[test]
fn test_prune_optimized_cache_keeps_fp32_only_head() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    // FP32-only install (no INT8): the FP32 stem's graph is the one a
    // cache-miss rebuild would write, so keep it.
    write_file(&dir.join(ModelVariant::Rnnt.encoder_file()), b"stub");
    let cache = dir.join("optimized_cache");
    let keep = content_cache_path(&cache, &dir.join("v3_rnnt_encoder.onnx"));
    let zombie = cache.join("v3_rnnt_encoder_int8_optimized.ort");
    write_file(&keep, &[1u8; 100]);
    write_file(&zombie, &[2u8; 200]);

    let report = prune_optimized_cache(dir, false).unwrap();
    assert_eq!(report.kept, vec![keep.clone()]);
    assert!(keep.exists());
    assert!(!zombie.exists(), "no INT8 encoder installed: zombie");
}

#[test]
fn test_prune_optimized_cache_keeps_manifest_named_encoder() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    // Manifest install with a custom encoder basename; the engine loads
    // that name, so its graph must survive alongside a hardcoded head.
    write_file(
        &dir.join("manifest.toml"),
        br#"architecture = "ml_ctc"
[files]
encoder = "custom_enc.onnx"
encoder_int8 = "custom_enc_int8.onnx"
vocab = "custom_vocab.txt"
"#,
    );
    write_file(&dir.join("custom_enc_int8.onnx"), b"stub");
    for f in ModelVariant::Rnnt.prequantized_files() {
        write_file(&dir.join(f), f.as_bytes());
    }
    let cache = dir.join("optimized_cache");
    let custom = content_cache_path(&cache, &dir.join("custom_enc_int8.onnx"));
    let rnnt = content_cache_path(&cache, &dir.join("v3_rnnt_encoder_int8.onnx"));
    write_file(&custom, &[1u8; 100]);
    write_file(&rnnt, &[2u8; 200]);

    let report = prune_optimized_cache(dir, false).unwrap();
    assert_eq!(report.kept.len(), 2);
    assert!(report.kept.contains(&custom));
    assert!(report.kept.contains(&rnnt));
    assert!(
        custom.exists(),
        "manifest-named graph must survive cache-gc"
    );
    assert!(rnnt.exists());
    assert!(report.removed.is_empty());
}

#[test]
fn test_prune_optimized_cache_dry_run_does_not_delete() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    for f in ModelVariant::Rnnt.prequantized_files() {
        write_file(&dir.join(f), f.as_bytes());
    }
    let cache = dir.join("optimized_cache");
    let zombie = cache.join("v3_rnnt_encoder_optimized.onnx");
    write_file(&zombie, &[9u8; 50]);

    let report = prune_optimized_cache(dir, true).unwrap();
    assert!(report.dry_run);
    assert_eq!(report.removed.len(), 1);
    assert!(zombie.exists(), "dry_run must not delete");
    assert_eq!(report.freed_bytes, 50);
}

#[test]
fn test_prune_optimized_cache_no_head_leaves_cache() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let cache = dir.join("optimized_cache");
    let zombie = cache.join("v3_rnnt_encoder_optimized.onnx");
    write_file(&zombie, b"x");

    let report = prune_optimized_cache(dir, false).unwrap();
    assert!(report.kept.is_empty());
    assert!(report.removed.is_empty());
    assert!(zombie.exists());
}

#[test]
fn test_prune_optimized_cache_dir_prunes_explicit_dir() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    // Active rnnt INT8 install with a relocated cache (e.g. systemd
    // CacheDirectory) outside the model dir.
    for f in ModelVariant::Rnnt.prequantized_files() {
        write_file(&dir.join(f), f.as_bytes());
    }
    let cache = tmp.path().join("relocated_cache");
    let keep = content_cache_path(&cache, &dir.join("v3_rnnt_encoder_int8.onnx"));
    let zombie = cache.join("v3_e2e_rnnt_encoder_int8_optimized.ort");
    write_file(&keep, &[1u8; 100]);
    write_file(&zombie, &[2u8; 200]);

    let report = prune_optimized_cache_dir(&cache, dir, false).unwrap();
    assert_eq!(report.kept, vec![keep.clone()]);
    assert_eq!(report.removed, vec![zombie.clone()]);
    assert!(keep.exists());
    assert!(!zombie.exists());

    // A missing explicit cache dir is a no-op, same as the default path.
    let report = prune_optimized_cache_dir(&tmp.path().join("absent"), dir, false).unwrap();
    assert!(report.kept.is_empty());
    assert!(report.removed.is_empty());
}

#[test]
fn test_prune_coreml_cache_keeps_current_version_drops_stale() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let root = dir.join("coreml_cache");
    // Current-version dir (must survive) with a compiled bundle inside.
    let keep = root.join(coreml_cache_version_dir());
    write_file(
        &keep.join("0_static_mlprogram").join("model.mil"),
        &[1u8; 100],
    );
    // A sibling from a different ORT version (stale).
    let stale_ver = root.join("ort-1");
    write_file(&stale_ver.join("model.txt"), &[2u8; 200]);
    // Legacy pre-versioning hash dirs written directly under coreml_cache/.
    let legacy_a = root.join("3716630788028257604");
    write_file(&legacy_a.join("model.txt"), &[3u8; 300]);
    let legacy_b = root.join("9059995084551308172");
    write_file(&legacy_b.join("weights.bin"), &[4u8; 400]);
    // Stray file must be left alone.
    write_file(&root.join("README"), b"keep me");

    let report = prune_coreml_cache(dir, false).unwrap();
    assert_eq!(report.kept, Some(keep.clone()));
    assert_eq!(report.removed.len(), 3);
    assert!(keep.exists());
    assert!(!stale_ver.exists());
    assert!(!legacy_a.exists());
    assert!(!legacy_b.exists());
    assert!(root.join("README").exists());
    assert_eq!(report.freed_bytes, 200 + 300 + 400);
}

#[test]
fn test_prune_coreml_cache_dry_run_does_not_delete() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let legacy = dir.join("coreml_cache").join("17316774604053445175");
    write_file(&legacy.join("model.txt"), &[7u8; 512]);

    let report = prune_coreml_cache(dir, true).unwrap();
    assert!(report.dry_run);
    assert!(report.kept.is_none());
    assert_eq!(report.removed.len(), 1);
    assert!(legacy.exists(), "dry_run must not delete");
    assert_eq!(report.freed_bytes, 512);
}

#[test]
fn test_prune_coreml_cache_absent_dir_is_noop() {
    let tmp = tempfile::tempdir().unwrap();
    let report = prune_coreml_cache(tmp.path(), false).unwrap();
    assert!(report.kept.is_none());
    assert!(report.removed.is_empty());
    assert_eq!(report.freed_bytes, 0);
}

#[test]
fn test_dedupe_hardlinks_exact_copies() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let a = dir.join("a.onnx");
    let b = dir.join("sub").join("b.onnx");
    let payload = b"identical-payload-bytes";
    write_file(&a, payload);
    write_file(&b, payload);
    // Distinct file must not be linked.
    write_file(&dir.join("c.onnx"), b"different");

    let report = dedupe_model_dir(dir, false).unwrap();
    assert_eq!(report.groups, 1);
    assert_eq!(report.hardlinked, 1);
    // `same_file` is inode-based and always returns false on stable Windows.
    #[cfg(unix)]
    assert!(same_file(&a, &b).unwrap());
    assert_eq!(std::fs::read(&a).unwrap(), payload);
    assert_eq!(std::fs::read(&b).unwrap(), payload);
    assert_eq!(std::fs::read(dir.join("c.onnx")).unwrap(), b"different");
}

#[test]
fn test_dedupe_dry_run_no_hardlink() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let a = dir.join("a.bin");
    let b = dir.join("b.bin");
    write_file(&a, b"same");
    write_file(&b, b"same");

    let report = dedupe_model_dir(dir, true).unwrap();
    assert!(report.dry_run);
    assert_eq!(report.hardlinked, 1);
    assert!(!same_file(&a, &b).unwrap());
}

#[test]
fn test_source_hash_rejects_external_marker_across_read_boundaries() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("encoder.onnx");
    for offset in [0, 65530, 65535, 65536, 65540, 65542, 131070] {
        let mut bytes = vec![0; offset];
        bytes.extend_from_slice(b"location");
        write_file(&path, &bytes);
        assert_eq!(
            optimized_source_hash(&path).unwrap(),
            None,
            "offset {offset}"
        );
    }
    write_file(&path, &[42; 131080]);
    assert_eq!(
        optimized_source_hash(&path).unwrap().unwrap(),
        sha256_file_streaming(&path).unwrap()
    );
}

#[test]
fn test_prune_optimized_cache_keeps_configurations_but_reclaims_old_content_and_legacy() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let encoder = dir.join(ModelVariant::Rnnt.encoder_int8_file());
    write_file(&encoder, b"original");
    let cache = dir.join("optimized_cache");
    let old = content_cache_path(&cache, &encoder);
    write_file(&old, b"old");
    write_file(&encoder, b"updated");
    let hash = optimized_source_hash(&encoder).unwrap().unwrap();
    let first = cache.join(optimized_content_basename(&hash, &"a".repeat(64)));
    let second = cache.join(optimized_content_basename(&hash, &"b".repeat(64)));
    let legacy = cache.join("v3_rnnt_encoder_int8_optimized.ort");
    for path in [&first, &second, &legacy] {
        write_file(path, b"graph");
    }
    let dry = prune_optimized_cache(dir, true).unwrap();
    assert_eq!(dry.kept.len(), 2);
    assert_eq!(dry.removed.len(), 2);
    assert!(old.exists() && legacy.exists());
    let report = prune_optimized_cache(dir, false).unwrap();
    assert_eq!(report.kept.len(), 2);
    assert!(first.exists() && second.exists());
    assert!(!old.exists() && !legacy.exists());
}

#[test]
fn test_versioned_model_pack_refuses_legacy_cache_mutation() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let cache = root.join("optimized_cache");
    std::fs::create_dir(&cache).unwrap();
    let orphan = cache.join("orphan.onnx");
    std::fs::write(&orphan, b"unmodified cache").unwrap();
    std::fs::write(
        root.join(crate::model::MANIFEST_FILE),
        "schema_version = 1\nfamily = \"gigaam_v4\"\n",
    )
    .unwrap();
    let coreml_stale = root.join("coreml_cache").join("stale");
    std::fs::create_dir_all(&coreml_stale).unwrap();
    assert!(prune_optimized_cache_dir(&cache, root, false).is_err());
    assert!(prune_coreml_cache(root, false).is_err());
    assert!(coreml_stale.is_dir());
    assert!(dedupe_model_dir(root, false).is_err());
    assert_eq!(std::fs::read(orphan).unwrap(), b"unmodified cache");
}
