use super::cache::{
    build_source_key, cached_meta_path, cached_model_path, compiled_cache_dir, copy_dir_recursive,
    current_source_key, meta_matches, populate_cache,
};
use super::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

mod objc_lifetime;

fn package_path() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME set");
    PathBuf::from(home).join(".gigastt/models/ane/gigaam_v3_encoder_768.mlpackage")
}

// ---- pure cache-logic tests (no Core ML / hardware) ------------------

#[test]
fn cache_paths_are_sibling_compiled_cache_dir() {
    let pkg = Path::new("/models/ane/gigaam_v3_encoder_768.mlpackage");
    assert_eq!(
        compiled_cache_dir(pkg),
        PathBuf::from("/models/ane/compiled_cache")
    );
    assert_eq!(
        cached_model_path(pkg),
        PathBuf::from("/models/ane/compiled_cache/gigaam_v3_encoder_768.mlmodelc")
    );
    assert_eq!(
        cached_meta_path(pkg),
        PathBuf::from("/models/ane/compiled_cache/gigaam_v3_encoder_768.mlmodelc.meta")
    );
}

#[test]
fn build_source_key_changes_with_size_and_os() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let pkg = tmp.path().join("pkg.mlpackage");
    fs::create_dir_all(pkg.join("Data")).unwrap();
    fs::write(pkg.join("Data").join("weight.bin"), b"abc").unwrap();

    let key_a = build_source_key(&pkg, "26.1").expect("key a");

    // Same source + OS -> identical key (hit).
    let key_a2 = build_source_key(&pkg, "26.1").expect("key a2");
    assert_eq!(key_a, key_a2, "same source+OS must produce the same key");

    // Different OS version -> different key (miss after OS update).
    let key_os = build_source_key(&pkg, "27.0").expect("key os");
    assert_ne!(key_a, key_os, "OS version must be part of the key");

    // Larger source (changed byte size) -> different key (miss).
    fs::write(pkg.join("Data").join("weight.bin"), b"abcdef").unwrap();
    let key_size = build_source_key(&pkg, "26.1").expect("key size");
    assert_ne!(key_a, key_size, "changed source size must change the key");
}

#[test]
fn meta_matches_only_on_exact_key() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let meta = tmp.path().join("model.mlmodelc.meta");

    // Missing sidecar -> miss.
    assert!(!meta_matches(&meta, "size=10 mtime_ns=5 os=26.1"));

    fs::write(&meta, "size=10 mtime_ns=5 os=26.1\n").unwrap();
    // Trailing newline is tolerated (trimmed) -> hit.
    assert!(meta_matches(&meta, "size=10 mtime_ns=5 os=26.1"));
    // Any difference -> miss.
    assert!(!meta_matches(&meta, "size=11 mtime_ns=5 os=26.1"));
    assert!(!meta_matches(&meta, "size=10 mtime_ns=5 os=27.0"));
}

#[test]
fn copy_dir_recursive_reproduces_tree() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let src = tmp.path().join("src");
    let dst = tmp.path().join("dst");
    fs::create_dir_all(src.join("nested")).unwrap();
    fs::write(src.join("top.bin"), b"top").unwrap();
    fs::write(src.join("nested").join("inner.bin"), b"inner").unwrap();

    copy_dir_recursive(&src, &dst).expect("copy");

    assert_eq!(fs::read(dst.join("top.bin")).unwrap(), b"top");
    assert_eq!(
        fs::read(dst.join("nested").join("inner.bin")).unwrap(),
        b"inner"
    );
}

#[test]
fn populate_cache_atomically_places_model_and_sidecar() {
    // No Core ML: stage a fake "compiled" dir + a fake source package, then
    // assert populate_cache mirrors it into compiled_cache/ with a sidecar
    // whose key matches the current source key (so a subsequent hit works).
    let tmp = tempfile::tempdir().expect("tempdir");
    let pkg = tmp.path().join("gigaam_v3_encoder_768.mlpackage");
    fs::create_dir_all(&pkg).unwrap();
    fs::write(pkg.join("Manifest.json"), b"{}").unwrap();

    let temp_compiled = tmp.path().join("temp.mlmodelc");
    fs::create_dir_all(temp_compiled.join("model")).unwrap();
    fs::write(temp_compiled.join("coremldata.bin"), b"compiled").unwrap();
    fs::write(temp_compiled.join("model").join("net.bin"), b"net").unwrap();

    populate_cache(&pkg, &temp_compiled).expect("populate");

    let cached = cached_model_path(&pkg);
    assert!(cached.is_dir(), "cached model dir must exist");
    assert_eq!(
        fs::read(cached.join("coremldata.bin")).unwrap(),
        b"compiled"
    );
    assert_eq!(
        fs::read(cached.join("model").join("net.bin")).unwrap(),
        b"net"
    );

    // The sidecar must match the current source key -> meta_matches hit.
    let key = current_source_key(&pkg).expect("source key");
    assert!(
        meta_matches(&cached_meta_path(&pkg), &key),
        "sidecar key must match the current source key after populate_cache"
    );

    // No staging dirs left behind.
    let leftover: Vec<_> = fs::read_dir(compiled_cache_dir(&pkg))
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with(".staging"))
        .collect();
    assert!(leftover.is_empty(), "no .staging dirs must remain");
}

fn ref_dir() -> PathBuf {
    PathBuf::from("/tmp/gigaam-ane-spike/bridge_ref")
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).expect("read f32 file");
    assert_eq!(bytes.len() % 4, 0, "f32 file length not a multiple of 4");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn read_shapes(path: &Path) -> (Vec<usize>, Vec<usize>) {
    let txt = fs::read_to_string(path).expect("read shapes.txt");
    let mut in_shape = Vec::new();
    let mut out_shape = Vec::new();
    for line in txt.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("in") => in_shape = it.map(|s| s.parse().unwrap()).collect(),
            Some("out") => out_shape = it.map(|s| s.parse().unwrap()).collect(),
            _ => {}
        }
    }
    (in_shape, out_shape)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// GO/NO-GO smoke test. Touches the filesystem + ANE, so `#[ignore]`d like the
/// e2e tests; run manually:
///   cargo test -p gigastt-core --features ane bridge -- --ignored --nocapture
#[test]
#[ignore = "requires the 768 bucket .mlpackage + Python bridge_ref/; runs on ANE"]
fn bridge_loads_predicts_matches_python_reference() {
    let pkg = package_path();
    let refd = ref_dir();
    if !pkg.exists() {
        eprintln!("SKIP: missing package {pkg:?} (run convert_gigaam_ane.py --buckets 768)");
        return;
    }
    if !refd.join("shapes.txt").exists() {
        eprintln!("SKIP: missing {refd:?}/shapes.txt (run dump_bridge_ref.py)");
        return;
    }

    let (in_shape, ref_out_shape) = read_shapes(&refd.join("shapes.txt"));
    let mel = read_f32(&refd.join("mel_in.f32"));
    let ref_out = read_f32(&refd.join("encoded_ref.f32"));
    assert_eq!(
        in_shape,
        vec![1, 64, 768],
        "unexpected reference input shape"
    );

    let model = compile_and_load(&pkg, true).expect("compile_and_load");

    let (out, out_shape) =
        predict_f32(&model, "mel", &mel, &in_shape, "encoded").expect("predict_f32");

    println!("out_shape={out_shape:?} ref_out_shape={ref_out_shape:?}");
    assert_eq!(
        out_shape, ref_out_shape,
        "output shape mismatch vs Python ref"
    );
    assert_eq!(
        out.len(),
        ref_out.len(),
        "output length mismatch vs Python ref"
    );
    assert!(
        out.iter().all(|v| v.is_finite()),
        "output has non-finite values"
    );

    let cos = cosine(&out, &ref_out);
    let max_abs = out
        .iter()
        .zip(ref_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("cosine={cos:.6}  max_abs={max_abs:.6}");
    assert!(cos > 0.999, "cosine {cos:.6} <= 0.999 vs Python reference");

    // RTFx: warm 4x, then time ~12 predicts. audio_secs = N/100 (mel hop 10ms).
    for _ in 0..4 {
        let _ = predict_f32(&model, "mel", &mel, &in_shape, "encoded").expect("warm predict");
    }
    let iters = 12;
    let mut times_ms = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let _ = predict_f32(&model, "mel", &mel, &in_shape, "encoded").expect("timed predict");
        times_ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ms = times_ms[times_ms.len() / 2];
    let audio_secs = in_shape[2] as f64 / 100.0;
    let rtfx = audio_secs / (median_ms / 1000.0);
    println!("median_ms={median_ms:.3}  audio_secs={audio_secs:.3}  RTFx={rtfx:.1}");
}

/// Cold-start cache GO/NO-GO. Loads the SAME package twice in one process:
/// the FIRST load compiles (~20 s) and populates the cache; the SECOND load
/// is a cache hit (no compile). Asserts the 2nd load is dramatically faster
/// AND produces byte-identical output (caching must not change results).
/// Touches the filesystem + ANE, so `#[ignore]`d; run manually:
///   cargo test -p gigastt-core --features ane bridge_disk_cache -- --ignored --nocapture
///
/// Hermetic: rather than wipe the developer's real warm cache, this
/// symlinks the real `.mlpackage` into a fresh tempdir and compiles from
/// there, so the cache derives to `<tmp>/compiled_cache/` (the cache path is
/// `package.parent()/compiled_cache`). The real cache is never touched; the
/// tempdir is removed on `TempDir` drop.
#[test]
#[ignore = "requires the 768 bucket .mlpackage; compiles on ANE (~20s first load)"]
fn bridge_disk_cache_skips_recompile_and_preserves_output() {
    let real_pkg = package_path();
    if !real_pkg.exists() {
        eprintln!("SKIP: missing package {real_pkg:?} (run convert_gigaam_ane.py --buckets 768)");
        return;
    }

    // Hermetic workspace: symlink the real package into a fresh tempdir so
    // the cache derives to <tmp>/compiled_cache/, leaving the real
    // ~/.gigastt/models/ane/compiled_cache/ untouched (TempDir cleans up).
    let tmp = tempfile::tempdir().expect("tempdir");
    let pkg = tmp.path().join("gigaam_v3_encoder_768.mlpackage");
    std::os::unix::fs::symlink(&real_pkg, &pkg).expect("symlink package into tempdir");

    let real_cache = compiled_cache_dir(&real_pkg);
    let real_cache_existed = real_cache.exists();

    // The cache must start empty inside the hermetic tempdir.
    let cache_dir = compiled_cache_dir(&pkg);
    assert!(!cached_model_path(&pkg).exists(), "cache must start empty");
    assert_eq!(
        cache_dir,
        tmp.path().join("compiled_cache"),
        "cache must derive inside the tempdir, not the real cache"
    );

    // Fixed input: a deterministic ramp over the 768 bucket's mel shape.
    let in_shape = vec![1usize, 64, 768];
    let n: usize = in_shape.iter().product();
    let mel: Vec<f32> = (0..n).map(|i| (i as f32 % 17.0) * 0.01).collect();

    // First load: cold compile + cache populate.
    let t0 = Instant::now();
    let model1 = compile_and_load(&pkg, true).expect("first compile_and_load");
    let first_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let (out1, shape1) =
        predict_f32(&model1, "mel", &mel, &in_shape, "encoded").expect("predict 1");

    assert!(
        cached_model_path(&pkg).exists(),
        "first load must populate the disk cache"
    );
    let key = current_source_key(&pkg).expect("source key");
    assert!(
        meta_matches(&cached_meta_path(&pkg), &key),
        "sidecar must match the current source key after first load"
    );

    // Second load: cache hit, no compile.
    let t1 = Instant::now();
    let model2 = compile_and_load(&pkg, true).expect("second compile_and_load");
    let second_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let (out2, shape2) =
        predict_f32(&model2, "mel", &mel, &in_shape, "encoded").expect("predict 2");

    println!("cold_start_first_ms={first_ms:.1}  cache_hit_second_ms={second_ms:.1}");

    // The cache hit must be dramatically faster than the cold compile.
    assert!(
        first_ms > 5_000.0,
        "expected cold compile > 5s, got {first_ms:.1} ms"
    );
    assert!(
        second_ms < 2_000.0,
        "expected cache-hit load < 2s, got {second_ms:.1} ms"
    );
    assert!(
        second_ms < first_ms / 2.0,
        "cache hit ({second_ms:.1} ms) must be much faster than cold ({first_ms:.1} ms)"
    );

    // Caching must not change results: byte-identical output both times.
    assert_eq!(shape1, shape2, "output shape changed across cache hit");
    assert_eq!(
        out1.len(),
        out2.len(),
        "output length changed across cache hit"
    );
    assert_eq!(
        out1.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
        out2.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
        "cache hit must produce byte-identical output"
    );

    // Hermetic guarantee: caching happened in the tempdir, so the real
    // cache must be in the exact state we found it (this test never created
    // or wiped the developer's warm ~/.gigastt cache).
    assert_eq!(
        real_cache.exists(),
        real_cache_existed,
        "the real cache dir must be untouched by this test"
    );
}
