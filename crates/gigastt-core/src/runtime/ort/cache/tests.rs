use super::*;
use std::io::Write;

fn write_file(path: &Path, bytes: &[u8]) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(bytes).unwrap();
}

#[test]
fn test_cache_settings_include_runtime_threads_and_prepacking() {
    let reference = settings_hash("runtime-a", 1, false, "avx2");
    assert_ne!(reference, settings_hash("runtime-a", 1, false, "avx512"));
    assert_ne!(reference, settings_hash("runtime-b", 1, false, "avx2"));
    assert_ne!(reference, settings_hash("runtime-a", 2, false, "avx2"));
    assert_ne!(reference, settings_hash("runtime-a", 1, true, "avx2"));
    assert_eq!(reference, settings_hash("runtime-a", 0, false, "avx2"));
}

#[test]
fn test_pending_cache_is_private_and_cleans_up_failed_writes() {
    let tmp = tempfile::tempdir().unwrap();
    let cache = tmp.path().join("encoder_optimized.ort");
    write_file(&cache, b"previous graph");
    let first = PendingCache::new(&cache).unwrap();
    let second = PendingCache::new(&cache).unwrap();
    assert_ne!(first.0, second.0);
    write_file(&first.0, b"incomplete write");
    assert_eq!(std::fs::read(&cache).unwrap(), b"previous graph");
    assert_eq!(std::fs::metadata(&second.0).unwrap().len(), 0);

    drop(first);
    drop(second);

    assert_eq!(std::fs::read(&cache).unwrap(), b"previous graph");
    assert_eq!(std::fs::read_dir(tmp.path()).unwrap().count(), 1);
}

#[test]
fn test_usable_optimized_cache_dir_creates_and_returns_dir() {
    let tmp = tempfile::tempdir().unwrap();
    let cache_dir = tmp.path().join("optimized_cache");
    assert!(!cache_dir.exists());
    assert_eq!(
        usable_optimized_cache_dir(&cache_dir),
        Some(cache_dir.clone())
    );
    assert!(cache_dir.is_dir());
    // The write probe must not leave files behind.
    assert_eq!(std::fs::read_dir(&cache_dir).unwrap().count(), 0);
}

#[test]
fn test_usable_optimized_cache_dir_preserves_existing_probe() {
    let tmp = tempfile::tempdir().unwrap();
    let cache_dir = tmp.path().join("optimized_cache");
    let existing_probe = cache_dir.join(format!(".write-probe-{}", std::process::id()));
    write_file(&existing_probe, b"another caller's probe");

    assert_eq!(
        usable_optimized_cache_dir(&cache_dir),
        Some(cache_dir.clone())
    );
    assert_eq!(
        std::fs::read(&existing_probe).unwrap(),
        b"another caller's probe"
    );
    assert_eq!(std::fs::read_dir(&cache_dir).unwrap().count(), 1);
}

#[test]
fn test_usable_optimized_cache_dir_uncreatable_returns_none() {
    let tmp = tempfile::tempdir().unwrap();
    // A path under a regular file fails `create_dir_all` (ENOTDIR) on
    // every platform and privilege level — no reliance on permissions.
    let blocker = tmp.path().join("blocker");
    write_file(&blocker, b"not a dir");
    let cache_dir = blocker.join("optimized_cache");
    assert_eq!(usable_optimized_cache_dir(&cache_dir), None);
}

#[cfg(unix)]
#[test]
fn test_usable_optimized_cache_dir_read_only_returns_none() {
    // Root bypasses directory write permission bits, so the probe would
    // succeed and this test cannot exercise the read-only path.
    if unsafe { libc::geteuid() } == 0 {
        eprintln!(
            "test_usable_optimized_cache_dir_read_only_returns_none: running as root, \
             skipping (root bypasses write permission bits) — vacuous pass"
        );
        return;
    }
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let cache_dir = tmp.path().join("optimized_cache");
    std::fs::create_dir_all(&cache_dir).unwrap();
    std::fs::set_permissions(&cache_dir, std::fs::Permissions::from_mode(0o555)).unwrap();
    // An existing-but-read-only dir passes `create_dir_all` yet cannot
    // hold the cache: the write probe must degrade to `None`.
    // Capture the result and restore permissions BEFORE asserting, so a
    // failing assert does not strand a 0o555 dir that tempdir's Drop
    // cannot remove (masking the real failure).
    let result = usable_optimized_cache_dir(&cache_dir);
    std::fs::set_permissions(&cache_dir, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert_eq!(result, None);
}

#[test]
fn test_usable_optimized_cache_dir_concurrent_probes() {
    // Each concurrent caller must see a usable directory and remove only
    // its own probe, without Windows sharing violations or leftovers.
    let tmp = tempfile::tempdir().unwrap();
    let cache_dir = tmp.path().join("optimized_cache");
    std::fs::create_dir_all(&cache_dir).unwrap();
    let barrier = std::sync::Barrier::new(4);
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                s.spawn(|| {
                    barrier.wait();
                    usable_optimized_cache_dir(&cache_dir)
                })
            })
            .collect();
        for h in handles {
            assert_eq!(h.join().unwrap(), Some(cache_dir.clone()));
        }
    });
    let leftover: Vec<_> = std::fs::read_dir(&cache_dir).unwrap().collect();
    assert!(leftover.is_empty(), "probe left files behind: {leftover:?}");
}

#[test]
fn test_source_hash_is_content_addressed() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("model.onnx");
    let cache = OptimizedCache::new(tmp.path().join("cache"));
    write_file(&model, b"first bytes");
    let first = cache.source_hash(&model, None).unwrap().unwrap();
    assert_eq!(first.len(), 64);
    assert_eq!(
        cache.source_hash(&model, Some(&first)).unwrap(),
        Some(first.clone())
    );
    write_file(&model, b"second bytes");
    let second = cache.source_hash(&model, None).unwrap().unwrap();
    assert_ne!(first, second);
    assert!(!cache.directory.exists());
}

#[test]
fn test_staged_graph_publishes_and_cleans_staging() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("model.onnx");
    write_file(&model, b"source model");
    let hash = crate::model::optimized_source_hash(&model)
        .unwrap()
        .unwrap();
    let settings = settings_hash("runtime-a", 1, false, "avx2");
    let destination = tmp
        .path()
        .join(crate::model::optimized_content_basename(&hash, &settings));
    assert_eq!(
        destination.file_name().unwrap().to_str().unwrap(),
        format!("{hash}-{settings}_optimized.ort"),
    );
    write_file(&destination, b"previous graph");
    let staged = StagedGraph::new(&destination, &hash).unwrap();
    let staging = staged.path().to_path_buf();
    write_file(&staging, b"complete graph");
    assert_eq!(std::fs::read(&destination).unwrap(), b"previous graph");
    staged.publish(&model);
    assert_eq!(std::fs::read(&destination).unwrap(), b"complete graph");
    drop(staged);
    assert!(!staging.exists());
}

#[test]
fn test_staged_graph_discards_when_source_changes() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("model.onnx");
    write_file(&model, b"source model");
    let hash = crate::model::optimized_source_hash(&model)
        .unwrap()
        .unwrap();
    let destination = tmp.path().join("graph.ort");
    write_file(&destination, b"previous graph");
    let staged = StagedGraph::new(&destination, &hash).unwrap();
    let staging = staged.path().to_path_buf();
    write_file(&staging, b"complete graph");
    write_file(&model, b"changed source");
    staged.publish(&model);
    drop(staged);
    assert_eq!(std::fs::read(&destination).unwrap(), b"previous graph");
    assert!(!staging.exists());
}
