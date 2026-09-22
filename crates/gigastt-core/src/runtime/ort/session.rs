use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use ort::session::Session;

use crate::runtime::{
    error::RuntimeError, factory::Runtime, session::RuntimeSession, tensor::Tensor,
};

use super::{factory::OrtExecutionProvider, tensor::value_to_tensor};

/// `ort`-backed runtime that loads sessions for a specific execution provider.
pub struct OrtRuntime {
    intra_threads: usize,
    provider: OrtExecutionProvider,
    secondary_cpu: bool,
    prepacked: Option<Arc<ort::session::builder::PrepackedWeights>>,
    optimized_cache_dir: Option<std::path::PathBuf>,
    /// Once-per-runtime result of the cache-dir writability probe. Pool
    /// triplets load concurrently on this shared runtime, so probing per
    /// `load_session` would repeat the probe (and its warning) pool-size
    /// times at every boot. Admin reload builds a fresh `OrtRuntime`, so
    /// re-probing on reload is preserved.
    usable_cache_dir: OnceLock<Option<std::path::PathBuf>>,
    /// Keep pool triplets from optimizing the same encoder concurrently.
    cache_load: parking_lot::Mutex<()>,
}

impl OrtRuntime {
    pub(crate) fn new(
        intra_threads: usize,
        provider: OrtExecutionProvider,
        secondary_cpu: bool,
        prepacked: Option<Arc<ort::session::builder::PrepackedWeights>>,
        optimized_cache_dir: Option<std::path::PathBuf>,
    ) -> Self {
        Self {
            intra_threads,
            provider,
            secondary_cpu,
            prepacked,
            optimized_cache_dir,
            usable_cache_dir: OnceLock::new(),
            cache_load: parking_lot::Mutex::new(()),
        }
    }
}

fn load_failed(path: &Path, e: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::LoadFailed {
        path: path.into(),
        message: e.to_string(),
    }
}

/// Pure freshness decision for an ORT optimized-graph cache entry: the cache
/// must be non-empty and no older than the source model it was derived from.
/// An ORT/binary upgrade typically rewrites the source model install or fails
/// to load the stale graph — the load-failure fallback covers the latter.
fn cache_is_fresh(
    cache_len: u64,
    cache_mtime: std::time::SystemTime,
    source_mtime: std::time::SystemTime,
) -> bool {
    cache_len > 0 && cache_mtime >= source_mtime
}

/// Filesystem wiring for [`cache_is_fresh`]: any metadata error (missing cache,
/// missing source, unreadable mtime) means "not fresh" so the caller falls
/// back to loading the source model and rewriting the cache.
fn optimized_cache_is_fresh(cache_path: &Path, source_path: &Path) -> bool {
    let (Ok(cache), Ok(source)) = (
        std::fs::metadata(cache_path),
        std::fs::metadata(source_path),
    ) else {
        return false;
    };
    match (cache.modified(), source.modified()) {
        (Ok(cache_mtime), Ok(source_mtime)) => {
            cache_is_fresh(cache.len(), cache_mtime, source_mtime)
        }
        _ => false,
    }
}

/// Path of the optimized graph ORT writes for `model_path` under `cache_dir`.
/// Uses the same basename rule as `model::cache::optimized_cache_basename` so
/// the file we write here is exactly the one `cache-gc` keeps.
fn optimized_cache_path(cache_dir: &Path, model_path: &Path) -> std::path::PathBuf {
    let basename = crate::model::optimized_cache_basename(model_path)
        .unwrap_or_else(|| "encoder_optimized.ort".into());
    cache_dir.join(basename)
}

/// Stage beside the destination so rename publishes a complete graph atomically.
/// Exclusive creation also separates writers in different runtimes/processes.
struct PendingCache(std::path::PathBuf);

impl PendingCache {
    fn new(cache_path: &Path) -> std::io::Result<Self> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);

        loop {
            let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            let mut path = cache_path.as_os_str().to_owned();
            path.push(format!(".partial.{}.{id}", std::process::id()));
            let path = std::path::PathBuf::from(path);
            match std::fs::File::create_new(&path) {
                Ok(_) => return Ok(Self(path)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
    }
}

impl Drop for PendingCache {
    fn drop(&mut self) {
        // Best effort on errors/unwind; after publication the path is absent.
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Decide whether the ORT optimized-graph cache under `cache_dir` is usable:
/// the directory must be creatable and writable. Read-only model installs
/// (e.g. systemd `ProtectSystem=strict` with models under `/usr/share`) make
/// either step fail with `EROFS`; boot must degrade to a cache-less load
/// (slower cold start, higher per-session RAM) instead of failing, so both
/// failures only warn and return `None`. Writability is probed by creating
/// and deleting a temp file — an existing-but-read-only dir passes
/// `create_dir_all` yet still cannot hold the cache.
fn usable_optimized_cache_dir(cache_dir: &Path) -> Option<std::path::PathBuf> {
    if let Err(e) = std::fs::create_dir_all(cache_dir) {
        tracing::warn!(
            path = %cache_dir.display(),
            error = %e,
            "encoder: optimized graph cache directory cannot be created; loading source model without the cache (slower cold start, higher per-session RAM)"
        );
        return None;
    }
    // Different runtimes can probe the same directory concurrently. Each
    // probe must own its file, including on Windows where deletion can make
    // a concurrent open fail with a sharing violation.
    match PendingCache::new(&cache_dir.join(".write-probe")) {
        Ok(_probe) => Some(cache_dir.to_path_buf()),
        Err(e) => {
            tracing::warn!(
                path = %cache_dir.display(),
                error = %e,
                "encoder: optimized graph cache directory is not writable; loading source model without the cache (slower cold start, higher per-session RAM)"
            );
            None
        }
    }
}

impl OrtRuntime {
    /// Assemble the session builder with prepacked weights, execution
    /// providers, and (CPU-only) thread counts. Cheap: no model parsing
    /// happens until `commit_from_file`, so the cache-miss fallback can
    /// simply build a second one.
    fn session_builder(
        &self,
        model_path: &Path,
        is_encoder: bool,
    ) -> Result<ort::session::builder::SessionBuilder, RuntimeError> {
        let mut builder = Session::builder().map_err(|e| load_failed(model_path, e))?;

        if let Some(prepacked) = self.prepacked.as_ref() {
            builder = builder
                .with_prepacked_weights(prepacked)
                .map_err(|e| load_failed(model_path, e))?;
        }

        let eps = self
            .provider
            .execution_providers(model_path, self.secondary_cpu);
        builder = builder
            .with_execution_providers(&eps)
            .map_err(|e| load_failed(model_path, e))?;

        if self.provider.is_cpu() {
            let intra_threads = if is_encoder {
                self.intra_threads.max(1)
            } else {
                1
            };
            builder = builder
                .with_intra_threads(intra_threads)
                .map_err(|e| load_failed(model_path, e))?;
            builder = builder
                .with_inter_threads(1)
                .map_err(|e| load_failed(model_path, e))?;
        }
        Ok(builder)
    }

    /// CPU-encoder fast path: when a fresh optimized graph from a previous
    /// boot exists, load the session from it and skip both re-optimizing the
    /// source model and re-serializing the cache (~224 MiB write per boot).
    /// Returns `None` when the fast path does not apply or the cached graph
    /// fails to load (the source load replaces the broken entry atomically).
    ///
    /// The cache is an ORT flatbuffer model (`.ort`), loaded with
    /// `session.use_memory_mapped_ort_model` +
    /// `session.use_ort_model_bytes_for_initializers`: sessions reference
    /// the encoder weights directly from a shared read-only file mapping
    /// instead of copying them into per-session anonymous memory, and the
    /// flatbuffer graph avoids retaining a parsed ModelProto object graph.
    /// Prepacking is disabled on this path — for this model it only
    /// duplicated the weights into per-session buffers (~145 MiB each) with
    /// no measurable RTF benefit.
    fn try_load_cached_encoder(&self, model_path: &Path) -> Option<Result<Session, RuntimeError>> {
        if !self.provider.is_cpu() {
            return None;
        }
        let cache_dir = self.optimized_cache_dir.as_ref()?;
        let cache_path = optimized_cache_path(cache_dir, model_path);
        if !optimized_cache_is_fresh(&cache_path, model_path) {
            return None;
        }
        let result = self
            .session_builder(model_path, true)
            .and_then(|mut builder| {
                builder = builder
                    .with_config_entry("session.load_model_format", "ORT")
                    .map_err(|e| load_failed(&cache_path, e))?;
                builder = builder
                    .with_config_entry("session.use_memory_mapped_ort_model", "1")
                    .map_err(|e| load_failed(&cache_path, e))?;
                builder = builder
                    .with_config_entry("session.use_ort_model_bytes_for_initializers", "1")
                    .map_err(|e| load_failed(&cache_path, e))?;
                builder = builder
                    .with_prepacking(false)
                    .map_err(|e| load_failed(&cache_path, e))?;
                builder
                    .commit_from_file(&cache_path)
                    .map_err(|e| load_failed(&cache_path, e))
            });
        match result {
            Ok(session) => {
                tracing::info!(
                    path = %cache_path.display(),
                    "encoder: loaded fresh optimized graph cache, skipped source model"
                );
                Some(Ok(session))
            }
            Err(e) => {
                tracing::warn!(
                    path = %cache_path.display(),
                    error = %e,
                    "encoder: optimized graph cache failed to load; falling back to the source model"
                );
                // Another runtime/process may have replaced this path since
                // our failed read. Only replace it with a complete new graph;
                // deleting here could unlink that writer's valid cache.
                None
            }
        }
    }
}

impl Runtime for OrtRuntime {
    fn load_session(
        &self,
        model_path: &Path,
        is_encoder: bool,
    ) -> Result<Box<dyn RuntimeSession>, RuntimeError> {
        // Check freshness under the same gate as publication. Waiting pool
        // triplets reuse the first writer's graph instead of re-optimizing it.
        let _cache_guard =
            (is_encoder && self.provider.is_cpu() && self.optimized_cache_dir.is_some())
                .then(|| self.cache_load.lock());
        if is_encoder && let Some(result) = self.try_load_cached_encoder(model_path) {
            let session = result?;
            return Ok(Box::new(OrtSession {
                session: Mutex::new(session),
            }));
        }

        let mut builder = self.session_builder(model_path, is_encoder)?;

        // Compute the cache path up front (probing writability once per
        // runtime via the OnceLock), then commit with the fallback: a
        // writable directory does not guarantee the ~224 MiB cache *file*
        // write succeeds (stale root-owned entry, ENOSPC mid-write), and ORT
        // treats that failure as fatal to `commit_from_file` — so on error
        // retry with a freshly built, cache-less builder (builders are cheap,
        // see `session_builder`).
        let cache_path = if self.provider.is_cpu() && is_encoder {
            self.usable_cache_dir
                .get_or_init(|| {
                    self.optimized_cache_dir
                        .as_deref()
                        .and_then(usable_optimized_cache_dir)
                })
                .as_ref()
                .map(|dir| optimized_cache_path(dir, model_path))
        } else {
            None
        };

        let pending_cache = cache_path.as_ref().and_then(|path| {
            match PendingCache::new(path) {
                Ok(pending) => Some(pending),
                Err(e) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %e,
                        "encoder: cannot stage optimized graph cache; loading source model without the cache"
                    );
                    None
                }
            }
        });

        if let Some(pending) = pending_cache.as_ref() {
            tracing::info!(
                path = %pending.0.display(),
                "encoder: loading source model and refreshing optimized graph cache"
            );
            builder = builder
                .with_optimized_model_path(&pending.0)
                .map_err(|e| load_failed(model_path, e))?;
            // Persist the optimized graph in ORT flatbuffer format so the
            // next boot can memory-map it (see `try_load_cached_encoder`).
            builder = builder
                .with_config_entry("session.save_model_format", "ORT")
                .map_err(|e| load_failed(model_path, e))?;
        }

        let session = match builder.commit_from_file(model_path) {
            Ok(session) => {
                if let (Some(pending), Some(cache_path)) = (&pending_cache, &cache_path)
                    && let Err(e) = std::fs::rename(&pending.0, cache_path)
                {
                    tracing::warn!(
                        path = %cache_path.display(),
                        error = %e,
                        "encoder: cannot publish optimized graph cache; keeping the source model session"
                    );
                }
                session
            }
            Err(e) => {
                let Some(pending) = pending_cache.as_ref() else {
                    return Err(load_failed(model_path, e));
                };
                tracing::warn!(
                    path = %pending.0.display(),
                    error = %e,
                    "encoder: optimized graph cache write failed; retrying without the cache (slower cold start, higher per-session RAM)"
                );
                self.session_builder(model_path, is_encoder)?
                    .commit_from_file(model_path)
                    .map_err(|e| load_failed(model_path, e))?
            }
        };
        Ok(Box::new(OrtSession {
            session: Mutex::new(session),
        }))
    }
}

/// `ort`-backed session wrapping a loaded ONNX model.
pub struct OrtSession {
    session: Mutex<Session>,
}

impl RuntimeSession for OrtSession {
    fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, RuntimeError> {
        let session_inputs: Vec<ort::session::SessionInputValue<'_>> = inputs
            .iter()
            .map(Tensor::as_ort_input)
            .collect::<Result<_, _>>()?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| RuntimeError::InferenceFailed("ort session mutex poisoned".into()))?;
        let outputs = session
            .run(&session_inputs[..])
            .map_err(|e| RuntimeError::InferenceFailed(e.to_string()))?;

        outputs
            .into_iter()
            .map(|(_name, value)| value_to_tensor(value))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::time::{Duration, SystemTime};

    fn write_file(path: &Path, bytes: &[u8]) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(bytes).unwrap();
    }

    fn write_identity_model(path: &Path) {
        // ONNX IR 8, opset 13: Identity(x: float[1]) -> y: float[1].
        write_file(path, b"\x08\x08\x3a\x40\x0a\x10\x0a\x01x\x12\x01y\x22\x08Identity\x12\x0acache-test\x5a\x0f\x0a\x01x\x12\x0a\x0a\x08\x08\x01\x12\x04\x0a\x02\x08\x01\x62\x0f\x0a\x01y\x12\x0a\x0a\x08\x08\x01\x12\x04\x0a\x02\x08\x01\x42\x02\x10\x0d");
    }

    fn cached_runtime(cache_dir: &Path) -> OrtRuntime {
        OrtRuntime::new(
            1,
            OrtExecutionProvider::Cpu,
            true,
            None,
            Some(cache_dir.to_path_buf()),
        )
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
    fn test_cache_invalid_source_leaves_no_partial_files() {
        let tmp = tempfile::tempdir().unwrap();
        let model = tmp.path().join("encoder.onnx");
        let cache_dir = tmp.path().join("cache");
        write_file(&model, b"invalid ONNX");
        assert!(
            cached_runtime(&cache_dir)
                .load_session(&model, true)
                .is_err()
        );
        assert_eq!(std::fs::read_dir(&cache_dir).unwrap().count(), 0);
    }

    #[test]
    fn test_cache_broken_graph_is_replaced_on_source_load() {
        let tmp = tempfile::tempdir().unwrap();
        let model = tmp.path().join("encoder.onnx");
        let cache_dir = tmp.path().join("cache");
        let cache = optimized_cache_path(&cache_dir, &model);
        write_identity_model(&model);
        write_file(&cache, b"invalid ORT");
        let runtime = cached_runtime(&cache_dir);
        assert!(runtime.try_load_cached_encoder(&model).is_none());
        // A failed reader must not delete a path another process can replace.
        assert_eq!(std::fs::read(&cache).unwrap(), b"invalid ORT");
        drop(runtime.load_session(&model, true).unwrap());
        assert!(runtime.try_load_cached_encoder(&model).unwrap().is_ok());
        assert_eq!(std::fs::read_dir(&cache_dir).unwrap().count(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn test_cache_refresh_replaces_file_without_truncating_readers() {
        use std::os::unix::fs::MetadataExt;

        let tmp = tempfile::tempdir().unwrap();
        let model = tmp.path().join("encoder.onnx");
        let cache_dir = tmp.path().join("cache");
        let cache = optimized_cache_path(&cache_dir, &model);
        write_identity_model(&model);
        let runtime = cached_runtime(&cache_dir);
        drop(runtime.load_session(&model, true).unwrap());
        let reader = std::fs::File::open(&cache).unwrap();
        reader.set_modified(SystemTime::UNIX_EPOCH).unwrap();

        drop(runtime.load_session(&model, true).unwrap());

        assert_ne!(
            reader.metadata().unwrap().ino(),
            std::fs::metadata(&cache).unwrap().ino()
        );
        assert!(runtime.try_load_cached_encoder(&model).unwrap().is_ok());
        assert_eq!(std::fs::read_dir(&cache_dir).unwrap().count(), 1);
    }

    #[test]
    fn test_cache_concurrent_cold_loads_leave_reusable_graph() {
        let tmp = tempfile::tempdir().unwrap();
        let model = tmp.path().join("encoder.onnx");
        let cache_dir = tmp.path().join("cache");
        write_identity_model(&model);
        let runtime = cached_runtime(&cache_dir);
        let barrier = std::sync::Barrier::new(4);
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        runtime.load_session(&model, true).unwrap()
                    })
                })
                .collect();
            for handle in handles {
                let session = handle.join().unwrap();
                let input = Tensor::new_checked(
                    crate::runtime::tensor::Shape::new(vec![1]),
                    crate::runtime::tensor::TensorData::F32(vec![42.0]),
                );
                assert_eq!(
                    session.run(std::slice::from_ref(&input)).unwrap(),
                    vec![input]
                );
            }
        });
        assert!(
            cached_runtime(&cache_dir)
                .try_load_cached_encoder(&model)
                .unwrap()
                .is_ok()
        );
        assert_eq!(std::fs::read_dir(&cache_dir).unwrap().count(), 1);
    }

    #[test]
    fn test_cache_publication_failure_keeps_loaded_session() {
        let tmp = tempfile::tempdir().unwrap();
        let model = tmp.path().join("encoder.onnx");
        let cache_dir = tmp.path().join("cache");
        write_identity_model(&model);
        // A directory at the destination makes publication fail on every OS.
        std::fs::create_dir_all(optimized_cache_path(&cache_dir, &model)).unwrap();
        assert!(
            cached_runtime(&cache_dir)
                .load_session(&model, true)
                .is_ok()
        );
        assert_eq!(std::fs::read_dir(&cache_dir).unwrap().count(), 1);
    }

    #[test]
    fn test_cache_is_fresh_nonempty_and_not_older_than_source() {
        let now = SystemTime::now();
        assert!(cache_is_fresh(100, now, now));
        assert!(cache_is_fresh(100, now + Duration::from_secs(1), now));
    }

    #[test]
    fn test_cache_is_fresh_rejects_stale_or_empty() {
        let now = SystemTime::now();
        assert!(!cache_is_fresh(100, now - Duration::from_secs(1), now));
        assert!(!cache_is_fresh(0, now + Duration::from_secs(1), now));
    }

    #[test]
    fn test_optimized_cache_is_fresh_with_real_files() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let source = dir.join("v3_rnnt_encoder_int8.onnx");
        let cache = dir.join("optimized_cache/v3_rnnt_encoder_int8_optimized.ort");
        write_file(&source, b"source");
        write_file(&cache, b"optimized");
        assert!(optimized_cache_is_fresh(&cache, &source));
    }

    #[test]
    fn test_optimized_cache_is_fresh_missing_or_empty_cache() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let source = dir.join("encoder.onnx");
        let cache = dir.join("optimized_cache/encoder_optimized.ort");
        write_file(&source, b"source");

        // Missing cache file → not fresh.
        assert!(!optimized_cache_is_fresh(&cache, &source));

        // Empty cache file → not fresh even though it is newer than the source.
        write_file(&cache, b"");
        assert!(!optimized_cache_is_fresh(&cache, &source));

        // Missing source model → not fresh (nothing trustworthy to compare).
        assert!(!optimized_cache_is_fresh(&cache, &dir.join("absent.onnx")));
    }

    #[test]
    fn test_optimized_cache_path_matches_gc_keep_name() {
        let cache_dir = Path::new("/models/optimized_cache");
        let encoder = Path::new("/models/v3_rnnt_encoder_int8.onnx");
        assert_eq!(
            optimized_cache_path(cache_dir, encoder),
            Path::new("/models/optimized_cache/v3_rnnt_encoder_int8_optimized.ort")
        );
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
}
