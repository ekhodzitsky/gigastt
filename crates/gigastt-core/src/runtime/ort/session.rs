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
    prepacked: Option<Arc<ort::session::builder::PrepackedWeights>>,
    optimized_cache_dir: Option<std::path::PathBuf>,
    /// Once-per-runtime result of the cache-dir writability probe. Pool
    /// triplets load concurrently on this shared runtime, so probing per
    /// `load_session` would repeat the probe (and its warning) pool-size
    /// times at every boot. Admin reload builds a fresh `OrtRuntime`, so
    /// re-probing on reload is preserved.
    usable_cache_dir: OnceLock<Option<std::path::PathBuf>>,
}

impl OrtRuntime {
    pub(crate) fn new(
        intra_threads: usize,
        provider: OrtExecutionProvider,
        prepacked: Option<Arc<ort::session::builder::PrepackedWeights>>,
        optimized_cache_dir: Option<std::path::PathBuf>,
    ) -> Self {
        Self {
            intra_threads,
            provider,
            prepacked,
            optimized_cache_dir,
            usable_cache_dir: OnceLock::new(),
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
    let probe = cache_dir.join(format!(".write-probe-{}", std::process::id()));
    match std::fs::File::create(&probe) {
        Ok(_) => {
            let _ = std::fs::remove_file(&probe);
            Some(cache_dir.to_path_buf())
        }
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

        let eps = self.provider.execution_providers(model_path);
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
    /// fails to load (the broken entry is deleted so the next boot rewrites
    /// it cleanly).
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
                    "encoder: optimized graph cache failed to load; deleting it and falling back to the source model"
                );
                if let Err(rm) = std::fs::remove_file(&cache_path) {
                    tracing::warn!(
                        path = %cache_path.display(),
                        error = %rm,
                        "encoder: failed to delete broken optimized graph cache"
                    );
                }
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

        if let Some(cache_path) = cache_path.as_ref() {
            tracing::info!(
                path = %cache_path.display(),
                "encoder: loading source model and refreshing optimized graph cache"
            );
            builder = builder
                .with_optimized_model_path(cache_path)
                .map_err(|e| load_failed(model_path, e))?;
            // Persist the optimized graph in ORT flatbuffer format so the
            // next boot can memory-map it (see `try_load_cached_encoder`).
            builder = builder
                .with_config_entry("session.save_model_format", "ORT")
                .map_err(|e| load_failed(model_path, e))?;
        }

        let session = match builder.commit_from_file(model_path) {
            Ok(session) => session,
            Err(e) => {
                let Some(cache_path) = cache_path.as_ref() else {
                    return Err(load_failed(model_path, e));
                };
                tracing::warn!(
                    path = %cache_path.display(),
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
        // Pool triplets load concurrently on the same runtime, so the probe
        // can race with itself within one process: the shared probe file name
        // (.write-probe-<pid>) must tolerate create/delete races and every
        // caller must still see a usable dir with no leftovers.
        let tmp = tempfile::tempdir().unwrap();
        let cache_dir = tmp.path().join("optimized_cache");
        std::fs::create_dir_all(&cache_dir).unwrap();
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..4)
                .map(|_| s.spawn(|| usable_optimized_cache_dir(&cache_dir)))
                .collect();
            for h in handles {
                assert_eq!(h.join().unwrap(), Some(cache_dir.clone()));
            }
        });
        let leftover: Vec<_> = std::fs::read_dir(&cache_dir).unwrap().collect();
        assert!(leftover.is_empty(), "probe left files behind: {leftover:?}");
    }
}
