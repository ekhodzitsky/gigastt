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
    allow_spinning: bool,
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
            allow_spinning: true,
            provider,
            secondary_cpu,
            prepacked,
            optimized_cache_dir,
            usable_cache_dir: OnceLock::new(),
            cache_load: parking_lot::Mutex::new(()),
        }
    }

    #[cfg(feature = "diarization")]
    pub(crate) fn without_spinning(mut self) -> Self {
        self.allow_spinning = false;
        self
    }
}

fn load_failed(path: &Path, e: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::LoadFailed {
        path: path.into(),
        message: e.to_string(),
    }
}

/// Level3 may specialize layouts for CPU vector capabilities. A graph built
/// on an AVX512 host must not be selected on an AVX2-only host with the same OS.
fn cpu_isa() -> String {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        format!(
            "sse2={};sse41={};avx={};avx2={};fma={};avx512f={};avx512bw={};avx512dq={};avx512vl={};avx512vnni={};avx512bf16={};avx512fp16={};avxvnni={}",
            std::is_x86_feature_detected!("sse2"),
            std::is_x86_feature_detected!("sse4.1"),
            std::is_x86_feature_detected!("avx"),
            std::is_x86_feature_detected!("avx2"),
            std::is_x86_feature_detected!("fma"),
            std::is_x86_feature_detected!("avx512f"),
            std::is_x86_feature_detected!("avx512bw"),
            std::is_x86_feature_detected!("avx512dq"),
            std::is_x86_feature_detected!("avx512vl"),
            std::is_x86_feature_detected!("avx512vnni"),
            std::is_x86_feature_detected!("avx512bf16"),
            std::is_x86_feature_detected!("avx512fp16"),
            std::is_x86_feature_detected!("avxvnni")
        )
    }
    #[cfg(target_arch = "aarch64")]
    {
        format!(
            "neon={};fp16={};dotprod={};i8mm={};bf16={};sve={};sve2={}",
            std::arch::is_aarch64_feature_detected!("neon"),
            std::arch::is_aarch64_feature_detected!("fp16"),
            std::arch::is_aarch64_feature_detected!("dotprod"),
            std::arch::is_aarch64_feature_detected!("i8mm"),
            std::arch::is_aarch64_feature_detected!("bf16"),
            std::arch::is_aarch64_feature_detected!("sve"),
            std::arch::is_aarch64_feature_detected!("sve2")
        )
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        std::env::consts::ARCH.to_owned()
    }
}

/// A cache entry is identified by source bytes and the exact optimization policy.
/// Bump this policy when changing serialization, provider, or builder settings.
fn settings_hash(runtime_info: &str, intra_threads: usize, prepacked: bool, isa: &str) -> String {
    use crate::sha256::{Sha256, hex_lower};
    let policy = format!(
        "v1;ort={runtime_info};api={};cpu;level=3;intra={};inter=1;prepacked={prepacked};format=ORT;mmap=1;initializers=1;cached-prepacking=0;os={};arch={};endian={};isa={isa}",
        ort::MINOR_VERSION,
        intra_threads.max(1),
        std::env::consts::OS,
        std::env::consts::ARCH,
        cfg!(target_endian = "little"),
    );
    let mut hash = Sha256::new();
    hash.update(policy.as_bytes());
    hex_lower(&hash.finalize())
}

impl OrtRuntime {
    fn cache_path(&self, cache_dir: &Path, source_hash: &str) -> std::path::PathBuf {
        cache_dir.join(crate::model::optimized_content_basename(
            source_hash,
            &settings_hash(
                ort::info(),
                self.intra_threads,
                self.prepacked.is_some(),
                &cpu_isa(),
            ),
        ))
    }
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
        let mut builder = Session::builder()
            .map_err(|e| load_failed(model_path, e))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| load_failed(model_path, e))?;

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
            builder = builder
                .with_intra_op_spinning(self.allow_spinning)
                .map_err(|e| load_failed(model_path, e))?
                .with_inter_op_spinning(self.allow_spinning)
                .map_err(|e| load_failed(model_path, e))?;
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
    fn try_load_cached_encoder(
        &self,
        model_path: &Path,
        source_hash: &str,
    ) -> Option<Result<Session, RuntimeError>> {
        if !self.provider.is_cpu() {
            return None;
        }
        let cache_dir = self.optimized_cache_dir.as_ref()?;
        let cache_path = self.cache_path(cache_dir, source_hash);
        if !std::fs::metadata(&cache_path).is_ok_and(|meta| meta.is_file() && meta.len() > 0) {
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
        self.load_session_with_model_hash(model_path, is_encoder, None)
    }

    fn load_session_with_model_hash(
        &self,
        model_path: &Path,
        is_encoder: bool,
        source_hash: Option<&str>,
    ) -> Result<Box<dyn RuntimeSession>, RuntimeError> {
        let source_hash =
            if is_encoder && self.provider.is_cpu() && self.optimized_cache_dir.is_some() {
                match source_hash {
                    Some(hash)
                        if hash.len() == 64
                            && hash
                                .bytes()
                                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) =>
                    {
                        Some(hash.to_owned())
                    }
                    _ => crate::model::optimized_source_hash(model_path)
                        .map_err(|e| load_failed(model_path, e))?,
                }
            } else {
                None
            };
        // Check identity under the same gate as publication. Waiting pool
        // triplets reuse the first writer's graph instead of re-optimizing it.
        let _cache_guard =
            (is_encoder && self.provider.is_cpu() && self.optimized_cache_dir.is_some())
                .then(|| self.cache_load.lock());
        if let Some(hash) = source_hash.as_deref()
            && let Some(result) = self.try_load_cached_encoder(model_path, hash)
        {
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
        let cache_path = if let Some(hash) = source_hash.as_deref() {
            self.usable_cache_dir
                .get_or_init(|| {
                    self.optimized_cache_dir
                        .as_deref()
                        .and_then(usable_optimized_cache_dir)
                })
                .as_ref()
                .map(|dir| self.cache_path(dir, hash))
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
                if let (Some(pending), Some(cache_path)) = (&pending_cache, &cache_path) {
                    // A stale caller hint or source replacement during compilation
                    // must never publish a graph under a different source's digest.
                    match crate::model::optimized_source_hash(model_path) {
                        Ok(current) if current.is_some() && current == source_hash => {
                            if let Err(e) = std::fs::rename(&pending.0, cache_path) {
                                tracing::warn!(path = %cache_path.display(), error = %e,
                                    "encoder: cannot publish optimized graph cache; keeping the source model session");
                            }
                        }
                        _ => tracing::warn!(path = %model_path.display(),
                            "encoder: source changed or is not self-contained; discarding optimized graph cache"),
                    }
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

    fn run_f32_into(
        &self,
        inputs: &[Tensor],
        destinations: &mut [&mut Vec<f32>],
    ) -> Result<(), RuntimeError> {
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
        if outputs.len() != destinations.len() {
            return Err(RuntimeError::InferenceFailed(format!(
                "expected {} output buffers, got {}",
                outputs.len(),
                destinations.len()
            )));
        }
        // Validate every borrowed output before changing any caller buffer.
        // ORT values and the session guard stay alive through both passes.
        for (_, output) in outputs.iter() {
            output
                .try_extract_tensor::<f32>()
                .map_err(|e| RuntimeError::InferenceFailed(e.to_string()))?;
        }
        for ((_, output), destination) in outputs.iter().zip(destinations) {
            let (_, data) = output
                .try_extract_tensor::<f32>()
                .map_err(|e| RuntimeError::InferenceFailed(e.to_string()))?;
            destination.clear();
            destination.extend_from_slice(data);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

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

    fn optimized_cache_path(cache_dir: &Path, model: &Path) -> std::path::PathBuf {
        cached_runtime(cache_dir).cache_path(
            cache_dir,
            &crate::model::optimized_source_hash(model).unwrap().unwrap(),
        )
    }

    #[test]
    #[cfg_attr(miri, ignore = "calls into onnxruntime FFI")]
    fn test_ort_f32_workspace_errors_preserve_destinations() {
        use crate::runtime::tensor::{Shape, TensorData};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("identity.onnx");
        write_identity_model(&path);
        let runtime = cached_runtime(dir.path());
        let session = runtime.load_session(&path, false).unwrap();
        let inputs = [Tensor::new(Shape::new(vec![1]), TensorData::F32(vec![3.0])).unwrap()];
        let mut first = vec![9.0];
        let mut second = vec![8.0];
        assert!(
            session
                .run_f32_into(&inputs, &mut [&mut first, &mut second])
                .is_err()
        );
        assert_eq!(first, [9.0]);
        assert_eq!(second, [8.0]);
        assert!(session.run_f32_into(&[], &mut [&mut first]).is_err());
        assert_eq!(first, [9.0]);
        // Change both ONNX tensor element types from float to int64, while
        // preserving the single-element shapes and Identity graph.
        let mut model = std::fs::read(&path).unwrap();
        for index in 0..model.len() - 5 {
            if model[index..index + 5] == [0x0a, 0x08, 0x08, 0x01, 0x12] {
                model[index + 3] = 7;
            }
        }
        let integer_path = dir.path().join("integer.onnx");
        std::fs::write(&integer_path, model).unwrap();
        let integer = runtime.load_session(&integer_path, false).unwrap();
        let inputs = [Tensor::new(Shape::new(vec![1]), TensorData::I64(vec![3])).unwrap()];
        assert!(integer.run_f32_into(&inputs, &mut [&mut first]).is_err());
        assert_eq!(first, [9.0]);
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
    #[cfg_attr(miri, ignore = "calls into onnxruntime FFI")]
    fn test_cache_changed_contents_with_preserved_size_and_time() {
        let tmp = tempfile::tempdir().unwrap();
        let model = tmp.path().join("encoder.onnx");
        let cache_dir = tmp.path().join("cache");
        write_identity_model(&model);
        let runtime = cached_runtime(&cache_dir);
        drop(runtime.load_session(&model, true).unwrap());
        let modified = std::fs::metadata(&model).unwrap().modified().unwrap();
        let old = std::fs::read(&model).unwrap();
        let replacement = String::from_utf8(old.clone())
            .unwrap()
            .replace("Identity", "Softsign");
        assert_eq!(old.len(), replacement.len());
        write_file(&model, replacement.as_bytes());
        std::fs::File::options()
            .write(true)
            .open(&model)
            .unwrap()
            .set_modified(modified)
            .unwrap();
        let session = runtime.load_session(&model, true).unwrap();
        let input = Tensor::new_checked(
            crate::runtime::tensor::Shape::new(vec![1]),
            crate::runtime::tensor::TensorData::F32(vec![42.0]),
        );
        let expected = Tensor::new_checked(
            crate::runtime::tensor::Shape::new(vec![1]),
            crate::runtime::tensor::TensorData::F32(vec![42.0 / 43.0]),
        );
        assert_eq!(session.run(&[input]).unwrap(), vec![expected]);
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
    #[cfg_attr(miri, ignore = "calls into onnxruntime FFI")]
    fn test_cache_same_name_different_directories_and_thread_settings() {
        let tmp = tempfile::tempdir().unwrap();
        let first = tmp.path().join("one/encoder.onnx");
        let second = tmp.path().join("two/encoder.onnx");
        let cache_dir = tmp.path().join("cache");
        write_identity_model(&first);
        let bytes = String::from_utf8(std::fs::read(&first).unwrap()).unwrap();
        write_file(&second, bytes.replace("Identity", "Softsign").as_bytes());
        let runtime = cached_runtime(&cache_dir);
        drop(runtime.load_session(&first, true).unwrap());
        drop(runtime.load_session(&second, true).unwrap());
        let mut other = cached_runtime(&cache_dir);
        other.intra_threads = 2;
        drop(other.load_session(&first, true).unwrap());
        assert_eq!(std::fs::read_dir(&cache_dir).unwrap().count(), 3);
        assert_ne!(
            optimized_cache_path(&cache_dir, &first),
            optimized_cache_path(&cache_dir, &second)
        );
    }

    #[test]
    #[cfg_attr(miri, ignore = "calls into onnxruntime FFI")]
    fn test_cache_legacy_graph_is_not_loaded() {
        let tmp = tempfile::tempdir().unwrap();
        let model = tmp.path().join("encoder.onnx");
        let cache_dir = tmp.path().join("cache");
        write_identity_model(&model);
        let legacy = cache_dir.join("encoder_optimized.ort");
        write_file(&legacy, b"old graph");
        drop(
            cached_runtime(&cache_dir)
                .load_session(&model, true)
                .unwrap(),
        );
        assert_eq!(std::fs::read(&legacy).unwrap(), b"old graph");
        assert_eq!(std::fs::read_dir(&cache_dir).unwrap().count(), 2);
    }

    #[test]
    #[cfg_attr(miri, ignore = "calls into onnxruntime FFI")]
    fn test_cache_stale_digest_hint_does_not_publish_source_graph() {
        let tmp = tempfile::tempdir().unwrap();
        let model = tmp.path().join("encoder.onnx");
        let cache_dir = tmp.path().join("cache");
        write_identity_model(&model);
        drop(
            cached_runtime(&cache_dir)
                .load_session_with_model_hash(&model, true, Some(&"f".repeat(64)))
                .unwrap(),
        );
        assert_eq!(std::fs::read_dir(&cache_dir).unwrap().count(), 0);
    }

    #[test]
    #[cfg_attr(miri, ignore = "calls into onnxruntime FFI")]
    fn test_cache_malformed_hash_hint_cannot_escape_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let model = tmp.path().join("encoder.onnx");
        let cache_dir = tmp.path().join("cache");
        write_identity_model(&model);
        let runtime = cached_runtime(&cache_dir);
        for hint in ["../escape", "/outside", "not-a-digest"] {
            drop(
                runtime
                    .load_session_with_model_hash(&model, true, Some(hint))
                    .unwrap(),
            );
        }
        assert!(optimized_cache_path(&cache_dir, &model).is_file());
        assert_eq!(std::fs::read_dir(&cache_dir).unwrap().count(), 1);
        assert_eq!(std::fs::read_dir(tmp.path()).unwrap().count(), 2);
    }

    fn protobuf_bytes(field: u8, data: &[u8]) -> Vec<u8> {
        let mut out = vec![field << 3 | 2];
        let mut len = data.len();
        while len >= 128 {
            out.push((len as u8 & 127) | 128);
            len >>= 7;
        }
        out.push(len as u8);
        out.extend_from_slice(data);
        out
    }

    #[test]
    #[cfg_attr(miri, ignore = "calls into onnxruntime FFI")]
    fn test_cache_external_weights_changes_are_observed() {
        let tmp = tempfile::tempdir().unwrap();
        let model = tmp.path().join("encoder.onnx");
        let weights = tmp.path().join("weights.bin");
        let cache_dir = tmp.path().join("cache");
        // Add(x, w) -> y with w stored outside the ModelProto.
        let mut node = protobuf_bytes(1, b"x");
        node.extend(protobuf_bytes(1, b"w"));
        node.extend(protobuf_bytes(2, b"y"));
        node.extend(protobuf_bytes(4, b"Add"));
        let mut tensor = vec![8, 1, 16, 1]; // dims=[1], FLOAT
        tensor.extend(protobuf_bytes(8, b"w"));
        let mut location = protobuf_bytes(1, b"location");
        location.extend(protobuf_bytes(2, b"weights.bin"));
        tensor.extend(protobuf_bytes(13, &location));
        tensor.extend([112, 1]); // data_location=EXTERNAL
        let mut graph = protobuf_bytes(1, &node);
        graph.extend(protobuf_bytes(2, b"external-test"));
        graph.extend(protobuf_bytes(5, &tensor));
        graph.extend(protobuf_bytes(
            11,
            b"\x0a\x01x\x12\x0a\x0a\x08\x08\x01\x12\x04\x0a\x02\x08\x01",
        ));
        graph.extend(protobuf_bytes(
            12,
            b"\x0a\x01y\x12\x0a\x0a\x08\x08\x01\x12\x04\x0a\x02\x08\x01",
        ));
        let mut bytes = vec![8, 8];
        bytes.extend(protobuf_bytes(7, &graph));
        bytes.extend([66, 2, 16, 13]);
        write_file(&model, &bytes);
        let runtime = cached_runtime(&cache_dir);
        for value in [42.0f32, 10.0] {
            write_file(&weights, &value.to_le_bytes());
            let session = runtime.load_session(&model, true).unwrap();
            let input = Tensor::new_checked(
                crate::runtime::tensor::Shape::new(vec![1]),
                crate::runtime::tensor::TensorData::F32(vec![1.0]),
            );
            let expected = Tensor::new_checked(
                crate::runtime::tensor::Shape::new(vec![1]),
                crate::runtime::tensor::TensorData::F32(vec![value + 1.0]),
            );
            assert_eq!(session.run(&[input]).unwrap(), vec![expected]);
        }
        assert!(!cache_dir.exists());
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
    #[cfg_attr(miri, ignore = "calls into onnxruntime FFI")]
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
    #[cfg_attr(miri, ignore = "calls into onnxruntime FFI")]
    fn test_cache_broken_graph_is_replaced_on_source_load() {
        let tmp = tempfile::tempdir().unwrap();
        let model = tmp.path().join("encoder.onnx");
        let cache_dir = tmp.path().join("cache");
        write_identity_model(&model);
        let cache = optimized_cache_path(&cache_dir, &model);
        write_file(&cache, b"invalid ORT");
        let runtime = cached_runtime(&cache_dir);
        assert!(
            runtime
                .try_load_cached_encoder(
                    &model,
                    &crate::model::optimized_source_hash(&model)
                        .unwrap()
                        .unwrap()
                )
                .is_none()
        );
        // A failed reader must not delete a path another process can replace.
        assert_eq!(std::fs::read(&cache).unwrap(), b"invalid ORT");
        drop(runtime.load_session(&model, true).unwrap());
        assert!(
            runtime
                .try_load_cached_encoder(
                    &model,
                    &crate::model::optimized_source_hash(&model)
                        .unwrap()
                        .unwrap()
                )
                .unwrap()
                .is_ok()
        );
        assert_eq!(std::fs::read_dir(&cache_dir).unwrap().count(), 1);
    }

    #[cfg(unix)]
    #[test]
    #[cfg_attr(miri, ignore = "calls into onnxruntime FFI")]
    fn test_cache_refresh_replaces_file_without_truncating_readers() {
        use std::os::unix::fs::MetadataExt;

        let tmp = tempfile::tempdir().unwrap();
        let model = tmp.path().join("encoder.onnx");
        let cache_dir = tmp.path().join("cache");
        write_identity_model(&model);
        let cache = optimized_cache_path(&cache_dir, &model);
        let runtime = cached_runtime(&cache_dir);
        drop(runtime.load_session(&model, true).unwrap());
        let reader = std::fs::File::open(&cache).unwrap();
        // Replace the cached graph with invalid bytes. Keep the old mapped inode
        // open; recovery must publish another file rather than truncating it.
        std::fs::remove_file(&cache).unwrap();
        write_file(&cache, b"broken cache");

        drop(runtime.load_session(&model, true).unwrap());

        assert_ne!(
            reader.metadata().unwrap().ino(),
            std::fs::metadata(&cache).unwrap().ino()
        );
        assert!(
            runtime
                .try_load_cached_encoder(
                    &model,
                    &crate::model::optimized_source_hash(&model)
                        .unwrap()
                        .unwrap()
                )
                .unwrap()
                .is_ok()
        );
        assert_eq!(std::fs::read_dir(&cache_dir).unwrap().count(), 1);
    }

    #[test]
    #[cfg_attr(miri, ignore = "calls into onnxruntime FFI")]
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
                .try_load_cached_encoder(
                    &model,
                    &crate::model::optimized_source_hash(&model)
                        .unwrap()
                        .unwrap()
                )
                .unwrap()
                .is_ok()
        );
        assert_eq!(std::fs::read_dir(&cache_dir).unwrap().count(), 1);
    }

    #[test]
    #[cfg_attr(miri, ignore = "calls into onnxruntime FFI")]
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
