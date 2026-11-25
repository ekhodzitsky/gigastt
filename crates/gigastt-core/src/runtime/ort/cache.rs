//! Filesystem ownership for optimized ORT graphs. Session construction and
//! provider-specific optimization policy remain in the runtime adapter.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub(super) struct OptimizedCache {
    directory: PathBuf,
    // Probe once per runtime, including after admin reload creates a fresh owner.
    usable_directory: OnceLock<Option<PathBuf>>,
    // Pool triplets reuse the first writer's graph instead of optimizing again.
    load: parking_lot::Mutex<()>,
}

impl OptimizedCache {
    pub(super) fn new(directory: PathBuf) -> Self {
        Self {
            directory,
            usable_directory: OnceLock::new(),
            load: parking_lot::Mutex::new(()),
        }
    }

    pub(super) fn source_hash(
        &self,
        model_path: &Path,
        hint: Option<&str>,
    ) -> std::io::Result<Option<String>> {
        match hint {
            Some(hash)
                if hash.len() == 64
                    && hash
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) =>
            {
                Ok(Some(hash.to_owned()))
            }
            _ => crate::model::optimized_source_hash(model_path),
        }
    }

    pub(super) fn lock(&self) -> parking_lot::MutexGuard<'_, ()> {
        self.load.lock()
    }

    pub(super) fn path(&self, source_hash: &str, intra_threads: usize, prepacked: bool) -> PathBuf {
        self.directory
            .join(crate::model::optimized_content_basename(
                source_hash,
                &settings_hash(ort::info(), intra_threads, prepacked, &cpu_isa()),
            ))
    }

    pub(super) fn existing_path(
        &self,
        source_hash: &str,
        intra_threads: usize,
        prepacked: bool,
    ) -> Option<PathBuf> {
        let path = self.path(source_hash, intra_threads, prepacked);
        std::fs::metadata(&path)
            .is_ok_and(|meta| meta.is_file() && meta.len() > 0)
            .then_some(path)
    }

    pub(super) fn stage(
        &self,
        source_hash: &str,
        intra_threads: usize,
        prepacked: bool,
    ) -> Option<StagedGraph> {
        self.usable_directory
            .get_or_init(|| usable_optimized_cache_dir(&self.directory))
            .as_ref()?;
        let path = self.path(source_hash, intra_threads, prepacked);
        match StagedGraph::new(&path, source_hash) {
            Ok(staged) => Some(staged),
            Err(e) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %e,
                    "encoder: cannot stage optimized graph cache; loading source model without the cache"
                );
                None
            }
        }
    }
}

/// Owns an exclusive staging file, its destination and the expected source
/// identity until the ORT builder completes. Dropping always cleans staging.
pub(super) struct StagedGraph {
    pending: PendingCache,
    destination: PathBuf,
    source_hash: String,
}

impl StagedGraph {
    fn new(destination: &Path, source_hash: &str) -> std::io::Result<Self> {
        Ok(Self {
            pending: PendingCache::new(destination)?,
            destination: destination.to_path_buf(),
            source_hash: source_hash.to_owned(),
        })
    }

    pub(super) fn path(&self) -> &Path {
        &self.pending.0
    }

    pub(super) fn publish(&self, model_path: &Path) {
        // A stale caller hint or source replacement during compilation must
        // never publish a graph under a different source's digest.
        match crate::model::optimized_source_hash(model_path) {
            Ok(Some(current)) if current == self.source_hash => {
                if let Err(e) = std::fs::rename(self.path(), &self.destination) {
                    tracing::warn!(path = %self.destination.display(), error = %e,
                        "encoder: cannot publish optimized graph cache; keeping the source model session");
                }
            }
            _ => tracing::warn!(path = %model_path.display(),
                "encoder: source changed or is not self-contained; discarding optimized graph cache"),
        }
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

#[cfg(test)]
mod tests;
