use std::path::Path;
use std::sync::{Arc, Mutex};

use ort::session::Session;

use crate::runtime::{
    error::RuntimeError, factory::Runtime, session::RuntimeSession, tensor::Tensor,
};

use super::{cache::OptimizedCache, factory::OrtExecutionProvider, tensor::value_to_tensor};

/// `ort`-backed runtime that loads sessions for a specific execution provider.
pub struct OrtRuntime {
    environment: ort::environment::Environment,
    intra_threads: usize,
    allow_spinning: bool,
    provider: OrtExecutionProvider,
    secondary_cpu: bool,
    prepacked: Option<Arc<ort::session::builder::PrepackedWeights>>,
    optimized_cache: Option<OptimizedCache>,
}

impl OrtRuntime {
    pub(crate) fn new(
        environment: ort::environment::Environment,
        intra_threads: usize,
        provider: OrtExecutionProvider,
        secondary_cpu: bool,
        prepacked: Option<Arc<ort::session::builder::PrepackedWeights>>,
        optimized_cache_dir: Option<std::path::PathBuf>,
    ) -> Self {
        Self {
            environment,
            intra_threads,
            allow_spinning: true,
            provider,
            secondary_cpu,
            prepacked,
            optimized_cache: optimized_cache_dir.map(OptimizedCache::new),
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
        let mut builder = Session::builder(&self.environment)
            .map_err(|e| load_failed(model_path, e))?
            .with_auto_device(ort::session::builder::AutoDevicePolicy::PreferCPU)
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
        let cache_path = self.optimized_cache.as_ref()?.existing_path(
            source_hash,
            self.intra_threads,
            self.prepacked.is_some(),
        )?;
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
        let cache = if is_encoder && self.provider.is_cpu() {
            self.optimized_cache.as_ref()
        } else {
            None
        };
        let source_hash = match cache {
            Some(cache) => cache
                .source_hash(model_path, source_hash)
                .map_err(|e| load_failed(model_path, e))?,
            None => None,
        };
        // Check identity under the same gate as publication. Waiting pool
        // triplets reuse the first writer's graph instead of re-optimizing it.
        let _cache_guard = cache.map(OptimizedCache::lock);
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
        let pending_cache = cache.zip(source_hash.as_deref()).and_then(|(cache, hash)| {
            cache.stage(hash, self.intra_threads, self.prepacked.is_some())
        });

        if let Some(pending) = pending_cache.as_ref() {
            tracing::info!(
                path = %pending.path().display(),
                "encoder: loading source model and refreshing optimized graph cache"
            );
            builder = builder
                .with_optimized_model_path(pending.path())
                .map_err(|e| load_failed(model_path, e))?;
            // Persist the optimized graph in ORT flatbuffer format so the
            // next boot can memory-map it (see `try_load_cached_encoder`).
            builder = builder
                .with_config_entry("session.save_model_format", "ORT")
                .map_err(|e| load_failed(model_path, e))?;
        }

        let session = match builder.commit_from_file(model_path) {
            Ok(session) => {
                if let Some(pending) = &pending_cache {
                    pending.publish(model_path);
                }
                session
            }
            Err(e) => {
                let Some(pending) = pending_cache.as_ref() else {
                    return Err(load_failed(model_path, e));
                };
                tracing::warn!(
                    path = %pending.path().display(),
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
mod tests;
