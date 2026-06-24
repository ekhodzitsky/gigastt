//! Composite ANE runtime: encoder on the Apple Neural Engine (per-bucket
//! fixed-shape `.mlpackage`, pad-up + fill-floor + ort fallback), decoder and
//! joiner delegated to an inner ort CPU runtime.
//!
//! ISOLATION: `objc2_core_ml` usage stays inside `runtime/coreml/`. Compiling +
//! holding the `MLModel` handles happens here and in [`super::encoder_session`];
//! the prediction itself goes through [`super::bridge`]. Gated
//! `#[cfg(all(feature = "ane", target_os = "macos"))]` (see `coreml/mod.rs`).

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::model::{ANE_BUCKETS, ane_package_complete, ane_package_dir_name};
use crate::runtime::{error::RuntimeError, factory::Runtime, session::RuntimeSession};

use super::bridge;
use super::encoder_session::{AneEncoderSession, BucketModel, SharedModel};

/// Composite runtime. Owns an inner ort runtime (for decoder/joiner + the
/// encoder fallback) and a lazily-populated, `Arc`-shared cache of compiled
/// bucket models so repeated `load_session(.., is_encoder=true)` calls — one per
/// pool slot — compile each ~167 MB package once and share it across every
/// encoder session instead of duplicating it per slot.
pub struct AneRuntime {
    ort: Box<dyn Runtime>,
    /// `bucket size -> compiled model`, shared across pool slots.
    bucket_cache: Arc<Mutex<HashMap<usize, Arc<SharedModel>>>>,
}

impl AneRuntime {
    pub fn new(ort: Box<dyn Runtime>) -> Self {
        Self {
            ort,
            bucket_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Compile `bucket`'s package (or return the cached handle). Holds the cache
    /// lock across compile so concurrent pool-slot loads coalesce onto one
    /// compile instead of each producing a duplicate ~167 MB model.
    fn shared_bucket_model(
        &self,
        ane_dir: &Path,
        bucket: usize,
    ) -> Result<Arc<SharedModel>, RuntimeError> {
        let mut cache = self
            .bucket_cache
            .lock()
            .map_err(|_| RuntimeError::InferenceFailed("ANE bucket cache poisoned".into()))?;
        if let Some(model) = cache.get(&bucket) {
            return Ok(Arc::clone(model));
        }
        let pkg = ane_dir.join(ane_package_dir_name(bucket));
        // Emit BEFORE the compile: `compile_and_load` does a synchronous Core ML
        // compile that can take several seconds on a cold cache, all while this
        // method holds the cache lock — without this line it looks like a hang.
        tracing::info!(bucket, package = %pkg.display(), "compiling ANE encoder bucket (cold-start, may take several seconds)");
        let model = bridge::compile_and_load(&pkg, true)?;
        let shared = Arc::new(SharedModel(model));
        cache.insert(bucket, Arc::clone(&shared));
        Ok(shared)
    }
}

impl Runtime for AneRuntime {
    fn load_session(
        &self,
        model_path: &Path,
        is_encoder: bool,
    ) -> Result<Box<dyn RuntimeSession>, RuntimeError> {
        // Decoder / joiner (and any non-encoder model) run on the inner ort CPU
        // runtime — they are small and the ANE only overrides the encoder.
        if !is_encoder {
            return self.ort.load_session(model_path, false);
        }

        // The per-bucket `.mlpackage`s live in an `ane/` sibling of the ONNX
        // encoder file (`<model_dir>/ane/gigaam_v3_encoder_<bucket>.mlpackage`).
        let ane_dir =
            model_path
                .parent()
                .map(|p| p.join("ane"))
                .ok_or_else(|| RuntimeError::LoadFailed {
                    path: model_path.to_path_buf(),
                    message: "encoder model path has no parent directory".to_string(),
                })?;

        let available: Vec<usize> = ANE_BUCKETS
            .iter()
            .copied()
            .filter(|&b| ane_package_complete(&ane_dir.join(ane_package_dir_name(b))))
            .collect();

        if available.is_empty() {
            return Err(RuntimeError::LoadFailed {
                path: model_path.to_path_buf(),
                message: format!(
                    "ANE encoder packages not found in {}; run `gigastt download --ane` (or convert locally)",
                    ane_dir.display()
                ),
            });
        }

        // Compile (once) + share each present bucket on CPU_AND_NE, then build the
        // ort encoder fallback for clips outside the fill-floor / bucket range.
        let mut buckets = Vec::with_capacity(available.len());
        for bucket in available {
            let model = self.shared_bucket_model(&ane_dir, bucket)?;
            buckets.push(BucketModel {
                size: bucket,
                model,
            });
        }
        let ort_fallback = self.ort.load_session(model_path, true)?;
        Ok(Box::new(AneEncoderSession::new(buckets, ort_fallback)))
    }
}
