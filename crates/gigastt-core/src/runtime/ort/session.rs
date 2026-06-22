use std::path::Path;
use std::sync::{Arc, Mutex};

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
        }
    }
}

fn load_failed(path: &Path, e: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::LoadFailed {
        path: path.into(),
        message: e.to_string(),
    }
}

impl Runtime for OrtRuntime {
    fn load_session(
        &self,
        model_path: &Path,
        is_encoder: bool,
    ) -> Result<Box<dyn RuntimeSession>, RuntimeError> {
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

            if is_encoder && let Some(cache_dir) = &self.optimized_cache_dir {
                std::fs::create_dir_all(cache_dir).map_err(|e| load_failed(model_path, e))?;
                let stem = model_path
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "encoder".into());
                builder = builder
                    .with_optimized_model_path(cache_dir.join(format!("{stem}_optimized.onnx")))
                    .map_err(|e| load_failed(model_path, e))?;
            }
        }

        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| load_failed(model_path, e))?;
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
