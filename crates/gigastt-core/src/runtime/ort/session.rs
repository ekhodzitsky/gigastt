use std::path::Path;
use std::sync::Mutex;

use ort::session::Session;

use crate::runtime::{
    error::RuntimeError, factory::Runtime, session::RuntimeSession, tensor::Tensor,
};

use super::{factory::OrtExecutionProvider, tensor::value_to_tensor};

/// `ort`-backed runtime that loads sessions for a specific execution provider.
pub struct OrtRuntime {
    intra_threads: usize,
    provider: OrtExecutionProvider,
}

impl OrtRuntime {
    pub(crate) fn new(intra_threads: usize, provider: OrtExecutionProvider) -> Self {
        Self {
            intra_threads,
            provider,
        }
    }
}

/// `ort`-backed session wrapping a loaded ONNX model.
pub struct OrtSession {
    session: Mutex<Session>,
}

impl Runtime for OrtRuntime {
    fn load_session(&self, model_path: &Path) -> Result<Box<dyn RuntimeSession>, RuntimeError> {
        let session = Session::builder()
            .map_err(|e| RuntimeError::LoadFailed {
                path: model_path.into(),
                message: e.to_string(),
            })?
            .with_execution_providers([self.provider.to_ort()])
            .map_err(|e| RuntimeError::LoadFailed {
                path: model_path.into(),
                message: e.to_string(),
            })?
            .with_intra_threads(self.intra_threads)
            .map_err(|e| RuntimeError::LoadFailed {
                path: model_path.into(),
                message: e.to_string(),
            })?
            .commit_from_file(model_path)
            .map_err(|e| RuntimeError::LoadFailed {
                path: model_path.into(),
                message: e.to_string(),
            })?;
        Ok(Box::new(OrtSession {
            session: Mutex::new(session),
        }))
    }
}

impl RuntimeSession for OrtSession {
    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RuntimeError> {
        let ort_inputs: Vec<ort::value::Value> = inputs
            .into_iter()
            .map(Tensor::into_ort_value)
            .collect::<Result<_, _>>()?;
        let session_inputs: Vec<ort::session::SessionInputValue<'_>> =
            ort_inputs.into_iter().map(Into::into).collect();

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
