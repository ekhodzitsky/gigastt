use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::runtime::{
    error::RuntimeError,
    factory::{Runtime, RuntimeFactory},
    session::RuntimeSession,
    tensor::{Shape, Tensor},
};

/// Factory that builds a [`MockRuntime`] from a map of scripted sessions.
///
/// Each entry is keyed by the stem of the model path the engine will load,
/// e.g. `v3_rnnt_encoder` for `/path/v3_rnnt_encoder.onnx`.
#[derive(Clone, Default)]
pub struct MockFactory {
    sessions: HashMap<String, Arc<MockSession>>,
}

#[allow(dead_code)]
impl MockFactory {
    pub fn new(sessions: HashMap<String, Arc<MockSession>>) -> Self {
        Self { sessions }
    }
}

impl RuntimeFactory for MockFactory {
    fn create(&self, _intra_threads: usize) -> Result<Box<dyn Runtime>, RuntimeError> {
        Ok(Box::new(MockRuntime {
            sessions: self.sessions.clone(),
        }))
    }

    fn cpu_fallback(&self) -> Box<dyn RuntimeFactory> {
        Box::new(MockFactory {
            sessions: self.sessions.clone(),
        })
    }
}

/// Mock runtime that hands out pre-configured [`MockSession`]s by path stem.
#[derive(Clone)]
pub struct MockRuntime {
    sessions: HashMap<String, Arc<MockSession>>,
}

impl Runtime for MockRuntime {
    fn load_session(
        &self,
        model_path: &Path,
        _is_encoder: bool,
    ) -> Result<Box<dyn RuntimeSession>, RuntimeError> {
        let key = model_path
            .file_stem()
            .ok_or_else(|| RuntimeError::LoadFailed {
                path: model_path.into(),
                message: "empty path".into(),
            })?
            .to_string_lossy()
            .to_string();
        let session = self
            .sessions
            .get(&key)
            .ok_or_else(|| RuntimeError::LoadFailed {
                path: model_path.into(),
                message: format!("no mock for {key}"),
            })?
            .clone();
        Ok(Box::new((*session).clone()))
    }
}

/// Mock ONNX session that validates input shapes and returns pre-recorded outputs.
///
/// Intended for unit tests that exercise the engine and decode loop without
/// loading real model files.
pub struct MockSession {
    pub expected_inputs: Vec<Shape>,
    pub outputs: Vec<Tensor>,
    call_count: AtomicUsize,
}

#[allow(dead_code)]
impl MockSession {
    pub fn new(expected_inputs: Vec<Shape>, outputs: Vec<Tensor>) -> Self {
        Self {
            expected_inputs,
            outputs,
            call_count: AtomicUsize::new(0),
        }
    }

    /// Number of times [`RuntimeSession::run`] has been called on this session.
    pub fn call_count(&self) -> usize {
        self.call_count.load(Ordering::Relaxed)
    }
}

impl Clone for MockSession {
    fn clone(&self) -> Self {
        Self {
            expected_inputs: self.expected_inputs.clone(),
            outputs: self.outputs.clone(),
            call_count: AtomicUsize::new(self.call_count.load(Ordering::Relaxed)),
        }
    }
}

impl RuntimeSession for MockSession {
    fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, RuntimeError> {
        if inputs.len() != self.expected_inputs.len() {
            return Err(RuntimeError::InvalidInputCount {
                expected: self.expected_inputs.len(),
                got: inputs.len(),
            });
        }
        for (actual, expected) in inputs.iter().zip(self.expected_inputs.iter()) {
            if actual.shape() != expected {
                return Err(RuntimeError::InvalidShape {
                    expected: expected.clone(),
                    got: actual.shape().clone(),
                });
            }
        }
        self.call_count.fetch_add(1, Ordering::Relaxed);
        Ok(self.outputs.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::tensor::TensorData;

    #[test]
    fn test_mock_session_returns_recorded_outputs() {
        let session = MockSession::new(
            vec![Shape::new(vec![1, 2])],
            vec![Tensor::new(Shape::new(vec![1]), TensorData::F32(vec![42.0])).unwrap()],
        );
        let input = Tensor::new(Shape::new(vec![1, 2]), TensorData::F32(vec![0.0, 0.0])).unwrap();
        let outputs = session.run(&[input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(session.call_count(), 1);
    }

    #[test]
    fn test_mock_session_rejects_mismatched_shape() {
        let session = MockSession::new(
            vec![Shape::new(vec![1, 2])],
            vec![Tensor::new(Shape::new(vec![1]), TensorData::F32(vec![42.0])).unwrap()],
        );
        let input = Tensor::new(Shape::new(vec![1, 3]), TensorData::F32(vec![0.0; 3])).unwrap();
        let err = session.run(&[input]).unwrap_err();
        match err {
            RuntimeError::InvalidShape { expected, got } => {
                assert_eq!(expected.dims(), &[1, 2]);
                assert_eq!(got.dims(), &[1, 3]);
            }
            other => panic!("expected InvalidShape, got {other:?}"),
        }
    }

    #[test]
    fn test_mock_session_rejects_wrong_input_count() {
        let session = MockSession::new(
            vec![Shape::new(vec![1]), Shape::new(vec![1])],
            vec![Tensor::new(Shape::new(vec![1]), TensorData::F32(vec![42.0])).unwrap()],
        );
        let input = Tensor::new(Shape::new(vec![1]), TensorData::F32(vec![0.0])).unwrap();
        let err = session.run(&[input]).unwrap_err();
        match err {
            RuntimeError::InvalidInputCount { expected, got } => {
                assert_eq!(expected, 2);
                assert_eq!(got, 1);
            }
            other => panic!("expected InvalidInputCount, got {other:?}"),
        }
    }

    #[test]
    fn test_mock_runtime_loads_session_by_stem() {
        let mut sessions = HashMap::new();
        sessions.insert(
            "encoder".into(),
            Arc::new(MockSession::new(
                vec![Shape::new(vec![1])],
                vec![Tensor::new(Shape::new(vec![1]), TensorData::F32(vec![1.0])).unwrap()],
            )),
        );
        let runtime = MockRuntime { sessions };
        let session = runtime
            .load_session(Path::new("/models/encoder.onnx"), true)
            .expect("load by stem");
        let input = Tensor::new(Shape::new(vec![1]), TensorData::F32(vec![0.0])).unwrap();
        let outputs = session.run(&[input]).unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_mock_runtime_missing_session_fails() {
        let runtime = MockRuntime {
            sessions: HashMap::new(),
        };
        let result = runtime.load_session(Path::new("/models/missing.onnx"), false);
        let err = match result {
            Ok(_) => panic!("expected load to fail"),
            Err(e) => e,
        };
        match err {
            RuntimeError::LoadFailed { message, .. } => {
                assert!(message.contains("no mock for missing"));
            }
            other => panic!("expected LoadFailed, got {other:?}"),
        }
    }
}
