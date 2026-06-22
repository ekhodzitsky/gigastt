use std::path::PathBuf;

use thiserror::Error;

use super::tensor::Shape;

/// Errors produced by the runtime abstraction layer.
#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("failed to load model {path}: {message}")]
    LoadFailed { path: PathBuf, message: String },

    #[error("inference failed: {0}")]
    InferenceFailed(String),

    #[error("invalid tensor shape: expected {expected:?}, got {got:?}")]
    InvalidShape { expected: Shape, got: Shape },

    #[error("unsupported element type")]
    UnsupportedElementType,

    #[error("invalid input count: expected {expected}, got {got}")]
    InvalidInputCount { expected: usize, got: usize },
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn test_load_failed_display() {
        let e = RuntimeError::LoadFailed {
            path: PathBuf::from("encoder.onnx"),
            message: "not found".into(),
        };
        assert_eq!(
            e.to_string(),
            "failed to load model encoder.onnx: not found"
        );
    }

    #[test]
    fn test_invalid_shape_display() {
        let expected = Shape::new(vec![2, 3]);
        let got = Shape::new(vec![3, 2]);
        let e = RuntimeError::InvalidShape {
            expected: expected.clone(),
            got: got.clone(),
        };
        assert!(e.to_string().contains("invalid tensor shape"));
        assert!(e.to_string().contains("[2, 3]"));
        assert!(e.to_string().contains("[3, 2]"));
    }

    #[test]
    fn test_inference_failed_display() {
        let e = RuntimeError::InferenceFailed("session is closed".into());
        assert_eq!(e.to_string(), "inference failed: session is closed");
    }

    #[test]
    fn test_invalid_input_count_display() {
        let e = RuntimeError::InvalidInputCount {
            expected: 3,
            got: 2,
        };
        assert_eq!(e.to_string(), "invalid input count: expected 3, got 2");
    }

    #[test]
    fn test_unsupported_element_type_display() {
        let e = RuntimeError::UnsupportedElementType;
        assert_eq!(e.to_string(), "unsupported element type");
    }
}
