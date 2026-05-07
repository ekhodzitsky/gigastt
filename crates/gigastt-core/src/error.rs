//! Error types for the gigastt public API.
//!
//! [`GigasttError`] is the primary error type returned by [`Engine`](crate::inference::Engine)
//! methods. It provides structured error variants so consumers can match on specific
//! failure modes without downcasting.

use thiserror::Error;

/// A validated model path string.
///
/// Invariant: non-empty, valid UTF-8.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPath(String);

impl ModelPath {
    /// { !s.is_empty() }
    /// fn new(s: &str) -> Result<ModelPath, GigasttError>
    /// { ret.as_ref().map(|p| !p.as_str().is_empty()).unwrap_or(true) }
    pub fn new(s: &str) -> Result<Self, GigasttError> {
        if s.is_empty() {
            return Err(GigasttError::InvalidAudio {
                reason: "empty model path".into(),
            });
        }
        Ok(ModelPath(s.to_string()))
    }

    /// { true }
    /// fn as_str(&self) -> &str
    /// { !ret.is_empty() }
    /// { true }
    /// fn as_str(&self) -> &str
    /// { !ret.is_empty() }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A human-readable error reason string.
///
/// Invariant: non-empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reason(String);

impl Reason {
    /// { !s.is_empty() }
    /// fn new(s: &str) -> Result<Reason, GigasttError>
    /// { ret.as_ref().map(|r| !r.as_str().is_empty()).unwrap_or(true) }
    pub fn new(s: &str) -> Result<Self, GigasttError> {
        if s.is_empty() {
            return Err(GigasttError::InvalidAudio {
                reason: "empty error reason".into(),
            });
        }
        Ok(Reason(s.to_string()))
    }

    /// { true }
    /// fn as_str(&self) -> &str
    /// { !ret.is_empty() }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Errors returned by gigastt public API methods.
///
/// This enum covers the main failure categories:
/// - Model loading failures ([`ModelLoad`](GigasttError::ModelLoad))
/// - Runtime inference errors ([`Inference`](GigasttError::Inference))
/// - Invalid audio input ([`InvalidAudio`](GigasttError::InvalidAudio))
/// - Filesystem / I/O errors ([`Io`](GigasttError::Io))
///
/// # Matching on errors
///
/// ```ignore
/// use gigastt::error::GigasttError;
///
/// match err {
///     GigasttError::ModelLoad { path, .. } => eprintln!("Model problem at {path}"),
///     GigasttError::Inference { .. } => eprintln!("Inference failed"),
///     GigasttError::InvalidAudio { reason } => eprintln!("Bad audio: {reason}"),
///     GigasttError::Io(e) => eprintln!("I/O error: {e}"),
///     _ => eprintln!("Other error"),
/// }
/// ```
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GigasttError {
    /// Model files not found or failed to load ONNX sessions.
    #[error("model load error at {path}")]
    ModelLoad {
        /// Path to the model file or directory that failed.
        path: String,
        /// Underlying error, if any.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
    /// ONNX inference failed during encode, decode, or join.
    #[error("inference failed")]
    Inference {
        /// Underlying error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// Invalid audio input (unsupported format, excessive duration, corrupt data).
    #[error("invalid audio: {reason}")]
    InvalidAudio {
        /// Human-readable description of why the audio was rejected.
        reason: String,
    },
    /// Filesystem or I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_path_rejects_empty() {
        assert!(ModelPath::new("").is_err());
    }

    #[test]
    fn test_model_path_accepts_valid() {
        let p = ModelPath::new("encoder.onnx").unwrap();
        assert_eq!(p.as_str(), "encoder.onnx");
    }

    #[test]
    fn test_reason_rejects_empty() {
        assert!(Reason::new("").is_err());
    }

    #[test]
    fn test_reason_accepts_valid() {
        let r = Reason::new("too long").unwrap();
        assert_eq!(r.as_str(), "too long");
    }

    #[test]
    fn test_display_model_load() {
        let e = GigasttError::ModelLoad {
            path: "encoder.onnx".into(),
            source: Some(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "missing weights",
            ))),
        };
        assert!(e.to_string().contains("encoder.onnx"));
    }

    #[test]
    fn test_display_inference() {
        let e = GigasttError::Inference {
            source: Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "decoder failed",
            )),
        };
        assert_eq!(e.to_string(), "inference failed");
    }

    #[test]
    fn test_display_invalid_audio() {
        let e = GigasttError::InvalidAudio {
            reason: "too long".into(),
        };
        assert_eq!(e.to_string(), "invalid audio: too long");
    }

    #[test]
    fn test_display_io() {
        let e = GigasttError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "gone"));
        assert!(e.to_string().contains("gone"));
    }

    #[test]
    fn test_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let e: GigasttError = io_err.into();
        assert!(matches!(e, GigasttError::Io(_)));
    }

    #[test]
    fn test_error_source_io() {
        let e = GigasttError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "x"));
        assert!(std::error::Error::source(&e).is_none());
    }

    #[test]
    fn test_into_anyhow() {
        // Verify GigasttError works with ? in anyhow::Result contexts
        fn returns_anyhow() -> anyhow::Result<()> {
            Err(GigasttError::Inference {
                source: Box::new(std::io::Error::new(std::io::ErrorKind::Other, "test")),
            })?;
            Ok(())
        }
        assert!(returns_anyhow().is_err());
    }
}
