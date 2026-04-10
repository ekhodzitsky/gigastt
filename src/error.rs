//! Error types for the gigastt public API.
//!
//! [`GigasttError`] is the primary error type returned by [`Engine`](crate::inference::Engine)
//! methods. It provides structured error variants so consumers can match on specific
//! failure modes without downcasting.

use std::fmt;

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
///     GigasttError::ModelLoad(msg) => eprintln!("Model problem: {msg}"),
///     GigasttError::Inference(msg) => eprintln!("Inference failed: {msg}"),
///     GigasttError::InvalidAudio(msg) => eprintln!("Bad audio: {msg}"),
///     GigasttError::Io(e) => eprintln!("I/O error: {e}"),
///     _ => eprintln!("Other error"),
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum GigasttError {
    /// Model files not found or failed to load ONNX sessions.
    ModelLoad(String),
    /// ONNX inference failed during encode, decode, or join.
    Inference(String),
    /// Invalid audio input (unsupported format, excessive duration, corrupt data).
    InvalidAudio(String),
    /// Filesystem or I/O error.
    Io(std::io::Error),
}

impl fmt::Display for GigasttError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GigasttError::ModelLoad(msg) => write!(f, "model load error: {msg}"),
            GigasttError::Inference(msg) => write!(f, "inference error: {msg}"),
            GigasttError::InvalidAudio(msg) => write!(f, "invalid audio: {msg}"),
            GigasttError::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for GigasttError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GigasttError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for GigasttError {
    fn from(e: std::io::Error) -> Self {
        GigasttError::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_model_load() {
        let e = GigasttError::ModelLoad("encoder not found".into());
        assert_eq!(e.to_string(), "model load error: encoder not found");
    }

    #[test]
    fn test_display_inference() {
        let e = GigasttError::Inference("decoder failed".into());
        assert_eq!(e.to_string(), "inference error: decoder failed");
    }

    #[test]
    fn test_display_invalid_audio() {
        let e = GigasttError::InvalidAudio("too long".into());
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
        assert!(std::error::Error::source(&e).is_some());
    }

    #[test]
    fn test_error_source_none_for_others() {
        let e = GigasttError::ModelLoad("x".into());
        assert!(std::error::Error::source(&e).is_none());
    }

    #[test]
    fn test_into_anyhow() {
        // Verify GigasttError works with ? in anyhow::Result contexts
        fn returns_anyhow() -> anyhow::Result<()> {
            Err(GigasttError::Inference("test".into()))?;
            Ok(())
        }
        assert!(returns_anyhow().is_err());
    }
}
