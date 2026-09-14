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
    /// Invalid user-supplied parameter or option (not audio-specific).
    #[error("invalid input: {message}")]
    InvalidInput { message: String },
    /// The run was cancelled cooperatively before it finished (client
    /// disconnect, `DELETE /v1/jobs/{id}`, a fired shutdown signal, or the
    /// no-progress inference watchdog). The decode loop observes the abort
    /// signal between tokens/frames and returns this so the pooled session is
    /// released promptly instead of running to completion. Additive: the enum
    /// is `#[non_exhaustive]`.
    #[error("cancelled")]
    Cancelled,
    /// The audio exceeded a duration limit and was rejected before it could
    /// exhaust memory. `observed_secs` is how long the decoded input turned out
    /// to be; `limit_secs` is the ceiling that fired. Two sources trip this: the
    /// opt-in `--max-audio-secs` (default `0` = unlimited), and the fixed safety
    /// ceiling that the whole-buffer paths (diarization, `channels=split` —
    /// including its per-channel Opus decode — and the raw telephony
    /// codecs) keep because they must materialize the entire decoded buffer in
    /// RAM. The default streaming file path, WAVE ingest, the VAD file path, and streamed
    /// OGG/Opus are O(one window) and have no length limit. Additive: the enum
    /// is `#[non_exhaustive]`.
    #[error("audio too long: {observed_secs:.0}s exceeds the maximum of {limit_secs:.0}s")]
    AudioTooLong {
        /// Observed decoded audio length, in seconds.
        observed_secs: f64,
        /// The limit that fired, in seconds.
        limit_secs: f64,
    },
}

impl GigasttError {
    /// Stable, machine-readable error code for wire payloads (WebSocket /
    /// SSE `error` events). Lets both streaming surfaces emit the same code
    /// for the same failure instead of collapsing everything to one generic
    /// string.
    pub fn code(&self) -> &'static str {
        match self {
            GigasttError::ModelLoad { .. } => "model_load_error",
            GigasttError::Inference { .. } => "inference_error",
            GigasttError::InvalidAudio { .. } => "invalid_audio",
            GigasttError::Io(_) => "io_error",
            GigasttError::InvalidInput { .. } => "invalid_input",
            GigasttError::Cancelled => "cancelled",
            GigasttError::AudioTooLong { .. } => "audio_too_long",
        }
    }
}

impl From<crate::runtime::RuntimeError> for GigasttError {
    fn from(err: crate::runtime::RuntimeError) -> Self {
        match err {
            crate::runtime::RuntimeError::LoadFailed { path, message } => GigasttError::ModelLoad {
                path: path.to_string_lossy().into_owned(),
                source: Some(Box::new(std::io::Error::other(message))),
            },
            other => GigasttError::Inference {
                source: Box::new(other),
            },
        }
    }
}

#[cfg(test)]
mod tests;
