//! Separate optional full-context v4 binding.

use crate::Word;
use std::sync::Arc;

#[cfg(all(
    feature = "v4-mkl",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
))]
use {
    crate::word_from,
    gigastt_core::{inference::v4::V4Engine as CoreV4Engine, model::TranscriptStyle},
    std::path::Path,
};

/// Errors specific to the separate v4 API; the existing v3 error enum is unchanged.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum V4BindingError {
    /// The backend is unavailable in this build or on this target.
    #[error("v4 requires a Linux x86-64 GNU build with v4-mkl")]
    UnsupportedCapability,
    /// An explicit caller argument is invalid.
    #[error("invalid argument: {msg}")]
    InvalidArgument { msg: String },
    /// The verified pack or required trusted local MKL runtime cannot be loaded.
    #[error("v4 model not found or failed to load: {msg}")]
    ModelNotFound { msg: String },
    /// Audio is invalid or cannot be decoded.
    #[error("invalid audio: {msg}")]
    InvalidAudio { msg: String },
    /// The audio exceeds the effective full-context duration limit.
    #[error("audio exceeds the v4 duration limit ({limit_secs}s)")]
    AudioTooLong { observed_secs: f64, limit_secs: f64 },
    /// Native inference failed.
    #[error("inference failed: {msg}")]
    Inference { msg: String },
}

#[cfg(all(
    feature = "v4-mkl",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
))]
impl From<gigastt_core::error::GigasttError> for V4BindingError {
    fn from(error: gigastt_core::error::GigasttError) -> Self {
        use gigastt_core::error::GigasttError as Core;
        match error {
            e @ Core::ModelLoad { .. } => Self::ModelNotFound {
                msg: format!(
                    "{e}; check the exact versioned v4 pack and trusted local MKL runtime"
                ),
            },
            Core::InvalidInput { message } => Self::InvalidArgument { msg: message },
            Core::InvalidAudio { reason } => Self::InvalidAudio { msg: reason },
            Core::AudioTooLong {
                observed_secs,
                limit_secs,
            } => Self::AudioTooLong {
                observed_secs,
                limit_secs,
            },
            e => Self::Inference { msg: e.to_string() },
        }
    }
}

/// Whether the optional v4 backend is available in this binding build.
#[uniffi::export]
pub fn v4_supported() -> bool {
    cfg!(all(
        feature = "v4-mkl",
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "gnu"
    ))
}

/// Exact emitted v4 token metadata. Confidence is a model score, not calibration.
#[derive(uniffi::Record)]
pub struct V4Token {
    pub id: u32,
    pub frame: u32,
    pub confidence: f64,
}

/// Full-context v4 result, with literal decoded text separate from word groups.
#[derive(uniffi::Record)]
pub struct V4Transcript {
    pub style: String,
    pub text: String,
    pub words: Vec<Word>,
    pub tokens: Vec<V4Token>,
    pub duration_s: f64,
    pub confidence: Option<f32>,
}

/// Explicit local v4 RNNT FP32 CPU file engine. It cannot open a legacy Stream.
#[derive(uniffi::Object)]
pub struct V4Engine {
    #[cfg(all(
        feature = "v4-mkl",
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "gnu"
    ))]
    inner: CoreV4Engine,
    #[cfg(all(
        feature = "v4-mkl",
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "gnu"
    ))]
    style: TranscriptStyle,
}

#[uniffi::export]
impl V4Engine {
    /// Load an exact versioned pack and explicitly trusted local MKL runtime,
    /// selecting "raw" or "e2e". There is no fallback or hidden download.
    #[uniffi::constructor]
    pub fn new(
        pack_dir: String,
        mkl_runtime_dir: String,
        style: String,
    ) -> Result<Arc<Self>, V4BindingError> {
        #[cfg(all(
            feature = "v4-mkl",
            target_os = "linux",
            target_arch = "x86_64",
            target_env = "gnu"
        ))]
        {
            let style = style
                .parse::<TranscriptStyle>()
                .map_err(|e| V4BindingError::InvalidArgument { msg: e.into() })?;
            let inner = CoreV4Engine::load(Path::new(&pack_dir), Path::new(&mkl_runtime_dir))?;
            Ok(Arc::new(Self { inner, style }))
        }
        #[cfg(not(all(
            feature = "v4-mkl",
            target_os = "linux",
            target_arch = "x86_64",
            target_env = "gnu"
        )))]
        {
            let _ = (pack_dir, mkl_runtime_dir, style);
            Err(V4BindingError::UnsupportedCapability)
        }
    }

    /// Recognize a supported container of at most 30 seconds, synchronously.
    pub fn transcribe_file(&self, path: String) -> Result<V4Transcript, V4BindingError> {
        #[cfg(all(
            feature = "v4-mkl",
            target_os = "linux",
            target_arch = "x86_64",
            target_env = "gnu"
        ))]
        {
            let r = self
                .inner
                .transcribe_file(Path::new(&path), self.style, None)?;
            let tokens = r
                .tokens
                .into_iter()
                .map(|t| -> Result<V4Token, V4BindingError> {
                    let error = |e: std::num::TryFromIntError| V4BindingError::Inference {
                        msg: e.to_string(),
                    };
                    Ok(V4Token {
                        id: u32::try_from(t.id).map_err(error)?,
                        frame: u32::try_from(t.frame).map_err(error)?,
                        confidence: t.confidence,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(V4Transcript {
                style: r.style.as_str().to_owned(),
                text: r.text,
                words: r.words.into_iter().map(word_from).collect(),
                tokens,
                duration_s: r.duration_s,
                confidence: r.confidence,
            })
        }
        #[cfg(not(all(
            feature = "v4-mkl",
            target_os = "linux",
            target_arch = "x86_64",
            target_env = "gnu"
        )))]
        {
            let _ = path;
            Err(V4BindingError::UnsupportedCapability)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v4_capability_and_explicit_errors() {
        let result = V4Engine::new("unused".into(), "unused".into(), "auto".into());
        if v4_supported() {
            assert!(matches!(
                result,
                Err(V4BindingError::InvalidArgument { .. })
            ));
        } else {
            assert!(matches!(result, Err(V4BindingError::UnsupportedCapability)));
        }
    }

    #[cfg(all(
        feature = "v4-mkl",
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "gnu"
    ))]
    #[test]
    fn test_v4_missing_pack_is_not_a_v3_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let result = V4Engine::new(
            dir.path().display().to_string(),
            dir.path().display().to_string(),
            "raw".into(),
        );
        assert!(matches!(result, Err(V4BindingError::ModelNotFound { .. })));
    }
}
