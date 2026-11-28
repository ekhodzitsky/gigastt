//! Separate optional full-context v4 binding; no legacy Stream compatibility.

use crate::Word;
use napi::bindgen_prelude::*;
use napi_derive::napi;

#[cfg(all(
    feature = "v4-mkl",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
))]
use {
    crate::word_from,
    gigastt_core::{inference::v4::V4Engine as CoreV4Engine, model::TranscriptStyle},
    std::{path::Path, sync::Arc},
};

/// Whether this addon was built with the qualified optional v4 backend.
#[napi]
pub fn v4_supported() -> bool {
    cfg!(all(
        feature = "v4-mkl",
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "gnu"
    ))
}

#[cfg(not(all(
    feature = "v4-mkl",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
)))]
fn unsupported() -> Error {
    Error::new(
        Status::GenericFailure,
        "UnsupportedCapability: v4 requires a Linux x86-64 GNU addon built with v4-mkl".to_owned(),
    )
}

#[cfg(all(
    feature = "v4-mkl",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
))]
fn v4_error(error: gigastt_core::error::GigasttError) -> Error {
    use gigastt_core::error::GigasttError;
    let prefix = match &error {
        GigasttError::ModelLoad { .. } => "ModelNotFound",
        GigasttError::InvalidInput { .. } => "InvalidArgument",
        GigasttError::InvalidAudio { .. } => "InvalidAudio",
        GigasttError::AudioTooLong { .. } => "AudioTooLong",
        GigasttError::Cancelled => "Cancelled",
        _ => "Inference",
    };
    let detail = if matches!(&error, GigasttError::ModelLoad { .. }) {
        format!("{error}; check the exact versioned v4 pack and trusted local MKL runtime")
    } else {
        error.to_string()
    };
    Error::new(Status::GenericFailure, format!("{prefix}: {detail}"))
}

/// Exact emitted v4 token metadata. Confidence is a model score, not calibration.
#[napi(object)]
pub struct V4Token {
    pub id: u32,
    pub frame: u32,
    pub confidence: f64,
}

/// Full-context v4 result, with literal decoded text separate from word groups.
#[napi(object)]
pub struct V4Transcript {
    pub style: String,
    pub text: String,
    pub words: Vec<Word>,
    pub tokens: Vec<V4Token>,
    pub duration_s: f64,
    pub confidence: Option<f64>,
}

/// Explicit local v4 RNNT FP32 CPU file engine. No streaming or fallback.
#[napi]
pub struct V4Engine {
    #[cfg(all(
        feature = "v4-mkl",
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "gnu"
    ))]
    inner: Arc<CoreV4Engine>,
    #[cfg(all(
        feature = "v4-mkl",
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "gnu"
    ))]
    style: TranscriptStyle,
}

#[napi]
impl V4Engine {
    /// Load the exact pack and explicit trusted local MKL runtime, selecting
    /// "raw" or "e2e". Unsupported builds return an actionable error.
    #[napi(constructor)]
    pub fn new(pack_dir: String, mkl_runtime_dir: String, style: String) -> Result<Self> {
        #[cfg(all(
            feature = "v4-mkl",
            target_os = "linux",
            target_arch = "x86_64",
            target_env = "gnu"
        ))]
        {
            let style = style
                .parse::<TranscriptStyle>()
                .map_err(|e| Error::new(Status::InvalidArg, format!("InvalidArgument: {e}")))?;
            let inner = CoreV4Engine::load(Path::new(&pack_dir), Path::new(&mkl_runtime_dir))
                .map_err(v4_error)?;
            Ok(Self {
                inner: Arc::new(inner),
                style,
            })
        }
        #[cfg(not(all(
            feature = "v4-mkl",
            target_os = "linux",
            target_arch = "x86_64",
            target_env = "gnu"
        )))]
        {
            let _ = (pack_dir, mkl_runtime_dir, style);
            Err(unsupported())
        }
    }

    /// Recognize a supported container of at most 30 seconds. The native
    /// engine serializes requests; inference runs off the JS event loop.
    #[napi(ts_return_type = "Promise<V4Transcript>")]
    pub fn transcribe_file(&self, path: String) -> AsyncTask<V4TranscribeFileTask> {
        #[cfg(not(all(
            feature = "v4-mkl",
            target_os = "linux",
            target_arch = "x86_64",
            target_env = "gnu"
        )))]
        let _ = path;
        AsyncTask::new(V4TranscribeFileTask {
            #[cfg(all(
                feature = "v4-mkl",
                target_os = "linux",
                target_arch = "x86_64",
                target_env = "gnu"
            ))]
            engine: Arc::clone(&self.inner),
            #[cfg(all(
                feature = "v4-mkl",
                target_os = "linux",
                target_arch = "x86_64",
                target_env = "gnu"
            ))]
            style: self.style,
            #[cfg(all(
                feature = "v4-mkl",
                target_os = "linux",
                target_arch = "x86_64",
                target_env = "gnu"
            ))]
            path,
        })
    }
}

pub struct V4TranscribeFileTask {
    #[cfg(all(
        feature = "v4-mkl",
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "gnu"
    ))]
    engine: Arc<CoreV4Engine>,
    #[cfg(all(
        feature = "v4-mkl",
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "gnu"
    ))]
    style: TranscriptStyle,
    #[cfg(all(
        feature = "v4-mkl",
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "gnu"
    ))]
    path: String,
}

#[napi]
impl Task for V4TranscribeFileTask {
    type Output = V4Transcript;
    type JsValue = V4Transcript;

    fn compute(&mut self) -> Result<Self::Output> {
        #[cfg(all(
            feature = "v4-mkl",
            target_os = "linux",
            target_arch = "x86_64",
            target_env = "gnu"
        ))]
        {
            let r = self
                .engine
                .transcribe_file(Path::new(&self.path), self.style, None)
                .map_err(v4_error)?;
            let tokens = r
                .tokens
                .into_iter()
                .map(|token| -> Result<V4Token> {
                    Ok(V4Token {
                        id: u32::try_from(token.id).map_err(crate::inference_err)?,
                        frame: u32::try_from(token.frame).map_err(crate::inference_err)?,
                        confidence: token.confidence,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(V4Transcript {
                style: r.style.as_str().to_owned(),
                text: r.text,
                words: r.words.into_iter().map(word_from).collect(),
                tokens,
                duration_s: r.duration_s,
                confidence: r.confidence.map(f64::from),
            })
        }
        #[cfg(not(all(
            feature = "v4-mkl",
            target_os = "linux",
            target_arch = "x86_64",
            target_env = "gnu"
        )))]
        {
            Err(unsupported())
        }
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v4_capability_and_explicit_errors() {
        let error = V4Engine::new("unused".into(), "unused".into(), "auto".into())
            .err()
            .expect("unsupported or invalid style must fail before model loading");
        assert!(error.reason.starts_with(if v4_supported() {
            "InvalidArgument:"
        } else {
            "UnsupportedCapability:"
        }));
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
        let error = V4Engine::new(
            dir.path().display().to_string(),
            dir.path().display().to_string(),
            "raw".into(),
        )
        .err()
        .expect("missing versioned pack");
        assert!(error.reason.starts_with("ModelNotFound:"));
    }
}
