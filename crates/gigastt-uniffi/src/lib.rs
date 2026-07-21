//! UniFFI bindings for gigastt — idiomatic Swift, Kotlin, and Python generated
//! from one Rust source.
//!
//! Wraps the synchronous `gigastt-core` engine: models are side-loaded (no HTTP
//! download) and inference uses the blocking pool path (no tokio runtime).
//! Errors are typed (`GigasttFfiError`) and map to Swift `throws` / Kotlin
//! exceptions / Python exceptions instead of the C-ABI's NULL sentinels; objects
//! are reference-counted, so there is no manual free.

use std::sync::{Arc, Mutex};

use gigastt_core::inference::{
    Engine as CoreEngine, OwnedReservation, SessionTriplet, StreamingState, audio,
};

uniffi::setup_scaffolding!();

/// Errors surfaced across the binding boundary.
///
/// Deliberately *not* named `GigasttError`: the core crate already exports a
/// public [`gigastt_core::error::GigasttError`], and two same-named public error
/// enums invite mix-ups. The `Ffi` marker also makes it clear at call sites
/// that this is the flat, string-carrying mirror intended for foreign bindings.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum GigasttFfiError {
    /// The model directory is missing files or the engine failed to load.
    #[error("model not found or failed to load: {msg}")]
    ModelNotFound { msg: String },
    /// The audio could not be decoded (unsupported format / corrupt input).
    #[error("invalid or undecodable audio: {msg}")]
    InvalidAudio { msg: String },
    /// No inference session triplet was available (pool closed/exhausted).
    #[error("inference session pool exhausted")]
    PoolExhausted,
    /// Inference failed at runtime.
    #[error("inference failed: {msg}")]
    Inference { msg: String },
    /// A caller-supplied argument was invalid.
    #[error("invalid argument: {msg}")]
    InvalidArgument { msg: String },
}

impl From<gigastt_core::error::GigasttError> for GigasttFfiError {
    fn from(e: gigastt_core::error::GigasttError) -> Self {
        use gigastt_core::error::GigasttError as Core;
        match e {
            e @ Core::ModelLoad { .. } => GigasttFfiError::ModelNotFound { msg: e.to_string() },
            Core::InvalidAudio { reason } => GigasttFfiError::InvalidAudio { msg: reason },
            Core::InvalidInput { message } => GigasttFfiError::InvalidArgument { msg: message },
            Core::Io(err) => GigasttFfiError::Inference {
                msg: err.to_string(),
            },
            Core::Inference { source } => GigasttFfiError::Inference {
                msg: source.to_string(),
            },
            // The core enum is `#[non_exhaustive]`, so a wildcard arm is
            // required here. Every current core variant is listed explicitly
            // above: removing or renaming one fails compilation in this match
            // instead of silently drifting the bindings.
            other => GigasttFfiError::Inference {
                msg: other.to_string(),
            },
        }
    }
}

/// A recognized word with timing, confidence, and optional speaker label.
#[derive(uniffi::Record)]
pub struct Word {
    pub text: String,
    pub start_s: f64,
    pub end_s: f64,
    pub confidence: f32,
    pub speaker: Option<u32>,
}

/// A transcript segment (interim or final) with its words.
#[derive(uniffi::Record)]
pub struct TranscriptSegment {
    pub text: String,
    pub words: Vec<Word>,
    pub is_final: bool,
}

/// The full result of transcribing a file.
#[derive(uniffi::Record)]
pub struct Transcript {
    pub text: String,
    pub words: Vec<Word>,
    pub duration_s: f64,
}

fn word_from(w: gigastt_core::inference::WordInfo) -> Word {
    Word {
        text: w.word,
        start_s: w.start,
        end_s: w.end,
        confidence: w.confidence,
        speaker: w.speaker,
    }
}

fn segment_from(s: gigastt_core::inference::TranscriptSegment) -> TranscriptSegment {
    TranscriptSegment {
        text: s.text,
        words: s.words.into_iter().map(word_from).collect(),
        is_final: s.is_final,
    }
}

/// On-device speech-recognition engine. Reference-counted and thread-safe; share
/// one instance across threads / streams.
#[derive(uniffi::Object)]
pub struct Engine {
    inner: CoreEngine,
}

#[uniffi::export]
impl Engine {
    /// Load the GigaAM v3 model from `model_dir` with the default pool size.
    #[uniffi::constructor]
    pub fn new(model_dir: String) -> Result<Arc<Self>, GigasttFfiError> {
        let inner = CoreEngine::load(&model_dir)
            .map_err(|e| GigasttFfiError::ModelNotFound { msg: e.to_string() })?;
        Ok(Arc::new(Self { inner }))
    }

    /// Load with an explicit session-pool size (concurrent inferences). On weak
    /// devices use `1` to bound resident memory.
    #[uniffi::constructor]
    pub fn new_with_pool_size(
        model_dir: String,
        pool_size: u32,
    ) -> Result<Arc<Self>, GigasttFfiError> {
        let inner = CoreEngine::load_with_pool_size(&model_dir, pool_size as usize)
            .map_err(|e| GigasttFfiError::ModelNotFound { msg: e.to_string() })?;
        Ok(Arc::new(Self { inner }))
    }

    /// Transcribe an audio file (WAV / MP3 / M4A / OGG / FLAC) to text + word
    /// timings. Blocks until inference completes.
    pub fn transcribe_file(&self, path: String) -> Result<Transcript, GigasttFfiError> {
        let mut guard = self
            .inner
            .pool
            .checkout_blocking()
            .map_err(|_| GigasttFfiError::PoolExhausted)?;
        let r = self
            .inner
            .transcribe_file(&path, &mut guard)
            .map_err(GigasttFfiError::from)?;
        Ok(Transcript {
            text: r.text,
            words: r.words.into_iter().map(word_from).collect(),
            duration_s: r.duration_s,
        })
    }
}

struct StreamInner {
    state: StreamingState,
    reservation: OwnedReservation<SessionTriplet>,
}

/// A streaming transcription session. Holds one pool triplet for its lifetime
/// (returned to the pool when this object is dropped).
#[derive(uniffi::Object)]
pub struct Stream {
    engine: Arc<Engine>,
    inner: Mutex<StreamInner>,
}

#[uniffi::export]
impl Stream {
    /// Open a streaming session against `engine`.
    #[uniffi::constructor]
    pub fn new(engine: Arc<Engine>) -> Result<Arc<Self>, GigasttFfiError> {
        let guard = engine
            .inner
            .pool
            .checkout_blocking()
            .map_err(|_| GigasttFfiError::PoolExhausted)?;
        let reservation = guard.into_owned();
        let state = engine.inner.create_state(false);
        Ok(Arc::new(Self {
            engine,
            inner: Mutex::new(StreamInner { state, reservation }),
        }))
    }

    /// Feed a chunk of little-endian mono PCM16 audio. `sample_rate` is
    /// resampled to 16 kHz internally. Returns any segments produced.
    pub fn process_chunk(
        &self,
        pcm16: Vec<u8>,
        sample_rate: u32,
    ) -> Result<Vec<TranscriptSegment>, GigasttFfiError> {
        let mut guard = self.inner.lock().expect("stream mutex poisoned");
        let StreamInner { state, reservation } = &mut *guard;

        let mut samples: Vec<f32> = pcm16
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect();

        if sample_rate != 16000 {
            audio::resample_with_cache(
                samples,
                audio::SampleRate(sample_rate),
                audio::SampleRate(16000),
                &mut state.resampler,
                &mut state.resample_output_buf,
            )
            .map_err(|e| GigasttFfiError::InvalidAudio { msg: e.to_string() })?;
            samples = std::mem::take(&mut state.resample_output_buf);
        }

        let segs = self
            .engine
            .inner
            .process_chunk(&samples, state, reservation)
            .map_err(GigasttFfiError::from)?;
        Ok(segs.into_iter().map(segment_from).collect())
    }

    /// Flush remaining buffered audio and return any final segment(s).
    pub fn flush(&self) -> Result<Vec<TranscriptSegment>, GigasttFfiError> {
        let mut guard = self.inner.lock().expect("stream mutex poisoned");
        let seg = self.engine.inner.flush_state(&mut guard.state);
        Ok(seg.into_iter().map(segment_from).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gigastt_core::error::GigasttError as CoreError;

    #[test]
    fn test_from_model_load_maps_to_model_not_found() {
        let core = CoreError::ModelLoad {
            path: "/models/encoder.onnx".into(),
            source: None,
        };
        match GigasttFfiError::from(core) {
            GigasttFfiError::ModelNotFound { msg } => {
                assert!(msg.contains("/models/encoder.onnx"));
            }
            other => panic!("expected ModelNotFound, got {other}"),
        }
    }

    #[test]
    fn test_from_invalid_audio_maps_to_invalid_audio() {
        let core = CoreError::InvalidAudio {
            reason: "too long".into(),
        };
        match GigasttFfiError::from(core) {
            GigasttFfiError::InvalidAudio { msg } => assert_eq!(msg, "too long"),
            other => panic!("expected InvalidAudio, got {other}"),
        }
    }

    #[test]
    fn test_from_invalid_input_maps_to_invalid_argument() {
        let core = CoreError::InvalidInput {
            message: "bad sample rate".into(),
        };
        match GigasttFfiError::from(core) {
            GigasttFfiError::InvalidArgument { msg } => assert_eq!(msg, "bad sample rate"),
            other => panic!("expected InvalidArgument, got {other}"),
        }
    }

    #[test]
    fn test_from_io_maps_to_inference() {
        let core = CoreError::Io(std::io::Error::other("disk gone"));
        match GigasttFfiError::from(core) {
            GigasttFfiError::Inference { msg } => assert!(msg.contains("disk gone")),
            other => panic!("expected Inference, got {other}"),
        }
    }

    #[test]
    fn test_from_inference_maps_to_inference_with_source_message() {
        let core = CoreError::Inference {
            source: Box::new(std::io::Error::other("decoder failed")),
        };
        match GigasttFfiError::from(core) {
            GigasttFfiError::Inference { msg } => assert_eq!(msg, "decoder failed"),
            other => panic!("expected Inference, got {other}"),
        }
    }
}
