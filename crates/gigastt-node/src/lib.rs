//! napi-rs Node.js bindings for gigastt — idiomatic JS over the synchronous
//! `gigastt-core` engine.
//!
//! Models are side-loaded (no HTTP download) and inference uses the blocking
//! pool path. CPU-bound work runs on a libuv worker thread via napi's
//! `AsyncTask`, so the JS event loop stays responsive and calls return Promises.
//! Errors map to thrown JS `Error`s (no NULL sentinels); objects are
//! garbage-collected (no manual free). onnxruntime is statically linked into the
//! addon (ort default `download-binaries`), so the `.node` is self-contained.

use std::sync::{Arc, Mutex};

use gigastt_core::inference::{
    Engine as CoreEngine, OwnedReservation, SessionTriplet, StreamingState, audio,
};
use napi::bindgen_prelude::*;
use napi_derive::napi;

// --- error mapping ---------------------------------------------------------
// gigastt-core errors flatten to thrown JS Errors. The stable variant name is
// the message prefix (e.g. "PoolExhausted: ...") so JS callers can branch on it;
// this matches the C-ABI/UniFFI error contract across bindings.

fn model_not_found(e: impl std::fmt::Display) -> Error {
    Error::new(Status::GenericFailure, format!("ModelNotFound: {e}"))
}

fn pool_exhausted() -> Error {
    Error::new(
        Status::GenericFailure,
        "PoolExhausted: inference session pool exhausted".to_string(),
    )
}

fn invalid_audio(e: impl std::fmt::Display) -> Error {
    Error::new(Status::GenericFailure, format!("InvalidAudio: {e}"))
}

fn inference_err(e: impl std::fmt::Display) -> Error {
    Error::new(Status::GenericFailure, format!("Inference: {e}"))
}

// --- value types -----------------------------------------------------------

/// A recognized word with timing, confidence, and optional speaker label.
#[napi(object)]
pub struct Word {
    pub text: String,
    pub start_s: f64,
    pub end_s: f64,
    pub confidence: f64,
    pub speaker: Option<u32>,
}

/// A transcript segment (interim or final) with its words.
#[napi(object)]
pub struct TranscriptSegment {
    pub text: String,
    pub words: Vec<Word>,
    pub is_final: bool,
}

/// The full result of transcribing a file.
#[napi(object)]
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
        confidence: w.confidence as f64,
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

// --- Engine ----------------------------------------------------------------

/// On-device speech-recognition engine. Thread-safe; share one instance across
/// streams. Loads the GigaAM v3 model from a side-loaded directory.
#[napi]
pub struct Engine {
    inner: Arc<CoreEngine>,
}

#[napi]
impl Engine {
    /// Load the model from `modelDir`. Pass `poolSize` to bound concurrent
    /// inferences (use `1` on weak devices to cap resident memory).
    #[napi(constructor)]
    pub fn new(model_dir: String, pool_size: Option<u32>) -> Result<Self> {
        let inner = match pool_size {
            Some(n) => CoreEngine::load_with_pool_size(&model_dir, n as usize),
            None => CoreEngine::load(&model_dir),
        }
        .map_err(model_not_found)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Transcribe an audio file (WAV / MP3 / M4A / OGG / FLAC) to text + word
    /// timings. Returns a Promise; inference runs off the event loop.
    #[napi(ts_return_type = "Promise<Transcript>")]
    pub fn transcribe_file(&self, path: String) -> AsyncTask<TranscribeFileTask> {
        AsyncTask::new(TranscribeFileTask {
            engine: self.inner.clone(),
            path,
        })
    }
}

pub struct TranscribeFileTask {
    engine: Arc<CoreEngine>,
    path: String,
}

#[napi]
impl Task for TranscribeFileTask {
    type Output = Transcript;
    type JsValue = Transcript;

    fn compute(&mut self) -> Result<Self::Output> {
        let mut guard = self
            .engine
            .pool
            .checkout_blocking()
            .map_err(|_| pool_exhausted())?;
        let r = self
            .engine
            .transcribe_file(&self.path, &mut guard)
            .map_err(inference_err)?;
        Ok(Transcript {
            text: r.text,
            words: r.words.into_iter().map(word_from).collect(),
            duration_s: r.duration_s,
        })
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
    }
}

// --- Stream ----------------------------------------------------------------

struct StreamInner {
    state: StreamingState,
    reservation: OwnedReservation<SessionTriplet>,
}

/// A streaming transcription session. Holds one pool triplet for its lifetime
/// (returned to the pool when garbage-collected). Chunks are processed in order.
#[napi]
pub struct Stream {
    engine: Arc<CoreEngine>,
    inner: Arc<Mutex<StreamInner>>,
}

#[napi]
impl Stream {
    /// Open a streaming session against `engine`.
    #[napi(constructor)]
    pub fn new(engine: &Engine) -> Result<Self> {
        let guard = engine
            .inner
            .pool
            .checkout_blocking()
            .map_err(|_| pool_exhausted())?;
        let reservation = guard.into_owned();
        let state = engine.inner.create_state(false);
        Ok(Self {
            engine: engine.inner.clone(),
            inner: Arc::new(Mutex::new(StreamInner { state, reservation })),
        })
    }

    /// Feed a chunk of little-endian mono PCM16 audio. `sampleRate` is resampled
    /// to 16 kHz internally. Returns a Promise of any segments produced. Await
    /// each call before sending the next chunk to preserve ordering.
    #[napi(ts_return_type = "Promise<TranscriptSegment[]>")]
    pub fn process_chunk(
        &self,
        pcm16: Uint8Array,
        sample_rate: u32,
    ) -> AsyncTask<ProcessChunkTask> {
        AsyncTask::new(ProcessChunkTask {
            engine: self.engine.clone(),
            inner: self.inner.clone(),
            pcm16: pcm16.to_vec(),
            sample_rate,
        })
    }

    /// Flush remaining buffered audio and return any final segment(s).
    #[napi(ts_return_type = "Promise<TranscriptSegment[]>")]
    pub fn flush(&self) -> AsyncTask<FlushTask> {
        AsyncTask::new(FlushTask {
            engine: self.engine.clone(),
            inner: self.inner.clone(),
        })
    }
}

fn lock_poisoned() -> Error {
    Error::new(
        Status::GenericFailure,
        "Inference: stream mutex poisoned".to_string(),
    )
}

pub struct ProcessChunkTask {
    engine: Arc<CoreEngine>,
    inner: Arc<Mutex<StreamInner>>,
    pcm16: Vec<u8>,
    sample_rate: u32,
}

#[napi]
impl Task for ProcessChunkTask {
    type Output = Vec<TranscriptSegment>;
    type JsValue = Vec<TranscriptSegment>;

    fn compute(&mut self) -> Result<Self::Output> {
        let mut guard = self.inner.lock().map_err(|_| lock_poisoned())?;
        let StreamInner { state, reservation } = &mut *guard;

        let mut samples: Vec<f32> = self
            .pcm16
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect();

        if self.sample_rate != 16000 {
            audio::resample_with_cache(
                samples,
                audio::SampleRate(self.sample_rate),
                audio::SampleRate(16000),
                &mut state.resampler,
                &mut state.resample_output_buf,
            )
            .map_err(invalid_audio)?;
            samples = std::mem::take(&mut state.resample_output_buf);
        }

        let segs = self
            .engine
            .process_chunk(&samples, state, reservation)
            .map_err(inference_err)?;
        Ok(segs.into_iter().map(segment_from).collect())
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
    }
}

pub struct FlushTask {
    engine: Arc<CoreEngine>,
    inner: Arc<Mutex<StreamInner>>,
}

#[napi]
impl Task for FlushTask {
    type Output = Vec<TranscriptSegment>;
    type JsValue = Vec<TranscriptSegment>;

    fn compute(&mut self) -> Result<Self::Output> {
        let mut guard = self.inner.lock().map_err(|_| lock_poisoned())?;
        let seg = self.engine.flush_state(&mut guard.state);
        Ok(seg.into_iter().map(segment_from).collect())
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
    }
}
