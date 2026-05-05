//! ONNX Runtime inference engine for GigaAM v3 e2e_rnnt.
//!
//! Loads encoder, decoder, and joiner ONNX models and runs the RNN-T streaming decode loop.

pub mod audio;
mod decode;
mod features;
mod tokenizer;

#[cfg(feature = "diarization")]
use polyvoice::{DiarizationConfig as DiaConfig, OfflineDiarizer, OnlineDiarizer, OnnxEmbeddingExtractor, SampleRate};

#[cfg(all(feature = "coreml", feature = "cuda"))]
compile_error!("Features `coreml` and `cuda` are mutually exclusive. Choose one.");

use anyhow::Context;
#[cfg(any(feature = "coreml", feature = "cuda"))]
use ort::ep;
use ort::session::Session;
use ort::value::TensorRef;
use serde::Serialize;
use std::ops::{Deref, DerefMut};
use std::path::Path;

use crate::error::GigasttError;

use features::MelSpectrogram;
use tokenizer::Tokenizer;

/// Number of mel frequency bins used for spectrogram features.
pub const N_MELS: usize = 64;
/// FFT window size in samples (320 samples = 20ms at 16kHz).
pub const N_FFT: usize = 320;
/// Hop length between consecutive FFT frames in samples (160 samples = 10ms at 16kHz).
pub const HOP_LENGTH: usize = 160;
/// Hidden dimension of the RNN-T prediction (decoder) network.
pub const PRED_HIDDEN: usize = 320;

fn ort_err(e: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

pub(crate) fn now_timestamp() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

/// Encoder time subsampling factor (4 frames → 1 encoder output frame).
const ENCODER_SUBSAMPLING: usize = 4;
/// Seconds per encoder frame (HOP_LENGTH * ENCODER_SUBSAMPLING / 16000 = 0.04s).
const SECONDS_PER_FRAME: f64 = (HOP_LENGTH as f64 * ENCODER_SUBSAMPLING as f64) / 16000.0;

/// Default number of session triplets in the pool.
#[cfg(target_os = "android")]
const DEFAULT_POOL_SIZE: usize = 1;
#[cfg(not(target_os = "android"))]
const DEFAULT_POOL_SIZE: usize = 4;

/// A set of ONNX sessions for one inference pipeline (encoder + decoder + joiner).
///
/// Moved out of the pool on checkout and returned on checkin.
/// Each triplet is independent and can run inference concurrently with others.
pub struct SessionTriplet {
    pub(crate) encoder: Session,
    pub(crate) decoder: Session,
    pub(crate) joiner: Session,
}

/// Errors returned by [`Pool::checkout`].
#[derive(Debug)]
pub enum PoolError {
    /// The pool was closed (graceful shutdown). All current and future
    /// waiters resolve to this variant; the caller should respond with a
    /// 503 / `pool_closed` to the client.
    Closed,
}

impl std::fmt::Display for PoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PoolError::Closed => write!(f, "session pool is closed"),
        }
    }
}

impl std::error::Error for PoolError {}

/// Pool of pre-loaded items of type `T`.
///
/// `SessionPool = Pool<SessionTriplet>` is the only public instantiation
/// outside this module. Generic `T` exists so the pool semantics can be
/// unit-tested without ONNX models.
///
/// Checkout = pop from the queue, checkin = push back via the
/// [`PoolGuard`] returned by [`checkout`](Self::checkout). The pool size acts
/// as the concurrency limit — no separate semaphore needed. FIFO ordering is
/// preserved because waiters are stored in a queue and served in order.
pub struct Pool<T> {
    inner: std::sync::Arc<PoolInner<T>>,
}

struct PoolInner<T> {
    items: std::sync::Mutex<std::collections::VecDeque<T>>,
    waiters: std::sync::Mutex<std::collections::VecDeque<Waiter<T>>>,
    closed: std::sync::atomic::AtomicBool,
    total: usize,
}

enum Waiter<T> {
    Async(tokio::sync::oneshot::Sender<T>),
    Blocking(std::sync::mpsc::Sender<T>),
}

/// Public alias for the production pool: holds [`SessionTriplet`] instances.
pub type SessionPool = Pool<SessionTriplet>;

impl<T: Send> Pool<T> {
    /// Create a pool pre-filled with the given items.
    pub fn new(items: Vec<T>) -> Self {
        let total = items.len();
        Self {
            inner: std::sync::Arc::new(PoolInner {
                items: std::sync::Mutex::new(std::collections::VecDeque::from(items)),
                waiters: std::sync::Mutex::new(std::collections::VecDeque::new()),
                closed: std::sync::atomic::AtomicBool::new(false),
                total,
            }),
        }
    }

    /// Checkout an item from the pool. Awaits FIFO if none available.
    ///
    /// Returns [`PoolError::Closed`] if the pool was shut down via
    /// [`close`](Self::close) before an item became available.
    pub async fn checkout(&self) -> Result<PoolGuard<T>, PoolError> {
        // Fast path
        {
            let mut items = self.inner.items.lock().unwrap();
            if self.inner.closed.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(PoolError::Closed);
            }
            if let Some(item) = items.pop_front() {
                return Ok(PoolGuard::new(self.inner.clone(), item));
            }
        }

        // Slow path: register as an async waiter
        let (tx, rx) = tokio::sync::oneshot::channel();
        {
            let mut waiters = self.inner.waiters.lock().unwrap();
            if self.inner.closed.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(PoolError::Closed);
            }
            waiters.push_back(Waiter::Async(tx));
        }

        match rx.await {
            Ok(item) => Ok(PoolGuard::new(self.inner.clone(), item)),
            Err(_) => Err(PoolError::Closed),
        }
    }

    /// Synchronous (blocking) checkout. Used by FFI and other synchronous callers.
    pub fn checkout_blocking(&self) -> Result<PoolGuard<T>, PoolError> {
        // Fast path
        {
            let mut items = self.inner.items.lock().unwrap();
            if self.inner.closed.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(PoolError::Closed);
            }
            if let Some(item) = items.pop_front() {
                return Ok(PoolGuard::new(self.inner.clone(), item));
            }
        }

        // Slow path: register as a blocking waiter
        let (tx, rx) = std::sync::mpsc::channel();
        {
            let mut waiters = self.inner.waiters.lock().unwrap();
            if self.inner.closed.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(PoolError::Closed);
            }
            waiters.push_back(Waiter::Blocking(tx));
        }

        match rx.recv() {
            Ok(item) => Ok(PoolGuard::new(self.inner.clone(), item)),
            Err(_) => Err(PoolError::Closed),
        }
    }

    /// Close the pool: all current and future [`checkout`](Self::checkout)
    /// callers resolve to [`PoolError::Closed`]. Used by graceful shutdown.
    /// Idempotent.
    pub fn close(&self) {
        self.inner
            .closed
            .store(true, std::sync::atomic::Ordering::SeqCst);
        // Drain all pending waiters so their receivers get Canceled / RecvError.
        let mut waiters = self.inner.waiters.lock().unwrap();
        waiters.clear();
    }

    /// Total number of items the pool was created with.
    pub fn total(&self) -> usize {
        self.inner.total
    }

    /// Number of currently available (not checked-out) items. O(1).
    pub fn available(&self) -> usize {
        let items = self.inner.items.lock().unwrap();
        items.len()
    }
}

impl<T> PoolInner<T> {
    fn checkin(&self, item: T) {
        if self.closed.load(std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        let mut waiters = self.waiters.lock().unwrap();
        if let Some(waiter) = waiters.pop_front() {
            drop(waiters);
            match waiter {
                Waiter::Async(tx) => {
                    let _ = tx.send(item);
                }
                Waiter::Blocking(tx) => {
                    let _ = tx.send(item);
                }
            }
        } else {
            drop(waiters);
            let mut items = self.items.lock().unwrap();
            items.push_back(item);
        }
    }
}

/// RAII guard that auto-checks-in an item when dropped.
///
/// Returned by [`Pool::checkout`]. Deref to access the inner item.
/// On drop (including panic unwind) the item is returned to the pool;
/// if the pool was closed in the meantime the item is silently dropped.
pub struct PoolGuard<T> {
    inner: Option<std::sync::Arc<PoolInner<T>>>,
    item: Option<T>,
}

impl<T> PoolGuard<T> {
    fn new(inner: std::sync::Arc<PoolInner<T>>, item: T) -> Self {
        Self {
            inner: Some(inner),
            item: Some(item),
        }
    }

    /// Strip the lifetime so the guard can be moved into a `'static`
    /// context (e.g. `tokio::task::spawn_blocking`). Returns the owned
    /// item together with an [`OwnedReservation`] that must receive the
    /// item back via [`OwnedReservation::checkin`] when the blocking task
    /// is done. Forgets the original guard so the inner Drop does not also
    /// try to check-in.
    pub fn into_owned(mut self) -> (T, OwnedReservation<T>) {
        let item = self
            .item
            .take()
            .unwrap_or_else(|| unreachable!("PoolGuard::into_owned called after drop"));
        let inner = self.inner.take().unwrap();
        std::mem::forget(self);
        let reservation = OwnedReservation { inner };
        (item, reservation)
    }
}

impl<T> Deref for PoolGuard<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.item
            .as_ref()
            .unwrap_or_else(|| unreachable!("PoolGuard accessed after item taken"))
    }
}

impl<T> DerefMut for PoolGuard<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.item
            .as_mut()
            .unwrap_or_else(|| unreachable!("PoolGuard accessed after item taken"))
    }
}

impl<T> Drop for PoolGuard<T> {
    fn drop(&mut self) {
        if let (Some(inner), Some(item)) = (self.inner.take(), self.item.take()) {
            inner.checkin(item);
        }
    }
}

/// Owned counterpart to [`PoolGuard`] for `'static` contexts (e.g.
/// `spawn_blocking`). The item must be returned via [`Self::checkin`].
///
/// This is intentionally not a Drop-guard: the blocking task takes ownership
/// of the item (and may even mutate it during inference), so the item must
/// travel back through the closure return path. After a panic the call site
/// is expected to recover the item via `catch_unwind` and call `checkin` to
/// keep the pool full.
pub struct OwnedReservation<T> {
    inner: std::sync::Arc<PoolInner<T>>,
}

impl<T> OwnedReservation<T> {
    /// Return the item to the pool from a synchronous (blocking) context.
    /// Silently drops the item if the pool has been closed.
    pub fn checkin(self, item: T) {
        self.inner.checkin(item);
    }
}

/// Decoder LSTM hidden state persisted across streaming chunks.
///
/// Created via [`DecoderState::new`] or obtained from [`StreamingState::decoder`].
/// Holds the RNN-T prediction network state between decode steps.
#[non_exhaustive]
pub struct DecoderState {
    /// LSTM hidden state vector (length [`PRED_HIDDEN`]).
    pub h: Vec<f32>,
    /// LSTM cell state vector (length [`PRED_HIDDEN`]).
    pub c: Vec<f32>,
    /// Previously emitted token ID (initialized to `blank_id`).
    pub prev_token: i64,
    /// Count of consecutive blank frames (used for endpointing).
    pub consecutive_blanks: usize,
}

impl DecoderState {
    /// Create a new decoder state initialized to zeros with the given blank token ID.
    pub fn new(blank_id: usize) -> Self {
        Self {
            h: vec![0.0; PRED_HIDDEN],
            c: vec![0.0; PRED_HIDDEN],
            prev_token: blank_id as i64,
            consecutive_blanks: 0,
        }
    }
}

/// A recognized word with timing and confidence metadata.
///
/// Produced by the RNN-T decoder during [`Engine::process_chunk`] or [`Engine::transcribe_file`].
/// Timestamps are in seconds relative to the start of the audio stream.
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct WordInfo {
    /// The recognized word text (BPE tokens joined, `▁` stripped).
    pub word: String,
    /// Start time in seconds from the beginning of the audio stream.
    pub start: f64,
    /// End time in seconds from the beginning of the audio stream.
    pub end: f64,
    /// Softmax confidence score (0.0–1.0), averaged over constituent BPE tokens.
    pub confidence: f32,
    /// Speaker label from diarization (zero-based index). Omitted if diarization is disabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<u32>,
}

/// Per-connection streaming state that persists across audio chunks.
///
/// Created via [`Engine::create_state`]. Holds the decoder LSTM state, an audio
/// sample buffer for incomplete frames, and accumulated transcript text/words.
/// Pass this to [`Engine::process_chunk`] for each incoming audio chunk and
/// [`Engine::flush_state`] when the stream ends.
#[non_exhaustive]
pub struct StreamingState {
    /// Decoder LSTM hidden state (persisted across chunks).
    pub decoder: DecoderState,
    /// Leftover audio samples that didn't fill a complete frame.
    pub audio_buffer: Vec<f32>,
    /// Accumulated transcript builder (reset on endpointing).
    pub assembler: TranscriptAssembler,
    /// Total encoder frames processed so far (for absolute timestamp offset).
    pub total_frames: usize,
    /// Optional cached resampler for non-16kHz streams.
    pub resampler: Option<rubato::SincFixedIn<f32>>,
    /// Reusable FFT buffer for mel spectrogram (avoids per-chunk allocation).
    pub mel_fft_input: Vec<rustfft::num_complex::Complex<f32>>,
    /// Reusable power spectrum buffer for mel spectrogram.
    pub mel_power: Vec<f32>,
    /// Diarization state (present only when diarization is enabled).
    #[cfg(feature = "diarization")]
    pub diarization_state: Option<OnlineDiarizer>,
}

/// Audio feature extraction pipeline.
///
/// Owns the [`MelSpectrogram`] and handles audio buffering, resampling,
/// and log-mel feature extraction. Extracted so `Engine` does not need to
/// own the low-level signal-processing details directly.
pub struct FeatureExtractor {
    mel: MelSpectrogram,
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureExtractor {
    pub fn new() -> Self {
        Self {
            mel: MelSpectrogram::new(),
        }
    }

    /// Prepare incoming samples (append to buffer, return a complete frame if available).
    pub fn prepare_buffer(&self, samples: &[f32], audio_buffer: &mut Vec<f32>) -> Option<Vec<f32>> {
        audio::prepare_audio_buffer(samples, audio_buffer)
    }

    /// Compute log-mel features from 16 kHz f32 samples, reusing state buffers.
    pub fn compute_mel(
        &self,
        samples: &[f32],
        fft_buf: &mut Vec<rustfft::num_complex::Complex<f32>>,
        power_buf: &mut Vec<f32>,
    ) -> (Vec<f32>, usize) {
        self.mel.compute_with_buffers(samples, fft_buf, power_buf)
    }

    /// One-shot mel computation (for file transcription where buffer reuse is unnecessary).
    pub fn compute(&self, samples: &[f32]) -> (Vec<f32>, usize) {
        self.mel.compute(samples)
    }
}

/// Streaming transcript assembler.
///
/// Accumulates recognized words and builds partial / final [`TranscriptSegment`]
/// payloads. Separated from `Engine` so the segment-building policy can be tested
/// in isolation without loading ONNX models.
pub struct TranscriptAssembler {
    text: String,
    words: Vec<WordInfo>,
}

impl Default for TranscriptAssembler {
    fn default() -> Self {
        Self::new()
    }
}

impl TranscriptAssembler {
    pub fn new() -> Self {
        Self {
            text: String::new(),
            words: Vec::new(),
        }
    }

    /// Append new words to the accumulated transcript.
    pub fn append(&mut self, new_words: Vec<WordInfo>) {
        for w in &new_words {
            if !self.text.is_empty() {
                self.text.push(' ');
            }
            self.text.push_str(&w.word);
        }
        self.words.extend(new_words);
    }

    /// Build a **final** segment and reset internal accumulation.
    pub fn finalize(&mut self, timestamp: f64) -> TranscriptSegment {
        TranscriptSegment {
            text: std::mem::take(&mut self.text),
            words: std::mem::take(&mut self.words),
            is_final: true,
            timestamp,
        }
    }

    /// Build a **partial** segment from current accumulation without resetting.
    pub fn partial(&self, timestamp: f64) -> TranscriptSegment {
        TranscriptSegment {
            text: self.text.clone(),
            words: self.words.clone(),
            is_final: false,
            timestamp,
        }
    }

    /// True if no words have been accumulated yet.
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
}

/// ONNX Runtime inference engine for GigaAM v3 e2e_rnnt.
///
/// Thread-safe: inference sessions live in a [`SessionPool`] so `Engine` can be
/// shared across connections via `Arc<Engine>`. The pool size acts as the
/// concurrency limit — no separate semaphore needed. Typical usage:
///
/// ```ignore
/// let engine = Engine::load("~/.gigastt/models")?;
/// let mut guard = engine.pool.checkout().await?;
/// let text = engine.transcribe_file("audio.wav", &mut guard)?;
/// // guard is returned to the pool on drop
/// ```
///
/// For streaming recognition, use [`create_state`](Engine::create_state) +
/// [`process_chunk`](Engine::process_chunk) + [`flush_state`](Engine::flush_state).
pub struct Engine {
    /// Pool of ONNX session triplets for concurrent inference.
    pub pool: SessionPool,
    tokenizer: Tokenizer,
    features: FeatureExtractor,
    /// Whether the INT8 quantized encoder is in use.
    int8: bool,
    /// Speaker encoder for diarization (None if model file is absent).
    #[cfg(feature = "diarization")]
    pub speaker_encoder: Option<OnnxEmbeddingExtractor>,
}

impl Engine {
    /// Whether the INT8 quantized encoder is loaded.
    pub fn is_int8(&self) -> bool {
        self.int8
    }

    /// Size of the BPE vocabulary the loaded tokenizer covers. Exposed so the
    /// REST `/v1/models` handler can report the real value instead of a
    /// hardcoded literal that would drift if the upstream model rev changes.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }

    /// Load ONNX models from the given directory and create an inference engine.
    ///
    /// Creates a pool of [`DEFAULT_POOL_SIZE`] session triplets for concurrent inference.
    /// Expects files: `v3_e2e_rnnt_encoder.onnx` (or `_int8.onnx`), `v3_e2e_rnnt_decoder.onnx`,
    /// `v3_e2e_rnnt_joint.onnx`, and `v3_e2e_rnnt_vocab.txt`.
    ///
    /// # Errors
    ///
    /// Returns [`GigasttError::ModelLoad`] if model files are missing or ONNX session creation fails.
    pub fn load(model_dir: &str) -> Result<Self, GigasttError> {
        Self::load_with_pool_size(model_dir, DEFAULT_POOL_SIZE)
    }

    /// Load ONNX models with a custom pool size.
    pub fn load_with_pool_size(model_dir: &str, pool_size: usize) -> Result<Self, GigasttError> {
        let dir = Path::new(model_dir);
        if !dir.join("v3_e2e_rnnt_encoder.onnx").exists() {
            return Err(GigasttError::ModelLoad {
                path: model_dir.to_string(),
                source: None,
            });
        }
        Self::load_inner(dir, model_dir, pool_size).map_err(|e| GigasttError::ModelLoad {
            path: model_dir.to_string(),
            source: Some(e.into()),
        })
    }

    /// Load a single set of encoder/decoder/joiner ONNX sessions from disk.
    fn load_sessions(
        dir: &Path,
        prepacked: &ort::session::builder::PrepackedWeights,
    ) -> anyhow::Result<(Session, Session, Session)> {
        let encoder_path = if dir.join("v3_e2e_rnnt_encoder_int8.onnx").exists() {
            dir.join("v3_e2e_rnnt_encoder_int8.onnx")
        } else {
            dir.join("v3_e2e_rnnt_encoder.onnx")
        };

        #[cfg(feature = "coreml")]
        let (encoder, decoder, joiner) = {
            // CoreML has its own cache (`coreml_cache/`) for compiled subgraphs.
            // We do NOT call `.with_optimized_model_path(...)` here: CoreML EP
            // replaces part of the graph with compiled nodes that cannot be
            // re-serialized as ONNX, and ORT errors out with
            // `Unable to serialize model as it contains compiled nodes`
            // on macOS 14+. The CoreML cache below is sufficient.
            let cache_dir = dir.join("coreml_cache");
            let coreml_ep = ep::CoreML::default()
                .with_compute_units(ep::coreml::ComputeUnits::CPUAndNeuralEngine)
                .with_specialization_strategy(ep::coreml::SpecializationStrategy::FastPrediction)
                .with_model_cache_dir(cache_dir.to_string_lossy())
                .build();

            let cpu_fallback = ort::execution_providers::CPUExecutionProvider::default();
            let eps = [coreml_ep.clone(), cpu_fallback.into()];
            let encoder = Session::builder()
                .map_err(ort_err)?
                .with_prepacked_weights(prepacked)
                .map_err(ort_err)?
                .with_execution_providers(&eps)
                .map_err(ort_err)?
                .commit_from_file(&encoder_path)
                .map_err(ort_err)?;
            let decoder = Session::builder()
                .map_err(ort_err)?
                .with_prepacked_weights(prepacked)
                .map_err(ort_err)?
                .with_execution_providers(&eps)
                .map_err(ort_err)?
                .commit_from_file(dir.join("v3_e2e_rnnt_decoder.onnx"))
                .map_err(ort_err)?;
            let joiner = Session::builder()
                .map_err(ort_err)?
                .with_prepacked_weights(prepacked)
                .map_err(ort_err)?
                .with_execution_providers(&eps)
                .map_err(ort_err)?
                .commit_from_file(dir.join("v3_e2e_rnnt_joint.onnx"))
                .map_err(ort_err)?;
            (encoder, decoder, joiner)
        };

        #[cfg(feature = "cuda")]
        let (encoder, decoder, joiner) = {
            // CUDA EP compiles subgraphs that cannot be re-serialized as ONNX,
            // so we do NOT call `.with_optimized_model_path(...)` here — same
            // reason as the CoreML block. ORT's CUDA EP keeps its own caches
            // internally.
            let cuda_ep = ep::CUDA::default().build();

            let cpu_fallback = ort::execution_providers::CPUExecutionProvider::default();
            let eps = [cuda_ep.clone(), cpu_fallback.into()];
            let encoder = Session::builder()
                .map_err(ort_err)?
                .with_prepacked_weights(prepacked)
                .map_err(ort_err)?
                .with_execution_providers(&eps)
                .map_err(ort_err)?
                .commit_from_file(&encoder_path)
                .map_err(ort_err)?;
            let decoder = Session::builder()
                .map_err(ort_err)?
                .with_prepacked_weights(prepacked)
                .map_err(ort_err)?
                .with_execution_providers(&eps)
                .map_err(ort_err)?
                .commit_from_file(dir.join("v3_e2e_rnnt_decoder.onnx"))
                .map_err(ort_err)?;
            let joiner = Session::builder()
                .map_err(ort_err)?
                .with_prepacked_weights(prepacked)
                .map_err(ort_err)?
                .with_execution_providers(&eps)
                .map_err(ort_err)?
                .commit_from_file(dir.join("v3_e2e_rnnt_joint.onnx"))
                .map_err(ort_err)?;
            (encoder, decoder, joiner)
        };

        #[cfg(not(any(feature = "coreml", feature = "cuda")))]
        let (encoder, decoder, joiner) = {
            let cache_dir = dir.join("optimized_cache");
            std::fs::create_dir_all(&cache_dir).ok();
            let cpu_fallback = ort::execution_providers::CPUExecutionProvider::default();
            let eps = [cpu_fallback.into()];
            let encoder = Session::builder()
                .map_err(ort_err)?
                .with_prepacked_weights(prepacked)
                .map_err(ort_err)?
                .with_intra_threads(1)
                .map_err(ort_err)?
                .with_inter_threads(1)
                .map_err(ort_err)?
                .with_optimized_model_path(cache_dir.join("encoder_optimized.onnx"))
                .map_err(ort_err)?
                .with_execution_providers(&eps)
                .map_err(ort_err)?
                .commit_from_file(&encoder_path)
                .map_err(ort_err)?;
            let decoder = Session::builder()
                .map_err(ort_err)?
                .with_prepacked_weights(prepacked)
                .map_err(ort_err)?
                .with_intra_threads(1)
                .map_err(ort_err)?
                .with_inter_threads(1)
                .map_err(ort_err)?
                .commit_from_file(dir.join("v3_e2e_rnnt_decoder.onnx"))
                .map_err(ort_err)?;
            let joiner = Session::builder()
                .map_err(ort_err)?
                .with_prepacked_weights(prepacked)
                .map_err(ort_err)?
                .with_intra_threads(1)
                .map_err(ort_err)?
                .with_inter_threads(1)
                .map_err(ort_err)?
                .commit_from_file(dir.join("v3_e2e_rnnt_joint.onnx"))
                .map_err(ort_err)?;
            (encoder, decoder, joiner)
        };

        Ok((encoder, decoder, joiner))
    }

    fn load_inner(dir: &Path, model_dir: &str, pool_size: usize) -> anyhow::Result<Self> {
        let is_int8 = dir.join("v3_e2e_rnnt_encoder_int8.onnx").exists();
        if is_int8 {
            tracing::info!("Using INT8 quantized encoder");
        }

        tracing::info!("Loading ONNX models from {model_dir} (pool_size={pool_size})...");

        #[cfg(feature = "coreml")]
        tracing::info!("Using CoreML execution provider (Neural Engine + CPU)");
        #[cfg(feature = "cuda")]
        tracing::info!("Using CUDA execution provider (falls back to CPU if unavailable)");
        #[cfg(not(any(feature = "coreml", feature = "cuda")))]
        tracing::info!("Using CPU execution provider");

        // Shared prepacked weights container (Arc-based, thread-safe)
        let prepacked = ort::session::builder::PrepackedWeights::new();

        let triplets: Vec<SessionTriplet> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..pool_size)
                .map(|i| {
                    let pp = &prepacked;
                    s.spawn(move || {
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            tracing::info!(
                                "Loading session triplet {}/{pool_size} (shared weights)",
                                i + 1
                            );
                            let (encoder, decoder, joiner) = Self::load_sessions(dir, pp)?;
                            Ok(SessionTriplet {
                                encoder,
                                decoder,
                                joiner,
                            })
                        }))
                        .map_err(|_| anyhow::anyhow!("model loading thread panicked"))?
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| match h.join() {
                    Ok(r) => r,
                    Err(_) => Err(anyhow::anyhow!("model loading thread panicked")),
                })
                .collect::<anyhow::Result<Vec<_>>>()
        })?;

        let tokenizer = Tokenizer::load(&dir.join("v3_e2e_rnnt_vocab.txt"))?;
        let features = FeatureExtractor::new();

        tracing::info!(
            "Models loaded (vocab_size={}, pool_size={pool_size})",
            tokenizer.vocab_size()
        );

        #[cfg(feature = "diarization")]
        let speaker_encoder = {
            let model_path = dir.join("wespeaker_resnet34.onnx");
            if model_path.exists() {
                match OnnxEmbeddingExtractor::new(&model_path, 256, 24000, 4) {
                    Ok(enc) => {
                        tracing::info!("Speaker encoder loaded (diarization available)");
                        Some(enc)
                    }
                    Err(e) => {
                        tracing::warn!("Speaker encoder not loaded, diarization unavailable: {e:#}");
                        None
                    }
                }
            } else {
                tracing::warn!("wespeaker_resnet34.onnx not found, diarization unavailable");
                None
            }
        };

        Ok(Self {
            pool: SessionPool::new(triplets),
            tokenizer,
            features,
            int8: is_int8,
            #[cfg(feature = "diarization")]
            speaker_encoder,
        })
    }

    /// Return `true` if a speaker encoder is loaded and diarization is available.
    #[cfg(feature = "diarization")]
    pub fn has_speaker_encoder(&self) -> bool {
        self.speaker_encoder.is_some()
    }

    /// Create a fresh streaming state for a new connection.
    ///
    /// Pass `diarization_enabled = true` to activate speaker diarization for
    /// this session. Without the `diarization` feature or a loaded speaker
    /// encoder, the flag is silently ignored (a `warn!` is emitted when the
    /// caller asked for diarization but the build does not support it, so the
    /// contract mismatch is visible in logs).
    pub fn create_state(&self, diarization_enabled: bool) -> StreamingState {
        #[cfg(feature = "diarization")]
        let diarization_state = if diarization_enabled && self.speaker_encoder.is_some() {
            Some(OnlineDiarizer::new(DiaConfig {
                threshold: 0.5,
                max_speakers: 64,
                window_secs: 1.5,
                hop_secs: 0.75,
                min_speech_secs: 0.25,
                max_gap_secs: 0.5,
                sample_rate: SampleRate::new(16000).expect("16kHz is valid"),
            }))
        } else {
            None
        };

        #[cfg(not(feature = "diarization"))]
        if diarization_enabled {
            tracing::warn!(
                "diarization_enabled=true ignored: build lacks the `diarization` feature"
            );
        }

        StreamingState {
            decoder: DecoderState::new(self.tokenizer.blank_id()),
            audio_buffer: Vec::new(),
            assembler: TranscriptAssembler::new(),
            total_frames: 0,
            resampler: None,
            mel_fft_input: Vec::new(),
            mel_power: Vec::new(),
            #[cfg(feature = "diarization")]
            diarization_state,
        }
    }

    /// Process a chunk of 16kHz f32 audio samples and return any new transcript segments.
    ///
    /// Returns [`TranscriptSegment`] with `is_final == false` during speech (Partial),
    /// and `is_final == true` on endpointing (~600ms silence detected).
    /// Streaming state (LSTM hidden/cell, leftover audio, accumulated text) is maintained in `state`.
    ///
    /// # Errors
    ///
    /// Returns [`GigasttError::Inference`] if the ONNX runtime fails.
    pub fn process_chunk(
        &self,
        samples: &[f32],
        state: &mut StreamingState,
        triplet: &mut SessionTriplet,
    ) -> Result<Vec<TranscriptSegment>, GigasttError> {
        if samples.is_empty() {
            return Ok(vec![]);
        }

        // Keep a copy of the 16kHz samples for diarization before the buffer
        // logic potentially pads/realigns them. Skip allocation when diarization
        // is not active for this session.
        #[cfg(feature = "diarization")]
        let samples_16k_copy = if state.diarization_state.is_some() {
            Some(samples.to_vec())
        } else {
            None
        };

        let samples = match self
            .features
            .prepare_buffer(samples, &mut state.audio_buffer)
        {
            Some(s) => s,
            None => return Ok(vec![]),
        };
        let samples = &samples[..];

        let mel_start = std::time::Instant::now();
        let (features, num_frames) =
            self.features
                .compute_mel(samples, &mut state.mel_fft_input, &mut state.mel_power);
        tracing::debug!(
            elapsed_us = mel_start.elapsed().as_micros() as u64,
            "mel_compute"
        );
        if num_frames == 0 {
            return Ok(vec![]);
        }

        #[cfg_attr(not(feature = "diarization"), allow(unused_mut))]
        let (mut new_words, endpoint) = self
            .run_inference(
                triplet,
                &features,
                num_frames,
                &mut state.decoder,
                state.total_frames,
            )
            .map_err(|e| GigasttError::Inference { source: e.into() })?;
        state.total_frames += num_frames;

        // --- Diarization: feed audio to OnlineDiarizer and assign speakers ---
        #[cfg(feature = "diarization")]
        if let (Some(dia), Some(copy), Some(enc)) = (
            state.diarization_state.as_mut(),
            samples_16k_copy.as_ref(),
            self.speaker_encoder.as_ref(),
        ) {
            if let Err(e) = dia.feed(copy, enc).map(|_segs| ()) {
                tracing::warn!("Diarization feed failed: {e:#}");
            }

            // Annotate all words in this chunk with current speaker
            if let Some(speaker_id) = dia.current_speaker() {
                for w in &mut new_words {
                    w.speaker = Some(speaker_id.0);
                }
            }
        }

        if new_words.is_empty() && !endpoint {
            return Ok(vec![]);
        }

        // Accumulate new words
        state.assembler.append(new_words);

        let ts = now_timestamp();

        if endpoint {
            state.decoder.consecutive_blanks = 0;
            Ok(vec![state.assembler.finalize(ts)])
        } else {
            Ok(vec![state.assembler.partial(ts)])
        }
    }

    /// Flush accumulated text as a Final segment (called on Stop/Close).
    pub fn flush_state(&self, state: &mut StreamingState) -> Option<TranscriptSegment> {
        if state.assembler.is_empty() {
            return None;
        }
        Some(state.assembler.finalize(now_timestamp()))
    }

    /// Transcribe an audio file to text (supports WAV, MP3, M4A/AAC, OGG, FLAC).
    ///
    /// Decodes the file to mono 16kHz, runs the full encoder+decoder pipeline,
    /// and returns the recognized text with word-level details and duration.
    ///
    /// # Errors
    ///
    /// Returns [`GigasttError::InvalidAudio`] if the file cannot be decoded, or
    /// [`GigasttError::Inference`] if the ONNX runtime fails.
    pub fn transcribe_file(
        &self,
        path: &str,
        triplet: &mut SessionTriplet,
    ) -> Result<TranscribeResult, GigasttError> {
        let float_samples =
            audio::decode_audio_file(path).map_err(|e| GigasttError::InvalidAudio {
                reason: format!("{e:#}"),
            })?;
        self.transcribe_samples(&float_samples, triplet)
    }

    /// Transcribe audio from raw bytes in memory (no temp file needed).
    ///
    /// Backwards-compatible shim: clones `data` into a [`bytes::Bytes`] and
    /// delegates to [`Engine::transcribe_bytes_shared`]. Prefer the shared
    /// variant on hot paths (REST/SSE) to avoid the extra copy.
    pub fn transcribe_bytes(
        &self,
        data: &[u8],
        triplet: &mut SessionTriplet,
    ) -> Result<TranscribeResult, GigasttError> {
        self.transcribe_bytes_shared(bytes::Bytes::copy_from_slice(data), triplet)
    }

    /// Transcribe audio from a reference-counted [`bytes::Bytes`] buffer
    /// without cloning.
    ///
    /// Reuses the same decode/inference pipeline as [`Engine::transcribe_bytes`]
    /// but hands the buffer straight to symphonia via [`audio::decode_audio_bytes_shared`].
    /// This is the zero-copy entry point used by the REST upload handler so a
    /// 50 MiB `axum::body::Bytes` body stays as a single in-memory buffer
    /// instead of being cloned into a `Vec<u8>` before decode.
    pub fn transcribe_bytes_shared(
        &self,
        data: bytes::Bytes,
        triplet: &mut SessionTriplet,
    ) -> Result<TranscribeResult, GigasttError> {
        let float_samples =
            audio::decode_audio_bytes_shared(data).map_err(|e| GigasttError::InvalidAudio {
                reason: format!("{e:#}"),
            })?;
        self.transcribe_samples(&float_samples, triplet)
    }

    /// Run the full mel + encoder + RNN-T decode pipeline on an already-decoded
    /// 16 kHz f32 sample buffer. Shared tail of [`Engine::transcribe_file`] and
    /// [`Engine::transcribe_bytes_shared`].
    fn transcribe_samples(
        &self,
        float_samples: &[f32],
        triplet: &mut SessionTriplet,
    ) -> Result<TranscribeResult, GigasttError> {
        let duration_s = float_samples.len() as f64 / 16000.0;

        let (features, num_frames) = self.features.compute(float_samples);
        tracing::info!("Extracted {} mel frames", num_frames);

        let mut decoder_state = DecoderState::new(self.tokenizer.blank_id());
        #[cfg_attr(not(feature = "diarization"), allow(unused_mut))]
        let (mut words, _endpoint) = self
            .run_inference(triplet, &features, num_frames, &mut decoder_state, 0)
            .map_err(|e| GigasttError::Inference { source: e.into() })?;
        let text: String = words
            .iter()
            .map(|w| w.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        #[cfg(feature = "diarization")]
        if let Some(ref enc) = self.speaker_encoder {
            let config = DiaConfig::default();
            let diarizer = OfflineDiarizer::new(config);
            match diarizer.run(float_samples, enc) {
                Ok(dia_result) => {
                    for word in &mut words {
                        let mid = (word.start + word.end) / 2.0;
                        if let Some(turn) = dia_result.turns.iter().find(|t| {
                            t.time.start <= mid && t.time.end >= mid
                        }) {
                            word.speaker = Some(turn.speaker.0);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Offline diarization failed: {e:#}");
                }
            }
        }

        Ok(TranscribeResult {
            text,
            words,
            duration_s,
        })
    }

    fn run_inference(
        &self,
        triplet: &mut SessionTriplet,
        features: &[f32],
        num_frames: usize,
        decoder_state: &mut DecoderState,
        frame_offset: usize,
    ) -> anyhow::Result<(Vec<WordInfo>, bool)> {
        // Encoder input: audio_signal [1, 64, num_frames], length [1]
        let signal_tensor = TensorRef::from_array_view(([1_usize, N_MELS, num_frames], features))?;
        let length_data = [num_frames as i64];
        let length_tensor = TensorRef::from_array_view(([1_usize], length_data.as_slice()))?;

        let enc_start = std::time::Instant::now();
        let encoder_outputs = triplet
            .encoder
            .run(ort::inputs![signal_tensor, length_tensor])
            .context("Encoder inference failed")?;
        tracing::info!(
            elapsed_ms = enc_start.elapsed().as_millis() as u64,
            "encoder_inference"
        );

        let (_enc_shape, enc_data) = encoder_outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract encoder output")?;
        let (_len_shape, len_data) = encoder_outputs[1]
            .try_extract_tensor::<i32>()
            .context("Failed to extract encoder length")?;

        let enc_len = usize::try_from(len_data[0]).context("Negative encoder length")?;

        tracing::debug!("Encoder output: {} frames", enc_len);

        // Copy encoder data so we can release the encoder output borrow
        let enc_data_owned: Vec<f32> = enc_data.to_vec();
        drop(encoder_outputs);

        // RNN-T greedy decode
        let dec_start = std::time::Instant::now();
        let result = decode::greedy_decode(
            &mut triplet.decoder,
            &mut triplet.joiner,
            &enc_data_owned,
            enc_len,
            self.tokenizer.blank_id(),
            decoder_state,
        )?;
        tracing::info!(
            elapsed_ms = dec_start.elapsed().as_millis() as u64,
            "greedy_decode"
        );

        // Convert token infos to words with timestamps
        let words = self.tokens_to_words(&result.tokens, frame_offset);

        tracing::info!(
            tokens = result.tokens.len(),
            words = words.len(),
            duration_ms = dec_start.elapsed().as_millis() as u64,
            "Decoded tokens"
        );

        Ok((words, result.endpoint_detected))
    }

    /// Convert decoded tokens into words with timestamps and confidence.
    fn tokens_to_words(&self, tokens: &[decode::TokenInfo], frame_offset: usize) -> Vec<WordInfo> {
        if tokens.is_empty() {
            return Vec::new();
        }

        // Fast path for the no-speech frame case. The word-boundary loop
        // below would also return `Vec::new()` on an empty input, but
        // bailing early skips the allocation of the intermediate state.
        if tokens.is_empty() {
            return Vec::new();
        }

        // Group tokens by words (BPE ▁ marks word boundaries)
        let mut words = Vec::new();
        let mut current_word = String::new();
        let mut word_start_frame: Option<usize> = None;
        let mut word_end_frame: usize = 0;
        let mut word_confidences: Vec<f32> = Vec::new();

        for token in tokens {
            let token_text = self.tokenizer.token_text(token.token_id);
            let is_word_boundary = token_text.starts_with('\u{2581}');

            if is_word_boundary && !current_word.is_empty() {
                // Emit previous word
                let avg_conf: f32 = if word_confidences.is_empty() {
                    1.0
                } else {
                    word_confidences.iter().sum::<f32>() / word_confidences.len() as f32
                };
                words.push(WordInfo {
                    word: std::mem::take(&mut current_word),
                    start: (word_start_frame.unwrap_or(0) + frame_offset) as f64
                        * SECONDS_PER_FRAME,
                    end: (word_end_frame + frame_offset) as f64 * SECONDS_PER_FRAME,
                    confidence: avg_conf,
                    speaker: None,
                });
                current_word.clear();
                word_confidences.clear();
                word_start_frame = None;
            }

            let clean = if let Some(stripped) = token_text.strip_prefix('\u{2581}') {
                stripped
            } else {
                token_text
            };
            if !clean.is_empty() {
                current_word.push_str(clean);
                if word_start_frame.is_none() {
                    word_start_frame = Some(token.frame_index);
                }
                word_end_frame = token.frame_index;
                word_confidences.push(token.confidence);
            }
        }

        // Emit last word
        if !current_word.is_empty() {
            let avg_conf: f32 = if word_confidences.is_empty() {
                1.0
            } else {
                word_confidences.iter().sum::<f32>() / word_confidences.len() as f32
            };
            words.push(WordInfo {
                word: current_word,
                start: (word_start_frame.unwrap_or(0) + frame_offset) as f64 * SECONDS_PER_FRAME,
                end: (word_end_frame + frame_offset) as f64 * SECONDS_PER_FRAME,
                confidence: avg_conf,
                speaker: None,
            });
        }

        words
    }
}

/// Result of file transcription, including word-level details.
#[derive(Debug, Clone, Serialize)]
pub struct TranscribeResult {
    pub text: String,
    pub words: Vec<WordInfo>,
    pub duration_s: f64,
}

/// A transcript segment emitted by the inference engine.
///
/// Partial segments (`is_final == false`) represent interim results that may change.
/// Final segments (`is_final == true`) represent completed utterances after endpointing.
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct TranscriptSegment {
    /// Recognized text for this segment.
    pub text: String,
    /// Individual words with timing and confidence metadata.
    pub words: Vec<WordInfo>,
    /// Whether this segment is final (utterance complete) or partial (interim).
    pub is_final: bool,
    /// Unix timestamp (seconds since epoch) when this segment was produced.
    pub timestamp: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_state_new_zeros() {
        let blank_id = 1024;
        let state = DecoderState::new(blank_id);
        assert!(state.h.iter().all(|&v| v == 0.0));
        assert!(state.c.iter().all(|&v| v == 0.0));
        assert_eq!(state.prev_token, blank_id as i64);
    }

    #[test]
    fn test_decoder_state_dimensions() {
        let state = DecoderState::new(1024);
        assert_eq!(state.h.len(), PRED_HIDDEN);
        assert_eq!(state.c.len(), PRED_HIDDEN);
    }

    #[test]
    fn test_decoder_state_custom_blank_id() {
        let state = DecoderState::new(42);
        assert_eq!(state.prev_token, 42);
    }

    // ---- Pool tests (B.7) ---------------------------------------------------
    //
    // These exercise `Pool<T>` with synthetic items so the contract is
    // observable without loading ONNX models. `SessionPool = Pool<SessionTriplet>`
    // is just an alias, so any property proven here also holds for the real
    // pool.

    #[tokio::test]
    async fn test_pool_guard_returns_triplet_on_normal_drop() {
        let pool = Pool::new(vec![1u32, 2, 3]);
        assert_eq!(pool.available(), 3);
        {
            let _guard = pool.checkout().await.expect("checkout");
            assert_eq!(pool.available(), 2);
        }
        // Dropping the guard returns the item.
        assert_eq!(pool.available(), 3);
    }

    #[tokio::test]
    async fn test_pool_guard_returns_triplet_on_panic_unwind() {
        // The guard's Drop impl runs during unwind, so a panic between
        // checkout and the natural end of scope still restores capacity.
        let pool = std::sync::Arc::new(Pool::new(vec![1u32]));
        assert_eq!(pool.available(), 1);

        let pool_clone = pool.clone();
        let result = tokio::spawn(async move {
            let _guard = pool_clone.checkout().await.expect("checkout");
            assert_eq!(pool_clone.available(), 0);
            panic!("synthetic inference panic");
        })
        .await;
        assert!(result.is_err(), "spawned task must report the panic");

        // Capacity is restored thanks to PoolGuard::drop running on unwind.
        assert_eq!(pool.available(), 1);
    }

    #[tokio::test]
    async fn test_pool_close_wakes_waiters_with_closed() {
        // A waiter blocked in `checkout` after the inventory is exhausted
        // must resolve to PoolError::Closed when `close()` fires. Map the
        // borrowed guard to the `()` success path so the spawn doesn't
        // need to carry the pool's lifetime.
        let pool = std::sync::Arc::new(Pool::<u32>::new(vec![]));
        let waiter = tokio::spawn({
            let pool = pool.clone();
            async move { pool.checkout().await.map(|_g| ()) }
        });

        // Give the waiter a moment to park on the channel.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        pool.close();

        let res = waiter.await.expect("join");
        assert!(matches!(res, Err(PoolError::Closed)));
    }

    #[tokio::test]
    async fn test_pool_fifo_under_contention() {
        // With a single-slot pool and three queued waiters, the order of
        // wake-ups must match the order in which `checkout` was called.
        // The mpsc channel itself is FIFO; the Mutex serializes waiters
        // so ordering is preserved under normal contention.
        let pool = std::sync::Arc::new(Pool::new(vec![0u32]));

        let primary = pool.checkout().await.expect("primary checkout");
        assert_eq!(pool.available(), 0);

        let waker_log = std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let mut handles = Vec::new();
        for id in 0u32..3 {
            let pool = pool.clone();
            let log = waker_log.clone();
            handles.push(tokio::spawn(async move {
                let g = pool.checkout().await.expect("checkout");
                log.lock().await.push(id);
                drop(g);
            }));
            // Stagger spawns so each waiter is parked before the next one
            // is registered with the channel.
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }

        // Release the only inventory slot so the queued waiters can run.
        drop(primary);
        for h in handles {
            h.await.expect("join");
        }

        let log = waker_log.lock().await.clone();
        assert_eq!(log, vec![0, 1, 2], "waiters must wake in FIFO order");
    }

    #[tokio::test]
    async fn test_into_owned_for_spawn_blocking() {
        // `into_owned` strips the lifetime so the item can be moved into a
        // blocking thread, then `OwnedReservation::checkin` returns it.
        let pool = std::sync::Arc::new(Pool::new(vec![String::from("triplet")]));
        let guard = pool.checkout().await.expect("checkout");
        let (item, reservation) = guard.into_owned();

        let item = tokio::task::spawn_blocking(move || {
            // Pretend we're running blocking inference.
            assert_eq!(item, "triplet");
            reservation.checkin(item.clone());
            item
        })
        .await
        .expect("join");

        // After the blocking task returns the item, the pool is full again.
        assert_eq!(pool.available(), 1);
        assert_eq!(item, "triplet");
    }

    #[tokio::test]
    async fn test_pool_close_is_idempotent() {
        // `pool.close()` is wired into the shutdown hook; calling it twice
        // (e.g. shutdown signal + Drop) must not panic.
        let pool = Pool::<u32>::new(vec![]);
        pool.close();
        pool.close();
    }
}
