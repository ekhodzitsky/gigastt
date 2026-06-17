//! ONNX Runtime inference engine for GigaAM v3 e2e_rnnt.
//!
//! Loads encoder, decoder, and joiner ONNX models and runs the RNN-T streaming decode loop.

pub mod audio;
mod decode;
mod features;
#[cfg(not(feature = "__internals"))]
mod tokenizer;
/// Tokenizer module, exposed for fuzzing/benchmarking under the private
/// `__internals` feature only. Not part of the stable public API.
#[cfg(feature = "__internals")]
pub mod tokenizer;

#[cfg(feature = "diarization")]
use polyvoice::streaming::StreamingPipeline;
// `EmbeddingError`, `EmbeddingExtractor`, and `OnnxEmbeddingExtractor` were
// deprecated in polyvoice 0.7.0 in favour of the v1.0 `polyvoice::embedder`
// `Embedder` trait. We keep them because our per-session diarization path,
// `StreamingPipeline`, is itself bound on the legacy `EmbeddingExtractor` trait
// (`impl<V, E> StreamingPipeline<V, E> where E: EmbeddingExtractor`) — polyvoice
// has not yet wired the new `Embedder` into streaming, and suppresses these same
// warnings crate-wide for that reason. Mirror it here; migrate once the streaming
// pipeline accepts `Embedder`.
#[cfg(feature = "diarization")]
#[allow(deprecated)]
use polyvoice::{
    ClusterConfig, DiarizationConfig as DiaConfig, EmbeddingError, EmbeddingExtractor, EnergyVad,
    OnnxEmbeddingExtractor, Pipeline, VadConfig,
};

#[cfg(feature = "diarization")]
const SPEAKER_EMBEDDING_DIM: usize = 256;
#[cfg(feature = "diarization")]
const SPEAKER_SEGMENT_SAMPLES: usize = 24000;
#[cfg(feature = "diarization")]
const SPEAKER_POOL_SIZE: usize = 4;

/// Adapter that lets a single shared [`OnnxEmbeddingExtractor`] back the
/// per-session [`StreamingPipeline`]s, which take ownership of their extractor.
/// The ONNX session pool inside the extractor is shared across sessions via `Arc`.
#[cfg(feature = "diarization")]
#[allow(deprecated)] // legacy OnnxEmbeddingExtractor — see import note above
pub struct SharedExtractor(std::sync::Arc<OnnxEmbeddingExtractor>);

#[cfg(feature = "diarization")]
#[allow(deprecated)] // legacy EmbeddingExtractor trait — see import note above
impl EmbeddingExtractor for SharedExtractor {
    fn extract(&self, samples: &[f32], config: &DiaConfig) -> Result<Vec<f32>, EmbeddingError> {
        self.0.extract(samples, config)
    }

    fn embedding_dim(&self) -> usize {
        self.0.embedding_dim()
    }
}

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

pub fn now_timestamp() -> f64 {
    use std::sync::OnceLock;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};
    // Anchor the wall-clock epoch to a monotonic `Instant` captured once, then
    // advance from it via `Instant::elapsed()`. Wire-visible timestamps stay
    // epoch-aligned (unchanged contract) but advance monotonically, immune to
    // NTP steps / wall-clock jumps mid-process.
    static ANCHOR: OnceLock<(SystemTime, Instant)> = OnceLock::new();
    let (epoch, start) = ANCHOR.get_or_init(|| (SystemTime::now(), Instant::now()));
    let base = match epoch.duration_since(UNIX_EPOCH) {
        Ok(d) => d.as_secs_f64(),
        Err(e) => {
            tracing::warn!("System clock is before Unix epoch: {e}");
            0.0
        }
    };
    base + start.elapsed().as_secs_f64()
}

/// Encoder time subsampling factor (4 frames → 1 encoder output frame).
const ENCODER_SUBSAMPLING: usize = 4;
/// Seconds per encoder frame (HOP_LENGTH * ENCODER_SUBSAMPLING / 16000 = 0.04s).
const SECONDS_PER_FRAME: f64 = (HOP_LENGTH as f64 * ENCODER_SUBSAMPLING as f64) / 16000.0;

/// Max streaming encoder window before forcing a finalize (samples @16kHz, 5s).
/// Re-decoding the whole window each chunk gives the offline Conformer left
/// context; this cap bounds the per-chunk encoder cost.
const STREAM_MAX_WINDOW_SAMPLES: usize = 16000 * 5;
/// Left-context audio retained across a streaming finalize/slide (samples @16kHz,
/// ~1.5s) so the next window keeps acoustic context instead of restarting cold.
const STREAM_LEFT_CONTEXT_SAMPLES: usize = 16000 * 3 / 2;
/// Decode stride: re-run the encoder only after this much NEW audio has
/// accumulated (samples @16kHz, 0.8s) instead of on every ~100ms chunk.
/// Re-decoding the window is the dominant streaming cost, so the stride keeps
/// the engine real-time; `finish_stream` decodes the sub-stride remainder at EOF.
const STREAM_DECODE_STRIDE_SAMPLES: usize = 16000 * 4 / 5;

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
    items: parking_lot::Mutex<std::collections::VecDeque<T>>,
    waiters: parking_lot::Mutex<std::collections::VecDeque<Waiter<T>>>,
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
                items: parking_lot::Mutex::new(std::collections::VecDeque::from(items)),
                waiters: parking_lot::Mutex::new(std::collections::VecDeque::new()),
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
            let mut items = self.inner.items.lock();
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
            let mut waiters = self.inner.waiters.lock();
            if self.inner.closed.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(PoolError::Closed);
            }
            // Re-check items under the waiters lock to prevent the lost-wakeup
            // race: between releasing items.lock() and acquiring waiters.lock(),
            // another thread may have checked in an item and pushed it back to
            // items because there were no waiters yet.
            let mut items = self.inner.items.lock();
            if let Some(item) = items.pop_front() {
                drop(items);
                drop(waiters);
                return Ok(PoolGuard::new(self.inner.clone(), item));
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
            let mut items = self.inner.items.lock();
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
            let mut waiters = self.inner.waiters.lock();
            if self.inner.closed.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(PoolError::Closed);
            }
            // Same lost-wakeup guard as the async variant.
            let mut items = self.inner.items.lock();
            if let Some(item) = items.pop_front() {
                drop(items);
                drop(waiters);
                return Ok(PoolGuard::new(self.inner.clone(), item));
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
        let mut waiters = self.inner.waiters.lock();
        waiters.clear();
    }

    /// Total number of items the pool was created with.
    pub fn total(&self) -> usize {
        self.inner.total
    }

    /// Number of currently available (not checked-out) items. O(1).
    pub fn available(&self) -> usize {
        let items = self.inner.items.lock();
        items.len()
    }

    /// Number of waiters currently blocked on checkout. O(1).
    pub fn waiters(&self) -> usize {
        let waiters = self.inner.waiters.lock();
        waiters.len()
    }
}

impl<T> PoolInner<T> {
    fn checkin(&self, mut item: T) {
        if self.closed.load(std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        // Retry loop: if the waiter at the front of the queue was abandoned
        // (its receiver was dropped because the checkout future was cancelled
        // via timeout, select!, or abort), we must skip it and try the next
        // one, or return the item to the pool. Without this retry a cancelled
        // waiter permanently leaks a pool slot.
        loop {
            let mut waiters = self.waiters.lock();
            if let Some(waiter) = waiters.pop_front() {
                drop(waiters);
                match waiter {
                    Waiter::Async(tx) => {
                        if let Err(returned_item) = tx.send(item) {
                            item = returned_item;
                            continue;
                        }
                    }
                    Waiter::Blocking(tx) => {
                        if let Err(std::sync::mpsc::SendError(returned_item)) = tx.send(item) {
                            item = returned_item;
                            continue;
                        }
                    }
                }
            } else {
                drop(waiters);
                let mut items = self.items.lock();
                items.push_back(item);
            }
            break;
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
    /// context (e.g. `tokio::task::spawn_blocking`). Returns an
    /// [`OwnedReservation`] that owns the item and automatically returns it
    /// to the pool on drop. Call [`OwnedReservation::checkin`] to return the
    /// item explicitly before the reservation is dropped.
    pub fn into_owned(mut self) -> OwnedReservation<T> {
        let item = self
            .item
            .take()
            .unwrap_or_else(|| unreachable!("PoolGuard::into_owned called after drop"));
        let inner = self.inner.take().unwrap();
        OwnedReservation {
            inner,
            item: Some(item),
        }
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
/// `spawn_blocking`). The item is returned to the pool automatically on drop.
///
/// Call [`Self::checkin`] to return the item explicitly and invalidate the
/// guard. If the reservation is dropped without calling `checkin`, the item
/// is still returned to the pool via the [`Drop`] impl. This guarantees that
/// the pool does not leak slots when a `spawn_blocking` task panics or is
/// cancelled.
pub struct OwnedReservation<T> {
    inner: std::sync::Arc<PoolInner<T>>,
    item: Option<T>,
}

impl<T> std::ops::Deref for OwnedReservation<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.item
            .as_ref()
            .unwrap_or_else(|| unreachable!("OwnedReservation accessed after checkin"))
    }
}

impl<T> std::ops::DerefMut for OwnedReservation<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.item
            .as_mut()
            .unwrap_or_else(|| unreachable!("OwnedReservation accessed after checkin"))
    }
}

impl<T> OwnedReservation<T> {
    /// Return the item to the pool explicitly. After this call the reservation
    /// is empty and its [`Drop`] is a no-op.
    pub fn checkin(mut self) {
        if let Some(item) = self.item.take() {
            self.inner.checkin(item);
        }
    }
}

impl<T> Drop for OwnedReservation<T> {
    fn drop(&mut self) {
        if let Some(item) = self.item.take() {
            self.inner.checkin(item);
        }
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
    /// Absolute sample offset (@16kHz) of `audio_buffer[0]`: how much committed
    /// audio has slid off the front. Drives absolute word timestamps
    /// (encoder-frame offset = this / (HOP_LENGTH * ENCODER_SUBSAMPLING)).
    pub window_start_samples: usize,
    /// Leading samples of `audio_buffer` that are already-emitted left context;
    /// words decoded within this region are suppressed (not re-emitted).
    pub context_samples: usize,
    /// New samples accumulated since the last decode. The encoder re-runs only
    /// once this reaches `STREAM_DECODE_STRIDE_SAMPLES`, then resets to 0 — this
    /// is what keeps the stream real-time (re-decoding the window is the cost).
    pub pending_samples: usize,
    /// Optional cached resampler for non-16kHz streams.
    pub resampler: Option<rubato::Async<f32>>,
    /// Reusable FFT buffer for mel spectrogram (avoids per-chunk allocation).
    pub mel_fft_input: Vec<rustfft::num_complex::Complex<f32>>,
    /// Reusable power spectrum buffer for mel spectrogram.
    pub mel_power: Vec<f32>,
    /// Reusable mel-output buffer (avoids per-chunk allocation).
    pub mel_output: Vec<f32>,
    /// Reusable resampler output buffer (avoids per-chunk allocation).
    pub resample_output_buf: Vec<f32>,
    /// Diarization state (present only when diarization is enabled).
    #[cfg(feature = "diarization")]
    pub diarization_state: Option<StreamingPipeline<EnergyVad, SharedExtractor>>,
}

/// Audio feature extraction pipeline.
///
/// Owns the `MelSpectrogram` and handles audio buffering, resampling,
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
    /// Create a new feature extractor with a freshly initialized mel spectrogram.
    pub fn new() -> Self {
        Self {
            mel: MelSpectrogram::new(),
        }
    }

    /// Prepare incoming samples (append to buffer, return the usable sample count if available).
    pub fn prepare_buffer(&self, samples: &[f32], audio_buffer: &mut Vec<f32>) -> Option<usize> {
        audio::prepare_audio_buffer(samples, audio_buffer)
    }

    /// Compute log-mel features from 16 kHz f32 samples, reusing state buffers.
    pub fn compute_mel(
        &self,
        samples: &[f32],
        fft_buf: &mut Vec<rustfft::num_complex::Complex<f32>>,
        power_buf: &mut Vec<f32>,
        output_buf: &mut Vec<f32>,
    ) -> usize {
        self.mel
            .compute_with_buffers(samples, fft_buf, power_buf, output_buf)
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
    /// Create a new, empty transcript assembler.
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

    /// Replace the accumulated transcript with a freshly decoded hypothesis.
    ///
    /// The sliding-window streaming path re-decodes its whole context window on
    /// every chunk, so it overwrites (rather than appends) the current tail.
    pub fn set_words(&mut self, words: Vec<WordInfo>) {
        self.text = words
            .iter()
            .map(|w| w.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        self.words = words;
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

/// Probe a freshly-built state; on failure, rebuild it once and re-probe.
///
/// `probe` is a runtime self-check, `rebuild` converts the failed state into
/// a replacement (receiving the probe error so it can log the cause). A
/// rebuilt state that still fails the probe is a hard error — there is no
/// second fallback level.
///
/// Extracted from the CoreML runtime-fallback path (issue #42) so the
/// decision logic stays unit-testable without ONNX sessions.
#[cfg_attr(not(feature = "coreml"), allow(dead_code))]
fn probe_or_rebuild<S>(
    state: S,
    probe: impl Fn(&S) -> anyhow::Result<()>,
    rebuild: impl FnOnce(S, anyhow::Error) -> anyhow::Result<S>,
) -> anyhow::Result<S> {
    match probe(&state) {
        Ok(()) => Ok(state),
        Err(probe_err) => {
            let rebuilt = rebuild(state, probe_err)?;
            probe(&rebuilt).context("state failed probe even after rebuild")?;
            Ok(rebuilt)
        }
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
    /// Pool of session triplets for interactive inference (WebSocket + SSE
    /// streaming). REST file transcription uses [`Engine::batch_pool`] when it
    /// is set, so a long batch job can't starve real-time streaming.
    pub pool: SessionPool,
    /// Optional dedicated pool for batch REST file transcription, split off
    /// from `pool` at load time. `None` means REST shares the interactive pool.
    pub batch_pool: Option<SessionPool>,
    tokenizer: Tokenizer,
    features: FeatureExtractor,
    /// Whether the INT8 quantized encoder is in use.
    int8: bool,
    /// Speaker encoder for diarization (None if model file is absent).
    ///
    /// Wrapped in `Arc` so per-session streaming pipelines can share the
    /// underlying ONNX session pool without each owning their own copy.
    #[cfg(feature = "diarization")]
    #[allow(deprecated)] // legacy OnnxEmbeddingExtractor — see import note above
    pub speaker_encoder: Option<std::sync::Arc<OnnxEmbeddingExtractor>>,
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
    /// Creates a pool of `DEFAULT_POOL_SIZE` session triplets for concurrent inference.
    /// Expects files: `v3_e2e_rnnt_encoder.onnx` (or `_int8.onnx`), `v3_e2e_rnnt_decoder.onnx`,
    /// `v3_e2e_rnnt_joint.onnx`, and `v3_e2e_rnnt_vocab.txt`.
    ///
    /// # Errors
    ///
    /// Returns [`GigasttError::ModelLoad`] if model files are missing or ONNX session creation fails.
    pub fn load(model_dir: &str) -> Result<Self, GigasttError> {
        Self::load_with_pool_size(model_dir, DEFAULT_POOL_SIZE)
    }

    /// Load ONNX models with a custom pool size. Requires the *full* pool to
    /// load (every triplet); use [`Engine::load_with_pool_size_min`] to boot on
    /// a partial pool.
    pub fn load_with_pool_size(model_dir: &str, pool_size: usize) -> Result<Self, GigasttError> {
        Self::load_with_pool_size_min(model_dir, pool_size, pool_size)
    }

    /// Load ONNX models with a custom pool size, tolerating a partial pool down
    /// to `min_size` triplets when the rest fail to load. Boots a degraded pool
    /// with a warning when `min_size <= loaded < pool_size`; errors only when
    /// fewer than `min_size` triplets load. `min_size` is clamped to
    /// `1..=pool_size`.
    pub fn load_with_pool_size_min(
        model_dir: &str,
        pool_size: usize,
        min_size: usize,
    ) -> Result<Self, GigasttError> {
        // No batch/stream split: the whole pool is interactive.
        Self::load_with_pools(model_dir, pool_size, min_size, 0)
    }

    /// Load ONNX models splitting the pool into an interactive pool (WebSocket +
    /// SSE) and a dedicated batch pool of `batch_pool_size` triplets for REST
    /// file transcription, so a long batch job can't starve real-time
    /// streaming. `batch_pool_size == 0` disables the split (REST shares the
    /// interactive pool); it is clamped to leave at least one interactive
    /// triplet. Partial-load tolerance follows `min_size` as in
    /// [`Engine::load_with_pool_size_min`].
    pub fn load_with_pools(
        model_dir: &str,
        pool_size: usize,
        min_size: usize,
        batch_pool_size: usize,
    ) -> Result<Self, GigasttError> {
        let dir = Path::new(model_dir);
        if !dir.join("v3_e2e_rnnt_encoder.onnx").exists() {
            return Err(GigasttError::ModelLoad {
                path: model_dir.to_string(),
                source: None,
            });
        }
        Self::load_inner(dir, model_dir, pool_size, min_size, batch_pool_size).map_err(|e| {
            GigasttError::ModelLoad {
                path: model_dir.to_string(),
                source: Some(e.into()),
            }
        })
    }

    /// Path to the preferred encoder model: INT8 quantized if present, FP32 otherwise.
    fn encoder_model_path(dir: &Path) -> std::path::PathBuf {
        if dir.join("v3_e2e_rnnt_encoder_int8.onnx").exists() {
            dir.join("v3_e2e_rnnt_encoder_int8.onnx")
        } else {
            dir.join("v3_e2e_rnnt_encoder.onnx")
        }
    }

    /// Load a single set of encoder/decoder/joiner ONNX sessions from disk,
    /// using the execution provider selected at compile time.
    fn load_sessions(
        dir: &Path,
        prepacked: &ort::session::builder::PrepackedWeights,
    ) -> anyhow::Result<(Session, Session, Session)> {
        #[cfg(feature = "coreml")]
        {
            Self::load_sessions_coreml(dir, prepacked)
        }
        #[cfg(feature = "cuda")]
        {
            Self::load_sessions_cuda(dir, prepacked)
        }
        #[cfg(not(any(feature = "coreml", feature = "cuda")))]
        {
            Self::load_sessions_cpu(dir, prepacked)
        }
    }

    #[cfg(feature = "coreml")]
    fn load_sessions_coreml(
        dir: &Path,
        prepacked: &ort::session::builder::PrepackedWeights,
    ) -> anyhow::Result<(Session, Session, Session)> {
        // CoreML has its own cache (`coreml_cache/`) for compiled subgraphs.
        // We do NOT call `.with_optimized_model_path(...)` here: CoreML EP
        // replaces part of the graph with compiled nodes that cannot be
        // re-serialized as ONNX, and ORT errors out with
        // `Unable to serialize model as it contains compiled nodes`
        // on macOS 14+. The CoreML cache below is sufficient.
        //
        // `with_static_input_shapes(true)` is load-bearing (issue #42): the
        // Conformer encoder has a dynamic time axis, and CoreML-compiled
        // partitions with dynamic shapes fail at prediction time with
        // `Error executing model: ... (error code: -1)` regardless of model
        // format or compute units. Restricting CoreML to statically-shaped
        // subgraphs keeps the heavy conv/matmul blocks accelerated and
        // leaves the dynamic-shape ops on the CPU EP.
        let cache_dir = dir.join("coreml_cache");
        let coreml_ep = ep::CoreML::default()
            .with_model_format(ep::coreml::ModelFormat::MLProgram)
            .with_static_input_shapes(true)
            .with_compute_units(ep::coreml::ComputeUnits::CPUAndNeuralEngine)
            .with_specialization_strategy(ep::coreml::SpecializationStrategy::FastPrediction)
            .with_model_cache_dir(cache_dir.to_string_lossy())
            .build();

        let cpu_fallback = ort::execution_providers::CPUExecutionProvider::default();
        let eps = [coreml_ep.clone(), cpu_fallback.into()];
        let encoder_path = Self::encoder_model_path(dir);
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
        Ok((encoder, decoder, joiner))
    }

    #[cfg(feature = "cuda")]
    fn load_sessions_cuda(
        dir: &Path,
        prepacked: &ort::session::builder::PrepackedWeights,
    ) -> anyhow::Result<(Session, Session, Session)> {
        // CUDA EP compiles subgraphs that cannot be re-serialized as ONNX,
        // so we do NOT call `.with_optimized_model_path(...)` here — same
        // reason as the CoreML block. ORT's CUDA EP keeps its own caches
        // internally.
        let cuda_ep = ep::CUDA::default().build();

        let cpu_fallback = ort::execution_providers::CPUExecutionProvider::default();
        let eps = [cuda_ep.clone(), cpu_fallback.into()];
        let encoder_path = Self::encoder_model_path(dir);
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
        Ok((encoder, decoder, joiner))
    }

    /// Load encoder/decoder/joiner sessions on the plain CPU EP.
    ///
    /// This is the default build's loader and the runtime-fallback target
    /// when the CoreML probe in [`Engine::load`] fails (issue #42). Not
    /// compiled under `cuda` (mutually exclusive with `coreml`), where it
    /// would be dead code.
    #[cfg(not(feature = "cuda"))]
    fn load_sessions_cpu(
        dir: &Path,
        prepacked: &ort::session::builder::PrepackedWeights,
    ) -> anyhow::Result<(Session, Session, Session)> {
        let cache_dir = dir.join("optimized_cache");
        std::fs::create_dir_all(&cache_dir)
            .with_context(|| format!("Failed to create ONNX cache dir: {}", cache_dir.display()))?;
        let cpu_fallback = ort::execution_providers::CPUExecutionProvider::default();
        let eps = [cpu_fallback.into()];
        let encoder_path = Self::encoder_model_path(dir);
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
        Ok((encoder, decoder, joiner))
    }

    /// Split loaded triplets into an interactive pool and an optional batch
    /// pool of `batch_pool_size` triplets. Always leaves at least one triplet
    /// for the interactive pool; `batch_pool_size == 0` (or a pool too small to
    /// split) yields no batch pool.
    fn split_triplets(
        mut triplets: Vec<SessionTriplet>,
        batch_pool_size: usize,
    ) -> (SessionPool, Option<SessionPool>) {
        let n = triplets.len();
        let batch = Self::batch_split_count(n, batch_pool_size);
        if batch == 0 {
            return (SessionPool::new(triplets), None);
        }
        let batch_triplets = triplets.split_off(n - batch);
        (
            SessionPool::new(triplets),
            Some(SessionPool::new(batch_triplets)),
        )
    }

    /// Number of triplets to reserve for the batch pool given `n` loaded and a
    /// requested `batch_pool_size`, always leaving at least one for the
    /// interactive pool (so `n <= 1` or `batch_pool_size == 0` yields 0).
    fn batch_split_count(n: usize, batch_pool_size: usize) -> usize {
        batch_pool_size.min(n.saturating_sub(1))
    }

    /// Decide the final pool from per-triplet load results: returns the
    /// successfully loaded triplets when at least `min_size` loaded (warning
    /// when the pool is degraded below `pool_size`), or an error describing the
    /// shortfall. `min_size` is clamped to `1..=pool_size`.
    fn finalize_pool_load<T>(
        results: Vec<anyhow::Result<T>>,
        pool_size: usize,
        min_size: usize,
    ) -> anyhow::Result<Vec<T>> {
        let min_size = min_size.clamp(1, pool_size.max(1));
        let mut loaded = Vec::with_capacity(results.len());
        let mut first_err: Option<anyhow::Error> = None;
        for r in results {
            match r {
                Ok(t) => loaded.push(t),
                Err(e) => {
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                }
            }
        }
        let n = loaded.len();
        if n >= min_size {
            if n < pool_size {
                let detail = first_err
                    .map(|e| format!("; first error: {e:#}"))
                    .unwrap_or_default();
                tracing::warn!(
                    "degraded pool: loaded {n}/{pool_size} session triplets ({} failed){detail}",
                    pool_size - n
                );
            }
            Ok(loaded)
        } else {
            let detail = first_err.map(|e| format!(": {e:#}")).unwrap_or_default();
            Err(anyhow::anyhow!(
                "loaded only {n}/{pool_size} session triplets, need at least {min_size}{detail}"
            ))
        }
    }

    /// Load up to `pool_size` session triplets in parallel via the given
    /// per-triplet loader (`load_sessions` for the compile-time-selected EP,
    /// `load_sessions_cpu` for the CoreML runtime fallback). Tolerates a
    /// partial pool down to `min_size` (see [`Engine::finalize_pool_load`]).
    fn load_triplets(
        dir: &Path,
        pool_size: usize,
        min_size: usize,
        prepacked: &ort::session::builder::PrepackedWeights,
        load_one: impl Fn(
            &Path,
            &ort::session::builder::PrepackedWeights,
        ) -> anyhow::Result<(Session, Session, Session)>
        + Sync,
    ) -> anyhow::Result<Vec<SessionTriplet>> {
        let results: Vec<anyhow::Result<SessionTriplet>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..pool_size)
                .map(|i| {
                    let pp = prepacked;
                    let load_one = &load_one;
                    s.spawn(move || {
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            tracing::info!(
                                "Loading session triplet {}/{pool_size} (shared weights)",
                                i + 1
                            );
                            let (encoder, decoder, joiner) = load_one(dir, pp)?;
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
                .collect()
        });
        Self::finalize_pool_load(results, pool_size, min_size)
    }

    fn load_inner(
        dir: &Path,
        model_dir: &str,
        pool_size: usize,
        min_size: usize,
        batch_pool_size: usize,
    ) -> anyhow::Result<Self> {
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

        // CoreML can reject a model at two distinct stages: session creation
        // (MLModel compilation) and the first `Run()` (issue #42). This guards
        // the first stage; the warmup probe below guards the second.
        #[cfg(feature = "coreml")]
        let triplets = match Self::load_triplets(
            dir,
            pool_size,
            min_size,
            &prepacked,
            Self::load_sessions,
        ) {
            Ok(triplets) => triplets,
            Err(load_err) => {
                tracing::warn!(
                    "CoreML EP failed to load sessions ({load_err:#}); falling back to CPU execution provider"
                );
                // Fresh container: the failed attempt may have pre-packed
                // weights with CoreML-specific kernel layouts.
                let prepacked = ort::session::builder::PrepackedWeights::new();
                Self::load_triplets(
                    dir,
                    pool_size,
                    min_size,
                    &prepacked,
                    Self::load_sessions_cpu,
                )?
            }
        };
        #[cfg(not(feature = "coreml"))]
        let triplets =
            Self::load_triplets(dir, pool_size, min_size, &prepacked, Self::load_sessions)?;

        let tokenizer = Tokenizer::load(&dir.join("v3_e2e_rnnt_vocab.txt"))?;
        let features = FeatureExtractor::new();

        tracing::info!(
            "Models loaded (vocab_size={}, pool_size={pool_size})",
            tokenizer.vocab_size()
        );

        #[cfg(feature = "diarization")]
        #[allow(deprecated)] // legacy OnnxEmbeddingExtractor::new — see import note above
        let speaker_encoder = {
            let model_path = dir.join("wespeaker_resnet34.onnx");
            if model_path.exists() {
                match OnnxEmbeddingExtractor::new(
                    &model_path,
                    SPEAKER_EMBEDDING_DIM,
                    SPEAKER_SEGMENT_SAMPLES,
                    SPEAKER_POOL_SIZE,
                ) {
                    Ok(enc) => {
                        tracing::info!("Speaker encoder loaded (diarization available)");
                        Some(std::sync::Arc::new(enc))
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Speaker encoder not loaded, diarization unavailable: {e:#}"
                        );
                        None
                    }
                }
            } else {
                tracing::warn!("wespeaker_resnet34.onnx not found, diarization unavailable");
                None
            }
        };

        let (pool, batch_pool) = Self::split_triplets(triplets, batch_pool_size);
        let engine = Self {
            pool,
            batch_pool,
            tokenizer,
            features,
            int8: is_int8,
            #[cfg(feature = "diarization")]
            speaker_encoder,
        };

        // CoreML compiles its graph partitions lazily, so sessions that
        // loaded fine can still fail at the first `Run()` (issue #42). Probe
        // one triplet now; if the probe fails, rebuild the pool on the CPU EP
        // instead of surfacing the error on every real request.
        #[cfg(feature = "coreml")]
        let engine = probe_or_rebuild(
            engine,
            |e: &Self| e.warmup_one().map_err(anyhow::Error::from),
            |mut e, probe_err| {
                tracing::warn!(
                    "CoreML EP failed at runtime ({probe_err:#}); falling back to CPU execution provider"
                );
                let prepacked = ort::session::builder::PrepackedWeights::new();
                let triplets = Self::load_triplets(
                    dir,
                    pool_size,
                    min_size,
                    &prepacked,
                    Self::load_sessions_cpu,
                )?;
                let (pool, batch_pool) = Self::split_triplets(triplets, batch_pool_size);
                e.pool = pool;
                e.batch_pool = batch_pool;
                Ok(e)
            },
        )?;

        Ok(engine)
    }

    /// Run one ~1 s silent inference on a single pooled session triplet.
    ///
    /// Exercises the full mel + encoder + RNN-T decode pipeline, forcing
    /// lazy EP work (CoreML partition compilation, first-run allocations) to
    /// happen now instead of on the first real request — and doubling as a
    /// runtime self-check for EPs that can fail at prediction time even
    /// though their sessions loaded fine (issue #42).
    fn warmup_one(&self) -> Result<(), GigasttError> {
        self.warmup_one_on(&self.pool)
    }

    /// Warm a single triplet from a specific pool with ~1 s of silence.
    fn warmup_one_on(&self, pool: &SessionPool) -> Result<(), GigasttError> {
        let silence = vec![0.0f32; 16000]; // 1 s at 16 kHz
        let mut guard = pool
            .checkout_blocking()
            .map_err(|e| GigasttError::Inference {
                source: Box::new(e),
            })?;
        self.transcribe_samples(&silence, &mut guard)?;
        Ok(())
    }

    /// Warm up every pooled session triplet with a ~1 s silent inference
    /// so the first real request doesn't pay the EP compile /
    /// first-allocation cost.
    ///
    /// Sequential checkouts visit each pooled triplet exactly once because
    /// check-in returns items to the back of the FIFO queue.
    ///
    /// # Errors
    ///
    /// Returns [`GigasttError::Inference`] if a warmup inference fails — with
    /// the `coreml` feature this is unexpected, because [`Engine::load`]
    /// already probed the pool and fell back to the CPU EP if needed.
    pub fn warmup(&self) -> Result<(), GigasttError> {
        for _ in 0..self.pool.total() {
            self.warmup_one()?;
        }
        if let Some(ref batch) = self.batch_pool {
            for _ in 0..batch.total() {
                self.warmup_one_on(batch)?;
            }
        }
        Ok(())
    }

    /// The pool REST file transcription should use: the dedicated batch pool
    /// when one was split off, otherwise the interactive pool.
    pub fn pool_for_batch(&self) -> &SessionPool {
        self.batch_pool.as_ref().unwrap_or(&self.pool)
    }

    /// Close both the interactive and batch pools so every waiter wakes with
    /// `PoolError::Closed` during graceful shutdown.
    pub fn close_pools(&self) {
        self.pool.close();
        if let Some(ref batch) = self.batch_pool {
            batch.close();
        }
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
        let diarization_state = match (diarization_enabled, &self.speaker_encoder) {
            (true, Some(enc)) => {
                let config = DiaConfig {
                    cluster: ClusterConfig {
                        threshold: 0.5,
                        ..ClusterConfig::default()
                    },
                    ..DiaConfig::default()
                };
                let vad_config = VadConfig::default();
                let vad = EnergyVad::new(-40.0, 16000, vad_config.frame_size);
                let extractor = SharedExtractor(std::sync::Arc::clone(enc));
                match StreamingPipeline::new(vad, extractor, config, vad_config) {
                    Ok(pipeline) => Some(pipeline),
                    Err(e) => {
                        tracing::warn!("Failed to initialize streaming diarization: {e:#}");
                        None
                    }
                }
            }
            _ => None,
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
            window_start_samples: 0,
            context_samples: 0,
            pending_samples: 0,
            resampler: None,
            mel_fft_input: Vec::new(),
            mel_power: Vec::new(),
            mel_output: Vec::new(),
            resample_output_buf: Vec::new(),
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

        // Diarization tracks speakers continuously, so feed every chunk's audio
        // even when this chunk doesn't trigger a decode (see the stride gate).
        #[cfg(feature = "diarization")]
        if let Some(dia) = state.diarization_state.as_mut()
            && let Err(e) = dia.feed(samples)
        {
            tracing::warn!("Diarization feed failed: {e:#}");
        }

        // Sliding-window streaming: accumulate audio; the encoder re-runs on the
        // whole retained window so the offline Conformer always has left context
        // (an isolated ~100ms chunk decodes to garbage). Re-decoding is the cost,
        // so we only decode once STREAM_DECODE_STRIDE_SAMPLES of NEW audio have
        // arrived (or the window hit its cap) — this keeps the engine real-time.
        // The window is bounded by STREAM_MAX_WINDOW_SAMPLES; on endpoint or cap
        // we finalize the tail and slide, retaining STREAM_LEFT_CONTEXT_SAMPLES.
        state.audio_buffer.extend_from_slice(samples);
        state.pending_samples += samples.len();
        let over_cap = state.audio_buffer.len() >= STREAM_MAX_WINDOW_SAMPLES;
        // Stride gate on NEW audio since the last decode (not since the last
        // slide): otherwise a non-finalizing partial would leave the counter
        // high and decode on every subsequent chunk.
        if state.pending_samples < STREAM_DECODE_STRIDE_SAMPLES && !over_cap {
            return Ok(vec![]);
        }
        if state.audio_buffer.len() < N_FFT {
            return Ok(vec![]);
        }

        let endpoint = self
            .decode_window(state, triplet)
            .map_err(|e| GigasttError::Inference { source: e.into() })?;
        state.pending_samples = 0;
        let ts = now_timestamp();

        if endpoint || over_cap {
            let seg = state.assembler.finalize(ts);
            // Slide: retain the trailing left-context window for the next decode.
            let keep = STREAM_LEFT_CONTEXT_SAMPLES.min(state.audio_buffer.len());
            let slide_off = state.audio_buffer.len() - keep;
            if slide_off > 0 {
                audio::consume_audio_buffer(&mut state.audio_buffer, slide_off);
                state.window_start_samples += slide_off;
            }
            state.context_samples = keep;
            if seg.text.trim().is_empty() {
                return Ok(vec![]);
            }
            return Ok(vec![seg]);
        }

        if state.assembler.is_empty() {
            return Ok(vec![]);
        }
        Ok(vec![state.assembler.partial(ts)])
    }

    /// Re-decode the whole retained window from a fresh decoder state and update
    /// the assembler with the context-suppressed tail. Returns whether the
    /// decoder detected an endpoint. Shared by [`Engine::process_chunk`]
    /// (strided) and [`Engine::finish_stream`] (forced at end of stream).
    fn decode_window(
        &self,
        state: &mut StreamingState,
        triplet: &mut SessionTriplet,
    ) -> anyhow::Result<bool> {
        let mel_start = std::time::Instant::now();
        let num_frames = self.features.compute_mel(
            &state.audio_buffer,
            &mut state.mel_fft_input,
            &mut state.mel_power,
            &mut state.mel_output,
        );
        tracing::debug!(
            elapsed_us = mel_start.elapsed().as_micros() as u64,
            "mel_compute"
        );
        if num_frames == 0 {
            return Ok(false);
        }

        // Encoder-frame offset of the window start (drift-free: a single division
        // over the cumulative slid-off sample count).
        let frame_offset = state.window_start_samples / (HOP_LENGTH * ENCODER_SUBSAMPLING);

        // The window overlaps the previous one, so persisting the LSTM state
        // would double-condition the prediction network — decode fresh.
        let mut decoder_state = DecoderState::new(self.tokenizer.blank_id());
        let (all_words, endpoint) = self.run_inference(
            triplet,
            &state.mel_output[..],
            num_frames,
            &mut decoder_state,
            frame_offset,
        )?;

        // Suppress words inside the already-emitted left context so a slid
        // window does not re-emit committed words.
        let window_start_s = frame_offset as f64 * SECONDS_PER_FRAME;
        let context_boundary_s = window_start_s + state.context_samples as f64 / 16000.0;
        #[cfg_attr(not(feature = "diarization"), allow(unused_mut))]
        let mut tail: Vec<WordInfo> = all_words
            .into_iter()
            .filter(|w| w.start + f64::EPSILON >= context_boundary_s)
            .collect();

        #[cfg(feature = "diarization")]
        if let Some(dia) = state.diarization_state.as_mut()
            && let Some(turn) = dia.turns().last()
        {
            let speaker = turn.speaker.0;
            for w in &mut tail {
                w.speaker = Some(speaker);
            }
        }

        state.assembler.set_words(tail);
        Ok(endpoint)
    }

    /// Decode any audio buffered since the last strided decode, then finalize.
    /// Call when the stream ends (Stop / EOF) so the decode-stride batching does
    /// not drop trailing words. Best-effort: on decode failure, falls back to a
    /// plain flush of whatever the assembler already holds.
    pub fn finish_stream(
        &self,
        state: &mut StreamingState,
        triplet: &mut SessionTriplet,
    ) -> Option<TranscriptSegment> {
        let has_pending = state.pending_samples > 0 && state.audio_buffer.len() >= N_FFT;
        if has_pending && let Err(e) = self.decode_window(state, triplet) {
            tracing::warn!("finish_stream decode failed: {e:#}");
        }
        self.flush_state(state)
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

        #[cfg(feature = "diarization")]
        if let Some(ref enc) = self.speaker_encoder {
            let config = DiaConfig::default();
            let vad_config = VadConfig::default();
            let pipeline = Pipeline::new(config, vad_config);
            let mut vad = EnergyVad::new(-40.0, 16000, vad_config.frame_size);
            match pipeline.run(float_samples, enc.as_ref(), &mut vad) {
                Ok(dia_result) => {
                    for word in &mut words {
                        let mid = (word.start + word.end) / 2.0;
                        if let Some(turn) = dia_result
                            .turns
                            .iter()
                            .find(|t| t.time.start <= mid && t.time.end >= mid)
                        {
                            word.speaker = Some(turn.speaker.0);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Offline diarization failed: {e:#}");
                }
            }
        }

        let text: String = words
            .iter()
            .map(|w| w.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");

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

        // RNN-T greedy decode — we pass the encoder-output borrow directly
        // instead of copying it.  The `encoder_outputs` variable is dropped
        // automatically at the end of this scope, after decode finishes.
        let dec_start = std::time::Instant::now();
        let result = decode::greedy_decode(
            &mut triplet.decoder,
            &mut triplet.joiner,
            enc_data,
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

        // Group tokens by words (BPE ▁ marks word boundaries)
        let mut words = Vec::new();
        let mut current_word = String::new();
        let mut word_start_frame: Option<usize> = None;
        let mut word_end_frame: usize = 0;
        let mut word_confidences: Vec<f32> = Vec::new();

        for token in tokens {
            let token_text = self.tokenizer.token_text(token.token_id);
            let is_word_boundary = token_text.starts_with(tokenizer::WORD_BOUNDARY);

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

            let clean = if let Some(stripped) = token_text.strip_prefix(tokenizer::WORD_BOUNDARY) {
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
    /// Full recognized transcript text (words joined with spaces).
    pub text: String,
    /// Word-level timing, confidence, and optional speaker annotations.
    pub words: Vec<WordInfo>,
    /// Duration of the decoded audio in seconds.
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

impl TranscriptSegment {
    pub fn empty_final() -> Self {
        Self {
            text: String::new(),
            words: vec![],
            is_final: true,
            timestamp: now_timestamp(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_split_count_clamps() {
        assert_eq!(Engine::batch_split_count(4, 1), 1); // typical: 1 batch, 3 stream
        assert_eq!(Engine::batch_split_count(4, 0), 0); // split disabled
        assert_eq!(Engine::batch_split_count(4, 10), 3); // clamped: leave 1 interactive
        assert_eq!(Engine::batch_split_count(1, 1), 0); // can't split a single triplet
        assert_eq!(Engine::batch_split_count(0, 1), 0); // empty pool
        assert_eq!(Engine::batch_split_count(2, 1), 1);
    }

    #[test]
    fn test_finalize_pool_load_full() {
        let r: Vec<anyhow::Result<u32>> = vec![Ok(1), Ok(2), Ok(3)];
        assert_eq!(Engine::finalize_pool_load(r, 3, 3).unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn test_finalize_pool_load_degraded_boots() {
        // 2 of 4 loaded with min_size 1 → degraded pool is accepted.
        let r: Vec<anyhow::Result<u32>> = vec![
            Ok(1),
            Err(anyhow::anyhow!("boom")),
            Ok(3),
            Err(anyhow::anyhow!("boom2")),
        ];
        assert_eq!(Engine::finalize_pool_load(r, 4, 1).unwrap(), vec![1, 3]);
    }

    #[test]
    fn test_finalize_pool_load_below_min_errors() {
        // Only 1 loaded but min_size 2 → error naming the shortfall.
        let r: Vec<anyhow::Result<u32>> = vec![
            Ok(1),
            Err(anyhow::anyhow!("boom")),
            Err(anyhow::anyhow!("boom2")),
        ];
        let err = Engine::finalize_pool_load(r, 3, 2).unwrap_err().to_string();
        assert!(err.contains("loaded only 1/3"), "got: {err}");
        assert!(err.contains("need at least 2"), "got: {err}");
    }

    #[test]
    fn test_finalize_pool_load_all_fail_errors() {
        let r: Vec<anyhow::Result<u32>> =
            vec![Err(anyhow::anyhow!("a")), Err(anyhow::anyhow!("b"))];
        assert!(Engine::finalize_pool_load(r, 2, 1).is_err());
    }

    #[test]
    fn test_finalize_pool_load_min_clamped_to_pool() {
        // min_size > pool_size is clamped down; a full load still succeeds.
        let r: Vec<anyhow::Result<u32>> = vec![Ok(1), Ok(2)];
        assert_eq!(Engine::finalize_pool_load(r, 2, 99).unwrap(), vec![1, 2]);
    }

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

    #[test]
    fn test_feature_extractor_default() {
        let _fe = FeatureExtractor::default();
    }

    #[test]
    fn test_transcript_assembler_default() {
        let ta = TranscriptAssembler::default();
        assert!(ta.text.is_empty());
        assert!(ta.words.is_empty());
    }

    #[test]
    fn test_pool_checkout_blocking_fast_path() {
        let pool = Pool::new(vec![42u32]);
        let guard = pool.checkout_blocking().expect("checkout_blocking");
        assert_eq!(*guard, 42);
        drop(guard);
        assert_eq!(pool.available(), 1);
    }

    #[test]
    fn test_pool_checkout_blocking_closed() {
        let pool = Pool::<u32>::new(vec![]);
        pool.close();
        assert!(matches!(pool.checkout_blocking(), Err(PoolError::Closed)));
    }

    #[test]
    fn test_pool_checkout_blocking_slow_path() {
        let pool = std::sync::Arc::new(Pool::new(vec![42u32]));
        let primary = pool.checkout_blocking().unwrap();

        let handle = std::thread::spawn({
            let pool = pool.clone();
            move || pool.checkout_blocking()
        });

        std::thread::sleep(std::time::Duration::from_millis(50));
        drop(primary);

        let guard = handle.join().expect("join").expect("checkout");
        assert_eq!(*guard, 42);
        drop(guard);
        assert_eq!(pool.available(), 1);
    }

    #[test]
    fn test_pool_error_display() {
        assert_eq!(format!("{}", PoolError::Closed), "session pool is closed");
    }

    #[test]
    fn test_ort_err() {
        let e = ort_err("test ort error");
        assert_eq!(format!("{e}"), "test ort error");
    }

    #[test]
    fn test_engine_load_missing_dir() {
        let result = Engine::load_with_pool_size("/nonexistent/path/for/tests", 1);
        assert!(matches!(result, Err(GigasttError::ModelLoad { .. })));
    }

    #[test]
    fn test_engine_load_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let result = Engine::load_with_pool_size(dir.path().to_str().unwrap(), 1);
        assert!(matches!(result, Err(GigasttError::ModelLoad { .. })));
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
        let reservation = guard.into_owned();

        let result = tokio::task::spawn_blocking(move || {
            // Pretend we're running blocking inference.
            assert_eq!(*reservation, "triplet");
            reservation.checkin();
            "done"
        })
        .await
        .expect("join");

        // After the blocking task returns the item, the pool is full again.
        assert_eq!(pool.available(), 1);
        assert_eq!(result, "done");
    }

    #[tokio::test]
    async fn test_owned_reservation_returns_on_spawn_blocking_panic() {
        // If the blocking task panics, the reservation's Drop must still
        // return the item so the pool does not leak capacity.
        let pool = std::sync::Arc::new(Pool::new(vec![String::from("triplet")]));
        let guard = pool.checkout().await.expect("checkout");
        let reservation = guard.into_owned();

        let result = tokio::task::spawn_blocking(move || {
            let _reservation = reservation;
            panic!("simulated inference panic");
        })
        .await;

        assert!(result.is_err(), "spawn_blocking must report the panic");
        assert_eq!(
            pool.available(),
            1,
            "reservation must be returned after panic"
        );
    }

    #[tokio::test]
    async fn test_owned_reservation_drop_returns_item() {
        // Dropping an unchecked-in reservation still returns the item.
        let pool = std::sync::Arc::new(Pool::new(vec![String::from("triplet")]));
        let guard = pool.checkout().await.expect("checkout");
        let reservation = guard.into_owned();

        tokio::task::spawn_blocking(move || {
            let _reservation = reservation;
            // reservation dropped here
        })
        .await
        .expect("join");

        assert_eq!(pool.available(), 1);
    }

    #[tokio::test]
    async fn test_pool_close_is_idempotent() {
        // `pool.close()` is wired into the shutdown hook; calling it twice
        // (e.g. shutdown signal + Drop) must not panic.
        let pool = Pool::<u32>::new(vec![]);
        pool.close();
        pool.close();
    }

    #[tokio::test]
    async fn test_pool_waiters_count() {
        let pool = std::sync::Arc::new(Pool::<u32>::new(vec![]));
        let w1 = tokio::spawn({
            let p = pool.clone();
            async move { p.checkout().await.map(|_| ()) }
        });
        let w2 = tokio::spawn({
            let p = pool.clone();
            async move { p.checkout().await.map(|_| ()) }
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert_eq!(pool.waiters(), 2, "both blocked tasks must be waiters");
        pool.close();
        let _ = w1.await;
        let _ = w2.await;
    }

    #[tokio::test]
    async fn test_owned_reservation_round_trip_through_option() {
        // Mirrors the pattern used by `handle_binary_frame` in ws.rs:
        // the reservation is temporarily moved out of an Option into
        // spawn_blocking and then placed back on success.
        let pool = std::sync::Arc::new(Pool::new(vec![42u32]));
        let guard = pool.checkout().await.expect("checkout");
        let mut reservation: Option<OwnedReservation<u32>> = Some(guard.into_owned());

        let (res_back, val) = tokio::task::spawn_blocking(move || {
            let mut r = reservation.take().unwrap();
            *r += 1;
            let v = *r;
            (r, v)
        })
        .await
        .expect("join");

        reservation = Some(res_back);
        assert_eq!(val, 43);
        drop(reservation);
        assert_eq!(pool.available(), 1);
    }

    #[tokio::test]
    async fn test_pool_slot_not_leaked_on_cancelled_checkout() {
        // If a checkout future is cancelled after registering a waiter but
        // before receiving an item, the oneshot receiver is dropped while the
        // sender remains in the waiters queue.  When another item is checked
        // in, the dead waiter must be skipped and the item returned to the
        // pool — otherwise the slot is leaked forever.
        let pool = std::sync::Arc::new(Pool::new(vec![42u32]));
        let primary = pool.checkout().await.expect("checkout");

        let aborted = tokio::spawn({
            let pool = pool.clone();
            async move { pool.checkout().await }
        });
        // Let the spawned task register as a waiter.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        aborted.abort();
        let _ = aborted.await;

        // The abandoned waiter is still queued.
        assert_eq!(pool.waiters(), 1);

        // Return the primary item.  Without the retry loop in checkin this
        // would silently drop the item because tx.send fails.
        drop(primary);

        assert_eq!(pool.available(), 1, "item must return to pool, not leak");
        assert_eq!(pool.waiters(), 0, "dead waiter must be removed");
    }

    #[tokio::test]
    async fn test_pool_slot_not_leaked_on_timeout_checkout() {
        // Same scenario as above, but using tokio::time::timeout instead of
        // abort — this is the exact path hit by the REST and WS handlers.
        let pool = std::sync::Arc::new(Pool::new(vec![42u32]));
        let primary = pool.checkout().await.expect("checkout");

        let result =
            tokio::time::timeout(std::time::Duration::from_millis(10), pool.checkout()).await;
        assert!(result.is_err(), "checkout must time out");

        assert_eq!(pool.waiters(), 1);

        drop(primary);

        assert_eq!(
            pool.available(),
            1,
            "item must return to pool after timeout"
        );
        assert_eq!(pool.waiters(), 0, "dead waiter must be removed");
    }

    #[tokio::test]
    async fn test_pool_multiple_dead_waiters_are_skipped() {
        // Several cancelled waiters in a row should all be skipped in one
        // checkin pass.
        let pool = std::sync::Arc::new(Pool::new(vec![0u32]));
        let primary = pool.checkout().await.expect("checkout");

        let mut handles = Vec::new();
        for _ in 0..3 {
            handles.push(tokio::spawn({
                let pool = pool.clone();
                async move { pool.checkout().await }
            }));
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        for h in handles {
            h.abort();
            let _ = h.await;
        }

        assert_eq!(pool.waiters(), 3);

        drop(primary);

        assert_eq!(
            pool.available(),
            1,
            "item returned after skipping 3 dead waiters"
        );
        assert_eq!(pool.waiters(), 0);
    }

    #[test]
    fn test_transcript_assembler_append_and_finalize() {
        let mut asm = TranscriptAssembler::new();
        assert!(asm.is_empty());
        asm.append(vec![
            WordInfo {
                word: "hello".into(),
                start: 0.0,
                end: 0.5,
                confidence: 0.9,
                speaker: None,
            },
            WordInfo {
                word: "world".into(),
                start: 0.6,
                end: 1.0,
                confidence: 0.85,
                speaker: None,
            },
        ]);
        assert!(!asm.is_empty());
        let seg = asm.finalize(1.0);
        assert_eq!(seg.text, "hello world");
        assert_eq!(seg.words.len(), 2);
        assert!(seg.is_final);
        assert_eq!(seg.timestamp, 1.0);
        // After finalize the assembler is reset.
        assert!(asm.is_empty());
    }

    #[test]
    fn test_transcript_assembler_partial() {
        let mut asm = TranscriptAssembler::new();
        asm.append(vec![WordInfo {
            word: "partial".into(),
            start: 0.0,
            end: 0.3,
            confidence: 0.8,
            speaker: None,
        }]);
        let seg = asm.partial(0.3);
        assert_eq!(seg.text, "partial");
        assert!(!seg.is_final);
        // partial must not reset the assembler.
        assert!(!asm.is_empty());
    }

    #[test]
    fn test_feature_extractor_compute_empty() {
        let fe = FeatureExtractor::new();
        let (mel, frames) = fe.compute(&[]);
        // When samples are shorter than N_FFT, compute_with_buffers returns
        // a single zero-filled frame with n_mels elements.
        assert_eq!(mel.len(), N_MELS);
        assert_eq!(frames, 1);
        assert!(mel.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_now_timestamp_non_negative() {
        let ts = now_timestamp();
        assert!(ts >= 0.0, "timestamp must be non-negative");
    }

    #[test]
    fn test_probe_or_rebuild_keeps_state_when_probe_passes() {
        let rebuilt = std::cell::Cell::new(false);
        let result = probe_or_rebuild(
            7u32,
            |v| {
                assert_eq!(*v, 7);
                Ok(())
            },
            |_, _| {
                rebuilt.set(true);
                Ok(99)
            },
        )
        .expect("healthy state must survive unchanged");
        assert_eq!(result, 7);
        assert!(!rebuilt.get(), "rebuild must not run when the probe passes");
    }

    #[test]
    fn test_probe_or_rebuild_rebuilds_when_probe_fails() {
        let result = probe_or_rebuild(
            1u32,
            |v| {
                if *v == 1 {
                    Err(anyhow::anyhow!("first probe failed"))
                } else {
                    Ok(())
                }
            },
            |old, probe_err| {
                assert_eq!(old, 1, "rebuild receives the failed state");
                assert!(
                    probe_err.to_string().contains("first probe failed"),
                    "rebuild receives the probe error for logging"
                );
                Ok(2)
            },
        )
        .expect("rebuilt state passing the probe is OK");
        assert_eq!(result, 2);
    }

    #[test]
    fn test_probe_or_rebuild_propagates_rebuild_error() {
        let result = probe_or_rebuild(
            1u32,
            |_| Err(anyhow::anyhow!("probe failed")),
            |_, _| Err(anyhow::anyhow!("rebuild failed")),
        );
        let err = result.expect_err("rebuild failure must be fatal");
        assert!(err.to_string().contains("rebuild failed"));
    }

    #[test]
    fn test_probe_or_rebuild_fails_when_rebuilt_state_fails_probe() {
        let result = probe_or_rebuild(
            1u32,
            |_| Err(anyhow::anyhow!("always fails")),
            |_, _| Ok(2u32),
        );
        assert!(
            result.is_err(),
            "a rebuilt state that still fails the probe must be a hard error"
        );
    }

    #[test]
    fn test_encoder_model_path_prefers_int8_when_present() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("v3_e2e_rnnt_encoder.onnx"), b"fp32").unwrap();
        std::fs::write(dir.path().join("v3_e2e_rnnt_encoder_int8.onnx"), b"int8").unwrap();
        let path = Engine::encoder_model_path(dir.path());
        assert_eq!(
            path.file_name().unwrap(),
            "v3_e2e_rnnt_encoder_int8.onnx",
            "INT8 encoder must win when both files exist"
        );
    }

    #[test]
    fn test_encoder_model_path_falls_back_to_fp32() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("v3_e2e_rnnt_encoder.onnx"), b"fp32").unwrap();
        let path = Engine::encoder_model_path(dir.path());
        assert_eq!(path.file_name().unwrap(), "v3_e2e_rnnt_encoder.onnx");
    }

    #[test]
    fn test_pool_sequential_checkouts_visit_every_item() {
        // Engine::warmup relies on this FIFO property: `total()` sequential
        // checkout/checkin cycles touch every pooled item exactly once.
        let pool = Pool::new(vec![1u32, 2, 3]);
        let mut seen = Vec::new();
        for _ in 0..pool.total() {
            let guard = pool.checkout_blocking().expect("checkout");
            seen.push(*guard);
            // guard drops here — the item returns to the back of the queue
        }
        seen.sort_unstable();
        assert_eq!(seen, vec![1, 2, 3]);
    }

    #[test]
    #[ignore = "requires model"]
    fn test_warmup_runs_silent_inference_on_every_triplet() {
        let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 2)
            .expect("engine should load");
        engine
            .warmup()
            .expect("warmup must succeed on a working engine");
        assert_eq!(
            engine.pool.available(),
            engine.pool.total(),
            "every triplet must be returned to the pool after warmup"
        );
    }
}
