//! ONNX Runtime inference engine for GigaAM v3 e2e_rnnt.
//!
//! Loads encoder, decoder, and joiner ONNX models and runs the RNN-T streaming decode loop.

pub mod audio;
mod bias;
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
use crate::model::ModelVariant;

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

/// Total physical RAM in bytes, or `0` if it can't be determined (in which case
/// the pool RAM cap is a no-op). macOS: `sysctl HW_MEMSIZE`; Linux/other unix:
/// `sysconf(_SC_PHYS_PAGES) * _SC_PAGESIZE`.
fn total_ram_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let mut mem: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let mib = [libc::CTL_HW, libc::HW_MEMSIZE];
        // SAFETY: `mib`/`mem`/`len` are valid for the duration of the call;
        // sysctl writes at most `len` bytes into `mem`.
        let rc = unsafe {
            libc::sysctl(
                mib.as_ptr() as *mut libc::c_int,
                mib.len() as libc::c_uint,
                &mut mem as *mut u64 as *mut libc::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc == 0 { mem } else { 0 }
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        // SAFETY: sysconf has no side effects and returns -1 on error.
        let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if pages > 0 && page_size > 0 {
            (pages as u64).saturating_mul(page_size as u64)
        } else {
            0
        }
    }
    #[cfg(not(unix))]
    {
        0
    }
}

/// Encoder time subsampling factor (4 frames → 1 encoder output frame).
const ENCODER_SUBSAMPLING: usize = 4;
/// Seconds per encoder frame (HOP_LENGTH * ENCODER_SUBSAMPLING / 16000 = 0.04s).
const SECONDS_PER_FRAME: f64 = (HOP_LENGTH as f64 * ENCODER_SUBSAMPLING as f64) / 16000.0;

/// Max streaming encoder window before forcing a finalize (samples @16kHz, 2.5s).
/// Re-decoding the whole window each stride gives the offline Conformer left
/// context; this cap bounds the per-stride encoder cost. With the 1.5s retained
/// left context and the 0.8s stride, a 2.5s window keeps the steady-state
/// re-encode overlap near ~3x (vs ~6.25x at a 5s window) — roughly half the
/// streaming encoder work — while retaining enough left context that streaming
/// quality stays on par with batch (covered by the `streaming_quality` tests).
const STREAM_MAX_WINDOW_SAMPLES: usize = 16000 * 5 / 2;
/// Left-context audio retained across a streaming finalize/slide (samples @16kHz,
/// ~1.5s) so the next window keeps acoustic context instead of restarting cold.
const STREAM_LEFT_CONTEXT_SAMPLES: usize = 16000 * 3 / 2;
/// Decode stride: re-run the encoder only after this much NEW audio has
/// accumulated (samples @16kHz, 0.8s) instead of on every ~100ms chunk.
/// Re-decoding the window is the dominant streaming cost, so the stride keeps
/// the engine real-time; `finish_stream` decodes the sub-stride remainder at EOF.
const STREAM_DECODE_STRIDE_SAMPLES: usize = 16000 * 4 / 5;

/// File-transcription chunking threshold (samples @16kHz, 30s). Inputs at or
/// below this length take the single-pass path unchanged; longer inputs are
/// split into overlapping windows so the encoder's peak activation memory is
/// bounded by the chunk size, not the file length. The Conformer encoder only
/// carries ~20–30s of useful context, so chunking above this costs no accuracy
/// in the common case.
const CHUNK_THRESHOLD_SAMPLES: usize = 16000 * 30;
/// Length of each long-form decode window (samples @16kHz, 24s). Bounds the
/// per-chunk encoder activation footprint.
const CHUNK_WINDOW_SAMPLES: usize = 16000 * 24;
/// Overlap retained between consecutive long-form windows (samples @16kHz, 2s),
/// so a word straddling a seam is decoded fully in at least one chunk. The
/// stitch step de-dups words in the overlap region (see [`stitch_chunk_words`]).
const CHUNK_OVERLAP_SAMPLES: usize = 16000 * 2;

/// Default number of session triplets in the pool.
///
/// Each pooled triplet deserializes its own copy of the encoder weights — ORT's
/// shared `PrepackedWeights` container shares prepacked kernel buffers, not the
/// raw initializer tensors, and there is no stable cross-session
/// initializer-sharing path in this ORT version (see
/// [`Engine::cap_pool_size_for_ram`]). A pooled INT8 encoder triplet costs
/// ~0.4 GB resident, so the default is kept at 2 (down from 4) to bound the
/// idle footprint: two concurrent inference slots cover typical local /
/// small-container deployments without quadrupling memory. Raise `--pool-size`
/// when higher concurrency is needed and RAM allows.
#[cfg(target_os = "android")]
const DEFAULT_POOL_SIZE: usize = 1;
#[cfg(not(target_os = "android"))]
const DEFAULT_POOL_SIZE: usize = 2;

/// Approximate resident bytes a single pooled encoder triplet costs, as a
/// multiple of the encoder file size on disk. Measured at ~1.9x the INT8
/// encoder file (225 MB file → ~0.4 GB resident per extra pooled slot, dynamic
/// INT8 graph, CPU EP, release). Used by [`Engine::cap_pool_size_for_ram`] to
/// keep `pool_size * encoder_file_bytes * this` under a fraction of total RAM.
const ENCODER_RESIDENT_MULTIPLIER: u64 = 2;

/// Fraction (denominator) of total system RAM the pooled encoder sessions are
/// allowed to occupy before [`Engine::cap_pool_size_for_ram`] clamps the pool.
/// `2` = at most half of total RAM budgeted to encoder slots, leaving headroom
/// for the decoder/joiner sessions, audio buffers, inference arenas, and the
/// rest of the system.
const POOL_RAM_FRACTION_DENOM: u64 = 2;

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

impl WordInfo {
    /// Create a new [`WordInfo`].
    pub fn new(
        word: impl Into<String>,
        start: f64,
        end: f64,
        confidence: f32,
        speaker: Option<u32>,
    ) -> Self {
        Self {
            word: word.into(),
            start,
            end,
            confidence,
            speaker,
        }
    }
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
    /// Optional VAD endpoint detector (present only when the engine has a VAD).
    /// Fed every chunk's raw samples to track trailing silence; when it fires,
    /// `process_chunk` finalizes the current segment. `None` = no VAD, and
    /// endpointing falls back to the decoder's blank-run heuristic alone.
    pub vad_endpointer: Option<crate::vad::VadEndpointer>,
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
    /// Recognition head detected on disk at load time. Drives the default
    /// punctuation policy (`auto`): on for [`ModelVariant::Rnnt`] (bare output),
    /// off for [`ModelVariant::E2eRnnt`] (already punctuated).
    variant: ModelVariant,
    /// Optional punctuation / casing restorer applied to file-transcription
    /// output. `None` = pass-through (the default, and the only behaviour when
    /// no punct model is installed). Attached via [`Engine::with_punctuator`].
    punctuator: Option<crate::punctuation::Punctuator>,
    /// Whether to run inverse text normalization (Russian number-words →
    /// digits) on file-transcription output, *before* the punctuation pass.
    /// Off by default; toggled via [`Engine::with_itn`].
    itn: bool,
    /// Optional contextual hotword biaser applied inside the greedy RNN-T decode
    /// loop (shallow fusion). `None` = no biasing (the default), and the decode
    /// path is then byte-for-byte identical to the un-biased engine. Attached
    /// via [`Engine::with_biaser`]. Shared across the session pool by reference.
    biaser: Option<bias::Biaser>,
    /// Optional Silero VAD. When set, file transcription skips silent regions
    /// (decoding only detected speech) and streaming finalizes a segment on
    /// VAD-detected trailing silence. `None` = no VAD: the file path decodes the
    /// whole buffer and streaming endpointing is byte-for-byte unchanged.
    /// Attached via [`Engine::with_vad`].
    vad: Option<crate::vad::SileroVad>,
    /// Thresholds for the VAD (speech threshold, min silence/speech, padding).
    /// Ignored when `vad` is `None`.
    vad_config: crate::vad::VadConfig,
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

    /// The recognition head ([`ModelVariant`]) detected on disk at load time.
    /// Lets callers decide the default punctuation policy (`auto`).
    pub fn variant(&self) -> ModelVariant {
        self.variant
    }

    /// Attach an optional punctuation / casing restorer, consuming and
    /// returning `self` (builder style). Pass `None` for pass-through. When set,
    /// the restorer post-processes the final text of file transcription
    /// ([`Engine::transcribe_file`] / [`Engine::transcribe_bytes_shared`]).
    pub fn with_punctuator(mut self, punctuator: Option<crate::punctuation::Punctuator>) -> Self {
        self.punctuator = punctuator;
        self
    }

    /// Whether a punctuation restorer is attached.
    pub fn has_punctuator(&self) -> bool {
        self.punctuator.is_some()
    }

    /// Enable or disable inverse text normalization (Russian number-words →
    /// digits) on file-transcription output, consuming and returning `self`
    /// (builder style). When enabled, ITN runs *before* the punctuation pass so
    /// the restorer cases the already-digitized text.
    pub fn with_itn(mut self, enabled: bool) -> Self {
        self.itn = enabled;
        self
    }

    /// Whether inverse text normalization is enabled.
    pub fn has_itn(&self) -> bool {
        self.itn
    }

    /// Attach a contextual hotword biaser built from `(phrase, weight)` pairs
    /// and an additive `boost`, consuming and returning `self` (builder style).
    /// Each phrase is tokenized with the engine's own `Tokenizer`, so biasing
    /// adapts to whichever recognition head is loaded.
    ///
    /// When `phrases` is empty, `boost <= 0`, or no phrase is representable in
    /// the active vocab, the biaser resolves to `None` and the decode path stays
    /// byte-for-byte unchanged. Replaces any previously attached biaser.
    pub fn with_hotwords(mut self, phrases: &[(String, f32)], boost: f32) -> Self {
        self.biaser = if phrases.is_empty() {
            None
        } else {
            bias::Biaser::from_phrases(&self.tokenizer, phrases, boost)
        };
        if let Some(b) = &self.biaser {
            tracing::info!(
                "Hotword biasing enabled ({} phrase(s), boost {boost})",
                b.phrase_count()
            );
        }
        self
    }

    /// Whether a hotword biaser is attached (biasing active).
    pub fn has_hotwords(&self) -> bool {
        self.biaser.is_some()
    }

    /// Attach an optional Silero VAD plus its config, consuming and returning
    /// `self` (builder style). Pass `None` for no VAD (the default): file
    /// transcription then decodes the whole buffer and streaming endpointing is
    /// byte-for-byte unchanged. When set, file transcription skips silence and
    /// streaming finalizes on VAD-detected trailing silence.
    pub fn with_vad(
        mut self,
        vad: Option<crate::vad::SileroVad>,
        config: crate::vad::VadConfig,
    ) -> Self {
        self.vad = vad;
        self.vad_config = config;
        if self.vad.is_some() {
            tracing::info!(
                "VAD enabled (threshold {}, min_silence {}ms)",
                self.vad_config.threshold,
                self.vad_config.min_silence_ms
            );
        }
        self
    }

    /// Whether a VAD is attached (silence skipping / VAD endpointing active).
    pub fn has_vad(&self) -> bool {
        self.vad.is_some()
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
    /// The recognition head ([`ModelVariant`]) is auto-detected from the encoder
    /// file present on disk: `v3_rnnt_encoder.onnx` (or `_int8.onnx`) selects the
    /// plain rnnt head, else `v3_e2e_rnnt_encoder.onnx` selects e2e_rnnt. The
    /// matching decoder, joiner, and vocab files must also be present.
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
        Self::load_with_pools_threads(model_dir, pool_size, min_size, batch_pool_size, 1)
    }

    /// Like [`Engine::load_with_pools`], but with a configurable encoder
    /// intra-op thread count for the CPU EP. `encoder_intra_threads == 1` (the
    /// default everywhere else) builds sessions identical to the prior
    /// behaviour. Values `> 1` give the dominant encoder more intra-op
    /// parallelism on weak CPUs / long single-file jobs; the count is clamped
    /// against the logical CPU count so `pool_size * threads` can't oversubscribe
    /// the machine (see `Engine::clamp_encoder_intra_threads`). Ignored by the
    /// CoreML / CUDA builds (the accelerator owns scheduling there).
    pub fn load_with_pools_threads(
        model_dir: &str,
        pool_size: usize,
        min_size: usize,
        batch_pool_size: usize,
        encoder_intra_threads: usize,
    ) -> Result<Self, GigasttError> {
        let dir = Path::new(model_dir);
        // Auto-detect which recognition head is present on disk (rnnt encoder
        // takes precedence, else e2e_rnnt). The on-disk layout fully determines
        // which head runs — callers select the variant only at download time.
        let Some(variant) = ModelVariant::detect_in_dir(dir) else {
            return Err(GigasttError::ModelLoad {
                path: model_dir.to_string(),
                source: None,
            });
        };
        // Bound the idle footprint: each pooled triplet deserializes its own
        // encoder copy, so a large `--pool-size` on a small host can OOM at
        // load. Clamp by available RAM (logs when it clamps); a no-op on hosts
        // with ample memory.
        let encoder_bytes = std::fs::metadata(Self::encoder_model_path(dir, variant))
            .map(|m| m.len())
            .unwrap_or(0);
        let pool_size = Self::cap_pool_size_for_ram(pool_size, encoder_bytes, total_ram_bytes());
        // Don't let `pool_size * encoder_intra_threads` oversubscribe the CPU
        // (no-op when the default `1` is requested).
        let logical_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let encoder_intra_threads =
            Self::clamp_encoder_intra_threads(pool_size, encoder_intra_threads, logical_cpus);
        Self::load_inner(
            dir,
            variant,
            model_dir,
            pool_size,
            min_size,
            batch_pool_size,
            encoder_intra_threads,
        )
        .map_err(|e| GigasttError::ModelLoad {
            path: model_dir.to_string(),
            source: Some(e.into()),
        })
    }

    /// Path to the preferred encoder model for `variant`: INT8 quantized if
    /// present, FP32 otherwise.
    fn encoder_model_path(dir: &Path, variant: ModelVariant) -> std::path::PathBuf {
        let int8 = dir.join(variant.encoder_int8_file());
        if int8.exists() {
            int8
        } else {
            dir.join(variant.encoder_file())
        }
    }

    /// Load a single set of encoder/decoder/joiner ONNX sessions from disk for
    /// `variant`, using the execution provider selected at compile time.
    ///
    /// `encoder_intra_threads` configures the encoder session's intra-op thread
    /// count on the CPU EP only; the CoreML / CUDA loaders ignore it (the
    /// accelerator owns scheduling there).
    fn load_sessions(
        dir: &Path,
        variant: ModelVariant,
        prepacked: &ort::session::builder::PrepackedWeights,
        encoder_intra_threads: usize,
    ) -> anyhow::Result<(Session, Session, Session)> {
        #[cfg(feature = "coreml")]
        {
            let _ = encoder_intra_threads;
            Self::load_sessions_coreml(dir, variant, prepacked)
        }
        #[cfg(feature = "cuda")]
        {
            let _ = encoder_intra_threads;
            Self::load_sessions_cuda(dir, variant, prepacked)
        }
        #[cfg(not(any(feature = "coreml", feature = "cuda")))]
        {
            Self::load_sessions_cpu(dir, variant, prepacked, encoder_intra_threads)
        }
    }

    #[cfg(feature = "coreml")]
    fn load_sessions_coreml(
        dir: &Path,
        variant: ModelVariant,
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
        let encoder_path = Self::encoder_model_path(dir, variant);
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
            .commit_from_file(dir.join(variant.decoder_file()))
            .map_err(ort_err)?;
        let joiner = Session::builder()
            .map_err(ort_err)?
            .with_prepacked_weights(prepacked)
            .map_err(ort_err)?
            .with_execution_providers(&eps)
            .map_err(ort_err)?
            .commit_from_file(dir.join(variant.joint_file()))
            .map_err(ort_err)?;
        Ok((encoder, decoder, joiner))
    }

    #[cfg(feature = "cuda")]
    fn load_sessions_cuda(
        dir: &Path,
        variant: ModelVariant,
        prepacked: &ort::session::builder::PrepackedWeights,
    ) -> anyhow::Result<(Session, Session, Session)> {
        // CUDA EP compiles subgraphs that cannot be re-serialized as ONNX,
        // so we do NOT call `.with_optimized_model_path(...)` here — same
        // reason as the CoreML block. ORT's CUDA EP keeps its own caches
        // internally.
        let cuda_ep = ep::CUDA::default().build();

        let cpu_fallback = ort::execution_providers::CPUExecutionProvider::default();
        let eps = [cuda_ep.clone(), cpu_fallback.into()];
        let encoder_path = Self::encoder_model_path(dir, variant);
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
            .commit_from_file(dir.join(variant.decoder_file()))
            .map_err(ort_err)?;
        let joiner = Session::builder()
            .map_err(ort_err)?
            .with_prepacked_weights(prepacked)
            .map_err(ort_err)?
            .with_execution_providers(&eps)
            .map_err(ort_err)?
            .commit_from_file(dir.join(variant.joint_file()))
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
        variant: ModelVariant,
        prepacked: &ort::session::builder::PrepackedWeights,
        encoder_intra_threads: usize,
    ) -> anyhow::Result<(Session, Session, Session)> {
        let cache_dir = dir.join("optimized_cache");
        std::fs::create_dir_all(&cache_dir)
            .with_context(|| format!("Failed to create ONNX cache dir: {}", cache_dir.display()))?;
        let cpu_fallback = ort::execution_providers::CPUExecutionProvider::default();
        let eps = [cpu_fallback.into()];
        let encoder_path = Self::encoder_model_path(dir, variant);
        // Only the encoder's intra-op count is configurable (it dominates the
        // single-utterance cost); inter-op stays 1 because the Conformer is a
        // near-linear chain, and the decoder/joiner stay intra=1 (tiny ops).
        // Default `encoder_intra_threads == 1` ⇒ identical to the prior build.
        let encoder = Session::builder()
            .map_err(ort_err)?
            .with_prepacked_weights(prepacked)
            .map_err(ort_err)?
            .with_intra_threads(encoder_intra_threads.max(1))
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
            .commit_from_file(dir.join(variant.decoder_file()))
            .map_err(ort_err)?;
        let joiner = Session::builder()
            .map_err(ort_err)?
            .with_prepacked_weights(prepacked)
            .map_err(ort_err)?
            .with_intra_threads(1)
            .map_err(ort_err)?
            .with_inter_threads(1)
            .map_err(ort_err)?
            .commit_from_file(dir.join(variant.joint_file()))
            .map_err(ort_err)?;
        Ok((encoder, decoder, joiner))
    }

    /// Split loaded triplets into an interactive pool and an optional batch
    /// pool of `batch_pool_size` triplets. Always leaves at least one triplet
    /// for the interactive pool; `batch_pool_size == 0` (or a pool too small to
    /// split) yields no batch pool.
    fn split_triplets(
        triplets: Vec<SessionTriplet>,
        batch_pool_size: usize,
    ) -> (SessionPool, Option<SessionPool>) {
        Self::split_pool(triplets, batch_pool_size)
    }

    /// Generic pool split underlying [`Engine::split_triplets`]: partition
    /// `items` into an interactive pool and an optional batch pool of
    /// `batch_pool_size` items, always leaving at least one item interactive.
    /// `batch_pool_size == 0` (or too few items to split) yields no batch pool.
    /// Generic over the item type so the routing can be unit-tested with a
    /// synthetic `Pool<u32>` instead of model-backed `SessionTriplet`s.
    fn split_pool<T: Send>(
        mut items: Vec<T>,
        batch_pool_size: usize,
    ) -> (Pool<T>, Option<Pool<T>>) {
        let n = items.len();
        let batch = Self::batch_split_count(n, batch_pool_size);
        if batch == 0 {
            return (Pool::new(items), None);
        }
        let batch_items = items.split_off(n - batch);
        (Pool::new(items), Some(Pool::new(batch_items)))
    }

    /// Number of triplets to reserve for the batch pool given `n` loaded and a
    /// requested `batch_pool_size`, always leaving at least one for the
    /// interactive pool (so `n <= 1` or `batch_pool_size == 0` yields 0).
    fn batch_split_count(n: usize, batch_pool_size: usize) -> usize {
        batch_pool_size.min(n.saturating_sub(1))
    }

    /// Clamp the requested `pool_size` so the pooled encoder sessions can't
    /// exceed [`POOL_RAM_FRACTION_DENOM`]⁻¹ of total RAM. Each triplet costs
    /// about `encoder_bytes * ENCODER_RESIDENT_MULTIPLIER` resident (the encoder
    /// dominates; decoder/joiner are small), so the max safe pool is
    /// `(total_ram / denom) / per_triplet`, never below 1. Logs a warning when
    /// it clamps. A no-op (returns `requested`) when RAM or encoder size is
    /// unknown (`0`) — we never *raise* the requested size, only lower it.
    ///
    /// Pure and total so the budgeting math is unit-tested without a model or a
    /// real `sysctl`/`sysconf` probe.
    fn cap_pool_size_for_ram(requested: usize, encoder_bytes: u64, total_ram: u64) -> usize {
        if requested <= 1 || encoder_bytes == 0 || total_ram == 0 {
            return requested.max(1);
        }
        let per_triplet = encoder_bytes.saturating_mul(ENCODER_RESIDENT_MULTIPLIER);
        let budget = total_ram / POOL_RAM_FRACTION_DENOM;
        // At least one slot always allowed even if a single triplet exceeds the
        // budget — the pool can't be empty, and partial-load tolerance
        // (`min_size`) handles a genuine OOM at load time.
        let max_slots = (budget / per_triplet.max(1)).max(1) as usize;
        if max_slots < requested {
            tracing::warn!(
                "Capping pool size {requested} -> {max_slots}: \
                 {requested} encoder slots (~{} MiB each) would exceed half of \
                 {} MiB total RAM. Concurrency is reduced; add RAM or lower \
                 --pool-size to silence this.",
                per_triplet / (1024 * 1024),
                total_ram / (1024 * 1024),
            );
            max_slots
        } else {
            requested
        }
    }

    /// Clamp the requested encoder intra-op thread count so the pooled encoder
    /// sessions can't oversubscribe the CPU. Each of `pool_size` triplets can
    /// run concurrently, so the total intra-op parallelism is
    /// `pool_size * threads`; capping that at `logical_cpus` keeps the machine
    /// from thrashing on context switches. The effective per-encoder count is
    /// therefore `clamp(requested, 1, logical_cpus / pool_size)`, never below 1.
    /// Logs a warning when it lowers the request. The default `requested == 1`
    /// always returns `1`, so the built sessions are unchanged.
    ///
    /// Pure and total so the budgeting math is unit-tested without spawning ORT
    /// sessions or probing the real CPU count.
    fn clamp_encoder_intra_threads(
        pool_size: usize,
        requested: usize,
        logical_cpus: usize,
    ) -> usize {
        let requested = requested.max(1);
        let pool_size = pool_size.max(1);
        let logical_cpus = logical_cpus.max(1);
        // Leave at least one thread per encoder even on a machine with fewer
        // logical CPUs than pooled triplets.
        let max_per_encoder = (logical_cpus / pool_size).max(1);
        if requested > max_per_encoder {
            tracing::warn!(
                "Capping encoder intra-op threads {requested} -> {max_per_encoder}: \
                 {pool_size} pooled encoder(s) x {requested} threads would exceed \
                 the {logical_cpus} logical CPU(s) available. Lower --pool-size or \
                 --encoder-intra-threads to silence this."
            );
            max_per_encoder
        } else {
            requested
        }
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
        variant: ModelVariant,
        pool_size: usize,
        min_size: usize,
        prepacked: &ort::session::builder::PrepackedWeights,
        load_one: impl Fn(
            &Path,
            ModelVariant,
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
                            let (encoder, decoder, joiner) = load_one(dir, variant, pp)?;
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
        variant: ModelVariant,
        model_dir: &str,
        pool_size: usize,
        min_size: usize,
        batch_pool_size: usize,
        encoder_intra_threads: usize,
    ) -> anyhow::Result<Self> {
        tracing::info!("Detected model variant: {variant:?}");
        let is_int8 = dir.join(variant.encoder_int8_file()).exists();
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
            variant,
            pool_size,
            min_size,
            &prepacked,
            |d, v, pp| Self::load_sessions(d, v, pp, encoder_intra_threads),
        ) {
            Ok(triplets) => triplets,
            Err(load_err) => {
                tracing::warn!(
                    "CoreML EP failed to load sessions ({load_err:#}); falling back to CPU execution provider"
                );
                // Fresh container: the failed attempt may have pre-packed
                // weights with CoreML-specific kernel layouts.
                let prepacked = ort::session::builder::PrepackedWeights::new();
                Self::load_triplets(dir, variant, pool_size, min_size, &prepacked, |d, v, pp| {
                    Self::load_sessions_cpu(d, v, pp, encoder_intra_threads)
                })?
            }
        };
        #[cfg(not(feature = "coreml"))]
        let triplets =
            Self::load_triplets(dir, variant, pool_size, min_size, &prepacked, |d, v, pp| {
                Self::load_sessions(d, v, pp, encoder_intra_threads)
            })?;

        let tokenizer = Tokenizer::load(&dir.join(variant.vocab_file()))?;
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
            variant,
            punctuator: None,
            itn: false,
            biaser: None,
            vad: None,
            vad_config: crate::vad::VadConfig::default(),
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
                    variant,
                    pool_size,
                    min_size,
                    &prepacked,
                    |d, v, pp| Self::load_sessions_cpu(d, v, pp, encoder_intra_threads),
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
            vad_endpointer: self
                .vad
                .as_ref()
                .map(|_| crate::vad::VadEndpointer::new(&self.vad_config)),
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

        // Feed the VAD on every chunk so trailing silence is tracked
        // continuously (independent of the decode stride). A VAD endpoint forces
        // a decode + finalize this chunk even if the stride gate wouldn't fire.
        // VAD is non-blocking: an inference error is logged and ignored, leaving
        // endpointing to the decoder's blank-run heuristic. With no VAD attached
        // `vad_endpoint` is always false and the path is byte-for-byte unchanged.
        let mut vad_endpoint = false;
        if let (Some(vad), Some(ep)) = (self.vad.as_ref(), state.vad_endpointer.as_mut()) {
            match ep.push(vad, samples) {
                Ok(fired) => vad_endpoint = fired,
                Err(e) => tracing::warn!("VAD endpoint detection failed: {e:#}"),
            }
        }

        let over_cap = state.audio_buffer.len() >= STREAM_MAX_WINDOW_SAMPLES;
        // Stride gate on NEW audio since the last decode (not since the last
        // slide): otherwise a non-finalizing partial would leave the counter
        // high and decode on every subsequent chunk. A VAD endpoint overrides
        // the gate so the utterance finalizes promptly.
        if state.pending_samples < STREAM_DECODE_STRIDE_SAMPLES && !over_cap && !vad_endpoint {
            return Ok(vec![]);
        }
        // Too little audio to extract a frame. Skip — but never when finalizing:
        // a fired VAD endpoint (or cap) must still flush the assembler below,
        // even though `decode_window` will add no new words from a sub-frame
        // buffer. (In practice a VAD endpoint needs ≥ min_silence_ms of trailing
        // audio, so the buffer is always ≫ N_FFT here; this guards the edge.)
        if state.audio_buffer.len() < N_FFT && !vad_endpoint && !over_cap {
            return Ok(vec![]);
        }

        let endpoint = self
            .decode_window(state, triplet)
            .map_err(|e| GigasttError::Inference { source: e.into() })?;
        state.pending_samples = 0;
        let ts = now_timestamp();

        if endpoint || over_cap || vad_endpoint {
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

        // When a VAD is attached, decode only the detected speech regions
        // (skipping silence) and remap word timestamps back to the original
        // timeline. VAD is non-blocking: on any VAD error we log and decode the
        // whole buffer, exactly as if no VAD were attached. With no VAD the path
        // is byte-for-byte the previous behaviour.
        #[cfg_attr(not(feature = "diarization"), allow(unused_mut))]
        let mut words = match &self.vad {
            Some(vad) => match vad.speech_regions(float_samples, &self.vad_config) {
                Ok(regions) => self.decode_speech_regions(float_samples, &regions, triplet)?,
                Err(e) => {
                    tracing::warn!("VAD failed, decoding full audio: {e:#}");
                    self.decode_words(float_samples, triplet)?
                }
            },
            None => self.decode_words(float_samples, triplet)?,
        };

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

        // Optional inverse text normalization (number-words → digits) for the
        // plain `rnnt` head. Runs BEFORE punctuation so the restorer cases the
        // already-digitized text. No-op when disabled; word-level timing is
        // left untouched — only the joined `text` is rewritten.
        let text = if self.itn {
            crate::itn::apply_itn(&text)
        } else {
            text
        };

        // Optional punctuation / casing restoration (plain `rnnt` head). When
        // no punctuator is attached this is a no-op; `restore` itself never
        // fails (returns the input unchanged on internal error). Word-level
        // timing is left untouched — only the joined `text` is restored.
        let text = match &self.punctuator {
            Some(p) => p.restore(&text),
            None => text,
        };

        Ok(TranscribeResult {
            text,
            words,
            duration_s,
        })
    }

    /// Decode a 16 kHz f32 buffer to words: single-pass for short inputs (one
    /// encoder Run over the whole buffer), chunked overlapping windows for long
    /// inputs so encoder activation memory stays O(chunk), not O(file). Both
    /// paths produce the same `Vec<WordInfo>` shape. This is the no-VAD path and
    /// the per-region decode used by [`Engine::decode_speech_regions`].
    fn decode_words(
        &self,
        samples: &[f32],
        triplet: &mut SessionTriplet,
    ) -> Result<Vec<WordInfo>, GigasttError> {
        if samples.len() <= CHUNK_THRESHOLD_SAMPLES {
            let (features, num_frames) = self.features.compute(samples);
            tracing::info!("Extracted {} mel frames", num_frames);
            let mut decoder_state = DecoderState::new(self.tokenizer.blank_id());
            Ok(self
                .run_inference(triplet, &features, num_frames, &mut decoder_state, 0)
                .map_err(|e| GigasttError::Inference { source: e.into() })?
                .0)
        } else {
            self.transcribe_samples_chunked(samples, triplet)
        }
    }

    /// Decode only the VAD-detected speech `regions` of `float_samples`: copy the
    /// speech spans into one silence-free buffer, decode it, then remap each
    /// word's start/end from the compressed (silence-removed) timeline back to
    /// the original timeline via [`crate::vad::remap_compressed_seconds`]. Empty
    /// `regions` (no speech) yields no words.
    fn decode_speech_regions(
        &self,
        float_samples: &[f32],
        regions: &[(usize, usize)],
        triplet: &mut SessionTriplet,
    ) -> Result<Vec<WordInfo>, GigasttError> {
        if regions.is_empty() {
            tracing::info!("VAD found no speech; skipping decode");
            return Ok(Vec::new());
        }
        let speech_len: usize = regions.iter().map(|(s, e)| e - s).sum();
        let mut speech = Vec::with_capacity(speech_len);
        for &(s, e) in regions {
            speech.extend_from_slice(&float_samples[s..e]);
        }
        tracing::info!(
            "VAD kept {}/{} samples ({} speech region(s))",
            speech_len,
            float_samples.len(),
            regions.len()
        );
        let mut words = self.decode_words(&speech, triplet)?;
        for w in &mut words {
            w.start = crate::vad::remap_compressed_seconds(w.start, regions, 16000.0);
            w.end = crate::vad::remap_compressed_seconds(w.end, regions, 16000.0);
        }
        Ok(words)
    }

    /// Long-form decode: split `float_samples` into overlapping windows, encode
    /// and decode each independently with a fresh [`DecoderState`], offset each
    /// chunk's word timestamps by the chunk's absolute start, then stitch the
    /// per-chunk word lists with overlap de-dup via [`stitch_chunk_words`].
    ///
    /// Peak encoder activation memory is bounded by [`CHUNK_WINDOW_SAMPLES`]
    /// rather than the full file length. Chunk starts are aligned to encoder
    /// frame boundaries (multiples of `HOP_LENGTH * ENCODER_SUBSAMPLING`) so the
    /// per-chunk frame offset is exact, matching the streaming path's math.
    fn transcribe_samples_chunked(
        &self,
        float_samples: &[f32],
        triplet: &mut SessionTriplet,
    ) -> Result<Vec<WordInfo>, GigasttError> {
        let total = float_samples.len();
        let stride = CHUNK_WINDOW_SAMPLES - CHUNK_OVERLAP_SAMPLES;
        // Align stride to an encoder-frame boundary so each chunk's frame offset
        // is integral; otherwise the offset would drift by a sub-frame each hop.
        let frame_samples = HOP_LENGTH * ENCODER_SUBSAMPLING;
        let stride = (stride / frame_samples) * frame_samples;
        tracing::info!(
            "Long-form chunked decode: {:.1}s in ~{}s windows ({}s overlap)",
            total as f64 / 16000.0,
            CHUNK_WINDOW_SAMPLES / 16000,
            CHUNK_OVERLAP_SAMPLES / 16000,
        );

        let mut merged: Vec<WordInfo> = Vec::new();
        let mut start = 0usize;
        while start < total {
            let end = (start + CHUNK_WINDOW_SAMPLES).min(total);
            let chunk = &float_samples[start..end];
            let (features, num_frames) = self.features.compute(chunk);
            let frame_offset = start / frame_samples;
            let mut decoder_state = DecoderState::new(self.tokenizer.blank_id());
            let (chunk_words, _endpoint) = self
                .run_inference(
                    triplet,
                    &features,
                    num_frames,
                    &mut decoder_state,
                    frame_offset,
                )
                .map_err(|e| GigasttError::Inference { source: e.into() })?;

            // Seam between the previous chunk's window and this one falls at the
            // midpoint of their overlap region, in absolute seconds.
            let overlap_mid_s = (start as f64 + CHUNK_OVERLAP_SAMPLES as f64 / 2.0) / 16000.0;
            merged = stitch_chunk_words(merged, chunk_words, overlap_mid_s);

            if end == total {
                break;
            }
            start += stride;
        }
        Ok(merged)
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
            self.biaser.as_ref(),
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
        TokenFormatter::tokens_to_words(&self.tokenizer, tokens, frame_offset)
    }
}

/// Merge a later chunk's words into the running `merged` list, de-duplicating
/// the overlap region around `seam_s` (absolute seconds).
///
/// Both lists carry absolute timestamps (each chunk's words were already offset
/// by the chunk start). The heuristic keeps `merged` words whose start is at or
/// before the seam and `next` words whose start is strictly after the seam, so
/// the ~2s overlap is attributed to exactly one chunk: the earlier chunk owns
/// the front half of the overlap, the later chunk owns the back half. A word
/// straddling the seam is decoded with full context in at least one chunk, so
/// no unique word is dropped and no overlap word is emitted twice in the common
/// case. The merged list is monotonic in `start` by construction (the earlier
/// chunk's kept words all start ≤ seam < the later chunk's kept words).
///
/// Pure and free-standing so the stitch policy is unit-testable without a
/// loaded model.
pub(crate) fn stitch_chunk_words(
    mut merged: Vec<WordInfo>,
    next: Vec<WordInfo>,
    seam_s: f64,
) -> Vec<WordInfo> {
    if merged.is_empty() {
        return next;
    }
    // Drop the earlier chunk's tail that reaches past the seam — those words are
    // re-decoded by `next` with more right context, so prefer the later chunk
    // for the back half of the overlap.
    merged.retain(|w| w.start <= seam_s);
    merged.extend(next.into_iter().filter(|w| w.start > seam_s));
    merged
}

/// Groups RNN-T decoded tokens into words at BPE word boundaries (`▁`).
///
/// Split out of `Engine` so the formatting logic is unit-testable without a
/// loaded model — it depends only on the [`Tokenizer`], not on any ONNX
/// session.
pub(crate) struct TokenFormatter;

impl TokenFormatter {
    /// Group `tokens` into words. `frame_offset` shifts per-token frame indices
    /// into absolute stream time; word confidence is the mean over the word's
    /// constituent BPE tokens.
    pub(crate) fn tokens_to_words(
        tokenizer: &Tokenizer,
        tokens: &[decode::TokenInfo],
        frame_offset: usize,
    ) -> Vec<WordInfo> {
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
            let token_text = tokenizer.token_text(token.token_id);
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
    fn test_cap_pool_size_for_ram_clamps_on_low_memory() {
        // 225 MiB encoder, 2x resident => ~450 MiB/triplet. Half of 2 GiB =
        // 1 GiB budget => floor(1024/450) = 2 slots; a request for 4 clamps.
        let enc = 225 * 1024 * 1024;
        let two_gib = 2 * 1024 * 1024 * 1024;
        assert_eq!(Engine::cap_pool_size_for_ram(4, enc, two_gib), 2);
    }

    #[test]
    fn test_cap_pool_size_for_ram_no_clamp_with_ample_ram() {
        // 64 GiB host easily fits a pool of 4 of the same encoder.
        let enc = 225 * 1024 * 1024;
        let sixty_four_gib = 64u64 * 1024 * 1024 * 1024;
        assert_eq!(Engine::cap_pool_size_for_ram(4, enc, sixty_four_gib), 4);
    }

    #[test]
    fn test_cap_pool_size_for_ram_never_below_one() {
        // Even a single triplet larger than the whole budget still yields 1 —
        // the pool can't be empty; partial-load tolerance handles real OOM.
        let huge_enc = 8 * 1024 * 1024 * 1024;
        let small_ram = 1024 * 1024 * 1024;
        assert_eq!(Engine::cap_pool_size_for_ram(4, huge_enc, small_ram), 1);
    }

    #[test]
    fn test_cap_pool_size_for_ram_noop_on_unknown_inputs() {
        // Unknown RAM or encoder size (0) => never lower the request.
        assert_eq!(Engine::cap_pool_size_for_ram(4, 0, 8 << 30), 4);
        assert_eq!(Engine::cap_pool_size_for_ram(4, 200 << 20, 0), 4);
        // pool_size <= 1 is returned as-is (min 1).
        assert_eq!(Engine::cap_pool_size_for_ram(1, 200 << 20, 1 << 30), 1);
        assert_eq!(Engine::cap_pool_size_for_ram(0, 200 << 20, 1 << 30), 1);
    }

    #[test]
    fn test_clamp_encoder_intra_threads() {
        // Default request of 1 is always returned unchanged (no behavior change).
        assert_eq!(Engine::clamp_encoder_intra_threads(2, 1, 10), 1);
        assert_eq!(Engine::clamp_encoder_intra_threads(4, 1, 4), 1);

        // Fits within budget: pool 2 x 4 threads = 8 <= 16 CPUs -> 4.
        assert_eq!(Engine::clamp_encoder_intra_threads(2, 4, 16), 4);

        // Over budget: pool 4 x 4 = 16 > 10 CPUs -> floor(10/4) = 2 per encoder.
        assert_eq!(Engine::clamp_encoder_intra_threads(4, 4, 10), 2);

        // More pooled encoders than CPUs still leaves at least 1 thread each.
        assert_eq!(Engine::clamp_encoder_intra_threads(8, 4, 4), 1);

        // Zero inputs are floored to 1 (total function, never panics/divides-by-0).
        assert_eq!(Engine::clamp_encoder_intra_threads(0, 4, 8), 4);
        assert_eq!(Engine::clamp_encoder_intra_threads(2, 0, 8), 1);
        assert_eq!(Engine::clamp_encoder_intra_threads(2, 4, 0), 1);
    }

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
    fn test_split_pool_routes_items_to_two_pools() {
        // Exercises the real split underlying `split_triplets` with a synthetic
        // `Pool<u32>` (no model). 4 items, batch 1 → interactive 3, batch 1.
        let (pool, batch) = Engine::split_pool(vec![1u32, 2, 3, 4], 1);
        assert_eq!(pool.total(), 3);
        assert_eq!(batch.as_ref().map(|b| b.total()), Some(1));

        // batch_pool_size 0 → split disabled, no batch pool.
        let (pool, batch) = Engine::split_pool(vec![1u32, 2, 3, 4], 0);
        assert_eq!(pool.total(), 4);
        assert!(batch.is_none());

        // Over-request clamps so at least one triplet stays interactive.
        let (pool, batch) = Engine::split_pool(vec![1u32, 2, 3], 9);
        assert_eq!(pool.total(), 1);
        assert_eq!(batch.as_ref().map(|b| b.total()), Some(2));

        // A single item can't be split.
        let (pool, batch) = Engine::split_pool(vec![1u32], 1);
        assert_eq!(pool.total(), 1);
        assert!(batch.is_none());
    }

    #[test]
    fn test_token_formatter_groups_words() {
        // `▁` (U+2581) marks a new word; continuation tokens have no prefix.
        let tok = Tokenizer::from_tokens(vec![
            "\u{2581}hel".into(), // 0: new word
            "lo".into(),          // 1: continuation
            "\u{2581}wor".into(), // 2: new word
            "ld".into(),          // 3: continuation
        ]);
        let tokens = vec![
            decode::TokenInfo {
                token_id: 0,
                frame_index: 0,
                confidence: 0.9,
            },
            decode::TokenInfo {
                token_id: 1,
                frame_index: 1,
                confidence: 0.8,
            },
            decode::TokenInfo {
                token_id: 2,
                frame_index: 2,
                confidence: 0.95,
            },
            decode::TokenInfo {
                token_id: 3,
                frame_index: 3,
                confidence: 0.85,
            },
        ];
        let words = TokenFormatter::tokens_to_words(&tok, &tokens, 0);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "hello");
        assert_eq!(words[1].word, "world");
        // Mean confidence per word.
        assert!((words[0].confidence - 0.85).abs() < 1e-6);
        assert!((words[1].confidence - 0.90).abs() < 1e-6);
        // Frame timing (SECONDS_PER_FRAME = 0.04).
        assert!((words[0].start - 0.0).abs() < 1e-9);
        assert!((words[0].end - 0.04).abs() < 1e-9);
        assert!((words[1].start - 0.08).abs() < 1e-9);
    }

    #[test]
    fn test_token_formatter_empty_tokens() {
        let tok = Tokenizer::from_tokens(vec!["\u{2581}a".into()]);
        assert!(TokenFormatter::tokens_to_words(&tok, &[], 0).is_empty());
    }

    #[test]
    fn test_token_formatter_frame_offset_shifts_time() {
        let tok = Tokenizer::from_tokens(vec!["\u{2581}x".into()]);
        let tokens = vec![decode::TokenInfo {
            token_id: 0,
            frame_index: 0,
            confidence: 1.0,
        }];
        let words = TokenFormatter::tokens_to_words(&tok, &tokens, 10);
        assert_eq!(words.len(), 1);
        // frame_offset 10 → start = 10 * 0.04 = 0.4.
        assert!((words[0].start - 0.4).abs() < 1e-9);
    }

    fn word(text: &str, start: f64, end: f64) -> WordInfo {
        WordInfo::new(text, start, end, 1.0, None)
    }

    #[test]
    fn test_stitch_first_chunk_passes_through() {
        // An empty `merged` (the very first chunk) is returned verbatim.
        let next = vec![word("a", 0.0, 0.5), word("b", 0.6, 1.0)];
        let out = stitch_chunk_words(Vec::new(), next.clone(), 11.0);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].word, "a");
        assert_eq!(out[1].word, "b");
    }

    #[test]
    fn test_stitch_dedups_overlap_no_drop_no_dup() {
        // Two 24s windows with a 22s stride: chunk B starts at 22s, overlap
        // [22s, 24s], seam at 23s. The word "dup" at ~22.5s is decoded by both
        // chunks; the seam attributes it to exactly one. No unique word lost.
        let chunk_a = vec![
            word("first", 1.0, 1.4),    // unique to A
            word("middle", 21.0, 21.4), // unique to A, before overlap
            word("dup", 22.4, 22.8),    // in overlap, before seam → kept from A
        ];
        // B's words are already offset by its 22s start.
        let chunk_b = vec![
            word("dup", 22.5, 22.9),   // same word re-decoded → after-seam copy dropped
            word("later", 25.0, 25.4), // unique to B
            word("end", 40.0, 40.4),   // unique to B
        ];
        let seam_s = 22.0 + CHUNK_OVERLAP_SAMPLES as f64 / 2.0 / 16000.0; // 23.0
        assert!((seam_s - 23.0).abs() < 1e-9);

        let out = stitch_chunk_words(chunk_a, chunk_b, seam_s);
        let texts: Vec<&str> = out.iter().map(|w| w.word.as_str()).collect();
        // "dup" appears exactly once (A's copy, before the seam); nothing dropped.
        assert_eq!(texts, vec!["first", "middle", "dup", "later", "end"]);
        // Monotonic in `start`.
        for w in out.windows(2) {
            assert!(w[0].start <= w[1].start, "not monotonic: {:?}", out);
        }
    }

    #[test]
    fn test_stitch_drops_a_tail_past_seam() {
        // A word decoded by A past the seam is dropped in favour of B's
        // fuller-context copy; the back half of the overlap belongs to B.
        let chunk_a = vec![word("keep", 22.0, 22.4), word("a_tail", 23.5, 23.9)];
        let chunk_b = vec![word("b_seam", 23.2, 23.6), word("b_late", 30.0, 30.4)];
        let out = stitch_chunk_words(chunk_a, chunk_b, 23.0);
        let texts: Vec<&str> = out.iter().map(|w| w.word.as_str()).collect();
        assert_eq!(texts, vec!["keep", "b_seam", "b_late"]);
    }

    #[test]
    fn test_stitch_timestamp_offset_math() {
        // The chunked path offsets a chunk's frame indices by
        // start_samples / (HOP_LENGTH * ENCODER_SUBSAMPLING). Verify that a word
        // at frame 0 of a chunk starting `start_samples` in lands at the right
        // absolute time via `tokens_to_words` (the same offset the engine feeds).
        let tok = Tokenizer::from_tokens(vec!["\u{2581}w".into()]);
        let tokens = vec![decode::TokenInfo {
            token_id: 0,
            frame_index: 0,
            confidence: 1.0,
        }];
        let start_samples = 16000 * 22; // chunk starts at 22s
        let frame_offset = start_samples / (HOP_LENGTH * ENCODER_SUBSAMPLING);
        let words = TokenFormatter::tokens_to_words(&tok, &tokens, frame_offset);
        assert_eq!(words.len(), 1);
        // frame 0 + offset → absolute start == 22.0s exactly (aligned stride).
        assert!(
            (words[0].start - 22.0).abs() < 1e-9,
            "got {}",
            words[0].start
        );
    }

    #[test]
    #[allow(clippy::assertions_on_constants)] // intentional compile-time sanity check on the chunk constants
    fn test_chunk_constants_sane() {
        // Window > overlap (positive stride) and threshold ≥ window so the
        // single-pass path covers everything up to one full window.
        assert!(CHUNK_WINDOW_SAMPLES > CHUNK_OVERLAP_SAMPLES);
        assert!(CHUNK_THRESHOLD_SAMPLES >= CHUNK_WINDOW_SAMPLES);
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
    fn test_now_timestamp_monotonic_and_epoch_aligned() {
        // Locks in the monotonic-anchor contract: two successive reads never go
        // backwards (immune to NTP steps), and the value stays Unix-epoch
        // aligned. A regression to a plain wall-clock read could violate either.
        let a = now_timestamp();
        let b = now_timestamp();
        assert!(
            b >= a,
            "now_timestamp must be non-decreasing (monotonic anchor)"
        );
        // Comfortably after 2023-11-14 and before a far-future sanity bound.
        assert!(
            a > 1_700_000_000.0,
            "timestamp must stay Unix-epoch aligned"
        );
        assert!(a < 4_000_000_000.0, "timestamp exceeds a sane upper bound");
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
        let path = Engine::encoder_model_path(dir.path(), ModelVariant::E2eRnnt);
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
        let path = Engine::encoder_model_path(dir.path(), ModelVariant::E2eRnnt);
        assert_eq!(path.file_name().unwrap(), "v3_e2e_rnnt_encoder.onnx");
    }

    #[test]
    fn test_encoder_model_path_rnnt_prefers_int8() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("v3_rnnt_encoder.onnx"), b"fp32").unwrap();
        std::fs::write(dir.path().join("v3_rnnt_encoder_int8.onnx"), b"int8").unwrap();
        let path = Engine::encoder_model_path(dir.path(), ModelVariant::Rnnt);
        assert_eq!(
            path.file_name().unwrap(),
            "v3_rnnt_encoder_int8.onnx",
            "INT8 rnnt encoder must win when both files exist"
        );
    }

    #[test]
    fn test_encoder_model_path_rnnt_falls_back_to_fp32() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("v3_rnnt_encoder.onnx"), b"fp32").unwrap();
        let path = Engine::encoder_model_path(dir.path(), ModelVariant::Rnnt);
        assert_eq!(path.file_name().unwrap(), "v3_rnnt_encoder.onnx");
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
