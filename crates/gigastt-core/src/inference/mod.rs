//! ONNX Runtime inference engine for GigaAM v3 e2e_rnnt.
//!
//! Loads encoder, decoder, and joiner ONNX models and runs the RNN-T streaming decode loop.

pub mod audio;
mod bias;
mod ctc;
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
// `EmbeddingError`, `EmbeddingExtractor`, and `FbankOnnxExtractor` were
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
    FbankOnnxExtractor, Pipeline, VadConfig,
};

#[cfg(feature = "diarization")]
const SPEAKER_EMBEDDING_DIM: usize = 256;
#[cfg(feature = "diarization")]
const SPEAKER_POOL_SIZE: usize = 4;

/// Adapter that lets a single shared [`FbankOnnxExtractor`] back the
/// per-session [`StreamingPipeline`]s, which take ownership of their extractor.
/// The ONNX session pool inside the extractor is shared across sessions via `Arc`.
#[cfg(feature = "diarization")]
#[allow(deprecated)] // legacy FbankOnnxExtractor — see import note above
pub struct SharedExtractor(std::sync::Arc<FbankOnnxExtractor>);

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
use serde::Serialize;
use std::ops::{Deref, DerefMut};
use std::path::Path;

use crate::error::GigasttError;
use crate::model::ModelVariant;
use crate::runtime::factory::Runtime;
#[allow(unused_imports)]
use crate::runtime::factory::RuntimeFactory;
use crate::runtime::production_factory_variant;
use crate::runtime::session::RuntimeSession;
use crate::runtime::tensor::{Shape, Tensor, TensorData, TensorDataView};

use features::MelSpectrogram;
use tokenizer::Tokenizer;

#[cfg(feature = "diarization")]
#[allow(deprecated)] // legacy FbankOnnxExtractor — see import note above
fn load_speaker_encoder(model_path: &Path, pool_size: usize) -> anyhow::Result<FbankOnnxExtractor> {
    FbankOnnxExtractor::new(model_path, SPEAKER_EMBEDDING_DIM, pool_size)
}

/// Number of mel frequency bins used for spectrogram features.
pub const N_MELS: usize = 64;
/// FFT window size in samples (320 samples = 20ms at 16kHz).
pub const N_FFT: usize = 320;
/// Hop length between consecutive FFT frames in samples (160 samples = 10ms at 16kHz).
pub const HOP_LENGTH: usize = 160;
/// Hidden dimension of the RNN-T prediction (decoder) network.
pub const PRED_HIDDEN: usize = 320;

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

/// Resolve which recognition head the engine should load: an explicit
/// `override_` (from `--model-variant`) always wins, else auto-detect from the
/// on-disk layout (`rnnt` precedence). Extracted so the override precedence —
/// the fix that keeps `--model-variant` effective when a directory holds more
/// than one head — is unit-testable without model files.
fn resolve_load_variant(override_: Option<ModelVariant>, model_dir: &Path) -> Option<ModelVariant> {
    override_.or_else(|| ModelVariant::detect_in_dir(model_dir))
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
///
/// The RNN-T heads populate all three sessions. The encoder-only CTC heads leave
/// `decoder` / `joiner` as `None` — the CTC branch in `run_inference` decodes
/// straight from the encoder output and never touches them, so loading them would
/// only waste encoder-sized RAM.
pub struct SessionTriplet {
    pub(crate) encoder: Box<dyn RuntimeSession>,
    pub(crate) decoder: Option<Box<dyn RuntimeSession>>,
    pub(crate) joiner: Option<Box<dyn RuntimeSession>>,
    /// Reusable encoder input tensors: `[audio_signal [1, N_MELS, num_frames], length [1]]`.
    /// Resized and overwritten in `run_inference` to avoid per-call allocations.
    pub(crate) encoder_inputs: Vec<Tensor>,
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
    #[cfg(feature = "async-pool")]
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
    #[cfg(feature = "async-pool")]
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
                    #[cfg(feature = "async-pool")]
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
    #[allow(deprecated)] // legacy FbankOnnxExtractor — see import note above
    pub speaker_encoder: Option<std::sync::Arc<FbankOnnxExtractor>>,
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

    /// Reject a [`TranscribeOverrides`] that turns a knob on without the backing
    /// resource loaded. Call this *before* checking out a session so the request
    /// fails fast (the REST layer maps the error to a `409`). Turning a knob off
    /// (`Some(false)`) and ITN in either direction are always valid — ITN is pure
    /// code with no model to load.
    ///
    /// # Errors
    ///
    /// - [`OverrideError::VadNotLoaded`] when `vad = Some(true)` but no VAD is
    ///   attached.
    /// - [`OverrideError::PunctuationNotAvailable`] when `punctuation = Some(true)`
    ///   but no punctuator is attached.
    pub fn validate_overrides(&self, o: &TranscribeOverrides) -> Result<(), OverrideError> {
        if o.vad == Some(true) && self.vad.is_none() {
            return Err(OverrideError::VadNotLoaded);
        }
        if o.punctuation == Some(true) && self.punctuator.is_none() {
            return Err(OverrideError::PunctuationNotAvailable);
        }
        Ok(())
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
        Self::load_with_pools_threads_variant(
            model_dir,
            None,
            pool_size,
            min_size,
            batch_pool_size,
            encoder_intra_threads,
        )
    }

    /// Like [`Engine::load_with_pools_threads`], but lets the caller force which
    /// recognition head to load instead of auto-detecting it.
    ///
    /// When `variant` is `Some(v)`, head `v` is loaded (and the load fails with a
    /// clear `ModelLoad` error if `v`'s files aren't in `model_dir`). When it is
    /// `None`, the on-disk layout is auto-detected with `rnnt` precedence, exactly
    /// as [`Engine::load_with_pools_threads`]. This is the entry point that makes
    /// `--model-variant` effective when a directory holds more than one head.
    pub fn load_with_pools_threads_variant(
        model_dir: &str,
        variant: Option<ModelVariant>,
        pool_size: usize,
        min_size: usize,
        batch_pool_size: usize,
        encoder_intra_threads: usize,
    ) -> Result<Self, GigasttError> {
        let dir = Path::new(model_dir);
        // Resolve the head once, up front: an explicit `variant` wins, else detect
        // from disk (rnnt precedence). Resolving here (not just inside
        // `load_with_factory`) keeps the RAM cap below sized to the head that will
        // actually load.
        let Some(variant) = resolve_load_variant(variant, dir) else {
            return Err(GigasttError::ModelLoad {
                path: dir.display().to_string(),
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

        let factory = production_factory_variant(dir, Some(variant));
        Self::load_with_factory(
            dir,
            Some(variant),
            pool_size,
            min_size,
            batch_pool_size,
            factory,
            encoder_intra_threads,
        )
    }

    /// Package-private factory-based loader. Used by production code paths and
    /// by tests that inject a [`crate::runtime::factory::RuntimeFactory`].
    pub(crate) fn load_with_factory(
        model_dir: &Path,
        variant_override: Option<ModelVariant>,
        pool_size: usize,
        min_size: usize,
        batch_pool_size: usize,
        factory: Box<dyn crate::runtime::factory::RuntimeFactory>,
        encoder_intra_threads: usize,
    ) -> Result<Self, GigasttError> {
        // Honor an explicit variant (e.g. from `--model-variant`) when the caller
        // resolved one; otherwise auto-detect from the on-disk layout (rnnt
        // precedence). Passing the override through is what makes `--model-variant`
        // effective when a directory holds more than one head — without it the
        // engine always re-detects and silently loads the highest-precedence head.
        let Some(variant) = resolve_load_variant(variant_override, model_dir) else {
            return Err(GigasttError::ModelLoad {
                path: model_dir.display().to_string(),
                source: None,
            });
        };
        let pool_size = pool_size.max(1);
        let min_size = min_size.clamp(1, pool_size);

        let runtime = factory.create(encoder_intra_threads)?;
        let model_load = |e: anyhow::Error| GigasttError::ModelLoad {
            path: model_dir.display().to_string(),
            source: Some(e.into()),
        };

        tracing::info!("Detected model variant: {variant:?}");
        // The candle backend loads FP32 `candle/*.safetensors` and ignores the
        // INT8 ONNX encoder, so report no INT8 there and gate out the INT8 / ONNX
        // / CPU-EP logs below (which are wrong for candle), emitting an accurate
        // candle line instead. The default (ort) logging is unchanged.
        let is_int8 =
            !cfg!(feature = "candle") && model_dir.join(variant.encoder_int8_file()).exists();
        if !cfg!(feature = "candle") {
            if is_int8 {
                tracing::info!("Using INT8 quantized encoder");
            } else if !cfg!(feature = "ane") {
                // On the default ORT path the INT8 encoder is expected. Warn so
                // the operator knows inference will be slower and larger than
                // intended. Fix: run `gigastt quantize` or re-run `gigastt download`.
                tracing::warn!(
                    "INT8 encoder not found — loading FP32 encoder (4× larger, slower). \
                     Run `gigastt download` or `gigastt quantize` to generate it."
                );
            }

            tracing::info!(
                "Loading ONNX models from {} (pool_size={pool_size})...",
                model_dir.display()
            );
        }

        #[cfg(feature = "candle")]
        tracing::info!(
            "Loading Candle models from {} (pool_size={pool_size})...",
            model_dir.display()
        );
        #[cfg(feature = "coreml")]
        tracing::info!("Using CoreML execution provider (Neural Engine + CPU)");
        #[cfg(feature = "cuda")]
        tracing::info!("Using CUDA execution provider (falls back to CPU if unavailable)");
        #[cfg(not(any(feature = "coreml", feature = "cuda", feature = "candle")))]
        tracing::info!("Using CPU execution provider");

        // CoreML can reject a model at load time; fall back to CPU if that happens.
        #[cfg(feature = "coreml")]
        let triplets = match Self::load_triplets_runtime(
            &*runtime, model_dir, variant, pool_size, min_size,
        )
        .map_err(model_load)
        {
            Ok(triplets) => triplets,
            Err(load_err) => {
                tracing::warn!(
                    "CoreML EP failed to load sessions ({load_err:#}); falling back to CPU execution provider"
                );
                let cpu_factory = factory.cpu_fallback();
                let runtime = cpu_factory.create(encoder_intra_threads)?;
                Self::load_triplets_runtime(&*runtime, model_dir, variant, pool_size, min_size)
                    .map_err(model_load)?
            }
        };
        #[cfg(not(feature = "coreml"))]
        let triplets =
            Self::load_triplets_runtime(&*runtime, model_dir, variant, pool_size, min_size)
                .map_err(model_load)?;

        let tokenizer =
            Tokenizer::load(&model_dir.join(variant.vocab_file())).map_err(model_load)?;
        let features = FeatureExtractor::new();

        tracing::info!(
            "Models loaded (vocab_size={}, pool_size={pool_size})",
            tokenizer.vocab_size()
        );

        #[cfg(feature = "diarization")]
        let speaker_encoder = {
            let model_path = model_dir.join("wespeaker_resnet34.onnx");
            if model_path.exists() {
                match load_speaker_encoder(&model_path, SPEAKER_POOL_SIZE) {
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

        // CoreML compiles its graph partitions lazily, so sessions that loaded
        // fine can still fail at the first `Run()`. Probe one triplet now; if the
        // probe fails, rebuild the pool on the CPU EP.
        #[cfg(feature = "coreml")]
        let engine = probe_or_rebuild(
            engine,
            |e: &Self| e.warmup_one().map_err(anyhow::Error::from),
            |mut e, probe_err| {
                tracing::warn!(
                    "CoreML EP failed at runtime ({probe_err:#}); falling back to CPU execution provider"
                );
                let cpu_factory = factory.cpu_fallback();
                let runtime = cpu_factory
                    .create(encoder_intra_threads)
                    .map_err(|e| anyhow::anyhow!(e))?;
                let triplets =
                    Self::load_triplets_runtime(&*runtime, model_dir, variant, pool_size, min_size)?;
                let (pool, batch_pool) = Self::split_triplets(triplets, batch_pool_size);
                e.pool = pool;
                e.batch_pool = batch_pool;
                Ok(e)
            },
        )
        .map_err(model_load)?;

        Ok(engine)
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

    /// Load up to `pool_size` session triplets in parallel through the given
    /// [`Runtime`], tolerating a partial pool down to `min_size`.
    fn load_triplets_runtime(
        runtime: &dyn Runtime,
        dir: &Path,
        variant: ModelVariant,
        pool_size: usize,
        min_size: usize,
    ) -> anyhow::Result<Vec<SessionTriplet>> {
        let encoder_path = Self::encoder_model_path(dir, variant);
        // CTC is encoder-only: no decoder/joiner ONNX exists on disk, and the CTC
        // branch in `run_inference` returns right after the encoder run without
        // touching them. Load them only for the RNN-T heads (leaving `None` for
        // CTC avoids holding an unused, never-run session per pool triplet).
        let is_ctc = variant.is_ctc();
        let decoder_path = dir.join(variant.decoder_file());
        let joiner_path = dir.join(variant.joint_file());

        let results: Vec<anyhow::Result<SessionTriplet>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..pool_size)
                .map(|i| {
                    let encoder_path = &encoder_path;
                    let decoder_path = &decoder_path;
                    let joiner_path = &joiner_path;
                    s.spawn(move || {
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            tracing::info!(
                                "Loading session triplet {}/{pool_size} (shared runtime)",
                                i + 1
                            );
                            let encoder = runtime
                                .load_session(encoder_path, true)
                                .map_err(|e| anyhow::anyhow!(e))?;
                            let (decoder, joiner) = if is_ctc {
                                (None, None)
                            } else {
                                let decoder = runtime
                                    .load_session(decoder_path, false)
                                    .map_err(|e| anyhow::anyhow!(e))?;
                                let joiner = runtime
                                    .load_session(joiner_path, false)
                                    .map_err(|e| anyhow::anyhow!(e))?;
                                (Some(decoder), Some(joiner))
                            };
                            Ok(SessionTriplet {
                                encoder,
                                decoder,
                                joiner,
                                encoder_inputs: vec![
                                    Tensor::new(
                                        Shape::new(vec![1, N_MELS, 1]),
                                        TensorData::F32(vec![0.0; N_MELS]),
                                    )?,
                                    Tensor::new(Shape::new(vec![1]), TensorData::I64(vec![0]))?,
                                ],
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
    #[cfg(feature = "file-decode")]
    pub fn transcribe_file(
        &self,
        path: &str,
        triplet: &mut SessionTriplet,
    ) -> Result<TranscribeResult, GigasttError> {
        self.transcribe_file_with_overrides(path, triplet, &TranscribeOverrides::default())
    }

    /// Like [`Engine::transcribe_file`] but applies per-request recognition-knob
    /// [`overrides`](TranscribeOverrides). With `TranscribeOverrides::default()`
    /// this is byte-for-byte [`Engine::transcribe_file`]; the plain method
    /// delegates here so binding call sites (FFI / UniFFI / Node) keep the
    /// no-override signature unchanged.
    #[cfg(feature = "file-decode")]
    pub fn transcribe_file_with_overrides(
        &self,
        path: &str,
        triplet: &mut SessionTriplet,
        overrides: &TranscribeOverrides,
    ) -> Result<TranscribeResult, GigasttError> {
        let float_samples =
            audio::decode_audio_file(path).map_err(|e| GigasttError::InvalidAudio {
                reason: format!("{e:#}"),
            })?;
        self.transcribe_samples_with_overrides(&float_samples, triplet, overrides)
    }

    /// Transcribe audio from raw bytes in memory (no temp file needed).
    ///
    /// Backwards-compatible shim: clones `data` into a [`bytes::Bytes`] and
    /// delegates to [`Engine::transcribe_bytes_shared`]. Prefer the shared
    /// variant on hot paths (REST/SSE) to avoid the extra copy.
    #[cfg(feature = "file-decode")]
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
    #[cfg(feature = "file-decode")]
    pub fn transcribe_bytes_shared(
        &self,
        data: bytes::Bytes,
        triplet: &mut SessionTriplet,
    ) -> Result<TranscribeResult, GigasttError> {
        self.transcribe_bytes_shared_with_overrides(data, triplet, &TranscribeOverrides::default())
    }

    /// Like [`Engine::transcribe_bytes_shared`] but applies per-request
    /// recognition-knob [`overrides`](TranscribeOverrides). With
    /// `TranscribeOverrides::default()` this is byte-for-byte
    /// [`Engine::transcribe_bytes_shared`]; the plain method delegates here so
    /// the zero-copy REST call site can opt into overrides without changing the
    /// no-override signature that other callers rely on.
    #[cfg(feature = "file-decode")]
    pub fn transcribe_bytes_shared_with_overrides(
        &self,
        data: bytes::Bytes,
        triplet: &mut SessionTriplet,
        overrides: &TranscribeOverrides,
    ) -> Result<TranscribeResult, GigasttError> {
        let float_samples =
            audio::decode_audio_bytes_shared(data).map_err(|e| GigasttError::InvalidAudio {
                reason: format!("{e:#}"),
            })?;
        self.transcribe_samples_with_overrides(&float_samples, triplet, overrides)
    }

    /// Transcribe a multi-channel recording with one speaker label per channel.
    ///
    /// Runs the engine once per channel sequentially on the supplied triplet. The
    /// caller is responsible for deciding whether to use this mode (e.g. after
    /// checking for dual-mono). Channel 0 becomes `speaker_0`, channel 1
    /// `speaker_1`, and so on. Results are merged into a single chronologically
    /// ordered transcript.
    ///
    /// Per-channel `text` fields are ignored: the merged transcript's text is
    /// rebuilt from the merged words after the final ITN/punctuation pass.
    #[cfg(feature = "file-decode")]
    pub fn transcribe_channels(
        &self,
        channels: &[Vec<f32>],
        triplet: &mut SessionTriplet,
    ) -> Result<TranscribeResult, GigasttError> {
        if channels.is_empty() {
            return Ok(TranscribeResult {
                text: String::new(),
                words: Vec::new(),
                duration_s: 0.0,
            });
        }

        let mut per_channel = Vec::with_capacity(channels.len());
        let overrides = TranscribeOverrides::default();
        for channel_samples in channels {
            let words = self.decode_words_for_samples(channel_samples, triplet, &overrides)?;
            let duration_s = channel_samples.len() as f64 / 16000.0;
            per_channel.push(TranscribeResult {
                text: String::new(),
                words,
                duration_s,
            });
        }

        let merged = merge_channel_results(per_channel);
        Ok(self.finish_transcribe_result(merged.words, merged.duration_s, &overrides))
    }

    /// Run the full mel + encoder + RNN-T decode pipeline on an already-decoded
    /// 16 kHz f32 sample buffer. Shared tail of [`Engine::transcribe_file`] and
    /// [`Engine::transcribe_bytes_shared`].
    fn transcribe_samples(
        &self,
        float_samples: &[f32],
        triplet: &mut SessionTriplet,
    ) -> Result<TranscribeResult, GigasttError> {
        self.transcribe_samples_with_overrides(
            float_samples,
            triplet,
            &TranscribeOverrides::default(),
        )
    }

    /// Override-aware tail of the file-transcription pipeline. With
    /// `TranscribeOverrides::default()` (all `None`) it is byte-for-byte the
    /// engine-default path; each `Some(_)` field flips the corresponding
    /// post-processing knob for this call only. `overrides` is assumed already
    /// validated by [`Engine::validate_overrides`] — an on-request with the
    /// resource missing degrades gracefully (VAD absent → whole-buffer decode)
    /// rather than erroring here.
    fn transcribe_samples_with_overrides(
        &self,
        float_samples: &[f32],
        triplet: &mut SessionTriplet,
        overrides: &TranscribeOverrides,
    ) -> Result<TranscribeResult, GigasttError> {
        let wall_start = std::time::Instant::now();
        let duration_s = float_samples.len() as f64 / 16000.0;

        #[cfg_attr(not(feature = "diarization"), allow(unused_mut))]
        let mut words = self.decode_words_for_samples(float_samples, triplet, overrides)?;

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

        let result = self.finish_transcribe_result(words, duration_s, overrides);

        let wall_s = wall_start.elapsed().as_secs_f64();
        let rtf = if duration_s > 0.0 {
            wall_s / duration_s
        } else {
            0.0
        };
        let encoder_label = if self.int8 { "int8" } else { "fp32" };
        let backend_label = if cfg!(feature = "candle") {
            "candle"
        } else if cfg!(feature = "ane") {
            "ane"
        } else if cfg!(feature = "coreml") {
            "coreml"
        } else if cfg!(feature = "cuda") {
            "cuda"
        } else {
            "cpu"
        };
        tracing::info!(
            audio_s = format_args!("{duration_s:.2}"),
            wall_s = format_args!("{wall_s:.2}"),
            rtf = format_args!("{rtf:.3}"),
            encoder = format_args!("{encoder_label}/{backend_label}"),
            "transcribe complete"
        );

        Ok(result)
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

    /// Decode a 16 kHz f32 buffer to words, applying VAD if configured.
    ///
    /// The VAD path is taken only when a VAD is attached AND the caller hasn't
    /// opted out via `overrides.vad`. `?vad=false` forces whole-buffer decode
    /// even on a VAD-enabled engine; `None` uses the engine default (VAD path
    /// iff a VAD is attached). `vad = Some(true)` on a VAD-less engine can't
    /// reach here — callers should validate overrides first — but the
    /// `self.vad.is_some()` guard keeps this correct regardless.
    fn decode_words_for_samples(
        &self,
        float_samples: &[f32],
        triplet: &mut SessionTriplet,
        overrides: &TranscribeOverrides,
    ) -> Result<Vec<WordInfo>, GigasttError> {
        let use_vad = self.vad.is_some() && overrides.vad.unwrap_or(true);
        match (use_vad, &self.vad) {
            (true, Some(vad)) => match vad.speech_regions(float_samples, &self.vad_config) {
                Ok(regions) => self.decode_speech_regions(float_samples, &regions, triplet),
                Err(e) => {
                    tracing::warn!("VAD failed, decoding full audio: {e:#}");
                    self.decode_words(float_samples, triplet)
                }
            },
            _ => self.decode_words(float_samples, triplet),
        }
    }

    /// Build the final [`TranscribeResult`] from raw words: join text, apply ITN,
    /// and apply punctuation restoration. Word-level timing is left untouched.
    /// Per-request overrides win over engine defaults; a `None` override keeps
    /// the boot policy.
    fn finish_transcribe_result(
        &self,
        words: Vec<WordInfo>,
        duration_s: f64,
        overrides: &TranscribeOverrides,
    ) -> TranscribeResult {
        let text: String = words
            .iter()
            .map(|w| w.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        // Optional inverse text normalization (number-words → digits) for the
        // plain `rnnt` head. Runs BEFORE punctuation so the restorer cases the
        // already-digitized text. No-op when disabled; word-level timing is
        // left untouched — only the joined `text` is rewritten. The per-request
        // override wins over the engine default; `None` keeps the boot policy.
        let text = if overrides.itn.unwrap_or(self.itn) {
            crate::itn::apply_itn(&text)
        } else {
            text
        };

        // Optional punctuation / casing restoration (plain `rnnt` head). The
        // engine default is "on iff a punctuator is attached"; a per-request
        // override can force it off (`Some(false)`) or on (`Some(true)`, only
        // reachable when a punctuator is loaded — the handler 409s otherwise).
        // The `self.punctuator` guard keeps this a no-op when none is attached;
        // `restore` itself never fails (returns the input unchanged on error).
        // Word-level timing is left untouched — only the joined `text` changes.
        let text = match &self.punctuator {
            Some(p) if overrides.punctuation.unwrap_or(true) => p.restore(&text),
            _ => text,
        };

        TranscribeResult {
            text,
            words,
            duration_s,
        }
    }

    fn run_inference(
        &self,
        triplet: &mut SessionTriplet,
        features: &[f32],
        num_frames: usize,
        decoder_state: &mut DecoderState,
        frame_offset: usize,
    ) -> anyhow::Result<(Vec<WordInfo>, bool)> {
        // Reuse the encoder input tensors: resize the signal tensor to the
        // current frame count and overwrite both buffers in place.
        triplet.encoder_inputs[0].resize_to(Shape::new(vec![1, N_MELS, num_frames]));
        triplet.encoder_inputs[0]
            .as_f32_mut()
            .context("encoder signal tensor is not f32")?
            .copy_from_slice(features);
        triplet.encoder_inputs[1]
            .as_i64_mut()
            .context("encoder length tensor is not i64")?[0] = num_frames as i64;

        let enc_start = std::time::Instant::now();
        let encoder_outputs = triplet
            .encoder
            .run(&triplet.encoder_inputs)
            .context("Encoder inference failed")?;
        tracing::info!(
            elapsed_ms = enc_start.elapsed().as_millis() as u64,
            "encoder_inference"
        );

        let enc_len = match encoder_outputs[1].view().data() {
            TensorDataView::I32(v) => usize::try_from(v[0]).context("Negative encoder length")?,
            TensorDataView::I64(v) => usize::try_from(v[0]).context("Negative encoder length")?,
            _ => anyhow::bail!("Unexpected encoder length tensor type"),
        };

        tracing::debug!("Encoder output: {} frames", enc_len);

        // CTC head: the single encoder emits per-frame class log-probs
        // (`[1, T', 71]`, row-major). Greedy-decode them directly — there is no
        // prediction network / joiner, so we return before the RNN-T block
        // borrows `encoder_outputs` for the decode loop.
        if self.variant.is_ctc() {
            let log_probs = encoder_outputs[0]
                .view()
                .data()
                .as_f32()
                .context("CTC log_probs tensor is not f32")?;
            let tokens = ctc::ctc_greedy_decode(
                log_probs,
                enc_len,
                self.tokenizer.vocab_size(),
                self.tokenizer.blank_id(),
            );
            let words = ctc::ctc_tokens_to_words(&self.tokenizer, &tokens, frame_offset);
            return Ok((words, false)); // CTC has no endpoint signal
        }

        // RNN-T greedy decode — the encoder output is borrowed for the decode loop.
        let dec_start = std::time::Instant::now();
        let result = decode::greedy_decode(
            triplet
                .decoder
                .as_deref()
                .expect("RNN-T decoder session must be loaded for a non-CTC head"),
            triplet
                .joiner
                .as_deref()
                .expect("RNN-T joiner session must be loaded for a non-CTC head"),
            &encoder_outputs[0].view(),
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

/// Per-request overrides for the recognition post-processing knobs, letting a
/// single loaded engine vary punctuation / ITN / VAD per file-transcription
/// call instead of only at boot. `None` on a field means "use the engine's
/// boot default", so a `TranscribeOverrides::default()` (all `None`) reproduces
/// the pre-feature behaviour byte-for-byte.
///
/// A knob can only be turned *on* per-request if the underlying resource is
/// loaded: `vad = Some(true)` requires a VAD to be attached, and
/// `punctuation = Some(true)` requires a punctuator. Call
/// [`Engine::validate_overrides`] before transcribing to reject impossible
/// requests (mapped to `409` on the REST surface); turning a knob *off*
/// (`Some(false)`) is always valid.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TranscribeOverrides {
    /// Override the punctuation / casing restoration pass. `Some(true)` forces
    /// it on (requires a punctuator), `Some(false)` skips it, `None` = engine
    /// default (on iff a punctuator is attached).
    pub punctuation: Option<bool>,
    /// Override inverse text normalization (number-words → digits).
    /// `Some(true)` / `Some(false)` force the state; `None` = engine default.
    /// ITN is pure code (no model), so `Some(true)` is always valid.
    pub itn: Option<bool>,
    /// Override VAD gating. `Some(true)` decodes only detected speech regions
    /// (requires a VAD to be attached), `Some(false)` decodes the whole buffer,
    /// `None` = engine default (VAD path iff a VAD is attached).
    pub vad: Option<bool>,
}

/// Why a [`TranscribeOverrides`] was rejected: a knob was turned on per-request
/// but the resource backing it isn't loaded. Carries a stable machine-readable
/// [`code`](OverrideError::code) so the REST layer can surface a `409` with a
/// consistent contract without re-deriving the string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverrideError {
    /// `vad = Some(true)` but no VAD is attached to the engine.
    VadNotLoaded,
    /// `punctuation = Some(true)` but no punctuator is attached to the engine.
    PunctuationNotAvailable,
}

impl OverrideError {
    /// Stable, machine-readable error code for the REST `409` payload.
    pub fn code(self) -> &'static str {
        match self {
            OverrideError::VadNotLoaded => "vad_not_loaded",
            OverrideError::PunctuationNotAvailable => "punctuation_not_available",
        }
    }

    /// Human-readable, non-sensitive message for the REST `409` payload.
    pub fn message(self) -> &'static str {
        match self {
            OverrideError::VadNotLoaded => {
                "VAD requested but not loaded; start the server with --vad"
            }
            OverrideError::PunctuationNotAvailable => {
                "punctuation requested but no punctuation model is loaded"
            }
        }
    }
}

impl std::fmt::Display for OverrideError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for OverrideError {}

/// Merge per-channel [`TranscribeResult`]s into a single chronologically ordered
/// result. Each channel is assigned a zero-based speaker label (`speaker_0`,
/// `speaker_1`, …). Words are sorted by `start`; equal timestamps are ordered by
/// channel index for stability.
pub fn merge_channel_results(per_channel: Vec<TranscribeResult>) -> TranscribeResult {
    let mut all_words = Vec::new();
    let mut duration_s = 0.0_f64;
    for (channel_idx, mut result) in per_channel.into_iter().enumerate() {
        let speaker = channel_idx as u32;
        for w in &mut result.words {
            w.speaker = Some(speaker);
        }
        duration_s = duration_s.max(result.duration_s);
        all_words.extend(result.words);
    }

    all_words.sort_by(|a, b| {
        a.start
            .total_cmp(&b.start)
            .then_with(|| a.speaker.cmp(&b.speaker))
    });

    let text = all_words
        .iter()
        .map(|w| w.word.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    TranscribeResult {
        text,
        words: all_words,
        duration_s,
    }
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
    fn test_transcribe_overrides_default_all_none() {
        // The default overrides must be all-`None` so a request with no knobs
        // reproduces the engine's boot behaviour byte-for-byte.
        let o = TranscribeOverrides::default();
        assert_eq!(o.punctuation, None);
        assert_eq!(o.itn, None);
        assert_eq!(o.vad, None);
    }

    #[test]
    fn test_override_error_codes_stable() {
        // Stable machine-readable codes surfaced as the REST 409 `code`.
        assert_eq!(OverrideError::VadNotLoaded.code(), "vad_not_loaded");
        assert_eq!(
            OverrideError::PunctuationNotAvailable.code(),
            "punctuation_not_available"
        );
        // Messages are non-empty and don't leak internals.
        assert!(!OverrideError::VadNotLoaded.message().is_empty());
        assert!(!OverrideError::PunctuationNotAvailable.message().is_empty());
        // Display matches message().
        assert_eq!(
            OverrideError::VadNotLoaded.to_string(),
            OverrideError::VadNotLoaded.message()
        );
    }

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

    fn sample_word(w: &str, start: f64, end: f64, speaker: Option<u32>) -> WordInfo {
        WordInfo::new(w, start, end, 0.9, speaker)
    }

    #[test]
    fn test_merge_channel_results_empty() {
        let merged = merge_channel_results(vec![
            TranscribeResult {
                text: String::new(),
                words: vec![],
                duration_s: 0.0,
            },
            TranscribeResult {
                text: String::new(),
                words: vec![],
                duration_s: 0.0,
            },
        ]);
        assert!(merged.words.is_empty());
        assert!(merged.text.is_empty());
    }

    #[test]
    fn test_merge_channel_results_interleaved_channels() {
        let ch0 = TranscribeResult {
            text: String::new(),
            words: vec![
                sample_word("привет", 0.0, 0.4, None),
                sample_word("как", 1.0, 1.3, None),
            ],
            duration_s: 1.5,
        };
        let ch1 = TranscribeResult {
            text: String::new(),
            words: vec![sample_word("да", 0.5, 0.8, None)],
            duration_s: 1.5,
        };
        let merged = merge_channel_results(vec![ch0, ch1]);
        assert_eq!(merged.words.len(), 3);
        assert_eq!(merged.words[0].word, "привет");
        assert_eq!(merged.words[0].speaker, Some(0));
        assert_eq!(merged.words[1].word, "да");
        assert_eq!(merged.words[1].speaker, Some(1));
        assert_eq!(merged.words[2].word, "как");
        assert_eq!(merged.words[2].speaker, Some(0));
    }

    #[test]
    fn test_merge_channel_results_tie_order_by_channel() {
        let ch0 = TranscribeResult {
            text: String::new(),
            words: vec![sample_word("а", 0.5, 0.7, None)],
            duration_s: 1.0,
        };
        let ch1 = TranscribeResult {
            text: String::new(),
            words: vec![sample_word("б", 0.5, 0.7, None)],
            duration_s: 1.0,
        };
        let merged = merge_channel_results(vec![ch0, ch1]);
        assert_eq!(merged.words[0].word, "а");
        assert_eq!(merged.words[0].speaker, Some(0));
        assert_eq!(merged.words[1].word, "б");
        assert_eq!(merged.words[1].speaker, Some(1));
    }

    #[test]
    fn test_merge_channel_results_no_channels() {
        let merged = merge_channel_results(vec![]);
        assert!(merged.words.is_empty());
        assert!(merged.text.is_empty());
        assert_eq!(merged.duration_s, 0.0);
    }

    #[test]
    fn test_merge_channel_results_max_duration() {
        let ch0 = TranscribeResult {
            text: String::new(),
            words: vec![sample_word("a", 0.0, 0.5, None)],
            duration_s: 5.0,
        };
        let ch1 = TranscribeResult {
            text: String::new(),
            words: vec![sample_word("b", 0.5, 1.0, None)],
            duration_s: 12.0,
        };
        let merged = merge_channel_results(vec![ch0, ch1]);
        assert_eq!(merged.duration_s, 12.0);
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

    #[test]
    fn test_resolve_load_variant_override_beats_disk_detection() {
        // A directory holding BOTH the rnnt and e2e_rnnt encoders: on-disk
        // detection returns rnnt (precedence), so without an explicit override the
        // engine would silently ignore `--model-variant e2e_rnnt`. This is the
        // regression this fix guards.
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join(ModelVariant::Rnnt.encoder_file()), b"x").unwrap();
        std::fs::write(dir.path().join(ModelVariant::E2eRnnt.encoder_file()), b"x").unwrap();

        // Sanity: bare on-disk detection prefers rnnt.
        assert_eq!(
            ModelVariant::detect_in_dir(dir.path()),
            Some(ModelVariant::Rnnt)
        );
        // No override → auto-detect (rnnt precedence): behavior is unchanged.
        assert_eq!(
            resolve_load_variant(None, dir.path()),
            Some(ModelVariant::Rnnt)
        );
        // Explicit override wins over the higher-precedence head on disk.
        assert_eq!(
            resolve_load_variant(Some(ModelVariant::E2eRnnt), dir.path()),
            Some(ModelVariant::E2eRnnt)
        );

        // The override is honored even when its files aren't present — the engine
        // load then fails with a clear ModelLoad error instead of silently loading
        // whatever else is on disk.
        let empty = tempfile::tempdir().expect("tempdir");
        assert_eq!(
            resolve_load_variant(Some(ModelVariant::E2eRnnt), empty.path()),
            Some(ModelVariant::E2eRnnt)
        );
        // No override + empty dir → nothing to load.
        assert_eq!(resolve_load_variant(None, empty.path()), None);
    }

    // ---- Pool tests (B.7) ---------------------------------------------------
    //
    // These exercise `Pool<T>` with synthetic items so the contract is
    // observable without loading ONNX models. `SessionPool = Pool<SessionTriplet>`
    // is just an alias, so any property proven here also holds for the real
    // pool.

    #[tokio::test]
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
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
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
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
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
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
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
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
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
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
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
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
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
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
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
    async fn test_pool_close_is_idempotent() {
        // `pool.close()` is wired into the shutdown hook; calling it twice
        // (e.g. shutdown signal + Drop) must not panic.
        let pool = Pool::<u32>::new(vec![]);
        pool.close();
        pool.close();
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
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
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
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
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
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
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
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
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
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

    // ---- More pure (no-model) coverage -------------------------------------

    #[test]
    fn test_finalize_pool_load_degraded_includes_error_detail() {
        // The degraded-pool branch logs the first error; exercise the
        // `first_err` formatting path (the loaded triplets are still returned).
        let r: Vec<anyhow::Result<u32>> = vec![Ok(1), Err(anyhow::anyhow!("first failure cause"))];
        assert_eq!(Engine::finalize_pool_load(r, 2, 1).unwrap(), vec![1]);
    }

    #[test]
    fn test_finalize_pool_load_below_min_no_errors_when_all_ok_but_short() {
        // No Err entries, but fewer results than pool_size with min above the
        // loaded count → still errors (loaded count is what matters).
        let r: Vec<anyhow::Result<u32>> = vec![Ok(1)];
        let err = Engine::finalize_pool_load(r, 3, 2).unwrap_err().to_string();
        assert!(err.contains("loaded only 1/3"), "got: {err}");
    }

    #[test]
    fn test_transcript_assembler_set_words_overwrites() {
        // The sliding-window streaming path overwrites (not appends) on each
        // re-decode via `set_words`. A second call replaces the first.
        let mut asm = TranscriptAssembler::new();
        asm.set_words(vec![word("alpha", 0.0, 0.4), word("beta", 0.5, 0.9)]);
        let p = asm.partial(0.0);
        assert_eq!(p.text, "alpha beta");
        assert_eq!(p.words.len(), 2);

        asm.set_words(vec![word("gamma", 1.0, 1.4)]);
        let p = asm.partial(0.0);
        assert_eq!(p.text, "gamma", "set_words must overwrite, not append");
        assert_eq!(p.words.len(), 1);
    }

    #[test]
    fn test_transcript_assembler_set_words_empty_resets_text() {
        let mut asm = TranscriptAssembler::new();
        asm.set_words(vec![word("x", 0.0, 0.4)]);
        assert!(!asm.is_empty());
        asm.set_words(vec![]);
        assert!(asm.is_empty(), "empty set_words clears the accumulation");
    }

    #[test]
    fn test_token_formatter_last_word_empty_confidences_defaults_to_one() {
        // A word whose only token is a bare boundary marker (`▁`, no body)
        // contributes no confidence sample; a following real word that itself
        // has no recorded confidences must default to 1.0 on the final-emit
        // path. We build a vocab whose tokens are pure boundary markers so the
        // `clean` body is empty and no confidence is pushed.
        let tok = Tokenizer::from_tokens(vec![
            "\u{2581}real".into(), // 0: a real word
            "\u{2581}".into(),     // 1: bare boundary, empty body
        ]);
        let tokens = vec![
            decode::TokenInfo {
                token_id: 0,
                frame_index: 0,
                confidence: 0.7,
            },
            // A bare boundary token forces emission of "real" (mid-loop emit),
            // then contributes nothing to a new word.
            decode::TokenInfo {
                token_id: 1,
                frame_index: 1,
                confidence: 0.5,
            },
        ];
        let words = TokenFormatter::tokens_to_words(&tok, &tokens, 0);
        // Only "real" is emitted; the trailing bare boundary leaves no word.
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].word, "real");
        assert!((words[0].confidence - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_feature_extractor_prepare_buffer_accumulates() {
        // `prepare_buffer` appends to the buffer and reports the usable sample
        // count once a full frame is available; below N_FFT it returns None.
        let fe = FeatureExtractor::new();
        let mut buf: Vec<f32> = Vec::new();
        // A handful of samples — fewer than N_FFT — yields no usable frame yet.
        let usable = fe.prepare_buffer(&[0.1; 10], &mut buf);
        assert_eq!(usable, None, "sub-frame input is buffered, not yet usable");
        assert_eq!(buf.len(), 10, "samples are retained in the buffer");

        // Append enough to cross a frame boundary; a usable count is reported.
        let usable = fe.prepare_buffer(&vec![0.2; N_FFT], &mut buf);
        assert!(
            usable.is_some(),
            "crossing a frame boundary yields a usable count"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore = "mel FFT over 1s of audio is too slow under Miri")]
    fn test_feature_extractor_compute_mel_reuses_buffers() {
        // `compute_mel` writes into caller-owned scratch buffers and returns the
        // frame count. One second of 16 kHz audio → ~100 frames; the output
        // buffer holds frames * N_MELS values.
        let fe = FeatureExtractor::new();
        let samples = vec![0.0f32; 16000];
        let mut fft_buf = Vec::new();
        let mut power_buf = Vec::new();
        let mut out_buf = Vec::new();
        let frames = fe.compute_mel(&samples, &mut fft_buf, &mut power_buf, &mut out_buf);
        assert!(frames > 0, "1s of audio yields at least one mel frame");
        assert_eq!(
            out_buf.len(),
            frames * N_MELS,
            "output buffer holds frames * N_MELS values"
        );
    }

    // ---- Model-backed coverage (process_chunk / transcribe / state) --------
    //
    // These need the GigaAM model on disk; CI / coverage runs them with
    // `--include-ignored`. They exercise the real streaming + file-decode
    // branches (empty input, sub-stride, sub-N_FFT, full decode + slide,
    // silence/short transcription, finish_stream / flush_state).

    #[test]
    #[ignore = "requires model"]
    fn test_create_state_initial_fields() {
        let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
            .expect("engine should load");
        let state = engine.create_state(false);
        assert!(state.audio_buffer.is_empty());
        assert!(state.assembler.is_empty());
        assert_eq!(state.window_start_samples, 0);
        assert_eq!(state.context_samples, 0);
        assert_eq!(state.pending_samples, 0);
        assert!(state.resampler.is_none());
        // No VAD attached on a default engine → no endpointer.
        assert!(state.vad_endpointer.is_none());
        // Decoder state seeded to blank.
        assert_eq!(state.decoder.consecutive_blanks, 0);
    }

    #[test]
    #[ignore = "requires model"]
    fn test_create_state_diarization_flag_ignored_without_feature() {
        // Without the `diarization` feature the flag is silently ignored and a
        // perfectly usable state still comes back.
        let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
            .expect("engine should load");
        let state = engine.create_state(true);
        assert!(state.audio_buffer.is_empty());
    }

    #[cfg(feature = "diarization")]
    #[test]
    #[ignore = "requires the WeSpeaker diarization model"]
    #[allow(deprecated)] // legacy EmbeddingExtractor — see import note above
    fn test_speaker_encoder_accepts_waveform_audio() {
        let model_path =
            Path::new(&crate::model::default_model_dir()).join("wespeaker_resnet34.onnx");
        let encoder = load_speaker_encoder(&model_path, 1).expect("speaker encoder should load");
        let samples: Vec<f32> = (0..24_000)
            .map(|i| {
                let phase = std::f32::consts::TAU * 220.0 * i as f32 / 16_000.0;
                0.1 * phase.sin()
            })
            .collect();

        let embedding = encoder
            .extract(&samples, &DiaConfig::default())
            .expect("waveform must be converted to rank-3 fbank features");

        assert_eq!(embedding.len(), SPEAKER_EMBEDDING_DIM);
        assert!(embedding.iter().all(|value| value.is_finite()));
    }

    #[test]
    #[ignore = "requires model"]
    fn test_process_chunk_empty_input_returns_no_segments() {
        let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
            .expect("engine should load");
        let mut guard = engine.pool.checkout_blocking().expect("checkout");
        let mut state = engine.create_state(false);
        let segs = engine
            .process_chunk(&[], &mut state, &mut guard)
            .expect("empty chunk must not error");
        assert!(segs.is_empty(), "empty input yields no segments");
        assert_eq!(state.audio_buffer.len(), 0);
    }

    #[test]
    #[ignore = "requires model"]
    fn test_process_chunk_sub_stride_buffers_without_decoding() {
        // A chunk smaller than the decode stride is buffered and triggers no
        // decode (the stride gate returns early).
        let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
            .expect("engine should load");
        let mut guard = engine.pool.checkout_blocking().expect("checkout");
        let mut state = engine.create_state(false);
        let small = vec![0.0f32; 1600]; // 0.1s ≪ 0.8s stride
        let segs = engine
            .process_chunk(&small, &mut state, &mut guard)
            .expect("sub-stride chunk must not error");
        assert!(segs.is_empty(), "sub-stride chunk yields no segments yet");
        assert_eq!(state.audio_buffer.len(), 1600, "samples are buffered");
        assert_eq!(state.pending_samples, 1600, "pending counter advances");
    }

    #[test]
    #[ignore = "requires model"]
    fn test_process_chunk_silence_over_stride_decodes_no_words() {
        // Enough silence to cross the decode stride: the encoder runs but
        // produces no words (silence), so the partial path returns no segments
        // and the pending counter resets.
        let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
            .expect("engine should load");
        let mut guard = engine.pool.checkout_blocking().expect("checkout");
        let mut state = engine.create_state(false);
        let chunk = vec![0.0f32; 16000]; // 1s of silence, > 0.8s stride
        let segs = engine
            .process_chunk(&chunk, &mut state, &mut guard)
            .expect("decode of silence must not error");
        // Silence → no words → empty assembler → no partial segment emitted.
        assert!(segs.is_empty(), "silence decodes to no words");
        assert_eq!(
            state.pending_samples, 0,
            "decode resets the pending counter"
        );
    }

    #[test]
    #[ignore = "requires model"]
    fn test_flush_state_empty_returns_none() {
        let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
            .expect("engine should load");
        let mut state = engine.create_state(false);
        assert!(
            engine.flush_state(&mut state).is_none(),
            "an empty assembler flushes to None"
        );
    }

    #[test]
    #[ignore = "requires model"]
    fn test_flush_state_nonempty_returns_final_segment() {
        let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
            .expect("engine should load");
        let mut state = engine.create_state(false);
        state.assembler.set_words(vec![word("hello", 0.0, 0.4)]);
        let seg = engine
            .flush_state(&mut state)
            .expect("non-empty assembler flushes to a Final segment");
        assert!(seg.is_final);
        assert_eq!(seg.text, "hello");
        assert!(
            engine.flush_state(&mut state).is_none(),
            "finalize resets the assembler"
        );
    }

    #[test]
    #[ignore = "requires model"]
    fn test_finish_stream_no_pending_flushes_assembler() {
        // No buffered audio and no pending samples: finish_stream skips the
        // forced decode and just flushes whatever the assembler holds.
        let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
            .expect("engine should load");
        let mut guard = engine.pool.checkout_blocking().expect("checkout");
        let mut state = engine.create_state(false);
        state.assembler.set_words(vec![word("trailing", 0.0, 0.4)]);
        let seg = engine
            .finish_stream(&mut state, &mut guard)
            .expect("finish_stream flushes the assembler");
        assert_eq!(seg.text, "trailing");
        assert!(seg.is_final);
    }

    #[test]
    #[ignore = "requires model"]
    fn test_finish_stream_empty_state_returns_none() {
        let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
            .expect("engine should load");
        let mut guard = engine.pool.checkout_blocking().expect("checkout");
        let mut state = engine.create_state(false);
        assert!(
            engine.finish_stream(&mut state, &mut guard).is_none(),
            "an idle stream finishes to None"
        );
    }

    #[test]
    #[ignore = "requires model"]
    fn test_transcribe_samples_silence_yields_empty_text() {
        // The single-pass file path on pure silence: the encoder runs, decode
        // produces no words, and the result text is empty with a correct
        // duration.
        let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
            .expect("engine should load");
        let mut guard = engine.pool.checkout_blocking().expect("checkout");
        let silence = vec![0.0f32; 16000 * 2]; // 2s
        let result = engine
            .transcribe_samples(&silence, &mut guard)
            .expect("silence transcription must not error");
        assert!(result.text.trim().is_empty(), "silence yields no text");
        assert!(result.words.is_empty());
        assert!((result.duration_s - 2.0).abs() < 1e-6);
    }

    #[test]
    #[ignore = "requires model"]
    fn test_transcribe_samples_short_sub_frame_audio() {
        // Audio shorter than a single FFT frame: the mel extractor pads to one
        // zero frame and the encoder still runs — exercising the short-input
        // single-pass branch without panicking. The decode output is whatever
        // the model emits on a lone padded frame; we only assert it doesn't
        // error and the reported duration matches the (tiny) input.
        let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
            .expect("engine should load");
        let mut guard = engine.pool.checkout_blocking().expect("checkout");
        let tiny = vec![0.0f32; 100]; // < N_FFT (320)
        let result = engine
            .transcribe_samples(&tiny, &mut guard)
            .expect("sub-frame audio must not error");
        assert!((result.duration_s - 100.0 / 16000.0).abs() < 1e-9);
    }

    #[test]
    #[ignore = "requires model"]
    fn test_transcribe_samples_below_chunk_threshold_single_pass() {
        // Just under the long-form chunk threshold (30s) takes the single-pass
        // path; pure silence still yields no words but must not error.
        let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
            .expect("engine should load");
        let mut guard = engine.pool.checkout_blocking().expect("checkout");
        let samples = vec![0.0f32; CHUNK_THRESHOLD_SAMPLES]; // exactly at threshold → single pass
        let result = engine
            .transcribe_samples(&samples, &mut guard)
            .expect("at-threshold audio must not error");
        assert!(result.words.is_empty(), "silence decodes to no words");
    }

    /// End-to-end proof that the Candle backend transcribes IDENTICALLY to the
    /// ort backend through the full engine pipeline (mel + encoder + RNN-T
    /// decode), not just per-stage tensors.
    ///
    /// Two engines are built on the SAME model dir — one forced onto the ort CPU
    /// backend, one forced onto the Candle backend (which reads the sibling
    /// `candle/*.safetensors`) — and the same fixture wav is transcribed by both.
    /// Per-stage parity is bit-exact, so the decoded text MUST be byte-identical.
    ///
    /// Run with:
    /// `cargo test -p gigastt-core --features candle --lib -- --ignored --nocapture candle_ort_transcription_parity`
    #[cfg(feature = "candle")]
    #[test]
    #[ignore = "requires v3_rnnt model + candle/*.safetensors"]
    fn candle_ort_transcription_parity() {
        let model_dir = crate::model::default_model_dir();
        let model_path = Path::new(&model_dir);

        let ort_engine = Engine::load_with_factory(
            model_path,
            None,
            1,
            1,
            0,
            Box::new(crate::runtime::ort::factory::OrtFactory::cpu()),
            1,
        )
        .expect("ort engine should load");
        let candle_engine = Engine::load_with_factory(
            model_path,
            None,
            1,
            1,
            0,
            Box::new(crate::runtime::candle::factory::CandleFactory::new()),
            1,
        )
        .expect("candle engine should load");

        let fixtures = [
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../gigastt/tests/fixtures/golos_00.wav"
            ),
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../gigastt/tests/fixtures/golos_01.wav"
            ),
        ];

        for fixture in fixtures {
            let mut ort_guard = ort_engine.pool.checkout_blocking().expect("ort checkout");
            let ort_text = ort_engine
                .transcribe_file(fixture, &mut ort_guard)
                .expect("ort transcription")
                .text;

            let mut candle_guard = candle_engine
                .pool
                .checkout_blocking()
                .expect("candle checkout");
            let candle_text = candle_engine
                .transcribe_file(fixture, &mut candle_guard)
                .expect("candle transcription")
                .text;

            eprintln!("fixture = {fixture}");
            eprintln!("ort    = {ort_text:?}");
            eprintln!("candle = {candle_text:?}");

            assert_eq!(
                ort_text, candle_text,
                "candle transcript diverges from ort for {fixture}:\n  ort    = {ort_text:?}\n  candle = {candle_text:?}"
            );
        }
    }

    /// End-to-end proof that the Candle backend produces IDENTICAL output to ort
    /// through the STREAMING path (sliding-window `process_chunk` + `finish_stream`),
    /// not just through the whole-file `transcribe_file` path.
    ///
    /// Both engines receive the SAME 8 000-sample (0.5 s) chunks fed in order;
    /// all returned segment texts are concatenated and compared byte-for-byte.
    /// Per-stage tensor parity is bit-exact, so the streamed transcripts must match.
    ///
    /// Run with:
    /// `cargo test -p gigastt-core --features candle --lib -- --ignored --nocapture candle_ort_streaming_parity`
    #[cfg(feature = "candle")]
    #[test]
    #[ignore = "requires v3_rnnt model + candle/*.safetensors"]
    fn candle_ort_streaming_parity() {
        const CHUNK_SAMPLES: usize = 8_000; // 0.5 s at 16 kHz

        let model_dir = crate::model::default_model_dir();
        let model_path = Path::new(&model_dir);

        let ort_engine = Engine::load_with_factory(
            model_path,
            None,
            1,
            1,
            0,
            Box::new(crate::runtime::ort::factory::OrtFactory::cpu()),
            1,
        )
        .expect("ort engine should load");
        let candle_engine = Engine::load_with_factory(
            model_path,
            None,
            1,
            1,
            0,
            Box::new(crate::runtime::candle::factory::CandleFactory::new()),
            1,
        )
        .expect("candle engine should load");

        let fixtures = [
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../gigastt/tests/fixtures/golos_00.wav"
            ),
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../gigastt/tests/fixtures/golos_01.wav"
            ),
        ];

        for fixture in fixtures {
            let samples = audio::decode_audio_file(fixture)
                .unwrap_or_else(|e| panic!("failed to decode {fixture}: {e:#}"));

            // --- ort streaming ---
            let mut ort_guard = ort_engine.pool.checkout_blocking().expect("ort checkout");
            let mut ort_state = ort_engine.create_state(false);
            let mut ort_text = String::new();
            for chunk in samples.chunks(CHUNK_SAMPLES) {
                let segs = ort_engine
                    .process_chunk(chunk, &mut ort_state, &mut ort_guard)
                    .expect("ort process_chunk must not error");
                for seg in segs {
                    if !ort_text.is_empty() {
                        ort_text.push(' ');
                    }
                    ort_text.push_str(seg.text.trim());
                }
            }
            if let Some(seg) = ort_engine.finish_stream(&mut ort_state, &mut ort_guard) {
                if !ort_text.is_empty() {
                    ort_text.push(' ');
                }
                ort_text.push_str(seg.text.trim());
            }

            // --- candle streaming ---
            let mut candle_guard = candle_engine
                .pool
                .checkout_blocking()
                .expect("candle checkout");
            let mut candle_state = candle_engine.create_state(false);
            let mut candle_text = String::new();
            for chunk in samples.chunks(CHUNK_SAMPLES) {
                let segs = candle_engine
                    .process_chunk(chunk, &mut candle_state, &mut candle_guard)
                    .expect("candle process_chunk must not error");
                for seg in segs {
                    if !candle_text.is_empty() {
                        candle_text.push(' ');
                    }
                    candle_text.push_str(seg.text.trim());
                }
            }
            if let Some(seg) = candle_engine.finish_stream(&mut candle_state, &mut candle_guard) {
                if !candle_text.is_empty() {
                    candle_text.push(' ');
                }
                candle_text.push_str(seg.text.trim());
            }

            eprintln!("fixture = {fixture}");
            eprintln!("ort    (streamed) = {ort_text:?}");
            eprintln!("candle (streamed) = {candle_text:?}");

            assert_eq!(
                ort_text, candle_text,
                "candle streamed transcript diverges from ort for {fixture}:\n  ort    = {ort_text:?}\n  candle = {candle_text:?}"
            );
        }
    }

    /// Word-level edit distance / WER between a reference and a hypothesis
    /// transcript (Levenshtein over whitespace tokens, normalized by reference
    /// word count). Used by the ANE measurement harness below.
    #[cfg(all(feature = "ane", target_os = "macos"))]
    fn word_error_rate(reference: &str, hypothesis: &str) -> f64 {
        let r: Vec<&str> = reference.split_whitespace().collect();
        let h: Vec<&str> = hypothesis.split_whitespace().collect();
        if r.is_empty() {
            return if h.is_empty() { 0.0 } else { 1.0 };
        }
        let mut prev: Vec<usize> = (0..=h.len()).collect();
        let mut cur = vec![0usize; h.len() + 1];
        for (i, rw) in r.iter().enumerate() {
            cur[0] = i + 1;
            for (j, hw) in h.iter().enumerate() {
                let cost = if rw == hw { 0 } else { 1 };
                cur[j + 1] = (prev[j] + cost).min(prev[j + 1] + 1).min(cur[j] + 1);
            }
            std::mem::swap(&mut prev, &mut cur);
        }
        prev[h.len()] as f64 / r.len() as f64
    }

    /// The 15 Golos fixtures with ground-truth references (from
    /// `crates/gigastt/tests/fixtures/manifest.json`). `(path, reference)`.
    #[cfg(all(feature = "ane", target_os = "macos"))]
    fn golos_fixtures() -> Vec<(String, &'static str)> {
        const REFS: &[(&str, &str)] = &[
            (
                "golos_00.wav",
                "шестьдесят тысяч тенге сколько будет стоить",
            ),
            (
                "golos_01.wav",
                "покажи мне на смотрешке телеканал синергия тв",
            ),
            ("golos_02.wav", "заказать яблоки зеленые"),
            (
                "golos_03.wav",
                "алиса закажи килограммовый торт графские развалины",
            ),
            ("golos_04.wav", "ищи телеканал про бизнес на тиви"),
            ("golos_05.wav", "михаила мурадяна"),
            (
                "golos_06.wav",
                "любовницы две тысячи тринадцать пятнадцатый сезон",
            ),
            ("golos_07.wav", "найди боевики"),
            ("golos_08.wav", "гетто сезон три"),
            ("golos_09.wav", "хочу посмотреть ростов папа на телевизоре"),
            ("golos_10.wav", "сбер какое твое самое ненавистное занятие"),
            ("golos_11.wav", "афина чем платят у китайцев"),
            (
                "golos_12.wav",
                "джой как работает досрочное погашение кредита",
            ),
            ("golos_13.wav", "у тебя найдется люк кейдж"),
            ("golos_14.wav", "у тебя будет лучшая часть пинк"),
        ];
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../gigastt/tests/fixtures/");
        REFS.iter()
            .map(|(f, r)| (format!("{dir}{f}"), *r))
            .collect()
    }

    /// Build the two engines (ort baseline + composite ANE) on the rnnt model in
    /// the default model dir. Returns `None` (with a printed SKIP) when the
    /// rnnt model or the bucket-768 `.mlpackage` is absent so the `#[ignore]`d
    /// measurement tests degrade cleanly on machines without the assets.
    #[cfg(all(feature = "ane", target_os = "macos"))]
    fn ane_measurement_engines() -> Option<(Engine, Engine)> {
        let model_dir = crate::model::default_model_dir();
        let model_path = Path::new(&model_dir);

        let ane_dir = model_path.join("ane");
        let bucket_768 = ane_dir.join(crate::model::ane_package_dir_name(768));
        if !crate::model::ane_package_complete(&bucket_768) {
            eprintln!(
                "SKIP: ANE bucket-768 package missing in {} (run `gigastt download --ane` or convert locally)",
                ane_dir.display()
            );
            return None;
        }
        if ModelVariant::detect_in_dir(model_path).is_none() {
            eprintln!("SKIP: no model files in {model_dir} (run `gigastt download`)");
            return None;
        }

        let ort_engine = Engine::load_with_factory(
            model_path,
            None,
            1,
            1,
            0,
            Box::new(crate::runtime::ort::factory::OrtFactory::cpu()),
            1,
        )
        .expect("ort engine should load");
        let ane_engine = Engine::load_with_factory(
            model_path,
            None,
            1,
            1,
            0,
            Box::new(crate::runtime::coreml::factory::AneFactory::new()),
            1,
        )
        .expect("ANE engine should load");
        Some((ort_engine, ane_engine))
    }

    /// Run one encoder pass directly through a checked-out triplet (mirrors
    /// [`Engine::run_inference`]'s encoder setup) and return the emitted
    /// `encoded_len`. Used to compare the ANE and ort encoders' length tensors
    /// for the SAME mel input, pinning [`calc_output_length`] against ONNX drift.
    #[cfg(all(feature = "ane", target_os = "macos"))]
    fn encoder_emitted_len(engine: &Engine, features: &[f32], num_frames: usize) -> usize {
        let mut guard = engine.pool.checkout_blocking().expect("checkout");
        let triplet = &mut *guard;
        triplet.encoder_inputs[0].resize_to(Shape::new(vec![1, N_MELS, num_frames]));
        triplet.encoder_inputs[0]
            .as_f32_mut()
            .expect("encoder signal tensor is f32")
            .copy_from_slice(features);
        triplet.encoder_inputs[1]
            .as_i64_mut()
            .expect("encoder length tensor is i64")[0] = num_frames as i64;
        let outputs = triplet
            .encoder
            .run(&triplet.encoder_inputs)
            .expect("encoder run");
        match outputs[1].view().data() {
            TensorDataView::I32(v) => usize::try_from(v[0]).expect("non-negative len"),
            TensorDataView::I64(v) => usize::try_from(v[0]).expect("non-negative len"),
            _ => panic!("unexpected encoder length tensor type"),
        }
    }

    /// FULL-GOLOS WER + frame-count-equality measurement (Part 2a + frame pin).
    ///
    /// For every Golos fixture, transcribes through BOTH the composite ANE engine
    /// and the pure-ort baseline, records mel length `T`, the bucket fill % and
    /// which PATH the ANE encoder took (ANE bucket vs ort fallback), and emits a
    /// per-clip table plus aggregate WER(ANE vs ort), WER(ANE vs truth) and
    /// WER(ort vs truth). Additionally asserts the ANE and ort encoders emit the
    /// SAME `encoded_len` for the same mel input across all fixtures' real `T`
    /// (pins [`calc_output_length`] == the ort ONNX length op against drift).
    ///
    /// ANE parity is SOFT (mask-free FP16 pad-up is not byte-exact: cosine >= 0.94
    /// at >= 50% fill, a borderline token can flip), so the per-clip ANE-vs-ort
    /// gate is a small WER threshold rather than byte equality.
    ///
    /// Run with:
    /// `cargo test -p gigastt-core --features ane --lib -- --ignored --nocapture ane_ort_transcription_parity`
    #[cfg(all(feature = "ane", target_os = "macos"))]
    #[test]
    #[ignore = "requires v3_rnnt model + ~/.gigastt/models/ane/*.mlpackage + ANE hardware"]
    fn ane_ort_transcription_parity() {
        let Some((ort_engine, ane_engine)) = ane_measurement_engines() else {
            return;
        };

        // Mirror the encoder-session selection policy (select_bucket over the
        // shipped ladder) so the table can label the bucket the ANE engine took
        // without instrumenting the session itself.
        use crate::model::ANE_BUCKETS;
        use crate::runtime::coreml::encoder_session::select_bucket;
        const FILL_FLOOR: f64 = 0.5;
        // Aggregate gate: the mean WER(ANE vs ort) across all 15 clips must stay
        // small. Per-clip gate: at most ONE word may differ from ort — the
        // documented FP16-pad-up borderline-token flip (see FILL_FLOOR) is allowed
        // on a single word, but a multi-word divergence is a real regression.
        const MAX_MEAN_WER: f64 = 0.05;
        const MAX_WORD_DIFF_PER_CLIP: usize = 1;

        let fixtures = golos_fixtures();
        let mut sum_wer_ane_ort = 0.0;
        let mut sum_wer_ane_truth = 0.0;
        let mut sum_wer_ort_truth = 0.0;
        let mut frame_eq_checked: Vec<usize> = Vec::new();
        // (clip, word-diff vs ort) for the post-table per-clip gate.
        let mut clip_word_diffs: Vec<(String, usize)> = Vec::new();

        eprintln!(
            "\n{:<12} {:>5} {:>6} {:>9} {:>6} texts",
            "clip", "T", "fill%", "path", "ident"
        );
        for (path, reference) in &fixtures {
            let samples = audio::decode_audio_file(path).expect("decode fixture");
            let (features, num_frames) = ane_engine.features.compute(&samples);
            let bucket = select_bucket(num_frames, ANE_BUCKETS, FILL_FLOOR);
            let on_ane = bucket.is_some();
            let fill = match bucket {
                Some(n) => num_frames as f64 / n as f64,
                None => 0.0,
            };
            let path_label = match bucket {
                Some(n) => format!("ANE-{n}"),
                None => "ort-fb".to_string(),
            };

            // Frame-count equality: the ANE and ort encoders must emit the SAME
            // encoded_len for the same mel input. (On the ort-fallback path this
            // is trivially the same session class, but we still assert it; on the
            // ANE path it pins calc_output_length against the ONNX length op.)
            let ane_len = encoder_emitted_len(&ane_engine, &features, num_frames);
            let ort_len = encoder_emitted_len(&ort_engine, &features, num_frames);
            assert_eq!(
                ane_len, ort_len,
                "encoded_len mismatch ANE={ane_len} ort={ort_len} for T={num_frames} ({path})"
            );
            if on_ane {
                let formula =
                    crate::runtime::coreml::encoder_session::calc_output_length(num_frames);
                assert_eq!(
                    formula, ort_len,
                    "calc_output_length({num_frames})={formula} != ort encoder emitted {ort_len}"
                );
                frame_eq_checked.push(num_frames);
            }

            let mut ort_guard = ort_engine.pool.checkout_blocking().expect("ort checkout");
            let ort_text = ort_engine
                .transcribe_file(path, &mut ort_guard)
                .expect("ort transcription")
                .text;
            drop(ort_guard);

            let mut ane_guard = ane_engine.pool.checkout_blocking().expect("ANE checkout");
            let ane_text = ane_engine
                .transcribe_file(path, &mut ane_guard)
                .expect("ANE transcription")
                .text;
            drop(ane_guard);

            let wer_ane_ort = word_error_rate(&ort_text, &ane_text);
            let wer_ane_truth = word_error_rate(reference, &ane_text);
            let wer_ort_truth = word_error_rate(reference, &ort_text);
            sum_wer_ane_ort += wer_ane_ort;
            sum_wer_ane_truth += wer_ane_truth;
            sum_wer_ort_truth += wer_ort_truth;

            // Absolute word-edit distance vs ort (WER * ort word count, rounded).
            let ort_words = ort_text.split_whitespace().count().max(1);
            let word_diff = (wer_ane_ort * ort_words as f64).round() as usize;

            let clip = std::path::Path::new(path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(path);
            clip_word_diffs.push((clip.to_string(), word_diff));
            eprintln!(
                "{:<12} {:>5} {:>5.1}% {:>9} {:>6} ort={:?}",
                clip,
                num_frames,
                fill * 100.0,
                path_label,
                ort_text == ane_text,
                ort_text
            );
            eprintln!(
                "{:<12} {:>5} {:>6} {:>9} {:>6} ane={:?}  truth={:?}",
                "", "", "", "", "", ane_text, reference
            );
            eprintln!(
                "{:<12} WER ANE-vs-ort={wer_ane_ort:.4} (word_diff={word_diff}) ANE-vs-truth={wer_ane_truth:.4} ort-vs-truth={wer_ort_truth:.4}",
                ""
            );
        }

        let n = fixtures.len() as f64;
        let mean_ane_ort = sum_wer_ane_ort / n;
        eprintln!("\n=== AGGREGATE (n={}) ===", fixtures.len());
        eprintln!("mean WER(ANE vs ort)   = {mean_ane_ort:.4}");
        eprintln!("mean WER(ANE vs truth) = {:.4}", sum_wer_ane_truth / n);
        eprintln!("mean WER(ort vs truth) = {:.4}", sum_wer_ort_truth / n);
        eprintln!(
            "frame-count equality (ANE==ort encoded_len & ==calc_output_length) verified for T = {:?}",
            frame_eq_checked
        );

        // Gate AFTER measuring every clip (a measurement harness must not abort
        // mid-run). Aggregate parity must be tight; per clip, at most one word may
        // differ from ort (the documented single FP16-pad-up borderline flip).
        assert!(
            mean_ane_ort <= MAX_MEAN_WER,
            "mean WER(ANE vs ort) {mean_ane_ort:.4} > {MAX_MEAN_WER}"
        );
        for (clip, diff) in &clip_word_diffs {
            assert!(
                *diff <= MAX_WORD_DIFF_PER_CLIP,
                "clip {clip}: {diff} words differ from ort (> {MAX_WORD_DIFF_PER_CLIP}) — multi-word ANE divergence is a regression, not a borderline flip"
            );
        }
    }

    /// END-TO-END RTFx measurement (Part 2b).
    ///
    /// For the fixtures that take the ANE path (>= 384 mel frames, >= 50% fill),
    /// measures FULL-PIPELINE wall time (audio decode -> mel -> encoder -> RNN-T
    /// greedy decode -> text) through the ANE engine and the ort baseline, warm
    /// (first run discarded), median of >= 5. Reports RTFx (audio_secs / median_s)
    /// for each engine and the speedup ratio ANE/ort. This quantifies how little
    /// of the encoder-only ~230x ANE speedup survives the CPU-bound RNN-T decode
    /// loop: the encoder is nearly free on the ANE, but end-to-end the pipeline is
    /// decode-bound, so the realized full-pipeline speedup is only ~3.7x.
    ///
    /// Run with:
    /// `cargo test -p gigastt-core --features ane --lib -- --ignored --nocapture ane_e2e_rtfx`
    #[cfg(all(feature = "ane", target_os = "macos"))]
    #[test]
    #[ignore = "requires v3_rnnt model + ~/.gigastt/models/ane/*.mlpackage + ANE hardware"]
    fn ane_e2e_rtfx() {
        let Some((ort_engine, ane_engine)) = ane_measurement_engines() else {
            return;
        };

        use crate::model::ANE_BUCKETS;
        use crate::runtime::coreml::encoder_session::select_bucket;
        const FILL_FLOOR: f64 = 0.5;
        const WARM: usize = 1;
        const TIMED: usize = 6;

        fn median_secs(engine: &Engine, path: &str) -> f64 {
            // Warmup (discarded) + timed full-pipeline runs.
            for _ in 0..WARM {
                let mut g = engine.pool.checkout_blocking().expect("checkout");
                let _ = engine.transcribe_file(path, &mut g).expect("transcribe");
            }
            let mut times: Vec<f64> = Vec::with_capacity(TIMED);
            for _ in 0..TIMED {
                let mut g = engine.pool.checkout_blocking().expect("checkout");
                let t = std::time::Instant::now();
                let _ = engine.transcribe_file(path, &mut g).expect("transcribe");
                times.push(t.elapsed().as_secs_f64());
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            times[times.len() / 2]
        }

        eprintln!(
            "\n{:<12} {:>5} {:>6} {:>6} {:>9} {:>9} {:>9} {:>9} {:>8}",
            "clip",
            "T",
            "bucket",
            "audio_s",
            "ort_med_s",
            "ane_med_s",
            "ort_RTFx",
            "ane_RTFx",
            "speedup"
        );
        let mut any_ane = false;
        for (path, _ref) in golos_fixtures() {
            let samples = audio::decode_audio_file(&path).expect("decode fixture");
            let (_features, num_frames) = ane_engine.features.compute(&samples);
            let Some(bucket) = select_bucket(num_frames, ANE_BUCKETS, FILL_FLOOR) else {
                continue; // only clips that exercise the ANE encoder path
            };
            any_ane = true;
            let audio_s = samples.len() as f64 / 16000.0;
            let ort_med = median_secs(&ort_engine, &path);
            let ane_med = median_secs(&ane_engine, &path);
            let ort_rtfx = audio_s / ort_med;
            let ane_rtfx = audio_s / ane_med;
            let speedup = ort_med / ane_med;
            let clip = std::path::Path::new(&path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(&path)
                .to_string();
            eprintln!(
                "{:<12} {:>5} {:>6} {:>6.2} {:>9.4} {:>9.4} {:>9.1} {:>9.1} {:>7.2}x",
                clip, num_frames, bucket, audio_s, ort_med, ane_med, ort_rtfx, ane_rtfx, speedup
            );
        }
        assert!(
            any_ane,
            "no Golos fixture took the ANE path (>= 256 mel frames at >= 50% fill); cannot measure e2e RTFx"
        );
    }

    /// CONCURRENT-PREDICTION test (Part 1 item 2).
    ///
    /// Builds ONE `AneEncoderSession` backed by a single shared `Arc<SharedModel>`
    /// and fires concurrent `run` calls from N >= 4 threads on the SAME model,
    /// asserting no crash/panic and that every thread gets the SAME output for the
    /// same input (deterministic). Exercises the `unsafe impl Send/Sync` under
    /// real `CPUAndNeuralEngine` multi-thread load.
    ///
    /// Run with:
    /// `cargo test -p gigastt-core --features ane --lib -- --ignored --nocapture ane_concurrent_prediction_deterministic`
    #[cfg(all(feature = "ane", target_os = "macos"))]
    #[test]
    #[ignore = "requires ~/.gigastt/models/ane/gigaam_v3_encoder_768.mlpackage + ANE hardware"]
    fn ane_concurrent_prediction_deterministic() {
        use crate::runtime::coreml::bridge;
        use crate::runtime::coreml::encoder_session::{SharedModel, pad_time};
        use std::sync::Arc;

        let model_dir = crate::model::default_model_dir();
        let pkg = Path::new(&model_dir)
            .join("ane")
            .join(crate::model::ane_package_dir_name(768));
        if !crate::model::ane_package_complete(&pkg) {
            eprintln!("SKIP: ANE bucket-768 package missing at {}", pkg.display());
            return;
        }

        // Compile + load ONCE; share across threads (the production sharing model).
        let model = Arc::new(SharedModel(
            bridge::compile_and_load(&pkg, true).expect("compile_and_load bucket-768"),
        ));

        // Deterministic-but-non-trivial mel input padded to the 768 window.
        const T: usize = 600;
        const N: usize = 768;
        let mut mel = vec![0.0f32; N_MELS * T];
        for (i, v) in mel.iter_mut().enumerate() {
            *v = ((i % 97) as f32 * 0.013 - 0.5).sin();
        }
        let padded: Arc<Vec<f32>> = Arc::new(pad_time(&mel, N_MELS, T, N));

        // Single-threaded reference output.
        let (reference, ref_shape) =
            bridge::predict_f32(&model.0, "mel", &padded, &[1, N_MELS, N], "encoded")
                .expect("reference predict");

        const THREADS: usize = 4;
        const PER_THREAD: usize = 5;
        let mut handles = Vec::with_capacity(THREADS);
        for _ in 0..THREADS {
            let model = Arc::clone(&model);
            let padded = Arc::clone(&padded);
            handles.push(std::thread::spawn(move || {
                let mut outs = Vec::with_capacity(PER_THREAD);
                for _ in 0..PER_THREAD {
                    let (out, shape) =
                        bridge::predict_f32(&model.0, "mel", &padded, &[1, N_MELS, N], "encoded")
                            .expect("concurrent predict");
                    outs.push((out, shape));
                }
                outs
            }));
        }

        let mut total = 0usize;
        for h in handles {
            let outs = h.join().expect("thread did not panic");
            for (out, shape) in outs {
                assert_eq!(shape, ref_shape, "concurrent output shape diverged");
                assert_eq!(
                    out.len(),
                    reference.len(),
                    "concurrent output length diverged"
                );
                assert!(
                    out.iter().all(|v| v.is_finite()),
                    "concurrent output has non-finite values"
                );
                // Bit-for-bit determinism: same model + same input -> same output.
                assert_eq!(
                    out, reference,
                    "concurrent prediction diverged from the single-threaded reference"
                );
                total += 1;
            }
        }
        eprintln!(
            "concurrent OK: {THREADS} threads x {PER_THREAD} predicts = {total} runs, all deterministic & finite"
        );
    }

    mod mock_runtime_tests {
        use std::collections::HashMap;
        use std::sync::Arc;

        use crate::inference::{Engine, PRED_HIDDEN};
        use crate::runtime::mock::{MockFactory, MockSession};
        use crate::runtime::tensor::{Shape, Tensor, TensorData};

        const ENC_DIM: usize = 768;

        fn tiny_mock_engine() -> (Engine, tempfile::TempDir) {
            let tmp = tempfile::tempdir().expect("tempdir");
            let dir = tmp.path();

            // Empty model files are enough for variant detection; the mock
            // runtime intercepts all session loading before the filesystem is
            // read for ONNX data.
            std::fs::write(dir.join("v3_rnnt_encoder.onnx"), b"").unwrap();
            std::fs::write(dir.join("v3_rnnt_decoder.onnx"), b"").unwrap();
            std::fs::write(dir.join("v3_rnnt_joint.onnx"), b"").unwrap();
            // vocab: index 0 = "▁hi", index 1 = "<blk>" (blank wins on ties).
            std::fs::write(dir.join("v3_vocab.txt"), "\u{2581}hi\n<blk>\n").unwrap();

            let mut sessions: HashMap<String, Arc<MockSession>> = HashMap::new();
            sessions.insert(
                "v3_rnnt_encoder".into(),
                Arc::new(MockSession::new(
                    vec![Shape::new(vec![1, 64, 1]), Shape::new(vec![1])],
                    vec![
                        Tensor::new(
                            Shape::new(vec![1, ENC_DIM, 1]),
                            TensorData::F32(vec![0.0; ENC_DIM]),
                        )
                        .unwrap(),
                        Tensor::new(Shape::new(vec![1]), TensorData::I64(vec![1])).unwrap(),
                    ],
                )),
            );
            sessions.insert(
                "v3_rnnt_decoder".into(),
                Arc::new(MockSession::new(
                    vec![
                        Shape::new(vec![1, 1]),
                        Shape::new(vec![1, 1, PRED_HIDDEN]),
                        Shape::new(vec![1, 1, PRED_HIDDEN]),
                    ],
                    vec![
                        Tensor::new(
                            Shape::new(vec![1, 1, PRED_HIDDEN]),
                            TensorData::F32(vec![0.0; PRED_HIDDEN]),
                        )
                        .unwrap(),
                        Tensor::new(
                            Shape::new(vec![1, 1, PRED_HIDDEN]),
                            TensorData::F32(vec![0.0; PRED_HIDDEN]),
                        )
                        .unwrap(),
                        Tensor::new(
                            Shape::new(vec![1, 1, PRED_HIDDEN]),
                            TensorData::F32(vec![0.0; PRED_HIDDEN]),
                        )
                        .unwrap(),
                    ],
                )),
            );
            sessions.insert(
                "v3_rnnt_joint".into(),
                Arc::new(MockSession::new(
                    vec![
                        Shape::new(vec![1, ENC_DIM, 1]),
                        Shape::new(vec![1, PRED_HIDDEN, 1]),
                    ],
                    vec![
                        Tensor::new(Shape::new(vec![1, 1, 2]), TensorData::F32(vec![0.0; 2]))
                            .unwrap(),
                    ],
                )),
            );

            let factory = Box::new(MockFactory::new(sessions));
            let engine = Engine::load_with_factory(dir, None, 1, 1, 0, factory, 1)
                .expect("engine should load with mock runtime");
            (engine, tmp)
        }

        #[test]
        fn test_engine_loads_with_mock_runtime() {
            let _ = tiny_mock_engine();
        }

        #[test]
        fn test_engine_mock_runtime_decodes_silence() {
            let (engine, _tmp) = tiny_mock_engine();
            let mut guard = engine.pool.checkout_blocking().expect("checkout");
            let samples = vec![0.0f32; 100]; // < N_FFT → one padded mel frame
            let result = engine
                .transcribe_samples(&samples, &mut guard)
                .expect("mock decode must not error");

            assert!(result.text.is_empty(), "blank-only decode yields no text");
            assert!(result.words.is_empty());
            assert!((result.duration_s - 100.0 / 16000.0).abs() < 1e-9);
        }

        #[test]
        fn test_validate_overrides_truth_table() {
            use crate::inference::{OverrideError, TranscribeOverrides};

            // The tiny mock engine loads with no VAD and no punctuator attached,
            // so any knob turned *on* per-request must be rejected, and any knob
            // turned *off* (or ITN in either direction) must be accepted.
            let (engine, _tmp) = tiny_mock_engine();
            assert!(!engine.has_vad(), "mock engine has no VAD");
            assert!(!engine.has_punctuator(), "mock engine has no punctuator");

            // All absent → OK (byte-unchanged default path).
            assert_eq!(
                engine.validate_overrides(&TranscribeOverrides::default()),
                Ok(())
            );

            // vad=Some(true) with no VAD → err vad_not_loaded.
            assert_eq!(
                engine.validate_overrides(&TranscribeOverrides {
                    vad: Some(true),
                    ..Default::default()
                }),
                Err(OverrideError::VadNotLoaded)
            );
            // vad=Some(false) is always OK (opting out never needs a resource).
            assert_eq!(
                engine.validate_overrides(&TranscribeOverrides {
                    vad: Some(false),
                    ..Default::default()
                }),
                Ok(())
            );

            // punctuation=Some(true) with no punctuator → err.
            assert_eq!(
                engine.validate_overrides(&TranscribeOverrides {
                    punctuation: Some(true),
                    ..Default::default()
                }),
                Err(OverrideError::PunctuationNotAvailable)
            );
            // punctuation=Some(false) is always OK.
            assert_eq!(
                engine.validate_overrides(&TranscribeOverrides {
                    punctuation: Some(false),
                    ..Default::default()
                }),
                Ok(())
            );

            // itn=Some(true) is always OK (pure code, no model to load).
            assert_eq!(
                engine.validate_overrides(&TranscribeOverrides {
                    itn: Some(true),
                    ..Default::default()
                }),
                Ok(())
            );
            assert_eq!(
                engine.validate_overrides(&TranscribeOverrides {
                    itn: Some(false),
                    ..Default::default()
                }),
                Ok(())
            );
        }

        #[test]
        fn test_transcribe_samples_with_overrides_vad_off_matches_default() {
            use crate::inference::TranscribeOverrides;

            // `?vad=false` on a VAD-less engine is a no-op relative to the default
            // path (both decode the whole buffer), so the output must be identical.
            let (engine, _tmp) = tiny_mock_engine();
            let mut guard = engine.pool.checkout_blocking().expect("checkout");
            let samples = vec![0.0f32; 100];
            let baseline = engine
                .transcribe_samples(&samples, &mut guard)
                .expect("baseline decode");
            let with_vad_off = engine
                .transcribe_samples_with_overrides(
                    &samples,
                    &mut guard,
                    &TranscribeOverrides {
                        vad: Some(false),
                        ..Default::default()
                    },
                )
                .expect("vad-off decode");
            assert_eq!(baseline.text, with_vad_off.text);
            assert_eq!(baseline.words.len(), with_vad_off.words.len());
        }
    }
}
