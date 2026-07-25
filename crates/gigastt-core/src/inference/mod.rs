//! ONNX Runtime inference engine for GigaAM v3 (rnnt head by default, e2e_rnnt optional).
//!
//! Loads encoder, decoder, and joiner ONNX models and runs the RNN-T streaming decode loop.

pub mod audio;
mod bias;
mod ctc;
mod decode;
mod engine;
mod features;
mod pool;
mod state;
mod types;

#[cfg(not(feature = "__internals"))]
mod tokenizer;
/// Tokenizer module, exposed for fuzzing/benchmarking under the private
/// `__internals` feature only. Not part of the stable public API.
#[cfg(feature = "__internals")]
pub mod tokenizer;

#[cfg(all(feature = "coreml", feature = "cuda"))]
compile_error!("Features `coreml` and `cuda` are mutually exclusive. Choose one.");

// ---------------------------------------------------------------------------
// Public re-exports (stable paths: `gigastt_core::inference::Engine`, etc.)
// ---------------------------------------------------------------------------

pub use engine::Engine;
pub use pool::{OwnedReservation, Pool, PoolError, PoolGuard, SessionPool, SessionTriplet};
#[cfg(feature = "diarization")]
pub use state::SharedExtractor;
pub use state::{
    DecoderState, EndpointMode, EndpointReason, FeatureExtractor, StreamingState,
    TranscriptAssembler, TranscriptSegment, WordInfo,
};
pub use types::{
    DEFAULT_HOTWORDS_BOOST, HotwordError, HotwordOverride, MAX_HOTWORD_PHRASE_CHARS,
    MAX_HOTWORDS_PER_REQUEST, OverrideError, TranscribeOverrides, TranscribeResult,
    merge_channel_results,
};

/// Number of mel frequency bins used for spectrogram features.
pub const N_MELS: usize = 64;
/// FFT window size in samples (320 samples = 20ms at 16kHz).
pub const N_FFT: usize = 320;
/// Hop length between consecutive FFT frames in samples (160 samples = 10ms at 16kHz).
pub const HOP_LENGTH: usize = 160;
/// Hidden dimension of the RNN-T prediction (decoder) network.
pub const PRED_HIDDEN: usize = 320;

/// Encoder time subsampling factor (4 frames → 1 encoder output frame).
const ENCODER_SUBSAMPLING: usize = 4;
/// Seconds per encoder frame (HOP_LENGTH * ENCODER_SUBSAMPLING / 16000 = 0.04s).
const SECONDS_PER_FRAME: f64 = (HOP_LENGTH as f64 * ENCODER_SUBSAMPLING as f64) / 16000.0;

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
