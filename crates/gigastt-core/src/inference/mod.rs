//! ONNX Runtime inference engine for GigaAM v3 (rnnt head by default, e2e_rnnt optional).
//!
//! Loads encoder, decoder, and joiner ONNX models and runs the RNN-T streaming decode loop.

pub mod audio;
mod bias;
mod ctc;
mod decode;
/// Speaker diarization (polyvoice). Feature-gated; see module docs for why the
/// deprecated polyvoice embedder surface is contained here rather than migrated.
#[cfg(feature = "diarization")]
mod diarization;
mod engine;
mod features;
mod load_files;
mod pool;
mod sizing;
mod state;
mod token_format;
mod types;
mod windows;

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
// Diarization adapter/types: all `#[allow(deprecated)]` polyvoice usage lives in
// `diarization` — see that module's docs for the migration blocker.
#[cfg(feature = "diarization")]
pub use diarization::{SharedExtractor, SpeakerEncoder, StreamingDiarizationState};
pub use state::{
    CommitPolicy, DecoderState, EndpointMode, EndpointReason, FeatureExtractor, StreamingState,
    TranscriptAssembler, TranscriptSegment, TranscriptSnapshot, WordInfo,
};
pub use types::{
    DEFAULT_HOTWORDS_BOOST, DiarizationOutcome, HotwordError, HotwordOverride,
    MAX_HOTWORD_PHRASE_CHARS, MAX_HOTWORDS_PER_REQUEST, OverrideError, TranscribeOverrides,
    TranscribeRequest, TranscribeResult, TranscribeSource, merge_channel_results,
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

#[cfg(test)]
mod timestamp_tests {
    use super::now_timestamp;

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
}
