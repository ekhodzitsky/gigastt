//! Audio decoding, resampling, and buffer management utilities.

#[cfg(feature = "file-decode")]
mod aac;
#[cfg(feature = "file-decode")]
mod chunks;
mod decode;
mod opus;
mod pcm;
mod resample;
mod stream;
mod telephony;
#[cfg(feature = "file-decode")]
mod vad_windows;
#[cfg(feature = "file-decode")]
mod wave;

#[cfg(test)]
mod tests;

// Excluded under Miri: proptest runs hundreds of cases per property, each
// driving the resampler / WAV decoder — orders of magnitude too slow under
// the Miri interpreter to finish in the nightly job's budget. The same
// properties run natively on every `cargo test`; this only trims the Miri
// coverage, not the stable-toolchain coverage.
#[cfg(all(test, not(miri)))]
#[path = "proptests.rs"]
mod proptests;

// Parent mel-frame constants used by the streaming audio buffer helpers.
pub(crate) use super::{HOP_LENGTH, N_FFT};

// Shared constants -----------------------------------------------------------

pub(crate) const MAX_BUFFER_SAMPLES: usize = 16000 * 5; // 5 seconds at 16kHz
/// Explicit, documented safety ceiling (seconds) for the decode paths that must
/// hold the **whole** decoded buffer in RAM: speaker diarization,
/// `channels=split` (including its per-channel Opus decode), and the raw
/// headerless telephony codecs, which have no packet-wise decoder. The default
/// file path streams overlapping windows (see `Engine::decode_words_streaming`),
/// the VAD file path streams through
/// [`VadWindows`](super::audio::VadWindows), and OGG/Opus streams packet-wise
/// through the `opus-rs` fallback, so all three have peak audio memory O(one
/// window) regardless of length and *no* duration limit; the remaining paths
/// keep this bound so a multi-hour input refuses with a typed
/// [`AudioTooLong`](crate::error::GigasttError::AudioTooLong) instead of driving
/// the process into OOM. Operators can lower the effective limit for every path
/// (including the streaming one) with `--max-audio-secs`; there is no way to
/// raise it above this ceiling for the whole-buffer paths.
#[cfg(feature = "file-decode")]
pub(crate) const WHOLE_BUFFER_MAX_AUDIO_SECS: f64 = 1800.0; // 30 minutes
/// Upper bound on a header-declared sample rate. Legal rates (8k–48k) stay well
/// below this; anything above is a malformed/adversarial header and is rejected
/// before it can scale a sample budget or the capacity hint.
#[cfg(feature = "file-decode")]
pub(crate) const MAX_SAMPLE_RATE: u32 = 192_000;

/// Normalized cross-correlation threshold for dual-mono detection.
/// Some PBXs record the same mixed call to both channels of a "stereo" file.
/// Transcribing them as independent speakers would duplicate every word, so
/// when the two channels are nearly identical we fall back to the mono path.
pub(crate) const DUAL_MONO_CORRELATION_THRESHOLD: f64 = 0.98;

/// Source-rate sample budget for a `limit` at `sample_rate`. `None` (or a
/// non-positive / non-finite limit) means **unbounded** — the streaming window
/// path, whose peak audio memory is O(one window) regardless of length.
/// `Some(secs)` yields `secs × sample_rate` samples with the rate **unclamped**,
/// so the budget is honest at 96 kHz / 192 kHz instead of silently expiring at a
/// fraction of the stated seconds. Pure so the budget math is testable without
/// decoding a file.
#[cfg(feature = "file-decode")]
pub(crate) fn max_samples_for_secs(limit: Option<f64>, sample_rate: u32) -> usize {
    match limit {
        Some(secs) if secs.is_finite() && secs > 0.0 => (secs * sample_rate as f64) as usize,
        _ => usize::MAX,
    }
}

/// Resolve the effective seconds limit for a path that must hold the whole
/// decoded buffer in RAM. Never exceeds [`WHOLE_BUFFER_MAX_AUDIO_SECS`]; an
/// operator-supplied `--max-audio-secs` only ever lowers it.
#[cfg(feature = "file-decode")]
pub(crate) fn whole_buffer_limit_secs(user: Option<f64>) -> f64 {
    match user {
        Some(secs) if secs.is_finite() && secs > 0.0 => secs.min(WHOLE_BUFFER_MAX_AUDIO_SECS),
        _ => WHOLE_BUFFER_MAX_AUDIO_SECS,
    }
}

/// Resolve a caller-supplied seconds limit into the pair the decode loops need:
/// the source-rate sample budget (`usize::MAX` when unbounded) and the finite
/// seconds figure to report on a trip (`f64::INFINITY` when unbounded, which the
/// budget then never reaches).
#[cfg(feature = "file-decode")]
pub(crate) fn resolve_budget(limit: Option<f64>, sample_rate: u32) -> (usize, f64) {
    let limit_secs = limit
        .filter(|s| s.is_finite() && *s > 0.0)
        .unwrap_or(f64::INFINITY);
    (max_samples_for_secs(limit, sample_rate), limit_secs)
}

/// Convert a decode-layer [`anyhow::Error`] into a typed
/// [`GigasttError`](crate::error::GigasttError), preserving a typed
/// [`AudioTooLong`](crate::error::GigasttError::AudioTooLong) rather than
/// collapsing it into the generic `InvalidAudio` bucket so the wire code stays
/// `audio_too_long`. Any other error becomes `InvalidAudio`.
#[cfg(feature = "file-decode")]
pub(crate) fn decode_error(e: anyhow::Error) -> crate::error::GigasttError {
    match e.downcast::<crate::error::GigasttError>() {
        Ok(g) => g,
        Err(e) => crate::error::GigasttError::InvalidAudio {
            reason: format!("{e:#}"),
        },
    }
}

/// Build the typed [`AudioTooLong`](crate::error::GigasttError::AudioTooLong) as
/// an [`anyhow::Error`] for the decode layer (which returns `anyhow::Result`).
/// The engine / server seams downcast it back to the concrete variant so the
/// wire code stays `audio_too_long` instead of collapsing into the generic
/// "invalid audio" bucket. `observed_source_frames` is counted at
/// `sample_rate`; `limit_secs` is the ceiling that fired.
#[cfg(feature = "file-decode")]
pub(crate) fn audio_too_long_err(
    observed_source_frames: usize,
    sample_rate: u32,
    limit_secs: f64,
) -> anyhow::Error {
    crate::error::GigasttError::AudioTooLong {
        observed_secs: observed_source_frames as f64 / sample_rate as f64,
        limit_secs,
    }
    .into()
}

// Public API re-exports (stable paths under `inference::audio::…`) ------------

// Re-export for unit tests (`use super::*`); production callers use the decode path only.
#[cfg(test)]
pub(crate) use decode::BytesMediaSource;
#[cfg(feature = "file-decode")]
pub use decode::{
    ChannelScan, PreparedChannels, prepare_channels_for_vad, prepare_channels_for_vad_with_abort,
    scan_channels, scan_channels_with_abort,
};
#[cfg(feature = "file-decode")]
pub use decode::{
    decode_audio_bytes, decode_audio_bytes_shared, decode_audio_bytes_shared_bounded,
    decode_audio_bytes_shared_channels, decode_audio_bytes_shared_channels_bounded,
    decode_audio_bytes_shared_channels_bounded_with_abort, decode_audio_file, load_audio_channels,
    probe_duration_bytes, probe_duration_file,
};
// Length-bounded file decode used by the engine's whole-buffer branch; the
// streaming path threads its budget through `FileWindows` instead. The bytes /
// channels bounded variants above are `pub` because the server decodes those
// buffers itself; the path variant is engine-only.
pub use decode::{DualMonoDetector, is_dual_mono, mix_channels_to_mono};
#[cfg(feature = "file-decode")]
pub(crate) use decode::{decode_audio_bytes_with_abort, decode_audio_file_with_abort};

pub(crate) use pcm::{consume_audio_buffer, prepare_audio_buffer};
pub use pcm::{parse_pcm16_with_carry, parse_pcm16_with_carry_into};

pub use resample::{SampleRate, resample, resample_with_cache};

// Long-form window source. Crate-internal: the file-decode loop is the only
// consumer today, and the streaming source lands behind the same trait.
pub(crate) use stream::{PcmWindow, PcmWindows, SliceWindows, WindowSpec};
// Streaming container-backed window source: keeps peak audio memory O(one
// window) regardless of file duration. `Engine::transcribe_request` pulls
// windows from it; the public `decode_audio_*` functions drain it flat.
#[cfg(feature = "file-decode")]
pub(crate) use stream::FileWindows;
// VAD-compressed window source: same geometry over the silence-free timeline,
// so the VAD file path is O(one window) too and needs no duration ceiling.
#[cfg(feature = "file-decode")]
pub(crate) use vad_windows::VadWindows;
// Fixed-size streaming decode for callers driving the streaming recognizer
// (SSE file transcription, embedders). Public: it is the only way to decode a
// container without materializing it.
#[cfg(feature = "file-decode")]
pub use chunks::AudioChunks;

pub use telephony::TelephonyCodec;
#[cfg(feature = "file-decode")]
pub use telephony::{
    decode_telephony_raw, decode_telephony_raw_bounded_with_abort, encode_wav_pcm16,
    quantize_wav_pcm16_in_place,
};

// Internals re-exported so unit tests (`use super::*`) keep resolving them.
#[cfg(all(test, feature = "file-decode"))]
#[allow(unused_imports)]
pub(crate) use opus::is_recoverable_packet_eof;

/// Cooperative boundary for synchronous container work. An active codec call
/// completes before this check can observe cancellation.
#[cfg(feature = "file-decode")]
pub(crate) fn check_decode_abort(abort: Option<&(dyn Fn() -> bool + Sync)>) -> anyhow::Result<()> {
    if abort.is_some_and(|abort| abort()) {
        return Err(crate::error::GigasttError::Cancelled.into());
    }
    Ok(())
}
