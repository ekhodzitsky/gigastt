//! Audio decoding, resampling, and buffer management utilities.

mod decode;
mod opus;
mod pcm;
mod resample;
mod stream;
mod telephony;

#[cfg(test)]
#[path = "tests.rs"]
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
/// Hard upper bound on file-transcription audio length (seconds). Long-form
/// inputs are decoded in bounded overlapping chunks (see
/// `Engine::decode_words_streaming`), so peak encoder memory is O(chunk)
/// regardless of file length; this cap bounds the fully decoded PCM buffer
/// instead. 30 minutes ≈ the largest uncompressed PCM16@16kHz upload the
/// default 50 MiB body limit admits (~27 min), and bounds the decoded f32
/// buffer at 30 min × 48 kHz × 4 B ≈ 346 MB per concurrent decode.
#[cfg(feature = "file-decode")]
pub(crate) const MAX_DURATION_S: f64 = 1800.0; // 30 minutes
/// Upper bound on a header-declared sample rate. Legal rates (8k–48k) stay well
/// below this; anything above is a malformed/adversarial header and is rejected
/// before it can scale the duration cap or the capacity hint.
#[cfg(feature = "file-decode")]
pub(crate) const MAX_SAMPLE_RATE: u32 = 192_000;
/// Ceiling used to size the duration cap and capacity hint. The header's
/// `sample_rate` is clamped to this when computing the sample budget, so a
/// crafted header cannot inflate either beyond `MAX_DURATION_S` × 48 kHz worth
/// of samples.
#[cfg(feature = "file-decode")]
pub(crate) const MAX_DECODE_SAMPLE_RATE: u32 = 48_000;

/// Normalized cross-correlation threshold for dual-mono detection.
/// Some PBXs record the same mixed call to both channels of a "stereo" file.
/// Transcribing them as independent speakers would duplicate every word, so
/// when the two channels are nearly identical we fall back to the mono path.
pub(crate) const DUAL_MONO_CORRELATION_THRESHOLD: f64 = 0.98;

/// Maximum number of decoded samples allowed for `sample_rate`, the budget used
/// by both the duration cap and the up-front capacity hint. The header rate is
/// clamped to [`MAX_DECODE_SAMPLE_RATE`] so a crafted header cannot inflate the
/// budget beyond [`MAX_DURATION_S`] × 48 kHz. Pure so the cap math is testable
/// without decoding a file.
#[cfg(feature = "file-decode")]
pub(crate) fn max_decode_samples(sample_rate: u32) -> usize {
    MAX_DURATION_S as usize * sample_rate.min(MAX_DECODE_SAMPLE_RATE) as usize
}

// Public API re-exports (stable paths under `inference::audio::…`) ------------

// Re-export for unit tests (`use super::*`); production callers use the decode path only.
#[cfg(test)]
pub(crate) use decode::BytesMediaSource;
#[cfg(feature = "file-decode")]
pub use decode::{
    decode_audio_bytes, decode_audio_bytes_shared, decode_audio_bytes_shared_channels,
    decode_audio_file, load_audio_channels,
};
pub use decode::{is_dual_mono, mix_channels_to_mono};

pub(crate) use pcm::{consume_audio_buffer, prepare_audio_buffer};
pub use pcm::{parse_pcm16_with_carry, parse_pcm16_with_carry_into};

pub use resample::{SampleRate, resample, resample_with_cache};

// Long-form window source. Crate-internal: the file-decode loop is the only
// consumer today, and the streaming source lands behind the same trait.
pub(crate) use stream::{PcmWindows, SliceWindows, WindowSpec};

pub use telephony::TelephonyCodec;
#[cfg(feature = "file-decode")]
pub use telephony::{decode_telephony_raw, encode_wav_pcm16};

// Internals re-exported so unit tests (`use super::*`) keep resolving them.
#[cfg(all(test, feature = "file-decode"))]
#[allow(unused_imports)]
pub(crate) use opus::{is_recoverable_packet_eof, opus_packet_frame_size};
#[cfg(all(test, feature = "file-decode"))]
#[allow(unused_imports)]
pub(crate) use telephony::try_decode_g722_wav;
