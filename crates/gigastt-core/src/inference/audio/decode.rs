//! Container decode (WAVE via ryf, AAC via syom, other formats via symphonia),
//! channel mix, and dual-mono detection.

#[cfg(feature = "file-decode")]
use anyhow::Result;
#[cfg(feature = "file-decode")]
use bytes::Bytes;

#[cfg(feature = "file-decode")]
use super::stream::FileWindows;
#[cfg(feature = "file-decode")]
use super::whole_buffer_limit_secs;

mod bytes_source;
#[cfg(any(test, feature = "file-decode"))]
pub(crate) use bytes_source::BytesMediaSource;
#[cfg(feature = "file-decode")]
mod probe;
#[cfg(feature = "file-decode")]
pub use probe::{probe_duration_bytes, probe_duration_file};
#[cfg(feature = "file-decode")]
mod channels;
#[cfg(feature = "file-decode")]
pub use channels::{
    decode_audio_bytes_shared_channels, decode_audio_bytes_shared_channels_bounded,
    decode_audio_bytes_shared_channels_bounded_with_abort, load_audio_channels,
};

// docs-drift: codecs
// Canonical decode surface, one token per supported input. Kept in sync with
// the FORMATS table in scripts/check-docs-drift.py and the format lists in
// docs/api.md ("Audio formats and telephony codecs") and docs/cli.md
// ("Supports:" line) — update all three together when adding a codec.
// wav
// wav-g711
// wav-g722
// wav-gsm
// mp3
// m4a
// ogg-vorbis
// ogg-opus
// webm-opus
// flac
// raw-pcmu
// raw-pcma
// raw-g722
// docs-drift: end

/// Decode any supported audio file to mono f32 samples at 16kHz.
///
/// Supports WAV (PCM/IEEE, G.711, G.722, GSM 06.10, ADPCM, RF64), MP3, M4A/AAC,
/// OGG/Vorbis, OGG/Opus (`.opus`), WebM/Opus, and FLAC.
/// Multi-channel audio is mixed to mono. This flat decode materializes the whole
/// buffer, so it is bounded by the ~30-minute whole-buffer safety ceiling; the
/// streaming file path (`Engine::transcribe_request`) pulls windows instead and
/// has no length limit.
///
/// # Errors
///
/// Returns an error if the file cannot be opened or decoded, or exceeds the
/// whole-buffer safety ceiling.
///
/// ```text
/// { !path.is_empty() }
/// fn decode_audio_file(path: &str) -> Result<Vec<f32>>
/// { ret.as_ref().map(|v| !v.is_empty() || path.is_empty()).unwrap_or(true) }
/// ```
#[cfg(feature = "file-decode")]
pub fn decode_audio_file(path: &str) -> Result<Vec<f32>> {
    decode_audio_file_bounded(path, None)
}

/// Flat decode with an explicit operator length budget. A flat drain
/// materializes the whole buffer, so it is always bounded by at least the
/// whole-buffer safety ceiling ([`whole_buffer_limit_secs`] clamps `max_audio_secs`
/// down to it); the engine's whole-buffer branch passes the request's
/// `--max-audio-secs`, the public wrapper passes `None` (ceiling only). Callers
/// that want peak memory independent of duration go through
/// `Engine::transcribe_request`, which pulls windows instead of draining.
#[cfg(feature = "file-decode")]
pub(crate) fn decode_audio_file_bounded(
    path: &str,
    max_audio_secs: Option<f64>,
) -> Result<Vec<f32>> {
    FileWindows::decode_file(path, Some(whole_buffer_limit_secs(max_audio_secs)))
}

/// Decode audio from raw bytes in memory (no temp file needed).
///
/// Backwards-compatible shim: clones `data` into a [`Bytes`] and delegates
/// to [`decode_audio_bytes_shared`]. New call sites should pass a
/// `bytes::Bytes` (or `axum::body::Bytes`) directly to avoid the copy.
///
/// # Errors
///
/// Returns an error if the bytes cannot be decoded or the audio exceeds the
/// whole-buffer safety ceiling.
///
/// ```text
/// { true }
/// fn decode_audio_bytes(data: &[u8]) -> Result<Vec<f32>>
/// { ret.as_ref().map(|v| !v.is_empty()).unwrap_or(true) }
/// ```
#[cfg(feature = "file-decode")]
pub fn decode_audio_bytes(data: &[u8]) -> Result<Vec<f32>> {
    decode_audio_bytes_shared(Bytes::copy_from_slice(data))
}

/// Decode audio from a shared [`Bytes`] buffer in place — no `to_vec()` clone.
///
/// Same logic as [`decode_audio_file`] but reads from a reference-counted
/// in-memory buffer. Supports WAV (PCM/IEEE, G.711, G.722, ADPCM, RF64), MP3,
/// M4A/AAC, OGG/Vorbis, OGG/Opus (`.opus`), WebM/Opus, and FLAC. Multi-channel
/// audio is mixed to mono. The whole-buffer safety ceiling is enforced
/// **incrementally** on each decoded packet: a
/// malicious or malformed upload is aborted before its decoded samples blow up
/// RAM.
///
/// # Errors
///
/// Returns an error if the bytes cannot be decoded or the audio exceeds the
/// whole-buffer safety ceiling.
///
/// ```text
/// { true }
/// fn decode_audio_bytes_shared(data: Bytes) -> Result<Vec<f32>>
/// { ret.as_ref().map(|v| !v.is_empty()).unwrap_or(true) }
/// ```
#[cfg(feature = "file-decode")]
pub fn decode_audio_bytes_shared(data: Bytes) -> Result<Vec<f32>> {
    decode_audio_bytes_shared_bounded(data, None)
}

/// Flat byte decode with an explicit operator length budget. `None` behaves
/// exactly like [`decode_audio_bytes_shared`] (whole-buffer ceiling); a
/// `Some(secs)` from `--max-audio-secs` lowers it. Public so the SSE streaming
/// handler, which materializes the whole buffer before chunking, can thread the
/// operator limit into its own decode.
#[cfg(feature = "file-decode")]
pub fn decode_audio_bytes_shared_bounded(
    data: Bytes,
    max_audio_secs: Option<f64>,
) -> Result<Vec<f32>> {
    FileWindows::decode_bytes(data, Some(whole_buffer_limit_secs(max_audio_secs)))
}

/// Flat decode with cooperative packet/block boundaries and the existing ceiling.
#[cfg(feature = "file-decode")]
pub(crate) fn decode_audio_file_with_abort(
    path: &str,
    max_audio_secs: Option<f64>,
    abort: Option<&(dyn Fn() -> bool + Sync)>,
) -> Result<Vec<f32>> {
    super::check_decode_abort(abort)?;
    FileWindows::open(
        path,
        super::WindowSpec::flat(),
        Some(whole_buffer_limit_secs(max_audio_secs)),
    )?
    .drain_to_vec_with_abort(abort)
}

#[cfg(feature = "file-decode")]
pub(crate) fn decode_audio_bytes_with_abort(
    data: Bytes,
    max_audio_secs: Option<f64>,
    abort: Option<&(dyn Fn() -> bool + Sync)>,
) -> Result<Vec<f32>> {
    super::check_decode_abort(abort)?;
    FileWindows::from_bytes(
        data,
        super::WindowSpec::flat(),
        Some(whole_buffer_limit_secs(max_audio_secs)),
    )?
    .drain_to_vec_with_abort(abort)
}

#[cfg(feature = "file-decode")]
mod scan;
#[cfg(feature = "file-decode")]
pub use scan::{
    ChannelScan, PreparedChannels, prepare_channels_for_vad, prepare_channels_for_vad_with_abort,
    scan_channels, scan_channels_with_abort,
};
mod dual_mono;
#[cfg(test)]
pub(crate) use dual_mono::normalized_correlation_for_test;
pub use dual_mono::{DualMonoDetector, is_dual_mono, mix_channels_to_mono};
