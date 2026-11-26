//! Headerless telephony codecs (G.711 / G.722 raw streams).
//!
//! WAVE-carried G.711 / G.722 is decoded by the ryf WAVE path — the container
//! declares the codec, so this module only covers RTP-dump / Asterisk Monitor
//! captures that have no RIFF header.

#[cfg(feature = "file-decode")]
use anyhow::Result;
#[cfg(feature = "file-decode")]
use ryf::{ChannelMode, G711Law};

#[cfg(feature = "file-decode")]
use super::resample::{RESAMPLE_STAGING_FRAMES, ResampleTo16k, SampleRate};
#[cfg(feature = "file-decode")]
use super::wave::{decode_options, map_ryf_err, take_mono};
#[cfg(feature = "file-decode")]
use super::{audio_too_long_err, check_decode_abort, resolve_budget, whole_buffer_limit_secs};

/// Headerless telephony codecs accepted for raw uploads (`?codec=` on REST,
/// `--codec` on the CLI). WAV-carried G.711/G.722 needs no such hint — the
/// container declares the codec — so this enum only covers the raw RTP-dump /
/// Asterisk Monitor case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TelephonyCodec {
    /// G.711 μ-law (PCMU): one byte per sample, typically 8 kHz.
    Pcmu,
    /// G.711 A-law (PCMA): one byte per sample, typically 8 kHz.
    Pcma,
    /// G.722 ADPCM @ 64 kbit/s: two PCM16 samples per byte, native 16 kHz.
    G722,
}

impl TelephonyCodec {
    /// Parse a codec name, case-insensitive. Accepts the RTP/SIP aliases
    /// `ulaw` (PCMU) and `alaw` (PCMA) alongside the canonical names.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "pcmu" | "ulaw" => Some(Self::Pcmu),
            "pcma" | "alaw" => Some(Self::Pcma),
            "g722" => Some(Self::G722),
            _ => None,
        }
    }

    /// Validate the caller-declared sample rate of a raw stream. A G.711 byte
    /// stream carries no rate of its own, so any rate inside the telephony
    /// band is accepted and resampled from; G.722 always decodes to its
    /// native 16 kHz, but 8000 is accepted too because SDP/RTP announces
    /// G.722 with an 8 kHz clock rate for historical reasons.
    pub fn validate_sample_rate(self, sample_rate: u32) -> Result<(), String> {
        match self {
            Self::G722 if sample_rate != 8000 && sample_rate != 16000 => Err(format!(
                "g722 decodes to 16 kHz natively; sample_rate must be 8000 (SDP convention) or 16000, got {sample_rate}"
            )),
            Self::Pcmu | Self::Pcma if !(8000..=48000).contains(&sample_rate) => Err(format!(
                "sample_rate must be within 8000..=48000 Hz for raw G.711, got {sample_rate}"
            )),
            _ => Ok(()),
        }
    }
}

/// Decode a headerless telephony byte stream to mono f32 at 16 kHz.
///
/// `sample_rate` is the declared rate of the input (see
/// [`TelephonyCodec::validate_sample_rate`]); G.722 ignores it and always
/// decodes to its native 16 kHz. The whole-buffer safety ceiling matches container decodes
/// (`MAX_DURATION_S`), evaluated on the decoded sample count before the f32
/// buffer is allocated.
#[cfg(feature = "file-decode")]
pub fn decode_telephony_raw(
    data: &[u8],
    codec: TelephonyCodec,
    sample_rate: u32,
) -> Result<Vec<f32>> {
    decode_telephony_raw_bounded_with_abort(data, codec, sample_rate, None, None)
}

/// Decode raw mono telephony with an operator duration limit and cancellation.
/// The limit is checked at the decoded source rate before codec allocation:
/// G.711 has one frame per byte; G.722 has two at 16 kHz, including its 8 kHz
/// RTP clock alias. The whole-buffer safety ceiling still applies.
/// Cancellation is checked around the synchronous codec call and between
/// resampling stages; an active third-party codec call is not interruptible.
#[cfg(feature = "file-decode")]
pub fn decode_telephony_raw_bounded_with_abort(
    data: &[u8],
    codec: TelephonyCodec,
    sample_rate: u32,
    max_audio_secs: Option<f64>,
    abort: Option<&(dyn Fn() -> bool + Sync)>,
) -> Result<Vec<f32>> {
    check_decode_abort(abort)?;
    if data.is_empty() {
        anyhow::bail!("Empty audio payload");
    }
    codec
        .validate_sample_rate(sample_rate)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let limit_secs = whole_buffer_limit_secs(max_audio_secs);
    let opts = decode_options(Some(limit_secs), ChannelMode::Mono);
    let (rate, pcm) = match codec {
        TelephonyCodec::Pcmu => {
            let decoded = ryf::decode_g711(data, G711Law::MuLaw, sample_rate, 1, &opts)
                .map_err(map_ryf_err)?;
            take_mono(decoded)?
        }
        TelephonyCodec::Pcma => {
            let decoded = ryf::decode_g711(data, G711Law::ALaw, sample_rate, 1, &opts)
                .map_err(map_ryf_err)?;
            take_mono(decoded)?
        }
        TelephonyCodec::G722 => {
            let decoded = ryf::decode_g722(data, sample_rate, 1, &opts).map_err(map_ryf_err)?;
            take_mono(decoded)?
        }
    };
    check_decode_abort(abort)?;
    let (max_samples, _) = resolve_budget(Some(limit_secs), rate);
    if pcm.len() > max_samples {
        return Err(audio_too_long_err(pcm.len(), rate, limit_secs));
    }
    // Preserve the cached resampler's established staging boundaries. The codec
    // owns a whole source-rate PCM buffer; resampling also builds 16 kHz output.
    let mut acc = ResampleTo16k::new(SampleRate(rate), Some(pcm.len()));
    for piece in pcm.chunks(RESAMPLE_STAGING_FRAMES) {
        check_decode_abort(abort)?;
        acc.stage().extend_from_slice(piece);
        acc.flush_full()?;
    }
    check_decode_abort(abort)?;
    let samples = acc.finish()?;
    check_decode_abort(abort)?;
    let (max_samples, _) = resolve_budget(Some(limit_secs), 16_000);
    if samples.len() > max_samples {
        return Err(audio_too_long_err(samples.len(), 16_000, limit_secs));
    }
    Ok(samples)
}

/// Wrap mono f32 samples in a PCM16 RIFF/WAVE container. Samples are
/// clamped to [-1.0, 1.0]; non-finite values become silence.
#[cfg(feature = "file-decode")]
pub fn encode_wav_pcm16(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_size = (samples.len() * 2) as u32;
    let mut buf = Vec::with_capacity(44 + data_size as usize);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(36 + data_size).to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&1u16.to_le_bytes()); // mono
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    buf.extend_from_slice(&2u16.to_le_bytes()); // block align
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());
    for &sample in samples {
        buf.extend_from_slice(&pcm16_sample(sample).to_le_bytes());
    }
    buf
}

/// Reproduce [`encode_wav_pcm16`] followed by 16 kHz PCM16 decoding in place.
/// This deliberately multiplies by 32767 before rounding, then divides the
/// signed sample by 32768. It is not interchangeable with narrowband resampler
/// rounding, and must still be applied after that separate precision stage.
#[cfg(feature = "file-decode")]
pub fn quantize_wav_pcm16_in_place(samples: &mut [f32]) {
    for sample in samples {
        *sample = f32::from(pcm16_sample(*sample)) / 32768.0;
    }
}

#[cfg(feature = "file-decode")]
fn pcm16_sample(sample: f32) -> i16 {
    let sample = if sample.is_finite() {
        sample.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    (sample * 32767.0).round() as i16
}

/// Locate a RIFF chunk payload by 4-byte id, tolerating a truncated final
/// chunk (clamped to the buffer end so decoders see the bytes that actually
/// arrived). Test helper for fixtures that are not WAVE-decoded end-to-end.
#[cfg(all(test, feature = "file-decode"))]
pub(super) fn find_riff_chunk<'a>(data: &'a [u8], want: &[u8; 4]) -> Option<&'a [u8]> {
    if data.len() < 12 {
        return None;
    }
    let mut pos = 12usize;
    while pos + 8 <= data.len() {
        let id = &data[pos..pos + 4];
        let size = u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]])
            as usize;
        let start = pos + 8;
        let end = start.saturating_add(size).min(data.len());
        if id == want {
            return Some(&data[start..end]);
        }
        pos = start.saturating_add(size).saturating_add(size & 1);
    }
    None
}
