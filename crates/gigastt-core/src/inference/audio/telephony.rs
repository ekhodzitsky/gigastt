//! Telephony codecs: G.711/G.722 raw streams and G.722-in-WAV fallback.

#[cfg(feature = "file-decode")]
use anyhow::Context;
#[cfg(feature = "file-decode")]
use anyhow::Result;

#[cfg(feature = "file-decode")]
use super::MAX_DURATION_S;
#[cfg(feature = "file-decode")]
use super::max_decode_samples;
#[cfg(feature = "file-decode")]
use super::resample::{RESAMPLE_STAGING_FRAMES, ResampleTo16k, SampleRate};

/// WAV format tags for ITU-T G.722 ADPCM. Symphonia's RIFF demuxer maps them
/// to `CODEC_TYPE_NULL` and there is no decoder for it, so G.722-in-WAV (what
/// Asterisk / Cisco / Teams players export) is detected up front and decoded
/// via the `audio-codec` crate. Both registered tags are accepted: 0x0064
/// (WAVE_FORMAT_G722_ADPCM, SBC/Asterisk exports) and 0x028F
/// (WAVE_FORMAT_ADPCM_G722, what ffmpeg/libavcodec writes).
#[cfg(feature = "file-decode")]
const WAV_FORMAT_TAGS_G722_ADPCM: [u16; 2] = [0x0064, 0x028F];

/// Size of the leading window inspected for a G.722 `fmt ` chunk. The `fmt `
/// chunk is virtually always the first chunk (ffmpeg, sox, and Asterisk all
/// write it at offset 12); when it lies beyond the window the file falls
/// through to symphonia and fails there as an unsupported codec, exactly as
/// before.
#[cfg(feature = "file-decode")]
const WAV_SNIFF_WINDOW: usize = 512;

/// Inspect the leading bytes of a RIFF/WAVE buffer for a G.722 ADPCM format
/// tag in the `fmt ` chunk. Returns `Some(is_g722)` when the `fmt ` chunk was
/// found inside the window, `None` when the buffer is not RIFF/WAVE or the
/// `fmt ` chunk lies beyond it.
#[cfg(feature = "file-decode")]
pub(super) fn sniff_wav_g722_tag(window: &[u8]) -> Option<bool> {
    if window.len() < 12 || &window[0..4] != b"RIFF" || &window[8..12] != b"WAVE" {
        return None;
    }
    let mut pos = 12usize;
    while pos + 8 <= window.len() {
        let id = &window[pos..pos + 4];
        let size = u32::from_le_bytes([
            window[pos + 4],
            window[pos + 5],
            window[pos + 6],
            window[pos + 7],
        ]) as usize;
        let start = pos + 8;
        if id == b"fmt " {
            // Need at least the 2-byte format tag.
            if size < 2 || start + 2 > window.len() {
                return None;
            }
            let tag = u16::from_le_bytes([window[start], window[start + 1]]);
            return Some(WAV_FORMAT_TAGS_G722_ADPCM.contains(&tag));
        }
        // RIFF chunks are word-aligned: odd sizes carry a pad byte.
        pos = start.saturating_add(size).saturating_add(size & 1);
    }
    None
}

/// Locate a RIFF chunk payload by 4-byte id, tolerating a truncated final
/// chunk (clamped to the buffer end so decoders see the bytes that actually
/// arrived).
#[cfg(feature = "file-decode")]
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

/// Read the header window of `path` and report whether it declares a
/// G.722-in-WAV stream. Open errors carry the same message the regular path
/// would produce; unreadable/short headers simply report `false` so the
/// symphonia path renders the canonical error.
#[cfg(feature = "file-decode")]
pub(super) fn sniffs_as_g722_wav(path: &str) -> Result<bool> {
    use std::io::Read as _;
    let mut file =
        std::fs::File::open(path).with_context(|| format!("Failed to open audio file: {path}"))?;
    let mut window = [0u8; WAV_SNIFF_WINDOW];
    let mut read = 0usize;
    while read < window.len() {
        match file.read(&mut window[read..]) {
            Ok(0) => break,
            Ok(n) => read += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => {
                return Err(e).with_context(|| format!("Failed to read audio file: {path}"));
            }
        }
    }
    Ok(sniff_wav_g722_tag(&window[..read]) == Some(true))
}

/// Decode a G.722-in-WAV buffer to mono f32 at 16 kHz (the native G.722
/// output rate). Returns `None` when the buffer is not G.722-in-WAV — the
/// caller then falls through to the symphonia pipeline; `Some(Err(..))` when
/// it IS G.722-in-WAV but malformed, so the error names the real problem
/// instead of surfacing as a generic "unsupported codec".
#[cfg(feature = "file-decode")]
pub(crate) fn try_decode_g722_wav(data: &[u8]) -> Option<Result<Vec<f32>>> {
    if sniff_wav_g722_tag(data) != Some(true) {
        return None;
    }
    let payload = match find_riff_chunk(data, b"data") {
        Some(p) if !p.is_empty() => p,
        _ => return Some(Err(anyhow::anyhow!("G.722 WAV has no data chunk"))),
    };
    // Duration cap, same budget as container decodes: two PCM16 samples per
    // encoded byte at the native 16 kHz rate.
    let num_samples = payload.len().saturating_mul(2);
    if num_samples > max_decode_samples(16000) {
        let observed_s = num_samples as f64 / 16000.0;
        return Some(Err(anyhow::anyhow!(
            "Audio file too long ({observed_s:.0}s). Maximum supported: {MAX_DURATION_S:.0}s."
        )));
    }
    let mut decoder = audio_codec::g722::G722Decoder::new();
    let pcm = audio_codec::Decoder::decode(&mut decoder, payload);
    tracing::info!(
        "Decoded G.722 WAV: {} samples at 16000Hz ({:.1}s)",
        pcm.len(),
        pcm.len() as f64 / 16000.0
    );
    Some(Ok(pcm.iter().map(|&s| f32::from(s) / 32768.0).collect()))
}

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
/// decodes to its native 16 kHz. The duration cap matches container decodes
/// (`MAX_DURATION_S`), evaluated on the decoded sample count before the f32
/// buffer is allocated.
#[cfg(feature = "file-decode")]
pub fn decode_telephony_raw(
    data: &[u8],
    codec: TelephonyCodec,
    sample_rate: u32,
) -> Result<Vec<f32>> {
    if data.is_empty() {
        anyhow::bail!("Empty audio payload");
    }
    codec
        .validate_sample_rate(sample_rate)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let (pcm, rate) = match codec {
        TelephonyCodec::Pcmu => {
            let mut decoder = audio_codec::pcmu::PcmuDecoder::new();
            (
                audio_codec::Decoder::decode(&mut decoder, data),
                sample_rate,
            )
        }
        TelephonyCodec::Pcma => {
            let mut decoder = audio_codec::pcma::PcmaDecoder::new();
            (
                audio_codec::Decoder::decode(&mut decoder, data),
                sample_rate,
            )
        }
        TelephonyCodec::G722 => {
            let mut decoder = audio_codec::g722::G722Decoder::new();
            (audio_codec::Decoder::decode(&mut decoder, data), 16000)
        }
    };
    if pcm.len() > max_decode_samples(rate) {
        let observed_s = pcm.len() as f64 / rate as f64;
        anyhow::bail!(
            "Audio file too long ({observed_s:.0}s). Maximum supported: {MAX_DURATION_S:.0}s."
        );
    }
    // Convert and resample in staged chunks so the full-length source-rate
    // f32 buffer is never materialized alongside the 16 kHz output.
    let mut acc = ResampleTo16k::new(SampleRate(rate), Some(pcm.len()));
    for piece in pcm.chunks(RESAMPLE_STAGING_FRAMES) {
        acc.stage()
            .extend(piece.iter().map(|&s| f32::from(s) / 32768.0));
        acc.flush_full()?;
    }
    acc.finish()
}

/// Wrap mono f32 samples in a PCM16 RIFF/WAVE container. Lets raw-codec
/// uploads (already decoded to 16 kHz) flow back through the standard
/// container-probing engine entry points without a temp file. Samples are
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
    for &s in samples {
        let v = if s.is_finite() {
            s.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        buf.extend_from_slice(&((v * 32767.0).round() as i16).to_le_bytes());
    }
    buf
}
