//! Audio decoding, resampling, and buffer management utilities.

#[cfg(feature = "file-decode")]
use anyhow::Context;
use anyhow::Result;
use bytes::Bytes;
use rubato::Resampler;
#[cfg(feature = "file-decode")]
use symphonia::core::codecs::audio::AudioDecoderOptions;
#[cfg(feature = "file-decode")]
use symphonia::core::formats::probe::Hint;
#[cfg(feature = "file-decode")]
use symphonia::core::formats::{FormatOptions, TrackType};
#[cfg(feature = "file-decode")]
use symphonia::core::io::{MediaSource, MediaSourceStream};
#[cfg(feature = "file-decode")]
use symphonia::core::meta::MetadataOptions;

use super::{HOP_LENGTH, N_FFT};

const MAX_BUFFER_SAMPLES: usize = 16000 * 5; // 5 seconds at 16kHz
/// Hard upper bound on file-transcription audio length (seconds). Long-form
/// inputs are decoded in bounded overlapping chunks (see
/// `Engine::transcribe_samples_chunked`), so peak encoder memory is O(chunk)
/// regardless of file length; this cap bounds the fully decoded PCM buffer
/// instead. 30 minutes ≈ the largest uncompressed PCM16@16kHz upload the
/// default 50 MiB body limit admits (~27 min), and bounds the decoded f32
/// buffer at 30 min × 48 kHz × 4 B ≈ 346 MB per concurrent decode.
#[cfg(feature = "file-decode")]
const MAX_DURATION_S: f64 = 1800.0; // 30 minutes
/// Upper bound on a header-declared sample rate. Legal rates (8k–48k) stay well
/// below this; anything above is a malformed/adversarial header and is rejected
/// before it can scale the duration cap or the capacity hint.
#[cfg(feature = "file-decode")]
const MAX_SAMPLE_RATE: u32 = 192_000;
/// Ceiling used to size the duration cap and capacity hint. The header's
/// `sample_rate` is clamped to this when computing the sample budget, so a
/// crafted header cannot inflate either beyond `MAX_DURATION_S` × 48 kHz worth
/// of samples.
#[cfg(feature = "file-decode")]
const MAX_DECODE_SAMPLE_RATE: u32 = 48_000;

/// Normalized cross-correlation threshold for dual-mono detection.
/// Some PBXs record the same mixed call to both channels of a "stereo" file.
/// Transcribing them as independent speakers would duplicate every word, so
/// when the two channels are nearly identical we fall back to the mono path.
const DUAL_MONO_CORRELATION_THRESHOLD: f64 = 0.98;

/// Maximum number of decoded samples allowed for `sample_rate`, the budget used
/// by both the duration cap and the up-front capacity hint. The header rate is
/// clamped to [`MAX_DECODE_SAMPLE_RATE`] so a crafted header cannot inflate the
/// budget beyond [`MAX_DURATION_S`] × 48 kHz. Pure so the cap math is testable
/// without decoding a file.
#[cfg(feature = "file-decode")]
fn max_decode_samples(sample_rate: u32) -> usize {
    MAX_DURATION_S as usize * sample_rate.min(MAX_DECODE_SAMPLE_RATE) as usize
}

/// Sample rate in Hz. Invariant: `rate > 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SampleRate(pub u32);

impl SampleRate {
    /// { rate > 0 }
    /// fn new(rate: u32) -> Result<SampleRate, String>
    /// { ret.as_ref().map(|r| r.0 > 0).unwrap_or(true) }
    pub fn new(rate: u32) -> Result<Self, String> {
        if rate == 0 {
            return Err("sample rate must be > 0".into());
        }
        Ok(SampleRate(rate))
    }

    /// { true }
    /// fn get(self) -> u32
    /// { ret > 0 }
    pub fn get(self) -> u32 {
        self.0
    }
}

/// A [`MediaSource`] that borrows its data from a reference-counted [`Bytes`]
/// buffer instead of cloning into a `Vec<u8>`.
///
/// Axum delivers REST upload bodies as `axum::body::Bytes`, which re-exports
/// `bytes::Bytes`. Before this type the decode path called `body.to_vec()` and
/// then wrapped the clone in `std::io::Cursor`, doubling the transient
/// memory footprint for every upload (a 50 MiB body briefly held 100 MiB in
/// RAM, plus another symphonia-side clone). `Bytes::clone` is a refcount bump,
/// so the shared variant decodes the original axum buffer in place.
///
/// The type is deliberately small and crate-private: it only needs to satisfy
/// `Read + Seek + Send + Sync` so symphonia's `MediaSourceStream` can drive it.
#[allow(dead_code)] // unused when `file-decode` is off (raw-PCM-only lean build)
pub(crate) struct BytesMediaSource {
    data: Bytes,
    pos: u64,
}

#[allow(dead_code)] // `new` is only called by the file-decode path
impl BytesMediaSource {
    pub(crate) fn new(data: Bytes) -> Self {
        Self { data, pos: 0 }
    }
}

impl std::io::Read for BytesMediaSource {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let len = self.data.len() as u64;
        if self.pos >= len {
            return Ok(0);
        }
        let start = self.pos as usize;
        let available = self.data.len() - start;
        let n = available.min(buf.len());
        buf[..n].copy_from_slice(&self.data[start..start + n]);
        self.pos += n as u64;
        Ok(n)
    }
}

impl std::io::Seek for BytesMediaSource {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        let len = self.data.len() as u64;
        // `std::io::Seek` semantics: seeking past the end is allowed; the next
        // read returns 0. Seeking to a negative offset is an error.
        let new_pos: i128 = match pos {
            std::io::SeekFrom::Start(n) => n as i128,
            std::io::SeekFrom::End(off) => len as i128 + off as i128,
            std::io::SeekFrom::Current(off) => self.pos as i128 + off as i128,
        };
        if new_pos < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "seek before start of buffer",
            ));
        }
        self.pos = new_pos as u64;
        Ok(self.pos)
    }
}

#[cfg(feature = "file-decode")]
impl MediaSource for BytesMediaSource {
    fn is_seekable(&self) -> bool {
        true
    }

    fn byte_len(&self) -> Option<u64> {
        Some(self.data.len() as u64)
    }
}

/// Decode any supported audio file to mono f32 samples at 16kHz.
///
/// Supports WAV, MP3, M4A/AAC, OGG/Vorbis, and FLAC via symphonia.
/// Multi-channel audio is mixed to mono. Files longer than the duration cap
/// (`MAX_DURATION_S`) are rejected; long files are decoded in bounded chunks.
///
/// # Errors
///
/// Returns an error if the file cannot be opened, decoded, or exceeds the duration limit.
///
/// ```text
/// { !path.is_empty() }
/// fn decode_audio_file(path: &str) -> Result<Vec<f32>>
/// { ret.as_ref().map(|v| !v.is_empty() || path.is_empty()).unwrap_or(true) }
/// ```
#[cfg(feature = "file-decode")]
pub fn decode_audio_file(path: &str) -> Result<Vec<f32>> {
    // G.722-in-WAV (format tag 0x0064) has no symphonia decoder: sniff the
    // first chunk headers, and only when they declare G.722 read the file
    // fully and decode it here. Other inputs stream through symphonia as
    // before, so plain WAVs keep their from-disk memory profile.
    if sniffs_as_g722_wav(path)? {
        let bytes =
            std::fs::read(path).with_context(|| format!("Failed to read audio file: {path}"))?;
        if let Some(result) = try_decode_g722_wav(&bytes) {
            return result;
        }
    }

    let file =
        std::fs::File::open(path).with_context(|| format!("Failed to open audio file: {path}"))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
    {
        hint.with_extension(ext);
    }

    let source_label = format!(
        "format={}",
        std::path::Path::new(path)
            .extension()
            .unwrap_or_default()
            .to_string_lossy()
    );

    decode_audio_inner(mss, hint, &source_label)
}

/// Decode audio from raw bytes in memory (no temp file needed).
///
/// Backwards-compatible shim: clones `data` into a [`Bytes`] and delegates
/// to [`decode_audio_bytes_shared`]. New call sites should pass a
/// `bytes::Bytes` (or `axum::body::Bytes`) directly to avoid the copy.
///
/// # Errors
///
/// Returns an error if the bytes cannot be decoded or the audio exceeds the duration limit.
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
/// in-memory buffer. Supports WAV, MP3, M4A/AAC, OGG/Vorbis, and FLAC via
/// symphonia. Multi-channel audio is mixed to mono. The duration cap
/// (`MAX_DURATION_S`) is enforced **incrementally** on each decoded packet: a
/// malicious or malformed upload is aborted before its decoded samples blow up
/// RAM.
///
/// # Errors
///
/// Returns an error if the bytes cannot be decoded or the audio exceeds the
/// duration limit.
///
/// ```text
/// { true }
/// fn decode_audio_bytes_shared(data: Bytes) -> Result<Vec<f32>>
/// { ret.as_ref().map(|v| !v.is_empty()).unwrap_or(true) }
/// ```
#[cfg(feature = "file-decode")]
pub fn decode_audio_bytes_shared(data: Bytes) -> Result<Vec<f32>> {
    // G.722-in-WAV (format tag 0x0064) has no symphonia decoder; detect it
    // before the generic probe and decode via the telephony fallback.
    if let Some(result) = try_decode_g722_wav(&data) {
        return result;
    }
    let source = BytesMediaSource::new(data);
    let mss = MediaSourceStream::new(Box::new(source), Default::default());
    let hint = Hint::new();
    decode_audio_inner(mss, hint, "bytes")
}

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
fn sniff_wav_g722_tag(window: &[u8]) -> Option<bool> {
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
fn find_riff_chunk<'a>(data: &'a [u8], want: &[u8; 4]) -> Option<&'a [u8]> {
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
fn sniffs_as_g722_wav(path: &str) -> Result<bool> {
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
fn try_decode_g722_wav(data: &[u8]) -> Option<Result<Vec<f32>>> {
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
    let mut samples: Vec<f32> = pcm.iter().map(|&s| f32::from(s) / 32768.0).collect();
    if rate != 16000 {
        samples =
            resample(&samples, SampleRate(rate), SampleRate(16000)).context("Resampling failed")?;
    }
    Ok(samples)
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

/// Decode an audio file to one f32 sample vector per channel at 16 kHz.
///
/// Same probe/decode/resample pipeline as [`decode_audio_file`], but keeps
/// channels separate. The mono mix path remains unchanged.
#[cfg(feature = "file-decode")]
pub fn load_audio_channels(path: &str) -> Result<Vec<Vec<f32>>> {
    let file =
        std::fs::File::open(path).with_context(|| format!("Failed to open audio file: {path}"))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
    {
        hint.with_extension(ext);
    }

    let source_label = format!(
        "format={}",
        std::path::Path::new(path)
            .extension()
            .unwrap_or_default()
            .to_string_lossy()
    );

    decode_audio_inner_channels(mss, hint, &source_label)
}

/// Decode raw audio bytes to one f32 sample vector per channel at 16 kHz.
#[cfg(feature = "file-decode")]
pub fn decode_audio_bytes_shared_channels(data: Bytes) -> Result<Vec<Vec<f32>>> {
    let source = BytesMediaSource::new(data);
    let mss = MediaSourceStream::new(Box::new(source), Default::default());
    decode_audio_inner_channels(mss, Hint::new(), "bytes")
}

/// Shared non-mixing decode: probe → format → decode → per-channel resample.
#[cfg(feature = "file-decode")]
fn decode_audio_inner_channels<'s>(
    mss: MediaSourceStream<'s>,
    hint: Hint,
    source_label: &str,
) -> Result<Vec<Vec<f32>>> {
    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .context("Unsupported audio format")?;

    let track = format
        .default_track(TrackType::Audio)
        .context("No audio track found")?;
    let track_id = track.id;
    let audio_params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .context("No audio codec parameters")?;
    let sample_rate = audio_params.sample_rate.context("Unknown sample rate")?;
    if sample_rate == 0 || sample_rate > MAX_SAMPLE_RATE {
        anyhow::bail!("Unsupported sample rate: {sample_rate}Hz");
    }
    let channels = audio_params
        .channels
        .as_ref()
        .map(|c| c.count())
        .unwrap_or(1);
    let n_frames_hint = track.num_frames;
    let max_samples = max_decode_samples(sample_rate);

    tracing::info!("Audio ({source_label}): {sample_rate}Hz, {channels}ch (split)");

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
        .context("Unsupported audio codec")?;

    let mut per_channel: Vec<Vec<f32>> = (0..channels)
        .map(|_| match n_frames_hint {
            Some(n) if n > 0 && n <= max_samples as u64 => Vec::with_capacity(n as usize),
            _ => Vec::new(),
        })
        .collect();

    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(e) => return Err(anyhow::anyhow!("Error reading packet: {e}")),
        };

        if packet.track_id != track_id {
            continue;
        }

        let decoded = decoder.decode(&packet).context("Decode error")?;
        let spec = decoded.spec().clone();
        let num_frames = decoded.frames();
        let ch = spec.channels().count();

        if ch > per_channel.len() {
            per_channel.resize_with(ch, Vec::new);
        }

        if ch > 1 {
            let mut interleaved: Vec<f32> = Vec::with_capacity(num_frames * ch);
            decoded.copy_to_vec_interleaved(&mut interleaved);
            for frame in 0..num_frames {
                for c in 0..ch {
                    per_channel[c].push(interleaved[frame * ch + c]);
                }
            }
        } else if !per_channel.is_empty() {
            let offset = per_channel[0].len();
            per_channel[0].resize(offset + num_frames, 0.0);
            decoded.copy_to_slice_interleaved(&mut per_channel[0][offset..]);
        }

        if per_channel.first().map(|v| v.len()).unwrap_or(0) > max_samples {
            let observed_s =
                per_channel.first().map(|v| v.len()).unwrap_or(0) as f64 / sample_rate as f64;
            anyhow::bail!(
                "Audio file too long ({:.0}s). Maximum supported: {MAX_DURATION_S:.0}s.",
                observed_s
            );
        }
    }

    let duration_s = per_channel
        .first()
        .map(|v| v.len() as f64 / sample_rate as f64)
        .unwrap_or(0.0);
    tracing::info!(
        "Decoded {} channel(s), first channel {} samples at {}Hz ({:.1}s)",
        per_channel.len(),
        per_channel.first().map(|v| v.len()).unwrap_or(0),
        sample_rate,
        duration_s
    );

    if sample_rate != 16000 {
        per_channel = per_channel
            .into_iter()
            .map(|ch| {
                resample(&ch, SampleRate(sample_rate), SampleRate(16000))
                    .context("Resampling failed")
            })
            .collect::<Result<Vec<_>>>()?;
        let channels = per_channel.len();
        tracing::info!("Resampled {channels} channel(s) to 16kHz");
    }

    Ok(per_channel)
}

/// Average multiple channels into a single mono vector.
pub fn mix_channels_to_mono(channels: &[Vec<f32>]) -> Vec<f32> {
    if channels.is_empty() {
        return Vec::new();
    }
    if channels.len() == 1 {
        return channels[0].clone();
    }
    let n = channels.iter().map(|c| c.len()).min().unwrap_or(0);
    (0..n)
        .map(|i| channels.iter().map(|c| c[i]).sum::<f32>() / channels.len() as f32)
        .collect()
}

/// Return `true` if a two-channel stream is dual-mono (both channels nearly
/// identical). Empty or single-channel input returns `false`.
pub fn is_dual_mono(channels: &[Vec<f32>]) -> bool {
    if channels.len() != 2 {
        return false;
    }
    let (left, right) = (&channels[0], &channels[1]);
    if left.is_empty() || right.is_empty() {
        return false;
    }
    let len = left.len().min(right.len());
    normalized_correlation(&left[..len], &right[..len]) > DUAL_MONO_CORRELATION_THRESHOLD
}

fn normalized_correlation(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len();
    if n == 0 || n != b.len() {
        return 0.0;
    }
    let mean_a = a.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let mean_b = b.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for (&x, &y) in a.iter().zip(b) {
        let dx = x as f64 - mean_a;
        let dy = y as f64 - mean_b;
        cov += dx * dy;
        var_a += dx * dx;
        var_b += dy * dy;
    }
    let denom = var_a.sqrt() * var_b.sqrt();
    if denom < 1e-12 {
        return 0.0;
    }
    cov / denom
}

/// Shared decode logic: probe → format → decode → mono mix → duration check → resample.
#[cfg(feature = "file-decode")]
fn decode_audio_inner<'s>(
    mss: MediaSourceStream<'s>,
    hint: Hint,
    source_label: &str,
) -> Result<Vec<f32>> {
    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .context("Unsupported audio format")?;

    let track = format
        .default_track(TrackType::Audio)
        .context("No audio track found")?;
    let track_id = track.id;
    let audio_params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .context("No audio codec parameters")?;
    let sample_rate = audio_params.sample_rate.context("Unknown sample rate")?;
    if sample_rate == 0 || sample_rate > MAX_SAMPLE_RATE {
        anyhow::bail!("Unsupported sample rate: {sample_rate}Hz");
    }
    let channels = audio_params
        .channels
        .as_ref()
        .map(|c| c.count())
        .unwrap_or(1);
    // Some formats (WAV, FLAC) publish the total frame count in the track;
    // reserve up-front to avoid `Vec` reallocation thrash for large uploads.
    // Streaming codecs (MP3) leave this as None and we fall back to the
    // default growth strategy.
    let n_frames_hint = track.num_frames;

    tracing::info!("Audio ({source_label}): {sample_rate}Hz, {channels}ch");

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
        .context("Unsupported audio codec")?;

    // Sample budget from a CLAMPED rate (header `sample_rate` capped at
    // MAX_DECODE_SAMPLE_RATE), so a crafted header cannot inflate the duration
    // cap or the capacity hint. Computed before the capacity match so the hint
    // is bounded by the same budget.
    let max_samples: usize = max_decode_samples(sample_rate);

    let mut all_samples: Vec<f32> = match n_frames_hint {
        Some(n) if n > 0 && n <= max_samples as u64 => Vec::with_capacity(n as usize),
        _ => Vec::new(),
    };

    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(e) => return Err(anyhow::anyhow!("Error reading packet: {e}")),
        };

        if packet.track_id != track_id {
            continue;
        }

        let decoded = decoder.decode(&packet).context("Decode error")?;
        let spec = decoded.spec().clone();
        let num_frames = decoded.frames();
        let ch = spec.channels().count();

        // Mix to mono if multi-channel
        if ch > 1 {
            let mut interleaved: Vec<f32> = Vec::with_capacity(num_frames * ch);
            decoded.copy_to_vec_interleaved(&mut interleaved);
            for frame in 0..num_frames {
                let mut sum = 0.0_f32;
                for c in 0..ch {
                    sum += interleaved[frame * ch + c];
                }
                all_samples.push(sum / ch as f32);
            }
        } else {
            let offset = all_samples.len();
            all_samples.resize(offset + num_frames, 0.0);
            decoded.copy_to_slice_interleaved(&mut all_samples[offset..]);
        }

        // Incremental duration cap: abort before the next packet is decoded
        // if the accumulated buffer already exceeds the duration budget.
        // This prevents a crafted upload from allocating hundreds of MiB of
        // PCM before the post-loop guard gets a chance to run.
        if all_samples.len() > max_samples {
            let observed_s = all_samples.len() as f64 / sample_rate as f64;
            anyhow::bail!(
                "Audio file too long ({:.0}s). Maximum supported: {MAX_DURATION_S:.0}s.",
                observed_s
            );
        }
    }

    let duration_s = all_samples.len() as f64 / sample_rate as f64;
    tracing::info!(
        "Decoded {} samples at {}Hz ({:.1}s)",
        all_samples.len(),
        sample_rate,
        duration_s
    );

    // Resample to 16kHz if needed
    if sample_rate != 16000 {
        all_samples = resample(&all_samples, SampleRate(sample_rate), SampleRate(16000))
            .context("Resampling failed")?;
        tracing::info!("Resampled to 16kHz: {} samples", all_samples.len());
    }

    Ok(all_samples)
}

/// High-quality polyphase FIR resampler (rubato Async, sinc interpolation).
///
/// Non-finite samples (NaN, infinity) are replaced with `0.0` before resampling.
///
/// ```text
/// { from_rate.0 > 0 && to_rate.0 > 0 }
/// fn resample(samples: &[f32], from_rate: SampleRate, to_rate: SampleRate) -> Result<Vec<f32>>
/// { ret.as_ref().map(|v| !v.is_empty() || samples.is_empty() || from_rate == to_rate).unwrap_or(true) }
/// ```
pub fn resample(samples: &[f32], from_rate: SampleRate, to_rate: SampleRate) -> Result<Vec<f32>> {
    if samples.is_empty() || from_rate.0 == 0 || to_rate.0 == 0 {
        return Ok(Vec::new());
    }
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }

    // Sanitize non-finite values
    let samples: Vec<f32> = samples
        .iter()
        .map(|&s| if s.is_finite() { s } else { 0.0 })
        .collect();

    use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
    use rubato::{
        Async, FixedAsync, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let ratio = to_rate.0 as f64 / from_rate.0 as f64;
    let chunk = samples.len();
    let mut resampler = Async::<f32>::new_sinc(ratio, 2.0, &params, chunk, 1, FixedAsync::Input)
        .map_err(|e| anyhow::anyhow!("Resampler init failed: {e}"))?;

    let input_data = [samples];
    let out_frames = resampler.output_frames_next();
    let mut output_data = [vec![0.0f32; out_frames]];
    {
        let input = SequentialSliceOfVecs::new(&input_data, 1, chunk)
            .map_err(|e| anyhow::anyhow!("Resampler input adapter failed: {e}"))?;
        let mut output = SequentialSliceOfVecs::new_mut(&mut output_data, 1, out_frames)
            .map_err(|e| anyhow::anyhow!("Resampler output adapter failed: {e}"))?;
        resampler
            .process_into_buffer(&input, &mut output, None)
            .map_err(|e| anyhow::anyhow!("Resampling failed: {e}"))?;
    }
    let [out_vec] = output_data;
    Ok(out_vec)
}

/// Lower bound for the cached streaming resampler's chunk capacity.
///
/// The capacity is fixed when the resampler is first created and cannot be
/// raised later without recreating it (which would reset the FIR state). A
/// tiny first frame would otherwise cap every later frame at that size and
/// make the oversized-frame split loop run many times per call; 4096 samples
/// covers ~85 ms at 48 kHz, so realistic frames never split. The value only
/// bounds per-call overhead — correctness holds for any capacity.
const MIN_RESAMPLER_CAPACITY: usize = 4096;

/// Resample audio using an optional cached resampler, writing into a caller-provided buffer.
///
/// The cached resampler is created once on first call and reused for the rest
/// of the session; it is never recreated, so the FIR history and fractional
/// phase survive across frames and no seam discontinuities appear at frame
/// boundaries. Chunk sizes may vary freely between calls: sizes up to the
/// resampler capacity are applied via `set_chunk_size` (which rubato supports
/// without touching filter state), and larger chunks are fed through the same
/// resampler in capacity-sized pieces.
///
/// Non-finite samples are sanitized in-place.
///
/// `samples` is consumed (moved) so that in-place sanitization avoids an
/// extra allocation. Callers that already own the input vector should pass
/// it directly; the buffer is not borrowed after the call.
///
/// ```text
/// { from_rate.0 > 0 && to_rate.0 > 0 }
/// fn resample_with_cache(samples: Vec<f32>, from_rate: SampleRate, to_rate: SampleRate, cache: &mut Option<rubato::Async<f32>>, out_buf: &mut Vec<f32>) -> anyhow::Result<()>
/// { ret.as_ref().map(|v| !v.is_empty() || samples.is_empty() || from_rate == to_rate).unwrap_or(true) }
/// ```
pub fn resample_with_cache(
    mut samples: Vec<f32>,
    from_rate: SampleRate,
    to_rate: SampleRate,
    cache: &mut Option<rubato::Async<f32>>,
    out_buf: &mut Vec<f32>,
) -> anyhow::Result<()> {
    if samples.is_empty() || from_rate.0 == 0 || to_rate.0 == 0 {
        out_buf.clear();
        return Ok(());
    }
    if from_rate == to_rate {
        *out_buf = samples;
        return Ok(());
    }

    // Sanitize non-finite values in-place
    for s in &mut samples {
        if !s.is_finite() {
            *s = 0.0;
        }
    }

    if cache.is_none() {
        use rubato::{
            Async, FixedAsync, SincInterpolationParameters, SincInterpolationType, WindowFunction,
        };
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };
        let ratio = to_rate.0 as f64 / from_rate.0 as f64;
        // Fix the capacity up front: it can never be raised without
        // recreating the resampler and losing the FIR state.
        let capacity = samples.len().max(MIN_RESAMPLER_CAPACITY);
        let r = Async::<f32>::new_sinc(ratio, 2.0, &params, capacity, 1, FixedAsync::Input)
            .map_err(|e| anyhow::anyhow!("Resampler init failed: {e}"))?;
        *cache = Some(r);
    }

    let resampler = match cache.as_mut() {
        Some(r) => r,
        None => anyhow::bail!("Resampler cache is None after initialization"),
    };
    out_buf.clear();
    let max_input = resampler.input_frames_max();
    if samples.len() <= max_input {
        process_cached_chunk(resampler, samples, out_buf)?;
    } else {
        // Frame exceeds the fixed capacity: feed it in capacity-sized pieces
        // through the same resampler so the FIR state carries across pieces.
        let mut piece_out = Vec::new();
        for piece in samples.chunks(max_input) {
            process_cached_chunk(resampler, piece.to_vec(), &mut piece_out)?;
            out_buf.extend_from_slice(&piece_out);
        }
    }
    Ok(())
}

/// Run one chunk through the cached resampler, replacing `dst` with the output.
///
/// `samples.len()` must not exceed `resampler.input_frames_max()`. The chunk
/// size is applied via `set_chunk_size`, which adjusts the required
/// input/output lengths while preserving the FIR history and fractional
/// phase — this is what keeps variable-sized streaming frames seamless.
fn process_cached_chunk(
    resampler: &mut rubato::Async<f32>,
    samples: Vec<f32>,
    dst: &mut Vec<f32>,
) -> anyhow::Result<()> {
    use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;

    let chunk = samples.len();
    resampler
        .set_chunk_size(chunk)
        .map_err(|e| anyhow::anyhow!("Resampler chunk resize failed: {e}"))?;
    let needed = resampler.output_frames_next();
    dst.clear();
    dst.resize(needed, 0.0);

    let input_data = [samples];
    let input = SequentialSliceOfVecs::new(&input_data, 1, chunk)
        .map_err(|e| anyhow::anyhow!("Resampler input adapter failed: {e}"))?;
    let mut output = SequentialSliceOfVecs::new_mut(std::slice::from_mut(dst), 1, needed)
        .map_err(|e| anyhow::anyhow!("Resampler output adapter failed: {e}"))?;
    resampler
        .process_into_buffer(&input, &mut output, None)
        .map_err(|e| anyhow::anyhow!("Resampling failed: {e}"))?;
    Ok(())
}

/// Parse PCM16 LE bytes into f32 samples, carrying a trailing odd byte across calls.
///
/// WebSocket clients may split their audio stream on arbitrary byte boundaries.
/// This function maintains a carry byte across frames so that odd-length payloads
/// don't introduce a 1-sample phase shift in the decoded audio.
pub fn parse_pcm16_with_carry(data: &[u8], pending: &mut Option<u8>) -> Vec<f32> {
    let mut out = Vec::new();
    parse_pcm16_with_carry_into(data, pending, &mut out);
    out
}

/// Parse PCM16 LE bytes into f32 samples, writing into a caller-provided buffer.
///
/// Same semantics as [`parse_pcm16_with_carry`] but avoids allocating a new
/// `Vec<f32>` on every call when the caller supplies a reusable buffer.
pub fn parse_pcm16_with_carry_into(data: &[u8], pending: &mut Option<u8>, out: &mut Vec<f32>) {
    out.clear();
    let carry_prev = pending.take();
    let needs_combine = carry_prev.is_some() || !data.len().is_multiple_of(2);

    if needs_combine {
        out.reserve(data.len().div_ceil(2));
        let mut bytes = data.iter().copied();
        if let Some(prev) = carry_prev {
            if let Some(b) = bytes.next() {
                out.push(i16::from_le_bytes([prev, b]) as f32 / 32768.0);
            } else {
                *pending = Some(prev);
                return;
            }
        }
        while let Some(b0) = bytes.next() {
            let b1 = match bytes.next() {
                Some(b) => b,
                None => {
                    *pending = Some(b0);
                    break;
                }
            };
            out.push(i16::from_le_bytes([b0, b1]) as f32 / 32768.0);
        }
    } else {
        out.reserve(data.len() / 2);
        for chunk in data.chunks_exact(2) {
            out.push(i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0);
        }
    }
}

/// Prepare audio buffer for processing: merge new samples with leftover,
/// truncate if too long, split into usable samples and new leftover.
///
/// Returns `Some(usable_samples)` if enough data for at least one frame,
/// `None` if all data was buffered for the next call.
/// Updates `buffer` in-place with leftover samples.
///
/// { true }
/// fn prepare_audio_buffer(new_samples: &[f32], buffer: &mut Vec<f32>) -> Option<usize>
/// { ret.is_none() == (buffer.len() < N_FFT) }
/// Determine how many samples at the front of `buffer` form complete frames.
///
/// Returns `Some(usable)` if enough data for at least one frame, `None` otherwise.
/// The caller should borrow `&buffer[..usable]`, then call
/// [`consume_audio_buffer`] to shift the leftovers.
pub(crate) fn prepare_audio_buffer(new_samples: &[f32], buffer: &mut Vec<f32>) -> Option<usize> {
    buffer.extend_from_slice(new_samples);

    if buffer.len() > MAX_BUFFER_SAMPLES {
        tracing::warn!("Audio buffer exceeded 5s limit, truncating");
        let excess = buffer.len() - MAX_BUFFER_SAMPLES;
        buffer.copy_within(excess.., 0);
        buffer.truncate(MAX_BUFFER_SAMPLES);
    }

    let hop_length = HOP_LENGTH;
    let n_fft = N_FFT;
    if buffer.len() >= n_fft {
        let num_frames = (buffer.len() - n_fft) / hop_length + 1;
        let usable = (num_frames - 1) * hop_length + n_fft;
        Some(usable)
    } else {
        None
    }
}

/// Shift leftover samples in `buffer` forward by `usable` samples and truncate.
pub(crate) fn consume_audio_buffer(buffer: &mut Vec<f32>, usable: usize) {
    buffer.copy_within(usable.., 0);
    buffer.truncate(buffer.len() - usable);
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- resample tests ---

    #[test]
    #[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
    fn test_resample_downsample_length() {
        let input: Vec<f32> = (0..4800).map(|i| (i as f32).sin()).collect();
        let output = resample(&input, SampleRate(48000), SampleRate(16000)).unwrap();
        // Rubato FIR filter has sinc_len/2 delay; output is shorter than ideal ratio.
        // For 4800 samples at 3:1 ratio, expect ~1556 (not exact 1600).
        assert!(!output.is_empty());
        assert!(
            output.len() > 1400 && output.len() < 1700,
            "Unexpected output length: {}",
            output.len()
        );
    }

    #[test]
    #[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
    fn test_resample_upsample_length() {
        let input: Vec<f32> = (0..800).map(|i| (i as f32).sin()).collect();
        let output = resample(&input, SampleRate(8000), SampleRate(16000)).unwrap();
        // Rubato FIR delay reduces output; expect ~1340 (not exact 1600).
        assert!(!output.is_empty());
        assert!(
            output.len() > 1200 && output.len() < 1700,
            "Unexpected output length: {}",
            output.len()
        );
    }

    #[test]
    #[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
    fn test_resample_preserves_dc() {
        // Constant signal should remain approximately constant after resampling.
        // Rubato FIR filter may cause transients at edges; check the middle 80%.
        let input = vec![0.5_f32; 4800];
        let output = resample(&input, SampleRate(48000), SampleRate(16000)).unwrap();
        let start = output.len() / 10;
        let end = output.len() - start;
        for &sample in &output[start..end] {
            assert!(
                (sample - 0.5).abs() < 0.05,
                "DC signal not preserved: {sample}"
            );
        }
    }

    #[test]
    fn test_resample_empty() {
        let output = resample(&[], SampleRate(48000), SampleRate(16000)).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn test_resample_zero_rate_returns_empty() {
        let input = vec![1.0, 2.0, 3.0];
        assert!(
            resample(&input, SampleRate(0), SampleRate(16000))
                .unwrap()
                .is_empty()
        );
        assert!(
            resample(&input, SampleRate(16000), SampleRate(0))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn test_resample_same_rate() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = resample(&input, SampleRate(16000), SampleRate(16000)).unwrap();
        assert_eq!(output.len(), input.len());
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    // --- prepare_audio_buffer tests ---

    #[test]
    fn test_buffer_short_input_returns_none() {
        // Less than N_FFT (320) samples → buffer everything
        let new_samples = vec![0.0; 100];
        let mut buffer = Vec::new();
        let result = prepare_audio_buffer(&new_samples, &mut buffer);
        assert!(result.is_none());
        assert_eq!(buffer.len(), 100);
    }

    #[test]
    fn test_buffer_exact_frame() {
        // Exactly N_FFT (320) samples → one frame, no leftover
        let new_samples = vec![1.0; N_FFT];
        let mut buffer = Vec::new();
        let result = prepare_audio_buffer(&new_samples, &mut buffer);
        assert!(result.is_some());
        let usable = result.unwrap();
        assert_eq!(usable, N_FFT);
        consume_audio_buffer(&mut buffer, usable);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_buffer_leftover_correct() {
        // N_FFT + 50 samples → one frame usable, 50 leftover
        let new_samples = vec![1.0; N_FFT + 50];
        let mut buffer = Vec::new();
        let result = prepare_audio_buffer(&new_samples, &mut buffer);
        assert!(result.is_some());
        let usable = result.unwrap();
        assert_eq!(usable, N_FFT); // one frame
        consume_audio_buffer(&mut buffer, usable);
        assert_eq!(buffer.len(), 50);
    }

    #[test]
    fn test_buffer_accumulates_across_calls() {
        let mut buffer = Vec::new();
        // First call: 200 samples (< 320) → buffered
        let result = prepare_audio_buffer(&vec![1.0; 200], &mut buffer);
        assert!(result.is_none());
        assert_eq!(buffer.len(), 200);

        // Second call: 200 more → total 400, enough for 1 frame (320), leftover 80
        let result = prepare_audio_buffer(&vec![2.0; 200], &mut buffer);
        assert!(result.is_some());
        let usable = result.unwrap();
        assert_eq!(usable, 320);
        consume_audio_buffer(&mut buffer, usable);
        assert_eq!(buffer.len(), 80);
    }

    #[test]
    fn test_buffer_truncation_at_5s() {
        // More than 80000 samples (5s at 16kHz) → truncate to last 80000
        let mut buffer = vec![0.0; 90000];
        let new_samples = vec![1.0; 1000];
        let result = prepare_audio_buffer(&new_samples, &mut buffer);
        // Total was 91000, truncated to 80000, then split into usable + leftover
        assert!(result.is_some());
        let usable = result.unwrap();
        consume_audio_buffer(&mut buffer, usable);
        assert!(usable + buffer.len() <= MAX_BUFFER_SAMPLES);
    }

    #[test]
    fn test_buffer_multi_frame() {
        // N_FFT + HOP_LENGTH = 480 → 2 frames, no leftover
        let new_samples = vec![1.0; N_FFT + HOP_LENGTH];
        let mut buffer = Vec::new();
        let result = prepare_audio_buffer(&new_samples, &mut buffer);
        assert!(result.is_some());
        // 2 frames: usable = (2-1)*160 + 320 = 480
        let usable = result.unwrap();
        assert_eq!(usable, N_FFT + HOP_LENGTH);
        consume_audio_buffer(&mut buffer, usable);
        assert!(buffer.is_empty());
    }

    // --- stress tests: robustness edge cases ---

    #[test]
    #[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
    fn test_resample_nan_input() {
        let input = vec![f32::NAN; 1000];
        let output = resample(&input, SampleRate(48000), SampleRate(16000)).unwrap();
        // NaN should be replaced with zeros
        assert!(!output.is_empty());
        for &s in &output {
            assert!(s.is_finite(), "NaN should be sanitized to zero, got {s}");
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
    fn test_resample_infinity_input() {
        let input = vec![f32::INFINITY; 500];
        let output = resample(&input, SampleRate(48000), SampleRate(16000)).unwrap();
        assert!(!output.is_empty());
        for &s in &output {
            assert!(
                s.is_finite(),
                "Infinity should be sanitized to zero, got {s}"
            );
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
    fn test_resample_mixed_nan_normal() {
        let mut input = vec![0.5_f32; 480];
        input[100] = f32::NAN;
        input[200] = f32::NEG_INFINITY;
        let output = resample(&input, SampleRate(48000), SampleRate(16000)).unwrap();
        assert!(!output.is_empty());
        for &s in &output {
            assert!(s.is_finite(), "Non-finite values should be sanitized");
        }
    }

    #[test]
    fn test_prepare_buffer_empty_input() {
        let mut buffer = vec![1.0; 100];
        let result = prepare_audio_buffer(&[], &mut buffer);
        // Empty new samples: buffer should retain its contents
        assert!(result.is_none());
        assert_eq!(buffer.len(), 100);
    }

    #[test]
    fn test_prepare_buffer_exactly_max() {
        // Exactly MAX_BUFFER_SAMPLES — should not trigger truncation warning
        let new_samples = vec![1.0; MAX_BUFFER_SAMPLES];
        let mut buffer = Vec::new();
        let result = prepare_audio_buffer(&new_samples, &mut buffer);
        assert!(result.is_some());
        let usable = result.unwrap();
        consume_audio_buffer(&mut buffer, usable);
        assert!(usable + buffer.len() <= MAX_BUFFER_SAMPLES);
    }

    #[test]
    fn test_prepare_buffer_one_over_max() {
        // MAX_BUFFER_SAMPLES + 1 — triggers truncation
        let new_samples = vec![1.0; MAX_BUFFER_SAMPLES + 1];
        let mut buffer = Vec::new();
        let result = prepare_audio_buffer(&new_samples, &mut buffer);
        assert!(result.is_some());
        let usable = result.unwrap();
        consume_audio_buffer(&mut buffer, usable);
        assert!(usable + buffer.len() <= MAX_BUFFER_SAMPLES);
    }

    // --- decode_audio_bytes tests ---

    pub(super) fn make_wav_bytes(samples: &[i16], sample_rate: u32) -> Vec<u8> {
        let data_size = (samples.len() * 2) as u32;
        let file_size = 36 + data_size;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&file_size.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes()); // chunk size
        buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
        buf.extend_from_slice(&1u16.to_le_bytes()); // mono
        buf.extend_from_slice(&sample_rate.to_le_bytes());
        buf.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
        buf.extend_from_slice(&2u16.to_le_bytes()); // block align
        buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_size.to_le_bytes());
        for &s in samples {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        buf
    }

    #[test]
    fn test_decode_audio_bytes_empty() {
        // Empty slice must return an error, not panic
        let result = decode_audio_bytes(&[]);
        assert!(result.is_err(), "Expected error for empty input, got Ok");
    }

    #[test]
    fn test_decode_audio_bytes_invalid_data() {
        // Random bytes that are not a valid audio file must return an error, not panic
        let garbage: Vec<u8> = (0u8..128).collect();
        let result = decode_audio_bytes(&garbage);
        assert!(
            result.is_err(),
            "Expected error for invalid audio data, got Ok"
        );
    }

    #[test]
    fn test_decode_audio_bytes_ape_overflow_crash_is_graceful() {
        // Regression: a crafted 36-byte APEv2 tag header (APE tags can ride on
        // MP3 uploads) sets an unbounded `size` field that made
        // symphonia-metadata's `size + 32` overflow and panic with "attempt to
        // add with overflow" (ape.rs). The vendored overflow-guard patch
        // saturates instead, so decode must return a graceful `Err` — never
        // panic. Fixture is the exact fuzz artifact that reddened the soak run.
        let crash = include_bytes!("../../tests/fixtures/ape_overflow_crash.bin");
        assert_eq!(crash.len(), 36, "fixture must stay the 36-byte crash input");
        let result = decode_audio_bytes(crash);
        assert!(
            result.is_err(),
            "crafted APEv2 header must yield a decode error, not panic or Ok"
        );
    }

    #[test]
    fn test_decode_audio_bytes_wav() {
        let silence: Vec<i16> = vec![0; 16000]; // 1 second at 16kHz
        let wav = make_wav_bytes(&silence, 16000);
        let samples = decode_audio_bytes(&wav).unwrap();
        assert!(!samples.is_empty());
        // Should be ~16000 samples (1 second at 16kHz)
        assert!((samples.len() as i64 - 16000).unsigned_abs() <= 100);
    }

    // --- BytesMediaSource tests ---

    use std::io::{Read, Seek, SeekFrom};

    #[test]
    fn bytes_media_source_read_full() {
        let data = Bytes::from_static(b"hello world");
        let mut src = BytesMediaSource::new(data.clone());
        let mut buf = vec![0u8; data.len()];
        let n = src.read(&mut buf).unwrap();
        assert_eq!(n, data.len());
        assert_eq!(buf, data.as_ref());
        // Next read returns 0 (EOF).
        let mut more = [0u8; 4];
        assert_eq!(src.read(&mut more).unwrap(), 0);
    }

    #[test]
    fn bytes_media_source_seek_end() {
        let data = Bytes::from_static(b"abcdefgh");
        let mut src = BytesMediaSource::new(data);
        let pos = src.seek(SeekFrom::End(0)).unwrap();
        assert_eq!(pos, 8);
        let mut buf = [0u8; 4];
        // Reading at EOF returns 0 bytes.
        assert_eq!(src.read(&mut buf).unwrap(), 0);
    }

    #[test]
    fn bytes_media_source_seek_past_end_ok() {
        let data = Bytes::from_static(b"abc");
        let mut src = BytesMediaSource::new(data);
        // std::io::Seek explicitly allows seeking past the end; the next read
        // returns 0. We mirror that behavior so symphonia's seek-then-read
        // dance on truncated files doesn't panic.
        let pos = src.seek(SeekFrom::Start(42)).unwrap();
        assert_eq!(pos, 42);
        let mut buf = [0u8; 4];
        assert_eq!(src.read(&mut buf).unwrap(), 0);
    }

    #[test]
    fn bytes_media_source_seek_before_start_err() {
        let data = Bytes::from_static(b"abc");
        let mut src = BytesMediaSource::new(data);
        let err = src.seek(SeekFrom::Start(2)).unwrap();
        assert_eq!(err, 2);
        // Relative seek that would land before byte 0 is an InvalidInput error.
        let result = src.seek(SeekFrom::Current(-100));
        assert!(result.is_err(), "seek before start should error");
    }

    #[test]
    fn bytes_media_source_partial_read_progress() {
        // Multiple partial reads must advance the cursor and stitch back to
        // the full buffer — protects against an off-by-one in the read loop.
        let data = Bytes::from_static(b"abcdefghij");
        let mut src = BytesMediaSource::new(data.clone());
        let mut out = Vec::new();
        let mut chunk = [0u8; 3];
        loop {
            let n = src.read(&mut chunk).unwrap();
            if n == 0 {
                break;
            }
            out.extend_from_slice(&chunk[..n]);
        }
        assert_eq!(out, data.as_ref());
    }

    #[test]
    fn bytes_media_source_byte_len_matches() {
        use symphonia::core::io::MediaSource as _;
        let data = Bytes::from_static(b"0123456789");
        let src = BytesMediaSource::new(data.clone());
        assert_eq!(src.byte_len(), Some(data.len() as u64));
        assert!(src.is_seekable());
    }

    // --- decode_audio_bytes_shared tests ---

    #[test]
    fn decode_audio_shim_matches_shared() {
        // Equivalence oracle: the &[u8] shim and the Bytes entry point must
        // produce byte-identical sample vectors for the same input. Protects
        // against the shim drifting from the shared implementation.
        let silence: Vec<i16> = vec![0; 16000];
        let wav = make_wav_bytes(&silence, 16000);
        let via_shim = decode_audio_bytes(&wav).unwrap();
        let via_shared = decode_audio_bytes_shared(Bytes::copy_from_slice(&wav)).unwrap();
        assert_eq!(via_shim.len(), via_shared.len());
        for (a, b) in via_shim.iter().zip(via_shared.iter()) {
            assert!((a - b).abs() < f32::EPSILON);
        }
    }

    // --- parse_pcm16_with_carry tests ---

    #[test]
    fn test_parse_pcm16_basic() {
        let data: &[u8] = &[0x00, 0x40, 0x00, 0xC0]; // two i16 samples: 16384, -16384
        let mut pending: Option<u8> = None;
        let samples = parse_pcm16_with_carry(data, &mut pending);
        assert_eq!(samples.len(), 2);
        assert!(pending.is_none());
        assert!((samples[0] - 0.5).abs() < 0.001);
        assert!((samples[1] + 0.5).abs() < 0.001);
    }

    #[test]
    fn test_parse_pcm16_odd_length_carry() {
        let mut pending: Option<u8> = None;
        let samples = parse_pcm16_with_carry(&[0x00, 0x00, 0xFF], &mut pending);
        assert_eq!(samples.len(), 1);
        assert_eq!(pending, Some(0xFF));

        let samples = parse_pcm16_with_carry(&[0x7F], &mut pending);
        assert_eq!(samples.len(), 1);
        assert!(pending.is_none());
    }

    #[test]
    fn test_parse_pcm16_empty() {
        let mut pending: Option<u8> = None;
        let samples = parse_pcm16_with_carry(&[], &mut pending);
        assert!(samples.is_empty());
        assert!(pending.is_none());
    }

    #[test]
    fn test_decode_duration_cap_pure() {
        // Pure cap math (testable without realizing a multi-minute PCM buffer):
        // the sample budget scales with the clamped rate and the duration cap.
        let budget_16k = max_decode_samples(16000);
        // 30-min cap at 16kHz => 1800 * 16000 samples.
        assert_eq!(budget_16k, 1800 * 16000);
        // 12 minutes (the old reject point) is comfortably under budget.
        assert!(12 * 60 * 16000 < budget_16k, "12-minute file must pass");
        // >30 min is over budget and would be rejected.
        assert!(
            (30 * 60 + 1) * 16000 > budget_16k,
            ">30min must exceed budget"
        );
        // Header rate is clamped: a crafted 192kHz header can't inflate the
        // budget past the 48kHz ceiling.
        assert_eq!(max_decode_samples(192_000), max_decode_samples(48_000));
    }

    // --- SampleRate tests ---

    #[test]
    fn test_sample_rate_new_zero_errors() {
        let result = SampleRate::new(0);
        assert!(result.is_err(), "zero sample rate must error");
    }

    #[test]
    fn test_sample_rate_new_positive_ok() {
        let sr = SampleRate::new(16000).unwrap();
        assert_eq!(sr.get(), 16000);
        assert_eq!(sr.0, 16000);
    }

    // --- stereo WAV helper + multi-channel mixing tests ---

    fn make_stereo_wav_from_frames(frames: &[(i16, i16)], sample_rate: u32) -> Vec<u8> {
        let data_size = (frames.len() * 4) as u32; // 2 channels * 2 bytes
        let file_size = 36 + data_size;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&file_size.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes()); // chunk size
        buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
        buf.extend_from_slice(&2u16.to_le_bytes()); // stereo
        buf.extend_from_slice(&sample_rate.to_le_bytes());
        buf.extend_from_slice(&(sample_rate * 4).to_le_bytes()); // byte rate
        buf.extend_from_slice(&4u16.to_le_bytes()); // block align
        buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_size.to_le_bytes());
        for &(l, r) in frames {
            buf.extend_from_slice(&l.to_le_bytes());
            buf.extend_from_slice(&r.to_le_bytes());
        }
        buf
    }

    fn make_stereo_wav_bytes(left: &[i16], right: &[i16], sample_rate: u32) -> Vec<u8> {
        assert_eq!(left.len(), right.len());
        let num_samples = left.len();
        let data_size = (num_samples * 4) as u32; // 2 channels * 2 bytes
        let file_size = 36 + data_size;
        let mut buf = Vec::with_capacity(file_size as usize);
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&file_size.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes()); // chunk size
        buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
        buf.extend_from_slice(&2u16.to_le_bytes()); // stereo
        buf.extend_from_slice(&sample_rate.to_le_bytes());
        buf.extend_from_slice(&(sample_rate * 4).to_le_bytes()); // byte rate
        buf.extend_from_slice(&4u16.to_le_bytes()); // block align
        buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_size.to_le_bytes());
        for i in 0..num_samples {
            buf.extend_from_slice(&left[i].to_le_bytes());
            buf.extend_from_slice(&right[i].to_le_bytes());
        }
        buf
    }

    #[test]
    fn test_decode_stereo_mixes_to_mono() {
        // Left = +16384 (0.5), Right = -16384 (-0.5) → mono average ≈ 0.0.
        // Exercises the multi-channel mixing branch in decode_audio_inner.
        let frames: Vec<(i16, i16)> = vec![(16384, -16384); 16000];
        let wav = make_stereo_wav_from_frames(&frames, 16000);
        let samples = decode_audio_bytes(&wav).unwrap();
        assert!(!samples.is_empty());
        // Output is mono (one sample per frame), not interleaved.
        assert!((samples.len() as i64 - 16000).unsigned_abs() <= 100);
        // The L/R cancel: each mono sample is ~0.0.
        for &s in &samples {
            assert!(s.abs() < 0.01, "stereo mix should cancel to ~0, got {s}");
        }
    }

    #[test]
    fn test_decode_stereo_constant_preserves_value() {
        // Both channels carry the same value → mono mix preserves it.
        let frames: Vec<(i16, i16)> = vec![(8192, 8192); 8000];
        let wav = make_stereo_wav_from_frames(&frames, 16000);
        let samples = decode_audio_bytes(&wav).unwrap();
        assert!(!samples.is_empty());
        for &s in &samples {
            assert!((s - 0.25).abs() < 0.01, "expected ~0.25, got {s}");
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
    fn test_decode_wav_resamples_to_16k() {
        // 48kHz mono WAV exercises the n_frames_hint capacity reservation and
        // the post-decode resample-to-16kHz branch.
        let silence: Vec<i16> = vec![0; 48000]; // 1 second at 48kHz
        let wav = make_wav_bytes(&silence, 48000);
        let samples = decode_audio_bytes(&wav).unwrap();
        assert!(!samples.is_empty());
        // Resampled to 16kHz → ~16000 samples (rubato FIR delay shortens it).
        assert!(
            samples.len() > 14000 && samples.len() < 17000,
            "expected ~16000 after resample, got {}",
            samples.len()
        );
    }

    #[test]
    fn test_decode_audio_bytes_shared_channels_8khz() {
        let sample_rate = 8000u32;
        let num_samples = sample_rate as usize;
        let left: Vec<i16> = (0..num_samples)
            .map(|i| ((i as f32 / num_samples as f32) * 6000.0) as i16)
            .collect();
        let right: Vec<i16> = (0..num_samples)
            .map(|i| ((1.0 - i as f32 / num_samples as f32) * 6000.0) as i16)
            .collect();
        let wav = make_stereo_wav_bytes(&left, &right, sample_rate);
        let channels = decode_audio_bytes_shared_channels(Bytes::from(wav)).unwrap();
        assert_eq!(channels.len(), 2);
        // Resampled to 16 kHz: expect roughly twice the length (allow FIR delay slack).
        assert!(
            channels[0].len() > num_samples * 15 / 10 && channels[0].len() < num_samples * 25 / 10
        );
        assert!(
            channels[1].len() > num_samples * 15 / 10 && channels[1].len() < num_samples * 25 / 10
        );
        // Channels should differ once the FIR resampler has passed its delay.
        assert!((channels[0][1000] - channels[1][1000]).abs() > 0.01);
    }

    #[test]
    fn test_is_dual_mono_identical_channels() {
        let samples: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
        assert!(is_dual_mono(&[samples.clone(), samples]));
    }

    #[test]
    fn test_is_dual_mono_independent_channels() {
        let left: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
        let right: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.03).cos()).collect();
        assert!(!is_dual_mono(&[left, right]));
    }

    #[test]
    fn test_mix_channels_to_mono() {
        let left = vec![1.0_f32];
        let right = vec![-1.0_f32];
        let mono = mix_channels_to_mono(&[left, right]);
        assert_eq!(mono.len(), 1);
        assert!(mono[0].abs() < 0.001);
    }

    #[test]
    fn test_is_dual_mono_empty_channels_returns_false() {
        assert!(!is_dual_mono(&[]));
    }

    #[test]
    fn test_is_dual_mono_single_channel_returns_false() {
        let samples: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01).sin()).collect();
        assert!(!is_dual_mono(&[samples]));
    }

    #[test]
    fn test_mix_channels_to_mono_empty_input() {
        let mono = mix_channels_to_mono(&[]);
        assert!(mono.is_empty());
    }

    #[test]
    fn test_decode_audio_bytes_shared_channels_mono_input() {
        // A mono WAV fed through the split decoder must return exactly one
        // channel whose samples match the regular mono decode path.
        let samples: Vec<i16> = (0..8000).map(|i| (i as f32 * 0.1).sin() as i16).collect();
        let wav = make_wav_bytes(&samples, 16000);
        let mono = decode_audio_bytes(&wav).unwrap();
        let channels = decode_audio_bytes_shared_channels(Bytes::copy_from_slice(&wav)).unwrap();
        assert_eq!(channels.len(), 1);
        assert_eq!(channels[0].len(), mono.len());
        for (a, b) in channels[0].iter().zip(mono.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "split mono decode diverged: {a} vs {b}"
            );
        }
    }

    // --- resample_with_cache tests ---

    #[test]
    fn test_resample_with_cache_empty_clears_buffer() {
        let mut cache: Option<rubato::Async<f32>> = None;
        let mut out = vec![1.0, 2.0, 3.0];
        resample_with_cache(
            Vec::new(),
            SampleRate(48000),
            SampleRate(16000),
            &mut cache,
            &mut out,
        )
        .unwrap();
        assert!(out.is_empty(), "empty input must clear the output buffer");
        assert!(cache.is_none(), "no resampler created for empty input");
    }

    #[test]
    fn test_resample_with_cache_zero_rate_clears_buffer() {
        let mut cache: Option<rubato::Async<f32>> = None;
        let mut out = vec![9.0];
        resample_with_cache(
            vec![1.0, 2.0],
            SampleRate(0),
            SampleRate(16000),
            &mut cache,
            &mut out,
        )
        .unwrap();
        assert!(out.is_empty());
        let mut out2 = vec![9.0];
        resample_with_cache(
            vec![1.0, 2.0],
            SampleRate(16000),
            SampleRate(0),
            &mut cache,
            &mut out2,
        )
        .unwrap();
        assert!(out2.is_empty());
    }

    #[test]
    fn test_resample_with_cache_same_rate_passthrough() {
        let mut cache: Option<rubato::Async<f32>> = None;
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = Vec::new();
        resample_with_cache(
            input.clone(),
            SampleRate(16000),
            SampleRate(16000),
            &mut cache,
            &mut out,
        )
        .unwrap();
        assert_eq!(out, input, "same rate must pass through unchanged");
        assert!(
            cache.is_none(),
            "no resampler created for same-rate passthrough"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
    fn test_resample_with_cache_sanitizes_non_finite() {
        let mut cache: Option<rubato::Async<f32>> = None;
        let mut input = vec![0.5_f32; 480];
        input[10] = f32::NAN;
        input[20] = f32::INFINITY;
        input[30] = f32::NEG_INFINITY;
        let mut out = Vec::new();
        resample_with_cache(
            input,
            SampleRate(48000),
            SampleRate(16000),
            &mut cache,
            &mut out,
        )
        .unwrap();
        assert!(!out.is_empty());
        assert!(
            cache.is_some(),
            "resampler should be cached after first use"
        );
        for &s in &out {
            assert!(
                s.is_finite(),
                "non-finite values must be sanitized, got {s}"
            );
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
    fn test_resample_with_cache_growing_chunks_match_one_shot() {
        use std::f32::consts::PI;

        // 1 s of a continuous two-tone signal at 48 kHz, continuous across the
        // whole stream so any seam glitch shows up against the reference.
        let n = 48_000usize;
        let signal: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / 48_000.0;
                0.5 * (2.0 * PI * 440.0 * t).sin() + 0.3 * (2.0 * PI * 1_200.0 * t).sin()
            })
            .collect();

        // Reference: one-shot resample of the whole signal in a single call.
        let reference = resample(&signal, SampleRate(48_000), SampleRate(16_000)).unwrap();

        // Stream the same signal in strictly growing frames (10 ms @ 48 kHz,
        // +10 ms per frame). Every growth step used to recreate the resampler,
        // resetting its FIR history and fractional phase at each seam.
        let mut cache: Option<rubato::Async<f32>> = None;
        let mut out = Vec::new();
        let mut streamed = Vec::new();
        let mut pos = 0usize;
        let mut chunk = 480usize;
        while pos < signal.len() {
            let end = (pos + chunk).min(signal.len());
            resample_with_cache(
                signal[pos..end].to_vec(),
                SampleRate(48_000),
                SampleRate(16_000),
                &mut cache,
                &mut out,
            )
            .unwrap();
            streamed.extend_from_slice(&out);
            pos = end;
            chunk += 480;
        }
        assert!(streamed.iter().all(|s| s.is_finite()));

        // A recreated resampler drops the output-delay tail (~85 samples at
        // 3:1) per recreation, so the streamed length collapses vs one-shot.
        let len_diff = reference.len().abs_diff(streamed.len());
        assert!(
            len_diff <= 2,
            "chunked stream diverged from one-shot reference: {} vs {} samples",
            streamed.len(),
            reference.len()
        );

        // Beyond the initial sinc transient (~sinc_len/2 * 1/3 ≈ 43 samples)
        // the streamed output must track the one-shot reference closely; a
        // seam discontinuity (FIR reset fade-in) shows up as a large spike.
        let skip = 128;
        let cmp_len = reference.len().min(streamed.len());
        assert!(cmp_len > skip + 1_000, "not enough overlap to compare");
        let mut max_diff = 0.0f32;
        let mut max_at = 0usize;
        for i in skip..cmp_len {
            let d = (reference[i] - streamed[i]).abs();
            if d > max_diff {
                max_diff = d;
                max_at = i;
            }
        }
        assert!(
            max_diff < 1e-3,
            "seam discontinuity: max |streamed - reference| = {max_diff} at sample {max_at}"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
    fn test_resample_with_cache_growth_keeps_instance() {
        let mut cache: Option<rubato::Async<f32>> = None;
        let mut out = Vec::new();
        let feed =
            |cache: &mut Option<rubato::Async<f32>>, out: &mut Vec<f32>, n: usize, seed: f32| {
                let input: Vec<f32> = (0..n).map(|i| (i as f32 * seed).sin()).collect();
                resample_with_cache(input, SampleRate(48_000), SampleRate(16_000), cache, out)
                    .unwrap();
            };

        // First frame fixes the resampler capacity.
        feed(&mut cache, &mut out, 480, 0.01);
        let capacity = cache.as_ref().unwrap().input_frames_max();
        assert!(capacity >= 480);

        // Growing frames must NOT change the capacity: a change means the
        // resampler was recreated and its FIR state was lost.
        feed(&mut cache, &mut out, 960, 0.02);
        assert_eq!(
            cache.as_ref().unwrap().input_frames_max(),
            capacity,
            "resampler recreated on frame growth"
        );
        feed(&mut cache, &mut out, 2_000, 0.03);
        assert_eq!(cache.as_ref().unwrap().input_frames_max(), capacity);

        // A frame larger than the initial capacity must also survive without
        // recreation (fed through in capacity-sized pieces).
        feed(&mut cache, &mut out, capacity + 1_001, 0.01);
        assert_eq!(
            cache.as_ref().unwrap().input_frames_max(),
            capacity,
            "oversized frame must be split, not trigger recreation"
        );
        assert!(out.iter().all(|s| s.is_finite()));

        // A frame one sample over capacity splits into a full piece plus a
        // 1-sample remainder (which defers its output via the fractional
        // phase); this must succeed and keep the instance.
        feed(&mut cache, &mut out, capacity + 1, 0.02);
        assert_eq!(cache.as_ref().unwrap().input_frames_max(), capacity);
        assert!(out.iter().all(|s| s.is_finite()));
    }

    #[test]
    #[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
    fn test_resample_with_cache_reuses_across_chunk_sizes() {
        let mut cache: Option<rubato::Async<f32>> = None;
        let mut out = Vec::new();
        // First call creates the resampler.
        let input1: Vec<f32> = (0..480).map(|i| (i as f32 * 0.01).sin()).collect();
        resample_with_cache(
            input1,
            SampleRate(48000),
            SampleRate(16000),
            &mut cache,
            &mut out,
        )
        .unwrap();
        assert!(cache.is_some());
        let len_first = out.len();
        assert!(len_first > 0);

        // Second call with the SAME chunk size reuses the cached resampler.
        let input2: Vec<f32> = (0..480).map(|i| (i as f32 * 0.02).cos()).collect();
        resample_with_cache(
            input2,
            SampleRate(48000),
            SampleRate(16000),
            &mut cache,
            &mut out,
        )
        .unwrap();
        assert!(cache.is_some());
        assert!(!out.is_empty());

        // Third call with a DIFFERENT chunk size resizes in place — the
        // resampler is never recreated, so its FIR state survives.
        let input3: Vec<f32> = (0..960).map(|i| (i as f32 * 0.01).sin()).collect();
        resample_with_cache(
            input3,
            SampleRate(48000),
            SampleRate(16000),
            &mut cache,
            &mut out,
        )
        .unwrap();
        assert!(cache.is_some());
        assert!(!out.is_empty());
        for &s in &out {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn test_decode_rejects_adversarial_sample_rate() {
        // A crafted header with an out-of-range sample rate must be rejected
        // before it can scale the duration cap or trigger an oversized
        // reservation — and must never panic.
        let silence: Vec<i16> = vec![0; 16]; // tiny payload — the header is the attack
        // Just above the ceiling: a well-formed header that the clamp must reject.
        let result = decode_audio_bytes(&make_wav_bytes(&silence, MAX_SAMPLE_RATE + 1));
        assert!(
            result.is_err(),
            "sample_rate above MAX_SAMPLE_RATE must be rejected"
        );
        // A grossly inflated rate must also be rejected (not panic / not allocate).
        let result = decode_audio_bytes(&make_wav_bytes(&silence, 1_000_000_000));
        assert!(result.is_err(), "absurd sample_rate must be rejected");
    }

    // --- telephony codecs: G.711 / G.722 ---

    /// Build a WAV buffer with an arbitrary format tag around an encoded
    /// payload (mono). The `fmt ` chunk carries the 2-byte `cbSize` extension
    /// field (18 bytes total) because symphonia rejects 16-byte `fmt ` chunks
    /// for the G.711 tags — and it is what ffmpeg writes for all of these.
    fn make_compressed_wav(tag: u16, sample_rate: u32, byte_rate: u32, payload: &[u8]) -> Vec<u8> {
        let data_size = payload.len() as u32;
        let mut buf = Vec::with_capacity(46 + payload.len());
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&(38 + data_size).to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&18u32.to_le_bytes()); // fmt chunk size (incl. cbSize)
        buf.extend_from_slice(&tag.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // mono
        buf.extend_from_slice(&sample_rate.to_le_bytes());
        buf.extend_from_slice(&byte_rate.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // block align
        buf.extend_from_slice(&8u16.to_le_bytes()); // bits per sample
        buf.extend_from_slice(&0u16.to_le_bytes()); // cbSize = 0
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_size.to_le_bytes());
        buf.extend_from_slice(payload);
        buf
    }

    fn test_tone_8k(n_samples: usize) -> Vec<i16> {
        (0..n_samples)
            .map(|i| ((i as f32 * 0.05).sin() * 12000.0) as i16)
            .collect()
    }

    #[test]
    fn test_telephony_codec_from_name() {
        assert_eq!(
            TelephonyCodec::from_name("pcmu"),
            Some(TelephonyCodec::Pcmu)
        );
        assert_eq!(
            TelephonyCodec::from_name("PCMU"),
            Some(TelephonyCodec::Pcmu)
        );
        assert_eq!(
            TelephonyCodec::from_name("ulaw"),
            Some(TelephonyCodec::Pcmu)
        );
        assert_eq!(
            TelephonyCodec::from_name("pcma"),
            Some(TelephonyCodec::Pcma)
        );
        assert_eq!(
            TelephonyCodec::from_name("alaw"),
            Some(TelephonyCodec::Pcma)
        );
        assert_eq!(
            TelephonyCodec::from_name("G722"),
            Some(TelephonyCodec::G722)
        );
        assert_eq!(TelephonyCodec::from_name("g729"), None);
        assert_eq!(TelephonyCodec::from_name(""), None);
    }

    #[test]
    fn test_telephony_codec_validate_sample_rate() {
        assert!(TelephonyCodec::Pcmu.validate_sample_rate(8000).is_ok());
        assert!(TelephonyCodec::Pcma.validate_sample_rate(16000).is_ok());
        assert!(TelephonyCodec::Pcma.validate_sample_rate(48000).is_ok());
        assert!(TelephonyCodec::Pcmu.validate_sample_rate(7999).is_err());
        assert!(TelephonyCodec::Pcma.validate_sample_rate(48001).is_err());
        // G.722 decodes to 16 kHz natively; 8000 is the SDP clock-rate alias.
        assert!(TelephonyCodec::G722.validate_sample_rate(8000).is_ok());
        assert!(TelephonyCodec::G722.validate_sample_rate(16000).is_ok());
        assert!(TelephonyCodec::G722.validate_sample_rate(44100).is_err());
    }

    #[test]
    #[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
    fn test_decode_telephony_raw_pcmu_roundtrip() {
        let source = test_tone_8k(8000);
        let mut encoder = audio_codec::pcmu::PcmuEncoder::new();
        let encoded = audio_codec::Encoder::encode(&mut encoder, &source);
        assert_eq!(encoded.len(), source.len(), "G.711 is one byte per sample");
        let decoded = decode_telephony_raw(&encoded, TelephonyCodec::Pcmu, 8000).unwrap();
        // Resampled 8k → 16k: roughly double, minus the FIR delay slack.
        assert!(
            decoded.len() > 12_000 && decoded.len() <= 16_000,
            "unexpected decoded length {}",
            decoded.len()
        );
        // G.711 is lossy but near-transparent: compare against the source
        // (resampled) with a loose bound instead of the raw encoded bytes.
        let expected = resample(
            &source
                .iter()
                .map(|&s| f32::from(s) / 32768.0)
                .collect::<Vec<_>>(),
            SampleRate(8000),
            SampleRate(16000),
        )
        .unwrap();
        let n = decoded.len().min(expected.len());
        let mse: f64 = decoded[..n]
            .iter()
            .zip(&expected[..n])
            .map(|(a, b)| f64::from((a - b) * (a - b)))
            .sum::<f64>()
            / n as f64;
        assert!(
            mse.sqrt() < 0.02,
            "G.711 μ-law roundtrip RMSE {}",
            mse.sqrt()
        );
    }

    #[test]
    #[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
    fn test_decode_telephony_raw_pcma_roundtrip() {
        let source = test_tone_8k(8000);
        let mut encoder = audio_codec::pcma::PcmaEncoder::new();
        let encoded = audio_codec::Encoder::encode(&mut encoder, &source);
        let decoded = decode_telephony_raw(&encoded, TelephonyCodec::Pcma, 8000).unwrap();
        assert!(decoded.len() > 12_000 && decoded.len() <= 16_000);
        assert!(decoded.iter().all(|s| s.is_finite()));
    }

    /// RMSE between two equal-rate signals at the best integer lag within
    /// ±`max_lag` samples. Lossy codecs carry an inherent group delay (the
    /// G.722 QMF bank), so a fixed-alignment RMSE would report the delay as
    /// distortion instead of measuring actual reconstruction error.
    fn best_lag_rmse(a: &[f32], b: &[f32], max_lag: usize) -> f64 {
        let mut best = f64::INFINITY;
        for lag in 0..=max_lag {
            for (a_slice, b_slice) in [
                (a.get(lag..).unwrap_or(&[]), b),
                (a, b.get(lag..).unwrap_or(&[])),
            ] {
                let n = a_slice.len().min(b_slice.len());
                if n < 100 {
                    continue;
                }
                let mse = a_slice[..n]
                    .iter()
                    .zip(&b_slice[..n])
                    .map(|(x, y)| {
                        let d = f64::from(x - y);
                        d * d
                    })
                    .sum::<f64>()
                    / n as f64;
                best = best.min(mse.sqrt());
            }
        }
        best
    }

    #[test]
    fn test_decode_telephony_raw_g722_roundtrip() {
        // 1 s of 16 kHz tone; G.722 output stays at its native 16 kHz.
        let source: Vec<i16> = (0..16000)
            .map(|i| ((i as f32 * 0.03).sin() * 10000.0) as i16)
            .collect();
        let mut encoder = audio_codec::g722::G722Encoder::new();
        let encoded = audio_codec::Encoder::encode(&mut encoder, &source);
        assert_eq!(encoded.len(), source.len() / 2, "64 kbit/s over 16 kHz");
        let decoded = decode_telephony_raw(&encoded, TelephonyCodec::G722, 8000).unwrap();
        assert_eq!(decoded.len(), source.len(), "G.722 stays at native 16 kHz");
        // ADPCM roundtrip: compare against the source at the best lag (the
        // codec's QMF bank delays the output by a few samples).
        let source_f32: Vec<f32> = source.iter().map(|&s| f32::from(s) / 32768.0).collect();
        let rmse = best_lag_rmse(&decoded, &source_f32, 64);
        assert!(rmse < 0.05, "G.722 roundtrip best-lag RMSE {rmse}");
    }

    #[test]
    fn test_decode_telephony_raw_empty_errors() {
        assert!(decode_telephony_raw(&[], TelephonyCodec::Pcmu, 8000).is_err());
        assert!(decode_telephony_raw(&[], TelephonyCodec::G722, 16000).is_err());
    }

    #[test]
    fn test_decode_telephony_raw_invalid_rate_errors() {
        let payload = vec![0xFFu8; 160];
        assert!(decode_telephony_raw(&payload, TelephonyCodec::Pcmu, 4000).is_err());
        assert!(decode_telephony_raw(&payload, TelephonyCodec::G722, 44100).is_err());
    }

    #[test]
    #[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
    fn test_decode_audio_bytes_g711_alaw_wav() {
        // G.711 A-law in WAV (tag 0x0006) is decoded by symphonia's PCM codec —
        // this pins the de-facto support so it cannot silently regress.
        let source = test_tone_8k(8000);
        let mut encoder = audio_codec::pcma::PcmaEncoder::new();
        let encoded = audio_codec::Encoder::encode(&mut encoder, &source);
        let wav = make_compressed_wav(0x0006, 8000, 8000, &encoded);
        let decoded = decode_audio_bytes(&wav).unwrap();
        assert!(
            decoded.len() > 12_000 && decoded.len() <= 16_000,
            "unexpected decoded length {}",
            decoded.len()
        );
        assert!(decoded.iter().all(|s| s.is_finite()));
    }

    #[test]
    #[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
    fn test_decode_audio_bytes_g711_mulaw_wav() {
        // G.711 μ-law in WAV (tag 0x0007), same symphonia PCM path.
        let source = test_tone_8k(8000);
        let mut encoder = audio_codec::pcmu::PcmuEncoder::new();
        let encoded = audio_codec::Encoder::encode(&mut encoder, &source);
        let wav = make_compressed_wav(0x0007, 8000, 8000, &encoded);
        let decoded = decode_audio_bytes(&wav).unwrap();
        assert!(
            decoded.len() > 12_000 && decoded.len() <= 16_000,
            "unexpected decoded length {}",
            decoded.len()
        );
        assert!(decoded.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn test_decode_audio_bytes_g722_wav_fallback() {
        // G.722-in-WAV (tag 0x0064) has no symphonia decoder; the fallback must
        // kick in and produce 2 samples per encoded byte at native 16 kHz.
        let source: Vec<i16> = (0..16000)
            .map(|i| ((i as f32 * 0.03).sin() * 10000.0) as i16)
            .collect();
        let mut encoder = audio_codec::g722::G722Encoder::new();
        let encoded = audio_codec::Encoder::encode(&mut encoder, &source);
        for tag in [0x0064u16, 0x028F] {
            let wav = make_compressed_wav(tag, 16000, 8000, &encoded);
            let decoded = decode_audio_bytes(&wav).unwrap_or_else(|e| {
                panic!("G.722 WAV (tag {tag:#06x}) must decode via the fallback: {e}")
            });
            assert_eq!(
                decoded.len(),
                source.len(),
                "G.722 WAV must decode to native 16 kHz (tag {tag:#06x})"
            );
        }
    }

    #[test]
    fn test_try_decode_g722_wav_malformed_inputs() {
        // Not RIFF at all → None (falls through to symphonia).
        assert!(try_decode_g722_wav(b"not a wave file").is_none());
        // PCM WAV → None (symphonia handles it).
        let pcm_wav = make_wav_bytes(&[0i16; 32], 16000);
        assert!(try_decode_g722_wav(&pcm_wav).is_none());
        // G.722 tag but no data chunk → Some(Err), not a panic or silent None.
        let mut header_only = make_compressed_wav(0x0064, 16000, 8000, &[]);
        header_only.truncate(38); // strip the data chunk header + payload
        let result = try_decode_g722_wav(&header_only);
        assert!(
            matches!(result, Some(Err(_))),
            "expected Some(Err), got {result:?}"
        );
        // Truncated data payload must decode the bytes present, not panic.
        let mut enc = audio_codec::g722::G722Encoder::new();
        let encoded = audio_codec::Encoder::encode(&mut enc, &[0i16; 320]);
        let mut wav = make_compressed_wav(0x0064, 16000, 8000, &encoded);
        wav.truncate(wav.len() - 3);
        let result = try_decode_g722_wav(&wav);
        assert!(
            matches!(result, Some(Ok(_))),
            "truncated data must not panic"
        );
    }

    #[test]
    fn test_decode_audio_bytes_g722_wav_ffmpeg_fixture_matches_reference() {
        // Independent-reference verification: `g722_tone.wav` was ENCODED by
        // ffmpeg (libavcodec G.722, tag 0x028F) and `g722_tone_ffmpeg.pcm` is
        // ffmpeg's own DECODE of it (see scripts/generate_telephony_fixtures.sh).
        // Our `audio-codec` decode is compared against ffmpeg's decode, so the
        // fixed-point port is validated against a second implementation rather
        // than against itself. Tolerance: RMSE below 1% of full scale.
        let wav = include_bytes!("../../tests/fixtures/telephony/g722_tone.wav");
        let reference_pcm = include_bytes!("../../tests/fixtures/telephony/g722_tone_ffmpeg.pcm");
        let ours = decode_audio_bytes(wav).expect("ffmpeg G.722 WAV must decode");
        let reference: Vec<f32> = reference_pcm
            .chunks_exact(2)
            .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) / 32768.0)
            .collect();
        assert_eq!(
            ours.len(),
            reference.len(),
            "sample count must match ffmpeg's decode exactly"
        );
        let mse: f64 = ours
            .iter()
            .zip(reference.iter())
            .map(|(a, b)| {
                let d = f64::from(a - b);
                d * d
            })
            .sum::<f64>()
            / ours.len() as f64;
        assert!(
            mse.sqrt() < 0.01,
            "G.722 decode diverged from ffmpeg reference: RMSE {}",
            mse.sqrt()
        );
    }

    #[test]
    fn test_encode_wav_pcm16_roundtrip() {
        let source: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.02).sin() * 0.5).collect();
        let wav = encode_wav_pcm16(&source, 16000);
        let decoded = decode_audio_bytes(&wav).unwrap();
        assert_eq!(decoded.len(), source.len());
        for (a, b) in decoded.iter().zip(source.iter()) {
            assert!((a - b).abs() < 1e-3, "PCM16 roundtrip drift: {a} vs {b}");
        }
    }

    #[test]
    fn test_encode_wav_pcm16_clamps_and_sanitizes() {
        let samples = [2.0f32, -2.0, f32::NAN, 0.5];
        let wav = encode_wav_pcm16(&samples, 16000);
        let decoded = decode_audio_bytes(&wav).unwrap();
        assert!((decoded[0] - 1.0).abs() < 1e-3, "must clamp to +1");
        assert!((decoded[1] + 1.0).abs() < 1e-3, "must clamp to -1");
        assert!(decoded[2].abs() < 1e-3, "NaN must become silence");
        assert!((decoded[3] - 0.5).abs() < 1e-3);
    }
}

// Excluded under Miri: proptest runs hundreds of cases per property, each
// driving the resampler / WAV decoder — orders of magnitude too slow under
// the Miri interpreter to finish in the nightly job's budget. The same
// properties run natively on every `cargo test`; this only trims the Miri
// coverage, not the stable-toolchain coverage.
#[cfg(all(test, not(miri)))]
mod proptests {
    use super::tests::make_wav_bytes;
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn proptest_pcm16_carry_invariant(
            chunks in proptest::collection::vec(
                proptest::collection::vec(any::<u8>(), 0..1000),
                1..20
            )
        ) {
            let mut pending: Option<u8> = None;
            let mut total_samples = 0usize;
            let mut total_bytes = 0usize;

            for chunk in &chunks {
                total_bytes += chunk.len();
                let samples = parse_pcm16_with_carry(chunk, &mut pending);
                total_samples += samples.len();
            }

            let expected = total_bytes / 2;
            prop_assert_eq!(total_samples, expected,
                "samples ({}) must equal total_bytes/2 ({})", total_samples, expected);

            if total_bytes % 2 == 1 {
                prop_assert!(pending.is_some());
            } else {
                prop_assert!(pending.is_none());
            }
        }

        #[test]
        fn proptest_resample_no_panic(
            samples in proptest::collection::vec(-1.0f32..1.0f32, 1..5_000),
            rate_idx in 0..5usize,
        ) {
            let rates = [8000u32, 16000, 24000, 44100, 48000];
            let from_rate = SampleRate(rates[rate_idx]);
            if from_rate.0 == 16000 {
                return Ok(());
            }
            let result = resample(&samples, from_rate, SampleRate(16000));
            prop_assert!(result.is_ok(), "resample failed: {:?}", result.err());
        }

        #[test]
        fn proptest_decode_header_sample_rate_never_panics(rate in 0u32..=300_000u32) {
            // Decoding a WAV with an arbitrary header sample rate must never panic;
            // any rate above the ceiling must be rejected, never accepted.
            let silence: Vec<i16> = vec![0; 8];
            let result = decode_audio_bytes(&make_wav_bytes(&silence, rate));
            if rate > MAX_SAMPLE_RATE {
                prop_assert!(result.is_err(), "rate {} above ceiling must be rejected", rate);
            }
        }
    }
}
