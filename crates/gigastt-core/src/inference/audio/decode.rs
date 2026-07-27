//! Container decode (symphonia paths), channel mix, and dual-mono detection.

#[cfg(feature = "file-decode")]
use anyhow::Context;
use anyhow::Result;
use bytes::Bytes;
#[cfg(feature = "file-decode")]
use symphonia::core::codecs::audio::AudioDecoderOptions;
#[cfg(feature = "file-decode")]
use symphonia::core::codecs::audio::well_known::CODEC_ID_OPUS;
#[cfg(feature = "file-decode")]
use symphonia::core::formats::probe::Hint;
#[cfg(feature = "file-decode")]
use symphonia::core::formats::{FormatOptions, TrackType};
#[cfg(feature = "file-decode")]
use symphonia::core::io::{MediaSource, MediaSourceStream};
#[cfg(feature = "file-decode")]
use symphonia::core::meta::MetadataOptions;

use super::DUAL_MONO_CORRELATION_THRESHOLD;
#[cfg(feature = "file-decode")]
use super::MAX_DURATION_S;
#[cfg(feature = "file-decode")]
use super::MAX_SAMPLE_RATE;
#[cfg(feature = "file-decode")]
use super::max_decode_samples;
#[cfg(feature = "file-decode")]
use super::opus::{decode_opus_channels, next_demux_packet};
#[cfg(feature = "file-decode")]
use super::resample::{RESAMPLE_STAGING_FRAMES, ResampleTo16k, SampleRate};
#[cfg(feature = "file-decode")]
use super::telephony::{sniffs_as_g722_wav, try_decode_g722_wav};

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

// docs-drift: codecs
// Canonical decode surface, one token per supported input. Kept in sync with
// the FORMATS table in scripts/check-docs-drift.py and the format lists in
// docs/api.md ("Audio formats and telephony codecs") and docs/cli.md
// ("Supports:" line) — update all three together when adding a codec.
// wav
// wav-g711
// wav-g722
// mp3
// m4a
// ogg-vorbis
// ogg-opus
// flac
// raw-pcmu
// raw-pcma
// raw-g722
// docs-drift: end

/// Decode any supported audio file to mono f32 samples at 16kHz.
///
/// Supports WAV, MP3, M4A/AAC, OGG/Vorbis, OGG/Opus (`.opus`), and FLAC.
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
/// in-memory buffer. Supports WAV, MP3, M4A/AAC, OGG/Vorbis, OGG/Opus
/// (`.opus`), and FLAC. Multi-channel audio is mixed to mono. The duration cap
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

    // Each channel is an independent stream, so each gets its own staging
    // buffer and cached resampler; none of them ever holds more than one
    // chunk of source-rate audio.
    let hint = match n_frames_hint {
        Some(n) if n > 0 && n <= max_samples as u64 => Some(n as usize),
        _ => None,
    };
    // Source-rate frame count of the first channel, tracked separately because
    // the accumulators now hold 16 kHz samples while the duration cap is
    // expressed in source-rate frames.
    let mut source_frames: usize = 0;

    // Symphonia demuxes OGG/Opus but ships no Opus decoder, so Opus packets
    // go through the `opus-rs` fallback and rejoin the shared resample tail.
    let acc: Vec<ResampleTo16k> = if audio_params.codec == CODEC_ID_OPUS {
        let decoded = decode_opus_channels(&mut *format, track_id, channels, max_samples)?;
        source_frames = decoded.first().map(|v| v.len()).unwrap_or(0);
        let mut acc = Vec::with_capacity(decoded.len());
        for samples in decoded {
            let mut chan = ResampleTo16k::new(SampleRate(sample_rate), Some(samples.len()));
            for piece in samples.chunks(RESAMPLE_STAGING_FRAMES) {
                chan.stage().extend_from_slice(piece);
                chan.flush_full()?;
            }
            acc.push(chan);
        }
        acc
    } else {
        let mut decoder = symphonia::default::get_codecs()
            .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
            .context("Unsupported audio codec")?;

        let mut acc: Vec<ResampleTo16k> = (0..channels)
            .map(|_| ResampleTo16k::new(SampleRate(sample_rate), hint))
            .collect();

        loop {
            let have_pcm = source_frames > 0;
            let Some(packet) = next_demux_packet(&mut *format, have_pcm)? else {
                break;
            };

            if packet.track_id != track_id {
                continue;
            }

            let decoded = decoder.decode(&packet).context("Decode error")?;
            let spec = decoded.spec().clone();
            let num_frames = decoded.frames();
            let ch = spec.channels().count();

            if ch > acc.len() {
                // A channel that appears mid-stream starts from this packet,
                // so it gets no length hint.
                acc.resize_with(ch, || ResampleTo16k::new(SampleRate(sample_rate), None));
            }

            if ch > 1 {
                let mut interleaved: Vec<f32> = Vec::with_capacity(num_frames * ch);
                decoded.copy_to_vec_interleaved(&mut interleaved);
                for frame in 0..num_frames {
                    for c in 0..ch {
                        acc[c].stage().push(interleaved[frame * ch + c]);
                    }
                }
            } else if !acc.is_empty() {
                let stage = acc[0].stage();
                let offset = stage.len();
                stage.resize(offset + num_frames, 0.0);
                decoded.copy_to_slice_interleaved(&mut stage[offset..]);
            }
            // Both branches above grow the first channel by `num_frames`; the
            // `ch <= 1` branch is skipped entirely when there is no channel.
            if !acc.is_empty() {
                source_frames += num_frames;
            }

            if source_frames > max_samples {
                let observed_s = source_frames as f64 / sample_rate as f64;
                anyhow::bail!(
                    "Audio file too long ({:.0}s). Maximum supported: {MAX_DURATION_S:.0}s.",
                    observed_s
                );
            }

            for chan in &mut acc {
                chan.flush_full()?;
            }
        }

        acc
    };

    let duration_s = source_frames as f64 / sample_rate as f64;
    tracing::info!(
        "Decoded {} channel(s), first channel {} samples at {}Hz ({:.1}s)",
        acc.len(),
        source_frames,
        sample_rate,
        duration_s
    );

    let channel_count = acc.len();
    let per_channel = acc
        .into_iter()
        .map(ResampleTo16k::finish)
        .collect::<Result<Vec<_>>>()?;
    if sample_rate != 16000 {
        tracing::info!("Resampled {channel_count} channel(s) to 16kHz");
    }

    Ok(per_channel)
}

/// Average multiple channels into a single mono vector.
pub fn mix_channels_to_mono(channels: &[Vec<f32>]) -> Vec<f32> {
    if channels.is_empty() {
        return Vec::new();
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

    // Sample budget from a CLAMPED rate (header `sample_rate` capped at
    // MAX_DECODE_SAMPLE_RATE), so a crafted header cannot inflate the duration
    // cap or the capacity hint. Computed before the capacity match so the hint
    // is bounded by the same budget.
    let max_samples: usize = max_decode_samples(sample_rate);

    // Decoded packets are staged at the source rate and drained to 16 kHz as
    // they fill, so peak memory is O(one staging chunk) instead of O(file).
    let mut acc = ResampleTo16k::new(
        SampleRate(sample_rate),
        match n_frames_hint {
            Some(n) if n > 0 && n <= max_samples as u64 => Some(n as usize),
            _ => None,
        },
    );
    // Source-rate frame count, tracked separately because the accumulator now
    // holds 16 kHz samples while the duration cap is expressed in source-rate
    // frames.
    let mut source_frames: usize = 0;

    // Symphonia demuxes OGG/Opus but ships no Opus decoder, so Opus packets
    // go through the `opus-rs` fallback (mono mix included) and rejoin the
    // shared duration-log / resample tail.
    if audio_params.codec == CODEC_ID_OPUS {
        let mono = mix_channels_to_mono(&decode_opus_channels(
            &mut *format,
            track_id,
            channels,
            max_samples,
        )?);
        source_frames = mono.len();
        for piece in mono.chunks(RESAMPLE_STAGING_FRAMES) {
            acc.stage().extend_from_slice(piece);
            acc.flush_full()?;
        }
    } else {
        let mut decoder = symphonia::default::get_codecs()
            .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
            .context("Unsupported audio codec")?;

        loop {
            let have_pcm = source_frames > 0;
            let Some(packet) = next_demux_packet(&mut *format, have_pcm)? else {
                break;
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
                let stage = acc.stage();
                for frame in 0..num_frames {
                    let mut sum = 0.0_f32;
                    for c in 0..ch {
                        sum += interleaved[frame * ch + c];
                    }
                    stage.push(sum / ch as f32);
                }
            } else {
                let stage = acc.stage();
                let offset = stage.len();
                stage.resize(offset + num_frames, 0.0);
                decoded.copy_to_slice_interleaved(&mut stage[offset..]);
            }
            source_frames += num_frames;

            // Incremental duration cap: abort before the next packet is decoded
            // if the accumulated buffer already exceeds the duration budget.
            // This prevents a crafted upload from allocating hundreds of MiB of
            // PCM before the post-loop guard gets a chance to run.
            if source_frames > max_samples {
                let observed_s = source_frames as f64 / sample_rate as f64;
                anyhow::bail!(
                    "Audio file too long ({:.0}s). Maximum supported: {MAX_DURATION_S:.0}s.",
                    observed_s
                );
            }

            acc.flush_full()?;
        }
    }

    let duration_s = source_frames as f64 / sample_rate as f64;
    tracing::info!(
        "Decoded {} samples at {}Hz ({:.1}s)",
        source_frames,
        sample_rate,
        duration_s
    );

    let all_samples = acc.finish()?;
    if sample_rate != 16000 {
        tracing::info!("Resampled to 16kHz: {} samples", all_samples.len());
    }

    Ok(all_samples)
}
