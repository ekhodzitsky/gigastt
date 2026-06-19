//! Audio decoding, resampling, and buffer management utilities.

use anyhow::{Context, Result};
use bytes::Bytes;
use rubato::Resampler;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::{MediaSource, MediaSourceStream};
use symphonia::core::meta::MetadataOptions;

use super::{HOP_LENGTH, N_FFT};

const MAX_BUFFER_SAMPLES: usize = 16000 * 5; // 5 seconds at 16kHz
/// Hard upper bound on file-transcription audio length (seconds). Long-form
/// inputs are decoded in bounded overlapping chunks (see
/// `Engine::transcribe_samples_chunked`), so peak memory is O(chunk) regardless
/// of file length; this cap now only fences off genuinely absurd / adversarial
/// uploads rather than the old 10-minute encoder-memory limit. Raised to 2h.
const MAX_DURATION_S: f64 = 7200.0; // 2 hours
/// Upper bound on a header-declared sample rate. Legal rates (8k–48k) stay well
/// below this; anything above is a malformed/adversarial header and is rejected
/// before it can scale the duration cap or the capacity hint.
const MAX_SAMPLE_RATE: u32 = 192_000;
/// Ceiling used to size the duration cap and capacity hint. The header's
/// `sample_rate` is clamped to this when computing the sample budget, so a
/// crafted header cannot inflate either beyond `MAX_DURATION_S` × 48 kHz worth
/// of samples.
const MAX_DECODE_SAMPLE_RATE: u32 = 48_000;

/// Maximum number of decoded samples allowed for `sample_rate`, the budget used
/// by both the duration cap and the up-front capacity hint. The header rate is
/// clamped to [`MAX_DECODE_SAMPLE_RATE`] so a crafted header cannot inflate the
/// budget beyond [`MAX_DURATION_S`] × 48 kHz. Pure so the cap math is testable
/// without decoding a file.
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
pub(crate) struct BytesMediaSource {
    data: Bytes,
    pos: u64,
}

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
pub fn decode_audio_file(path: &str) -> Result<Vec<f32>> {
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
pub fn decode_audio_bytes_shared(data: Bytes) -> Result<Vec<f32>> {
    let source = BytesMediaSource::new(data);
    let mss = MediaSourceStream::new(Box::new(source), Default::default());
    let hint = Hint::new();
    decode_audio_inner(mss, hint, "bytes")
}

/// Shared decode logic: probe → format → decode → mono mix → duration check → resample.
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

/// Resample audio using an optional cached resampler.
///
/// The cached resampler is created on first call and reused when the input
/// chunk size matches. If the chunk size changes, the cache is recreated.
///
/// ```text
/// { from_rate.0 > 0 && to_rate.0 > 0 }
/// fn resample_with_cache(samples: Vec<f32>, from_rate: SampleRate, to_rate: SampleRate, cache: &mut Option<rubato::Async<f32>>, out_buf: &mut Vec<f32>) -> anyhow::Result<()>
/// { ret.as_ref().map(|v| !v.is_empty() || samples.is_empty() || from_rate == to_rate).unwrap_or(true) }
/// ```
/// Resample audio using an optional cached resampler, writing into a caller-provided buffer.
///
/// The cached resampler is created on first call and reused when the input
/// chunk size matches. If the chunk size changes, the cache is recreated.
/// Non-finite samples are sanitized in-place.
///
/// `samples` is consumed (moved) so that in-place sanitization avoids an
/// extra allocation. Callers that already own the input vector should pass
/// it directly; the buffer is not borrowed after the call.
pub fn resample_with_cache(
    mut samples: Vec<f32>,
    from_rate: SampleRate,
    to_rate: SampleRate,
    cache: &mut Option<rubato::Async<f32>>,
    out_buf: &mut Vec<f32>,
) -> anyhow::Result<()> {
    use rubato::Resampler;

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

    let ratio = to_rate.0 as f64 / from_rate.0 as f64;
    let chunk = samples.len();

    let needs_new = match cache {
        Some(r) => r.set_chunk_size(chunk).is_err(),
        None => true,
    };

    if needs_new {
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
        let r = Async::<f32>::new_sinc(ratio, 2.0, &params, chunk, 1, FixedAsync::Input)
            .map_err(|e| anyhow::anyhow!("Resampler init failed: {e}"))?;
        *cache = Some(r);
    }

    let resampler = match cache.as_mut() {
        Some(r) => r,
        None => anyhow::bail!("Resampler cache is None after initialization"),
    };
    let needed = resampler.output_frames_next();
    out_buf.clear();
    out_buf.resize(needed, 0.0);

    use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
    let input_data = [samples];
    let input = SequentialSliceOfVecs::new(&input_data, 1, chunk)
        .map_err(|e| anyhow::anyhow!("Resampler input adapter failed: {e}"))?;
    let mut output = SequentialSliceOfVecs::new_mut(std::slice::from_mut(out_buf), 1, needed)
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
        // Pure cap math (testable without realizing a multi-hour PCM buffer):
        // the sample budget scales with the clamped rate and the raised cap.
        // A >10-minute file is now under budget (chunked long-form decode bounds
        // memory); only genuinely absurd lengths trip the cap.
        let budget_16k = max_decode_samples(16000);
        // 2h cap at 16kHz => 7200 * 16000 samples.
        assert_eq!(budget_16k, 7200 * 16000);
        // 12 minutes (the old reject point) is now comfortably under budget.
        assert!(12 * 60 * 16000 < budget_16k, "12-minute file must pass now");
        // >2h is over budget and would be rejected.
        assert!(
            (2 * 3600 + 1) * 16000 > budget_16k,
            ">2h must exceed budget"
        );
        // Header rate is clamped: a crafted 192kHz header can't inflate the
        // budget past the 48kHz ceiling.
        assert_eq!(max_decode_samples(192_000), max_decode_samples(48_000));
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
}

#[cfg(test)]
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
