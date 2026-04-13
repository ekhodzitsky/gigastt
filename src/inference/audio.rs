//! Audio decoding, resampling, and buffer management utilities.

use anyhow::{Context, Result};

use super::{HOP_LENGTH, N_FFT};

const MAX_BUFFER_SAMPLES: usize = 16000 * 5; // 5 seconds at 16kHz
const MAX_DURATION_S: f64 = 600.0; // 10 minutes

/// Decode any supported audio file to mono f32 samples at 16kHz.
///
/// Supports WAV, MP3, M4A/AAC, OGG/Vorbis, and FLAC via symphonia.
/// Multi-channel audio is mixed to mono. Files longer than 10 minutes are rejected.
///
/// # Errors
///
/// Returns an error if the file cannot be opened, decoded, or exceeds the duration limit.
pub fn decode_audio_file(path: &str) -> Result<Vec<f32>> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open audio file: {path}"))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = std::path::Path::new(path).extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .context("Unsupported audio format")?;

    let mut format = probed.format;

    let track = format
        .default_track()
        .context("No audio track found")?;
    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .context("Unknown sample rate")?;
    let channels = track
        .codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(1);

    tracing::info!("Audio: {sample_rate}Hz, {channels}ch, format={}",
        std::path::Path::new(path).extension().unwrap_or_default().to_string_lossy());

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("Unsupported audio codec")?;

    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(anyhow::anyhow!("Error reading packet: {e}")),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = decoder.decode(&packet).context("Decode error")?;
        let spec = *decoded.spec();
        let num_frames = decoded.frames();

        let mut sample_buf = SampleBuffer::<f32>::new(num_frames as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        let samples = sample_buf.samples();

        // Mix to mono if multi-channel
        if spec.channels.count() > 1 {
            let ch = spec.channels.count();
            for frame in 0..num_frames {
                let mut sum = 0.0_f32;
                for c in 0..ch {
                    sum += samples[frame * ch + c];
                }
                all_samples.push(sum / ch as f32);
            }
        } else {
            all_samples.extend_from_slice(samples);
        }
    }

    let duration_s = all_samples.len() as f64 / sample_rate as f64;
    tracing::info!(
        "Decoded {} samples at {}Hz ({:.1}s)",
        all_samples.len(),
        sample_rate,
        duration_s
    );

    if duration_s > MAX_DURATION_S {
        anyhow::bail!(
            "Audio file too long ({:.0}s). Maximum supported: {MAX_DURATION_S:.0}s.",
            duration_s
        );
    }

    // Resample to 16kHz if needed
    if sample_rate != 16000 {
        all_samples = resample(&all_samples, sample_rate, 16000).context("Resampling failed")?;
        tracing::info!("Resampled to 16kHz: {} samples", all_samples.len());
    }

    Ok(all_samples)
}

/// Decode audio from raw bytes in memory (no temp file needed).
///
/// Same logic as [`decode_audio_file`] but reads from an in-memory buffer.
/// Supports WAV, MP3, M4A/AAC, OGG/Vorbis, and FLAC via symphonia.
/// Multi-channel audio is mixed to mono. Audio longer than 10 minutes is rejected.
///
/// # Errors
///
/// Returns an error if the bytes cannot be decoded or the audio exceeds the duration limit.
pub fn decode_audio_bytes(data: &[u8]) -> Result<Vec<f32>> {
    use std::io::Cursor;
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let cursor = Cursor::new(data.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    let hint = Hint::new();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .context("Unsupported audio format")?;

    let mut format = probed.format;

    let track = format
        .default_track()
        .context("No audio track found")?;
    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .context("Unknown sample rate")?;
    let channels = track
        .codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(1);

    tracing::info!("Audio (bytes): {sample_rate}Hz, {channels}ch");

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("Unsupported audio codec")?;

    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(anyhow::anyhow!("Error reading packet: {e}")),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = decoder.decode(&packet).context("Decode error")?;
        let spec = *decoded.spec();
        let num_frames = decoded.frames();

        let mut sample_buf = SampleBuffer::<f32>::new(num_frames as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        let samples = sample_buf.samples();

        // Mix to mono if multi-channel
        if spec.channels.count() > 1 {
            let ch = spec.channels.count();
            for frame in 0..num_frames {
                let mut sum = 0.0_f32;
                for c in 0..ch {
                    sum += samples[frame * ch + c];
                }
                all_samples.push(sum / ch as f32);
            }
        } else {
            all_samples.extend_from_slice(samples);
        }
    }

    let duration_s = all_samples.len() as f64 / sample_rate as f64;
    tracing::info!(
        "Decoded {} samples at {}Hz ({:.1}s)",
        all_samples.len(),
        sample_rate,
        duration_s
    );

    if duration_s > MAX_DURATION_S {
        anyhow::bail!(
            "Audio file too long ({:.0}s). Maximum supported: {MAX_DURATION_S:.0}s.",
            duration_s
        );
    }

    // Resample to 16kHz if needed
    if sample_rate != 16000 {
        all_samples = resample(&all_samples, sample_rate, 16000).context("Resampling failed")?;
        tracing::info!("Resampled to 16kHz: {} samples", all_samples.len());
    }

    Ok(all_samples)
}

/// High-quality polyphase FIR resampler (rubato SincFixedIn).
///
/// Non-finite samples (NaN, infinity) are replaced with `0.0` before resampling.
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    if samples.is_empty() || from_rate == 0 || to_rate == 0 {
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

    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType,
        WindowFunction,
    };

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let ratio = to_rate as f64 / from_rate as f64;
    let mut resampler = SincFixedIn::<f32>::new(
        ratio,
        2.0,
        params,
        samples.len(),
        1, // mono
    )
    .map_err(|e| anyhow::anyhow!("Resampler init failed: {e}"))?;

    let waves_in = vec![samples];
    let mut waves_out = resampler
        .process(&waves_in, None)
        .map_err(|e| anyhow::anyhow!("Resampling failed: {e}"))?;
    Ok(waves_out.remove(0))
}

/// Prepare audio buffer for processing: merge new samples with leftover,
/// truncate if too long, split into usable samples and new leftover.
///
/// Returns `Some(usable_samples)` if enough data for at least one frame,
/// `None` if all data was buffered for the next call.
/// Updates `buffer` in-place with leftover samples.
pub(crate) fn prepare_audio_buffer(
    new_samples: &[f32],
    buffer: &mut Vec<f32>,
) -> Option<Vec<f32>> {
    let mut all_samples = std::mem::take(buffer);
    all_samples.extend_from_slice(new_samples);

    if all_samples.len() > MAX_BUFFER_SAMPLES {
        tracing::warn!("Audio buffer exceeded 5s limit, truncating");
        all_samples = all_samples[all_samples.len() - MAX_BUFFER_SAMPLES..].to_vec();
    }

    let hop_length = HOP_LENGTH;
    let n_fft = N_FFT;
    let usable = if all_samples.len() >= n_fft {
        let num_frames = (all_samples.len() - n_fft) / hop_length + 1;
        (num_frames - 1) * hop_length + n_fft
    } else {
        0
    };

    if usable == 0 {
        *buffer = all_samples;
        return None;
    }

    *buffer = all_samples[usable..].to_vec();
    Some(all_samples[..usable].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- resample tests ---

    #[test]
    fn test_resample_downsample_length() {
        let input: Vec<f32> = (0..4800).map(|i| (i as f32).sin()).collect();
        let output = resample(&input, 48000, 16000).unwrap();
        // Rubato FIR filter has sinc_len/2 delay; output is shorter than ideal ratio.
        // For 4800 samples at 3:1 ratio, expect ~1556 (not exact 1600).
        assert!(!output.is_empty());
        assert!(output.len() > 1400 && output.len() < 1700,
            "Unexpected output length: {}", output.len());
    }

    #[test]
    fn test_resample_upsample_length() {
        let input: Vec<f32> = (0..800).map(|i| (i as f32).sin()).collect();
        let output = resample(&input, 8000, 16000).unwrap();
        // Rubato FIR delay reduces output; expect ~1340 (not exact 1600).
        assert!(!output.is_empty());
        assert!(output.len() > 1200 && output.len() < 1700,
            "Unexpected output length: {}", output.len());
    }

    #[test]
    fn test_resample_preserves_dc() {
        // Constant signal should remain approximately constant after resampling.
        // Rubato FIR filter may cause transients at edges; check the middle 80%.
        let input = vec![0.5_f32; 4800];
        let output = resample(&input, 48000, 16000).unwrap();
        let start = output.len() / 10;
        let end = output.len() - start;
        for &sample in &output[start..end] {
            assert!((sample - 0.5).abs() < 0.05, "DC signal not preserved: {sample}");
        }
    }

    #[test]
    fn test_resample_empty() {
        let output = resample(&[], 48000, 16000).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn test_resample_zero_rate_returns_empty() {
        let input = vec![1.0, 2.0, 3.0];
        assert!(resample(&input, 0, 16000).unwrap().is_empty());
        assert!(resample(&input, 16000, 0).unwrap().is_empty());
    }

    #[test]
    fn test_resample_same_rate() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = resample(&input, 16000, 16000).unwrap();
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
        assert_eq!(result.unwrap().len(), N_FFT);
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
        assert_eq!(usable.len(), N_FFT); // one frame
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
        assert_eq!(usable.len(), 320);
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
        assert!(usable.len() + buffer.len() <= MAX_BUFFER_SAMPLES);
    }

    #[test]
    fn test_buffer_multi_frame() {
        // N_FFT + HOP_LENGTH = 480 → 2 frames, no leftover
        let new_samples = vec![1.0; N_FFT + HOP_LENGTH];
        let mut buffer = Vec::new();
        let result = prepare_audio_buffer(&new_samples, &mut buffer);
        assert!(result.is_some());
        // 2 frames: usable = (2-1)*160 + 320 = 480
        assert_eq!(result.unwrap().len(), N_FFT + HOP_LENGTH);
        assert!(buffer.is_empty());
    }

    // --- stress tests: robustness edge cases ---

    #[test]
    fn test_resample_nan_input() {
        let input = vec![f32::NAN; 1000];
        let output = resample(&input, 48000, 16000).unwrap();
        // NaN should be replaced with zeros
        assert!(!output.is_empty());
        for &s in &output {
            assert!(s.is_finite(), "NaN should be sanitized to zero, got {s}");
        }
    }

    #[test]
    fn test_resample_infinity_input() {
        let input = vec![f32::INFINITY; 500];
        let output = resample(&input, 48000, 16000).unwrap();
        assert!(!output.is_empty());
        for &s in &output {
            assert!(s.is_finite(), "Infinity should be sanitized to zero, got {s}");
        }
    }

    #[test]
    fn test_resample_mixed_nan_normal() {
        let mut input = vec![0.5_f32; 480];
        input[100] = f32::NAN;
        input[200] = f32::NEG_INFINITY;
        let output = resample(&input, 48000, 16000).unwrap();
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
        assert!(usable.len() + buffer.len() <= MAX_BUFFER_SAMPLES);
    }

    #[test]
    fn test_prepare_buffer_one_over_max() {
        // MAX_BUFFER_SAMPLES + 1 — triggers truncation
        let new_samples = vec![1.0; MAX_BUFFER_SAMPLES + 1];
        let mut buffer = Vec::new();
        let result = prepare_audio_buffer(&new_samples, &mut buffer);
        assert!(result.is_some());
        let usable = result.unwrap();
        assert!(usable.len() + buffer.len() <= MAX_BUFFER_SAMPLES);
    }
}
