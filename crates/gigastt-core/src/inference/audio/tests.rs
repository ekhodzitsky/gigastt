use super::*;
use bytes::Bytes;
use rubato::Resampler;

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
    let crash = include_bytes!("../../../tests/fixtures/ape_overflow_crash.bin");
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

#[test]
fn test_probe_duration_wav_reports_declared_seconds() {
    // A WAV header declares its frame count, so the probe returns the duration
    // without decoding a single packet.
    let wav = make_wav_bytes(&vec![0i16; 16000], 16000); // exactly 1.0 s
    let probed = probe_duration_bytes(Bytes::from(wav)).unwrap();
    assert!(
        matches!(probed, Some(s) if (s - 1.0).abs() < 1e-6),
        "expected ~1.0 s, got {probed:?}"
    );
}

#[test]
fn test_probe_duration_agrees_with_decoded_length() {
    // The probe's declared duration must match the decoded sample count: the
    // job executor uses the two interchangeably to size the progress bar, so a
    // divergence would move the bar when the probe fast-path kicks in.
    let wav = make_wav_bytes(&vec![0i16; 24000], 16000); // 1.5 s at 16 kHz
    let probed = probe_duration_bytes(Bytes::from(wav.clone()))
        .unwrap()
        .expect("WAV declares its duration");
    let decoded_s = decode_audio_bytes_shared(Bytes::from(wav)).unwrap().len() as f64 / 16_000.0;
    assert!(
        (probed - decoded_s).abs() < 1e-3,
        "probe {probed} vs decode {decoded_s}"
    );
}

#[test]
fn test_probe_duration_non_container_does_not_claim_duration() {
    // Bytes that are not a supported container must not panic and must never
    // report a duration — the caller falls back to a real decode, which
    // surfaces the proper "invalid audio" error.
    let r = probe_duration_bytes(Bytes::from_static(b"definitely not audio"));
    assert!(
        r.is_err() || matches!(r, Ok(None)),
        "garbage bytes must be Err or Ok(None), got {r:?}"
    );
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
    assert!(channels[0].len() > num_samples * 15 / 10 && channels[0].len() < num_samples * 25 / 10);
    assert!(channels[1].len() > num_samples * 15 / 10 && channels[1].len() < num_samples * 25 / 10);
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
    let feed = |cache: &mut Option<rubato::Async<f32>>, out: &mut Vec<f32>, n: usize, seed: f32| {
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * seed).sin()).collect();
        resample_with_cache(input, SampleRate(48_000), SampleRate(16_000), cache, out).unwrap();
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
    let wav = include_bytes!("../../../tests/fixtures/telephony/g722_tone.wav");
    let reference_pcm = include_bytes!("../../../tests/fixtures/telephony/g722_tone_ffmpeg.pcm");
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

// --- Opus (OGG container, pure-Rust opus-rs fallback decoder) ---

#[test]
fn test_is_recoverable_packet_eof_matches_unexpected_eof_only() {
    use std::io::{Error as IoError, ErrorKind};
    use symphonia::core::errors::Error as SymError;

    let eof = SymError::IoError(IoError::new(
        ErrorKind::UnexpectedEof,
        "unexpected end of file",
    ));
    assert!(is_recoverable_packet_eof(&eof));

    let other_io = SymError::IoError(IoError::other("disk full"));
    assert!(!is_recoverable_packet_eof(&other_io));

    let decode = SymError::DecodeError("bad page");
    assert!(!is_recoverable_packet_eof(&decode));

    let unsupported = SymError::Unsupported("codec");
    assert!(!is_recoverable_packet_eof(&unsupported));
}

#[test]
fn test_decode_audio_bytes_opus_ogg_missing_eos_succeeds() {
    // Telegram Android (and some MediaRecorder paths) write Ogg/Opus without
    // the EOS flag on the final page. Symphonia then ends the demux with
    // UnexpectedEof instead of Ok(None). Soft-EOF must still return audio
    // (issue #217). Fixture is opus_tone.ogg with the EOS bit cleared and
    // the page CRC recomputed.
    let no_eos = include_bytes!("../../../tests/fixtures/opus/opus_tone_no_eos.ogg");
    let with_eos = include_bytes!("../../../tests/fixtures/opus/opus_tone.ogg");
    let decoded_no_eos = decode_audio_bytes(no_eos).expect("OGG/Opus without EOS must decode");
    let decoded_with_eos =
        decode_audio_bytes(with_eos).expect("OGG/Opus with EOS must still decode");
    assert!(
        !decoded_no_eos.is_empty(),
        "missing-EOS stream must yield non-empty PCM"
    );
    // Same content; length must match the EOS sibling within a sample or two.
    let delta = (decoded_no_eos.len() as i64 - decoded_with_eos.len() as i64).unsigned_abs();
    assert!(
        delta <= 2,
        "no-EOS length {} diverged from with-EOS length {}",
        decoded_no_eos.len(),
        decoded_with_eos.len()
    );
    // Spot-check a stretch of samples (pre-skip / tone body) for identity.
    let start = decoded_no_eos.len().min(decoded_with_eos.len()) / 4;
    let end = start + 1000;
    for (a, b) in decoded_no_eos[start..end]
        .iter()
        .zip(decoded_with_eos[start..end].iter())
    {
        assert!((a - b).abs() < f32::EPSILON);
    }
}

#[test]
fn test_decode_audio_file_opus_missing_eos_matches_bytes() {
    let no_eos = include_bytes!("../../../tests/fixtures/opus/opus_tone_no_eos.ogg");
    let mut tmp = tempfile::NamedTempFile::with_suffix(".ogg").expect("temp file");
    std::io::Write::write_all(&mut tmp, no_eos).expect("write temp file");
    let via_file = decode_audio_file(tmp.path().to_str().expect("utf-8 path"))
        .expect("missing-EOS OGG/Opus file must decode");
    let via_bytes = decode_audio_bytes(no_eos).expect("missing-EOS bytes must decode");
    assert_eq!(via_file.len(), via_bytes.len());
    for (a, b) in via_file.iter().zip(via_bytes.iter()) {
        assert!((a - b).abs() < f32::EPSILON);
    }
}

#[test]
fn test_decode_audio_bytes_truncated_opus_headers_only_still_errors() {
    // Truncate after OpusHead+OpusTags pages so demux may open the track but
    // no audio packets arrive. Soft-EOF must NOT turn this into silence.
    let full = include_bytes!("../../../tests/fixtures/opus/opus_tone.ogg");
    // First two Ogg pages only (~header); keep under 200 bytes of safety.
    // Find end of page 1 (seq 1) more carefully: second page ends before first audio.
    let mut pages = Vec::new();
    let mut i = 0usize;
    let data = full;
    while i + 27 <= data.len() {
        if &data[i..i + 4] != b"OggS" {
            break;
        }
        let nseg = data[i + 26] as usize;
        let body: usize = data[i + 27..i + 27 + nseg]
            .iter()
            .map(|&s| s as usize)
            .sum();
        let page_end = i + 27 + nseg + body;
        pages.push(page_end);
        i = page_end;
        if pages.len() == 2 {
            break;
        }
    }
    assert!(pages.len() >= 2, "fixture must have header pages");
    let headers_only = &data[..pages[1]];
    let err = decode_audio_bytes(headers_only).expect_err("headers-only Opus must fail");
    let msg = format!("{err:#}");
    // Must not succeed with empty/near-empty PCM via soft-EOF.
    assert!(
        msg.contains("packet")
            || msg.contains("end of file")
            || msg.contains("audio")
            || msg.contains("Decode")
            || msg.contains("Unsupported")
            || msg.contains("malformed")
            || msg.contains("Opus")
            || msg.contains("track")
            || msg.contains("empty")
            || msg.contains("No "),
        "unexpected error for headers-only: {msg}"
    );
}

#[test]
fn test_decode_audio_bytes_random_bytes_still_errors() {
    let junk = [0u8; 64];
    assert!(decode_audio_bytes(&junk).is_err());
}

#[test]
fn test_opus_packet_frame_size_toc_parsing() {
    // Config 0 (SILK 10 ms), code 0: one 10 ms frame = 480 samples.
    assert_eq!(opus_packet_frame_size(&[0b0000_0000]), Some(480));
    // Config 3 (SILK 60 ms), code 1: two 60 ms frames = 5760 (the max).
    assert_eq!(opus_packet_frame_size(&[0b0001_1001]), Some(5760));
    // Config 16 (CELT 2.5 ms), code 0: 120 samples.
    assert_eq!(opus_packet_frame_size(&[0b1000_0000]), Some(120));
    // Config 31 (CELT 20 ms), code 2: two 20 ms frames = 1920.
    assert_eq!(opus_packet_frame_size(&[0b1111_1010]), Some(1920));
    // Config 12 (hybrid 10 ms), code 3: M=3 frames from the second byte.
    assert_eq!(opus_packet_frame_size(&[0b0110_0011, 3]), Some(1440));
    // Code 3 with M=0 frames is invalid.
    assert_eq!(opus_packet_frame_size(&[0b0110_0011, 0]), None);
    // Over 120 ms total (60 ms x 3) exceeds the RFC 6716 packet maximum.
    assert_eq!(opus_packet_frame_size(&[0b0001_1011, 3]), None);
    // Empty packet.
    assert_eq!(opus_packet_frame_size(&[]), None);
}

#[test]
fn test_decode_audio_bytes_opus_ogg_matches_ffmpeg_reference() {
    // Independent-reference verification: `opus_tone.ogg` was ENCODED by
    // ffmpeg (libopus) and `opus_tone_ffmpeg.pcm` is ffmpeg's own DECODE
    // of it resampled to 16 kHz mono (see
    // scripts/generate_opus_fixtures.sh). Our opus-rs decode is compared
    // against libopus, so the pure-Rust port is validated against a
    // second implementation rather than against itself. We do not trim
    // the OpusHead pre-skip (ffmpeg does), so the comparison runs at the
    // best lag. Tolerance: RMSE below 2% of full scale.
    let ogg = include_bytes!("../../../tests/fixtures/opus/opus_tone.ogg");
    let reference_pcm = include_bytes!("../../../tests/fixtures/opus/opus_tone_ffmpeg.pcm");
    let ours = decode_audio_bytes(ogg).expect("OGG/Opus must decode");
    let reference: Vec<f32> = reference_pcm
        .chunks_exact(2)
        .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) / 32768.0)
        .collect();
    // 3 s of tone at 16 kHz; the untrimmed pre-skip on our side and the
    // resampler's FIR delay shift the exact count by a few hundred.
    assert!(
        ours.len() > 46_000 && ours.len() < 50_000,
        "unexpected decoded length {}",
        ours.len()
    );
    let rmse = best_lag_rmse(&ours, &reference, 1024);
    assert!(
        rmse < 0.02,
        "Opus decode diverged from ffmpeg reference: RMSE {rmse}"
    );
}

#[test]
fn test_decode_audio_file_opus_extension_matches_bytes() {
    // The file path probes with an `.opus` extension hint; the bytes path
    // sniffs content only. Both must decode the same OGG/Opus stream
    // identically (the CLI transcribes via `decode_audio_file`).
    let ogg = include_bytes!("../../../tests/fixtures/opus/opus_tone.ogg");
    let mut tmp = tempfile::NamedTempFile::with_suffix(".opus").expect("temp file");
    std::io::Write::write_all(&mut tmp, ogg).expect("write temp file");
    let via_file = decode_audio_file(tmp.path().to_str().expect("utf-8 path"))
        .expect("OGG/Opus file must decode");
    let via_bytes = decode_audio_bytes(ogg).expect("OGG/Opus bytes must decode");
    assert_eq!(via_file.len(), via_bytes.len());
    for (a, b) in via_file.iter().zip(via_bytes.iter()) {
        assert!((a - b).abs() < f32::EPSILON);
    }
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

// --- streaming resample equivalence (whole-buffer reference) ---

/// PCM16 samples of a committed 16 kHz mono WAV fixture, used as a real-signal
/// input for the streaming-resample equivalence tests.
fn fixture_tone_pcm() -> Vec<i16> {
    let wav = include_bytes!("../../../tests/fixtures/telephony/tone_src.wav");
    let data = super::telephony::find_riff_chunk(wav, b"data").expect("fixture data chunk");
    data.chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect()
}

/// Linear sine sweep, PCM16. Sweeping across the whole band is the adversarial
/// case for a resampler seam: any FIR-history reset shows up as a spike.
fn sweep_pcm(rate: u32, seconds: f32) -> Vec<i16> {
    let n = (rate as f32 * seconds) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / rate as f32;
            let f = 50.0 + (0.45 * rate as f32 - 50.0) * (t / seconds);
            (0.8 * (std::f32::consts::PI * f * t).sin() * 32000.0) as i16
        })
        .collect()
}

/// Pure tone, PCM16, with the phase accumulated in f64 so the *input* carries
/// no drift of its own. Where `sweep_pcm` probes chunk seams, a single tone
/// probes long-run phase accumulation: it has an analytic ground truth, so each
/// path can be scored against the truth instead of only against the other one.
/// `freq` is chosen to complete a whole number of cycles per 16 kHz analysis
/// window, which makes the phase estimate below leakage-free.
fn tone_pcm(rate: u32, seconds: f64, freq: f64) -> Vec<i16> {
    let n = (f64::from(rate) * seconds) as usize;
    (0..n)
        .map(|i| {
            let t = i as f64 / f64::from(rate);
            (0.8 * (std::f64::consts::TAU * freq * t).sin() * 32000.0) as i16
        })
        .collect()
}

/// Largest deviation, across one-second windows, of the measured phase of
/// `freq` from the phase of the first window. A resampler whose fractional read
/// position accumulates error stretches time slightly, which shows up here as a
/// phase that walks away from where it started.
fn max_phase_drift_16k(samples: &[f32], freq: f64) -> f64 {
    const WINDOW: usize = 16_000;
    let phase_of = |start: usize| {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (i, &v) in samples[start..start + WINDOW].iter().enumerate() {
            let w = std::f64::consts::TAU * freq * ((start + i) as f64 / 16_000.0);
            re += f64::from(v) * w.cos();
            im += f64::from(v) * w.sin();
        }
        im.atan2(re)
    };
    let first = phase_of(0);
    let mut worst = 0.0f64;
    for w in 0..samples.len() / WINDOW {
        let mut d = phase_of(w * WINDOW) - first;
        // Wrap into (-pi, pi] so a drift that crosses a cycle stays comparable.
        d -= std::f64::consts::TAU * (d / std::f64::consts::TAU).round();
        worst = worst.max(d.abs());
    }
    worst
}

/// Signal-to-error ratio, in dB, of `candidate` measured against `reference`.
fn signal_to_error_db(reference: &[f32], candidate: &[f32]) -> f64 {
    let mut err = 0.0f64;
    let mut sig = 0.0f64;
    for (&r, &c) in reference.iter().zip(candidate) {
        let d = f64::from(r) - f64::from(c);
        err += d * d;
        sig += f64::from(r) * f64::from(r);
    }
    if err == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (sig / err).log10()
}

/// Assert the streaming path agrees with a whole-buffer `resample()` of the
/// same decoded signal: same length within one sample, max per-sample delta
/// below 1e-4.
fn assert_matches_whole_buffer(streamed: &[f32], reference: &[f32], what: &str) {
    let len_diff = streamed.len().abs_diff(reference.len());
    assert!(
        len_diff <= 1,
        "{what}: length diverged, streaming {} vs whole-buffer {}",
        streamed.len(),
        reference.len()
    );
    let cmp = streamed.len().min(reference.len());
    assert!(cmp > 0, "{what}: nothing to compare");
    let mut max_diff = 0.0f32;
    let mut max_at = 0usize;
    for i in 0..cmp {
        let d = (streamed[i] - reference[i]).abs();
        if d > max_diff {
            max_diff = d;
            max_at = i;
        }
    }
    assert!(
        max_diff <= 1e-4,
        "{what}: max |streaming - whole-buffer| = {max_diff} at sample {max_at}"
    );
}

/// Decoding the same PCM twice — once with a 16 kHz header (passthrough, so
/// the result is exactly what the decoder produced) and once with a `rate`
/// header (the streaming resample path) — must agree with running the
/// passthrough result through the whole-buffer `resample()`.
fn check_mono_equivalence(pcm: &[i16], rate: u32, what: &str) {
    let at_source = decode_audio_bytes(&make_wav_bytes(pcm, 16000)).unwrap();
    let reference = resample(&at_source, SampleRate(rate), SampleRate(16000)).unwrap();
    let streamed = decode_audio_bytes(&make_wav_bytes(pcm, rate)).unwrap();
    assert_matches_whole_buffer(&streamed, &reference, what);
}

fn check_stereo_equivalence(left: &[i16], right: &[i16], rate: u32, what: &str) {
    // Mono-mix path.
    let mixed_at_source = decode_audio_bytes(&make_stereo_wav_bytes(left, right, 16000)).unwrap();
    let mixed_reference = resample(&mixed_at_source, SampleRate(rate), SampleRate(16000)).unwrap();
    let mixed_streamed = decode_audio_bytes(&make_stereo_wav_bytes(left, right, rate)).unwrap();
    assert_matches_whole_buffer(&mixed_streamed, &mixed_reference, &format!("{what} mixed"));

    // Split-channel path.
    let split_at_source =
        decode_audio_bytes_shared_channels(Bytes::from(make_stereo_wav_bytes(left, right, 16000)))
            .unwrap();
    let split_streamed =
        decode_audio_bytes_shared_channels(Bytes::from(make_stereo_wav_bytes(left, right, rate)))
            .unwrap();
    assert_eq!(split_streamed.len(), split_at_source.len());
    for (c, (streamed, source)) in split_streamed.iter().zip(&split_at_source).enumerate() {
        let reference = resample(source, SampleRate(rate), SampleRate(16000)).unwrap();
        assert_matches_whole_buffer(streamed, &reference, &format!("{what} channel {c}"));
    }
}

#[test]
#[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
fn test_streaming_decode_matches_whole_buffer_resample_48k() {
    let pcm = sweep_pcm(48_000, 2.5);
    check_mono_equivalence(&pcm, 48_000, "48k sweep mono");
}

#[test]
#[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
fn test_streaming_decode_matches_whole_buffer_resample_44k1() {
    let pcm = sweep_pcm(44_100, 2.5);
    check_mono_equivalence(&pcm, 44_100, "44.1k sweep mono");
}

#[test]
#[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
fn test_streaming_decode_matches_whole_buffer_resample_stereo_48k() {
    let left = sweep_pcm(48_000, 2.5);
    let right: Vec<i16> = left.iter().rev().copied().collect();
    check_stereo_equivalence(&left, &right, 48_000, "48k sweep stereo");
}

#[test]
#[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
fn test_streaming_decode_matches_whole_buffer_resample_stereo_44k1() {
    let left = sweep_pcm(44_100, 2.5);
    let right: Vec<i16> = left.iter().rev().copied().collect();
    check_stereo_equivalence(&left, &right, 44_100, "44.1k sweep stereo");
}

/// Long-input gate for the staged resample at a NON-INTEGER ratio.
///
/// The fixtures above hold both paths to 1e-4 per sample, but they are 2.5 s
/// long and that tolerance is only reachable at that scale. rubato carries its
/// fractional read position in a single f64 that runs monotonically for the
/// whole of one `process` call (`idx += 1/ratio` per output sample) and takes
/// the sub-sample offset as `idx * 256 - floor(idx * 256)`, so the resolution
/// of that offset halves every time `idx` doubles. One whole-buffer call over
/// 300 s of 44.1 kHz audio drives `idx` to 13.2 million, where the offset is
/// quantised to ~1e-6; the staged path restarts `idx` near zero on every flush
/// and holds ~1e-9. The two therefore separate as the input grows, and past
/// ~300 s the gap is over the 1e-4 per-sample bound the short fixtures use —
/// this case measures 1.2e-4, which is why it is scored on the error-to-signal
/// ratio instead. Sweeping the duration over the same comparison gives max
/// per-sample deltas of 6.7e-7 at 30 s, 1.2e-5 at 120 s, 1.3e-4 at 300 s and
/// 4.1e-4 at 600 s.
///
/// What separates is the *reference*, not the path under test. Against the
/// analytic tone the staged path's phase is flat with duration (9.2e-5 rad at
/// 30 s, 9.3e-5 rad at 600 s) while the whole-buffer path's walks (9.3e-5 rad
/// at 30 s, 4.2e-4 rad at 600 s); this case measures 8.4e-5 rad staged against
/// 2.4e-4 rad whole-buffer. Asserting the ordering pins that direction: a plain
/// delta cannot say which side moved, but this fails if the staged path ever
/// becomes the one that drifts. Integer ratios are exempt from all of it —
/// `1/ratio` is exact in f64 at 48/32/8 kHz, so those stay bit-identical at any
/// length and keep the strict per-sample gate.
///
/// Costs ~50 s in a debug build, which is why one duration is covered rather
/// than a sweep; 300 s is the shortest that reaches the regime. That cost is
/// also why it is `#[ignore]`d: PR CI runs `cargo test --workspace --lib` and
/// stays fast, while the main-push lane runs this by name.
#[test]
#[ignore = "~50 s in debug; long-duration numeric gate, run on main push"]
#[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
fn test_streaming_decode_long_44k1_input_holds_phase_better_than_whole_buffer() {
    // A whole number of cycles per one-second analysis window, so the phase
    // estimate sees no spectral leakage.
    const FREQ: f64 = 1_000.0;
    let pcm = tone_pcm(44_100, 300.0, FREQ);

    let at_source = decode_audio_bytes(&make_wav_bytes(&pcm, 16000)).unwrap();
    let reference = resample(&at_source, SampleRate(44_100), SampleRate(16000)).unwrap();
    let streamed = decode_audio_bytes(&make_wav_bytes(&pcm, 44_100)).unwrap();

    // Length stays exact: the divergence is sub-sample phase, never a dropped
    // or duplicated frame at a flush boundary.
    assert_eq!(
        streamed.len(),
        reference.len(),
        "long 44.1k input: length diverged"
    );

    // Error-to-signal floor for non-integer ratios at this length, in place of
    // the per-sample tolerance the short fixtures use.
    let snr = signal_to_error_db(&reference, &streamed);
    assert!(
        snr >= 70.0,
        "long 44.1k input: streaming vs whole-buffer SNR {snr:.1} dB below the 70 dB floor"
    );

    let streamed_drift = max_phase_drift_16k(&streamed, FREQ);
    let reference_drift = max_phase_drift_16k(&reference, FREQ);
    assert!(
        streamed_drift <= reference_drift,
        "long 44.1k input: staged path drifted {streamed_drift:.3e} rad, \
         more than the whole-buffer reference's {reference_drift:.3e} rad"
    );
    // Absolute floor as well, so the ordering assertion cannot be satisfied by
    // both paths degrading together.
    assert!(
        streamed_drift <= 2e-4,
        "long 44.1k input: staged path phase drift {streamed_drift:.3e} rad exceeds 2e-4"
    );
}

#[test]
#[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
fn test_streaming_decode_matches_whole_buffer_resample_fixture() {
    // The fixture is exactly one staging chunk long, so on its own it drains
    // once and never crosses a chunk boundary — the seam this test exists to
    // guard would go unobserved. Doubling it puts a full flush on each side of
    // a boundary plus a short tail, so a resampler that dropped its FIR history
    // between flushes shows up here.
    let pcm = [fixture_tone_pcm(), fixture_tone_pcm()].concat();
    assert!(
        pcm.len() > super::resample::RESAMPLE_STAGING_FRAMES,
        "fixture must span more than one staging flush, got {} samples",
        pcm.len()
    );
    check_mono_equivalence(&pcm, 48_000, "fixture 48k");
    check_mono_equivalence(&pcm, 44_100, "fixture 44.1k");
}

#[test]
fn test_streaming_decode_16k_input_is_bit_identical() {
    // A 16 kHz source must never reach the resampler: every sample stays the
    // raw PCM16 conversion and the frame count is preserved exactly.
    let pcm = fixture_tone_pcm();
    let mono = decode_audio_bytes(&make_wav_bytes(&pcm, 16000)).unwrap();
    assert_eq!(mono.len(), pcm.len());
    for (i, (&raw, &got)) in pcm.iter().zip(&mono).enumerate() {
        let expected = f32::from(raw) / 32768.0;
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "sample {i} was filtered: {got} vs {expected}"
        );
    }

    let right: Vec<i16> = pcm.iter().rev().copied().collect();
    let channels =
        decode_audio_bytes_shared_channels(Bytes::from(make_stereo_wav_bytes(&pcm, &right, 16000)))
            .unwrap();
    assert_eq!(channels.len(), 2);
    for (c, raw) in [&pcm, &right].iter().enumerate() {
        assert_eq!(channels[c].len(), raw.len());
        for (i, (&r, &got)) in raw.iter().zip(&channels[c]).enumerate() {
            let expected = f32::from(r) / 32768.0;
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "channel {c} sample {i} was filtered: {got} vs {expected}"
            );
        }
    }
}

#[test]
#[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
fn test_telephony_raw_streaming_matches_whole_buffer_resample() {
    // 8 kHz G.711 upsamples to 16 kHz through the same staged path. Long
    // enough to fill the staging buffer more than once — at 2.5 s the whole
    // clip fits in a single flush and no chunk boundary is ever crossed.
    let pcm = sweep_pcm(8_000, 12.5);
    assert!(
        pcm.len() > super::resample::RESAMPLE_STAGING_FRAMES,
        "clip must span more than one staging flush, got {} samples",
        pcm.len()
    );
    let mut encoder = audio_codec::pcmu::PcmuEncoder::new();
    let encoded = audio_codec::Encoder::encode(&mut encoder, &pcm);
    let mut decoder = audio_codec::pcmu::PcmuDecoder::new();
    let round_tripped = audio_codec::Decoder::decode(&mut decoder, &encoded);
    let at_source: Vec<f32> = round_tripped
        .iter()
        .map(|&s| f32::from(s) / 32768.0)
        .collect();
    let reference = resample(&at_source, SampleRate(8_000), SampleRate(16_000)).unwrap();
    let streamed = decode_telephony_raw(&encoded, TelephonyCodec::Pcmu, 8_000).unwrap();
    assert_matches_whole_buffer(&streamed, &reference, "pcmu 8k");
}
