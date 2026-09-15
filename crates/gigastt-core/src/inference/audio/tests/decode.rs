use super::*;
use bytes::Bytes;
use std::io::{Read, Seek, SeekFrom};

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
    // Regression: a crafted APEv2 tag header (APE tags can ride on MP3
    // uploads) sets an unbounded `size` field that made crates.io
    // symphonia-metadata 0.6.0 panic with "attempt to add with overflow" on
    // `size + 32` (ape.rs). Symphonia 0.6.1 widens the size to u64 before
    // adding the header; decode must return a graceful `Err` — never panic.
    //
    // Two fixtures exercise the same path: the original on-disk seed (36 B),
    // and the exact Continuous Fuzz artifact that reddened Nightly Soak
    // (crash-2cd57d8e…, 38 B, 2026-08-11).
    let fixtures: &[&[u8]] = &[
        include_bytes!("../../../../tests/fixtures/ape_overflow_crash.bin"),
        &[
            0xff, 0xf0, 0xff, 0x41, 0x50, 0x45, 0x54, 0x41, 0x47, 0x45, 0x58, 0xd0, 0x07, 0x00,
            0x00, 0xf8, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xf1, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xf8, 0xf0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xf8,
        ],
    ];
    assert_eq!(fixtures[0].len(), 36);
    assert_eq!(fixtures[1].len(), 38);
    for (i, crash) in fixtures.iter().enumerate() {
        let result = decode_audio_bytes(crash);
        assert!(
            result.is_err(),
            "fixture {i}: crafted APEv2 header must yield a decode error, not panic or Ok"
        );
    }
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

#[test]
fn test_decode_rejects_adversarial_sample_rate() {
    // A crafted header with an out-of-range sample rate must be rejected
    // before it can scale the length budget or trigger an oversized
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
