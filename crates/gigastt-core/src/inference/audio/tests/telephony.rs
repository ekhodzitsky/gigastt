use super::*;

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
    let encoded = encode_telephony(&mut encoder, &source);
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
    let encoded = encode_telephony(&mut encoder, &source);
    let decoded = decode_telephony_raw(&encoded, TelephonyCodec::Pcma, 8000).unwrap();
    assert!(decoded.len() > 12_000 && decoded.len() <= 16_000);
    assert!(decoded.iter().all(|s| s.is_finite()));
}

#[test]
fn test_decode_telephony_raw_g722_roundtrip() {
    // 1 s of 16 kHz tone natively; G.722 output stays at its native 16 kHz.
    // Miri uses a tenth of a second. The QMF delay and the RMSE bound are the
    // same either way.
    let n = if cfg!(miri) { 1_600 } else { 16_000 };
    let source: Vec<i16> = (0..n)
        .map(|i| ((i as f32 * 0.03).sin() * 10000.0) as i16)
        .collect();
    let mut encoder = audio_codec::g722::G722Encoder::new();
    let encoded = encode_telephony(&mut encoder, &source);
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
    // G.711 A-law in WAV (tag 0x0006) is decoded by ryf.
    let source = test_tone_8k(8000);
    let mut encoder = audio_codec::pcma::PcmaEncoder::new();
    let encoded = encode_telephony(&mut encoder, &source);
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
    // G.711 μ-law in WAV (tag 0x0007), same ryf WAVE path.
    let source = test_tone_8k(8000);
    let mut encoder = audio_codec::pcmu::PcmuEncoder::new();
    let encoded = encode_telephony(&mut encoder, &source);
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
    // G.722-in-WAV (tags 0x0064 / 0x0065 / 0x028F) is decoded by ryf
    // and produces 2 samples per encoded byte at native 16 kHz.
    // One second natively. A tenth of a second is the same codec path under Miri.
    let n = if cfg!(miri) { 1_600 } else { 16_000 };
    let source: Vec<i16> = (0..n)
        .map(|i| ((i as f32 * 0.03).sin() * 10000.0) as i16)
        .collect();
    let mut encoder = audio_codec::g722::G722Encoder::new();
    let encoded = encode_telephony(&mut encoder, &source);
    for tag in [0x0064u16, 0x0065, 0x028F] {
        let wav = make_compressed_wav(tag, 16000, 8000, &encoded);
        let decoded = decode_audio_bytes(&wav)
            .unwrap_or_else(|e| panic!("G.722 WAV (tag {tag:#06x}) must decode via ryf: {e}"));
        assert_eq!(
            decoded.len(),
            source.len(),
            "G.722 WAV must decode to native 16 kHz (tag {tag:#06x})"
        );
    }
}

#[test]
fn test_wave_g722_malformed_inputs() {
    // Not RIFF at all → symphonia rejects it as an unsupported container.
    assert!(decode_audio_bytes(b"not a wave file").is_err());
    // PCM WAV still decodes.
    let pcm_wav = make_wav_bytes(&[0i16; 32], 16000);
    assert!(decode_audio_bytes(&pcm_wav).is_ok());
    // G.722 tag but no data chunk → error, not a panic.
    let mut header_only = make_compressed_wav(0x0064, 16000, 8000, &[]);
    header_only.truncate(38); // strip the data chunk header + payload
    assert!(
        decode_audio_bytes(&header_only).is_err(),
        "G.722 WAV with no data chunk must error"
    );
    // Truncated data payload must decode the bytes present, not panic.
    let mut enc = audio_codec::g722::G722Encoder::new();
    let encoded = encode_telephony(&mut enc, &[0i16; 320]);
    let mut wav = make_compressed_wav(0x0064, 16000, 8000, &encoded);
    wav.truncate(wav.len() - 3);
    assert!(
        decode_audio_bytes(&wav).is_ok(),
        "truncated G.722 data must not panic"
    );
}

#[test]
#[cfg_attr(
    miri,
    ignore = "fixed-point G.722 compared with ffmpeg; numeric, runs natively"
)]
fn test_decode_audio_bytes_g722_wav_ffmpeg_fixture_matches_reference() {
    // Independent-reference verification: `g722_tone.wav` was ENCODED by
    // ffmpeg (libavcodec G.722, tag 0x028F) and `g722_tone_ffmpeg.pcm` is
    // ffmpeg's own DECODE of it (see scripts/generate_telephony_fixtures.sh).
    // Our ryf G.722 decode is compared against ffmpeg's decode, so the
    // fixed-point port is validated against a second implementation rather
    // than against itself. Tolerance: RMSE below 1% of full scale.
    let wav = include_bytes!("../../../../tests/fixtures/telephony/g722_tone.wav");
    let reference_pcm = include_bytes!("../../../../tests/fixtures/telephony/g722_tone_ffmpeg.pcm");
    let ours = decode_audio_bytes(wav).expect("ffmpeg G.722 WAV must decode");
    let reference: Vec<f32> = reference_pcm
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from(i16::from_le_bytes(*c)) / 32768.0)
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
    let n = if cfg!(miri) { 2_000 } else { 16_000 };
    let source: Vec<f32> = (0..n).map(|i| (i as f32 * 0.02).sin() * 0.5).collect();
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
