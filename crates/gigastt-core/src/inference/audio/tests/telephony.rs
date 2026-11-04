use super::*;

#[test]
fn test_g726_wave_matches_independent_reference_and_streaming() {
    use super::super::stream::FileWindows;
    use bytes::Bytes;

    let fixtures: [(&[u8], &[u8]); 4] = [
        (
            include_bytes!("../../../../tests/fixtures/telephony/g726_16k.wav"),
            include_bytes!("../../../../tests/fixtures/telephony/g726_16k_reference.wav"),
        ),
        (
            include_bytes!("../../../../tests/fixtures/telephony/g726_24k.wav"),
            include_bytes!("../../../../tests/fixtures/telephony/g726_24k_reference.wav"),
        ),
        (
            include_bytes!("../../../../tests/fixtures/telephony/g726_32k.wav"),
            include_bytes!("../../../../tests/fixtures/telephony/g726_32k_reference.wav"),
        ),
        (
            include_bytes!("../../../../tests/fixtures/telephony/g726_40k.wav"),
            include_bytes!("../../../../tests/fixtures/telephony/g726_40k_reference.wav"),
        ),
    ];
    for (encoded, reference) in fixtures {
        // Both signals use the same production resampler; the reference PCM
        // was decoded independently by FFmpeg, never by the updated library.
        let expected = decode_audio_bytes(reference).unwrap();
        // Even the 2-bit fixture must cross the decoder's 4096-byte input
        // block; the 3- and 5-bit rates also exercise carried partial bytes.
        assert!(expected.len() > 32_768, "fixture must cross decode blocks");
        let actual = decode_audio_bytes(encoded).unwrap();
        assert!(!actual.is_empty());
        assert_eq!(actual, expected);
        assert_eq!(
            FileWindows::decode_bytes(Bytes::copy_from_slice(encoded), None).unwrap(),
            expected
        );
        let error =
            FileWindows::decode_bytes(Bytes::copy_from_slice(encoded), Some(0.05)).unwrap_err();
        assert!(matches!(
            error.downcast_ref::<crate::error::GigasttError>(),
            Some(crate::error::GigasttError::AudioTooLong { .. })
        ));
    }
}

#[test]
fn test_wave_ambiguous_0064_is_rejected_in_both_decode_paths() {
    use super::super::stream::FileWindows;
    use bytes::Bytes;

    // A 32 kbit/s header could mean G.726; interpreting it as G.722 would
    // silently return the wrong signal. Valid 64 kbit/s aliases are covered
    // by test_decode_audio_bytes_g722_wav_fallback.
    let wav = make_compressed_wav(0x0064, 8000, 4000, &[0x55; 80]);
    for result in [
        decode_audio_bytes(&wav),
        FileWindows::decode_bytes(Bytes::from(wav), None),
    ] {
        assert!(result.is_err(), "ambiguous codec must not be guessed");
    }
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

#[test]
fn test_in_place_pcm16_precision_matches_wav_for_every_code_and_edges() {
    let mut samples: Vec<f32> = (i16::MIN..=i16::MAX)
        .map(|code| f32::from(code) / 32768.0 + 0.25 / 32768.0)
        .collect();
    samples.extend([
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0,
        0.0,
        -2.0,
        2.0,
    ]);
    let expected = decode_audio_bytes(&encode_wav_pcm16(&samples, 16000)).unwrap();
    quantize_wav_pcm16_in_place(&mut samples);
    assert_eq!(
        samples.iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|s| s.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
#[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
fn test_raw_budget_uses_audio_rate_and_exact_sample_boundaries() {
    for (codec, rate, bytes_per_second) in [
        (TelephonyCodec::Pcmu, 8000, 8000),
        (TelephonyCodec::Pcma, 8000, 8000),
        (TelephonyCodec::Pcmu, 16000, 16000),
        (TelephonyCodec::Pcma, 11025, 11025),
        (TelephonyCodec::Pcma, 48000, 48000),
        (TelephonyCodec::G722, 8000, 8000),
        (TelephonyCodec::G722, 16000, 8000),
    ] {
        let data = vec![0xff; bytes_per_second];
        let exact =
            decode_telephony_raw_bounded_with_abort(&data, codec, rate, Some(1.0), None).unwrap();
        let legacy = decode_telephony_raw(&data, codec, rate).unwrap();
        assert_eq!(exact, legacy);
        for (length, limit) in [
            (bytes_per_second + 1, 1.0),
            (bytes_per_second, f64::from_bits(1.0f64.to_bits() - 1)),
        ] {
            let err = decode_telephony_raw_bounded_with_abort(
                &vec![0xff; length],
                codec,
                rate,
                Some(limit),
                None,
            )
            .unwrap_err();
            assert!(matches!(
                err.downcast_ref::<crate::error::GigasttError>(),
                Some(crate::error::GigasttError::AudioTooLong { .. })
            ));
        }
        // G.722's 8 kHz hint is an RTP clock: one byte still decodes two
        // 16 kHz samples, so a one-sample budget cannot admit one byte.
        if codec == TelephonyCodec::G722 {
            assert!(
                decode_telephony_raw_bounded_with_abort(
                    &[0xff],
                    codec,
                    rate,
                    Some(1.0 / 16000.0),
                    None
                )
                .is_err()
            );
            assert_eq!(
                decode_telephony_raw_bounded_with_abort(
                    &[0xff],
                    codec,
                    rate,
                    Some(2.0 / 16000.0),
                    None
                )
                .unwrap()
                .len(),
                2
            );
        }
    }
}

#[test]
fn test_raw_budget_rejects_before_codec_and_preserves_disabled_limit_semantics() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let checks = AtomicUsize::new(0);
    let abort = || {
        checks.fetch_add(1, Ordering::Relaxed);
        false
    };
    let result = decode_telephony_raw_bounded_with_abort(
        &[0xff; 1000],
        TelephonyCodec::Pcmu,
        16000,
        Some(0.001),
        Some(&abort),
    );
    assert!(result.is_err());
    // No post-codec checkpoint was reached: ryf rejects the known frame count
    // before materializing decoded PCM.
    assert_eq!(checks.load(Ordering::Relaxed), 1);
    for limit in [
        None,
        Some(0.0),
        Some(-1.0),
        Some(f64::NAN),
        Some(f64::INFINITY),
    ] {
        assert_eq!(
            decode_telephony_raw_bounded_with_abort(
                &[0xff; 100],
                TelephonyCodec::Pcmu,
                16000,
                limit,
                None
            )
            .unwrap()
            .len(),
            100
        );
    }
}

#[test]
#[cfg_attr(miri, ignore = "rubato sinc resampler is too slow under Miri")]
fn test_raw_decode_cancels_before_codec_and_between_resampling_stages() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    for stop in 0..=4 {
        let checks = AtomicUsize::new(0);
        let abort = || checks.fetch_add(1, Ordering::Relaxed) >= stop;
        let error = decode_telephony_raw_bounded_with_abort(
            &vec![0xff; 150_000],
            TelephonyCodec::Pcmu,
            8000,
            None,
            Some(&abort),
        )
        .unwrap_err();
        assert!(matches!(
            error.downcast_ref::<crate::error::GigasttError>(),
            Some(crate::error::GigasttError::Cancelled)
        ));
    }
    let error = decode_telephony_raw_bounded_with_abort(
        &[],
        TelephonyCodec::Pcmu,
        1,
        Some(0.001),
        Some(&|| true),
    )
    .unwrap_err();
    assert!(matches!(
        error.downcast_ref::<crate::error::GigasttError>(),
        Some(crate::error::GigasttError::Cancelled)
    ));
}

#[test]
fn test_fixture_mulaw_encoder_preserves_independent_codepoints() {
    // FFmpeg decoded every mu-law byte (0..=255) into this independent PCM
    // reference. Both zero codes decode to zero, whose canonical encoding is FF.
    let reference =
        include_bytes!("../../../../tests/fixtures/telephony/pcmu_codepoints_ffmpeg.pcm");
    assert_eq!(reference.len(), 256 * 2);
    let pcm: Vec<i16> = reference
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&pair| i16::from_le_bytes(pair))
        .collect();
    let canonical: Vec<u8> = (0..=u8::MAX)
        .map(|code| if code == 0x7f { 0xff } else { code })
        .collect();
    let encoded = encode_telephony(&mut audio_codec::pcmu::PcmuEncoder::new(), &pcm);
    assert_eq!(
        encoded, canonical,
        "fixture encoder changed canonical levels"
    );

    let codes: Vec<u8> = (0..=u8::MAX).collect();
    let samples: Vec<f32> = pcm.iter().map(|&x| f32::from(x) / 32768.0).collect();
    let expected = resample(&samples, SampleRate(8000), SampleRate(16000)).unwrap();
    assert_eq!(
        decode_telephony_raw(&codes, TelephonyCodec::Pcmu, 8000).unwrap(),
        expected
    );
}
