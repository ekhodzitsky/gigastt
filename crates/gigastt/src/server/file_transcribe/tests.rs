use super::*;

fn channel_fixture(channels: u16, dual_mono: bool) -> Bytes {
    let frames = 1600u32;
    let data_bytes = frames * u32::from(channels) * 2;
    let mut wav = Vec::new();
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_bytes).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&channels.to_le_bytes());
    wav.extend_from_slice(&16000u32.to_le_bytes());
    wav.extend_from_slice(&(16000 * u32::from(channels) * 2).to_le_bytes());
    wav.extend_from_slice(&(channels * 2).to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_bytes.to_le_bytes());
    for frame in 0..frames {
        for channel in 0..channels {
            let frequency = if channel == 0 || dual_mono {
                440.0
            } else {
                710.0
            };
            let sample = ((frame as f32 * frequency * std::f32::consts::TAU / 16000.0).sin()
                * 10000.0) as i16;
            wav.extend_from_slice(&sample.to_le_bytes());
        }
    }
    Bytes::from(wav)
}

#[test]
fn test_cancelled_preparation_never_decodes_or_transcribes() {
    for split_channels in [false, true] {
        for raw_codec in [
            None,
            Some((gigastt_core::inference::audio::TelephonyCodec::Pcmu, 8000)),
        ] {
            let opts = FileTranscribeOpts {
                split_channels,
                raw_codec,
                abort: Some(Arc::new(AtomicBool::new(true))),
                ..Default::default()
            };
            let result = with_file_transcribe_request(Bytes::new(), &opts, true, |_| {
                panic!("cancelled preparation reached inference")
            });
            assert!(matches!(result, Err::<(), _>(GigasttError::Cancelled)));
        }
    }
}

#[test]
fn test_channel_routing_preserves_request_context() {
    for (fixture, channels, dual_mono) in [
        (channel_fixture(1, false), 1, false),
        (channel_fixture(2, true), 2, true),
        (channel_fixture(2, false), 2, false),
        (
            Bytes::from_static(include_bytes!(
                "../../../../gigastt-core/tests/fixtures/opus/dual.ogg"
            )),
            2,
            true,
        ),
        (
            Bytes::from_static(include_bytes!(
                "../../../../gigastt-core/tests/fixtures/opus/late_stereo.ogg"
            )),
            2,
            false,
        ),
    ] {
        for has_vad in [false, true] {
            for vad in [None, Some(false), Some(true)] {
                for enabled in [false, true] {
                    for split_channels in [false, true] {
                        let opts = FileTranscribeOpts {
                            overrides: TranscribeOverrides {
                                punctuation: Some(enabled),
                                itn: Some(!enabled),
                                vad,
                            },
                            hotwords: Some(HotwordOverride::new(vec!["тест".into()], Some(3.0))),
                            split_channels,
                            diarization: true,
                            abort: Some(Arc::new(AtomicBool::new(false))),
                            partial: Some(Arc::new(Default::default())),
                            progress: Some(Arc::new(AtomicU64::new(0))),
                            progress_channels: Some(Arc::new(AtomicU64::new(0))),
                            diarization_outcome: Some(Arc::new(Default::default())),
                            max_audio_secs: Some(3.0),
                            file_zero_context: false,
                            ..Default::default()
                        };
                        with_file_transcribe_request(fixture.clone(), &opts, has_vad, |request| {
                            let split = split_channels && channels == 2 && !dual_mono;
                            assert_eq!(
                                opts.progress_channels
                                    .as_ref()
                                    .unwrap()
                                    .load(Ordering::Relaxed),
                                if split { 2 } else { 1 },
                            );
                            match &request.source {
                                TranscribeSource::Bytes(_) => assert!(!split),
                                TranscribeSource::Channels(audio) => {
                                    assert!(split && has_vad && vad.unwrap_or(true));
                                    assert_eq!(audio.len(), 2);
                                }
                                TranscribeSource::ChannelStreams { channels, .. } => {
                                    assert!(split && !(has_vad && vad.unwrap_or(true)));
                                    assert_eq!(*channels, 2);
                                }
                                _ => panic!("unexpected source"),
                            }
                            assert_eq!(request.overrides.punctuation, opts.overrides.punctuation);
                            assert_eq!(request.overrides.itn, opts.overrides.itn);
                            assert_eq!(request.overrides.vad, opts.overrides.vad);
                            assert!(std::ptr::eq(
                                request.hotwords.unwrap(),
                                opts.hotwords.as_ref().unwrap()
                            ));
                            assert!(Arc::ptr_eq(
                                request.abort.as_ref().unwrap(),
                                opts.abort.as_ref().unwrap()
                            ));
                            assert!(Arc::ptr_eq(
                                request.partial.as_ref().unwrap(),
                                opts.partial.as_ref().unwrap()
                            ));
                            assert!(Arc::ptr_eq(
                                request.progress.as_ref().unwrap(),
                                opts.progress.as_ref().unwrap()
                            ));
                            assert_eq!(request.max_audio_secs, opts.max_audio_secs);
                            // Channel speaker labels take precedence over offline diarization,
                            // including mono fallback, as on the existing split path.
                            assert_eq!(request.diarization, !split_channels);
                            if split_channels {
                                assert!(request.diarization_outcome.is_none());
                            } else {
                                assert!(Arc::ptr_eq(
                                    request.diarization_outcome.as_ref().unwrap(),
                                    opts.diarization_outcome.as_ref().unwrap()
                                ));
                            }
                            Ok(())
                        })
                        .unwrap();
                    }
                }
            }
        }
    }
}

#[test]
fn test_raw_pcm_routing_preserves_legacy_samples_and_request_context() {
    use gigastt_core::inference::audio::{
        TelephonyCodec, decode_audio_bytes, decode_telephony_raw, encode_wav_pcm16,
    };
    let body: Bytes = (0..8000).map(|i| i as u8).collect::<Vec<_>>().into();
    for codec in [
        TelephonyCodec::Pcmu,
        TelephonyCodec::Pcma,
        TelephonyCodec::G722,
    ] {
        let decoded = decode_telephony_raw(&body, codec, 8000).unwrap();
        let expected = decode_audio_bytes(&encode_wav_pcm16(&decoded, 16000)).unwrap();
        for split_channels in [false, true] {
            let opts = FileTranscribeOpts {
                raw_codec: Some((codec, 8000)),
                split_channels,
                diarization: true,
                overrides: TranscribeOverrides {
                    punctuation: Some(false),
                    itn: Some(true),
                    vad: Some(true),
                },
                hotwords: Some(HotwordOverride::new(vec!["тест".into()], Some(3.0))),
                abort: Some(Arc::new(AtomicBool::new(false))),
                partial: Some(Arc::new(Default::default())),
                progress: Some(Arc::new(AtomicU64::new(17))),
                progress_channels: Some(Arc::new(AtomicU64::new(0))),
                diarization_outcome: Some(Arc::new(Default::default())),
                max_audio_secs: Some(2.0),
                file_zero_context: false,
            };
            with_file_transcribe_request(body.clone(), &opts, true, |request| {
                let TranscribeSource::Samples(samples) = request.source else {
                    panic!("raw audio should bypass the WAV container");
                };
                assert_eq!(
                    samples.iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|s| s.to_bits()).collect::<Vec<_>>()
                );
                assert_eq!(request.overrides.punctuation, Some(false));
                assert_eq!(request.overrides.itn, Some(true));
                assert_eq!(request.overrides.vad, Some(true));
                assert!(std::ptr::eq(
                    request.hotwords.unwrap(),
                    opts.hotwords.as_ref().unwrap()
                ));
                assert!(Arc::ptr_eq(
                    request.abort.as_ref().unwrap(),
                    opts.abort.as_ref().unwrap()
                ));
                assert!(Arc::ptr_eq(
                    request.partial.as_ref().unwrap(),
                    opts.partial.as_ref().unwrap()
                ));
                assert!(Arc::ptr_eq(
                    request.progress.as_ref().unwrap(),
                    opts.progress.as_ref().unwrap()
                ));
                assert_eq!(request.max_audio_secs, opts.max_audio_secs);
                assert_eq!(request.diarization, !split_channels);
                assert_eq!(request.diarization_outcome.is_some(), !split_channels);
                assert_eq!(
                    opts.progress_channels
                        .as_ref()
                        .unwrap()
                        .load(Ordering::Relaxed),
                    1
                );
                Ok(())
            })
            .unwrap();
        }
    }
}

#[test]
fn test_raw_routing_threshold_and_one_sample_over_preserve_precision() {
    use gigastt_core::inference::audio::{
        TelephonyCodec, decode_audio_bytes, decode_telephony_raw, encode_wav_pcm16,
    };
    for samples in [RAW_PCM_MAX_SAMPLES, RAW_PCM_MAX_SAMPLES + 1] {
        // G.711 at 16 kHz decodes one byte to exactly one output sample.
        let body: Bytes = (0..samples).map(|i| i as u8).collect::<Vec<_>>().into();
        let expected = encode_wav_pcm16(
            &decode_telephony_raw(&body, TelephonyCodec::Pcma, 16000).unwrap(),
            16000,
        );
        let opts = FileTranscribeOpts {
            raw_codec: Some((TelephonyCodec::Pcma, 16000)),
            ..Default::default()
        };
        with_file_transcribe_request(body, &opts, false, |request| {
            match request.source {
                TranscribeSource::Samples(pcm) => {
                    assert_eq!(samples, RAW_PCM_MAX_SAMPLES);
                    assert_eq!(pcm, decode_audio_bytes(&expected).unwrap());
                }
                TranscribeSource::Bytes(wav) => {
                    assert_eq!(samples, RAW_PCM_MAX_SAMPLES + 1);
                    assert_eq!(wav.as_ref(), expected);
                }
                _ => panic!("unexpected raw source"),
            }
            Ok(())
        })
        .unwrap();
    }
}

#[test]
fn test_raw_preparation_checks_cancellation_after_both_representations() {
    use gigastt_core::inference::audio::TelephonyCodec;
    use std::sync::atomic::AtomicUsize;
    for samples in [160, RAW_PCM_MAX_SAMPLES + 1] {
        let body = vec![0xff; samples];
        let checks = AtomicUsize::new(0);
        prepare_raw_audio(
            &body,
            TelephonyCodec::Pcmu,
            16000,
            None,
            Some(&|| {
                checks.fetch_add(1, Ordering::Relaxed);
                false
            }),
        )
        .unwrap();
        let final_check = checks.load(Ordering::Relaxed) - 1;
        checks.store(0, Ordering::Relaxed);
        let result = prepare_raw_audio(
            &body,
            TelephonyCodec::Pcmu,
            16000,
            None,
            Some(&|| checks.fetch_add(1, Ordering::Relaxed) == final_check),
        );
        assert!(matches!(result, Err(GigasttError::Cancelled)));
    }
}

#[test]
fn test_raw_pcm_duration_budget_rejects_before_engine_routing() {
    use gigastt_core::inference::audio::TelephonyCodec;
    for codec in [
        TelephonyCodec::Pcmu,
        TelephonyCodec::Pcma,
        TelephonyCodec::G722,
    ] {
        let opts = FileTranscribeOpts {
            raw_codec: Some((codec, 8000)),
            max_audio_secs: Some(0.01),
            ..Default::default()
        };
        let result =
            with_file_transcribe_request(Bytes::from(vec![0xff; 8000]), &opts, false, |_| Ok(()));
        assert!(matches!(result, Err(GigasttError::AudioTooLong { .. })));
    }
}

#[test]
fn test_file_transcribe_opts_default() {
    let opts = FileTranscribeOpts::default();
    assert!(!opts.split_channels);
    assert!(!opts.diarization);
    assert!(opts.hotwords.is_none());
    assert!(opts.raw_codec.is_none());
    assert!(opts.overrides.punctuation.is_none());
    assert!(opts.overrides.itn.is_none());
    assert!(opts.overrides.vad.is_none());
}

#[test]
fn test_prepare_raw_audio_rejects_empty_pcmu() {
    // Empty PCMU body is invalid for telephony decode.
    let err = prepare_raw_audio(
        &[],
        gigastt_core::inference::audio::TelephonyCodec::Pcmu,
        8000,
        None,
        None,
    )
    .unwrap_err();
    match err {
        GigasttError::InvalidAudio { .. } => {}
        other => panic!("expected InvalidAudio, got {other:?}"),
    }
}

#[test]
fn test_file_zero_context_routes_default_off_and_opt_in_to_native_request() {
    for enabled in [false, true] {
        let opts = FileTranscribeOpts {
            file_zero_context: enabled,
            ..Default::default()
        };
        with_file_transcribe_request(Bytes::from_static(b"file"), &opts, false, |request| {
            assert!(matches!(request.source, TranscribeSource::Bytes(_)));
            assert_eq!(request.file_zero_context, enabled);
            Ok(())
        })
        .unwrap();
    }
}
