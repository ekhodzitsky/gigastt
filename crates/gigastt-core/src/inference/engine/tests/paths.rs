//! File / bytes / channels / cancel paths on the mock engine.

use super::*;
use crate::inference::{TranscribeOverrides, TranscribeRequest, TranscribeSource};
use crate::model::ModelVariant;
use crate::test_support;
use crate::vad::VadConfig;

#[test]
fn test_rnnt_precision_uses_selected_backend_not_build_features() {
    let tmp = tempfile::tempdir().expect("tempdir");
    test_support::write_rnnt_layout(tmp.path()).expect("rnnt layout");
    for (backend, expected_int8) in [("cpu", true), ("candle", false)] {
        // Mock sessions exercise loader metadata, not actual Metal execution.
        let engine = Engine::load_with_factory(
            tmp.path(),
            Some(ModelVariant::Rnnt),
            1,
            1,
            0,
            Box::new(test_support::rnnt_factory()),
            1,
            false,
            backend,
        )
        .expect("mock engine");
        assert_eq!(engine.execution_provider(), backend);
        assert_eq!(engine.is_int8(), expected_int8, "backend: {backend}");
    }
}

#[test]
fn test_engine_config_getters_and_builders() {
    let (engine, _tmp) = test_support::rnnt_engine();
    // This helper injects CPU sessions, even when Candle support is compiled.
    assert_eq!(engine.execution_provider(), "cpu");
    assert!(engine.is_int8());
    assert_eq!(engine.variant(), ModelVariant::Rnnt);
    assert_eq!(engine.vocab_size(), 2);
    assert!(!engine.has_punctuator());
    assert!(!engine.has_itn());
    assert!(!engine.has_vad());
    assert!(!engine.has_hotwords());
    assert_eq!(engine.endpoint_mode(), crate::inference::EndpointMode::Auto);

    let engine = engine
        .with_itn(true)
        .with_punctuator(None)
        .with_vad(None, VadConfig::default())
        .with_endpoint_mode(crate::inference::EndpointMode::Assistant)
        .with_hotwords(&[], 5.0);
    assert!(engine.has_itn());
    assert!(!engine.has_punctuator());
    assert!(!engine.has_vad());
    assert!(!engine.has_hotwords());
    assert_eq!(
        engine.endpoint_mode(),
        crate::inference::EndpointMode::Assistant
    );
    assert_eq!(
        engine.vad_config().threshold,
        VadConfig::default().threshold
    );
    assert_eq!(
        engine
            .apply_text_postprocess(
                "двадцать один".into(),
                true,
                false,
                DecodeControls::default()
            )
            .unwrap(),
        "21"
    );
    assert_eq!(
        engine
            .apply_text_postprocess(
                "двадцать один".into(),
                false,
                true,
                DecodeControls::default()
            )
            .unwrap(),
        "двадцать один"
    );
}

#[test]
fn test_transcribe_bytes_and_file_wrappers() {
    let (engine, tmp) = test_support::rnnt_engine();
    let mut guard = engine.pool.checkout_blocking().expect("checkout");
    let wav = test_support::pcm16_wav(&[0i16; 320], 16_000);

    let from_bytes = engine
        .transcribe_bytes(&wav, &mut guard)
        .expect("bytes wrapper");
    assert!(from_bytes.text.is_empty());
    assert!((from_bytes.duration_s - 320.0 / 16_000.0).abs() < 1e-6);

    let shared = engine
        .transcribe_bytes_shared(bytes::Bytes::from(wav.clone()), &mut guard)
        .expect("shared bytes");
    assert_eq!(shared.duration_s, from_bytes.duration_s);

    let with_overrides = engine
        .transcribe_bytes_shared_with_overrides(
            bytes::Bytes::from(wav.clone()),
            &mut guard,
            &TranscribeOverrides::default(),
        )
        .expect("bytes overrides");
    assert_eq!(with_overrides.duration_s, from_bytes.duration_s);

    let diarized = engine
        .transcribe_bytes_shared_with_overrides_diarized(
            bytes::Bytes::from(wav.clone()),
            &mut guard,
            &TranscribeOverrides::default(),
        )
        .expect("diarized wrapper degrades without a speaker model");
    assert_eq!(diarized.duration_s, from_bytes.duration_s);

    let path = tmp.path().join("clip.wav");
    std::fs::write(&path, &wav).expect("write wav");
    let from_file = engine
        .transcribe_file(path.to_str().expect("utf8"), &mut guard)
        .expect("file wrapper");
    assert!((from_file.duration_s - from_bytes.duration_s).abs() < 1e-6);

    let file_overrides = engine
        .transcribe_file_with_overrides(
            path.to_str().expect("utf8"),
            &mut guard,
            &TranscribeOverrides {
                itn: Some(false),
                ..Default::default()
            },
        )
        .expect("file overrides");
    assert!((file_overrides.duration_s - from_bytes.duration_s).abs() < 1e-6);
}

#[test]
fn test_transcribe_channels_empty_and_one_channel() {
    let (engine, _tmp) = test_support::rnnt_engine();
    let mut guard = engine.pool.checkout_blocking().expect("checkout");

    let empty = engine
        .transcribe_channels(&[], &mut guard)
        .expect("empty channels");
    assert!(empty.text.is_empty());
    assert!(empty.words.is_empty());
    assert_eq!(empty.duration_s, 0.0);

    let one = vec![vec![0.0f32; 160]];
    let result = engine
        .transcribe_channels(&one, &mut guard)
        .expect("one channel");
    assert!((result.duration_s - 160.0 / 16_000.0).abs() < 1e-6);
}

#[test]
fn test_decode_speech_regions_empty_is_noop() {
    let (engine, _tmp) = test_support::rnnt_engine();
    let mut guard = engine.pool.checkout_blocking().expect("checkout");
    let words = engine
        .decode_speech_regions(
            &[0.0; 320],
            &[],
            &mut guard,
            None,
            DecodeControls::default(),
        )
        .expect("empty regions");
    assert!(words.is_empty());
}

#[test]
fn test_decode_words_honours_abort_before_single_pass() {
    let (engine, _tmp) = test_support::rnnt_engine();
    let mut guard = engine.pool.checkout_blocking().expect("checkout");
    let aborted = DecodeControls {
        on_partial: None,
        abort: Some(&|| true),
        on_progress: None,
    };
    let err = engine
        .decode_words(&[0.0; 100], &mut guard, None, aborted)
        .expect_err("cancelled single-pass");
    assert!(matches!(err, crate::error::GigasttError::Cancelled));
}

#[test]
fn test_transcribe_request_abort_flag() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    let (engine, _tmp) = test_support::rnnt_engine();
    let mut guard = engine.pool.checkout_blocking().expect("checkout");
    let flag = Arc::new(AtomicBool::new(true));
    let samples = vec![0.0f32; 100];
    let req = TranscribeRequest::new(TranscribeSource::Samples(&samples)).with_abort(Some(flag));
    let err = engine
        .transcribe_request(req, &mut guard)
        .expect_err("aborted request");
    assert!(matches!(err, crate::error::GigasttError::Cancelled));

    let flag = Arc::new(AtomicBool::new(false));
    flag.store(false, Ordering::Relaxed);
    let req = TranscribeRequest::new(TranscribeSource::Samples(&samples)).with_abort(Some(flag));
    engine
        .transcribe_request(req, &mut guard)
        .expect("not aborted");
}

#[test]
fn test_stream_eligible_without_diarization_request() {
    let (engine, _tmp) = test_support::rnnt_engine();
    assert!(engine.stream_eligible(false));
}

#[test]
fn test_channel_progress_accumulates_across_sample_buffers() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    let (engine, _tmp) = test_support::rnnt_engine();
    let mut guard = engine.pool.checkout_blocking().expect("checkout");
    let channels = vec![vec![0.0; 320], vec![0.0; 480]];
    let progress = Arc::new(AtomicU64::new(0));
    engine
        .transcribe_request(
            TranscribeRequest::new(TranscribeSource::Channels(&channels))
                .with_progress(Some(progress.clone())),
            &mut guard,
        )
        .expect("channels");
    assert_eq!(progress.load(Ordering::Relaxed), 800);

    // Watchdog work retains even one-sample increments across channels.
    let tiny = vec![vec![0.0; 1], vec![0.0; 1]];
    progress.store(0, Ordering::Relaxed);
    engine
        .transcribe_request(
            TranscribeRequest::new(TranscribeSource::Channels(&tiny))
                .with_progress(Some(progress.clone())),
            &mut guard,
        )
        .expect("tiny channels");
    assert_eq!(progress.load(Ordering::Relaxed), 2);

    let observed = std::cell::RefCell::new(Vec::new());
    let report = |n| observed.borrow_mut().push(n);
    engine
        .transcribe_channels_inner(
            &channels,
            &mut guard,
            &TranscribeOverrides::default(),
            None,
            DecodeControls {
                on_progress: Some(&report),
                ..Default::default()
            },
        )
        .expect("channel progress");
    let observed = observed.into_inner();
    assert!(
        observed.windows(2).all(|pair| pair[0] <= pair[1]),
        "{observed:?}"
    );
    assert!(observed.contains(&320));
    assert_eq!(observed.last(), Some(&800));
}

#[test]
fn test_channel_progress_accumulates_across_file_streams() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    let (engine, _tmp) = test_support::rnnt_engine();
    let mut guard = engine.pool.checkout_blocking().expect("checkout");
    let mut wav = test_support::pcm16_wav(&[0; 640], 16_000);
    wav[22..24].copy_from_slice(&2u16.to_le_bytes());
    wav[28..32].copy_from_slice(&64_000u32.to_le_bytes());
    wav[32..34].copy_from_slice(&4u16.to_le_bytes());
    let progress = Arc::new(AtomicU64::new(0));
    let data = bytes::Bytes::from(wav);
    engine
        .transcribe_request(
            TranscribeRequest::new(TranscribeSource::ChannelStreams {
                data: data.clone(),
                channels: 2,
            })
            .with_progress(Some(progress.clone())),
            &mut guard,
        )
        .expect("channel streams");
    assert_eq!(progress.load(Ordering::Relaxed), 640);
    let observed = std::cell::RefCell::new(Vec::new());
    let report = |n| observed.borrow_mut().push(n);
    engine
        .transcribe_channel_streams(
            data,
            2,
            None,
            &mut guard,
            &TranscribeOverrides::default(),
            None,
            DecodeControls {
                on_progress: Some(&report),
                ..Default::default()
            },
        )
        .expect("streamed channel progress");
    let observed = observed.into_inner();
    assert!(
        observed.windows(2).all(|pair| pair[0] <= pair[1]),
        "{observed:?}"
    );
    assert!(observed.contains(&320));
    assert_eq!(observed.last(), Some(&640));
}

#[test]
fn test_channel_progress_does_not_advance_cancelled_channels() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let (engine, _tmp) = test_support::rnnt_engine();
    let mut guard = engine.pool.checkout_blocking().expect("checkout");
    let cancelled = AtomicBool::new(false);
    let observed = std::cell::RefCell::new(Vec::new());
    let report = |n| {
        observed.borrow_mut().push(n);
        cancelled.store(true, Ordering::Relaxed);
    };
    let abort = || cancelled.load(Ordering::Relaxed);
    let result = engine.transcribe_channels_inner(
        &[vec![0.0; 320], vec![0.0; 480]],
        &mut guard,
        &TranscribeOverrides::default(),
        None,
        DecodeControls {
            on_progress: Some(&report),
            abort: Some(&abort),
            ..Default::default()
        },
    );
    assert!(matches!(result, Err(GigasttError::Cancelled)));
    assert_eq!(observed.into_inner(), vec![320]);
}
