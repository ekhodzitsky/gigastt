//! File / bytes / channels / cancel paths on the mock engine.

use super::*;
use crate::inference::{TranscribeOverrides, TranscribeRequest, TranscribeSource};
use crate::model::ModelVariant;
use crate::test_support;
use crate::vad::VadConfig;

#[test]
fn test_engine_config_getters_and_builders() {
    let (engine, _tmp) = test_support::rnnt_engine();
    // Candle ignores the INT8 ONNX encoder and loads FP32 safetensors, so
    // `Engine::load` reports `is_int8() == false` under that feature.
    assert_eq!(engine.is_int8(), !cfg!(feature = "candle"));
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
        engine.apply_text_postprocess("двадцать один".into(), true, false),
        "21"
    );
    assert_eq!(
        engine.apply_text_postprocess("двадцать один".into(), false, true),
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
