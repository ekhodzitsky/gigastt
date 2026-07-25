//! Unit tests for HTTP handlers (no model required).

use super::super::config::{RuntimeLimits, pool_retry_after_ms, pool_retry_after_secs};
use super::super::metrics::MetricsRegistry;
use super::admin::peer_is_loopback;
use super::error::{
    api_error, api_inference_timeout_error, api_pool_closed_error, api_timeout_error,
};
use super::export::render_export_response;
use super::stream::{StreamError, sse_data_payload};
use super::transcribe::{raw_codec_to_wav, resolve_raw_codec};
use super::*;

use arc_swap::ArcSwap;
use axum::body::Bytes;
use axum::extract::{Query, State};
use axum::http::{StatusCode, header};
use gigastt_core::inference::Engine;
use std::sync::Arc;

#[test]
fn test_health_response_serialization() {
    let resp = HealthResponse {
        status: "ok".into(),
        model: "gigaam-v3-rnnt".into(),
        variant: "rnnt".into(),
        version: "0.3.0".into(),
        punctuation: true,
        itn: true,
    };
    let json = serde_json::to_string(&resp).unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["status"], "ok");
    assert_eq!(v["model"], "gigaam-v3-rnnt");
    assert_eq!(v["variant"], "rnnt");
    assert_eq!(v["punctuation"], true);
    assert_eq!(v["itn"], true);
}

#[test]
fn test_transcribe_response_serialization() {
    let resp = TranscribeResponse {
        text: "hello".into(),
        words: vec![],

        duration: 1.5,
        confidence: None,
        segments: None,
    };
    let json = serde_json::to_string(&resp).unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["text"], "hello");
    assert_eq!(v["duration"], 1.5);
}

#[test]
fn test_readiness_response_ready_serialization() {
    let resp = ReadinessResponse {
        status: "ready".into(),
        pool_available: 3,
        pool_total: 4,
        reason: None,
    };
    let json = serde_json::to_value(&resp).unwrap();
    assert_eq!(json["status"], "ready");
    assert_eq!(json["pool_available"], 3);
    assert_eq!(json["pool_total"], 4);
    assert!(json.get("reason").is_none() || json["reason"].is_null());
}

#[test]
fn test_readiness_response_not_ready_serialization() {
    let resp = ReadinessResponse {
        status: "not_ready".into(),
        pool_available: 0,
        pool_total: 4,
        reason: Some("pool_exhausted".into()),
    };
    let json = serde_json::to_value(&resp).unwrap();
    assert_eq!(json["status"], "not_ready");
    assert_eq!(json["reason"], "pool_exhausted");
}

#[tokio::test]
async fn test_api_error_basic() {
    let resp = api_error(StatusCode::BAD_REQUEST, "bad request", "bad_request");
    let (parts, body) = resp.into_parts();
    assert_eq!(parts.status, StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(body, 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["error"], "bad request");
    assert_eq!(v["code"], "bad_request");
}

#[tokio::test]
async fn test_override_conflict_error_mapping() {
    // The per-request-knob 409s reuse the shared `api_error` machinery, so
    // they must carry StatusCode::CONFLICT and the stable code an operator's
    // client keys off. Drive it via the same `OverrideError::{code,message}`
    // the handler maps, plus the standalone `variant_not_loaded` guard.
    use gigastt_core::inference::OverrideError;
    for err in [
        OverrideError::VadNotLoaded,
        OverrideError::PunctuationNotAvailable,
    ] {
        let resp = api_error(StatusCode::CONFLICT, err.message(), err.code());
        let (parts, body) = resp.into_parts();
        assert_eq!(parts.status, StatusCode::CONFLICT);
        let bytes = axum::body::to_bytes(body, 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["code"], err.code());
        assert!(v["error"].as_str().is_some_and(|s| !s.is_empty()));
    }
    assert_eq!(OverrideError::VadNotLoaded.code(), "vad_not_loaded");
    assert_eq!(
        OverrideError::PunctuationNotAvailable.code(),
        "punctuation_not_available"
    );

    // Hotword DoS limit violations map to 400 (not 409).
    use gigastt_core::inference::HotwordError;
    for err in [HotwordError::TooManyHotwords, HotwordError::PhraseTooLong] {
        let resp = api_error(StatusCode::BAD_REQUEST, err.message(), err.code());
        let (parts, body) = resp.into_parts();
        assert_eq!(parts.status, StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(body, 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["code"], err.code());
    }

    // The variant guard is a standalone literal (no engine needed to check
    // the code/status contract it emits).
    let resp = api_error(
        StatusCode::CONFLICT,
        "Requested model variant is not loaded",
        "variant_not_loaded",
    );
    let (parts, body) = resp.into_parts();
    assert_eq!(parts.status, StatusCode::CONFLICT);
    let bytes = axum::body::to_bytes(body, 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["code"], "variant_not_loaded");
}

#[tokio::test]
async fn test_api_timeout_error_includes_retry_after() {
    let limits = RuntimeLimits::default();
    let resp = api_timeout_error(&limits);
    let (parts, body) = resp.into_parts();
    assert_eq!(parts.status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        parts
            .headers
            .get(header::RETRY_AFTER)
            .unwrap()
            .to_str()
            .unwrap(),
        pool_retry_after_secs(&limits).to_string()
    );
    let bytes = axum::body::to_bytes(body, 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["code"], "timeout");
    assert_eq!(v["retry_after_ms"], pool_retry_after_ms(&limits));
}

#[tokio::test]
async fn test_api_pool_closed_error_no_retry() {
    let resp = api_pool_closed_error();
    let (parts, body) = resp.into_parts();
    assert_eq!(parts.status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(parts.headers.get(header::RETRY_AFTER).is_none());
    let bytes = axum::body::to_bytes(body, 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["code"], "pool_closed");
    assert!(v.get("retry_after_ms").is_none());
}

#[tokio::test]
async fn test_api_inference_timeout_error_is_504() {
    let resp = api_inference_timeout_error();
    let (parts, body) = resp.into_parts();
    assert_eq!(parts.status, StatusCode::GATEWAY_TIMEOUT);
    // A wedged run would just time out again, so no Retry-After hint.
    assert!(parts.headers.get(header::RETRY_AFTER).is_none());
    let bytes = axum::body::to_bytes(body, 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["code"], "inference_timeout");
}

#[test]
fn test_resolve_raw_codec_absent_is_none() {
    let params = ExportParams::default();
    assert!(resolve_raw_codec(&params).unwrap().is_none());
}

#[test]
fn test_resolve_raw_codec_valid_pairs() {
    for (name, rate) in [
        ("pcmu", 8000),
        ("ulaw", 8000),
        ("pcma", 8000),
        ("alaw", 16000),
        ("g722", 8000),
        ("g722", 16000),
    ] {
        let params = ExportParams {
            codec: Some(name.into()),
            sample_rate: Some(rate),
            ..Default::default()
        };
        assert!(
            resolve_raw_codec(&params).unwrap().is_some(),
            "{name}@{rate} must resolve"
        );
    }
}

#[tokio::test]
async fn test_resolve_raw_codec_unknown_codec_is_400() {
    let params = ExportParams {
        codec: Some("g729".into()),
        sample_rate: Some(8000),
        ..Default::default()
    };
    let resp = resolve_raw_codec(&params).unwrap_err();
    let (parts, body) = resp.into_parts();
    assert_eq!(parts.status, StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(body, 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["code"], "unsupported_codec");
}

#[tokio::test]
async fn test_resolve_raw_codec_missing_sample_rate_is_400() {
    let params = ExportParams {
        codec: Some("pcmu".into()),
        sample_rate: None,
        ..Default::default()
    };
    let resp = resolve_raw_codec(&params).unwrap_err();
    let (parts, body) = resp.into_parts();
    assert_eq!(parts.status, StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(body, 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["code"], "invalid_sample_rate");
}

#[tokio::test]
async fn test_resolve_raw_codec_bad_sample_rate_is_400() {
    for (name, rate) in [("pcmu", 4000), ("g722", 44100)] {
        let params = ExportParams {
            codec: Some(name.into()),
            sample_rate: Some(rate),
            ..Default::default()
        };
        let resp = resolve_raw_codec(&params).unwrap_err();
        let (parts, body) = resp.into_parts();
        assert_eq!(parts.status, StatusCode::BAD_REQUEST, "{name}@{rate}");
        let bytes = axum::body::to_bytes(body, 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["code"], "invalid_sample_rate", "{name}@{rate}");
    }
}

#[test]
fn test_raw_codec_to_wav_produces_decodable_wav() {
    // μ-law silence (0xFF ≈ 0) re-wraps into a WAV the standard pipeline
    // accepts: the full raw→16kHz-WAV transform without a model.
    let raw = vec![0xFFu8; 8000]; // 1 s of μ-law silence at 8 kHz
    let wav = raw_codec_to_wav(
        &raw,
        gigastt_core::inference::audio::TelephonyCodec::Pcmu,
        8000,
    )
    .unwrap();
    let samples = gigastt_core::inference::audio::decode_audio_bytes_shared(wav).unwrap();
    assert!(
        samples.len() > 12_000 && samples.len() <= 16_000,
        "expected ~1 s at 16 kHz, got {}",
        samples.len()
    );
    assert!(
        samples.iter().all(|s| s.abs() < 0.01),
        "μ-law silence must decode to near-silence"
    );
}

#[test]
fn test_raw_codec_to_wav_rejects_bad_input() {
    let result = raw_codec_to_wav(
        &[],
        gigastt_core::inference::audio::TelephonyCodec::Pcmu,
        8000,
    );
    assert!(result.is_err(), "empty raw payload must error");
}

#[test]
fn test_export_params_parse_codec_query() {
    // The query string is how REST clients pass the pair; pin it through
    // axum's Query extractor itself (the exact extraction path the
    // handler uses).
    let uri: axum::http::Uri = "http://localhost/v1/transcribe?codec=pcmu&sample_rate=8000"
        .parse()
        .unwrap();
    let axum::extract::Query(params) =
        axum::extract::Query::<ExportParams>::try_from_uri(&uri).expect("codec query must parse");
    assert_eq!(params.codec.as_deref(), Some("pcmu"));
    assert_eq!(params.sample_rate, Some(8000));
}

#[test]
fn test_sse_data_payload_preserves_error_codes() {
    // Per-variant code is preserved (not collapsed to a generic string),
    // including the distinct inference_panic / inference_timeout events.
    for code in [
        "invalid_audio",
        "inference_error",
        "inference_panic",
        "inference_timeout",
    ] {
        let payload = sse_data_payload(&Err(StreamError {
            code,
            message: "sanitized".into(),
        }));
        let v: serde_json::Value = serde_json::from_str(&payload).unwrap();
        assert_eq!(v["type"], "error");
        assert_eq!(v["code"], code);
        assert_eq!(v["message"], "sanitized");
    }
}

#[test]
fn test_sse_data_payload_segment_framing() {
    // A final segment renders as type "final"; a non-final one as "partial".
    let seg = gigastt_core::inference::TranscriptSegment::empty_final();
    let final_payload = sse_data_payload(&Ok(seg));
    let v: serde_json::Value = serde_json::from_str(&final_payload).unwrap();
    assert_eq!(v["type"], "final");

    let mut partial = gigastt_core::inference::TranscriptSegment::empty_final();
    partial.is_final = false;
    let partial_payload = sse_data_payload(&Ok(partial));
    let v: serde_json::Value = serde_json::from_str(&partial_payload).unwrap();
    assert_eq!(v["type"], "partial");
}

#[test]
fn test_sse_data_payload_confidence_present_only_when_some() {
    // A segment with words carries the aggregate; an empty one omits the
    // key entirely, matching the WS payload contract.
    let mut seg = gigastt_core::inference::TranscriptSegment::empty_final();
    seg.confidence = Some(0.85);
    let payload = sse_data_payload(&Ok(seg));
    let v: serde_json::Value = serde_json::from_str(&payload).unwrap();
    let c = v["confidence"].as_f64().expect("numeric confidence");
    assert!((c - 0.85).abs() < 1e-6, "got {c}");

    let empty = gigastt_core::inference::TranscriptSegment::empty_final();
    let payload = sse_data_payload(&Ok(empty));
    let v: serde_json::Value = serde_json::from_str(&payload).unwrap();
    assert!(v.get("confidence").is_none());
}

#[tokio::test]
#[ignore = "requires model"]
async fn test_readiness_when_shutdown_cancelled() {
    let state = Arc::new(AppState {
        engine: engine_swap(test_engine()),
        limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
        metrics_registry: None,
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        shutdown: tokio_util::sync::CancellationToken::new(),
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: None,
    });
    state.shutdown.cancel();
    let resp = readiness(State(state)).await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let bytes = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["status"], "not_ready");
    assert_eq!(v["reason"], "shutting_down");
}

#[tokio::test]
#[ignore = "requires model"]
async fn test_readiness_when_pool_exhausted() {
    let engine = fresh_engine();
    let _guards: Vec<_> = (0..engine.pool.total())
        .map(|_| engine.pool.checkout_blocking().unwrap())
        .collect();
    let state = Arc::new(AppState {
        engine: engine_swap(engine),
        limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
        metrics_registry: None,
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        shutdown: tokio_util::sync::CancellationToken::new(),
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: None,
    });
    let resp = readiness(State(state)).await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let bytes = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["status"], "not_ready");
    assert_eq!(v["reason"], "pool_exhausted");
}

#[tokio::test]
#[ignore = "requires model"]
async fn test_transcribe_payload_too_large() {
    let state = Arc::new(AppState {
        engine: engine_swap(test_engine()),
        limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits {
            body_limit_bytes: 10,
            ..RuntimeLimits::default()
        })),
        metrics_registry: None,
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        shutdown: tokio_util::sync::CancellationToken::new(),
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: None,
    });
    let body = Bytes::from(vec![0u8; 100]);
    let result = transcribe(State(state), Query(ExportParams::default()), body).await;
    match result {
        Err(resp) => assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE),
        Ok(_) => panic!("expected payload_too_large error"),
    }
}

#[tokio::test]
#[ignore = "requires model"]
async fn test_transcribe_channels_split_diarization_conflict_returns_400() {
    let state = Arc::new(AppState {
        engine: engine_swap(test_engine()),
        limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
        metrics_registry: None,
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        shutdown: tokio_util::sync::CancellationToken::new(),
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: None,
    });
    let params = ExportParams {
        channels: Some("split".into()),
        diarization: Some(true),
        ..ExportParams::default()
    };
    let resp = transcribe(State(state), Query(params), minimal_wav())
        .await
        .unwrap_err();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["code"], "conflicting_modes");
}

#[tokio::test]
#[ignore = "requires model"]
async fn test_models_with_metrics() {
    let state = Arc::new(AppState {
        engine: engine_swap(test_engine()),
        limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
        metrics_registry: Some(Arc::new(MetricsRegistry::new())),
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        shutdown: tokio_util::sync::CancellationToken::new(),
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: None,
    });
    let resp = models(State(state)).await;
    let json = serde_json::to_value(&*resp).unwrap();
    // The id reflects the head actually loaded on disk (rnnt or e2e_rnnt),
    // not a hardcoded literal, so assert the stable shape instead.
    let id = json["id"].as_str().unwrap();
    assert!(
        id == "gigaam-v3-rnnt" || id == "gigaam-v3-e2e-rnnt",
        "unexpected model id: {id}"
    );
    assert_eq!(
        json["variant"],
        if id.contains("e2e") {
            "e2e_rnnt"
        } else {
            "rnnt"
        }
    );
}

#[tokio::test]
#[ignore = "requires model"]
async fn test_readiness_with_metrics() {
    let state = Arc::new(AppState {
        engine: engine_swap(fresh_engine()),
        limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
        metrics_registry: Some(Arc::new(MetricsRegistry::new())),
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        shutdown: tokio_util::sync::CancellationToken::new(),
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: None,
    });
    let resp = readiness(State(state)).await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
#[ignore = "requires model"]
async fn test_transcribe_pool_closed() {
    let engine = fresh_engine();
    engine.pool.close();
    let state = Arc::new(AppState {
        engine: engine_swap(engine),
        limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
        metrics_registry: None,
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        shutdown: tokio_util::sync::CancellationToken::new(),
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: None,
    });
    let body = Bytes::from(vec![0u8; 100]);
    let result = transcribe(State(state), Query(ExportParams::default()), body).await;
    match result {
        Err(resp) => assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE),
        Ok(_) => panic!("expected pool_closed error"),
    }
}

#[tokio::test]
#[ignore = "requires model"]
async fn test_transcribe_stream_invalid_audio() {
    let state = Arc::new(AppState {
        engine: engine_swap(test_engine()),
        limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
        metrics_registry: None,
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        shutdown: tokio_util::sync::CancellationToken::new(),
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: None,
    });
    let body = Bytes::from(vec![0u8; 100]);
    let result = transcribe_stream(State(state), body).await;
    match result {
        Err(resp) => assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY),
        Ok(_) => panic!("expected invalid_audio error"),
    }
}

#[tokio::test]
#[ignore = "requires model"]
async fn test_transcribe_stream_payload_too_large() {
    let state = Arc::new(AppState {
        engine: engine_swap(test_engine()),
        limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits {
            body_limit_bytes: 10,
            ..RuntimeLimits::default()
        })),
        metrics_registry: None,
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        shutdown: tokio_util::sync::CancellationToken::new(),
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: None,
    });
    let body = Bytes::from(vec![0u8; 100]);
    let result = transcribe_stream(State(state), body).await;
    match result {
        Err(resp) => assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE),
        Ok(_) => panic!("expected payload_too_large error"),
    }
}

#[tokio::test]
#[ignore = "requires model"]
async fn test_transcribe_stream_pool_closed() {
    let engine = fresh_engine();
    engine.pool.close();
    let state = Arc::new(AppState {
        engine: engine_swap(engine),
        limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
        metrics_registry: None,
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        shutdown: tokio_util::sync::CancellationToken::new(),
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: None,
    });
    let body = minimal_wav();
    let result = transcribe_stream(State(state), body).await;
    match result {
        Err(resp) => assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE),
        Ok(_) => panic!("expected pool_closed error"),
    }
}

#[tokio::test]
#[ignore = "requires model"]
async fn test_transcribe_with_metrics() {
    let state = Arc::new(AppState {
        engine: engine_swap(test_engine()),
        limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
        metrics_registry: Some(Arc::new(MetricsRegistry::new())),
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        shutdown: tokio_util::sync::CancellationToken::new(),
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: None,
    });
    let body = short_wav();
    match transcribe(State(state), Query(ExportParams::default()), body).await {
        Ok(_) => {}
        Err(_) => panic!("transcribe with metrics failed"),
    }
}

#[tokio::test]
#[ignore = "requires model"]
async fn test_transcribe_stream_with_metrics() {
    let state = Arc::new(AppState {
        engine: engine_swap(test_engine()),
        limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
        metrics_registry: Some(Arc::new(MetricsRegistry::new())),
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        shutdown: tokio_util::sync::CancellationToken::new(),
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: None,
    });
    let body = short_wav();
    match transcribe_stream(State(state), body).await {
        Ok(_) => {}
        Err(_) => panic!("transcribe_stream with metrics failed"),
    }
}

#[tokio::test]
#[ignore = "requires model"]
async fn test_transcribe_segments_json() {
    // `?segments=true` on the default JSON response adds a `segments` array
    // with sane start/end ordering and per-segment words, while keeping the
    // top-level text/words/duration contract.
    let state = Arc::new(AppState {
        engine: engine_swap(test_engine()),
        limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
        metrics_registry: None,
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        shutdown: tokio_util::sync::CancellationToken::new(),
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: None,
    });
    let params = ExportParams {
        segments: Some(true),
        ..ExportParams::default()
    };
    let resp = transcribe(State(state), Query(params), short_wav())
        .await
        .expect("transcribe with segments should succeed");
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    // Top-level contract is preserved.
    assert!(v.get("text").is_some());
    assert!(v.get("words").is_some());
    assert!(v.get("duration").is_some());
    // The segments array is present and every segment has monotonic timing.
    let segments = v["segments"].as_array().expect("segments array present");
    for seg in segments {
        let start = seg["start"].as_f64().unwrap();
        let end = seg["end"].as_f64().unwrap();
        assert!(end >= start, "segment end {end} < start {start}");
        assert!(seg["words"].is_array());
    }
}

fn sample_export_result() -> gigastt_core::inference::TranscribeResult {
    use gigastt_core::inference::WordInfo;
    gigastt_core::inference::TranscribeResult {
        text: "привет мир".into(),
        words: vec![
            WordInfo::new("привет", 0.0, 0.5, 0.98, Some(0)),
            WordInfo::new("мир", 0.6, 1.0, 0.97, Some(0)),
        ],
        duration_s: 1.0,
        confidence: None,
    }
}

#[tokio::test]
async fn test_render_export_default_returns_none() {
    let result = sample_export_result();
    let params = ExportParams::default();
    assert!(render_export_response(&result, &params).unwrap().is_none());
}

#[tokio::test]
async fn test_render_export_txt() {
    let result = sample_export_result();
    let params = ExportParams {
        format: Some("txt".into()),
        ..ExportParams::default()
    };
    let resp = render_export_response(&result, &params).unwrap().unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    assert_eq!(body, "привет мир");
}

#[tokio::test]
async fn test_render_export_srt_content_type() {
    let result = sample_export_result();
    let params = ExportParams {
        format: Some("srt".into()),
        ..ExportParams::default()
    };
    let resp = render_export_response(&result, &params).unwrap().unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers().get(header::CONTENT_TYPE).unwrap(),
        "application/x-subrip; charset=utf-8"
    );
    let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("[SPEAKER_0] привет мир"));
}

#[tokio::test]
async fn test_render_export_vtt_download_header() {
    let result = sample_export_result();
    let params = ExportParams {
        format: Some("vtt".into()),
        download: Some("recording.vtt".into()),
        ..ExportParams::default()
    };
    let resp = render_export_response(&result, &params).unwrap().unwrap();
    assert_eq!(
        resp.headers().get(header::CONTENT_DISPOSITION).unwrap(),
        "attachment; filename=\"recording.vtt\"; filename*=UTF-8''recording.vtt"
    );
}

#[tokio::test]
async fn test_render_export_download_filename_with_control_char_does_not_panic() {
    // The download filename is user-controlled; control characters must not
    // produce an invalid header value / panic — they are sanitized out of
    // the quoted fallback and percent-encoded in `filename*`.
    let result = sample_export_result();
    let params = ExportParams {
        format: Some("srt".into()),
        download: Some("evil\r\nInjected: x".into()),
        ..ExportParams::default()
    };
    let resp = render_export_response(&result, &params).unwrap().unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers().get(header::CONTENT_DISPOSITION).unwrap(),
        "attachment; filename=\"evil__Injected: x\"; filename*=UTF-8''evil%0D%0AInjected%3A%20x"
    );
}

#[tokio::test]
async fn test_render_export_invalid_format() {
    let result = sample_export_result();
    let params = ExportParams {
        format: Some("docx".into()),
        ..ExportParams::default()
    };
    let err = render_export_response(&result, &params).unwrap_err();
    assert_eq!(err.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_render_export_invalid_format_body_code() {
    // The invalid-format error carries the machine-readable `invalid_format`
    // code so clients can distinguish it from other 400s.
    let result = sample_export_result();
    let params = ExportParams {
        format: Some("xml".into()),
        ..ExportParams::default()
    };
    let err = render_export_response(&result, &params).unwrap_err();
    assert_eq!(err.status(), StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(err.into_body(), 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["code"], "invalid_format");
}

#[tokio::test]
async fn test_render_export_uppercase_json_returns_none() {
    // Format negotiation is case-insensitive: an explicit (any-case) "json"
    // means "keep the default TranscribeResponse contract", so the helper
    // returns None instead of building a Response.
    let result = sample_export_result();
    let params = ExportParams {
        format: Some("JSON".into()),
        ..ExportParams::default()
    };
    assert!(render_export_response(&result, &params).unwrap().is_none());
}

#[tokio::test]
async fn test_render_export_uppercase_format_renders() {
    // Non-JSON format strings are also case-insensitive (parsed via
    // ExportFormat::from_str), so "SRT" still renders subtitles.
    let result = sample_export_result();
    let params = ExportParams {
        format: Some("SRT".into()),
        ..ExportParams::default()
    };
    let resp = render_export_response(&result, &params).unwrap().unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers().get(header::CONTENT_TYPE).unwrap(),
        "application/x-subrip; charset=utf-8"
    );
}

#[tokio::test]
async fn test_render_export_empty_download_uses_default_name() {
    // An empty `download` value still requests an attachment; the helper
    // synthesizes the default `transcript.<ext>` filename.
    let result = sample_export_result();
    let params = ExportParams {
        format: Some("vtt".into()),
        download: Some(String::new()),
        ..ExportParams::default()
    };
    let resp = render_export_response(&result, &params).unwrap().unwrap();
    assert_eq!(
        resp.headers().get(header::CONTENT_DISPOSITION).unwrap(),
        "attachment; filename=\"transcript.vtt\"; filename*=UTF-8''transcript.vtt"
    );
}

#[tokio::test]
async fn test_render_export_download_filename_injection_neutralized() {
    // A crafted `download` value trying to splice a second `filename*`
    // parameter must survive only as inert data: the quote becomes `_` in
    // the fallback, and the `filename*` bytes are percent-encoded, so the
    // spoofed `spoofed.exe` never appears as a real header parameter.
    let result = sample_export_result();
    let params = ExportParams {
        format: Some("srt".into()),
        download: Some("evil\"; filename*=UTF-8''spoofed.exe".into()),
        ..ExportParams::default()
    };
    let resp = render_export_response(&result, &params).unwrap().unwrap();
    assert_eq!(
        resp.headers().get(header::CONTENT_DISPOSITION).unwrap(),
        "attachment; filename=\"evil_; filename*=UTF-8''spoofed.exe\"; \
         filename*=UTF-8''evil%22%3B%20filename%2A%3DUTF-8%27%27spoofed.exe"
    );
}

#[tokio::test]
async fn test_render_export_download_filename_unicode_percent_encoded() {
    // Non-ASCII names get an ASCII-safe fallback for legacy clients and
    // keep the full UTF-8 name percent-encoded in `filename*` (RFC 6266).
    let result = sample_export_result();
    let params = ExportParams {
        format: Some("txt".into()),
        download: Some("é.txt".into()),
        ..ExportParams::default()
    };
    let resp = render_export_response(&result, &params).unwrap().unwrap();
    assert_eq!(
        resp.headers().get(header::CONTENT_DISPOSITION).unwrap(),
        "attachment; filename=\"_.txt\"; filename*=UTF-8''%C3%A9.txt"
    );
}

#[tokio::test]
async fn test_render_export_download_filename_cyrillic_percent_encoded() {
    // Cyrillic input must not leak raw non-ASCII bytes into the header:
    // the fallback replaces each non-ASCII character, `filename*` carries
    // the percent-encoded UTF-8, and the whole value stays ASCII.
    let result = sample_export_result();
    let params = ExportParams {
        format: Some("srt".into()),
        download: Some("отчёт.srt".into()),
        ..ExportParams::default()
    };
    let resp = render_export_response(&result, &params).unwrap().unwrap();
    let value = resp
        .headers()
        .get(header::CONTENT_DISPOSITION)
        .unwrap()
        .to_str()
        .unwrap();
    let encoded_name = format!("{}.srt", "%d0%be%d1%82%d1%87%d1%91%d1%82".to_uppercase());
    assert_eq!(
        value,
        format!("attachment; filename=\"_____.srt\"; filename*=UTF-8''{encoded_name}")
    );
    assert!(value.is_ascii());
}

#[tokio::test]
async fn test_render_export_md_includes_word_timestamps() {
    // The Markdown path honours `word_timestamps` and renders the per-word
    // table; the content type is text/markdown.
    let result = sample_export_result();
    let params = ExportParams {
        format: Some("md".into()),
        word_timestamps: Some(true),
        ..ExportParams::default()
    };
    let resp = render_export_response(&result, &params).unwrap().unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers().get(header::CONTENT_TYPE).unwrap(),
        "text/markdown; charset=utf-8"
    );
    let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("# Transcript"));
    assert!(text.contains("| Word | Start | End |"));
}

#[tokio::test]
async fn test_render_export_line_break_opts_passed_through() {
    // Tight per-line caps must be threaded into RenderOpts so the rendered
    // subtitles actually break — proving the params override the defaults.
    let result = sample_export_result();
    let loose = ExportParams {
        format: Some("srt".into()),
        ..ExportParams::default()
    };
    let tight = ExportParams {
        format: Some("srt".into()),
        max_words_per_line: Some(1),
        ..ExportParams::default()
    };
    let loose_resp = render_export_response(&result, &loose).unwrap().unwrap();
    let tight_resp = render_export_response(&result, &tight).unwrap().unwrap();
    let loose_body = axum::body::to_bytes(loose_resp.into_body(), 4096)
        .await
        .unwrap();
    let tight_body = axum::body::to_bytes(tight_resp.into_body(), 4096)
        .await
        .unwrap();
    let loose_text = String::from_utf8(loose_body.to_vec()).unwrap();
    let tight_text = String::from_utf8(tight_body.to_vec()).unwrap();
    // One word per line yields one cue per word (more "-->" arrows) than the
    // default 14-words-per-line grouping.
    let loose_cues = loose_text.matches("-->").count();
    let tight_cues = tight_text.matches("-->").count();
    assert!(
        tight_cues > loose_cues,
        "tight={tight_cues} should exceed loose={loose_cues}"
    );
}

#[tokio::test]
async fn test_render_export_md_segments_emits_headers() {
    // `format=md` + `segments=true` switches Markdown to `### [mm:ss]`
    // section headers over the cue boundaries, dropping the flat
    // `# Transcript` blob.
    let result = sample_export_result();
    let params = ExportParams {
        format: Some("md".into()),
        segments: Some(true),
        ..ExportParams::default()
    };
    let resp = render_export_response(&result, &params).unwrap().unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers().get(header::CONTENT_TYPE).unwrap(),
        "text/markdown; charset=utf-8"
    );
    let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("### [00:00]"));
    assert!(text.contains("[SPEAKER_0] привет мир"));
    // Segment mode replaces the flat transcript blob.
    assert!(!text.contains("# Transcript"));
}

#[tokio::test]
async fn test_render_export_md_without_segments_unchanged() {
    // Plain `format=md` (no segments) keeps the existing frontmatter +
    // `# Transcript` layout — segment mode is strictly opt-in.
    let result = sample_export_result();
    let params = ExportParams {
        format: Some("md".into()),
        ..ExportParams::default()
    };
    let resp = render_export_response(&result, &params).unwrap().unwrap();
    let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("# Transcript"));
    assert!(!text.contains("### ["));
}

#[tokio::test]
async fn test_render_export_segments_ignored_for_srt() {
    // `segments=true` is a JSON/Markdown affordance; SRT is already
    // cue-based and must render identically with or without the flag.
    let result = sample_export_result();
    let plain = ExportParams {
        format: Some("srt".into()),
        ..ExportParams::default()
    };
    let with_segments = ExportParams {
        format: Some("srt".into()),
        segments: Some(true),
        ..ExportParams::default()
    };
    let a = render_export_response(&result, &plain).unwrap().unwrap();
    let b = render_export_response(&result, &with_segments)
        .unwrap()
        .unwrap();
    let a_body = axum::body::to_bytes(a.into_body(), 4096).await.unwrap();
    let b_body = axum::body::to_bytes(b.into_body(), 4096).await.unwrap();
    assert_eq!(a_body, b_body);
}

#[test]
fn test_transcribe_response_omits_segments_when_none() {
    // The default response must be byte-identical to the pre-feature shape:
    // no `segments` key when the caller didn't ask for it.
    let resp = TranscribeResponse {
        text: "hello".into(),
        words: vec![],
        duration: 1.5,
        confidence: None,
        segments: None,
    };
    let v = serde_json::to_value(&resp).unwrap();
    assert!(v.get("segments").is_none());
    assert_eq!(v["text"], "hello");
    assert_eq!(v["duration"], 1.5);
}

#[test]
fn test_transcribe_response_confidence_present_only_when_some() {
    // With words decoded, the aggregate rides the top-level response;
    // without words the key is omitted so the response shape matches the
    // pre-field contract exactly.
    let with = TranscribeResponse {
        text: "hello".into(),
        words: vec![],
        duration: 1.5,
        confidence: Some(0.87),
        segments: None,
    };
    let v = serde_json::to_value(&with).unwrap();
    let c = v["confidence"].as_f64().expect("numeric confidence");
    assert!((c - 0.87).abs() < 1e-6, "got {c}");

    let without = TranscribeResponse {
        text: String::new(),
        words: vec![],
        duration: 0.0,
        confidence: None,
        segments: None,
    };
    let v = serde_json::to_value(&without).unwrap();
    assert!(v.get("confidence").is_none());
}

#[test]
fn test_transcribe_response_includes_segments_when_present() {
    use gigastt_core::export::to_segments;
    use gigastt_core::inference::WordInfo;
    let words = vec![
        WordInfo::new("привет", 0.0, 0.5, 0.98, None),
        WordInfo::new("мир", 0.6, 1.0, 0.97, None),
    ];
    let resp = TranscribeResponse {
        text: "привет мир".into(),
        words: words.clone(),
        duration: 1.0,
        confidence: None,
        segments: Some(to_segments(&words, 80, 14)),
    };
    let v = serde_json::to_value(&resp).unwrap();
    let segments = v["segments"].as_array().unwrap();
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0]["start"], 0.0);
    assert_eq!(segments[0]["end"], 1.0);
    assert_eq!(segments[0]["text"], "привет мир");
    assert_eq!(segments[0]["words"][0]["word"], "привет");
}

#[test]
fn test_export_params_deserialize_from_query() {
    // The query-param shape drives format negotiation; confirm axum's Query
    // extractor maps every field so the handler sees the caller's choices.
    let uri: axum::http::Uri = "http://x/?format=srt&download=out.srt&max_chars_per_line=20&max_words_per_line=3&word_timestamps=true&segments=true&channels=split&diarization=true"
        .parse()
        .unwrap();
    let Query(params): Query<ExportParams> = Query::try_from_uri(&uri).unwrap();
    assert_eq!(params.format.as_deref(), Some("srt"));
    assert_eq!(params.download.as_deref(), Some("out.srt"));
    assert_eq!(params.max_chars_per_line, Some(20));
    assert_eq!(params.max_words_per_line, Some(3));
    assert_eq!(params.word_timestamps, Some(true));
    assert_eq!(params.segments, Some(true));
    assert_eq!(params.channels.as_deref(), Some("split"));
    assert_eq!(params.diarization, Some(true));
}

#[test]
fn test_export_params_default_empty_query() {
    // No query params -> all None, which the handler maps to JSON defaults.
    let uri: axum::http::Uri = "http://x/".parse().unwrap();
    let Query(params): Query<ExportParams> = Query::try_from_uri(&uri).unwrap();
    assert!(params.format.is_none());
    assert!(params.download.is_none());
    assert!(params.max_chars_per_line.is_none());
    // The per-request knob overrides default to absent so the handler falls
    // back to the engine's boot policy (byte-unchanged response).
    assert!(params.punctuation.is_none());
    assert!(params.itn.is_none());
    assert!(params.vad.is_none());
    assert!(params.hotwords.is_none());
    assert!(params.hotwords_boost.is_none());
    assert!(params.variant.is_none());
}

#[test]
fn test_transcribe_knob_params_deserialize_from_query() {
    // `?punctuation=false&itn=false&vad=false&variant=rnnt` maps to
    // `Some(false)`/`Some("rnnt")`, letting the handler override the boot
    // policy per request.
    let uri: axum::http::Uri = "http://x/?punctuation=false&itn=false&vad=false&variant=rnnt"
        .parse()
        .unwrap();
    let Query(params): Query<ExportParams> = Query::try_from_uri(&uri).unwrap();
    assert_eq!(params.punctuation, Some(false));
    assert_eq!(params.itn, Some(false));
    assert_eq!(params.vad, Some(false));
    assert_eq!(params.variant.as_deref(), Some("rnnt"));

    // The `true` direction deserializes symmetrically.
    let uri: axum::http::Uri = "http://x/?punctuation=true&itn=true&vad=true"
        .parse()
        .unwrap();
    let Query(params): Query<ExportParams> = Query::try_from_uri(&uri).unwrap();
    assert_eq!(params.punctuation, Some(true));
    assert_eq!(params.itn, Some(true));
    assert_eq!(params.vad, Some(true));
}

#[test]
fn test_hotwords_query_param_parsing() {
    // Comma-separated phrases + optional boost deserialize from the query
    // string and map to HotwordOverride via hotwords_from_export_params.
    let uri: axum::http::Uri =
        "http://x/?hotwords=%D1%81%D0%B1%D0%B5%D1%80,%D1%82%D0%B8%D0%BD%D1%8C%D0%BA%D0%BE%D1%84%D1%84&hotwords_boost=3.5"
            .parse()
            .unwrap();
    let Query(params): Query<ExportParams> = Query::try_from_uri(&uri).unwrap();
    assert_eq!(params.hotwords.as_deref(), Some("сбер,тинькофф"));
    assert_eq!(params.hotwords_boost, Some(3.5));

    let hw = hotwords_from_export_params(&params).expect("hotwords present");
    assert_eq!(hw.phrases, vec!["сбер".to_string(), "тинькофф".to_string()]);
    assert_eq!(hw.boost, Some(3.5));

    // Absent hotwords → engine default (None), even if boost is set alone.
    let uri: axum::http::Uri = "http://x/?hotwords_boost=9".parse().unwrap();
    let Query(params): Query<ExportParams> = Query::try_from_uri(&uri).unwrap();
    assert!(hotwords_from_export_params(&params).is_none());

    // Empty key present → Some(empty) force-off.
    let uri: axum::http::Uri = "http://x/?hotwords=".parse().unwrap();
    let Query(params): Query<ExportParams> = Query::try_from_uri(&uri).unwrap();
    let hw = hotwords_from_export_params(&params).expect("key present means override");
    assert!(hw.phrases.is_empty());

    assert_eq!(
        parse_hotwords_query(" сбер , , тинькофф "),
        vec!["сбер".to_string(), "тинькофф".to_string()]
    );
    assert!(parse_hotwords_query("").is_empty());
    assert!(parse_hotwords_query(",,,").is_empty());
}

#[test]
fn test_model_info_serialization_shape() {
    // ModelInfo is the /v1/models contract; assert the field names/values
    // clients depend on are present and correctly typed.
    let info = ModelInfo {
        id: "gigaam-v3-rnnt".into(),
        name: "GigaAM v3 RNN-T".into(),
        variant: "rnnt".into(),
        version: "0.9.0".into(),
        encoder: "int8".into(),
        vocab_size: 34,
        sample_rate: 16000,
        pool_size: 4,
        pool_available: 3,
        supported_formats: vec!["wav".into(), "mp3".into()],
        supported_rates: vec![16000, 48000],
        punctuation: true,
        itn: true,
        diarization: false,
    };
    let v = serde_json::to_value(&info).unwrap();
    assert_eq!(v["id"], "gigaam-v3-rnnt");
    assert_eq!(v["variant"], "rnnt");
    assert_eq!(v["encoder"], "int8");
    assert_eq!(v["vocab_size"], 34);
    assert_eq!(v["sample_rate"], 16000);
    assert_eq!(v["punctuation"], true);
    assert_eq!(v["itn"], true);
    assert_eq!(v["diarization"], false);
    assert_eq!(v["supported_rates"][1], 48000);
}

#[tokio::test]
async fn test_api_inference_timeout_error_body_message() {
    // The 504 inference-timeout body should not leak internals, just the
    // stable code + a sanitized message.
    let resp = api_inference_timeout_error();
    let bytes = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["code"], "inference_timeout");
    assert_eq!(v["error"], "Inference timed out.");
}

#[tokio::test]
async fn test_api_pool_closed_error_status_and_message() {
    // pool_closed is a 503 with a sanitized "shutting down" message and no
    // retry hint (the pool is not coming back).
    let resp = api_pool_closed_error();
    let (parts, body) = resp.into_parts();
    assert_eq!(parts.status, StatusCode::SERVICE_UNAVAILABLE);
    let bytes = axum::body::to_bytes(body, 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["error"], "Server is shutting down");
    assert_eq!(v["code"], "pool_closed");
}

#[test]
fn test_peer_is_loopback_guard() {
    use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr};
    // IPv4 + IPv6 loopback are accepted regardless of source port.
    assert!(peer_is_loopback(&SocketAddr::from((
        Ipv4Addr::LOCALHOST,
        5000
    ))));
    assert!(peer_is_loopback(&SocketAddr::from((
        Ipv6Addr::LOCALHOST,
        5000
    ))));
    // A non-loopback peer (LAN / public) is rejected — reload must stay local
    // even under --bind-all / --cors-allow-any.
    assert!(!peer_is_loopback(&SocketAddr::from((
        Ipv4Addr::new(192, 168, 1, 10),
        9876
    ))));
    assert!(!peer_is_loopback(&SocketAddr::from((
        Ipv4Addr::new(10, 0, 0, 1),
        9876
    ))));
    assert!(!peer_is_loopback(&SocketAddr::from((
        Ipv4Addr::new(8, 8, 8, 8),
        443
    ))));
}

#[tokio::test]
#[ignore = "requires model"]
async fn test_reload_rejects_non_loopback_peer() {
    // The loopback guard fires before any engine work: a non-loopback
    // ConnectInfo yields 403 `loopback_only` even with a builder present.
    // Model-gated only because `AppState` needs a concrete `Engine`; the
    // pure guard logic is covered model-free by `test_peer_is_loopback_guard`.
    use std::net::{Ipv4Addr, SocketAddr};
    let state = Arc::new(AppState {
        engine: engine_swap(test_engine()),
        engine_builder: Some(Arc::new(|| {
            anyhow::bail!("builder must not run for a rejected peer")
        })),
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
        metrics_registry: None,
        shutdown: tokio_util::sync::CancellationToken::new(),
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: None,
    });
    let peer = SocketAddr::from((Ipv4Addr::new(203, 0, 113, 7), 40000));
    let resp = reload(axum::extract::ConnectInfo(peer), State(state)).await;
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let bytes = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["code"], "loopback_only");
}

#[tokio::test]
#[ignore = "requires model"]
async fn test_reload_unsupported_when_no_builder() {
    // A loopback peer with no stored builder (the thin `run_with_shutdown` /
    // test path) gets `reload_unsupported`, not a swap.
    use std::net::{Ipv4Addr, SocketAddr};
    let state = Arc::new(AppState {
        engine: engine_swap(test_engine()),
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
        metrics_registry: None,
        shutdown: tokio_util::sync::CancellationToken::new(),
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: None,
    });
    let peer = SocketAddr::from((Ipv4Addr::LOCALHOST, 40000));
    let resp = reload(axum::extract::ConnectInfo(peer), State(state)).await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let bytes = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["code"], "reload_unsupported");
}

#[test]
fn test_sse_data_payload_includes_words_and_timestamp() {
    // A successful segment carries text, timestamp and words through
    // unchanged so SSE clients can render word-level UI.
    use gigastt_core::inference::WordInfo;
    let mut seg = gigastt_core::inference::TranscriptSegment::empty_final();
    seg.text = "привет".into();
    seg.timestamp = 1.25;
    seg.words = vec![WordInfo::new("привет", 0.0, 0.5, 0.99, Some(0))];
    let payload = sse_data_payload(&Ok(seg));
    let v: serde_json::Value = serde_json::from_str(&payload).unwrap();
    assert_eq!(v["type"], "final");
    assert_eq!(v["text"], "привет");
    assert_eq!(v["timestamp"], 1.25);
    assert_eq!(v["words"][0]["word"], "привет");
}

fn test_engine() -> Arc<Engine> {
    use std::sync::OnceLock;
    static ENGINE: OnceLock<Arc<Engine>> = OnceLock::new();
    ENGINE
        .get_or_init(|| {
            Arc::new(
                Engine::load_with_pool_size(&gigastt_core::model::default_model_dir(), 1).unwrap(),
            )
        })
        .clone()
}

fn fresh_engine() -> Arc<Engine> {
    Arc::new(Engine::load_with_pool_size(&gigastt_core::model::default_model_dir(), 1).unwrap())
}

/// Wrap an engine handle in the `ArcSwap` the `AppState` now holds. Keeps
/// the model-gated test constructors terse after the hot-reload swap change.
fn engine_swap(engine: Arc<Engine>) -> Arc<ArcSwap<Engine>> {
    Arc::new(ArcSwap::new(engine))
}

fn minimal_wav() -> Bytes {
    let data_size = 4u32;
    let file_size = 44 + data_size - 8;
    let mut wav = vec![];
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&file_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&16000u32.to_le_bytes());
    wav.extend_from_slice(&(16000u32 * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());
    wav.extend_from_slice(&0i16.to_le_bytes());
    wav.extend_from_slice(&0i16.to_le_bytes());
    Bytes::from(wav)
}

fn short_wav() -> Bytes {
    let sample_rate = 16000u32;
    let duration_s = 0.1f32;
    let num_samples = (sample_rate as f32 * duration_s) as u32;
    let data_size = num_samples * 2;
    let file_size = 44 + data_size - 8;
    let mut wav = vec![];
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&file_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());
    for _ in 0..num_samples {
        wav.extend_from_slice(&0i16.to_le_bytes());
    }
    Bytes::from(wav)
}
