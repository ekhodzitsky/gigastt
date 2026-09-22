use super::*;

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
        diarization: None,
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
        diarization: None,
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
        diarization: None,
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
        diarization: None,
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
        diarization: None,
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
        execution_provider: "cpu".into(),
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
    assert_eq!(v["execution_provider"], "cpu");
    assert_eq!(v["supported_rates"][1], 48000);
}
