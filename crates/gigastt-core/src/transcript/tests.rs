use super::*;

#[test]
fn test_aggregate_confidence_duration_weighted() {
    // The longer word dominates: (0.9*1.0 + 0.5*3.0) / 4.0 = 0.6.
    let words = vec![
        WordInfo::new("short", 0.0, 1.0, 0.9, None),
        WordInfo::new("long", 1.0, 4.0, 0.5, None),
    ];
    let c = aggregate_confidence(&words).expect("non-empty words give a score");
    assert!((c - 0.6).abs() < 1e-6, "duration-weighted mean, got {c}");
}

#[test]
fn test_aggregate_confidence_zero_duration_plain_mean() {
    // Zero-duration words carry no weight; fall back to the plain mean
    // (0.8 + 0.6) / 2 = 0.7 instead of a 0/0 divide.
    let words = vec![
        WordInfo::new("a", 1.0, 1.0, 0.8, None),
        WordInfo::new("b", 2.0, 2.0, 0.6, None),
    ];
    let c = aggregate_confidence(&words).expect("non-empty words give a score");
    assert!((c - 0.7).abs() < 1e-6, "plain mean, got {c}");
}

#[test]
fn test_aggregate_confidence_empty_words_none() {
    assert_eq!(aggregate_confidence(&[]), None);
}

#[test]
fn test_segment_confidence_omitted_from_json_when_none() {
    // Backward-compatible payload: a segment without words must serialize
    // exactly like before the field existed — no `confidence` key at all.
    let seg = TranscriptSegment::empty_final();
    let v = serde_json::to_value(&seg).unwrap();
    assert!(v.get("confidence").is_none());
    assert!(v.get("truncated").is_none());
}

#[test]
fn test_truncated_final_serializes_the_flag_and_keeps_text() {
    let mut seg = TranscriptSegment::empty_final();
    seg.text = "привет".into();
    seg.tentative = seg.text.clone();
    seg.committed.clear();
    let seg = seg.into_truncated_final();
    let v = serde_json::to_value(&seg).unwrap();
    assert_eq!(v["truncated"], true);
    assert_eq!(v["committed"], "привет");
    assert_eq!(v["tentative"], "");
    assert_eq!(v["text"], "привет");
}

#[test]
fn test_segment_old_json_without_confidence_still_deserializes() {
    // Client-side view of the wire contract: payloads written before the
    // `confidence` field existed (no key) must keep parsing into a typed
    // client that knows the field — it simply defaults to `None`. The
    // core segment type is Serialize-only; this mirrors a typed SDK.
    #[derive(serde::Deserialize)]
    struct ClientSegmentView {
        #[serde(default)]
        confidence: Option<f32>,
    }
    let old_json = r#"{"text":"привет","words":[],"is_final":true,"timestamp":1.0}"#;
    let view: ClientSegmentView = serde_json::from_str(old_json).unwrap();
    assert_eq!(view.confidence, None);
}

#[test]
fn test_inference_paths_remain_aliases_and_export_timing_is_unchanged() {
    // These assignments exercise the old public paths, not wrapper conversions.
    let word: crate::inference::WordInfo = WordInfo::new("слово", 0.125, 1.5, 0.75, None);
    let result: crate::inference::TranscribeResult = TranscribeResult {
        text: word.word.clone(),
        words: vec![word],
        duration_s: 1.5,
        confidence: None,
    };
    assert_eq!(
        serde_json::to_value(&result).unwrap(),
        serde_json::json!({
            "text": "слово",
            "words": [{"word": "слово", "start": 0.125, "end": 1.5, "confidence": 0.75}],
            "duration_s": 1.5,
        })
    );
    assert_eq!(
        crate::export::to_srt(&result.words, 80, 14),
        "1\n00:00:00,125 --> 00:00:01,500\nслово\n"
    );
    assert_eq!(
        crate::export::to_vtt(&result.words, 80, 14),
        "WEBVTT\n\n00:00:00.125 --> 00:00:01.500\nслово\n\n"
    );
}

#[test]
fn test_protocol_segment_reasons_and_legacy_omissions() {
    use crate::protocol::ServerMessage;

    let mut segment: crate::inference::TranscriptSegment = TranscriptSegment::empty_final();
    segment.text = "hello".into();
    segment.committed = segment.text.clone();
    segment.timestamp = 1.0;
    segment.is_final = false;
    segment.speech_final = false;
    segment.endpoint_reason = None;
    assert_eq!(
        serde_json::to_value(ServerMessage::Partial(segment.clone())).unwrap(),
        serde_json::json!({
            "type": "partial", "text": "hello", "committed": "hello",
            "tentative": "", "words": [], "is_final": false, "timestamp": 1.0,
        })
    );
    segment.is_final = true;
    segment.speech_final = true;
    for (reason, expected) in [
        (EndpointReason::Vad, "vad"),
        (EndpointReason::Blank, "blank"),
        (EndpointReason::Stop, "stop"),
    ] {
        let reason: crate::inference::EndpointReason = reason;
        segment.endpoint_reason = Some(reason);
        assert_eq!(
            serde_json::to_value(ServerMessage::Final(segment.clone())).unwrap(),
            serde_json::json!({
                "type": "final", "text": "hello", "committed": "hello",
                "tentative": "", "words": [], "is_final": true, "timestamp": 1.0,
                "speech_final": true, "endpoint_reason": expected,
            })
        );
    }
}
