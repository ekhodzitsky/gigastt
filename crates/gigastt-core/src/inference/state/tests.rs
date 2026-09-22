use super::super::{N_FFT, N_MELS, PRED_HIDDEN};
use super::*;

fn word(text: &str, start: f64, end: f64) -> WordInfo {
    WordInfo::new(text, start, end, 1.0, None)
}

#[test]
fn test_stream_text_parts_preserve_spaces_and_reset_after_final() {
    let mut asm = TranscriptAssembler::new();
    asm.append(vec![word("привет", 0.0, 0.5)]);
    asm.commit_live();
    asm.set_words(vec![word("мир", 0.5, 1.0)]);
    let partial = serde_json::to_value(asm.partial(1.0)).unwrap();
    assert_eq!(partial["committed"], "привет");
    assert_eq!(partial["tentative"], " мир");
    assert_eq!(partial["text"], "привет мир");

    let final_ = serde_json::to_value(asm.finalize(2.0)).unwrap();
    assert_eq!(final_["committed"], "привет мир");
    assert_eq!(final_["tentative"], "");
    asm.append(vec![word("снова", 1.0, 1.5)]);
    let next = serde_json::to_value(asm.partial(3.0)).unwrap();
    assert_eq!(next["committed"], "");
    assert_eq!(next["tentative"], "снова");
}

#[test]
fn test_decoder_state_new_zeros() {
    let blank_id = 1024;
    let state = DecoderState::new(blank_id);
    assert!(state.h.iter().all(|&v| v == 0.0));
    assert!(state.c.iter().all(|&v| v == 0.0));
    assert_eq!(state.prev_token, blank_id as i64);
}

#[test]
fn test_decoder_state_dimensions() {
    let state = DecoderState::new(1024);
    assert_eq!(state.h.len(), PRED_HIDDEN);
    assert_eq!(state.c.len(), PRED_HIDDEN);
}

#[test]
fn test_decoder_state_custom_blank_id() {
    let state = DecoderState::new(42);
    assert_eq!(state.prev_token, 42);
}

#[test]
fn test_feature_extractor_default() {
    let _fe = FeatureExtractor::default();
}

#[test]
fn test_transcript_assembler_default() {
    let ta = TranscriptAssembler::default();
    assert!(ta.is_empty());
}

#[test]
fn test_transcript_assembler_append_and_finalize() {
    let mut asm = TranscriptAssembler::new();
    assert!(asm.is_empty());
    asm.append(vec![
        WordInfo {
            word: "hello".into(),
            start: 0.0,
            end: 0.5,
            confidence: 0.9,
            speaker: None,
        },
        WordInfo {
            word: "world".into(),
            start: 0.6,
            end: 1.0,
            confidence: 0.85,
            speaker: None,
        },
    ]);
    assert!(!asm.is_empty());
    let seg = asm.finalize(1.0);
    assert_eq!(seg.text, "hello world");
    assert_eq!(seg.words.len(), 2);
    assert!(seg.is_final);
    assert!(seg.speech_final);
    assert_eq!(seg.endpoint_reason, Some(EndpointReason::Stop));
    assert_eq!(seg.timestamp, 1.0);
    // After finalize the assembler is reset.
    assert!(asm.is_empty());
}

#[test]
fn test_transcript_assembler_commit_live_preserves_prefix_across_set_words() {
    let mut asm = TranscriptAssembler::new();
    asm.append(vec![WordInfo::new("hello", 0.0, 0.5, 0.9, None)]);
    asm.commit_live();
    // Window slide re-decode replaces only the live tail.
    asm.set_words(vec![WordInfo::new("world", 0.6, 1.0, 0.8, None)]);
    let partial = asm.partial(1.0);
    assert!(!partial.is_final);
    assert!(!partial.speech_final);
    assert_eq!(partial.text, "hello world");
    assert_eq!(partial.words.len(), 2);
    let fin = asm.finalize_with_reason(2.0, EndpointReason::Vad);
    assert!(fin.speech_final);
    assert_eq!(fin.endpoint_reason, Some(EndpointReason::Vad));
    assert_eq!(fin.text, "hello world");
    assert!(asm.is_empty());
}

#[test]
fn test_transcript_assembler_commit_prefix_partial_commit_keeps_tail_live() {
    let mut asm = TranscriptAssembler::new();
    asm.set_words(vec![
        WordInfo::new("a", 0.0, 0.5, 0.9, None),
        WordInfo::new("b", 0.5, 1.0, 0.9, None),
        WordInfo::new("c", 1.0, 1.5, 0.9, None),
    ]);
    assert_eq!(asm.commit_prefix(2), 2);
    // Committed prefix + still-live tail both surface in partials.
    let p = asm.partial(1.0);
    assert_eq!(p.text, "a b c");
    assert_eq!(asm.live_word_count(), 1);
    assert_eq!(asm.committed_coverage_end(), Some(1.0));
    // A re-decode may still revise the live tail; the committed prefix ("a b")
    // stays put.
    asm.set_words(vec![WordInfo::new("cee", 1.0, 1.6, 0.9, None)]);
    assert_eq!(asm.partial(1.1).text, "a b cee");
    // Committing more than live is clamped; zero is a no-op.
    assert_eq!(asm.commit_prefix(0), 0);
    assert_eq!(asm.commit_prefix(42), 1);
    let fin = asm.finalize(2.0);
    assert_eq!(fin.text, "a b cee");
    assert!(asm.is_empty());
    assert_eq!(asm.committed_coverage_end(), None);
}

#[test]
fn test_endpoint_mode_parse_token() {
    assert_eq!(EndpointMode::parse_token("auto"), Some(EndpointMode::Auto));
    assert_eq!(
        EndpointMode::parse_token("ASSISTANT"),
        Some(EndpointMode::Assistant)
    );
    assert_eq!(
        EndpointMode::parse_token("manual"),
        Some(EndpointMode::Manual)
    );
    assert_eq!(EndpointMode::parse_token("nope"), None);
}

#[test]
fn test_transcript_assembler_partial() {
    let mut asm = TranscriptAssembler::new();
    asm.append(vec![WordInfo {
        word: "partial".into(),
        start: 0.0,
        end: 0.3,
        confidence: 0.8,
        speaker: None,
    }]);
    let seg = asm.partial(0.3);
    assert_eq!(seg.text, "partial");
    assert!(!seg.is_final);
    // partial must not reset the assembler.
    assert!(!asm.is_empty());
}

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
fn test_assembler_fills_segment_confidence() {
    let mut asm = TranscriptAssembler::new();
    asm.append(vec![
        WordInfo::new("hello", 0.0, 0.5, 0.9, None),
        WordInfo::new("world", 0.5, 1.0, 0.8, None),
    ]);
    let partial = asm.partial(0.5);
    let c = partial.confidence.expect("partial carries the aggregate");
    assert!(
        (c - 0.85).abs() < 1e-6,
        "equal durations → plain mean, got {c}"
    );
    let final_seg = asm.finalize(1.0);
    let c = final_seg.confidence.expect("final carries the aggregate");
    assert!((c - 0.85).abs() < 1e-6, "got {c}");
    // An empty assembler yields an empty segment with no score.
    let empty = asm.finalize(1.0);
    assert_eq!(empty.confidence, None);
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
fn test_feature_extractor_compute_empty() {
    let fe = FeatureExtractor::new();
    let (mel, frames) = fe.compute(&[]);
    // When samples are shorter than N_FFT, compute_with_buffers returns
    // a single zero-filled frame with n_mels elements.
    assert_eq!(mel.len(), N_MELS);
    assert_eq!(frames, 1);
    assert!(mel.iter().all(|&v| v == 0.0));
}

// ---- More pure (no-model) coverage -------------------------------------

#[test]
fn test_transcript_assembler_set_words_overwrites() {
    // The sliding-window streaming path overwrites (not appends) on each
    // re-decode via `set_words`. A second call replaces the first.
    let mut asm = TranscriptAssembler::new();
    asm.set_words(vec![word("alpha", 0.0, 0.4), word("beta", 0.5, 0.9)]);
    let p = asm.partial(0.0);
    assert_eq!(p.text, "alpha beta");
    assert_eq!(p.words.len(), 2);

    asm.set_words(vec![word("gamma", 1.0, 1.4)]);
    let p = asm.partial(0.0);
    assert_eq!(p.text, "gamma", "set_words must overwrite, not append");
    assert_eq!(p.words.len(), 1);
}

#[test]
fn test_transcript_assembler_set_words_empty_resets_text() {
    let mut asm = TranscriptAssembler::new();
    asm.set_words(vec![word("x", 0.0, 0.4)]);
    assert!(!asm.is_empty());
    asm.set_words(vec![]);
    assert!(asm.is_empty(), "empty set_words clears the accumulation");
}

#[test]
fn test_feature_extractor_prepare_buffer_accumulates() {
    // `prepare_buffer` appends to the buffer and reports the usable sample
    // count once a full frame is available; below N_FFT it returns None.
    let fe = FeatureExtractor::new();
    let mut buf: Vec<f32> = Vec::new();
    // A handful of samples — fewer than N_FFT — yields no usable frame yet.
    let usable = fe.prepare_buffer(&[0.1; 10], &mut buf);
    assert_eq!(usable, None, "sub-frame input is buffered, not yet usable");
    assert_eq!(buf.len(), 10, "samples are retained in the buffer");

    // Append enough to cross a frame boundary; a usable count is reported.
    let usable = fe.prepare_buffer(&vec![0.2; N_FFT], &mut buf);
    assert!(
        usable.is_some(),
        "crossing a frame boundary yields a usable count"
    );
}

#[test]
#[cfg_attr(miri, ignore = "mel FFT over 1s of audio is too slow under Miri")]
fn test_feature_extractor_compute_mel_reuses_buffers() {
    // `compute_mel` writes into caller-owned scratch buffers and returns the
    // frame count. One second of 16 kHz audio → ~100 frames; the output
    // buffer holds frames * N_MELS values.
    let fe = FeatureExtractor::new();
    let samples = vec![0.0f32; 16000];
    let mut fft_buf = Vec::new();
    let mut power_buf = Vec::new();
    let mut out_buf = Vec::new();
    let frames = fe.compute_mel(&samples, &mut fft_buf, &mut power_buf, &mut out_buf);
    assert!(frames > 0, "1s of audio yields at least one mel frame");
    assert_eq!(
        out_buf.len(),
        frames * N_MELS,
        "output buffer holds frames * N_MELS values"
    );
}
