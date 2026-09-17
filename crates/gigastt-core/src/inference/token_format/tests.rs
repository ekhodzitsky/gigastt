use super::super::decode;
use super::super::state::WordInfo;
use super::super::tokenizer::Tokenizer;
use super::super::windows::CHUNK_OVERLAP_SAMPLES;
use super::super::{ENCODER_SUBSAMPLING, HOP_LENGTH};
use super::*;

#[test]
fn test_token_formatter_groups_words() {
    // `▁` (U+2581) marks a new word; continuation tokens have no prefix.
    let tok = Tokenizer::from_tokens(vec![
        "\u{2581}hel".into(), // 0: new word
        "lo".into(),          // 1: continuation
        "\u{2581}wor".into(), // 2: new word
        "ld".into(),          // 3: continuation
    ]);
    let tokens = vec![
        decode::TokenInfo {
            token_id: 0,
            frame_index: 0,
            confidence: 0.9,
        },
        decode::TokenInfo {
            token_id: 1,
            frame_index: 1,
            confidence: 0.8,
        },
        decode::TokenInfo {
            token_id: 2,
            frame_index: 2,
            confidence: 0.95,
        },
        decode::TokenInfo {
            token_id: 3,
            frame_index: 3,
            confidence: 0.85,
        },
    ];
    let words = TokenFormatter::tokens_to_words(&tok, &tokens, 0);
    assert_eq!(words.len(), 2);
    assert_eq!(words[0].word, "hello");
    assert_eq!(words[1].word, "world");
    // Mean confidence per word.
    assert!((words[0].confidence - 0.85).abs() < 1e-6);
    assert!((words[1].confidence - 0.90).abs() < 1e-6);
    // Frame timing (SECONDS_PER_FRAME = 0.04).
    assert!((words[0].start - 0.0).abs() < 1e-9);
    assert!((words[0].end - 0.04).abs() < 1e-9);
    assert!((words[1].start - 0.08).abs() < 1e-9);
}

#[test]
fn test_token_formatter_empty_tokens() {
    let tok = Tokenizer::from_tokens(vec!["\u{2581}a".into()]);
    assert!(TokenFormatter::tokens_to_words(&tok, &[], 0).is_empty());
}

#[test]
fn test_token_formatter_frame_offset_shifts_time() {
    let tok = Tokenizer::from_tokens(vec!["\u{2581}x".into()]);
    let tokens = vec![decode::TokenInfo {
        token_id: 0,
        frame_index: 0,
        confidence: 1.0,
    }];
    let words = TokenFormatter::tokens_to_words(&tok, &tokens, 10);
    assert_eq!(words.len(), 1);
    // frame_offset 10 → start = 10 * 0.04 = 0.4.
    assert!((words[0].start - 0.4).abs() < 1e-9);
}

fn word(text: &str, start: f64, end: f64) -> WordInfo {
    WordInfo::new(text, start, end, 1.0, None)
}

#[test]
fn test_stitch_first_chunk_passes_through() {
    // An empty `merged` (the very first chunk) is returned verbatim.
    let next = vec![word("a", 0.0, 0.5), word("b", 0.6, 1.0)];
    let out = stitch_chunk_words(Vec::new(), next.clone(), 11.0);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].word, "a");
    assert_eq!(out[1].word, "b");
}

#[test]
fn test_stitch_preserves_quick_repeated_words() {
    let a = vec![word("да", 22.95, 23.02), word("да", 23.04, 23.11)];
    let b = vec![word("да", 23.04, 23.11), word("да", 23.12, 23.19)];
    let out = stitch_chunk_words(a, b, 23.0);
    assert_eq!(
        out.iter().map(|w| w.word.as_str()).collect::<Vec<_>>(),
        ["да", "да"]
    );
    assert!(out.windows(2).all(|w| w[0].start <= w[1].start));
}

#[test]
fn test_stitch_repetition_count_survives_uniform_timestamp_jitter() {
    for shift in [-0.3, -0.16, -0.04, 0.0, 0.04, 0.16, 0.3] {
        let a: Vec<_> = (0..4)
            .map(|i| word("да", 22.84 + i as f64 * 0.12, 22.92 + i as f64 * 0.12))
            .collect();
        let b = a
            .iter()
            .map(|w| word(&w.word, w.start + shift, w.end + shift))
            .collect();
        let out = stitch_chunk_words(a, b, 23.0);
        assert_eq!(out.len(), 4, "shift {shift}");
        assert!(out.windows(2).all(|w| w[0].start <= w[1].start));
    }
}

#[test]
fn test_stitch_pathological_overlap_uses_bounded_fallback() {
    let a: Vec<_> = (0..100).map(|_| word("да", 22.9, 23.1)).collect();
    let b: Vec<_> = (0..100).map(|_| word("нет", 23.1, 23.3)).collect();
    let out = stitch_chunk_words(a, b, 23.0);
    assert_eq!(out.len(), 200);
    assert!(out.windows(2).all(|w| w[0].start <= w[1].start));
}

#[test]
fn test_stitch_punctuation_only_is_not_an_alignment_anchor() {
    let out = stitch_chunk_words(
        vec![word("...", 22.96, 23.0)],
        vec![word("!", 23.04, 23.1)],
        23.0,
    );
    assert_eq!(out.len(), 2);
}

#[test]
fn test_stitch_matches_case_punctuation_and_yo_without_rewriting_words() {
    let a = vec![word("Ещё,", 22.96, 23.36)];
    let b = vec![word("еще", 23.04, 23.44), word("раз", 23.6, 23.8)];
    let out = stitch_chunk_words(a, b, 23.0);
    assert_eq!(
        out.iter().map(|w| w.word.as_str()).collect::<Vec<_>>(),
        ["Ещё,", "раз"]
    );
}

#[test]
fn test_stitch_does_not_match_distant_repeated_words() {
    let a = vec![word("да", 22.5, 22.7)];
    let b = vec![word("да", 23.5, 23.7)];
    let out = stitch_chunk_words(a, b, 23.0);
    assert_eq!(out.len(), 2);
}

#[test]
fn test_stitch_partial_word_uses_complete_hypothesis() {
    let a = vec![word("говорил", 22.7, 23.1), word("по", 23.5, 24.0)];
    let b = vec![word("говорил", 22.75, 23.15), word("потом", 23.5, 24.2)];
    let out = stitch_chunk_words(a, b, 23.0);
    assert_eq!(
        out.iter().map(|w| w.word.as_str()).collect::<Vec<_>>(),
        ["говорил", "потом"]
    );
}

#[test]
fn test_stitch_unmatched_continuous_speech_keeps_midpoint_fallback() {
    let a = vec![word("один", 22.8, 23.3), word("хвост", 23.4, 23.9)];
    let b = vec![word("другой", 22.9, 23.4), word("далее", 23.5, 24.0)];
    let out = stitch_chunk_words(a, b, 23.0);
    assert_eq!(
        out.iter().map(|w| w.word.as_str()).collect::<Vec<_>>(),
        ["один", "далее"]
    );
}

#[test]
fn test_seam_seconds_is_midpoint_of_actual_overlap() {
    // Regular grid: a 24 s window ending at 24 s and the next one starting at
    // the 22 s stride (352_000) share 2 s, so the seam is 23.0 s.
    assert_eq!(seam_seconds(384_000, 352_000), 23.0);
    // An end-anchored trailing window overlaps more: [44, 68] then
    // [47.25, 71.25] seams at the middle of [47.25, 68].
    assert_eq!(seam_seconds(68 * 16_000, 756_000), (47.25 + 68.0) / 2.0);
    // No overlap (or no previous window): the seam is the window start.
    assert_eq!(seam_seconds(0, 352_000), 22.0);
    assert_eq!(seam_seconds(0, 0), 0.0);
}

#[test]
fn test_stitch_dedups_overlap_no_drop_no_dup() {
    // Two 24s windows with a 22s stride: chunk B starts at 22s, overlap
    // [22s, 24s], seam at 23s. The word "dup" at ~22.5s is decoded by both
    // chunks; the seam attributes it to exactly one. No unique word lost.
    let chunk_a = vec![
        word("first", 1.0, 1.4),    // unique to A
        word("middle", 21.0, 21.4), // unique to A, before overlap
        word("dup", 22.4, 22.8),    // in overlap, before seam → kept from A
    ];
    // B's words are already offset by its 22s start.
    let chunk_b = vec![
        word("dup", 22.5, 22.9),   // same word re-decoded → after-seam copy dropped
        word("later", 25.0, 25.4), // unique to B
        word("end", 40.0, 40.4),   // unique to B
    ];
    let seam_s = 22.0 + CHUNK_OVERLAP_SAMPLES as f64 / 2.0 / 16000.0; // 23.0
    assert!((seam_s - 23.0).abs() < 1e-9);

    let out = stitch_chunk_words(chunk_a, chunk_b, seam_s);
    let texts: Vec<&str> = out.iter().map(|w| w.word.as_str()).collect();
    // "dup" appears exactly once (A's copy, before the seam); nothing dropped.
    assert_eq!(texts, vec!["first", "middle", "dup", "later", "end"]);
    // Monotonic in `start`.
    for w in out.windows(2) {
        assert!(w[0].start <= w[1].start, "not monotonic: {:?}", out);
    }
}

#[test]
fn test_stitch_drops_a_tail_past_seam() {
    // A word decoded by A past the seam is dropped in favour of B's
    // fuller-context copy; the back half of the overlap belongs to B.
    let chunk_a = vec![word("keep", 22.0, 22.4), word("a_tail", 23.5, 23.9)];
    let chunk_b = vec![word("b_seam", 23.2, 23.6), word("b_late", 30.0, 30.4)];
    let out = stitch_chunk_words(chunk_a, chunk_b, 23.0);
    let texts: Vec<&str> = out.iter().map(|w| w.word.as_str()).collect();
    assert_eq!(texts, vec!["keep", "b_seam", "b_late"]);
}

// Matching copies must survive once even when their start times straddle
// the nominal midpoint in either direction.

#[test]
fn test_stitch_straddling_word_survives_once_when_starts_diverge() {
    // The old timestamp cut retained both copies.
    let chunk_a = vec![word("на", 22.0, 22.32), word("мосту", 22.96, 23.36)];
    let chunk_b = vec![word("мосту", 23.04, 23.44), word("стоял", 24.0, 24.4)];
    let out = stitch_chunk_words(chunk_a, chunk_b, 23.0);
    let texts: Vec<&str> = out.iter().map(|w| w.word.as_str()).collect();
    assert_eq!(
        texts,
        vec!["на", "мосту", "стоял"],
        "timestamp jitter must not duplicate the same spoken word"
    );
}

#[test]
fn test_stitch_straddling_word_survives_once_when_starts_converge() {
    // The old timestamp cut discarded both copies.
    let chunk_a = vec![word("на", 22.0, 22.32), word("мосту", 23.04, 23.44)];
    let chunk_b = vec![word("мосту", 22.96, 23.36), word("стоял", 24.0, 24.4)];
    let out = stitch_chunk_words(chunk_a, chunk_b, 23.0);
    let texts: Vec<&str> = out.iter().map(|w| w.word.as_str()).collect();
    assert_eq!(
        texts,
        vec!["на", "мосту", "стоял"],
        "timestamp jitter must not discard both copies of a spoken word"
    );
}

#[test]
fn test_stitch_word_exactly_on_seam_kept_from_earlier_chunk() {
    // Equal midpoint starts choose the earlier copy, preserving its metadata.
    let chunk_a = vec![WordInfo::new("шов", 23.0, 23.4, 0.5, None)];
    let chunk_b = vec![
        WordInfo::new("шов", 23.0, 23.4, 0.9, None),
        word("после", 24.0, 24.4),
    ];
    let out = stitch_chunk_words(chunk_a, chunk_b, 23.0);
    let texts: Vec<&str> = out.iter().map(|w| w.word.as_str()).collect();
    assert_eq!(
        texts,
        vec!["шов", "после"],
        "no duplicate exactly on the seam"
    );
    assert_eq!(
        out[0].confidence, 0.5,
        "the surviving copy comes from the earlier chunk"
    );
}

#[test]
fn test_stitch_empty_next_chunk_preserves_previous_words() {
    // No new hypothesis provides evidence for replacing the old tail.
    let chunk_a = vec![word("до", 22.0, 22.4), word("хвост", 23.04, 23.44)];
    let out = stitch_chunk_words(chunk_a, Vec::new(), 23.0);
    let texts: Vec<&str> = out.iter().map(|w| w.word.as_str()).collect();
    assert_eq!(
        texts,
        vec!["до", "хвост"],
        "an empty hypothesis must not erase words without a replacement"
    );
}

#[test]
fn test_stitch_silence_at_seam_loses_nothing() {
    // A gap without a matching word uses the lossless midpoint fallback.
    let chunk_a = vec![word("перед", 22.6, 22.9)];
    let chunk_b = vec![word("после", 23.1, 23.5), word("конец", 24.0, 24.4)];
    let out = stitch_chunk_words(chunk_a, chunk_b, 23.0);
    let texts: Vec<&str> = out.iter().map(|w| w.word.as_str()).collect();
    assert_eq!(texts, vec!["перед", "после", "конец"]);
    for w in out.windows(2) {
        assert!(w[0].start <= w[1].start, "not monotonic: {:?}", out);
    }
}

#[test]
fn test_stitch_empty_next_preserves_all_words_at_every_seam() {
    // An empty hypothesis preserves the available tail at any cut position.
    let merged: Vec<WordInfo> = (0..40)
        .map(|i| word(&format!("w{i}"), i as f64 * 0.5, i as f64 * 0.5 + 0.3))
        .collect();
    for step in 0..=80 {
        let seam_s = step as f64 * 0.25;
        let expected = merged.clone();
        let got = stitch_chunk_words(merged.clone(), Vec::new(), seam_s);
        assert_eq!(
            got.iter().map(|w| w.word.as_str()).collect::<Vec<_>>(),
            expected.iter().map(|w| w.word.as_str()).collect::<Vec<_>>(),
            "diverged at seam {seam_s}"
        );
    }
}

#[test]
fn test_stitch_timestamp_offset_math() {
    // The chunked path offsets a chunk's frame indices by
    // start_samples / (HOP_LENGTH * ENCODER_SUBSAMPLING). Verify that a word
    // at frame 0 of a chunk starting `start_samples` in lands at the right
    // absolute time via `tokens_to_words` (the same offset the engine feeds).
    let tok = Tokenizer::from_tokens(vec!["\u{2581}w".into()]);
    let tokens = vec![decode::TokenInfo {
        token_id: 0,
        frame_index: 0,
        confidence: 1.0,
    }];
    let start_samples = 16000 * 22; // chunk starts at 22s
    let frame_offset = start_samples / (HOP_LENGTH * ENCODER_SUBSAMPLING);
    let words = TokenFormatter::tokens_to_words(&tok, &tokens, frame_offset);
    assert_eq!(words.len(), 1);
    // frame 0 + offset → absolute start == 22.0s exactly (aligned stride).
    assert!(
        (words[0].start - 22.0).abs() < 1e-9,
        "got {}",
        words[0].start
    );
}

#[test]
fn test_token_formatter_last_word_empty_confidences_defaults_to_one() {
    // A word whose only token is a bare boundary marker (`▁`, no body)
    // contributes no confidence sample; a following real word that itself
    // has no recorded confidences must default to 1.0 on the final-emit
    // path. We build a vocab whose tokens are pure boundary markers so the
    // `clean` body is empty and no confidence is pushed.
    let tok = Tokenizer::from_tokens(vec![
        "\u{2581}real".into(), // 0: a real word
        "\u{2581}".into(),     // 1: bare boundary, empty body
    ]);
    let tokens = vec![
        decode::TokenInfo {
            token_id: 0,
            frame_index: 0,
            confidence: 0.7,
        },
        // A bare boundary token forces emission of "real" (mid-loop emit),
        // then contributes nothing to a new word.
        decode::TokenInfo {
            token_id: 1,
            frame_index: 1,
            confidence: 0.5,
        },
    ];
    let words = TokenFormatter::tokens_to_words(&tok, &tokens, 0);
    // Only "real" is emitted; the trailing bare boundary leaves no word.
    assert_eq!(words.len(), 1);
    assert_eq!(words[0].word, "real");
    assert!((words[0].confidence - 0.7).abs() < 1e-6);
}
