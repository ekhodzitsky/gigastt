use super::*;

#[test]
fn test_abort_between_ctc_frames_keeps_prefix() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let lp = logits(&[0, 1, 0, 1], 3);
    let biaser = Biaser::from_sequences(vec![vec![99]], 3.0).unwrap();
    for beam in [false, true] {
        let checks = AtomicUsize::new(0);
        let abort = || checks.fetch_add(1, Ordering::Relaxed) >= 2;
        let tokens = if beam {
            ctc_prefix_beam_decode_with_abort(&lp, 4, 3, 2, &biaser, Some(&abort))
        } else {
            ctc_greedy_decode_with_abort(&lp, 4, 3, 2, Some(&abort))
        };
        assert_eq!(
            tokens.iter().map(|t| t.token_id).collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert!(tokens.iter().all(|t| t.frame_index < 2));
    }
}

/// Build a `[t, vocab]` row-major log-prob buffer where frame `t` argmaxes to
/// `ids[t]`.
fn logits(ids: &[usize], vocab: usize) -> Vec<f32> {
    let mut lp = vec![-10.0f32; ids.len() * vocab];
    for (t, &id) in ids.iter().enumerate() {
        lp[t * vocab + id] = 5.0;
    }
    lp
}

#[test]
fn collapses_repeats_and_drops_blank() {
    // vocab=3, blank=2. frames: a a <blk> b  ->  [a, b]
    let lp = logits(&[0, 0, 2, 1], 3);
    let toks = ctc_greedy_decode(&lp, 4, 3, 2);
    assert_eq!(
        toks.iter().map(|t| t.token_id).collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(toks[0].frame_index, 0);
    assert_eq!(toks[1].frame_index, 3);
}

#[test]
fn blank_separates_identical_labels() {
    // a a <blk> a  ->  [a, a]  (blank breaks the run of identical labels)
    let lp = logits(&[0, 0, 2, 0], 3);
    let toks = ctc_greedy_decode(&lp, 4, 3, 2);
    assert_eq!(
        toks.iter().map(|t| t.token_id).collect::<Vec<_>>(),
        vec![0, 0]
    );
}

#[test]
fn honours_t_len_truncation() {
    // 3 frames in the buffer, but only the first 2 are valid.
    let lp = logits(&[0, 1, 0], 3);
    let toks = ctc_greedy_decode(&lp, 2, 3, 2);
    assert_eq!(toks.len(), 2);
    assert_eq!(toks[1].frame_index, 1);
}

/// Two-class-plus-blank frames where `ids[t]` leads by `margin` nats over
/// every other class. A small margin is what a boost can overturn.
fn logits_with_margin(ids: &[usize], vocab: usize, margin: f32) -> Vec<f32> {
    let mut lp = vec![0.0f32; ids.len() * vocab];
    for (t, &id) in ids.iter().enumerate() {
        lp[t * vocab + id] = margin;
    }
    lp
}

fn ids_of(tokens: &[TokenInfo]) -> Vec<usize> {
    tokens.iter().map(|t| t.token_id).collect()
}

#[test]
fn beam_without_a_hotword_hit_matches_greedy() {
    // The biaser names a token this vocabulary does not have, so nothing is
    // ever boosted and the beam must land where the argmax does — including
    // the collapse rules.
    let b = Biaser::from_sequences(vec![vec![99]], 5.0).expect("biaser");
    for ids in [
        vec![0usize, 0, 2, 1],
        vec![0, 0, 2, 0],
        vec![1, 2, 2, 1, 0],
        vec![2, 2, 2],
    ] {
        let lp = logits(&ids, 3);
        let beam = ctc_prefix_beam_decode(&lp, ids.len(), 3, 2, &b);
        let greedy = ctc_greedy_decode(&lp, ids.len(), 3, 2);
        assert_eq!(
            ids_of(&beam),
            ids_of(&greedy),
            "beam diverged from greedy on {ids:?}"
        );
    }
}

#[test]
fn beam_keeps_the_frame_of_each_emitted_label() {
    // Word timestamps are built from these, so a prefix has to carry its
    // alignment, not just its text.
    let b = Biaser::from_sequences(vec![vec![99]], 5.0).expect("biaser");
    let lp = logits(&[0, 0, 2, 1], 3);
    let beam = ctc_prefix_beam_decode(&lp, 4, 3, 2, &b);
    let greedy = ctc_greedy_decode(&lp, 4, 3, 2);
    assert_eq!(
        beam.iter().map(|t| t.frame_index).collect::<Vec<_>>(),
        greedy.iter().map(|t| t.frame_index).collect::<Vec<_>>()
    );
}

#[test]
fn boost_recovers_a_hotword_the_model_narrowly_missed() {
    // vocab: 0, 1, 2 are labels, 3 is blank. Frame 0 is label 0; on frame 1
    // the model puts label 2 narrowly ahead of label 1. The hotword is
    // [0, 1] — the phrase the model just missed.
    let vocab = 4;
    let mut lp = logits_with_margin(&[0, 2], vocab, 5.0);
    lp[vocab + 1] = 4.6; // frame 1: label 1 just behind label 2

    let inert = Biaser::from_sequences(vec![vec![99]], 3.0).expect("biaser");
    let unbiased = ids_of(&ctc_prefix_beam_decode(&lp, 2, vocab, 3, &inert));
    assert_eq!(unbiased, vec![0, 2], "model's own pick");

    let hot = Biaser::from_sequences(vec![vec![0, 1]], 3.0).expect("biaser");
    let biased = ids_of(&ctc_prefix_beam_decode(&lp, 2, vocab, 3, &hot));
    assert_eq!(biased, vec![0, 1], "boost recovers the hotword");
}

#[test]
fn abandoned_partial_match_is_refunded() {
    // The hotword is [0, 1, 1, 1] but the audio only supports its first
    // token, so no beam can finish the phrase. The refund must leave the
    // transcript exactly where the unbiased decode put it — a partial match
    // that wins would be the greedy path's failure mode all over again.
    let lp = logits(&[0, 2, 1, 2, 0], 3);
    let inert = Biaser::from_sequences(vec![vec![99]], 8.0).expect("biaser");
    let hot = Biaser::from_sequences(vec![vec![0, 1, 1, 1]], 8.0).expect("biaser");
    assert_eq!(
        ids_of(&ctc_prefix_beam_decode(&lp, 5, 3, 2, &hot)),
        ids_of(&ctc_prefix_beam_decode(&lp, 5, 3, 2, &inert)),
        "an unfinishable hotword must not bend the transcript"
    );
}

#[test]
fn beam_tolerates_a_hotword_token_outside_the_vocab() {
    // A glossary compiled against another head can name ids this one does
    // not have. Those must be dropped, not indexed.
    let b = Biaser::from_sequences(vec![vec![0, 250]], 5.0).expect("biaser");
    let lp = logits(&[0, 1], 3);
    let out = ctc_prefix_beam_decode(&lp, 2, 3, 2, &b);
    assert_eq!(ids_of(&out), vec![0, 1]);
}

#[test]
fn beam_honours_t_len_truncation() {
    let b = Biaser::from_sequences(vec![vec![99]], 5.0).expect("biaser");
    let lp = logits(&[0, 1, 0], 3);
    let out = ctc_prefix_beam_decode(&lp, 2, 3, 2, &b);
    assert_eq!(ids_of(&out), vec![0, 1]);
}

#[test]
fn all_blank_is_empty() {
    let lp = logits(&[2, 2, 2], 3);
    assert!(ctc_greedy_decode(&lp, 3, 3, 2).is_empty());
}

/// Build a CTC-style char tokenizer whose vocab[0] is the `▁` word-boundary
/// marker (matching istupakov's `multilingual_vocab.txt`), followed by the
/// letters used below and a trailing `<blk>`.
fn ctc_tokenizer(letters: &[&str]) -> Tokenizer {
    let mut toks = vec!["\u{2581}".to_string()];
    toks.extend(letters.iter().map(|s| s.to_string()));
    toks.push("<blk>".to_string());
    Tokenizer::from_tokens(toks)
}

fn tok(id: usize, frame: usize) -> TokenInfo {
    TokenInfo {
        token_id: id,
        frame_index: frame,
        confidence: 1.0,
    }
}

#[test]
fn groups_words_on_boundary_marker() {
    // vocab: 0=▁, 1..=6 = п р и в е т, 7..=9 = м и р, 10=<blk>
    let t = ctc_tokenizer(&["п", "р", "и", "в", "е", "т", "м", "и", "р"]);
    // "привет мир": п р и в е т ▁ м и р
    let toks = [
        tok(1, 0),
        tok(2, 1),
        tok(3, 2),
        tok(4, 3),
        tok(5, 4),
        tok(6, 5),
        tok(0, 6), // ▁ separator
        tok(7, 7),
        tok(8, 8),
        tok(9, 9),
    ];
    let words = ctc_tokens_to_words(&t, &toks, 0);
    assert_eq!(
        words.iter().map(|w| w.word.as_str()).collect::<Vec<_>>(),
        vec!["привет", "мир"]
    );
    // Timestamps come from the first/last frame of each word.
    assert!((words[0].start - 0.0).abs() < 1e-9);
    assert!(words[1].start > words[0].end);
}

#[test]
fn leading_and_trailing_boundaries_emit_no_empty_words() {
    let t = ctc_tokenizer(&["а", "б"]);
    // ▁ а б ▁  → one word "аб", no empty words from the edge separators.
    let toks = [tok(0, 0), tok(1, 1), tok(2, 2), tok(0, 3)];
    let words = ctc_tokens_to_words(&t, &toks, 0);
    assert_eq!(
        words.iter().map(|w| w.word.as_str()).collect::<Vec<_>>(),
        vec!["аб"]
    );
}
