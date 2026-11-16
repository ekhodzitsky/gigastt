use super::*;
use crate::inference::tokenizer::Tokenizer;
use crate::model::ModelVariant;
use crate::runtime::mock::MockSession;
use crate::runtime::tensor::{Shape, Tensor, TensorData};

fn decode_raw_words(
    committed: Vec<WordInfo>,
    raw: &[(&str, usize)],
    origin: usize,
    context: usize,
) -> Vec<WordInfo> {
    let (mut engine, _tmp) = crate::test_support::rnnt_engine();
    engine.variant = ModelVariant::MlCtcLarge;
    let mut vocabulary = vec!["▁".to_string()];
    for (text, _) in raw {
        if !vocabulary.iter().any(|token| token == text) {
            vocabulary.push((*text).to_string());
        }
    }
    vocabulary.push("<blk>".into());
    let classes = vocabulary.len();
    let blank = classes - 1;
    let frames = raw.iter().map(|(_, frame)| frame + 2).max().unwrap_or(1);
    let mut logits = vec![-10.0; frames * classes];
    for row in logits.chunks_mut(classes) {
        row[blank] = 0.0;
    }
    for (i, (text, frame)) in raw.iter().enumerate() {
        let id = vocabulary.iter().position(|token| token == text).unwrap();
        logits[frame * classes + id] = 10.0;
        if i > 0 {
            logits[(frame - 1) * classes] = 10.0;
        }
    }
    engine.tokenizer = Tokenizer::from_tokens(vocabulary);
    let mut guard = engine.pool.checkout_blocking().unwrap();
    guard.encoder = Box::new(MockSession::unconstrained(vec![
        Tensor::new_checked(
            Shape::new(vec![1, frames, classes]),
            TensorData::F32(logits),
        ),
        Tensor::new_checked(Shape::new(vec![1]), TensorData::I64(vec![frames as i64])),
    ]));
    let mut state = engine.create_state(false);
    state.commit_policy = crate::inference::CommitPolicy::StablePrefix;
    state.window_start_samples = origin;
    state.context_samples = context;
    state.audio_buffer = vec![0.0; 80000];
    state.assembler.append(committed.clone());
    state.assembler.commit_live();
    engine.decode_window(&mut state, &mut guard).unwrap();
    let output = state.assembler.partial(0.0).words;
    assert_eq!(
        serde_json::to_value(&output[..committed.len()]).unwrap(),
        serde_json::to_value(&committed).unwrap()
    );
    output
}

fn text(words: &[WordInfo]) -> String {
    words
        .iter()
        .map(|word| word.word.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn test_ctc_covered_prefix_updates_assembler_without_repeating_committed_word() {
    let out = decode_raw_words(
        vec![word("вас", 2.16, 2.24)],
        &[("вас", 41), ("приветствует", 47)],
        11840,
        24000,
    );
    assert_eq!(text(&out), "вас приветствует");
}

#[test]
fn test_ctc_covered_prefix_keeps_close_genuine_repetition() {
    let out = decode_raw_words(
        vec![word("да", 1.0, 1.11)],
        &[("да", 28), ("да", 30), ("нет", 33)],
        0,
        17760,
    );
    assert_eq!(text(&out), "да да нет");
}

#[test]
fn test_ctc_covered_prefix_accepts_point_anchors_and_preserves_future_negation() {
    let old = vec![
        word("скажите", 5.88, 6.2),
        word("с", 6.36, 6.36),
        word("кем", 6.48, 6.56),
        word("или", 6.76, 6.88),
        word("по", 6.96, 7.0),
    ];
    let out = decode_raw_words(
        old,
        &[
            ("скажите", 11),
            ("с", 23),
            ("кем", 26),
            ("или", 34),
            ("по", 40),
            ("не", 45),
            ("молчи", 50),
        ],
        88000,
        24000,
    );
    assert_eq!(text(&out), "скажите с кем или по не молчи");
}

#[test]
fn test_ctc_covered_prefix_does_not_reconcile_an_evicted_anchor() {
    let out = decode_raw_words(vec![word("привет", 1.0, 1.1)], &[("привет", 25)], 48000, 0);
    assert_eq!(text(&out), "привет привет");
}

#[test]
fn test_ctc_covered_prefix_keeps_time_coverage_filter_after_match() {
    let out = decode_raw_words(
        vec![word("да", 1.0, 1.1)],
        &[("да", 20), ("старое", 25), ("не", 33)],
        0,
        17600,
    );
    assert_eq!(text(&out), "да не");
}

#[test]
fn test_ctc_unmatched_prefix_keeps_legacy_same_word_seam() {
    let out = decode_raw_words(
        vec![word("старое", 0.2, 0.3), word("да", 1.0, 1.1)],
        &[("да", 28), ("нет", 33)],
        0,
        17600,
    );
    assert_eq!(text(&out), "старое да нет");
}

#[test]
fn test_ctc_covered_leading_word_allows_committed_chain_reconciliation() {
    let old = vec![
        word("супермодели", 1.4, 1.96),
        word("не", 2.08, 2.12),
        word("участвуют", 2.52, 2.92),
        word("в", 3.04, 3.04),
    ];
    let out = decode_raw_words(
        old,
        &[
            ("супермодели", 10),
            ("не", 28),
            ("участвуют", 41),
            ("в", 43),
            ("конкурсах", 50),
        ],
        24640,
        24000,
    );
    assert_eq!(text(&out), "супермодели не участвуют в конкурсах");
}

#[test]
fn test_ctc_live_leading_negation_prevents_skipping_to_anchor() {
    let out = decode_raw_words(
        vec![word("да", 1.0, 1.1)],
        &[("не", 28), ("да", 32)],
        0,
        17600,
    );
    assert_eq!(text(&out), "да не да");
}
