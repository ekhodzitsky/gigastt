use std::sync::Arc;

use super::windows::{WINDOW_OVERLAP_WORDS, WINDOW_WORDS, Window};
use super::*;
use crate::runtime::{RuntimeError, factory::Runtime};

#[test]
fn test_capitalize_python_semantics() {
    // Python str.capitalize(): first upper, rest lower.
    assert_eq!(capitalize("привет"), "Привет");
    assert_eq!(capitalize("ПРИВЕТ"), "Привет");
    assert_eq!(capitalize("пРиВеТ"), "Привет");
    assert_eq!(capitalize(""), "");
    assert_eq!(capitalize("a"), "A");
}

#[test]
fn test_process_token_lower_modes() {
    assert_eq!(process_token("слово", "LOWER_O"), "слово");
    assert_eq!(process_token("слово", "LOWER_PERIOD"), "слово.");
    assert_eq!(process_token("слово", "LOWER_COMMA"), "слово,");
    assert_eq!(process_token("слово", "LOWER_QUESTION"), "слово?");
    assert_eq!(process_token("слово", "LOWER_VOSKL"), "слово!");
    assert_eq!(process_token("слово", "LOWER_DVOETOCHIE"), "слово:");
    assert_eq!(process_token("слово", "LOWER_PERIODCOMMA"), "слово;");
    assert_eq!(process_token("слово", "LOWER_DEFIS"), "слово-");
    assert_eq!(process_token("слово", "LOWER_MNOGOTOCHIE"), "слово...");
    assert_eq!(process_token("слово", "LOWER_QUESTIONVOSKL"), "слово?!");
}

#[test]
fn test_process_token_upper_capitalizes_first_lowercases_rest() {
    // UPPER_* uses Python capitalize: ПРИВЕТ → Привет, then suffix.
    assert_eq!(process_token("анна", "UPPER_O"), "Анна");
    assert_eq!(process_token("анна", "UPPER_COMMA"), "Анна,");
    assert_eq!(process_token("ПРИВЕТ", "UPPER_PERIOD"), "Привет.");
}

#[test]
fn test_process_token_upper_total_uppercases_all() {
    assert_eq!(process_token("ооо", "UPPER_TOTAL_O"), "ООО");
    assert_eq!(process_token("ссср", "UPPER_TOTAL_PERIOD"), "СССР.");
    assert_eq!(process_token("ооо", "UPPER_TOTAL_COMMA"), "ООО,");
}

#[test]
fn test_process_token_tire_spacing_quirk() {
    // LOWER_TIRE: no leading space before em-dash.
    assert_eq!(process_token("это", "LOWER_TIRE"), "это—");
    // UPPER_TIRE and UPPER_TOTAL_TIRE: leading space before em-dash.
    assert_eq!(process_token("это", "UPPER_TIRE"), "Это —");
    assert_eq!(process_token("это", "UPPER_TOTAL_TIRE"), "ЭТО —");
}

#[test]
fn test_process_token_unknown_label_is_identity() {
    assert_eq!(process_token("слово", "GARBAGE"), "слово");
    assert_eq!(process_token("слово", "LOWER_BOGUS"), "слово");
}

#[test]
fn test_first_subword_labels_picks_first_subtoken() {
    // Tokens: [CLS]=word None, word0 has 2 subtokens (idx1,2), word1 has 1
    // subtoken (idx3), [SEP]=None.
    let word_ids = vec![None, Some(0), Some(0), Some(1), None];
    // argmax label per token; word0's FIRST subtoken (idx1) is label 3,
    // its second (idx2) is 9 (must be ignored). word1 (idx3) is label 7.
    let argmax = vec![0, 3, 9, 7, 0];
    let labels = first_subword_labels(&word_ids, &argmax, 2);
    assert_eq!(labels, vec![3, 7]);
}

#[test]
fn test_first_subword_labels_missing_word_defaults_zero() {
    // No subtoken maps to word index 1 → defaults to label id 0.
    let word_ids = vec![None, Some(0), None];
    let argmax = vec![0, 5, 0];
    let labels = first_subword_labels(&word_ids, &argmax, 2);
    assert_eq!(labels, vec![5, 0]);
}

#[test]
fn test_argmax_returns_index_of_max() {
    assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
    assert_eq!(argmax(&[5.0, 1.0, 2.0]), 0);
    assert_eq!(argmax(&[1.0, 1.0, 3.0]), 2);
}

/// A minimal but valid HuggingFace tokenizer.json (WordPiece model, the only
/// scheme the in-tree tokenizer supports). Used to drive `Punctuator::load`
/// past the tokenizer step so the ONNX-session failure branch is exercised
/// without the real model.
const MINIMAL_TOKENIZER_JSON: &str = r###"{
    "version": "1.0",
    "truncation": null,
    "padding": null,
    "added_tokens": [],
    "normalizer": null,
    "pre_tokenizer": {"type": "BertPreTokenizer"},
    "post_processor": null,
    "decoder": null,
    "model": {
        "type": "WordPiece",
        "unk_token": "[UNK]",
        "continuing_subword_prefix": "##",
        "max_input_chars_per_word": 100,
        "vocab": {"[UNK]": 0, "a": 1, "b": 2}
    }
}"###;

/// A valid config.json with an `id2label` map: lets `load` clear the
/// id2label step so later failures (tokenizer / model) are reached.
const MINIMAL_CONFIG_JSON: &str = r#"{"id2label": {"0": "LOWER_O"}}"#;

/// `load` must surface the id2label parse failure (missing object) before it
/// ever touches the tokenizer or ONNX session.
#[test]
fn test_load_missing_id2label_errors() {
    let tmp = tempfile::tempdir().expect("tempdir");
    std::fs::write(tmp.path().join(PUNCT_CONFIG_FILE), r#"{"foo": 1}"#).unwrap();
    assert!(Punctuator::load(tmp.path()).is_err());
}

/// config.json parses but tokenizer.json is malformed: `load` must fail at
/// the `Tokenizer::from_file` step (graceful "punct unavailable", no panic).
#[test]
fn test_load_valid_config_invalid_tokenizer_errors() {
    let tmp = tempfile::tempdir().expect("tempdir");
    std::fs::write(tmp.path().join(PUNCT_CONFIG_FILE), MINIMAL_CONFIG_JSON).unwrap();
    std::fs::write(tmp.path().join(PUNCT_TOKENIZER_FILE), "{ not valid json").unwrap();
    // `Punctuator` is not `Debug`, so match instead of `expect_err`.
    match Punctuator::load(tmp.path()) {
        Ok(_) => panic!("malformed tokenizer must error"),
        Err(e) => assert!(e.to_string().contains("tokenizer")),
    }
}

/// config + tokenizer both load, but the ONNX model file is absent: `load`
/// must fail inside `OrtRuntime::load_session` (the `RuntimeError::LoadFailed`
/// branch), never panic. This is the last gate the caller turns into "punct disabled".
// Skipped under Miri: this is the one punctuation-load test that drives
// past the config + tokenizer gates into `OrtRuntime::load_session`, which
// calls the onnxruntime C API — a foreign function Miri cannot interpret.
#[test]
#[cfg_attr(miri, ignore = "reaches onnxruntime FFI via Punctuator::load")]
fn test_load_valid_config_and_tokenizer_missing_model_errors() {
    let tmp = tempfile::tempdir().expect("tempdir");
    std::fs::write(tmp.path().join(PUNCT_CONFIG_FILE), MINIMAL_CONFIG_JSON).unwrap();
    std::fs::write(
        tmp.path().join(PUNCT_TOKENIZER_FILE),
        MINIMAL_TOKENIZER_JSON,
    )
    .unwrap();
    // No rupunct_small_int8.onnx written → session build must fail.
    assert!(Punctuator::load(tmp.path()).is_err());
}

#[test]
fn test_load_punctuator_missing_dir_errors() {
    // Graceful fallback contract: loading from an absent dir must error
    // (the caller turns this into "punctuation disabled"), never panic.
    let tmp = tempfile::tempdir().expect("tempdir");
    let missing = tmp.path().join("does-not-exist");
    assert!(Punctuator::load(&missing).is_err());
}

#[test]
fn test_load_id2label_parses_contiguous_map() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cfg = tmp.path().join("config.json");
    std::fs::write(
        &cfg,
        r#"{"id2label": {"0": "UPPER_PERIOD", "1": "LOWER_PERIOD", "2": "UPPER_TOTAL_PERIOD"}}"#,
    )
    .unwrap();
    let labels = load_id2label(&cfg).expect("parse");
    assert_eq!(
        labels,
        vec!["UPPER_PERIOD", "LOWER_PERIOD", "UPPER_TOTAL_PERIOD"]
    );
}

#[test]
fn test_load_id2label_rejects_gap() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cfg = tmp.path().join("config.json");
    // Index 1 missing → non-contiguous.
    std::fs::write(&cfg, r#"{"id2label": {"0": "A", "2": "C"}}"#).unwrap();
    assert!(load_id2label(&cfg).is_err());
}

#[test]
fn test_word_spans_match_split_whitespace() {
    let text = "привет\tмир\n\nвот   так";
    let spans = word_spans(text);
    let words: Vec<&str> = spans.iter().map(|&(a, b)| &text[a..b]).collect();
    assert_eq!(words, text.split_whitespace().collect::<Vec<_>>());
}

#[test]
fn test_word_spans_empty_and_whitespace_only() {
    assert!(word_spans("").is_empty());
    assert!(word_spans("  \n\t ").is_empty());
}

/// Backward-compat gate (structural half): any transcript short enough for
/// one window is planned as a single window covering every word, and that
/// window's slice is the input string itself — so the model is handed
/// byte-for-byte what the un-windowed implementation handed it.
#[test]
fn test_short_text_is_one_window_over_the_whole_input() {
    let mut cases: Vec<String> = SHORT_FIXTURE_GOLDENS
        .iter()
        .map(|(input, _)| (*input).to_string())
        .collect();
    cases.push("одно".to_string());
    cases.push(
        (0..WINDOW_WORDS)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" "),
    );

    for text in &cases {
        let spans = word_spans(text);
        let windows = plan_windows(spans.len());
        assert_eq!(windows.len(), 1, "{} words", spans.len());
        assert_eq!(
            windows[0],
            Window {
                start: 0,
                end: spans.len(),
                keep_start: 0,
                keep_end: spans.len(),
            }
        );
        let slice = &text[spans[0].0..spans[spans.len() - 1].1];
        assert_eq!(
            slice, text,
            "the single window must encode the input verbatim"
        );
    }
}

#[test]
fn test_plan_windows_empty_input_has_no_windows() {
    assert!(plan_windows(0).is_empty());
}

/// The kept ranges must tile `0..num_words` exactly: no word labelled twice,
/// none left out, and every window small enough for the model.
#[test]
fn test_plan_windows_keep_ranges_tile_without_gap_or_overlap() {
    for num_words in [1, 2, 249, 250, 251, 600, 5000, 20_000] {
        let windows = plan_windows(num_words);
        let mut next = 0usize;
        for w in &windows {
            assert!(w.end - w.start <= WINDOW_WORDS, "{num_words}: {w:?}");
            assert!(w.start <= w.keep_start && w.keep_end <= w.end, "{w:?}");
            assert!(w.keep_start < w.keep_end, "{w:?}");
            assert_eq!(w.keep_start, next, "{num_words}: gap/overlap at {w:?}");
            next = w.keep_end;
        }
        assert_eq!(next, num_words, "{num_words} words not fully covered");
    }
}

/// Every kept word except at the transcript's own edges must sit at least
/// half an overlap away from its window's borders, i.e. be labelled with
/// real left and right context.
#[test]
fn test_plan_windows_interior_words_keep_context_on_both_sides() {
    let num_words = 5000;
    let windows = plan_windows(num_words);
    assert!(windows.len() > 1);
    let half = WINDOW_OVERLAP_WORDS / 2;
    for w in &windows {
        if w.start > 0 {
            assert!(w.keep_start - w.start >= half, "{w:?}");
        }
        if w.end < num_words {
            assert!(w.end - w.keep_end >= half, "{w:?}");
        }
    }
}

/// Pure split/splice round-trip on a synthetic 5000-word transcript, with no
/// model involved: each window is "labelled" with the global index of every
/// word it covers, so the spliced result proves that each word kept exactly
/// one label, from a window that actually covered it, at its own position.
#[test]
fn test_splice_window_labels_round_trips_5000_words() {
    // Three windows still prove the splice. Five thousand owned strings do
    // not finish under Miri inside the nightly budget.
    let n = if cfg!(miri) { 600 } else { 5000 };
    let words: Vec<String> = (0..n).map(|i| format!("w{i}")).collect();
    let text = words.join(" ");
    let spans = word_spans(&text);
    assert_eq!(spans.len(), n);

    let windows = plan_windows(spans.len());
    let per_window: Vec<Option<Vec<usize>>> = windows
        .iter()
        .map(|w| Some((w.start..w.end).collect()))
        .collect();

    let merged = splice_window_labels(&windows, &per_window, spans.len());
    let expected: Vec<Option<usize>> = (0..n).map(Some).collect();
    assert_eq!(merged, expected, "zero lost, duplicated or reordered words");

    // The word list the assembler walks is still the original one, in order.
    let round_tripped: Vec<&str> = spans.iter().map(|&(a, b)| &text[a..b]).collect();
    assert_eq!(round_tripped, words);
}

#[test]
fn test_splice_window_labels_failed_window_leaves_its_words_unlabelled() {
    let windows = plan_windows(600);
    assert_eq!(windows.len(), 3);
    let mut per_window: Vec<Option<Vec<usize>>> = windows
        .iter()
        .map(|w| Some((w.start..w.end).map(|_| 7usize).collect()))
        .collect();
    per_window[1] = None;

    let merged = splice_window_labels(&windows, &per_window, 600);
    for (i, label) in merged.iter().enumerate() {
        let bare = i >= windows[1].keep_start && i < windows[1].keep_end;
        assert_eq!(*label, if bare { None } else { Some(7) }, "word {i}");
    }
}

/// A tokenizer whose WordPiece vocab only knows `a` / `##a`, so an N-char
/// word explodes into N subtokens. Used to drive a window past the subtoken
/// ceiling without a real model.
const SPLITTING_TOKENIZER_JSON: &str = r###"{
    "version": "1.0",
    "truncation": null,
    "padding": null,
    "added_tokens": [],
    "normalizer": null,
    "pre_tokenizer": {"type": "BertPreTokenizer"},
    "post_processor": null,
    "decoder": null,
    "model": {
        "type": "WordPiece",
        "unk_token": "[UNK]",
        "continuing_subword_prefix": "##",
        "max_input_chars_per_word": 200,
        "vocab": {"[UNK]": 0, "a": 1, "##a": 2}
    }
}"###;

/// Stand-in for the ONNX punct session: answers with `[1, seq, num_labels]`
/// logits whose argmax is `label` on every token, records the sequence length
/// of every run, and can fail one chosen run. Lets the windowing path be
/// exercised with no model on disk.
#[derive(Clone)]
struct StubSession {
    num_labels: usize,
    label: usize,
    fail_on_call: Option<usize>,
    seqs: Arc<Mutex<Vec<usize>>>,
}

impl RuntimeSession for StubSession {
    fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, RuntimeError> {
        let dims = inputs[0].shape().dims().to_vec();
        assert_eq!(dims.len(), 2, "punct inputs are [1, seq]");
        let seq = dims[1];
        let call = {
            let mut seqs = self.seqs.lock();
            seqs.push(seq);
            seqs.len() - 1
        };
        if self.fail_on_call == Some(call) {
            return Err(RuntimeError::InferenceFailed("stub window failure".into()));
        }
        let mut logits = vec![0.0f32; seq * self.num_labels];
        for t in 0..seq {
            logits[t * self.num_labels + self.label] = 1.0;
        }
        Ok(vec![Tensor::new_checked(
            Shape::new(vec![1, seq, self.num_labels]),
            TensorData::F32(logits),
        )])
    }
}

impl RuntimeFactory for StubSession {
    fn create(&self, _intra_threads: usize) -> Result<Box<dyn Runtime>, RuntimeError> {
        Ok(Box::new(self.clone()))
    }
    fn cpu_fallback(&self) -> Box<dyn RuntimeFactory> {
        Box::new(self.clone())
    }
}

impl Runtime for StubSession {
    fn load_session(
        &self,
        _model_path: &Path,
        _is_encoder: bool,
    ) -> Result<Box<dyn RuntimeSession>, RuntimeError> {
        Ok(Box::new(self.clone()))
    }
}

/// Punctuator over a temp-dir tokenizer + label map, with the ONNX session
/// replaced by [`StubSession`]. Returns the punctuator and the shared log of
/// per-run sequence lengths.
fn stub_punctuator(
    tokenizer_json: &str,
    labels: &[&str],
    label: usize,
    fail_on_call: Option<usize>,
) -> (Punctuator, Arc<Mutex<Vec<usize>>>) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let entries: Vec<String> = labels
        .iter()
        .enumerate()
        .map(|(i, l)| format!("\"{i}\": \"{l}\""))
        .collect();
    std::fs::write(
        tmp.path().join(PUNCT_CONFIG_FILE),
        format!("{{\"id2label\": {{{}}}}}", entries.join(", ")),
    )
    .unwrap();
    std::fs::write(tmp.path().join(PUNCT_TOKENIZER_FILE), tokenizer_json).unwrap();

    let seqs = Arc::new(Mutex::new(Vec::new()));
    let stub = StubSession {
        num_labels: labels.len(),
        label,
        fail_on_call,
        seqs: Arc::clone(&seqs),
    };
    let punct = match Punctuator::load_with_factory(tmp.path(), &stub) {
        Ok(p) => p,
        Err(e) => panic!("stub punctuator load failed: {e:#}"),
    };
    (punct, seqs)
}

/// A transcript far longer than one window comes back with every word
/// labelled, in order, one run per planned window.
#[test]
fn test_restore_long_text_labels_every_word_one_run_per_window() {
    // Several windows prove one run per window. Five thousand words through
    // the tokenizer does not finish under Miri inside the nightly budget.
    let n = if cfg!(miri) { 600 } else { 5000 };
    let words: Vec<String> = (0..n).map(|i| format!("w{i}")).collect();
    let text = words.join(" ");
    let (punct, seqs) = stub_punctuator(MINIMAL_TOKENIZER_JSON, &["LOWER_O", "UPPER_O"], 1, None);

    let out = punct.restore(&text);

    let expected: Vec<String> = words.iter().map(|w| capitalize(w)).collect();
    assert_eq!(out, expected.join(" "));
    let seqs = seqs.lock();
    assert_eq!(seqs.len(), plan_windows(n).len());
    assert!(seqs.iter().all(|&s| s <= WINDOW_WORDS), "{seqs:?}");
    assert_eq!(punct.failed_windows(), 0);
}

/// A window whose lexis blows past the subtoken ceiling is split until every
/// submitted sequence fits the model's position table.
// WordPiece over a window of 40-character words does not finish under Miri.
// A shorter word no longer crosses the ceiling inside one window, so the
// branch stays a native numeric test.
#[test]
#[cfg_attr(
    miri,
    ignore = "wordpiece of a full window of long words does not finish under Miri"
)]
fn test_restore_splits_a_window_over_the_subtoken_ceiling() {
    // 250 words × 40 subtokens ≈ 10k subtokens in one window.
    let word = "a".repeat(40);
    let text = vec![word.as_str(); WINDOW_WORDS].join(" ");
    let (punct, seqs) = stub_punctuator(SPLITTING_TOKENIZER_JSON, &["LOWER_O", "UPPER_O"], 1, None);

    let out = punct.restore(&text);

    assert_eq!(
        out,
        vec![capitalize(&word); WINDOW_WORDS].join(" "),
        "every word must still be labelled"
    );
    let seqs = seqs.lock();
    assert!(seqs.len() > 1, "the oversized window must have been split");
    assert!(
        seqs.iter().all(|&s| s <= MAX_WINDOW_SUBTOKENS),
        "a run exceeded the ceiling: {seqs:?}"
    );
}

/// One failing window must not cost the rest of the transcript its
/// punctuation — only its own words come back bare, and the failure is
/// counted instead of vanishing.
#[test]
fn test_restore_partial_window_failure_only_bares_that_window() {
    let words: Vec<String> = (0..600).map(|i| format!("w{i}")).collect();
    let text = words.join(" ");
    let windows = plan_windows(600);
    assert_eq!(windows.len(), 3);
    let (punct, _seqs) = stub_punctuator(
        MINIMAL_TOKENIZER_JSON,
        &["LOWER_O", "UPPER_O"],
        1,
        Some(1), // the middle window's run fails
    );

    let out = punct.restore(&text);

    let got: Vec<&str> = out.split(' ').collect();
    assert_eq!(got.len(), words.len());
    for (i, word) in words.iter().enumerate() {
        let bare = i >= windows[1].keep_start && i < windows[1].keep_end;
        let expected = if bare { word.clone() } else { capitalize(word) };
        assert_eq!(got[i], expected, "word {i}");
    }
    assert_eq!(punct.failed_windows(), 1);
}

/// When nothing can be labelled the un-windowed contract stands: the input
/// comes back untouched (original whitespace included) and the failure is
/// counted.
#[test]
fn test_restore_returns_input_unchanged_when_every_window_fails() {
    let text = "  привет   мир  ";
    let (punct, _seqs) = stub_punctuator(MINIMAL_TOKENIZER_JSON, &["LOWER_O"], 0, Some(0));

    assert_eq!(punct.restore(text), text);
    assert_eq!(punct.failed_windows(), 1);
}

/// Short transcripts (one window each) paired with the output captured from
/// the single-run implementation that preceded windowing.
const SHORT_FIXTURE_GOLDENS: &[(&str, &str)] = &[
    (
        "привет меня зовут анна сколько будет стоить шестьдесят тысяч тенге",
        "Привет меня зовут Анна, Сколько будет стоить шестьдесят тысяч тенге.",
    ),
    (
        "здравствуйте я хотел бы узнать когда открывается магазин и сколько стоит доставка до города",
        "Здравствуйте, Я хотел бы узнать, когда открывается магазин и сколько стоит доставка до города.",
    ),
    ("нет спасибо не надо", "Нет, Спасибо. Не надо."),
    (
        "он сказал что завтра будет дождь а послезавтра выпадет снег и станет холодно",
        "Он сказал, что завтра будет дождь, а послезавтра выпадет снег и станет холодно.",
    ),
    (
        "один два три четыре пять шесть семь восемь девять десять",
        "Один — два, три, четыре, пять, шесть, семь, восемь, девять, десять.",
    ),
];

/// Backward-compat gate (model half): a transcript that fits in one window
/// must come back byte-identical to the pre-windowing output.
#[test]
#[ignore = "requires punct model at ~/.gigastt/models/punct"]
fn test_restore_short_fixtures_match_unwindowed_output() {
    let dir = default_punct_model_dir();
    let punct = Punctuator::load(Path::new(&dir)).expect("load punct model");
    for (input, expected) in SHORT_FIXTURE_GOLDENS {
        assert_eq!(&punct.restore(input), expected);
    }
    assert_eq!(punct.failed_windows(), 0);
}

/// A 20 000-word transcript — several times the model's position table —
/// must come back punctuated and cased end to end, with every word intact.
/// Before windowing this returned the input verbatim.
#[test]
#[ignore = "requires punct model at ~/.gigastt/models/punct"]
fn test_restore_very_long_transcript_is_punctuated() {
    let dir = default_punct_model_dir();
    let punct = Punctuator::load(Path::new(&dir)).expect("load punct model");
    let sentence = "сегодня мы обсудим важный вопрос который волнует многих наших слушателей";
    let text = std::iter::repeat_n(sentence, 2000)
        .collect::<Vec<_>>()
        .join(" ");
    assert_eq!(text.split_whitespace().count(), 20_000);

    let out = punct.restore(&text);

    assert_ne!(out, text, "a 20k-word transcript must not come back bare");
    assert_eq!(
        out.split_whitespace().count(),
        20_000,
        "no word may be lost or duplicated"
    );
    assert_eq!(punct.failed_windows(), 0);
    let tail: String = out
        .split_whitespace()
        .skip(15_000)
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        tail.contains('.') && tail.chars().any(char::is_uppercase),
        "punctuation and casing must reach the end of the transcript"
    );
}

/// End-to-end on the real ONNX model (model-gated, like other model tests).
/// Validates the full tokenizer → ONNX → first-subword → process_token
/// pipeline against the RUPunct reference string.
#[test]
#[ignore = "requires punct model at ~/.gigastt/models/punct"]
fn test_restore_reference_string() {
    let dir = default_punct_model_dir();
    let punct = Punctuator::load(Path::new(&dir)).expect("load punct model");
    let out = punct.restore("привет меня зовут анна сколько будет стоить шестьдесят тысяч тенге");
    assert_eq!(
        out,
        "Привет меня зовут Анна, Сколько будет стоить шестьдесят тысяч тенге."
    );
}

/// Latency probe for the streaming use case: `restore` runs synchronously
/// on the finalization boundary of every streaming segment, so its cost on
/// short (1–10 word) segments adds directly to final-segment latency.
/// Prints p50/p95 per segment length; a generous sanity ceiling keeps the
/// run self-checking without flaking on slow machines (the probe is
/// model-gated and runs manually, not in CI).
#[test]
#[ignore = "requires punct model at ~/.gigastt/models/punct"]
fn test_restore_latency_short_segments() {
    let dir = default_punct_model_dir();
    let punct = Punctuator::load(Path::new(&dir)).expect("load punct model");

    let cases: &[(&str, &str)] = &[
        ("1 word", "привет"),
        ("5 words", "привет меня зовут анна"),
        (
            "10 words",
            "привет меня зовут анна сколько будет стоить шестьдесят тысяч тенге",
        ),
    ];
    const ITERS: usize = 50;

    for (label, text) in cases {
        // Warmup: first runs pay tokenizer/thread-pool lazy init.
        for _ in 0..5 {
            let _ = punct.restore(text);
        }
        let mut samples = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let start = std::time::Instant::now();
            let _ = punct.restore(text);
            samples.push(start.elapsed());
        }
        samples.sort();
        let p50 = samples[ITERS / 2];
        let p95 = samples[ITERS * 95 / 100];
        eprintln!(
            "restore latency {label}: p50={p50:?} p95={p95:?} max={:?}",
            samples[ITERS - 1]
        );
        assert!(
            p95 < std::time::Duration::from_millis(500),
            "restore p95 on a short segment must stay well under 500ms, got {p95:?} ({label})"
        );
    }
}

use crate::model::default_punct_model_dir;
