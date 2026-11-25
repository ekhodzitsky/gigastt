use super::*;
use serde::Deserialize;

fn fixtures_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

#[derive(Deserialize)]
struct GoldenCase {
    input: String,
    ids: Vec<u32>,
    tokens: Vec<String>,
    attention_mask: Vec<u32>,
    word_ids: Vec<Option<u32>>,
}

fn load_golden() -> Vec<GoldenCase> {
    let raw = std::fs::read_to_string(fixtures_dir().join("wordpiece_golden.json"))
        .expect("read golden fixtures");
    serde_json::from_str(&raw).expect("parse golden fixtures")
}

fn assert_replays(tokenizer: &Tokenizer) {
    let cases = load_golden();
    assert_eq!(cases.len(), 39, "fixture corpus size changed unexpectedly");
    for case in &cases {
        let enc = tokenizer.encode(&case.input, true);
        let ctx = || format!("input {:?}\nexpected tokens {:?}", case.input, case.tokens);
        assert_eq!(enc.get_ids(), case.ids.as_slice(), "ids: {}", ctx());
        assert_eq!(
            enc.get_attention_mask(),
            case.attention_mask.as_slice(),
            "attention_mask: {}",
            ctx()
        );
        assert_eq!(
            enc.get_word_ids(),
            case.word_ids.as_slice(),
            "word_ids: {}",
            ctx()
        );
    }
}

/// Hermetic golden replay: the in-tree tokenizer over the reduced vocab
/// fixture must reproduce, id for id and word-id for word-id, what the
/// upstream `tokenizers` 0.23 crate produced for the corpus (Russian
/// phrases, punctuation, digits, mixed case, Cf chars, CJK, U+FFFD,
/// emoji, literal special tokens, empty/whitespace input, word-length
/// boundary at 100 chars).
///
/// The reduced vocab contains every token the upstream crate emitted for
/// the corpus with its original id; greedy longest-match therefore takes
/// the identical path (longer candidates are absent from both vocabs,
/// shorter or equal ones are committed identically, and words that fell
/// back to [UNK] find nothing in either).
#[test]
fn test_golden_fixtures_replay_reduced_vocab() {
    let tokenizer = Tokenizer::from_file(&fixtures_dir().join("wordpiece_tokenizer.json"))
        .expect("load reduced vocab fixture");
    assert_replays(&tokenizer);
}

/// Pretokenized input keeps the caller's whitespace word identity while
/// preserving the model's token IDs and applying the special template once.
#[test]
fn test_encode_words_retains_input_word_indices() {
    let tokenizer = Tokenizer::from_file(&fixtures_dir().join("wordpiece_tokenizer.json"))
        .expect("load reduced vocab fixture");
    let text = "какой-то \u{200b} 3.14 中 [SEP] привет";
    let raw = tokenizer.encode(text, true);
    let words = tokenizer.encode_words(text, true);
    assert_eq!(words.get_ids(), raw.get_ids());
    assert_eq!(words.get_attention_mask(), raw.get_attention_mask());

    let mut expected = vec![None];
    for (i, word) in text.split_whitespace().enumerate() {
        let count = tokenizer.encode(word, false).get_ids().len();
        expected.extend(std::iter::repeat_n(Some(i as u32), count));
    }
    expected.push(None);
    assert_eq!(words.get_word_ids(), expected);
    assert!(!expected.contains(&Some(1))); // normalization removes the word
    assert_ne!(words.get_word_ids(), raw.get_word_ids());
}

/// Same replay against the real punct model tokenizer.json (full 83 828
/// entry vocab). Guards against the reduced vocab masking a divergence.
#[test]
#[ignore = "requires punct model at ~/.gigastt/models/punct"]
fn test_golden_fixtures_replay_full_vocab() {
    let path = Path::new(&std::env::var("HOME").expect("HOME"))
        .join(".gigastt/models/punct/tokenizer.json");
    let tokenizer = Tokenizer::from_file(&path).expect("load real tokenizer.json");
    assert_replays(&tokenizer);
}

/// Minimal WordPiece tokenizer.json builder for behavioral unit tests.
fn write_tokenizer(dir: &Path, vocab: &str) -> std::path::PathBuf {
    let json = format!(
        r###"{{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": {{"type": "BertNormalizer", "clean_text": true,
                             "handle_chinese_chars": true, "strip_accents": false,
                             "lowercase": false}},
            "pre_tokenizer": {{"type": "BertPreTokenizer"}},
            "post_processor": null,
            "decoder": null,
            "model": {{"type": "WordPiece", "unk_token": "[UNK]",
                       "continuing_subword_prefix": "##",
                       "max_input_chars_per_word": 100,
                       "vocab": {{{vocab}}}
            }}
        }}"###
    );
    let path = dir.join("tokenizer.json");
    std::fs::write(&path, json).unwrap();
    path
}

const TINY_VOCAB: &str = r###""[UNK]": 0, "привет": 1, "мир": 2, "##а": 3, "а": 4,
    "при": 5, "##вет": 6, "!": 7, "中": 8"###;

#[test]
fn test_encode_basic_and_unk() {
    let tmp = tempfile::tempdir().unwrap();
    let tok = Tokenizer::from_file(&write_tokenizer(tmp.path(), TINY_VOCAB)).unwrap();
    // Whole word in vocab, unknown word -> [UNK], punctuation isolated.
    let enc = tok.encode("привет мир! неизвестно", true);
    assert_eq!(enc.get_ids(), &[1, 2, 7, 0]);
    assert_eq!(enc.get_word_ids(), &[Some(0), Some(1), Some(2), Some(3)]);
    assert_eq!(enc.get_attention_mask(), &[1, 1, 1, 1]);
}

#[test]
fn test_encode_greedy_wordpiece_and_no_specials() {
    let tmp = tempfile::tempdir().unwrap();
    let tok = Tokenizer::from_file(&write_tokenizer(tmp.path(), TINY_VOCAB)).unwrap();
    // "приветик": "привет" matches, then "##и"/"##ик" are absent, so the
    // whole word collapses to a single [UNK].
    let enc = tok.encode("приветик", false);
    assert_eq!(enc.get_ids(), &[0]);
    // Longest match keeps "привет" whole.
    let enc = tok.encode("привет", false);
    assert_eq!(enc.get_ids(), &[1]);
    // "привета" -> "привет" + "##а" (both in vocab).
    let enc = tok.encode("привета", false);
    assert_eq!(enc.get_ids(), &[1, 3]);
    assert_eq!(enc.get_word_ids(), &[Some(0), Some(0)]);
}

#[test]
fn test_word_length_cap_boundary() {
    let tmp = tempfile::tempdir().unwrap();
    let tok = Tokenizer::from_file(&write_tokenizer(tmp.path(), TINY_VOCAB)).unwrap();
    let word_100 = "а".repeat(100); // "а" + "##а" * 99, all in vocab
    let enc = tok.encode(&word_100, false);
    assert_eq!(enc.get_ids().len(), 100);
    assert_eq!(enc.get_ids()[0], 4);
    assert!(enc.get_ids()[1..].iter().all(|&id| id == 3));
    let word_101 = "а".repeat(101);
    let enc = tok.encode(&word_101, false);
    assert_eq!(enc.get_ids(), &[0]);
}

#[test]
fn test_clean_text_and_chinese_spacing() {
    let tmp = tempfile::tempdir().unwrap();
    let tok = Tokenizer::from_file(&write_tokenizer(tmp.path(), TINY_VOCAB)).unwrap();
    // NUL / ZWSP (Cf) removed; tab -> space; CJK char spaced out.
    let enc = tok.encode("при\u{0}ве\u{200B}т\t中", false);
    assert_eq!(enc.get_ids(), &[1, 8]);
    assert_eq!(enc.get_word_ids(), &[Some(0), Some(1)]);
}

#[test]
fn test_empty_and_whitespace_input() {
    let tmp = tempfile::tempdir().unwrap();
    let tok = Tokenizer::from_file(&write_tokenizer(tmp.path(), TINY_VOCAB)).unwrap();
    for input in ["", "   \t\n "] {
        let enc = tok.encode(input, true);
        assert!(enc.get_ids().is_empty());
        assert!(enc.get_word_ids().is_empty());
    }
}

fn expect_load_error(json: &str, needle: &str) {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("tokenizer.json");
    std::fs::write(&path, json).unwrap();
    match Tokenizer::from_file(&path) {
        Ok(_) => panic!("expected load error containing '{needle}'"),
        Err(e) => {
            let msg = format!("{e:#}");
            assert!(msg.contains(needle), "error '{msg}' lacks '{needle}'");
        }
    }
}

fn base_json(model: &str, normalizer: &str, post: &str, added: &str) -> String {
    format!(
        r###"{{"version": "1.0", "truncation": null, "padding": null,
            "added_tokens": {added}, "normalizer": {normalizer},
            "pre_tokenizer": {{"type": "BertPreTokenizer"}},
            "post_processor": {post}, "decoder": null,
            "model": {model}}}"###
    )
}

const GOOD_MODEL: &str = r###"{"type": "WordPiece", "unk_token": "[UNK]",
    "continuing_subword_prefix": "##", "max_input_chars_per_word": 100,
    "vocab": {"[UNK]": 0, "a": 1}}"###;
const GOOD_NORMALIZER: &str = r#"{"type": "BertNormalizer", "clean_text": true,
    "handle_chinese_chars": true, "strip_accents": false, "lowercase": false}"#;

#[test]
fn test_load_rejects_lowercase_normalizer() {
    let norm = r#"{"type": "BertNormalizer", "clean_text": true,
        "handle_chinese_chars": true, "strip_accents": false, "lowercase": true}"#;
    expect_load_error(
        &base_json(GOOD_MODEL, norm, "null", "[]"),
        "lowercase=true is not supported",
    );
}

#[test]
fn test_load_rejects_strip_accents_normalizer() {
    let norm = r#"{"type": "BertNormalizer", "clean_text": true,
        "handle_chinese_chars": true, "strip_accents": true, "lowercase": false}"#;
    expect_load_error(
        &base_json(GOOD_MODEL, norm, "null", "[]"),
        "strip_accents=true is not supported",
    );
}

#[test]
fn test_load_rejects_non_wordpiece_model() {
    let model = r#"{"type": "WordLevel", "vocab": {"[UNK]": 0}, "unk_token": "[UNK]"}"#;
    expect_load_error(
        &base_json(model, GOOD_NORMALIZER, "null", "[]"),
        "unsupported model type 'WordLevel'",
    );
}

#[test]
fn test_load_rejects_missing_unk_token() {
    let model = r###"{"type": "WordPiece", "unk_token": "[UNK]",
        "continuing_subword_prefix": "##", "max_input_chars_per_word": 100,
        "vocab": {"a": 1}}"###;
    expect_load_error(
        &base_json(model, GOOD_NORMALIZER, "null", "[]"),
        "missing unk_token",
    );
}

#[test]
fn test_load_rejects_truncation_and_padding() {
    let json = base_json(GOOD_MODEL, GOOD_NORMALIZER, "null", "[]").replace(
        r#""truncation": null"#,
        r#""truncation": {"max_length": 128}"#,
    );
    expect_load_error(&json, "truncation is not supported");
    let json = base_json(GOOD_MODEL, GOOD_NORMALIZER, "null", "[]").replace(
        r#""padding": null"#,
        r#""padding": {"strategy": "BatchLongest"}"#,
    );
    expect_load_error(&json, "padding is not supported");
}

#[test]
fn test_load_rejects_unsupported_added_token_flags() {
    let added = r#"[{"id": 5, "content": "tok", "single_word": true, "lstrip": false,
        "rstrip": false, "normalized": false, "special": false}]"#;
    expect_load_error(
        &base_json(GOOD_MODEL, GOOD_NORMALIZER, "null", added),
        "single_word=true",
    );
    let added = r#"[{"id": 5, "content": "tok", "single_word": false, "lstrip": false,
        "rstrip": false, "normalized": true, "special": false}]"#;
    expect_load_error(
        &base_json(GOOD_MODEL, GOOD_NORMALIZER, "null", added),
        "normalized=true",
    );
}

#[test]
fn test_load_rejects_bad_version_and_unknown_parts() {
    let json = base_json(GOOD_MODEL, GOOD_NORMALIZER, "null", "[]")
        .replace(r#""version": "1.0"#, r#""version": "2.0"#);
    expect_load_error(&json, "unknown tokenizer version");
    let json = base_json(GOOD_MODEL, GOOD_NORMALIZER, "null", "[]").replace(
        r#"{"type": "BertPreTokenizer"}"#,
        r#"{"type": "Whitespace"}"#,
    );
    expect_load_error(&json, "unsupported pre_tokenizer type 'Whitespace'");
    let json = base_json(GOOD_MODEL, GOOD_NORMALIZER, "null", "[]")
        .replace(GOOD_NORMALIZER, r#"{"type": "Lowercase"}"#);
    expect_load_error(&json, "unsupported normalizer type 'Lowercase'");
}

#[test]
fn test_added_token_extraction_and_template() {
    // Full BERT scheme in miniature: [CLS]/[SEP] template + added tokens.
    let json = r###"{
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [
            {"id": 0, "content": "[UNK]", "single_word": false, "lstrip": false,
             "rstrip": false, "normalized": false, "special": true},
            {"id": 2, "content": "[CLS]", "single_word": false, "lstrip": false,
             "rstrip": false, "normalized": false, "special": true},
            {"id": 3, "content": "[SEP]", "single_word": false, "lstrip": false,
             "rstrip": false, "normalized": false, "special": true}
        ],
        "normalizer": null,
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
            ],
            "pair": [],
            "special_tokens": {
                "[CLS]": {"id": "[CLS]", "ids": [2], "tokens": ["[CLS]"]},
                "[SEP]": {"id": "[SEP]", "ids": [3], "tokens": ["[SEP]"]}
            }
        },
        "decoder": null,
        "model": {"type": "WordPiece", "unk_token": "[UNK]",
                  "continuing_subword_prefix": "##",
                  "max_input_chars_per_word": 100,
                  "vocab": {"[UNK]": 0, "a": 1, "[CLS]": 2, "[SEP]": 3, "b": 4}}
    }"###;
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("tokenizer.json");
    std::fs::write(&path, json).unwrap();
    let tok = Tokenizer::from_file(&path).unwrap();

    let enc = tok.encode("a[SEP]b", true);
    assert_eq!(enc.get_ids(), &[2, 1, 3, 4, 3]);
    assert_eq!(enc.get_word_ids(), &[None, Some(0), Some(1), Some(2), None]);
    // add_special_tokens = false: no [CLS]/[SEP] wrapper.
    let enc = tok.encode("a b", false);
    assert_eq!(enc.get_ids(), &[1, 4]);
}

#[test]
fn test_added_token_lstrip_rstrip() {
    let json = r###"{
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [
            {"id": 9, "content": "<T>", "single_word": false, "lstrip": true,
             "rstrip": true, "normalized": false, "special": false}
        ],
        "normalizer": null, "pre_tokenizer": {"type": "BertPreTokenizer"},
        "post_processor": null, "decoder": null,
        "model": {"type": "WordPiece", "unk_token": "[UNK]",
                  "continuing_subword_prefix": "##",
                  "max_input_chars_per_word": 100,
                  "vocab": {"[UNK]": 0, "a": 1, "b": 2, "<T>": 9}}
    }"###;
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("tokenizer.json");
    std::fs::write(&path, json).unwrap();
    let tok = Tokenizer::from_file(&path).unwrap();
    // Surrounding whitespace is absorbed into the added-token span, so no
    // text span materializes around it.
    let enc = tok.encode("a <T> b", false);
    assert_eq!(enc.get_ids(), &[1, 9, 2]);
    assert_eq!(enc.get_word_ids(), &[Some(0), Some(1), Some(2)]);
}

#[test]
fn test_unicode_category_spot_checks() {
    // C*: control (Cc), format (Cf), private use (Co); \t\n\r excluded.
    assert!(is_control('\x00'));
    assert!(is_control('\x7F'));
    assert!(is_control('\u{200B}')); // ZWSP (Cf)
    assert!(is_control('\u{AD}')); // soft hyphen (Cf)
    assert!(is_control('\u{E000}')); // PUA (Co)
    assert!(is_control('\u{10FFFD}')); // plane 16 PUA (Co)
    assert!(!is_control('\t'));
    assert!(!is_control('\n'));
    assert!(!is_control('а'));
    // Unassigned (Cn) is NOT control: unicode_categories has no Cn table.
    assert!(!is_control('\u{10FFFF}'));
    // P*: dashes, quotes, CJK brackets; symbols are not punctuation.
    assert!(is_bert_punc('!'));
    assert!(is_bert_punc('+')); // ASCII branch (Unicode Sm)
    assert!(is_bert_punc('_')); // Pc
    assert!(is_bert_punc('\u{2014}')); // em dash (Pd)
    assert!(is_bert_punc('\u{300A}')); // 《 (Ps)
    assert!(!is_bert_punc('\u{2212}')); // minus sign (Sm)
    assert!(!is_bert_punc('中'));
    // CJK ranges.
    assert!(is_chinese_char('中'));
    assert!(is_chinese_char('野'));
    assert!(!is_chinese_char('が')); // hiragana is not CJK ideograph
    assert!(!is_chinese_char('а'));
}
