//! Optional punctuation + capitalization restoration for the plain `rnnt` head.
//!
//! The plain RNN-T recognition head ([`ModelVariant::Rnnt`](crate::model::ModelVariant::Rnnt))
//! emits bare lowercase Russian with no punctuation, e.g.
//! `"шестьдесят тысяч тенге сколько будет стоить"`. This module restores
//! punctuation and casing as an *optional* post-processing pass, producing
//! e.g. `"Шестьдесят тысяч тенге, сколько будет стоить?"`.
//!
//! The model is `RUPunct/RUPunct_small` (MIT), exported to ONNX and INT8-quantized
//! (dynamic MatMulInteger — runs on the CPU EP like the encoder). It is a BERT
//! token-classification head: each WordPiece subtoken gets one of 33 labels
//! (`{LOWER, UPPER, UPPER_TOTAL}` × 11 punctuation classes). We replicate the
//! RUPunct `aggregation_strategy="first"` inference: take the label of each
//! word's FIRST subtoken and apply [`process_token`].
//!
//! This is *optional*: a build or run without the punct model behaves exactly as
//! before. If the model dir / files are absent or the model fails to load,
//! [`Punctuator::load`] returns an error which the caller treats as "punctuation
//! disabled" (the engine logs a warning once and returns input text unchanged).
//!
//! NOTE (distribution): the exported ONNX artifact is published at the
//! `ekhodzitsky/rupunct-small-onnx` HuggingFace repo (public, MIT) and
//! auto-downloads into the punct model dir (`--punct-model-dir`, default
//! `~/.gigastt/models/punct/`) on first use via
//! [`crate::model::ensure_punct_model`]. A local dir is still honoured if
//! pre-populated. sha256 of the int8 ONNX:
//! `b105da023474d98aa13ba18953ae67b04b17bd0595034bc06030c17536893933`.

use std::path::Path;

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::TensorRef;
use parking_lot::Mutex;
use tokenizers::Tokenizer;

/// Basename of the INT8 ONNX punctuation model inside the punct model dir.
pub const PUNCT_MODEL_FILE: &str = "rupunct_small_int8.onnx";
/// Basename of the HuggingFace tokenizer JSON inside the punct model dir.
pub const PUNCT_TOKENIZER_FILE: &str = "tokenizer.json";
/// Basename of the model config JSON (carries `id2label`) inside the punct model dir.
pub const PUNCT_CONFIG_FILE: &str = "config.json";

fn ort_err(e: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

/// Apply Python `str.capitalize()` semantics to a token: first character
/// uppercased, every following character lowercased. Operates over Unicode
/// `char`s (Russian Cyrillic), matching RUPunct's reference decode.
fn capitalize(token: &str) -> String {
    let mut chars = token.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => {
            let mut out: String = first.to_uppercase().collect();
            for c in chars {
                out.extend(c.to_lowercase());
            }
            out
        }
    }
}

/// Cased + punctuated rendering of one word given its RUPunct label.
///
/// Verbatim port of the reference `process_token(token, label)` from the
/// `RUPunct/RUPunct_small` model card. Case transform:
/// `LOWER_*` keeps the token, `UPPER_*` applies `capitalize` (Python
/// `str.capitalize`), `UPPER_TOTAL_*` upper-cases the whole token. Punctuation
/// is appended as a suffix. SPACING QUIRK preserved exactly: `LOWER_TIRE`
/// appends `"—"` (no leading space) while `UPPER_TIRE` / `UPPER_TOTAL_TIRE`
/// append `" —"` (leading space). Unknown labels leave the token unchanged.
pub fn process_token(token: &str, label: &str) -> String {
    // Split the label into its case prefix and punctuation suffix. The longest
    // prefix `UPPER_TOTAL_` must be tried before `UPPER_`.
    let (cased, punct_class) = if let Some(rest) = label.strip_prefix("UPPER_TOTAL_") {
        (token.to_uppercase(), rest)
    } else if let Some(rest) = label.strip_prefix("UPPER_") {
        (capitalize(token), rest)
    } else if let Some(rest) = label.strip_prefix("LOWER_") {
        (token.to_string(), rest)
    } else {
        // Unknown / malformed label: leave the token untouched.
        return token.to_string();
    };

    let is_upper = !label.starts_with("LOWER_");
    let suffix: &str = match punct_class {
        "O" => "",
        "PERIOD" => ".",
        "COMMA" => ",",
        "QUESTION" => "?",
        "VOSKL" => "!",
        "DVOETOCHIE" => ":",
        "PERIODCOMMA" => ";",
        "DEFIS" => "-",
        "MNOGOTOCHIE" => "...",
        "QUESTIONVOSKL" => "?!",
        // Em-dash spacing quirk: lower has no leading space, upper variants do.
        "TIRE" => {
            if is_upper {
                " —"
            } else {
                "—"
            }
        }
        // Unknown punctuation class: no suffix.
        _ => "",
    };

    let mut out = cased;
    out.push_str(suffix);
    out
}

/// For each whitespace word index `0..num_words`, return the label id of its
/// FIRST subtoken — the token whose `word_id == Some(w)` with the lowest
/// position. This is RUPunct's `aggregation_strategy="first"`.
///
/// `word_ids` is the per-token word mapping (special tokens are `None`);
/// `argmax_per_token` is the pre-computed argmax label id for each token.
/// Words with no subtoken (should not happen for real input) get label id 0.
///
/// Pure (no model / I/O) so the first-subword selection is unit-testable.
fn first_subword_labels(
    word_ids: &[Option<u32>],
    argmax_per_token: &[usize],
    num_words: usize,
) -> Vec<usize> {
    let mut labels = vec![0usize; num_words];
    let mut seen = vec![false; num_words];
    for (tok_idx, wid) in word_ids.iter().enumerate() {
        let Some(w) = wid else { continue };
        let w = *w as usize;
        if w < num_words && !seen[w] {
            seen[w] = true;
            labels[w] = argmax_per_token.get(tok_idx).copied().unwrap_or(0);
        }
    }
    labels
}

/// Argmax over the last `num_labels`-sized window of a logits row.
fn argmax(row: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// Punctuation + capitalization restorer backed by the RUPunct ONNX model.
///
/// Loaded from a model dir via [`Punctuator::load`]. The single ONNX session is
/// guarded by a [`Mutex`] because the punct pass runs on already-decoded text
/// (off the hot inference loop) and is not worth pooling. [`restore`](Self::restore)
/// is the public entry point and never panics: on any internal failure it logs
/// and returns the input text unchanged.
pub struct Punctuator {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    /// `id2label[i]` is the label name for logit index `i`.
    id2label: Vec<String>,
}

impl Punctuator {
    /// Load the punctuation model, tokenizer, and label map from `model_dir`.
    ///
    /// Expects `rupunct_small_int8.onnx`, `tokenizer.json`, and `config.json`
    /// (with an `id2label` map) in `model_dir`.
    ///
    /// # Errors
    ///
    /// Returns an error if any file is missing or fails to parse / load. The
    /// caller treats an error as "punctuation unavailable" and proceeds without
    /// it — restoration is optional post-processing.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let model_path = model_dir.join(PUNCT_MODEL_FILE);
        let tokenizer_path = model_dir.join(PUNCT_TOKENIZER_FILE);
        let config_path = model_dir.join(PUNCT_CONFIG_FILE);

        let id2label = load_id2label(&config_path)
            .with_context(|| format!("Failed to load id2label from {}", config_path.display()))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            anyhow::anyhow!("Failed to load tokenizer {}: {e}", tokenizer_path.display())
        })?;

        let session = Session::builder()
            .map_err(ort_err)?
            .commit_from_file(&model_path)
            .map_err(ort_err)
            .with_context(|| format!("Failed to load punct model {}", model_path.display()))?;

        tracing::info!(
            "Punctuation model loaded ({} labels) from {}",
            id2label.len(),
            model_dir.display()
        );

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            id2label,
        })
    }

    /// Restore punctuation + capitalization on a space-separated transcript.
    ///
    /// Replicates RUPunct's pipeline: encode the text, run the BERT token
    /// classifier, take each word's first-subtoken label, apply [`process_token`],
    /// and join with single spaces (trimmed).
    ///
    /// Never fails: on empty input or any internal error it returns the input
    /// text unchanged (the error is logged at `warn`). This keeps the punct pass
    /// strictly optional — a transcription is never blocked by it.
    pub fn restore(&self, text: &str) -> String {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return text.to_string();
        }
        match self.restore_inner(trimmed) {
            Ok(out) => out,
            Err(e) => {
                tracing::warn!("Punctuation restore failed, returning bare text: {e:#}");
                text.to_string()
            }
        }
    }

    fn restore_inner(&self, text: &str) -> Result<String> {
        // Whitespace words: the decoder output is space-separated, so this is
        // the word granularity the labels are aggregated to.
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Ok(text.to_string());
        }

        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&i| i as i64).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let seq = ids.len();
        let token_type_ids = vec![0i64; seq];

        let input_ids = TensorRef::from_array_view(([1_usize, seq], ids.as_slice()))?;
        let attention_mask = TensorRef::from_array_view(([1_usize, seq], mask.as_slice()))?;
        let token_type = TensorRef::from_array_view(([1_usize, seq], token_type_ids.as_slice()))?;

        // Run the session and reduce the borrowed logits to an owned
        // per-token argmax inside this scope, so the `outputs` borrow (which
        // ties the lifetime to the session guard) is released before the
        // session guard is dropped at end of scope.
        let num_labels = self.id2label.len();
        let argmax_per_token: Vec<usize> = {
            let mut session = self.session.lock();
            let outputs = session
                .run(ort::inputs![
                    "input_ids" => input_ids,
                    "attention_mask" => attention_mask,
                    "token_type_ids" => token_type,
                ])
                .context("punct model inference failed")?;

            let (shape, logits) = outputs["logits"]
                .try_extract_tensor::<f32>()
                .context("failed to extract punct logits")?;

            // Expect [1, seq, num_labels].
            if shape.len() != 3 || shape[2] as usize != num_labels {
                anyhow::bail!(
                    "unexpected punct logits shape {shape:?} (expected [1, {seq}, {num_labels}])"
                );
            }

            (0..seq)
                .map(|t| {
                    let start = t * num_labels;
                    argmax(&logits[start..start + num_labels])
                })
                .collect()
        };

        let label_ids =
            first_subword_labels(encoding.get_word_ids(), &argmax_per_token, words.len());

        let mut out = String::new();
        for (word, &lid) in words.iter().zip(label_ids.iter()) {
            let label = self
                .id2label
                .get(lid)
                .map(String::as_str)
                .unwrap_or("LOWER_O");
            let processed = process_token(word, label);
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(&processed);
        }
        Ok(out.trim().to_string())
    }
}

/// Parse the `id2label` map from a HuggingFace `config.json` into a dense
/// `Vec<String>` indexed by label id.
fn load_id2label(config_path: &Path) -> Result<Vec<String>> {
    let raw = std::fs::read_to_string(config_path)
        .with_context(|| format!("Failed to read {}", config_path.display()))?;
    let config: serde_json::Value =
        serde_json::from_str(&raw).context("config.json is not valid JSON")?;
    let map = config
        .get("id2label")
        .and_then(|v| v.as_object())
        .context("config.json missing id2label object")?;

    // Keys are stringified indices ("0".."32"); place each at its index.
    let mut labels = vec![String::new(); map.len()];
    for (k, v) in map {
        let idx: usize = k
            .parse()
            .with_context(|| format!("id2label key '{k}' is not an integer"))?;
        let label = v
            .as_str()
            .with_context(|| format!("id2label['{k}'] is not a string"))?;
        if idx >= labels.len() {
            anyhow::bail!("id2label index {idx} out of range ({} labels)", map.len());
        }
        labels[idx] = label.to_string();
    }
    if labels.iter().any(|l| l.is_empty()) {
        anyhow::bail!("id2label has a gap (non-contiguous indices)");
    }
    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// End-to-end on the real ONNX model (model-gated, like other model tests).
    /// Validates the full tokenizer → ONNX → first-subword → process_token
    /// pipeline against the RUPunct reference string.
    #[test]
    #[ignore = "requires punct model at ~/.gigastt/models/punct"]
    fn test_restore_reference_string() {
        let dir = default_punct_model_dir();
        let punct = Punctuator::load(Path::new(&dir)).expect("load punct model");
        let out =
            punct.restore("привет меня зовут анна сколько будет стоить шестьдесят тысяч тенге");
        assert_eq!(
            out,
            "Привет меня зовут Анна, Сколько будет стоить шестьдесят тысяч тенге."
        );
    }

    use crate::model::default_punct_model_dir;
}
