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
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use parking_lot::Mutex;
use tokenizers::Tokenizer;

use crate::runtime::{
    factory::RuntimeFactory,
    session::RuntimeSession,
    tensor::{Shape, Tensor, TensorData},
};

/// Basename of the INT8 ONNX punctuation model inside the punct model dir.
pub const PUNCT_MODEL_FILE: &str = "rupunct_small_int8.onnx";
/// Basename of the HuggingFace tokenizer JSON inside the punct model dir.
pub const PUNCT_TOKENIZER_FILE: &str = "tokenizer.json";
/// Basename of the model config JSON (carries `id2label`) inside the punct model dir.
pub const PUNCT_CONFIG_FILE: &str = "config.json";

/// Whitespace words labelled in one model run.
///
/// The exported RUPunct graph is fully dynamic but its position-embedding table
/// has 2048 rows, so a single run over a whole long transcript overflows the
/// embedding and fails the entire pass. 250 Russian words are roughly 600–900
/// WordPiece subtokens, which leaves a wide margin under that ceiling.
const WINDOW_WORDS: usize = 250;

/// Words shared by neighbouring windows; must be even and below [`WINDOW_WORDS`].
///
/// Each window keeps only the labels of its middle and drops half of the overlap
/// on either side, so (except at the very start / end of the transcript) every
/// word is labelled from a window in which it has real left and right context.
const WINDOW_OVERLAP_WORDS: usize = 40;

/// Hard ceiling on subtokens submitted in one run, kept below the model's 2048
/// position rows. A window whose lexis still encodes above this is halved until
/// it fits.
const MAX_WINDOW_SUBTOKENS: usize = 2000;

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

/// Byte spans of the whitespace-separated words of `text`, in order.
///
/// Same split as [`str::split_whitespace`], but each word keeps its byte range so
/// a run of words can be sliced back out of the original string. The slice of a
/// window spanning every word is the input string itself, which is what keeps a
/// single-window transcript byte-identical to the un-windowed path.
fn word_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start: Option<usize> = None;
    for (i, c) in text.char_indices() {
        if c.is_whitespace() {
            if let Some(s) = start.take() {
                spans.push((s, i));
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        spans.push((s, text.len()));
    }
    spans
}

/// One model window: the words encoded together (`start..end`) and the sub-range
/// whose labels are kept (`keep_start..keep_end`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Window {
    start: usize,
    end: usize,
    keep_start: usize,
    keep_end: usize,
}

/// Tile `num_words` words with overlapping windows of at most [`WINDOW_WORDS`].
///
/// Windows advance by `WINDOW_WORDS - WINDOW_OVERLAP_WORDS` and the kept ranges
/// cut each overlap in half, so the kept ranges tile `0..num_words` with no gap
/// and no repeat while every window still sees `WINDOW_OVERLAP_WORDS / 2` words
/// of context beyond what it labels.
fn plan_windows(num_words: usize) -> Vec<Window> {
    if num_words == 0 {
        return Vec::new();
    }
    if num_words <= WINDOW_WORDS {
        return vec![Window {
            start: 0,
            end: num_words,
            keep_start: 0,
            keep_end: num_words,
        }];
    }

    let stride = WINDOW_WORDS - WINDOW_OVERLAP_WORDS;
    let half = WINDOW_OVERLAP_WORDS / 2;
    let mut windows = Vec::new();
    let mut start = 0usize;
    loop {
        let end = (start + WINDOW_WORDS).min(num_words);
        let is_last = end == num_words;
        windows.push(Window {
            start,
            end,
            keep_start: if start == 0 { 0 } else { start + half },
            keep_end: if is_last { num_words } else { end - half },
        });
        if is_last {
            break;
        }
        start += stride;
    }
    windows
}

/// Merge the per-window label vectors into one label per word.
///
/// `per_window[i]` holds a label for every word of `windows[i]` (word `start + j`
/// is at index `j`), or `None` when that window's inference failed. Words only a
/// failed window covered stay `None` and are rendered unchanged.
fn splice_window_labels(
    windows: &[Window],
    per_window: &[Option<Vec<usize>>],
    num_words: usize,
) -> Vec<Option<usize>> {
    let mut merged = vec![None; num_words];
    for (window, labels) in windows.iter().zip(per_window.iter()) {
        let Some(labels) = labels else { continue };
        let keep_end = window.keep_end.min(num_words);
        let keep_start = window.keep_start.min(keep_end);
        let Some(base) = keep_start.checked_sub(window.start) else {
            continue;
        };
        for (offset, slot) in merged[keep_start..keep_end].iter_mut().enumerate() {
            *slot = labels.get(base + offset).copied();
        }
    }
    merged
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
    session: Mutex<Box<dyn RuntimeSession>>,
    tokenizer: Tokenizer,
    /// `id2label[i]` is the label name for logit index `i`.
    id2label: Vec<String>,
    /// Windows whose inference failed since load — see [`Punctuator::failed_windows`].
    failed_windows: AtomicU64,
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
        let factory = crate::runtime::cpu_factory();
        Self::load_with_factory(model_dir, factory.as_ref())
    }

    /// Like [`Punctuator::load`], but loads the ONNX session through a
    /// caller-supplied `RuntimeFactory` (e.g. a non-`ort` backend or a test
    /// mock) instead of the default CPU `ort` runtime.
    pub fn load_with_factory(model_dir: &Path, factory: &dyn RuntimeFactory) -> Result<Self> {
        let model_path = model_dir.join(PUNCT_MODEL_FILE);
        let tokenizer_path = model_dir.join(PUNCT_TOKENIZER_FILE);
        let config_path = model_dir.join(PUNCT_CONFIG_FILE);

        let id2label = load_id2label(&config_path)
            .with_context(|| format!("Failed to load id2label from {}", config_path.display()))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            anyhow::anyhow!("Failed to load tokenizer {}: {e}", tokenizer_path.display())
        })?;

        tracing::debug!("Loading punctuation model from {}", model_path.display());
        let runtime = factory
            .cpu_fallback()
            .create(1)
            .map_err(|e| anyhow::anyhow!(e))
            .context("Failed to create runtime for punctuation model")?;
        let session = runtime
            .load_session(&model_path, false)
            .map_err(|e| anyhow::anyhow!(e))
            .context("Failed to load punctuation model")?;

        tracing::info!(
            "Punctuation model loaded ({} labels) from {}",
            id2label.len(),
            model_dir.display()
        );

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            id2label,
            failed_windows: AtomicU64::new(0),
        })
    }

    /// Number of windows whose inference has failed since this model was loaded.
    ///
    /// [`restore`](Self::restore) degrades quietly by contract — the words of a
    /// failed window come back bare — so this counter, together with the `warn!`
    /// it logs, is how a caller notices that punctuation was applied only
    /// partially (or not at all).
    pub fn failed_windows(&self) -> u64 {
        self.failed_windows.load(Ordering::Relaxed)
    }

    /// Restore punctuation + capitalization on a space-separated transcript.
    ///
    /// Replicates RUPunct's pipeline: encode the text, run the BERT token
    /// classifier, take each word's first-subtoken label, apply [`process_token`],
    /// and join with single spaces (trimmed).
    ///
    /// A transcript of more than a couple hundred words is labelled in
    /// overlapping windows — the model's position table would otherwise overflow
    /// and cost the whole transcript its punctuation. Anything that fits in one
    /// window takes the same single encode + single run it always did.
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
        let spans = word_spans(text);
        if spans.is_empty() {
            return Ok(text.to_string());
        }

        // A transcript longer than one window is labelled window by window: one
        // encode + one run each, so no sequence can outgrow the model's position
        // table. A transcript that fits in one window takes exactly the single
        // encode + single run it always did.
        let windows = plan_windows(spans.len());
        let mut per_window: Vec<Option<Vec<usize>>> = Vec::with_capacity(windows.len());
        let mut first_error: Option<anyhow::Error> = None;
        for window in &windows {
            match self.label_word_range(text, &spans, window.start, window.end) {
                Ok(labels) => per_window.push(Some(labels)),
                Err(e) => {
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                    per_window.push(None);
                }
            }
        }

        let failed = per_window.iter().filter(|labels| labels.is_none()).count();
        if failed > 0 {
            self.failed_windows
                .fetch_add(failed as u64, Ordering::Relaxed);
        }
        if failed == windows.len() {
            // Nothing could be labelled: keep the un-windowed contract and let
            // `restore` log and hand back the input text untouched.
            return Err(
                first_error.unwrap_or_else(|| anyhow::anyhow!("punct model produced no labels"))
            );
        }
        if failed > 0 {
            let detail = first_error.map_or_else(String::new, |e| format!("{e:#}"));
            tracing::warn!(
                "Punctuation restore: {failed} of {} windows failed, their words stay bare: {detail}",
                windows.len()
            );
        }

        let label_ids = splice_window_labels(&windows, &per_window, spans.len());
        let mut out = String::new();
        for (&(from, to), lid) in spans.iter().zip(label_ids.iter()) {
            let word = &text[from..to];
            // A word whose window failed keeps `LOWER_O`, i.e. comes back bare.
            let label = lid
                .and_then(|lid| self.id2label.get(lid))
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

    /// Label the words `start..end`, returning one label id per word of the range.
    ///
    /// One encode + one run, unless the range encodes above
    /// [`MAX_WINDOW_SUBTOKENS`] — then it is halved and each half labelled
    /// separately, so no sequence longer than the model's position table is ever
    /// submitted.
    fn label_word_range(
        &self,
        text: &str,
        spans: &[(usize, usize)],
        start: usize,
        end: usize,
    ) -> Result<Vec<usize>> {
        if start >= end || end > spans.len() {
            anyhow::bail!(
                "invalid word window {start}..{end} over {} words",
                spans.len()
            );
        }
        let chunk = &text[spans[start].0..spans[end - 1].1];

        let encoding = self
            .tokenizer
            .encode(chunk, true)
            .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;

        let seq = encoding.get_ids().len();
        if seq > MAX_WINDOW_SUBTOKENS {
            let num_words = end - start;
            if num_words < 2 {
                anyhow::bail!(
                    "a single word encodes to {seq} subtokens (max {MAX_WINDOW_SUBTOKENS})"
                );
            }
            let mid = start + num_words / 2;
            let mut labels = self.label_word_range(text, spans, start, mid)?;
            labels.extend(self.label_word_range(text, spans, mid, end)?);
            return Ok(labels);
        }

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&i| i as i64).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let token_type_ids = vec![0i64; seq];

        let input_ids = Tensor::new(Shape::new(vec![1, seq]), TensorData::I64(ids))?;
        let attention_mask = Tensor::new(Shape::new(vec![1, seq]), TensorData::I64(mask))?;
        let token_type = Tensor::new(Shape::new(vec![1, seq]), TensorData::I64(token_type_ids))?;

        // Run the session and reduce the borrowed logits to an owned
        // per-token argmax inside this scope.
        let num_labels = self.id2label.len();
        let argmax_per_token: Vec<usize> = {
            let session = self.session.lock();
            let outputs = session
                .run(&[input_ids, attention_mask, token_type])
                .context("punct model inference failed")?;

            let logits_view = outputs[0].view();
            let logits = logits_view
                .data()
                .as_f32()
                .context("failed to extract punct logits")?;

            // Expect [1, seq, num_labels].
            let shape = logits_view.shape().dims();
            if shape != [1, seq, num_labels] {
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

        Ok(first_subword_labels(
            encoding.get_word_ids(),
            &argmax_per_token,
            end - start,
        ))
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
    use std::sync::Arc;

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

    /// A minimal but valid HuggingFace tokenizer.json (WordLevel model). Used to
    /// drive `Punctuator::load` past the tokenizer step so the ONNX-session
    /// failure branch is exercised without the real model.
    const MINIMAL_TOKENIZER_JSON: &str = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": {"[UNK]": 0, "a": 1, "b": 2},
            "unk_token": "[UNK]"
        }
    }"#;

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
        let words: Vec<String> = (0..5000).map(|i| format!("w{i}")).collect();
        let text = words.join(" ");
        let spans = word_spans(&text);
        assert_eq!(spans.len(), 5000);

        let windows = plan_windows(spans.len());
        let per_window: Vec<Option<Vec<usize>>> = windows
            .iter()
            .map(|w| Some((w.start..w.end).collect()))
            .collect();

        let merged = splice_window_labels(&windows, &per_window, spans.len());
        let expected: Vec<Option<usize>> = (0..5000).map(Some).collect();
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
        "pre_tokenizer": {"type": "Whitespace"},
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
        let words: Vec<String> = (0..5000).map(|i| format!("w{i}")).collect();
        let text = words.join(" ");
        let (punct, seqs) =
            stub_punctuator(MINIMAL_TOKENIZER_JSON, &["LOWER_O", "UPPER_O"], 1, None);

        let out = punct.restore(&text);

        let expected: Vec<String> = words.iter().map(|w| capitalize(w)).collect();
        assert_eq!(out, expected.join(" "));
        let seqs = seqs.lock();
        assert_eq!(seqs.len(), plan_windows(5000).len());
        assert!(seqs.iter().all(|&s| s <= WINDOW_WORDS), "{seqs:?}");
        assert_eq!(punct.failed_windows(), 0);
    }

    /// A window whose lexis blows past the subtoken ceiling is split until every
    /// submitted sequence fits the model's position table.
    #[test]
    fn test_restore_splits_a_window_over_the_subtoken_ceiling() {
        // 250 words × 40 subtokens ≈ 10k subtokens in one window.
        let word = "a".repeat(40);
        let text = vec![word.as_str(); WINDOW_WORDS].join(" ");
        let (punct, seqs) =
            stub_punctuator(SPLITTING_TOKENIZER_JSON, &["LOWER_O", "UPPER_O"], 1, None);

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
        let out =
            punct.restore("привет меня зовут анна сколько будет стоить шестьдесят тысяч тенге");
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
}
