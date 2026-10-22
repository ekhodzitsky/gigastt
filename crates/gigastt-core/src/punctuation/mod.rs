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

use crate::runtime::{
    factory::RuntimeFactory,
    session::RuntimeSession,
    tensor::{Shape, Tensor, TensorData},
};
use crate::wordpiece::Tokenizer;

/// Basename of the INT8 ONNX punctuation model inside the punct model dir.
pub const PUNCT_MODEL_FILE: &str = "rupunct_small_int8.onnx";
/// Basename of the HuggingFace tokenizer JSON inside the punct model dir.
pub const PUNCT_TOKENIZER_FILE: &str = "tokenizer.json";
/// Basename of the model config JSON (carries `id2label`) inside the punct model dir.
pub const PUNCT_CONFIG_FILE: &str = "config.json";

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

mod windows;
use windows::{first_subword_labels, plan_windows, splice_window_labels, word_spans};

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

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .with_context(|| format!("Failed to load tokenizer {}", tokenizer_path.display()))?;

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

        let encoding = self.tokenizer.encode(chunk, true);

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
        #[cfg(test)]
        let mut probe = crate::sidecar_probe::Probe::new("punctuation");
        let argmax_per_token: Vec<usize> = {
            let session = self.session.lock();
            #[cfg(test)]
            probe.acquired();
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
        #[cfg(test)]
        drop(probe);

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
mod tests;

#[cfg(feature = "rupunct")]
mod native;
#[cfg(feature = "rupunct")]
pub use native::NativePunctuator;

pub(crate) enum Backend {
    Onnx(Punctuator),
    #[cfg(feature = "rupunct")]
    Native(Box<NativePunctuator>),
}
impl Backend {
    pub(crate) fn restore(&self, text: &str) -> String {
        match self {
            Self::Onnx(model) => model.restore(text),
            #[cfg(feature = "rupunct")]
            Self::Native(model) => model.restore(text),
        }
    }
}
