//! In-tree WordPiece tokenizer for the RUPunct punctuation model.
//!
//! Replaces the `tokenizers` crate (0.23) on the punctuation path. The scope is
//! exactly the scheme shipped in the punct model's `tokenizer.json`:
//! `BertNormalizer { clean_text, handle_chinese_chars, strip_accents: false,
//! lowercase: false }` + `BertPreTokenizer` + greedy WordPiece (`##`
//! continuation, `[UNK]` fallback, 100-char word cap) + the `[CLS] A [SEP]`
//! template. Anything outside this scheme (lowercase / accent-stripping
//! normalizers, other pre-tokenizers or models, truncation / padding,
//! normalized or single-word added tokens) fails loudly at load time instead
//! of being silently mis-implemented.
//!
//! Byte-for-byte parity with the upstream `tokenizers` crate is enforced by
//! the golden fixtures in `tests/fixtures/wordpiece_golden.json` — produced by
//! the real crate over a corpus of Russian phrases and Unicode edges — which
//! the unit tests replay against the reduced vocab in
//! `tests/fixtures/wordpiece_tokenizer.json`.
//!
//! The `CONTROL_FORMAT_RANGES` / `PUNCTUATION_RANGES` tables are a compact
//! range encoding of the category tables in the `unicode_categories` 0.1.1
//! crate — the exact data source the upstream `tokenizers` 0.23 normalizer and
//! pre-tokenizer consult — so `is_other` / `is_punctuation` decisions match
//! the upstream crate char for char.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};

mod unicode;
pub(super) use unicode::{is_bert_punc, is_chinese_char, is_control};

/// WordPiece tokenizer loaded from a HuggingFace `tokenizer.json`.
///
/// Exposes only the surface the punctuation backends use: [`Tokenizer::from_file`],
/// [`Tokenizer::encode_words`], and the [`Encoding`] getters.
pub(crate) struct Tokenizer {
    vocab: HashMap<String, u32>,
    unk_id: u32,
    continuing_subword_prefix: String,
    max_input_chars_per_word: usize,
    normalizer: Normalizer,
    /// Whether the `BertPreTokenizer` split step is active (a null
    /// `pre_tokenizer` sends the whole normalized span to WordPiece as one
    /// word, as upstream does).
    bert_pre_tokenizer: bool,
    /// Added tokens extracted from the input before normalization, matched
    /// leftmost-longest (all `special: true, normalized: false` in the punct
    /// model's tokenizer.json).
    added_tokens: Vec<AddedToken>,
    /// `[CLS]` / `[SEP]` from the `TemplateProcessing` post-processor; `None`
    /// when the file declares no post-processor.
    cls: Option<u32>,
    sep: Option<u32>,
}

/// Encoded token ids, an all-ones attention mask (no padding is ever applied),
/// and per-token word indices (`None` for the `[CLS]` / `[SEP]` template tokens).
/// Indices identify caller-supplied words for pre-tokenized input, otherwise
/// the BERT pre-tokenizer's splits.
pub(crate) struct Encoding {
    ids: Vec<u32>,
    attention_mask: Vec<u32>,
    word_ids: Vec<Option<u32>>,
}

impl Encoding {
    pub(crate) fn get_ids(&self) -> &[u32] {
        &self.ids
    }

    pub(crate) fn get_attention_mask(&self) -> &[u32] {
        &self.attention_mask
    }

    pub(crate) fn get_word_ids(&self) -> &[Option<u32>] {
        &self.word_ids
    }
}

/// One `added_tokens[]` entry: a literal string pulled out of the input before
/// normalization and emitted as a single token with the given id.
struct AddedToken {
    id: u32,
    content: String,
    lstrip: bool,
    rstrip: bool,
}

/// Normalizer configuration; only the modes the punct model uses are
/// supported (see module docs).
pub(super) enum Normalizer {
    None,
    Bert {
        clean_text: bool,
        handle_chinese_chars: bool,
    },
}

impl Tokenizer {
    /// Load from a HuggingFace `tokenizer.json`. Fails loudly on any construct
    /// outside the supported scheme.
    pub(crate) fn from_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read tokenizer file {}", path.display()))?;
        let file: TokenizerFile = serde_json::from_str(&raw)
            .with_context(|| format!("failed to parse tokenizer file {}", path.display()))?;
        Self::from_parsed(file)
            .with_context(|| format!("unsupported tokenizer file {}", path.display()))
    }

    fn from_parsed(file: TokenizerFile) -> Result<Self> {
        if let Some(version) = &file.version
            && version != "1.0"
        {
            bail!("unknown tokenizer version '{version}'");
        }
        if file.truncation.is_some() {
            bail!("truncation is not supported (expected null)");
        }
        if file.padding.is_some() {
            bail!("padding is not supported (expected null)");
        }
        if file.model.kind != "WordPiece" {
            bail!(
                "unsupported model type '{}' (expected 'WordPiece')",
                file.model.kind
            );
        }
        if file.model.vocab.is_empty() {
            bail!("WordPiece vocabulary is empty");
        }

        let unk_token = file.model.unk_token.unwrap_or_else(|| "[UNK]".to_string());
        let unk_id =
            *file.model.vocab.get(&unk_token).ok_or_else(|| {
                anyhow!("WordPiece vocabulary is missing unk_token '{unk_token}'")
            })?;

        let normalizer = parse_normalizer(file.normalizer)?;
        let bert_pre_tokenizer = parse_pre_tokenizer(file.pre_tokenizer)?;
        let (cls, sep) = parse_post_processor(file.post_processor)?;

        let mut added_tokens = Vec::with_capacity(file.added_tokens.len());
        for token in &file.added_tokens {
            if token.content.is_empty() {
                continue;
            }
            if token.normalized {
                bail!(
                    "added token '{}' has normalized=true, which is not supported",
                    token.content
                );
            }
            if token.single_word {
                bail!(
                    "added token '{}' has single_word=true, which is not supported",
                    token.content
                );
            }
            // The upstream crate resolves the id by vocab lookup, falling back
            // to fresh ids; the JSON `id` field only feeds a mismatch warning.
            let id = file
                .model
                .vocab
                .get(&token.content)
                .copied()
                .unwrap_or(token.id);
            added_tokens.push(AddedToken {
                id,
                content: token.content.clone(),
                lstrip: token.lstrip,
                rstrip: token.rstrip,
            });
        }

        Ok(Self {
            vocab: file.model.vocab,
            unk_id,
            continuing_subword_prefix: file
                .model
                .continuing_subword_prefix
                .unwrap_or_else(|| "##".to_string()),
            max_input_chars_per_word: file.model.max_input_chars_per_word.unwrap_or(100),
            normalizer,
            bert_pre_tokenizer,
            added_tokens,
            cls,
            sep,
        })
    }

    /// Encode a single sequence. Mirrors `tokenizers::Tokenizer::encode(text,
    /// add_special_tokens)` for the supported scheme: added-token extraction,
    /// BertNormalizer, BertPreTokenizer, greedy WordPiece, `[CLS]`/`[SEP]`
    /// template. Infallible: every fallible condition is rejected at load.
    #[cfg(test)]
    pub(crate) fn encode(&self, text: &str, add_special_tokens: bool) -> Encoding {
        self.encode_segments([(text, None)], add_special_tokens)
    }

    /// Encode caller-supplied whitespace words, as pre-tokenized input.
    /// Punctuation and CJK splits within a word inherit its index, and a
    /// word removed by normalization does not consume the next word's index.
    /// The special-token template is still applied once to the full sequence.
    pub(crate) fn encode_words(&self, text: &str, add_special_tokens: bool) -> Encoding {
        let segments = text
            .split_whitespace()
            .enumerate()
            .map(|(i, word)| (word, Some(i as u32)));
        self.encode_segments(segments, add_special_tokens)
    }

    fn encode_segments<'a>(
        &self,
        segments: impl IntoIterator<Item = (&'a str, Option<u32>)>,
        add_special_tokens: bool,
    ) -> Encoding {
        let mut ids: Vec<u32> = Vec::new();
        let mut word_ids: Vec<Option<u32>> = Vec::new();

        if add_special_tokens && let Some(cls) = self.cls {
            ids.push(cls);
            word_ids.push(None);
        }

        let mut next_word: u32 = 0;
        for (text, input_word) in segments {
            for span in self.extract_added(text) {
                match span {
                    Span::Added(idx) => {
                        ids.push(self.added_tokens[idx].id);
                        word_ids.push(Some(input_word.unwrap_or(next_word)));
                        next_word += 1;
                    }
                    Span::Text(slice) => {
                        let normalized = self.normalize(slice);
                        let words = if self.bert_pre_tokenizer {
                            bert_pre_tokenize(&normalized)
                        } else if normalized.is_empty() {
                            Vec::new()
                        } else {
                            vec![normalized.as_str()]
                        };
                        for word in words {
                            self.tokenize_word(word, &mut ids);
                            word_ids.resize(ids.len(), Some(input_word.unwrap_or(next_word)));
                            next_word += 1;
                        }
                    }
                }
            }
        }

        if add_special_tokens && let Some(sep) = self.sep {
            ids.push(sep);
            word_ids.push(None);
        }

        let attention_mask = vec![1; ids.len()];
        Encoding {
            ids,
            attention_mask,
            word_ids,
        }
    }

    /// BertNormalizer: `clean_text` drops NUL / U+FFFD / control chars (any
    /// Unicode "Other" — Cc, Cf, Co — except \t \n \r) and maps all whitespace
    /// to ' '; `handle_chinese_chars` puts spaces around CJK ideographs.
    fn normalize(&self, text: &str) -> String {
        let (clean_text, handle_chinese_chars) = match &self.normalizer {
            Normalizer::None => return text.to_string(),
            Normalizer::Bert {
                clean_text,
                handle_chinese_chars,
            } => (*clean_text, *handle_chinese_chars),
        };

        let mut out = String::with_capacity(text.len());
        for c in text.chars() {
            if clean_text {
                if c == '\0' || c == '\u{FFFD}' || is_control(c) {
                    continue;
                }
                out.push(if c.is_whitespace() { ' ' } else { c });
            } else {
                out.push(c);
            }
        }
        if handle_chinese_chars {
            let mut spaced = String::with_capacity(out.len());
            for c in out.chars() {
                if is_chinese_char(c) {
                    spaced.push(' ');
                    spaced.push(c);
                    spaced.push(' ');
                } else {
                    spaced.push(c);
                }
            }
            out = spaced;
        }
        out
    }

    /// Greedy longest-match-first WordPiece over one pre-tokenized word,
    /// appending token ids to `out`. A word longer than
    /// `max_input_chars_per_word` chars, or one whose chars cannot all be
    /// covered by vocab entries, collapses to a single `[UNK]`.
    fn tokenize_word(&self, word: &str, out: &mut Vec<u32>) {
        if word.chars().count() > self.max_input_chars_per_word {
            out.push(self.unk_id);
            return;
        }

        let mut start = 0;
        let mut is_bad = false;
        // Checkpoint so a failed word rolls back only its own subtokens, not
        // tokens emitted for earlier words (upstream returns a per-word Vec).
        let checkpoint = out.len();
        while start < word.len() {
            let mut end = word.len();
            let mut found = None;
            while start < end {
                let substr = &word[start..end];
                let id = if start > 0 {
                    // Only continuation probes allocate; words are short
                    // (<= 100 chars) so this stays cheap.
                    let mut candidate =
                        String::with_capacity(self.continuing_subword_prefix.len() + substr.len());
                    candidate.push_str(&self.continuing_subword_prefix);
                    candidate.push_str(substr);
                    self.vocab.get(candidate.as_str()).copied()
                } else {
                    self.vocab.get(substr).copied()
                };
                if let Some(id) = id {
                    found = Some(id);
                    break;
                }
                end -= substr.chars().last().map_or(1, |c| c.len_utf8());
            }
            let Some(id) = found else {
                is_bad = true;
                break;
            };
            out.push(id);
            start = end;
        }

        if is_bad {
            // Any uncovered char dooms the whole word to a single [UNK].
            out.truncate(checkpoint);
            out.push(self.unk_id);
        }
    }

    /// Leftmost-longest scan for added tokens, splitting the input into text
    /// spans (to be normalized + tokenized) and added-token spans (emitted
    /// as-is). Mirrors `AddedVocabulary::find_matches` upstream.
    fn extract_added<'a>(&self, text: &'a str) -> Vec<Span<'a>> {
        let mut spans = Vec::new();
        if self.added_tokens.is_empty() {
            if !text.is_empty() {
                spans.push(Span::Text(text));
            }
            return spans;
        }

        let mut span_start = 0;
        let mut pos = 0;
        while pos < text.len() {
            // Longest match at this position (patterns are plain literals).
            let mut best: Option<usize> = None;
            for (i, token) in self.added_tokens.iter().enumerate() {
                if text[pos..].starts_with(token.content.as_str()) {
                    match best {
                        Some(b) if self.added_tokens[b].content.len() >= token.content.len() => {}
                        _ => best = Some(i),
                    }
                }
            }
            let Some(idx) = best else {
                pos += text[pos..].chars().next().map_or(1, |c| c.len_utf8());
                continue;
            };

            let token = &self.added_tokens[idx];
            let mut start = pos;
            let mut stop = pos + token.content.len();
            if token.lstrip {
                while start > span_start {
                    let prev = text[..start].chars().next_back();
                    match prev {
                        Some(c) if c.is_whitespace() => start -= c.len_utf8(),
                        _ => break,
                    }
                }
            }
            if token.rstrip {
                while let Some(c) = text[stop..].chars().next() {
                    if !c.is_whitespace() {
                        break;
                    }
                    stop += c.len_utf8();
                }
            }

            if start > span_start {
                spans.push(Span::Text(&text[span_start..start]));
            }
            spans.push(Span::Added(idx));
            pos = stop;
            span_start = stop;
        }
        if span_start < text.len() {
            spans.push(Span::Text(&text[span_start..]));
        }
        spans
    }
}

/// One piece of the input after added-token extraction.
enum Span<'a> {
    /// Raw text still to be normalized + tokenized.
    Text(&'a str),
    /// Index into `Tokenizer::added_tokens`.
    Added(usize),
}

/// BertPreTokenizer: split on whitespace (removed), then isolate every
/// punctuation char (ASCII punctuation or Unicode P* category) into its own
/// word. Borrows from `text`; never yields empty words.
fn bert_pre_tokenize(text: &str) -> Vec<&str> {
    let mut words = Vec::new();
    let mut word_start: Option<usize> = None;
    for (i, c) in text.char_indices() {
        if c.is_whitespace() {
            if let Some(s) = word_start.take() {
                words.push(&text[s..i]);
            }
        } else if is_bert_punc(c) {
            if let Some(s) = word_start.take() {
                words.push(&text[s..i]);
            }
            words.push(&text[i..i + c.len_utf8()]);
        } else if word_start.is_none() {
            word_start = Some(i);
        }
    }
    if let Some(s) = word_start {
        words.push(&text[s..]);
    }
    words
}

mod json;
use json::{TokenizerFile, parse_normalizer, parse_post_processor, parse_pre_tokenizer};

#[cfg(test)]
mod tests;
