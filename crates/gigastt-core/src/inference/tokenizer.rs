//! BPE tokenizer for GigaAM v3 e2e_rnnt.

use anyhow::{Context, Result, ensure};
use std::path::Path;

/// BPE word-boundary marker (U+2581 "▁"). Tokens that begin a new word are
/// prefixed with it; decoding replaces it with a space. Single source of truth
/// so the encoder/decoder split logic and the decode step agree.
pub const WORD_BOUNDARY: char = '\u{2581}';

pub struct Tokenizer {
    tokens: Vec<String>,
    blank_id: usize,
}

impl Tokenizer {
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read vocab file: {}", path.display()))?;

        let mut tokens = Vec::new();

        for line in content.lines() {
            if line.is_empty() {
                continue;
            }
            // Skip lines that are a bare integer (header like "1025\n") — no
            // real vocab entry ever hashes to just decimal digits, so treating
            // such a line as a token would poison the ID space with a ghost
            // entry.
            if line.parse::<usize>().is_ok() {
                continue;
            }
            // Try "token<whitespace>id" format first, fall back to just token
            let token = if let Some(pos) = line.rfind(['\t', ' ']) {
                // Check if what follows is a valid number
                let after = line[pos + 1..].trim();
                if after.parse::<usize>().is_ok() {
                    line[..pos].to_string()
                } else {
                    line.to_string()
                }
            } else {
                line.to_string()
            };
            tokens.push(token);
        }

        ensure!(
            !tokens.is_empty(),
            "Vocabulary file is empty: {}",
            path.display()
        );
        let blank_id = tokens
            .iter()
            .position(|t| t == "<blk>")
            .unwrap_or_else(|| tokens.len().saturating_sub(1));

        tracing::info!(
            "Loaded vocabulary: {} tokens, blank_id={}",
            tokens.len(),
            blank_id
        );

        Ok(Self { tokens, blank_id })
    }

    /// Build a tokenizer directly from an in-memory vocab list, without
    /// touching the filesystem or the 850 MB model. Exposed under the private
    /// `__internals` feature (fuzzing, benchmarking) and in unit tests; it is
    /// not part of the stable public API. The blank token is located the same
    /// way [`Tokenizer::load`] does (`<blk>` if present, else the last index).
    #[cfg(any(test, feature = "__internals"))]
    pub fn from_tokens(tokens: Vec<String>) -> Self {
        let blank_id = tokens
            .iter()
            .position(|t| t == "<blk>")
            .unwrap_or_else(|| tokens.len().saturating_sub(1));
        Self { tokens, blank_id }
    }

    /// Construct a tiny synthetic tokenizer (Cyrillic letters, a couple of
    /// `▁`-prefixed word-boundary tokens, `<unk>`, and `<blk>`) for fuzzing /
    /// benchmarking the decode path with no model on disk. Exposed only under
    /// the private `__internals` feature.
    #[cfg(feature = "__internals")]
    pub fn synthetic() -> Self {
        let mut tokens: Vec<String> = Vec::new();
        for ch in 'а'..='я' {
            tokens.push(ch.to_string());
        }
        tokens.push("\u{2581}привет".to_string());
        tokens.push("\u{2581}мир".to_string());
        tokens.push("\u{2581}".to_string());
        tokens.push("<unk>".to_string());
        tokens.push("<blk>".to_string());
        Self::from_tokens(tokens)
    }

    pub fn blank_id(&self) -> usize {
        self.blank_id
    }

    /// Encode a hotword phrase into the token-id sequence the decoder would
    /// emit for it, using greedy longest-match over the vocabulary.
    ///
    /// Spaces (and a leading position) map to the `▁` word-boundary marker the
    /// same way [`Tokenizer::decode`] reverses it, so the encoding matches what
    /// the RNN-T head produces token-for-token. Works for both heads: the plain
    /// `rnnt` char vocab (single-codepoint tokens) and the `e2e_rnnt` BPE vocab
    /// (multi-codepoint `▁`-prefixed pieces).
    ///
    /// Returns `None` if any codepoint of the phrase can't be matched against
    /// the vocabulary (so an unrepresentable hotword is dropped rather than
    /// silently biasing toward a wrong sub-sequence). Special tokens (`<blk>`,
    /// `<unk>`) are never matched.
    pub(crate) fn encode_phrase(&self, phrase: &str) -> Option<Vec<usize>> {
        let phrase = phrase.trim();
        if phrase.is_empty() {
            return None;
        }
        // Build the marked form: leading word + every space become `▁`, so the
        // first piece of each word carries the boundary marker the head emits.
        let mut marked = String::new();
        marked.push(WORD_BOUNDARY);
        for ch in phrase.chars() {
            if ch.is_whitespace() {
                marked.push(WORD_BOUNDARY);
            } else {
                marked.push(ch);
            }
        }

        let chars: Vec<char> = marked.chars().collect();
        let mut ids = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            // Longest vocab token that matches starting at position `i`.
            let mut best: Option<(usize, usize)> = None; // (token_id, char_len)
            for (id, tok) in self.tokens.iter().enumerate() {
                if tok == "<blk>" || tok == "<unk>" || tok.is_empty() {
                    continue;
                }
                let tok_chars = tok.chars().count();
                if tok_chars == 0 || i + tok_chars > chars.len() {
                    continue;
                }
                if chars[i..i + tok_chars].iter().copied().eq(tok.chars())
                    && best.is_none_or(|(_, len)| tok_chars > len)
                {
                    best = Some((id, tok_chars));
                }
            }
            match best {
                Some((id, len)) => {
                    ids.push(id);
                    i += len;
                }
                // A bare `▁` boundary with no vocab token for it: skip it and
                // keep going (the next word still encodes). Any other unmatched
                // codepoint makes the phrase unrepresentable.
                None if chars[i] == WORD_BOUNDARY => i += 1,
                None => return None,
            }
        }
        if ids.is_empty() { None } else { Some(ids) }
    }

    /// Get raw token text by id (returns empty string for out-of-range or special tokens).
    pub fn token_text(&self, id: usize) -> &str {
        if id >= self.tokens.len() {
            return "";
        }
        let t = &self.tokens[id];
        if t == "<blk>" || t == "<unk>" { "" } else { t }
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    #[allow(dead_code)]
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut text = String::new();
        for &id in ids {
            if id == self.blank_id || id >= self.tokens.len() {
                continue;
            }
            let token = &self.tokens[id];
            if token == "<unk>" {
                continue;
            }
            text.push_str(token);
        }
        // Replace the word-boundary marker with a space, then trim.
        text.replace(WORD_BOUNDARY, " ").trim().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn create_test_vocab(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "{content}").unwrap();
        f
    }

    #[test]
    fn test_load_vocab() {
        let f = create_test_vocab(".\t0\n,\t1\n▁в\t2\n<blk>\t3\n");
        let tok = Tokenizer::load(f.path()).unwrap();
        assert_eq!(tok.vocab_size(), 4);
        assert_eq!(tok.blank_id(), 3);
    }

    #[test]
    fn test_decode_basic() {
        let f = create_test_vocab(".\t0\n,\t1\n▁в\t2\n<blk>\t3\n");
        let tok = Tokenizer::load(f.path()).unwrap();
        assert_eq!(tok.decode(&[2, 0]), "в.");
    }

    #[test]
    fn test_decode_blank_skipped() {
        let f = create_test_vocab("а\t0\nб\t1\n<blk>\t2\n");
        let tok = Tokenizer::load(f.path()).unwrap();
        assert_eq!(tok.decode(&[0, 2, 1]), "аб");
    }

    #[test]
    fn test_decode_unk_skipped() {
        let f = create_test_vocab("<unk>\t0\nа\t1\n<blk>\t2\n");
        let tok = Tokenizer::load(f.path()).unwrap();
        assert_eq!(tok.decode(&[0, 1]), "а");
    }

    #[test]
    fn test_decode_word_boundary() {
        let f = create_test_vocab("▁привет\t0\n▁мир\t1\n<blk>\t2\n");
        let tok = Tokenizer::load(f.path()).unwrap();
        assert_eq!(tok.decode(&[0, 1]), "привет мир");
    }

    #[test]
    fn test_load_vocab_missing_file() {
        let result = Tokenizer::load(std::path::Path::new("/nonexistent/vocab.txt"));
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_empty() {
        let f = create_test_vocab("a\t0\n<blk>\t1\n");
        let tok = Tokenizer::load(f.path()).unwrap();
        assert_eq!(tok.decode(&[]), "");
    }

    #[test]
    fn test_token_text_out_of_range() {
        let f = create_test_vocab("a\t0\n<blk>\t1\n");
        let tok = Tokenizer::load(f.path()).unwrap();
        assert_eq!(tok.token_text(999), "");
    }

    #[test]
    fn test_token_text_unk_is_empty() {
        let f = create_test_vocab("<unk>\t0\na\t1\n<blk>\t2\n");
        let tok = Tokenizer::load(f.path()).unwrap();
        assert_eq!(tok.token_text(0), "");
        assert_eq!(tok.token_text(2), "");
        assert_eq!(tok.token_text(1), "a");
    }

    #[test]
    fn test_load_vocab_with_empty_lines_and_bare_integers() {
        let f = create_test_vocab("\n1025\na\t0\n\nb\t1\n<blk>\t2\n");
        let tok = Tokenizer::load(f.path()).unwrap();
        assert_eq!(tok.vocab_size(), 3);
        assert_eq!(tok.token_text(0), "a");
        assert_eq!(tok.token_text(1), "b");
    }

    #[test]
    fn test_load_vocab_no_whitespace_fallback() {
        let f = create_test_vocab("a\nb\t1\n<blk>\t2\n");
        let tok = Tokenizer::load(f.path()).unwrap();
        assert_eq!(tok.vocab_size(), 3);
        assert_eq!(tok.token_text(0), "a");
    }

    #[test]
    fn test_encode_phrase_char_vocab_longest_match() {
        // Char-style vocab plus a `▁`-prefixed boundary token. "аб" → [▁а? no].
        // Here "▁а", "б", "в" exist: "▁аб" greedily matches "▁а" then "б".
        let f = create_test_vocab("▁а\t0\nб\t1\nв\t2\n▁\t3\n<blk>\t4\n");
        let tok = Tokenizer::load(f.path()).unwrap();
        let ids = tok.encode_phrase("аб").unwrap();
        assert_eq!(ids, vec![0, 1]);
        // Round-trips back to the phrase via decode.
        assert_eq!(tok.decode(&ids), "аб");
    }

    #[test]
    fn test_encode_phrase_bpe_word_boundary() {
        // BPE-style vocab where whole words are single tokens.
        let f = create_test_vocab("▁привет\t0\n▁мир\t1\n<blk>\t2\n");
        let tok = Tokenizer::load(f.path()).unwrap();
        let ids = tok.encode_phrase("привет мир").unwrap();
        assert_eq!(ids, vec![0, 1]);
        assert_eq!(tok.decode(&ids), "привет мир");
    }

    #[test]
    fn test_encode_phrase_prefers_longest_token() {
        // "▁аб" must win over "▁а" + "б" when the longer piece exists.
        let f = create_test_vocab("▁а\t0\nб\t1\n▁аб\t2\n<blk>\t3\n");
        let tok = Tokenizer::load(f.path()).unwrap();
        assert_eq!(tok.encode_phrase("аб").unwrap(), vec![2]);
    }

    #[test]
    fn test_encode_phrase_unrepresentable_returns_none() {
        // 'я' is not in the vocab → the phrase can't be encoded.
        let f = create_test_vocab("▁а\t0\nб\t1\n<blk>\t2\n");
        let tok = Tokenizer::load(f.path()).unwrap();
        assert!(tok.encode_phrase("яб").is_none());
    }

    #[test]
    fn test_encode_phrase_empty_returns_none() {
        let f = create_test_vocab("▁а\t0\n<blk>\t1\n");
        let tok = Tokenizer::load(f.path()).unwrap();
        assert!(tok.encode_phrase("   ").is_none());
        assert!(tok.encode_phrase("").is_none());
    }

    #[test]
    fn test_load_vocab_non_numeric_after_whitespace_fallback() {
        let f = create_test_vocab("hello world abc\n<blk>\t1\n");
        let tok = Tokenizer::load(f.path()).unwrap();
        // "hello world abc" has whitespace but "abc" isn't a valid number, so whole line kept
        assert_eq!(tok.vocab_size(), 2);
        assert_eq!(tok.token_text(0), "hello world abc");
    }
}
