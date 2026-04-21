//! BPE tokenizer for GigaAM v3 e2e_rnnt.

use anyhow::{Context, Result};
use std::path::Path;

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

        let blank_id = tokens
            .iter()
            .position(|t| t == "<blk>")
            .unwrap_or(tokens.len() - 1);

        tracing::info!(
            "Loaded vocabulary: {} tokens, blank_id={}",
            tokens.len(),
            blank_id
        );

        Ok(Self { tokens, blank_id })
    }

    pub fn blank_id(&self) -> usize {
        self.blank_id
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
        // Replace ▁ (U+2581) with space, then trim
        text.replace('▁', " ").trim().to_string()
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
}
