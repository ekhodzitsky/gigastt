//! Default Russian brand / acronym hotword lexicon.
//!
//! A small curated list of brands, media/streaming services, and acronyms that
//! a Russian STT model commonly mis-recognizes (rare or out-of-distribution
//! against conversational training data). It seeds the default hotword pack:
//! passing [`DEFAULT_HOTWORDS`] to
//! [`Engine::with_hotwords`](crate::inference::Engine::with_hotwords) biases the
//! greedy decode toward these spellings.
//!
//! Stored as a plain `&[&str]` rather than a `phf` compile-time map: the list is
//! consumed exactly once (iterated to tokenize each phrase at biaser-build time)
//! and never key-looked-up, so a hash map would add a build dependency for no
//! runtime benefit. The Cyrillic forms mirror the transliteration targets used
//! by the benchmark's anglicism normalization (`benchmark/common.py`), so this
//! list doubles as the en→ru source for a future Transliterated-WER metric.

/// Curated Russian brand / acronym hotwords, in their canonical Cyrillic
/// spelling. Default-OFF: only applied when a caller opts into the default pack.
pub const DEFAULT_HOTWORDS: &[&str] = &[
    // Device / service brands.
    "эпл",
    "айфон",
    "самсунг",
    "сони",
    "гугл",
    "яндекс",
    "сбер",
    "алиса",
    "маруся",
    // Social / media / streaming.
    "ютуб",
    "фейсбук",
    "инстаграм",
    "телеграм",
    "вконтакте",
    "нетфликс",
    "спотифай",
    "ватсап",
    "тикток",
    "твич",
    "кинопоиск",
    "окко",
    "иви",
    "смотрешка",
    // Marketplaces.
    "алиэкспресс",
    "озон",
    "вайлдберриз",
    // Acronyms / short brands.
    "вк",
    "тв",
    "синергия",
    "пинк",
];

/// Build the default hotword pack as `(phrase, weight)` pairs with unit weight,
/// ready to pass to
/// [`Engine::with_hotwords`](crate::inference::Engine::with_hotwords).
pub fn default_hotword_pairs() -> Vec<(String, f32)> {
    DEFAULT_HOTWORDS
        .iter()
        .map(|w| ((*w).to_string(), 1.0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_hotwords_non_empty_and_unique() {
        assert!(!DEFAULT_HOTWORDS.is_empty());
        let mut seen = std::collections::HashSet::new();
        for w in DEFAULT_HOTWORDS {
            assert!(seen.insert(*w), "duplicate hotword in lexicon: {w}");
            assert!(!w.trim().is_empty(), "empty hotword in lexicon");
        }
    }

    #[test]
    fn test_default_hotword_pairs_unit_weight() {
        let pairs = default_hotword_pairs();
        assert_eq!(pairs.len(), DEFAULT_HOTWORDS.len());
        assert!(pairs.iter().all(|(_, w)| *w == 1.0));
    }
}
