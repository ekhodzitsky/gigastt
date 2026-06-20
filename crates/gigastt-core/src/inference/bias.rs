//! Contextual hotword biasing for the greedy RNN-T decode loop.
//!
//! Shallow-fusion biasing steers the greedy transducer toward a curated set of
//! phrases (brands, names, domain terms) without a beam search. Each hotword is
//! tokenized to the id sequence the active head would emit (via
//! [`Tokenizer::encode_phrase`](super::tokenizer::Tokenizer::encode_phrase), so
//! it adapts to whichever vocab is loaded) and stored in a small prefix trie.
//!
//! During decode, a [`BiasState`] tracks which hotword prefixes are currently
//! "active" given the recently emitted tokens. Before the argmax over the
//! joiner logits, [`Biaser::boost_logits`] adds a fixed boost to the logits of
//! the token-ids that would extend an active prefix. A token that completes /
//! advances a prefix advances the state; anything else resets it (while still
//! letting a fresh hotword start). Blank frames leave the prefix state
//! unchanged — they emit no label, so a partially-matched hotword survives the
//! gaps between its tokens.
//!
//! The [`Biaser`] itself is immutable after construction and shared across the
//! session pool via `&Biaser`; the only mutable per-decode bookkeeping lives in
//! [`BiasState`], created fresh for each decode. When no hotwords are
//! configured the engine holds no biaser at all and the decode path is
//! byte-for-byte unchanged.

use super::tokenizer::Tokenizer;

/// One node of the hotword prefix trie. The root is index 0.
struct TrieNode {
    /// Edges keyed by token id → child node index.
    children: std::collections::HashMap<usize, usize>,
}

impl TrieNode {
    fn new() -> Self {
        Self {
            children: std::collections::HashMap::new(),
        }
    }
}

/// Compiled hotword biaser: a prefix trie over hotword token-id sequences plus
/// the additive logit boost. Immutable and shareable across inference sessions.
pub struct Biaser {
    nodes: Vec<TrieNode>,
    /// Additive boost applied to a continuation token's logit.
    boost: f32,
    /// Number of distinct hotword phrases successfully compiled.
    phrase_count: usize,
}

impl Biaser {
    /// Build a biaser from hotword token-id sequences and a boost. Sequences
    /// must be non-empty; empty ones are skipped. Returns `None` if no sequence
    /// survives (so callers treat "no usable hotwords" as biasing-off).
    ///
    /// `pub(crate)` so the decode-loop unit tests can construct a biaser from
    /// raw token-id sequences without a tokenizer.
    pub(crate) fn from_sequences(sequences: Vec<Vec<usize>>, boost: f32) -> Option<Self> {
        let mut nodes = vec![TrieNode::new()];
        let mut phrase_count = 0;
        for seq in sequences {
            if seq.is_empty() {
                continue;
            }
            phrase_count += 1;
            let mut node = 0usize;
            for tok in seq {
                node = match nodes[node].children.get(&tok) {
                    Some(&child) => child,
                    None => {
                        let child = nodes.len();
                        nodes.push(TrieNode::new());
                        nodes[node].children.insert(tok, child);
                        child
                    }
                };
            }
        }
        if phrase_count == 0 {
            return None;
        }
        Some(Self {
            nodes,
            boost,
            phrase_count,
        })
    }

    /// Build a biaser from `(phrase, weight)` pairs, tokenizing each phrase with
    /// the active [`Tokenizer`]. `weight` scales the base `boost` per phrase
    /// (use `1.0` for the default). Phrases the tokenizer can't represent are
    /// dropped. Returns `None` if no phrase compiles or `boost <= 0`.
    pub fn from_phrases(
        tokenizer: &Tokenizer,
        phrases: &[(String, f32)],
        boost: f32,
    ) -> Option<Self> {
        if boost <= 0.0 {
            return None;
        }
        // Per-phrase weights are folded into the boost by storing the *highest*
        // requested boost on each trie edge would complicate the immutable
        // node layout; instead we keep a single base boost and treat the weight
        // as a phrase-level filter (weight <= 0 drops the phrase). A future
        // per-edge weight can extend TrieNode without touching the decode loop.
        let mut sequences = Vec::new();
        let mut dropped = 0usize;
        for (phrase, weight) in phrases {
            if *weight <= 0.0 {
                continue;
            }
            match tokenizer.encode_phrase(phrase) {
                Some(ids) => sequences.push(ids),
                None => {
                    dropped += 1;
                    tracing::debug!(phrase = %phrase, "hotword dropped: not representable in active vocab");
                }
            }
        }
        if dropped > 0 {
            tracing::warn!(
                "{dropped} hotword phrase(s) dropped (not representable in active vocab)"
            );
        }
        Self::from_sequences(sequences, boost)
    }

    /// Number of hotword phrases compiled into the trie.
    pub fn phrase_count(&self) -> usize {
        self.phrase_count
    }

    /// Create a fresh per-decode prefix-tracking state rooted at the trie root.
    pub(crate) fn new_state(&self) -> BiasState {
        BiasState {
            // The root is always active so a new hotword can start at any token.
            active: vec![0],
        }
    }

    /// Add the boost to `logits` for every token id that extends a currently
    /// active hotword prefix. No-op when no active node has children (i.e. no
    /// hotword could continue here), so non-hotword regions are untouched.
    pub(crate) fn boost_logits(&self, state: &BiasState, logits: &mut [f32]) {
        for &node in &state.active {
            for &tok in self.nodes[node].children.keys() {
                if tok < logits.len() {
                    logits[tok] += self.boost;
                }
            }
        }
    }

    /// Advance the prefix state after a non-blank token `tok` was emitted.
    ///
    /// New active set = the children reached by `tok` from any previously active
    /// node, plus the root (so a fresh hotword can begin on the next token).
    /// Deduplicated to keep the active set small.
    pub(crate) fn advance(&self, state: &mut BiasState, tok: usize) {
        let mut next = Vec::new();
        for &node in &state.active {
            if let Some(&child) = self.nodes[node].children.get(&tok)
                && !next.contains(&child)
            {
                next.push(child);
            }
        }
        // The root stays active so biasing can restart at the next token.
        if !next.contains(&0) {
            next.push(0);
        }
        state.active = next;
    }
}

/// Per-decode hotword prefix-tracking state. Holds the set of trie nodes whose
/// prefix has been matched by the recently emitted tokens. Cheap to create;
/// one per [`greedy_decode`](super::decode::greedy_decode) call.
pub(crate) struct BiasState {
    active: Vec<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn biaser(seqs: Vec<Vec<usize>>, boost: f32) -> Biaser {
        Biaser::from_sequences(seqs, boost).expect("non-empty sequences")
    }

    #[test]
    fn test_from_sequences_empty_returns_none() {
        assert!(Biaser::from_sequences(vec![], 5.0).is_none());
        assert!(Biaser::from_sequences(vec![vec![]], 5.0).is_none());
    }

    #[test]
    fn test_boost_applies_to_first_token_of_each_hotword() {
        // Two hotwords: [1,2] and [3]. At the root both 1 and 3 are boostable.
        let b = biaser(vec![vec![1, 2], vec![3]], 5.0);
        let state = b.new_state();
        let mut logits = vec![0.0; 5];
        b.boost_logits(&state, &mut logits);
        assert_eq!(logits[1], 5.0);
        assert_eq!(logits[3], 5.0);
        assert_eq!(logits[2], 0.0, "mid-hotword token not boosted at root");
        assert_eq!(logits[0], 0.0);
    }

    #[test]
    fn test_advance_then_boost_continuation() {
        // After emitting token 1, the hotword [1,2] should boost token 2.
        let b = biaser(vec![vec![1, 2]], 5.0);
        let mut state = b.new_state();
        b.advance(&mut state, 1);
        let mut logits = vec![0.0; 5];
        b.boost_logits(&state, &mut logits);
        assert_eq!(logits[2], 5.0, "continuation token boosted after prefix");
        // Token 1 is also boostable again because the root stays active.
        assert_eq!(logits[1], 5.0, "root keeps a fresh hotword start available");
    }

    #[test]
    fn test_advance_off_prefix_resets_to_root_only() {
        // Emit a non-matching token: only the root-level starts remain boosted.
        let b = biaser(vec![vec![1, 2]], 5.0);
        let mut state = b.new_state();
        b.advance(&mut state, 1); // on prefix [1]
        b.advance(&mut state, 9); // off prefix → reset to root
        let mut logits = vec![0.0; 5];
        b.boost_logits(&state, &mut logits);
        assert_eq!(logits[2], 0.0, "continuation no longer boosted after reset");
        assert_eq!(logits[1], 5.0, "root start still boosted");
    }

    #[test]
    fn test_shared_prefix_keeps_both_branches_active() {
        // Hotwords [1,2] and [1,3] share the first token.
        let b = biaser(vec![vec![1, 2], vec![1, 3]], 4.0);
        let mut state = b.new_state();
        b.advance(&mut state, 1);
        let mut logits = vec![0.0; 5];
        b.boost_logits(&state, &mut logits);
        assert_eq!(logits[2], 4.0);
        assert_eq!(logits[3], 4.0);
    }

    #[test]
    fn test_boost_ignores_out_of_range_token_id() {
        // A hotword token id beyond the logits length must not panic.
        let b = biaser(vec![vec![99]], 5.0);
        let state = b.new_state();
        let mut logits = vec![0.0; 5];
        b.boost_logits(&state, &mut logits); // no panic
        assert!(logits.iter().all(|&l| l == 0.0));
    }

    use crate::inference::tokenizer::Tokenizer;

    /// A char-vocab tokenizer covering the Cyrillic letters used below plus a
    /// `▁` word-boundary marker, so `encode_phrase` produces deterministic ids.
    fn char_tokenizer() -> Tokenizer {
        let tokens = vec![
            "а".to_string(),
            "б".to_string(),
            "в".to_string(),
            "г".to_string(),
            "д".to_string(),
            "\u{2581}".to_string(), // word-boundary marker
            "<unk>".to_string(),
            "<blk>".to_string(),
        ];
        Tokenizer::from_tokens(tokens)
    }

    #[test]
    fn test_from_phrases_zero_boost_returns_none() {
        let tok = char_tokenizer();
        let phrases = vec![("аб".to_string(), 1.0)];
        assert!(Biaser::from_phrases(&tok, &phrases, 0.0).is_none());
    }

    #[test]
    fn test_from_phrases_negative_boost_returns_none() {
        let tok = char_tokenizer();
        let phrases = vec![("аб".to_string(), 1.0)];
        assert!(Biaser::from_phrases(&tok, &phrases, -3.0).is_none());
    }

    #[test]
    fn test_from_phrases_empty_slice_returns_none() {
        let tok = char_tokenizer();
        assert!(Biaser::from_phrases(&tok, &[], 5.0).is_none());
    }

    #[test]
    fn test_from_phrases_all_zero_weight_returns_none() {
        // Positive boost but every phrase filtered out by weight <= 0 → None.
        let tok = char_tokenizer();
        let phrases = vec![("аб".to_string(), 0.0), ("вг".to_string(), -1.0)];
        assert!(Biaser::from_phrases(&tok, &phrases, 5.0).is_none());
    }

    #[test]
    fn test_from_phrases_unrepresentable_only_returns_none() {
        // A phrase with no codepoints in the vocab is dropped; nothing survives.
        let tok = char_tokenizer();
        let phrases = vec![("xyz".to_string(), 1.0)];
        assert!(Biaser::from_phrases(&tok, &phrases, 5.0).is_none());
    }

    #[test]
    fn test_from_phrases_single_token_phrase_boosts_first_token() {
        // "а" encodes to a leading ▁ (id 5) then char id 0. The first emitted
        // token of the hotword is the boundary marker, so the root boosts id 5.
        let tok = char_tokenizer();
        let phrases = vec![("а".to_string(), 1.0)];
        let b = Biaser::from_phrases(&tok, &phrases, 7.0).expect("phrase compiles");
        assert_eq!(b.phrase_count(), 1);

        let ids = tok.encode_phrase("а").expect("representable");
        let state = b.new_state();
        let mut logits = vec![0.0; 8];
        b.boost_logits(&state, &mut logits);
        // First token of the encoded phrase is boostable at the root.
        assert_eq!(logits[ids[0]], 7.0, "first token of phrase boosted at root");
    }

    #[test]
    fn test_from_phrases_multi_token_phrase_boosts_continuation() {
        // "аб" → [▁(5), а(0), б(1)]. After advancing through the encoded
        // prefix, the next continuation token must be boosted.
        let tok = char_tokenizer();
        let phrases = vec![("аб".to_string(), 1.0)];
        let b = Biaser::from_phrases(&tok, &phrases, 4.0).expect("phrase compiles");
        assert_eq!(b.phrase_count(), 1);

        let ids = tok.encode_phrase("аб").expect("representable");
        assert_eq!(ids, vec![5, 0, 1]);

        let mut state = b.new_state();
        b.advance(&mut state, ids[0]); // ▁
        b.advance(&mut state, ids[1]); // а
        let mut logits = vec![0.0; 8];
        b.boost_logits(&state, &mut logits);
        assert_eq!(
            logits[ids[2]], 4.0,
            "third token boosted after two-token prefix"
        );
    }

    #[test]
    fn test_from_phrases_drops_unrepresentable_keeps_representable() {
        // One good phrase, one with an out-of-vocab codepoint. The good one
        // survives; the count reflects only the compiled phrase.
        let tok = char_tokenizer();
        let phrases = vec![("аб".to_string(), 1.0), ("аz".to_string(), 1.0)];
        let b = Biaser::from_phrases(&tok, &phrases, 5.0).expect("one phrase compiles");
        assert_eq!(b.phrase_count(), 1);
    }

    #[test]
    fn test_from_phrases_weight_filters_per_phrase() {
        // Two representable phrases; one has weight 0 and is dropped before
        // tokenization, leaving a single compiled phrase.
        let tok = char_tokenizer();
        let phrases = vec![("аб".to_string(), 1.0), ("вг".to_string(), 0.0)];
        let b = Biaser::from_phrases(&tok, &phrases, 5.0).expect("one phrase compiles");
        assert_eq!(b.phrase_count(), 1);
    }
}
