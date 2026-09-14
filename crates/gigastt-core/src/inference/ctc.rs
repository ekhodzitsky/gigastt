//! CTC greedy decoding for the GigaAM Multilingual CTC head.
//!
//! Unlike the RNN-T path (encoder + prediction network + joiner), the CTC head is
//! a single encoder that emits per-frame class log-probabilities. Greedy CTC
//! decode = per-frame argmax over the vocab, collapse consecutive repeats (a blank
//! between two identical labels breaks the run and keeps both), then drop blanks.
//!
//! `log_probs` is the encoder output `[1, T', V]` read row-major, so frame `t`'s
//! logits are `log_probs[t*V .. (t+1)*V]` — **frame-major**: `T'` is the outer
//! axis, the vocab the inner. This differs from the RNN-T encoder's channels-first
//! `[1, D, T]` layout (see [`super::decode::extract_encoder_frame`]).

use super::bias::{BiasPath, Biaser};
use super::decode::{TokenInfo, argmax_with_confidence};
use super::tokenizer::{Tokenizer, WORD_BOUNDARY};
use super::{SECONDS_PER_FRAME, WordInfo};

/// Hypotheses kept per frame by [`ctc_prefix_beam_decode_with_abort`].
///
/// Small on purpose: the vocabulary is 71 classes and biasing needs breadth
/// only where a hotword competes with what the model heard, not everywhere.
const BEAM_WIDTH: usize = 8;

/// Acoustic candidates considered per frame before hotword continuations are
/// added. Everything below this rank is too unlikely to survive pruning anyway.
const BEAM_TOP_K: usize = 6;

/// `ln(exp(a) + exp(b))`, stable for the log-domain accumulation below.
fn log_add_exp(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let (hi, lo) = if a > b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp().ln_1p()
}

/// One prefix under consideration, with the two probabilities CTC prefix search
/// has to keep apart: paths that end in a blank and paths that end in the
/// prefix's own last label. Collapsing them would lose the distinction between
/// "aa" written as one label and as two.
struct Hypothesis {
    /// Log-probability of paths reaching this prefix and ending in blank.
    p_blank: f32,
    /// Log-probability of paths reaching this prefix and ending in its last label.
    p_nonblank: f32,
    /// Emitted labels with the frame and confidence that produced them; word
    /// timestamps downstream are built from these, so a prefix has to carry its
    /// alignment rather than just its text.
    tokens: Vec<TokenInfo>,
    /// Where this hypothesis sits in the hotword trie.
    bias: BiasPath,
    /// Probability of the contribution that supplied `tokens` and `bias`. When
    /// two extensions land on the same prefix, the better-supported alignment
    /// is the one worth keeping.
    anchor: f32,
}

impl Hypothesis {
    fn total(&self) -> f32 {
        log_add_exp(self.p_blank, self.p_nonblank)
    }

    /// Score with any un-earned hotword boost taken back.
    ///
    /// Pruning uses [`Self::total`], boost included — that credit is what keeps
    /// a half-matched phrase alive long enough to finish. Choosing the winner
    /// uses this instead: a hypothesis still inside a phrase when the audio ran
    /// out never finished it, and paying it for the attempt is how a glossary
    /// starts inventing words that were not said.
    fn settled_total(&self) -> f32 {
        self.total() - self.bias.pending()
    }
}

/// Contextual CTC decoding by prefix beam search.
///
/// Greedy CTC takes a per-frame argmax and has no continuation state, so there
/// is nothing for a hotword boost to steer — biasing was inert on these heads.
/// A beam keeps competing prefixes alive, which gives a boost something to act
/// on and, just as importantly, gives a wrong guess somewhere to lose: a
/// hypothesis that walks into a hotword and abandons it is refunded the boost
/// it was granted (see [`Biaser::score_token`]), so only a phrase the audio
/// actually supports keeps its advantage.
///
/// Arguments match [`ctc_greedy_decode_with_abort`]; `log_probs` is likewise the raw
/// encoder output, normalized per frame here.
#[cfg(test)]
pub(crate) fn ctc_prefix_beam_decode(
    log_probs: &[f32],
    t_len: usize,
    vocab: usize,
    blank_id: usize,
    biaser: &Biaser,
) -> Vec<TokenInfo> {
    ctc_prefix_beam_decode_with_abort(log_probs, t_len, vocab, blank_id, biaser, None)
}

pub(crate) fn ctc_prefix_beam_decode_with_abort(
    log_probs: &[f32],
    t_len: usize,
    vocab: usize,
    blank_id: usize,
    biaser: &Biaser,
    abort: Option<&(dyn Fn() -> bool + Sync)>,
) -> Vec<TokenInfo> {
    if vocab == 0 {
        return Vec::new();
    }
    let usable = t_len.min(log_probs.len() / vocab);

    // The empty prefix, reached by a blank-only path with probability 1.
    let mut beams: Vec<(Vec<usize>, Hypothesis)> = vec![(
        Vec::new(),
        Hypothesis {
            p_blank: 0.0,
            p_nonblank: f32::NEG_INFINITY,
            tokens: Vec::new(),
            bias: BiasPath::default(),
            anchor: 0.0,
        },
    )];

    let mut lp = vec![0.0_f32; vocab];
    let mut candidates: Vec<usize> = Vec::new();

    for t in 0..usable {
        if abort.is_some_and(|abort| abort()) {
            break;
        }
        log_softmax(&log_probs[t * vocab..(t + 1) * vocab], &mut lp);

        // Candidates: the most likely classes this frame, plus every token that
        // could extend a hotword any live beam is inside. The second half is
        // what lets a boosted continuation be considered at all when the model
        // ranks it below the cut.
        candidates.clear();
        top_k_into(&lp, BEAM_TOP_K, &mut candidates);
        if !candidates.contains(&blank_id) {
            candidates.push(blank_id);
        }
        for (_, hyp) in &beams {
            biaser.continuations(hyp.bias, &mut candidates);
        }
        // A hotword compiled against a different vocabulary can name a token id
        // this head does not have; drop those rather than index past the frame.
        candidates.retain(|&c| c < vocab);
        candidates.sort_unstable();
        candidates.dedup();

        let mut next: Vec<(Vec<usize>, Hypothesis)> = Vec::new();
        for (labels, hyp) in &beams {
            for &c in &candidates {
                if lp[c] == f32::NEG_INFINITY {
                    continue;
                }
                if c == blank_id {
                    // No label emitted: the prefix is unchanged and the path
                    // now ends in blank.
                    let p = hyp.total() + lp[c];
                    merge(&mut next, labels, hyp, Emission::Blank(p));
                    continue;
                }
                if labels.last() == Some(&c) {
                    // Repeat of the last label. Without an intervening blank it
                    // collapses into the same prefix; with one it emits a
                    // second copy.
                    let same = hyp.p_nonblank + lp[c];
                    merge(&mut next, labels, hyp, Emission::Repeat(same));
                    let (bonus, bias) = biaser.score_token(hyp.bias, c);
                    let extended = hyp.p_blank + lp[c] + bonus;
                    merge_extension(&mut next, labels, hyp, c, extended, bias, t, lp[c].exp());
                    continue;
                }
                let (bonus, bias) = biaser.score_token(hyp.bias, c);
                let extended = hyp.total() + lp[c] + bonus;
                merge_extension(&mut next, labels, hyp, c, extended, bias, t, lp[c].exp());
            }
        }

        if next.is_empty() {
            break;
        }
        next.sort_by(|a, b| b.1.total().total_cmp(&a.1.total()));
        next.truncate(BEAM_WIDTH);
        beams = next;
    }

    beams
        .into_iter()
        .max_by(|a, b| a.1.settled_total().total_cmp(&b.1.settled_total()))
        .map(|(_, hyp)| hyp.tokens)
        .unwrap_or_default()
}

/// A contribution to a prefix that emits no new label.
enum Emission {
    /// Path ends in blank.
    Blank(f32),
    /// Path ends in the prefix's own last label.
    Repeat(f32),
}

/// Fold a same-prefix contribution into `next`.
fn merge(
    next: &mut Vec<(Vec<usize>, Hypothesis)>,
    labels: &[usize],
    from: &Hypothesis,
    emission: Emission,
) {
    let slot = match next.iter_mut().find(|(l, _)| l == labels) {
        Some((_, hyp)) => hyp,
        None => {
            next.push((
                labels.to_vec(),
                Hypothesis {
                    p_blank: f32::NEG_INFINITY,
                    p_nonblank: f32::NEG_INFINITY,
                    tokens: from.tokens.clone(),
                    bias: from.bias,
                    anchor: from.anchor,
                },
            ));
            &mut next.last_mut().expect("just pushed").1
        }
    };
    match emission {
        Emission::Blank(p) => slot.p_blank = log_add_exp(slot.p_blank, p),
        Emission::Repeat(p) => slot.p_nonblank = log_add_exp(slot.p_nonblank, p),
    }
}

/// Fold an extension by `token` into `next`, keeping the alignment of whichever
/// contribution is best supported.
#[allow(clippy::too_many_arguments)]
fn merge_extension(
    next: &mut Vec<(Vec<usize>, Hypothesis)>,
    labels: &[usize],
    from: &Hypothesis,
    token: usize,
    p: f32,
    bias: BiasPath,
    frame: usize,
    confidence: f32,
) {
    if p == f32::NEG_INFINITY {
        return;
    }
    let mut extended = Vec::with_capacity(labels.len() + 1);
    extended.extend_from_slice(labels);
    extended.push(token);

    let build = |from: &Hypothesis| {
        let mut tokens = Vec::with_capacity(from.tokens.len() + 1);
        tokens.extend_from_slice(&from.tokens);
        tokens.push(TokenInfo {
            token_id: token,
            frame_index: frame,
            confidence,
        });
        tokens
    };

    match next.iter_mut().find(|(l, _)| *l == extended) {
        Some((_, hyp)) => {
            hyp.p_nonblank = log_add_exp(hyp.p_nonblank, p);
            if p > hyp.anchor {
                hyp.tokens = build(from);
                hyp.bias = bias;
                hyp.anchor = p;
            }
        }
        None => next.push((
            extended,
            Hypothesis {
                p_blank: f32::NEG_INFINITY,
                p_nonblank: p,
                tokens: build(from),
                bias,
                anchor: p,
            },
        )),
    }
}

/// Normalize one frame of encoder output into log-probabilities.
fn log_softmax(row: &[f32], out: &mut [f32]) {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = row.iter().map(|&l| (l - max).exp()).sum();
    let log_sum = max + sum.ln();
    for (o, &l) in out.iter_mut().zip(row) {
        *o = l - log_sum;
    }
}

/// Indices of the `k` largest values in `lp`, appended to `out`.
fn top_k_into(lp: &[f32], k: usize, out: &mut Vec<usize>) {
    let mut idx: Vec<usize> = (0..lp.len()).collect();
    let k = k.min(idx.len());
    idx.select_nth_unstable_by(k.saturating_sub(1), |&a, &b| lp[b].total_cmp(&lp[a]));
    out.extend_from_slice(&idx[..k]);
}

/// Greedy CTC decode over a flat `log_probs` buffer of shape `[t_total, vocab]`
/// (row-major). Returns the collapsed, blank-stripped tokens with a per-token
/// frame index (relative to this window) and softmax confidence — the same
/// [`TokenInfo`] the RNN-T path emits, so downstream word formatting is shared.
///
/// - `t_len`: number of valid frames (from the encoder's `encoded_lengths[0]`);
///   frames past it are right-padding and ignored, even if the tensor's outer dim
///   is larger.
/// - `vocab`: class count (71 for GigaAM Multilingual).
/// - `blank_id`: CTC blank (70 = `vocab - 1`).
#[cfg(test)]
pub(crate) fn ctc_greedy_decode(
    log_probs: &[f32],
    t_len: usize,
    vocab: usize,
    blank_id: usize,
) -> Vec<TokenInfo> {
    ctc_greedy_decode_with_abort(log_probs, t_len, vocab, blank_id, None)
}

pub(crate) fn ctc_greedy_decode_with_abort(
    log_probs: &[f32],
    t_len: usize,
    vocab: usize,
    blank_id: usize,
    abort: Option<&(dyn Fn() -> bool + Sync)>,
) -> Vec<TokenInfo> {
    if vocab == 0 {
        return Vec::new();
    }
    let usable = t_len.min(log_probs.len() / vocab);
    let mut out = Vec::new();
    let mut prev: Option<usize> = None;
    for t in 0..usable {
        if abort.is_some_and(|abort| abort()) {
            break;
        }
        let row = &log_probs[t * vocab..(t + 1) * vocab];
        let (id, confidence) = argmax_with_confidence(row, blank_id);
        // Collapse: skip a frame whose argmax equals the previous frame's argmax
        // (blank or not). Tracking the raw argmax — not the last *emitted* token —
        // is what makes a blank between two identical labels keep both.
        if Some(id) == prev {
            continue;
        }
        prev = Some(id);
        if id == blank_id {
            continue;
        }
        out.push(TokenInfo {
            token_id: id,
            frame_index: t,
            confidence,
        });
    }
    out
}

/// Group CTC-decoded tokens into words with timestamps and confidence.
///
/// istupakov's CTC vocab encodes the inter-word space as the `▁` word-boundary
/// marker at `vocab[0]` (mirroring the RNN-T vocab), NOT a literal `' '`. Words
/// split on that marker; every other token is a single character concatenated
/// into the current word. The blank token never appears here (it is dropped in
/// [`ctc_greedy_decode_with_abort`]).
pub(crate) fn ctc_tokens_to_words(
    tokenizer: &Tokenizer,
    tokens: &[TokenInfo],
    frame_offset: usize,
) -> Vec<WordInfo> {
    let mut words = Vec::new();
    let mut current_word = String::new();
    let mut word_start_frame: Option<usize> = None;
    let mut word_end_frame: usize = 0;
    let mut word_confidences: Vec<f32> = Vec::new();

    let flush = |word: &mut String,
                 start: &mut Option<usize>,
                 end: usize,
                 confs: &mut Vec<f32>,
                 out: &mut Vec<WordInfo>| {
        if word.is_empty() {
            return;
        }
        let avg_conf: f32 = if confs.is_empty() {
            1.0
        } else {
            confs.iter().sum::<f32>() / confs.len() as f32
        };
        out.push(WordInfo {
            word: std::mem::take(word),
            start: (start.unwrap_or(0) + frame_offset) as f64 * SECONDS_PER_FRAME,
            end: (end + frame_offset) as f64 * SECONDS_PER_FRAME,
            confidence: avg_conf,
            speaker: None,
        });
        *start = None;
        confs.clear();
    };

    for token in tokens {
        let ch = tokenizer.token_text(token.token_id);
        if ch.starts_with(WORD_BOUNDARY) {
            flush(
                &mut current_word,
                &mut word_start_frame,
                word_end_frame,
                &mut word_confidences,
                &mut words,
            );
            continue;
        }
        if !ch.is_empty() {
            current_word.push_str(ch);
            if word_start_frame.is_none() {
                word_start_frame = Some(token.frame_index);
            }
            word_end_frame = token.frame_index;
            word_confidences.push(token.confidence);
        }
    }

    flush(
        &mut current_word,
        &mut word_start_frame,
        word_end_frame,
        &mut word_confidences,
        &mut words,
    );

    words
}

#[cfg(test)]
mod tests;
