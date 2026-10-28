//! Word alignment at a file-window seam. All times are on the same timeline.

use super::WordInfo;
use std::ops::RangeInclusive;

// Bound work even for a pathological decoder emitting many words at one time.
const MAX_OVERLAP_WORDS: usize = 64;
// Ten encoder frames: text alone must not merge distinct repetitions.
const MAX_START_DRIFT: f64 = 0.4;

fn word_key(word: &str) -> String {
    word.chars()
        .flat_map(char::to_lowercase)
        .filter(|c| c.is_alphanumeric())
        .map(|c| if c == 'ё' { 'е' } else { c })
        .collect()
}

#[derive(Clone, Copy, Default, PartialEq)]
struct Score {
    matches: u16,
    drift: f64,
}

impl Score {
    fn with_match(self, drift: f64) -> Self {
        Self {
            matches: self.matches + 1,
            drift: self.drift + drift,
        }
    }

    fn better_than(self, other: Self) -> bool {
        self.matches > other.matches || (self.matches == other.matches && self.drift < other.drift)
    }
}

/// Find an order-preserving matching word near the midpoint and cut on the
/// same side of its two copies. Maximizing the number of matches before
/// minimizing time drift preserves rapid repetitions; nearest-word matching
/// alone can mistake the second "yes" for the first one in the next window.
fn aligned_cut(
    merged: &[WordInfo],
    next: &[WordInfo],
    overlap: &RangeInclusive<f64>,
) -> Option<(usize, usize)> {
    let seam = (overlap.start() + overlap.end()) / 2.0;
    let old_start = merged.partition_point(|w| w.start < *overlap.start());
    let old_end = merged.partition_point(|w| w.start <= *overlap.end());
    let new_start = next.partition_point(|w| w.start < *overlap.start());
    let new_end = next.partition_point(|w| w.start <= *overlap.end());
    let old = &merged[old_start..old_end];
    let new = &next[new_start..new_end];
    if old.is_empty()
        || new.is_empty()
        || old.len() > MAX_OVERLAP_WORDS
        || new.len() > MAX_OVERLAP_WORDS
    {
        return None;
    }
    let old_keys: Vec<_> = old.iter().map(|w| word_key(&w.word)).collect();
    let new_keys: Vec<_> = new.iter().map(|w| word_key(&w.word)).collect();
    let matches = |i: usize, j: usize| {
        !old_keys[i].is_empty()
            && old_keys[i] == new_keys[j]
            && (old[i].start - new[j].start).abs() <= MAX_START_DRIFT
    };
    let width = new.len() + 1;
    let mut scores = vec![Score::default(); (old.len() + 1) * width];
    for i in 1..=old.len() {
        for j in 1..=new.len() {
            let mut best = scores[(i - 1) * width + j];
            let left = scores[i * width + j - 1];
            if left.better_than(best) {
                best = left;
            }
            if matches(i - 1, j - 1) {
                let paired = scores[(i - 1) * width + j - 1]
                    .with_match((old[i - 1].start - new[j - 1].start).abs());
                if paired.better_than(best) {
                    best = paired;
                }
            }
            scores[i * width + j] = best;
        }
    }

    let (mut i, mut j) = (old.len(), new.len());
    let mut best_cut = None;
    let mut best_distance = f64::INFINITY;
    while i > 0 && j > 0 {
        let score = scores[i * width + j];
        let drift = (old[i - 1].start - new[j - 1].start).abs();
        if matches(i - 1, j - 1) && score == scores[(i - 1) * width + j - 1].with_match(drift) {
            let center = (old[i - 1].start + new[j - 1].start) / 2.0;
            // Keep the copy with more encoder context: old before the
            // midpoint, new after it. Preserve the chosen WordInfo; the caller
            // may reconcile initial case using both aligned hypotheses.
            let distance = (center - seam).abs();
            let preferred = usize::from(center <= seam);
            // If timestamps move past the neighbouring word, the other
            // copy can preserve both sequence identity and chronological
            // order. Never repair ordering by dropping extra words.
            for keep_old in [preferred, 1 - preferred] {
                let a = old_start + i - 1 + keep_old;
                let b = new_start + j - 1 + keep_old;
                let ordered = a == 0 || b == next.len() || merged[a - 1].start <= next[b].start;
                if ordered && distance < best_distance {
                    best_cut = Some((a, b));
                    best_distance = distance;
                    break;
                }
            }
            i -= 1;
            j -= 1;
        } else if score == scores[(i - 1) * width + j] {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    best_cut
}

/// Merge independently decoded file windows using aligned overlap words.
/// Empty next hypotheses preserve the available tail. Without a trustworthy
/// match the midpoint timestamp rule remains the conservative fallback.
/// Only the bounded overlap is aligned; the accumulated transcript is not
/// rescanned or cloned on each window.
#[cfg(test)]
pub(crate) fn stitch_chunk_words(
    merged: Vec<WordInfo>,
    next: Vec<WordInfo>,
    seam_s: f64,
) -> Vec<WordInfo> {
    stitch_chunk_words_retained(merged, next, (seam_s - 1.0)..=(seam_s + 1.0)).0
}

/// Return the exact unchanged prefix length alongside the stitched words.
pub(crate) fn stitch_chunk_words_retained(
    mut merged: Vec<WordInfo>,
    mut next: Vec<WordInfo>,
    overlap: RangeInclusive<f64>,
) -> (Vec<WordInfo>, usize) {
    let seam_s = (overlap.start() + overlap.end()) / 2.0;
    if merged.is_empty() {
        return (next, 0);
    }
    if next.is_empty() {
        let retained = merged.len();
        return (merged, retained);
    }
    let aligned = aligned_cut(&merged, &next, &overlap);
    if let Some((old_end, 0)) = aligned {
        reconcile_seam_case(&merged, old_end, &mut next, &overlap);
    }
    let (old_end, new_start) = aligned.unwrap_or_else(|| {
        (
            merged.partition_point(|w| w.start <= seam_s),
            next.partition_point(|w| w.start <= seam_s),
        )
    });
    merged.truncate(old_end);
    merged.extend(next.into_iter().skip(new_start));
    (merged, old_end)
}

/// Only the first decoded word can carry a window-start capital. Require its
/// aligned older copy to differ solely in that initial case: guessing from the
/// preceding punctuation alone would lowercase proper names. Ambiguous repeated
/// words and fallback cuts retain the selected hypothesis unchanged.
fn reconcile_seam_case(
    old: &[WordInfo],
    boundary: usize,
    next: &mut [WordInfo],
    overlap: &RangeInclusive<f64>,
) {
    let Some(first) = next.first() else {
        return;
    };
    if boundary == 0 || boundary >= old.len() || ends_sentence(&old[boundary - 1].word) {
        return;
    }
    let Some((idx, ch)) = first.word.char_indices().find(|(_, c)| c.is_alphabetic()) else {
        return;
    };
    if !matches!(ch, 'А'..='Я' | 'Ё') {
        return;
    }
    let mut letters = first.word.chars().filter(|c| c.is_alphabetic());
    letters.next();
    // Preserve multi-letter acronyms, including ones followed by punctuation.
    if letters.clone().next().is_some() && letters.all(char::is_uppercase) {
        return;
    }
    let mut lower = first.word.clone();
    lower.replace_range(idx..idx + ch.len_utf8(), &ch.to_lowercase().to_string());
    if old[boundary].word != lower {
        return;
    }

    // Require a unique correspondence in both windows. An insertion or deletion
    // among rapid repetitions can otherwise lend case from another occurrence.
    // aligned_cut already bounds both overlap slices to MAX_OVERLAP_WORDS.
    let key = word_key(&first.word);
    if unique_overlap_match(old, first, &key, overlap)
        && unique_overlap_match(next, &old[boundary], &key, overlap)
    {
        next[0].word = lower;
    }
}

fn unique_overlap_match(
    words: &[WordInfo],
    anchor: &WordInfo,
    key: &str,
    overlap: &RangeInclusive<f64>,
) -> bool {
    let start = words.partition_point(|w| w.start < *overlap.start());
    let end = words.partition_point(|w| w.start <= *overlap.end());
    words[start..end]
        .iter()
        .filter(|w| (w.start - anchor.start).abs() <= MAX_START_DRIFT && word_key(&w.word) == key)
        .take(2)
        .count()
        == 1
}

/// Does `word` end a sentence, ignoring closing quotes and brackets?
fn ends_sentence(word: &str) -> bool {
    word.trim_end_matches(['"', '\'', '»', ')', ']', '”', '’', ' '])
        .chars()
        .last()
        .is_some_and(|c| matches!(c, '.' | '!' | '?' | '…'))
}
