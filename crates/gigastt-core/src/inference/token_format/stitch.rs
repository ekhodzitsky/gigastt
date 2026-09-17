//! Word alignment at a file-window seam. All times are on the same timeline.

use super::WordInfo;
use crate::inference::windows::CHUNK_OVERLAP_SAMPLES;

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
fn aligned_cut(merged: &[WordInfo], next: &[WordInfo], seam: f64) -> Option<(usize, usize)> {
    let half_overlap = CHUNK_OVERLAP_SAMPLES as f64 / 32_000.0;
    let old_start = merged.partition_point(|w| w.start < seam - half_overlap);
    let old_end = merged.partition_point(|w| w.start <= seam + half_overlap);
    let new_start = next.partition_point(|w| w.start < seam - half_overlap);
    let new_end = next.partition_point(|w| w.start <= seam + half_overlap);
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
            // midpoint, new after it. The text, punctuation and confidence
            // of the chosen WordInfo stay intact.
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
pub(crate) fn stitch_chunk_words(
    mut merged: Vec<WordInfo>,
    next: Vec<WordInfo>,
    seam_s: f64,
) -> Vec<WordInfo> {
    if merged.is_empty() {
        return next;
    }
    if next.is_empty() {
        return merged;
    }
    let (old_end, new_start) = aligned_cut(&merged, &next, seam_s).unwrap_or_else(|| {
        (
            merged.partition_point(|w| w.start <= seam_s),
            next.partition_point(|w| w.start <= seam_s),
        )
    });
    merged.truncate(old_end);
    merged.extend(next.into_iter().skip(new_start));
    merged
}
