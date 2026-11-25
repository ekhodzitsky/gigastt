//! Overlapping-window tiling for long-form punctuation restore.

/// Whitespace words labelled in one model run.
///
/// The exported RUPunct graph is fully dynamic but its position-embedding table
/// has 2048 rows, so a single run over a whole long transcript overflows the
/// embedding and fails the entire pass. 250 Russian words are roughly 600–900
/// WordPiece subtokens, which leaves a wide margin under that ceiling.
pub(super) const WINDOW_WORDS: usize = 250;

/// Words shared by neighbouring windows; must be even and below [`WINDOW_WORDS`].
///
/// Each window keeps only the labels of its middle and drops half of the overlap
/// on either side, so (except at the very start / end of the transcript) every
/// word is labelled from a window in which it has real left and right context.
pub(super) const WINDOW_OVERLAP_WORDS: usize = 40;

/// For each whitespace word index `0..num_words`, return the label id of its
/// FIRST subtoken — the token whose `word_id == Some(w)` with the lowest
/// position. This is RUPunct's `aggregation_strategy="first"`.
///
/// `word_ids` maps tokens to caller-supplied whitespace words (special tokens
/// are `None`);
/// `argmax_per_token` is the pre-computed argmax label id for each token.
/// Words with no subtoken (e.g. removed by normalization) have no label.
///
/// Pure (no model / I/O) so the first-subword selection is unit-testable.
pub(super) fn first_subword_labels(
    word_ids: &[Option<u32>],
    argmax_per_token: &[usize],
    num_words: usize,
) -> Vec<Option<usize>> {
    let mut labels = vec![None; num_words];
    for (tok_idx, wid) in word_ids.iter().enumerate() {
        let Some(w) = wid else { continue };
        let w = *w as usize;
        if let Some(slot) = labels.get_mut(w)
            && slot.is_none()
        {
            *slot = argmax_per_token.get(tok_idx).copied();
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
pub(super) fn word_spans(text: &str) -> Vec<(usize, usize)> {
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
pub(super) struct Window {
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) keep_start: usize,
    pub(super) keep_end: usize,
}

/// Tile `num_words` words with overlapping windows of at most [`WINDOW_WORDS`].
///
/// Windows advance by `WINDOW_WORDS - WINDOW_OVERLAP_WORDS` and the kept ranges
/// cut each overlap in half, so the kept ranges tile `0..num_words` with no gap
/// and no repeat while every window still sees `WINDOW_OVERLAP_WORDS / 2` words
/// of context beyond what it labels.
pub(super) fn plan_windows(num_words: usize) -> Vec<Window> {
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
/// is at index `j`), or `None` when that window's inference failed. Unlabelled
/// words and words covered only by a failed window remain `None` and are
/// rendered unchanged.
pub(super) fn splice_window_labels(
    windows: &[Window],
    per_window: &[Option<Vec<Option<usize>>>],
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
            *slot = labels.get(base + offset).copied().flatten();
        }
    }
    merged
}
