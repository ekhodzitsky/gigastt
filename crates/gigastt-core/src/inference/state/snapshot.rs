//! Readable snapshots with immutable file-window chunks and owned read results.
use std::sync::Arc;

use crate::transcript::{TranscriptSegment, WordInfo, aggregate_confidence};

struct WordChunk {
    previous: Option<Arc<WordChunk>>,
    words: Arc<[WordInfo]>,
    used: usize,
    total: usize,
}

impl std::fmt::Debug for WordChunk {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WordChunk")
            .field("used", &self.used)
            .field("total", &self.total)
            .field("has_previous", &self.previous.is_some())
            .finish()
    }
}

impl Drop for WordChunk {
    fn drop(&mut self) {
        // Long recordings can contain many windows. Release unshared history
        // iteratively rather than overflowing the stack through Arc's drops.
        let mut previous = self.previous.take();
        while let Some(chunk) = previous {
            match Arc::try_unwrap(chunk) {
                Ok(mut chunk) => previous = chunk.previous.take(),
                Err(_) => break,
            }
        }
    }
}

fn retain_prefix(mut head: Option<Arc<WordChunk>>, retained: usize) -> Option<Arc<WordChunk>> {
    if retained == 0 {
        return None;
    }
    while let Some(chunk) = head {
        if retained >= chunk.total {
            return Some(chunk);
        }
        let before = chunk.total - chunk.used;
        if retained <= before {
            head = chunk.previous.clone();
            continue;
        }
        let used = retained - before;
        // A truncated view normally shares a bounded window allocation. Compact
        // when more than half would be abandoned, bounding retained word storage
        // to twice the visible words even after repeated deep truncation.
        let words = if used < chunk.words.len().div_ceil(2) {
            Arc::from(&chunk.words[..used])
        } else {
            chunk.words.clone()
        };
        return Some(Arc::new(WordChunk {
            previous: chunk.previous.clone(),
            words,
            used,
            total: retained,
        }));
    }
    None
}

fn collect_words(head: &Option<Arc<WordChunk>>, output: &mut Vec<WordInfo>) {
    let mut chunks = Vec::new();
    let mut cursor = head.as_ref();
    while let Some(chunk) = cursor {
        chunks.push(chunk);
        cursor = chunk.previous.as_ref();
    }
    for chunk in chunks.into_iter().rev() {
        output.extend_from_slice(&chunk.words[..chunk.used]);
    }
}

#[derive(Debug, Clone)]
enum SnapshotData {
    Segment(Arc<TranscriptSegment>),
    File {
        channels: Vec<Option<Arc<WordChunk>>>,
        split: bool,
        timestamp: f64,
    },
}

/// Last readable transcript, shared with the caller of a blocking decode.
/// Updated after each file window, streaming hypothesis, or interrupted decode.
/// A snapshot is provisional and never represents successful completion.
#[derive(Debug, Default)]
pub struct TranscriptSnapshot(parking_lot::Mutex<Option<SnapshotData>>);

impl TranscriptSnapshot {
    pub(crate) fn clear(&self) {
        let old = self.0.lock().take();
        drop(old);
    }

    /// Read the latest snapshot, including after cancellation or timeout.
    /// Copies words/text outside the state lock; split-channel reads additionally
    /// sort the result into chronological order.
    pub fn get(&self) -> Option<TranscriptSegment> {
        let snapshot = self.0.lock().clone()?;
        match snapshot {
            SnapshotData::Segment(segment) => Some((*segment).clone()),
            SnapshotData::File {
                channels,
                split,
                timestamp,
            } => {
                let capacity = channels.iter().flatten().map(|head| head.total).sum();
                let mut words = Vec::with_capacity(capacity);
                for channel in &channels {
                    collect_words(channel, &mut words);
                }
                if split {
                    words.sort_by(|a, b| {
                        a.start
                            .total_cmp(&b.start)
                            .then_with(|| a.speaker.cmp(&b.speaker))
                    });
                }
                let text_capacity = words.iter().map(|word| word.word.len()).sum::<usize>()
                    + words.len().saturating_sub(1);
                let mut text = String::with_capacity(text_capacity);
                for word in &words {
                    if !text.is_empty() {
                        text.push(' ');
                    }
                    text.push_str(&word.word);
                }
                let confidence = aggregate_confidence(&words);
                Some(TranscriptSegment {
                    tentative: text.clone(),
                    text,
                    committed: String::new(),
                    words,
                    is_final: false,
                    speech_final: false,
                    endpoint_reason: None,
                    timestamp,
                    confidence,
                    truncated: false,
                })
            }
        }
    }

    pub(crate) fn store(&self, mut segment: TranscriptSegment) {
        segment.is_final = false;
        segment.speech_final = false;
        segment.endpoint_reason = None;
        let old = self
            .0
            .lock()
            .replace(SnapshotData::Segment(Arc::new(segment)));
        drop(old);
    }
}

/// One decode request's independent publication history. The shared sink never
/// applies deltas: concurrent requests can safely publish complete snapshots.
#[derive(Default)]
pub(crate) struct SnapshotPublisher {
    channels: Vec<Option<Arc<WordChunk>>>,
    split: bool,
}

impl SnapshotPublisher {
    pub(crate) fn publish(
        &mut self,
        snapshot: &TranscriptSnapshot,
        retained: usize,
        mut words: Vec<WordInfo>,
        channel: Option<usize>,
        timestamp: f64,
    ) {
        if let Some(channel) = channel {
            for word in &mut words {
                word.speaker = Some(channel as u32);
            }
        }
        let words: Arc<[WordInfo]> = words.into();
        let index = channel.unwrap_or(0);
        if self.split != channel.is_some() || (index == 0 && retained == 0) {
            self.channels.clear();
        }
        self.split = channel.is_some();
        self.channels
            .resize_with(self.channels.len().max(index + 1), || None);
        let previous = retain_prefix(self.channels[index].take(), retained);
        let before = previous.as_ref().map_or(0, |chunk| chunk.total);
        debug_assert_eq!(before, retained, "publication retained unavailable words");
        self.channels[index] = if words.is_empty() {
            previous
        } else {
            Some(Arc::new(WordChunk {
                previous,
                total: before + words.len(),
                used: words.len(),
                words,
            }))
        };
        let next = SnapshotData::File {
            channels: self.channels.clone(),
            split: self.split,
            timestamp,
        };
        let old = snapshot.0.lock().replace(next);
        drop(old);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::TranscriptAssembler;

    fn words(start: usize, count: usize) -> Vec<WordInfo> {
        (start..start + count)
            .map(|n| WordInfo::new(format!("word{n}"), n as f64, n as f64 + 0.5, 0.9, None))
            .collect()
    }

    fn legacy(words: Vec<WordInfo>) -> TranscriptSegment {
        let mut assembler = TranscriptAssembler::new();
        assembler.append(words);
        assembler.partial(123.0)
    }

    #[test]
    fn test_snapshot_replacements_match_full_rebuild_after_every_update() {
        let snapshot = TranscriptSnapshot::default();
        let mut publisher = SnapshotPublisher::default();
        let mut expected = Vec::new();
        for (retained, start, count) in [
            (0, 0, 20),
            (15, 30, 12),
            (2, 50, 3),
            (5, 60, 20),
            (1, 80, 0),
            (0, 0, 0),
            (0, 90, 4),
        ] {
            expected.truncate(retained);
            expected.extend(words(start, count));
            publisher.publish(&snapshot, retained, words(start, count), None, 123.0);
            assert_eq!(
                serde_json::to_value(snapshot.get().unwrap()).unwrap(),
                serde_json::to_value(legacy(expected.clone())).unwrap()
            );
        }
    }

    #[test]
    fn test_snapshot_deep_chain_drop_and_truncation_are_bounded() {
        let snapshot = TranscriptSnapshot::default();
        let mut publisher = SnapshotPublisher::default();
        let depth = if cfg!(miri) { 40 } else { 20_000 };
        for n in 0..depth {
            publisher.publish(&snapshot, n, words(n, 1), None, 123.0);
        }
        assert_eq!(snapshot.get().unwrap().words.len(), depth);
        assert!(format!("{snapshot:?}").len() < 512);
        publisher.publish(&snapshot, 1, words(depth, 2), None, 123.0);
        assert_eq!(snapshot.get().unwrap().words.len(), 3);
        snapshot.clear();
        assert!(snapshot.get().is_none());
        for n in 0..depth {
            publisher.publish(&snapshot, n, words(n, 1), None, 123.0);
        }
        drop(snapshot);
    }

    #[test]
    fn test_snapshot_truncated_chunk_does_not_retain_abandoned_storage() {
        let snapshot = TranscriptSnapshot::default();
        let mut publisher = SnapshotPublisher::default();
        publisher.publish(&snapshot, 0, words(0, 1025), None, 123.0);
        for retained in [512, 128, 32, 8, 1] {
            publisher.publish(&snapshot, retained, Vec::new(), None, 123.0);
            let guard = snapshot.0.lock();
            let Some(SnapshotData::File { channels, .. }) = &*guard else {
                panic!("file snapshot")
            };
            let head = channels[0].as_ref().unwrap();
            assert!(head.words.len() <= 2 * retained);
        }
    }

    #[test]
    fn test_snapshot_concurrent_readers_see_complete_consistent_updates() {
        let snapshot = TranscriptSnapshot::default();
        let mut publisher = SnapshotPublisher::default();
        let iterations = if cfg!(miri) { 16 } else { 256 };
        std::thread::scope(|scope| {
            scope.spawn(|| {
                for n in 0..iterations {
                    publisher.publish(&snapshot, n, words(n, 1), None, 123.0);
                }
            });
            for _ in 0..4 {
                scope.spawn(|| {
                    for _ in 0..iterations {
                        if let Some(segment) = snapshot.get() {
                            assert_eq!(
                                serde_json::to_value(&segment).unwrap(),
                                serde_json::to_value(legacy(words(0, segment.words.len())))
                                    .unwrap()
                            );
                        }
                    }
                });
            }
        });
        assert_eq!(snapshot.get().unwrap().words.len(), iterations);
    }
    #[test]
    fn test_snapshot_channel_updates_preserve_ties_and_completed_channels() {
        use crate::inference::{TranscribeResult, merge_channel_results};
        let snapshot = TranscriptSnapshot::default();
        let mut publisher = SnapshotPublisher::default();
        let mut expected = [Vec::new(), Vec::new()];
        for (channel, retained, start, count) in
            [(0, 0, 0, 5), (0, 3, 3, 5), (1, 0, 0, 4), (1, 2, 2, 5)]
        {
            expected[channel].truncate(retained);
            expected[channel].extend(words(start, count));
            publisher.publish(
                &snapshot,
                retained,
                words(start, count),
                Some(channel),
                123.0,
            );
            let merged = merge_channel_results(
                expected
                    .iter()
                    .map(|words| TranscribeResult {
                        words: words.clone(),
                        text: String::new(),
                        duration_s: 0.0,
                        confidence: None,
                    })
                    .collect(),
            );
            assert_eq!(
                serde_json::to_value(snapshot.get().unwrap()).unwrap(),
                serde_json::to_value(legacy(merged.words)).unwrap()
            );
        }
        publisher.publish(&snapshot, 0, words(30, 1), Some(0), 123.0);
        assert_eq!(snapshot.get().unwrap().words.len(), 1);
    }
    #[test]
    fn test_snapshot_empty_word_spacing_matches_legacy_assembler() {
        let snapshot = TranscriptSnapshot::default();
        let mut publisher = SnapshotPublisher::default();
        for labels in [vec!["", "a"], vec!["", "", "a", "", "b", ""], vec!["", ""]] {
            let words: Vec<_> = labels
                .into_iter()
                .map(|label| WordInfo::new(label, 0.0, 1.0, 0.8, None))
                .collect();
            publisher.publish(&snapshot, 0, words.clone(), None, 123.0);
            assert_eq!(
                serde_json::to_value(snapshot.get().unwrap()).unwrap(),
                serde_json::to_value(legacy(words)).unwrap()
            );
        }
    }
    #[test]
    fn test_first_publication_on_later_channel_replaces_previous_request() {
        let snapshot = TranscriptSnapshot::default();
        let mut publisher = SnapshotPublisher::default();
        publisher.publish(&snapshot, 0, words(100, 4), Some(0), 123.0);
        publisher.publish(&snapshot, 0, words(200, 4), Some(1), 123.0);
        // The next request's first channel is empty and publishes no window.
        publisher = SnapshotPublisher::default();
        publisher.publish(&snapshot, 0, words(0, 2), Some(1), 123.0);
        let result = snapshot.get().unwrap();
        assert_eq!(result.words.len(), 2);
        assert!(result.words.iter().all(|word| word.speaker == Some(1)));
        assert_eq!(result.text, "word0 word1");
    }
    #[test]
    fn test_independent_publishers_share_snapshot_without_mixing_histories() {
        let snapshot = TranscriptSnapshot::default();
        let mut first = SnapshotPublisher::default();
        let mut second = SnapshotPublisher::default();
        first.publish(&snapshot, 0, words(0, 100), None, 123.0);
        second.publish(&snapshot, 0, words(200, 20), None, 123.0);
        first.publish(&snapshot, 90, words(90, 20), None, 123.0);
        assert_eq!(
            serde_json::to_value(snapshot.get().unwrap()).unwrap(),
            serde_json::to_value(legacy(words(0, 110))).unwrap()
        );
        second.publish(&snapshot, 15, words(215, 20), None, 123.0);
        assert_eq!(
            serde_json::to_value(snapshot.get().unwrap()).unwrap(),
            serde_json::to_value(legacy(words(200, 35))).unwrap()
        );
    }
}
