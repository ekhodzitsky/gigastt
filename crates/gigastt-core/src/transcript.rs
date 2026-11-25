//! Transcript value types shared by recognition, wire protocols and exports.
//!
//! No engine or decoder state is needed to construct or serialize these values.

use serde::Serialize;

/// A recognized word with timing and confidence metadata.
///
/// Timestamps are in seconds relative to the start of the audio stream.
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct WordInfo {
    /// The recognized word text (BPE tokens joined, `▁` stripped).
    pub word: String,
    /// Start time in seconds from the beginning of the audio stream.
    pub start: f64,
    /// End time in seconds from the beginning of the audio stream.
    pub end: f64,
    /// Softmax confidence score (0.0–1.0), averaged over constituent BPE tokens.
    pub confidence: f32,
    /// Speaker label from diarization (zero-based index). Omitted if diarization is disabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<u32>,
}

impl WordInfo {
    /// Create a new [`WordInfo`].
    pub fn new(
        word: impl Into<String>,
        start: f64,
        end: f64,
        confidence: f32,
        speaker: Option<u32>,
    ) -> Self {
        Self {
            word: word.into(),
            start,
            end,
            confidence,
            speaker,
        }
    }
}

/// Aggregate per-word confidences into a single segment-level score:
/// a duration-weighted mean of `words[].confidence` (longer words count
/// more). Falls back to a plain mean when every word has zero duration, and
/// returns `None` for an empty word list. The result is an average of
/// per-word softmax scores — **not** a calibrated probability that the
/// segment is correct.
pub(crate) fn aggregate_confidence(words: &[WordInfo]) -> Option<f32> {
    if words.is_empty() {
        return None;
    }
    let mut weighted_sum = 0.0_f64;
    let mut total_weight = 0.0_f64;
    for w in words {
        let weight = (w.end - w.start).max(0.0);
        weighted_sum += f64::from(w.confidence) * weight;
        total_weight += weight;
    }
    let mean = if total_weight > 0.0 {
        weighted_sum / total_weight
    } else {
        words.iter().map(|w| f64::from(w.confidence)).sum::<f64>() / words.len() as f64
    };
    Some(mean as f32)
}

#[derive(Debug, Clone, Serialize)]
pub struct TranscribeResult {
    /// Full recognized transcript text (words joined with spaces).
    pub text: String,
    /// Word-level timing, confidence, and optional speaker annotations.
    pub words: Vec<WordInfo>,
    /// Duration of the decoded audio in seconds.
    pub duration_s: f64,
    /// Mean confidence across all words (duration-weighted average of
    /// `words[].confidence`; plain average when every word has zero
    /// duration). An average of per-word softmax scores — **not** a
    /// calibrated probability that the transcript is correct. `None` when no
    /// words were decoded; omitted from JSON in that case.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

/// Why a streaming segment was closed as a true utterance endpoint.
///
/// Window-cap slides are **not** utterance endpoints and never set this field —
/// they emit a non-final partial after committing a stable prefix instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum EndpointReason {
    /// Silero VAD detected trailing silence (`--vad` / `min_silence_ms`).
    Vad,
    /// Decoder blank-run heuristic (~600 ms) with no VAD attached.
    Blank,
    /// Client `stop`, connection drain, or explicit flush.
    Stop,
}

/// Recognized transcript segment, either interim or final.
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct TranscriptSegment {
    /// Recognized text for this segment.
    pub text: String,
    /// Immutable prefix of this utterance. Resets after its Final.
    pub committed: String,
    /// Revisable suffix, including any separating space. `text` is exactly
    /// `committed + tentative`; successful finals have an empty suffix.
    pub tentative: String,
    /// Individual words with timing and confidence metadata.
    pub words: Vec<WordInfo>,
    /// Whether this segment is final (utterance complete) or partial (interim).
    pub is_final: bool,
    /// True only for a true end-of-utterance (`final`). Always false on partials.
    /// Omitted from JSON when false so older clients never see a new key on partials.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub speech_final: bool,
    /// Why the utterance closed. Present only on true finals; omitted on partials
    /// and when the reason is not set (legacy empty finals).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint_reason: Option<EndpointReason>,
    /// The utterance was cut by a terminal cap (session limit or shutdown),
    /// not by a recognition failure. Omitted when false. Recognized text, if
    /// any, stays in `committed`.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
    /// Unix timestamp (seconds since epoch) when this segment was produced.
    pub timestamp: f64,
    /// Mean confidence across the segment's words (duration-weighted average
    /// of `words[].confidence`; plain average when every word has zero
    /// duration). An average of per-word softmax scores — **not** a
    /// calibrated probability that the segment is correct. `None` when the
    /// segment has no words; omitted from JSON in that case.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

impl TranscriptSegment {
    pub fn empty_final() -> Self {
        Self {
            text: String::new(),
            committed: String::new(),
            tentative: String::new(),
            words: vec![],
            is_final: true,
            speech_final: true,
            endpoint_reason: Some(EndpointReason::Stop),
            timestamp: now_timestamp(),
            confidence: None,
            truncated: false,
        }
    }

    /// Empty final that says the stream was cut before any text existed.
    pub fn empty_truncated_final() -> Self {
        let mut segment = Self::empty_final();
        segment.truncated = true;
        segment
    }

    /// Turn a partial (or an already-final segment) into a terminal final.
    ///
    /// `text` wins over the public `committed` field: `on_finalize` partials
    /// publish an empty committed prefix even when the assembler already
    /// holds words. A cap must not replace that text with an empty final.
    pub fn into_truncated_final(mut self) -> Self {
        if self.text.is_empty() && !self.committed.is_empty() {
            self.text = self.committed.clone();
        }
        if !self.text.is_empty() {
            self.committed = self.text.clone();
            self.tentative.clear();
        }
        self.is_final = true;
        self.speech_final = true;
        self.truncated = true;
        if self.endpoint_reason.is_none() {
            self.endpoint_reason = Some(EndpointReason::Stop);
        }
        self
    }
}

pub fn now_timestamp() -> f64 {
    use std::sync::OnceLock;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};
    // Anchor the wall-clock epoch to a monotonic `Instant` captured once, then
    // advance from it via `Instant::elapsed()`. Wire-visible timestamps stay
    // epoch-aligned (unchanged contract) but advance monotonically, immune to
    // NTP steps / wall-clock jumps mid-process.
    static ANCHOR: OnceLock<(SystemTime, Instant)> = OnceLock::new();
    let (epoch, start) = ANCHOR.get_or_init(|| (SystemTime::now(), Instant::now()));
    let base = match epoch.duration_since(UNIX_EPOCH) {
        Ok(d) => d.as_secs_f64(),
        Err(e) => {
            tracing::warn!("System clock is before Unix epoch: {e}");
            0.0
        }
    };
    base + start.elapsed().as_secs_f64()
}

#[cfg(test)]
mod timestamp_tests {
    use super::now_timestamp;

    #[test]
    fn test_now_timestamp_non_negative() {
        let ts = now_timestamp();
        assert!(ts >= 0.0, "timestamp must be non-negative");
    }

    #[test]
    fn test_now_timestamp_monotonic_and_epoch_aligned() {
        // Locks in the monotonic-anchor contract: two successive reads never go
        // backwards (immune to NTP steps), and the value stays Unix-epoch
        // aligned. A regression to a plain wall-clock read could violate either.
        let a = now_timestamp();
        let b = now_timestamp();
        assert!(
            b >= a,
            "now_timestamp must be non-decreasing (monotonic anchor)"
        );
        // Comfortably after 2023-11-14 and before a far-future sanity bound.
        assert!(
            a > 1_700_000_000.0,
            "timestamp must stay Unix-epoch aligned"
        );
        assert!(a < 4_000_000_000.0, "timestamp exceeds a sane upper bound");
    }
}

#[cfg(test)]
mod tests;
