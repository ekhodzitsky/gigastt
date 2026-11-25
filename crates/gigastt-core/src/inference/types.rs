//! File-transcription requests, overrides and channel result assembly.

use crate::transcript::{TranscribeResult, aggregate_confidence};

mod overrides;
mod request;

pub use overrides::{
    DEFAULT_HOTWORDS_BOOST, HotwordError, HotwordOverride, MAX_HOTWORD_PHRASE_CHARS,
    MAX_HOTWORDS_PER_REQUEST, OverrideError, TranscribeOverrides,
};
pub use request::{TranscribeRequest, TranscribeSource};

/// Merge per-channel [`TranscribeResult`]s into a single chronologically ordered
/// result. Each channel is assigned a zero-based speaker label (`speaker_0`,
/// `speaker_1`, …). Words are sorted by `start`; equal timestamps are ordered by
/// channel index for stability.
pub fn merge_channel_results(per_channel: Vec<TranscribeResult>) -> TranscribeResult {
    let mut all_words = Vec::new();
    let mut duration_s = 0.0_f64;
    for (channel_idx, mut result) in per_channel.into_iter().enumerate() {
        let speaker = channel_idx as u32;
        for w in &mut result.words {
            w.speaker = Some(speaker);
        }
        duration_s = duration_s.max(result.duration_s);
        all_words.extend(result.words);
    }

    all_words.sort_by(|a, b| {
        a.start
            .total_cmp(&b.start)
            .then_with(|| a.speaker.cmp(&b.speaker))
    });

    let text = all_words
        .iter()
        .map(|w| w.word.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    TranscribeResult {
        confidence: aggregate_confidence(&all_words),
        text,
        words: all_words,
        duration_s,
    }
}

/// Outcome of an offline speaker-diarization attempt for a file-transcription
/// request.
///
/// Recorded into the caller-supplied [`TranscribeRequest::diarization_outcome`]
/// sink so a `?diarization=true` request that ends up with no speaker labels can
/// be surfaced *with a reason* instead of returning an all-empty-speaker
/// transcript silently (HTTP 200 today). The sink is written only when
/// diarization was requested; a plain transcript leaves it untouched.
///
/// A new variant is additive: the enum is `#[non_exhaustive]`, and downstream
/// mappers already need a catch-all arm.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum DiarizationOutcome {
    /// Speaker turns were produced and each word labeled.
    Applied,
    /// Requested, but no speaker encoder is available — the model file is
    /// absent or failed to load, or this build lacks the `diarization` feature.
    /// Server capability is advertised on `/health` and the WebSocket `Ready`
    /// message; this reports it per request.
    NoSpeakerModel,
    /// The clusterer refused the input because it exceeds the maximum duration
    /// it can process in a single global pass. Both fields are seconds of audio,
    /// as reported by the clusterer (not re-derived here).
    DurationCeiling {
        /// Length of the submitted audio.
        input_secs: f64,
        /// The clusterer's single-pass ceiling.
        ceiling_secs: f64,
    },
    /// Attempted but the diarization pipeline failed for another reason (already
    /// logged); no numbers to report.
    Failed,
}

#[cfg(test)]
mod override_and_merge_tests {
    use super::*;
    use crate::transcript::WordInfo;

    #[test]
    fn test_transcribe_overrides_default_all_none() {
        // The default overrides must be all-`None` so a request with no knobs
        // reproduces the engine's boot behaviour byte-for-byte.
        let o = TranscribeOverrides::default();
        assert_eq!(o.punctuation, None);
        assert_eq!(o.itn, None);
        assert_eq!(o.vad, None);
    }

    #[test]
    fn test_hotword_override_limits_constants() {
        // Documented DoS caps used by validate_overrides and the REST 400 path.
        assert_eq!(MAX_HOTWORDS_PER_REQUEST, 64);
        assert_eq!(MAX_HOTWORD_PHRASE_CHARS, 64);
        assert_eq!(DEFAULT_HOTWORDS_BOOST, 5.0);
    }

    #[test]
    fn test_override_error_codes_stable() {
        // Stable machine-readable codes surfaced as the REST error `code`.
        assert_eq!(OverrideError::VadNotLoaded.code(), "vad_not_loaded");
        assert_eq!(
            OverrideError::PunctuationNotAvailable.code(),
            "punctuation_not_available"
        );
        assert_eq!(HotwordError::TooManyHotwords.code(), "too_many_hotwords");
        assert_eq!(
            HotwordError::PhraseTooLong.code(),
            "hotword_phrase_too_long"
        );
        // Messages are non-empty and don't leak internals.
        assert!(!OverrideError::VadNotLoaded.message().is_empty());
        assert!(!OverrideError::PunctuationNotAvailable.message().is_empty());
        assert!(!HotwordError::TooManyHotwords.message().is_empty());
        assert!(!HotwordError::PhraseTooLong.message().is_empty());
        // Display matches message().
        assert_eq!(
            OverrideError::VadNotLoaded.to_string(),
            OverrideError::VadNotLoaded.message()
        );
        // Limit violations are client errors (400); missing models are 409.
    }

    fn sample_word(w: &str, start: f64, end: f64, speaker: Option<u32>) -> WordInfo {
        WordInfo::new(w, start, end, 0.9, speaker)
    }

    #[test]
    fn test_merge_channel_results_empty() {
        let merged = merge_channel_results(vec![
            TranscribeResult {
                text: String::new(),
                words: vec![],
                duration_s: 0.0,
                confidence: None,
            },
            TranscribeResult {
                text: String::new(),
                words: vec![],
                duration_s: 0.0,
                confidence: None,
            },
        ]);
        assert!(merged.words.is_empty());
        assert!(merged.text.is_empty());
    }

    #[test]
    fn test_merge_channel_results_interleaved_channels() {
        let ch0 = TranscribeResult {
            text: String::new(),
            words: vec![
                sample_word("привет", 0.0, 0.4, None),
                sample_word("как", 1.0, 1.3, None),
            ],
            duration_s: 1.5,
            confidence: None,
        };
        let ch1 = TranscribeResult {
            text: String::new(),
            words: vec![sample_word("да", 0.5, 0.8, None)],
            duration_s: 1.5,
            confidence: None,
        };
        let merged = merge_channel_results(vec![ch0, ch1]);
        assert_eq!(merged.words.len(), 3);
        assert_eq!(merged.words[0].word, "привет");
        assert_eq!(merged.words[0].speaker, Some(0));
        assert_eq!(merged.words[1].word, "да");
        assert_eq!(merged.words[1].speaker, Some(1));
        assert_eq!(merged.words[2].word, "как");
        assert_eq!(merged.words[2].speaker, Some(0));
    }

    #[test]
    fn test_merge_channel_results_tie_order_by_channel() {
        let ch0 = TranscribeResult {
            text: String::new(),
            words: vec![sample_word("а", 0.5, 0.7, None)],
            duration_s: 1.0,
            confidence: None,
        };
        let ch1 = TranscribeResult {
            text: String::new(),
            words: vec![sample_word("б", 0.5, 0.7, None)],
            duration_s: 1.0,
            confidence: None,
        };
        let merged = merge_channel_results(vec![ch0, ch1]);
        assert_eq!(merged.words[0].word, "а");
        assert_eq!(merged.words[0].speaker, Some(0));
        assert_eq!(merged.words[1].word, "б");
        assert_eq!(merged.words[1].speaker, Some(1));
    }

    #[test]
    fn test_merge_channel_results_no_channels() {
        let merged = merge_channel_results(vec![]);
        assert!(merged.words.is_empty());
        assert!(merged.text.is_empty());
        assert_eq!(merged.duration_s, 0.0);
    }

    #[test]
    fn test_merge_channel_results_max_duration() {
        let ch0 = TranscribeResult {
            text: String::new(),
            words: vec![sample_word("a", 0.0, 0.5, None)],
            duration_s: 5.0,
            confidence: None,
        };
        let ch1 = TranscribeResult {
            text: String::new(),
            words: vec![sample_word("b", 0.5, 1.0, None)],
            duration_s: 12.0,
            confidence: None,
        };
        let merged = merge_channel_results(vec![ch0, ch1]);
        assert_eq!(merged.duration_s, 12.0);
    }
}
