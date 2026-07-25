//! File-transcription result types and per-request overrides.

use serde::Serialize;

use super::state::{WordInfo, aggregate_confidence};

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

/// Maximum number of hotword phrases accepted on a single request. Larger
/// payloads are rejected by [`crate::inference::Engine::validate_hotwords`] (mapped to HTTP 400).
pub const MAX_HOTWORDS_PER_REQUEST: usize = 64;

/// Maximum length of a single hotword phrase in Unicode scalar values (chars).
/// Longer phrases are rejected by [`crate::inference::Engine::validate_hotwords`] (HTTP 400).
pub const MAX_HOTWORD_PHRASE_CHARS: usize = 64;

/// Default additive logit boost when a per-request hotword override omits
/// `boost` (matches the CLI `--hotwords-boost` default).
pub const DEFAULT_HOTWORDS_BOOST: f32 = 5.0;

/// Per-request hotword biasing override. Replaces the engine's boot-time
/// biaser for a single file-transcription call.
///
/// Semantics when passed to the hotwords parameter of file-transcription APIs:
/// - `None` (argument absent) → keep the engine boot biaser unchanged.
/// - `Some(empty phrases)` → force biasing **off** for this request.
/// - `Some(non-empty phrases)` → build a temporary hotword biaser for this
///   request only (engine boot biaser is not consulted).
///
/// Kept separate from [`TranscribeOverrides`] so that type remains `Copy`/`Eq`
/// (semver-stable for external struct literals and trait bounds).
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct HotwordOverride {
    /// Phrases to boost. Empty means force biasing off for the request.
    pub phrases: Vec<String>,
    /// Additive logit boost. `None` uses [`DEFAULT_HOTWORDS_BOOST`].
    pub boost: Option<f32>,
}

impl HotwordOverride {
    /// Construct a hotword override (preferred over struct-literal from outside
    /// this crate because the type is `#[non_exhaustive]`).
    pub fn new(phrases: Vec<String>, boost: Option<f32>) -> Self {
        Self { phrases, boost }
    }
}

/// Per-request overrides for the recognition post-processing knobs, letting a
/// single loaded engine vary punctuation / ITN / VAD per file-transcription
/// call instead of only at boot. `None` on a field means "use the engine's
/// boot default", so a `TranscribeOverrides::default()` (all `None`) reproduces
/// the pre-feature behaviour byte-for-byte.
///
/// A knob can only be turned *on* per-request if the underlying resource is
/// loaded: `vad = Some(true)` requires a VAD to be attached, and
/// `punctuation = Some(true)` requires a punctuator. Call
/// [`crate::inference::Engine::validate_overrides`] before transcribing to reject impossible
/// requests (mapped to `409` on the REST surface); turning a knob *off*
/// (`Some(false)`) is always valid.
///
/// Per-request hotwords live on [`HotwordOverride`] (validated via
/// [`crate::inference::Engine::validate_hotwords`]) so this struct stays `Copy`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TranscribeOverrides {
    /// Override the punctuation / casing restoration pass. `Some(true)` forces
    /// it on (requires a punctuator), `Some(false)` skips it, `None` = engine
    /// default (on iff a punctuator is attached).
    pub punctuation: Option<bool>,
    /// Override inverse text normalization (number-words → digits).
    /// `Some(true)` / `Some(false)` force the state; `None` = engine default.
    /// ITN is pure code (no model), so `Some(true)` is always valid.
    pub itn: Option<bool>,
    /// Override VAD gating. `Some(true)` decodes only detected speech regions
    /// (requires a VAD to be attached), `Some(false)` decodes the whole buffer,
    /// `None` = engine default (VAD path iff a VAD is attached).
    pub vad: Option<bool>,
}

/// Why a [`TranscribeOverrides`] was rejected: a knob was turned on per-request
/// but the resource backing it isn't loaded. Carries a stable machine-readable
/// [`code`](OverrideError::code) so the REST layer can surface a `409` with a
/// consistent contract without re-deriving the string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverrideError {
    /// `vad = Some(true)` but no VAD is attached to the engine.
    VadNotLoaded,
    /// `punctuation = Some(true)` but no punctuator is attached to the engine.
    PunctuationNotAvailable,
}

impl OverrideError {
    /// Stable, machine-readable error code for the REST `409` payload.
    pub fn code(self) -> &'static str {
        match self {
            OverrideError::VadNotLoaded => "vad_not_loaded",
            OverrideError::PunctuationNotAvailable => "punctuation_not_available",
        }
    }

    /// Human-readable, non-sensitive message for the REST `409` payload.
    pub fn message(self) -> &'static str {
        match self {
            OverrideError::VadNotLoaded => {
                "VAD requested but not loaded; start the server with --vad"
            }
            OverrideError::PunctuationNotAvailable => {
                "punctuation requested but no punctuation model is loaded"
            }
        }
    }
}

/// Why a [`HotwordOverride`] was rejected (DoS limits). New type so
/// [`OverrideError`] stays exhaustively matchable without a major bump.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HotwordError {
    /// More than [`MAX_HOTWORDS_PER_REQUEST`] phrases in the hotword override.
    TooManyHotwords,
    /// A hotword phrase exceeds [`MAX_HOTWORD_PHRASE_CHARS`] characters.
    PhraseTooLong,
}

impl HotwordError {
    /// Stable, machine-readable error code for the REST `400` payload.
    pub fn code(self) -> &'static str {
        match self {
            HotwordError::TooManyHotwords => "too_many_hotwords",
            HotwordError::PhraseTooLong => "hotword_phrase_too_long",
        }
    }

    /// Human-readable, non-sensitive message for the REST `400` payload.
    pub fn message(self) -> &'static str {
        match self {
            HotwordError::TooManyHotwords => "too many hotwords in request (max 64)",
            HotwordError::PhraseTooLong => "hotword phrase exceeds max length (64 characters)",
        }
    }
}

impl std::fmt::Display for OverrideError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for OverrideError {}

impl std::fmt::Display for HotwordError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for HotwordError {}

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
