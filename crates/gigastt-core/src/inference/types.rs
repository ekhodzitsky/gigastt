//! File-transcription result types and per-request overrides.

use serde::Serialize;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, OnceLock};

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

/// Input audio for a single file-transcription request.
///
/// Prefer constructing a [`TranscribeRequest`] and calling
/// [`crate::inference::Engine::transcribe_request`] instead of the combinatorial
/// `transcribe_*_with_overrides_*` entry points (kept as thin wrappers).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum TranscribeSource<'a> {
    /// Filesystem path decoded via the file pipeline (WAV/MP3/M4A/OGG/FLAC/…).
    #[cfg(feature = "file-decode")]
    Path(&'a str),
    /// Reference-counted byte buffer (zero-copy REST / jobs upload path).
    #[cfg(feature = "file-decode")]
    Bytes(bytes::Bytes),
    /// Pre-decoded mono 16 kHz f32 samples.
    Samples(&'a [f32]),
    /// Pre-decoded per-channel 16 kHz mono samples (`channels=split` /
    /// `--stereo-speakers`). Channel index becomes the speaker label;
    /// [`TranscribeRequest::diarization`] is ignored for this source.
    Channels(&'a [Vec<f32>]),
}

/// Unified file-transcription request (builder-friendly).
///
/// Collapses the combinatorial `transcribe_file` / `transcribe_bytes*` /
/// `transcribe_channels` entry points into one path. Construct with
/// [`TranscribeRequest::new`] and chain [`with_overrides`](Self::with_overrides)
/// / [`with_hotwords`](Self::with_hotwords) / [`with_diarization`](Self::with_diarization).
///
/// Defaults match the historical plain methods: engine boot overrides, no
/// per-request hotwords, diarization off.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TranscribeRequest<'a> {
    /// Audio input (path, bytes, samples, or split channels).
    pub source: TranscribeSource<'a>,
    /// Per-request recognition knobs (`None` fields = engine boot default).
    pub overrides: TranscribeOverrides,
    /// Optional per-request hotword biaser override. See [`HotwordOverride`].
    pub hotwords: Option<&'a HotwordOverride>,
    /// When `true` and the source is mono samples/bytes/path, run offline
    /// speaker diarization after decode (no-op without a loaded speaker
    /// encoder / `diarization` feature). Ignored for
    /// [`TranscribeSource::Channels`].
    pub diarization: bool,
    /// Optional cooperative-cancellation flag. When set and flipped to `true`
    /// by another thread (client disconnect, `DELETE /v1/jobs/{id}`, shutdown,
    /// or the no-progress inference watchdog), the decode loop observes it at a
    /// window boundary and returns
    /// [`GigasttError::Cancelled`](crate::error::GigasttError::Cancelled),
    /// releasing the pooled session within one window instead of running to
    /// completion. `None` (the default) is the historical, non-cancellable
    /// behaviour.
    pub abort: Option<Arc<AtomicBool>>,
    /// Optional progress sink. When set, the long-form decode stores the number
    /// of 16 kHz samples processed so far (monotonically increasing, ending at
    /// the decoded length) after each window completes. A server watchdog reads
    /// it both to reset its no-progress deadline and to drive a real per-window
    /// job progress bar. `None` (the default) reports nothing.
    pub progress: Option<Arc<AtomicU64>>,
    /// Optional write-once sink for the offline speaker-diarization outcome.
    /// When set and [`diarization`](Self::diarization) is true, the engine
    /// records why speakers were or were not labeled ([`DiarizationOutcome`])
    /// so the caller can surface a capability notice on the response instead of
    /// returning an all-empty-speaker transcript silently. `None` (the default)
    /// records nothing, reproducing the historical behaviour.
    pub diarization_outcome: Option<Arc<OnceLock<DiarizationOutcome>>>,
    /// Optional opt-in maximum decoded audio length, in seconds. `None` (the
    /// default) leaves the streaming file path unbounded — a file of any length
    /// transcribes with O(one window) peak memory. When `Some(secs)`, audio
    /// longer than `secs` is rejected with
    /// [`GigasttError::AudioTooLong`](crate::error::GigasttError::AudioTooLong).
    /// The whole-buffer paths (VAD, diarization, `channels=split`, telephony /
    /// Opus) additionally clamp to a fixed safety ceiling regardless of this
    /// value, so they refuse rather than exhaust memory.
    pub max_audio_secs: Option<f64>,
}

impl<'a> TranscribeRequest<'a> {
    /// Build a request with default overrides, no hotwords, and diarization off.
    pub fn new(source: TranscribeSource<'a>) -> Self {
        Self {
            source,
            overrides: TranscribeOverrides::default(),
            hotwords: None,
            diarization: false,
            abort: None,
            progress: None,
            diarization_outcome: None,
            max_audio_secs: None,
        }
    }

    /// Set per-request recognition-knob overrides.
    pub fn with_overrides(mut self, overrides: TranscribeOverrides) -> Self {
        self.overrides = overrides;
        self
    }

    /// Set optional per-request hotword override.
    pub fn with_hotwords(mut self, hotwords: Option<&'a HotwordOverride>) -> Self {
        self.hotwords = hotwords;
        self
    }

    /// Enable or disable offline speaker diarization for mono sources.
    pub fn with_diarization(mut self, diarization: bool) -> Self {
        self.diarization = diarization;
        self
    }

    /// Attach a cooperative-cancellation flag. Flipping the shared
    /// [`AtomicBool`] to `true` from another thread makes the decode return
    /// [`GigasttError::Cancelled`](crate::error::GigasttError::Cancelled) at the
    /// next window boundary. `None` restores the non-cancellable default.
    pub fn with_abort(mut self, abort: Option<Arc<AtomicBool>>) -> Self {
        self.abort = abort;
        self
    }

    /// Attach a progress sink that receives the cumulative count of processed
    /// 16 kHz samples after each long-form window. `None` reports nothing.
    pub fn with_progress(mut self, progress: Option<Arc<AtomicU64>>) -> Self {
        self.progress = progress;
        self
    }

    /// Attach a write-once sink that receives the offline-diarization
    /// [`DiarizationOutcome`] for this request. `None` records nothing.
    pub fn with_diarization_outcome(
        mut self,
        sink: Option<Arc<OnceLock<DiarizationOutcome>>>,
    ) -> Self {
        self.diarization_outcome = sink;
        self
    }

    /// Set an opt-in maximum decoded audio length in seconds. `None` (the
    /// default) leaves the streaming path unbounded; the whole-buffer paths keep
    /// their fixed safety ceiling either way.
    pub fn with_max_audio_secs(mut self, max_audio_secs: Option<f64>) -> Self {
        self.max_audio_secs = max_audio_secs;
        self
    }
}

#[cfg(test)]
mod request_tests {
    use super::*;

    #[test]
    fn test_transcribe_request_builder_defaults() {
        let samples: &[f32] = &[];
        let req = TranscribeRequest::new(TranscribeSource::Samples(samples));
        assert!(matches!(req.source, TranscribeSource::Samples(_)));
        assert!(req.overrides.punctuation.is_none());
        assert!(req.hotwords.is_none());
        assert!(!req.diarization);
    }

    #[test]
    fn test_transcribe_request_builder_chain() {
        let samples: &[f32] = &[];
        let hw = HotwordOverride::new(vec!["тест".into()], Some(3.0));
        let req = TranscribeRequest::new(TranscribeSource::Samples(samples))
            .with_overrides(TranscribeOverrides {
                punctuation: Some(false),
                itn: Some(true),
                vad: Some(false),
            })
            .with_hotwords(Some(&hw))
            .with_diarization(true);
        assert_eq!(req.overrides.punctuation, Some(false));
        assert_eq!(req.overrides.itn, Some(true));
        assert!(req.hotwords.is_some());
        assert!(req.diarization);
    }
}
