//! Unified file-transcription request and source enum.

use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, OnceLock};

use super::{DiarizationOutcome, HotwordOverride, TranscribeOverrides};

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
    /// `channels=split` over a container that is **not** materialized: each
    /// channel is pulled through the windowed decode in turn, so peak audio
    /// memory is one window rather than every channel of the whole file.
    ///
    /// Prefer this over [`TranscribeSource::Channels`] when the caller has the
    /// encoded bytes: that variant needs every channel decoded up front, which
    /// is what puts a duration ceiling on the split path. Decide the channel
    /// count — and whether splitting is right at all — with
    /// [`scan_channels`](crate::inference::audio::scan_channels).
    #[cfg(feature = "file-decode")]
    ChannelStreams {
        /// Encoded container bytes; cloned per channel (a refcount bump).
        data: bytes::Bytes,
        /// Channels to decode, each transcribed as its own speaker.
        channels: usize,
    },
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
    /// token/frame boundary and returns
    /// [`GigasttError::Cancelled`](crate::error::GigasttError::Cancelled),
    /// releasing the pooled session after the current runtime call instead of
    /// running to completion. `None` (the default) is the historical, non-cancellable
    /// behaviour.
    pub abort: Option<Arc<AtomicBool>>,
    /// Optional readable partial, retained when the request returns Cancelled.
    pub partial: Option<Arc<crate::inference::TranscriptSnapshot>>,
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
    /// The whole-buffer paths (diarization, `channels=split` — including its
    /// per-channel Opus decode — and the raw telephony codecs)
    /// additionally clamp to a fixed safety ceiling regardless of this value,
    /// so they refuse rather than exhaust memory. The VAD file path,
    /// WAVE ingest, and streamed OGG/Opus decode in bounded windows and stay
    /// unbounded.
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
            partial: None,
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
    /// next decode step. A native encoder call must return first. `None`
    /// restores the non-cancellable default.
    pub fn with_abort(mut self, abort: Option<Arc<AtomicBool>>) -> Self {
        self.abort = abort;
        self
    }

    /// Attach a sink for provisional text, including the interrupted window.
    pub fn with_partial(
        mut self,
        partial: Option<Arc<crate::inference::TranscriptSnapshot>>,
    ) -> Self {
        self.partial = partial;
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
