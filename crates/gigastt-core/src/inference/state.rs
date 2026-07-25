//! Streaming/session state, word timing, and transcript assembly.

use serde::{Deserialize, Serialize};

use super::PRED_HIDDEN;
use super::audio;
#[cfg(feature = "diarization")]
use super::diarization::StreamingDiarizationState;
use super::features::MelSpectrogram;
use super::now_timestamp;

#[non_exhaustive]
pub struct DecoderState {
    /// LSTM hidden state vector (length [`PRED_HIDDEN`]).
    pub h: Vec<f32>,
    /// LSTM cell state vector (length [`PRED_HIDDEN`]).
    pub c: Vec<f32>,
    /// Previously emitted token ID (initialized to `blank_id`).
    pub prev_token: i64,
    /// Count of consecutive blank frames (used for endpointing).
    pub consecutive_blanks: usize,
}

impl DecoderState {
    /// Create a new decoder state initialized to zeros with the given blank token ID.
    pub fn new(blank_id: usize) -> Self {
        Self {
            h: vec![0.0; PRED_HIDDEN],
            c: vec![0.0; PRED_HIDDEN],
            prev_token: blank_id as i64,
            consecutive_blanks: 0,
        }
    }
}

/// A recognized word with timing and confidence metadata.
///
/// Produced by the RNN-T decoder during [`crate::inference::Engine::process_chunk`] or [`crate::inference::Engine::transcribe_file`].
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

/// Per-connection streaming state that persists across audio chunks.
///
/// Created via [`crate::inference::Engine::create_state`]. Holds the decoder LSTM state, an audio
/// sample buffer for incomplete frames, and accumulated transcript text/words.
/// Pass this to [`crate::inference::Engine::process_chunk`] for each incoming audio chunk and
/// [`crate::inference::Engine::flush_state`] when the stream ends.
#[non_exhaustive]
pub struct StreamingState {
    /// Decoder LSTM hidden state (persisted across chunks).
    pub decoder: DecoderState,
    /// Leftover audio samples that didn't fill a complete frame.
    pub audio_buffer: Vec<f32>,
    /// Accumulated transcript builder (reset on endpointing).
    pub assembler: TranscriptAssembler,
    /// Absolute sample offset (@16kHz) of `audio_buffer[0]`: how much committed
    /// audio has slid off the front. Drives absolute word timestamps
    /// (encoder-frame offset = this / (HOP_LENGTH * ENCODER_SUBSAMPLING)).
    pub window_start_samples: usize,
    /// Leading samples of `audio_buffer` that are already-emitted left context;
    /// words decoded within this region are suppressed (not re-emitted).
    pub context_samples: usize,
    /// New samples accumulated since the last decode. The encoder re-runs only
    /// once this reaches `STREAM_DECODE_STRIDE_SAMPLES`, then resets to 0 — this
    /// is what keeps the stream real-time (re-decoding the window is the cost).
    pub pending_samples: usize,
    /// Optional cached resampler for non-16kHz streams.
    pub resampler: Option<rubato::Async<f32>>,
    /// Reusable FFT buffer for mel spectrogram (avoids per-chunk allocation).
    pub mel_fft_input: Vec<rustfft::num_complex::Complex<f32>>,
    /// Reusable power spectrum buffer for mel spectrogram.
    pub mel_power: Vec<f32>,
    /// Reusable mel-output buffer (avoids per-chunk allocation).
    pub mel_output: Vec<f32>,
    /// Reusable resampler output buffer (avoids per-chunk allocation).
    pub resample_output_buf: Vec<f32>,
    /// Optional VAD endpoint detector (present only when the engine has a VAD).
    /// Fed every chunk's raw samples to track trailing silence; when it fires,
    /// `process_chunk` finalizes the current segment. `None` = no VAD, and
    /// endpointing falls back to the decoder's blank-run heuristic alone;
    /// `Some` = the VAD owns endpointing and the blank-run heuristic is ignored.
    pub vad_endpointer: Option<crate::vad::VadEndpointer>,
    /// Per-session punctuation/casing-restoration override applied to **final**
    /// segments only (`None` = engine boot default). Set by the WS `configure`
    /// message; SSE and FFI streaming leave it `None` so the boot policy
    /// applies. Partials always stay raw.
    pub punctuation: Option<bool>,
    /// Per-session inverse-text-normalization override applied to **final**
    /// segments only (`None` = engine boot default).
    pub itn: Option<bool>,
    /// Per-session utterance-end policy (from engine boot or WS `configure`).
    pub endpoint_mode: EndpointMode,
    /// Diarization state (present only when diarization is enabled).
    #[cfg(feature = "diarization")]
    pub diarization_state: Option<StreamingDiarizationState>,
}

/// Audio feature extraction pipeline.
///
/// Owns the `MelSpectrogram` and handles audio buffering, resampling,
/// and log-mel feature extraction. Extracted so `Engine` does not need to
/// own the low-level signal-processing details directly.
pub struct FeatureExtractor {
    mel: MelSpectrogram,
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureExtractor {
    /// Create a new feature extractor with a freshly initialized mel spectrogram.
    pub fn new() -> Self {
        Self {
            mel: MelSpectrogram::new(),
        }
    }

    /// Prepare incoming samples (append to buffer, return the usable sample count if available).
    pub fn prepare_buffer(&self, samples: &[f32], audio_buffer: &mut Vec<f32>) -> Option<usize> {
        audio::prepare_audio_buffer(samples, audio_buffer)
    }

    /// Compute log-mel features from 16 kHz f32 samples, reusing state buffers.
    pub fn compute_mel(
        &self,
        samples: &[f32],
        fft_buf: &mut Vec<rustfft::num_complex::Complex<f32>>,
        power_buf: &mut Vec<f32>,
        output_buf: &mut Vec<f32>,
    ) -> usize {
        self.mel
            .compute_with_buffers(samples, fft_buf, power_buf, output_buf)
    }

    /// One-shot mel computation (for file transcription where buffer reuse is unnecessary).
    pub fn compute(&self, samples: &[f32]) -> (Vec<f32>, usize) {
        self.mel.compute(samples)
    }
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

/// Streaming utterance-end policy for WebSocket (and other streaming) sessions.
///
/// The encoder window cap is **never** an utterance endpoint under any mode —
/// it only commits a stable prefix so the next partial keeps growing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum EndpointMode {
    /// Default: VAD silence (if attached) or decoder blank-run ends utterances.
    #[default]
    Auto,
    /// Voice-assistant friendly: only VAD silence ends utterances automatically.
    /// Blank-run is ignored even without a VAD — pair with `--vad` or rely on
    /// client `stop`. Window cap never finalizes.
    Assistant,
    /// Only explicit `stop` / flush ends utterances (no blank, no VAD endpoint).
    Manual,
}

impl EndpointMode {
    /// Parse a wire / CLI token (`auto` | `assistant` | `manual`).
    pub fn parse_token(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "assistant" => Some(Self::Assistant),
            "manual" => Some(Self::Manual),
            _ => None,
        }
    }

    /// Canonical wire token for this mode.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Assistant => "assistant",
            Self::Manual => "manual",
        }
    }
}

/// Streaming transcript assembler.
///
/// Accumulates recognized words and builds partial / final [`TranscriptSegment`]
/// payloads. Separated from `Engine` so the segment-building policy can be tested
/// in isolation without loading ONNX models.
///
/// Holds a **committed** (stable) prefix plus a **live** tail. Window-cap slides
/// move the live tail into the committed prefix without ending the utterance;
/// true endpoints (`finalize`) take both and reset.
pub struct TranscriptAssembler {
    committed_text: String,
    committed_words: Vec<WordInfo>,
    text: String,
    words: Vec<WordInfo>,
}

impl Default for TranscriptAssembler {
    fn default() -> Self {
        Self::new()
    }
}

impl TranscriptAssembler {
    /// Create a new, empty transcript assembler.
    pub fn new() -> Self {
        Self {
            committed_text: String::new(),
            committed_words: Vec::new(),
            text: String::new(),
            words: Vec::new(),
        }
    }

    /// Append new words to the **live** tail of the transcript.
    pub fn append(&mut self, new_words: Vec<WordInfo>) {
        for w in &new_words {
            if !self.text.is_empty() {
                self.text.push(' ');
            }
            self.text.push_str(&w.word);
        }
        self.words.extend(new_words);
    }

    /// Replace the **live** tail with a freshly decoded hypothesis.
    ///
    /// The sliding-window streaming path re-decodes its whole context window on
    /// every chunk, so it overwrites (rather than appends) the current tail.
    /// The committed (stable) prefix is left untouched.
    pub fn set_words(&mut self, words: Vec<WordInfo>) {
        self.text = words
            .iter()
            .map(|w| w.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        self.words = words;
    }

    /// Move the live tail into the committed (stable) prefix without ending the
    /// utterance. Used when the encoder window slides: words that leave the
    /// re-decode context must not reappear, but the client must not see a
    /// `final` (that would mean "command complete" for voice assistants).
    pub fn commit_live(&mut self) {
        if self.words.is_empty() {
            return;
        }
        if !self.committed_text.is_empty() && !self.text.is_empty() {
            self.committed_text.push(' ');
        }
        self.committed_text.push_str(&self.text);
        self.committed_words.append(&mut self.words);
        self.text.clear();
    }

    /// Full utterance text (committed prefix + live tail).
    fn full_text(&self) -> String {
        if self.committed_text.is_empty() {
            self.text.clone()
        } else if self.text.is_empty() {
            self.committed_text.clone()
        } else {
            format!("{} {}", self.committed_text, self.text)
        }
    }

    /// Full utterance words (committed prefix + live tail).
    fn full_words(&self) -> Vec<WordInfo> {
        let mut out = Vec::with_capacity(self.committed_words.len() + self.words.len());
        out.extend_from_slice(&self.committed_words);
        out.extend_from_slice(&self.words);
        out
    }

    /// Build a **final** segment for a true utterance endpoint and reset.
    pub fn finalize(&mut self, timestamp: f64) -> TranscriptSegment {
        self.finalize_with_reason(timestamp, EndpointReason::Stop)
    }

    /// Build a **final** segment with an explicit endpoint reason and reset.
    pub fn finalize_with_reason(
        &mut self,
        timestamp: f64,
        reason: EndpointReason,
    ) -> TranscriptSegment {
        self.commit_live();
        let words = std::mem::take(&mut self.committed_words);
        let text = std::mem::take(&mut self.committed_text);
        let confidence = aggregate_confidence(&words);
        self.text.clear();
        self.words.clear();
        TranscriptSegment {
            text,
            words,
            is_final: true,
            speech_final: true,
            endpoint_reason: Some(reason),
            timestamp,
            confidence,
        }
    }

    /// Build a **partial** segment from committed + live without resetting.
    pub fn partial(&self, timestamp: f64) -> TranscriptSegment {
        let words = self.full_words();
        TranscriptSegment {
            text: self.full_text(),
            words: words.clone(),
            is_final: false,
            speech_final: false,
            endpoint_reason: None,
            timestamp,
            confidence: aggregate_confidence(&words),
        }
    }

    /// True if neither the committed prefix nor the live tail has words.
    pub fn is_empty(&self) -> bool {
        self.committed_text.is_empty() && self.text.is_empty()
    }
}

/// Probe a freshly-built state; on failure, rebuild it once and re-probe.
///
/// `probe` is a runtime self-check, `rebuild` converts the failed state into
/// a replacement (receiving the probe error so it can log the cause). A
/// rebuilt state that still fails the probe is a hard error — there is no
/// second fallback level.
///
/// Extracted from the CoreML runtime-fallback path (issue #42) so the

#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct TranscriptSegment {
    /// Recognized text for this segment.
    pub text: String,
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
            words: vec![],
            is_final: true,
            speech_final: true,
            endpoint_reason: Some(EndpointReason::Stop),
            timestamp: now_timestamp(),
            confidence: None,
        }
    }
}
