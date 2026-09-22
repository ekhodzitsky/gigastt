//! `impl Engine` methods — split out of the former god-file.
use super::*;
impl Engine {
    /// Provider bound at load (`cpu`, `coreml`, `cuda`, `candle`, `ane`).
    pub fn execution_provider(&self) -> &str {
        &self.execution_provider
    }

    pub fn is_int8(&self) -> bool {
        self.int8
    }

    /// The recognition head ([`ModelVariant`]) detected on disk at load time.
    /// Lets callers decide the default punctuation policy (`auto`).
    pub fn variant(&self) -> ModelVariant {
        self.variant
    }

    /// Attach an optional punctuation / casing restorer, consuming and
    /// returning `self` (builder style). Pass `None` for pass-through. When set,
    /// the restorer post-processes the final text of file transcription
    /// ([`Engine::transcribe_file`] / [`Engine::transcribe_bytes_shared`]) and
    /// of finalized streaming segments ([`Engine::process_chunk`] /
    /// [`Engine::flush_state`]).
    pub fn with_punctuator(mut self, punctuator: Option<crate::punctuation::Punctuator>) -> Self {
        self.punctuator = punctuator;
        self
    }

    /// Whether a punctuation restorer is attached.
    pub fn has_punctuator(&self) -> bool {
        self.punctuator.is_some()
    }

    /// Reject a [`TranscribeOverrides`] that turns a knob on without the backing
    /// resource loaded. Call this *before* checking out a session so the request
    /// fails fast (the REST layer maps the error to a `409`). Turning a knob off
    /// (`Some(false)`) and ITN in either direction are always valid — ITN is pure
    /// code with no model to load.
    ///
    /// Hotword DoS limits are validated separately via
    /// [`Engine::validate_hotwords`] so [`TranscribeOverrides`] stays `Copy` (semver).
    ///
    /// # Errors
    ///
    /// - [`OverrideError::VadNotLoaded`] when `vad = Some(true)` but no VAD is
    ///   attached.
    /// - [`OverrideError::PunctuationNotAvailable`] when `punctuation = Some(true)`
    ///   but no punctuator is attached.
    pub fn validate_overrides(&self, o: &TranscribeOverrides) -> Result<(), OverrideError> {
        if o.vad == Some(true) && self.vad.is_none() {
            return Err(OverrideError::VadNotLoaded);
        }
        if o.punctuation == Some(true) && self.punctuator.is_none() {
            return Err(OverrideError::PunctuationNotAvailable);
        }
        Ok(())
    }

    /// Reject a [`HotwordOverride`] that exceeds DoS limits. Call before checkout
    /// so oversized requests fail fast (REST maps to HTTP 400).
    ///
    /// # Errors
    ///
    /// - [`HotwordError::TooManyHotwords`] when more than
    ///   [`MAX_HOTWORDS_PER_REQUEST`] phrases are supplied.
    /// - [`HotwordError::PhraseTooLong`] when any phrase exceeds
    ///   [`MAX_HOTWORD_PHRASE_CHARS`] Unicode scalar values.
    pub fn validate_hotwords(&self, hw: &HotwordOverride) -> Result<(), HotwordError> {
        if hw.phrases.len() > MAX_HOTWORDS_PER_REQUEST {
            return Err(HotwordError::TooManyHotwords);
        }
        for phrase in &hw.phrases {
            if phrase.chars().count() > MAX_HOTWORD_PHRASE_CHARS {
                return Err(HotwordError::PhraseTooLong);
            }
        }
        Ok(())
    }

    /// Build a temporary per-request [`bias::Biaser`] from a
    /// [`HotwordOverride`]. Empty phrases force biasing off (`None`).
    /// Unrepresentable phrases are dropped by [`Biaser::from_phrases`]; if none
    /// survive, returns `None` (decode continues without biasing).
    pub(crate) fn build_request_biaser(&self, hw: &HotwordOverride) -> Option<bias::Biaser> {
        if hw.phrases.is_empty() {
            return None;
        }
        let boost = hw.boost.unwrap_or(DEFAULT_HOTWORDS_BOOST);
        let pairs: Vec<(String, f32)> = hw.phrases.iter().cloned().map(|p| (p, 1.0)).collect();
        bias::Biaser::from_phrases(&self.tokenizer, &pairs, boost)
    }

    /// Effective hotword biaser for one decode call.
    ///
    /// - `hotwords == None` → engine boot biaser (may itself be `None`)
    /// - `hotwords == Some(_)` → temporary request biaser (may be `None` when
    ///   phrases are empty / unrepresentable — deliberately *not* the boot biaser)
    ///
    /// `request_biaser` must be the result of [`build_request_biaser`] (or
    /// `None` when `hotwords` is absent); it is only borrowed, never moved.
    pub(crate) fn select_biaser<'a>(
        &'a self,
        hotwords: Option<&HotwordOverride>,
        request_biaser: &'a Option<bias::Biaser>,
    ) -> Option<&'a bias::Biaser> {
        match hotwords {
            None => self.biaser.as_ref(),
            Some(_) => request_biaser.as_ref(),
        }
    }

    /// Apply ITN then punctuation to joined text. Shared by the streaming
    /// finalizer and the file-transcription result builder so the order and
    /// guards stay identical.
    pub(crate) fn apply_text_postprocess(
        &self,
        text: String,
        itn: bool,
        punctuation: bool,
    ) -> String {
        let text = if itn {
            crate::itn::apply_itn(&text)
        } else {
            text
        };
        match &self.punctuator {
            Some(p) if punctuation => p.restore(&text),
            _ => text,
        }
    }

    /// Enable or disable inverse text normalization (Russian number-words →
    /// digits) on file-transcription output and finalized streaming segments,
    /// consuming and returning `self` (builder style). When enabled, ITN runs
    /// *before* the punctuation pass so the restorer cases the
    /// already-digitized text.
    pub fn with_itn(mut self, enabled: bool) -> Self {
        self.itn = enabled;
        self
    }

    /// Whether inverse text normalization is enabled.
    pub fn has_itn(&self) -> bool {
        self.itn
    }

    /// Attach a contextual hotword biaser built from `(phrase, weight)` pairs
    /// and an additive `boost`, consuming and returning `self` (builder style).
    /// Each phrase is tokenized with the engine's own `Tokenizer`, so biasing
    /// adapts to whichever recognition head is loaded.
    ///
    /// When `phrases` is empty, `boost <= 0`, or no phrase is representable in
    /// the active vocab, the biaser resolves to `None` and the decode path stays
    /// byte-for-byte unchanged. Replaces any previously attached biaser.
    pub fn with_hotwords(mut self, phrases: &[(String, f32)], boost: f32) -> Self {
        self.biaser = if phrases.is_empty() {
            None
        } else {
            bias::Biaser::from_phrases(&self.tokenizer, phrases, boost)
        };
        if let Some(b) = &self.biaser {
            tracing::info!(
                "Hotword biasing enabled ({} phrase(s), boost {boost})",
                b.phrase_count()
            );
        }
        self
    }

    /// Whether a hotword biaser is attached (biasing active).
    pub fn has_hotwords(&self) -> bool {
        self.biaser.is_some()
    }

    /// Attach an optional Silero VAD plus its config, consuming and returning
    /// `self` (builder style). Pass `None` for no VAD (the default): file
    /// transcription then decodes the whole buffer and streaming endpointing is
    /// byte-for-byte unchanged. When set, file transcription skips silence and
    /// streaming endpointing is owned by VAD-detected trailing silence (the
    /// decoder's blank-run heuristic is ignored).
    pub fn with_vad(
        mut self,
        vad: Option<crate::vad::SileroVad>,
        config: crate::vad::VadConfig,
    ) -> Self {
        self.vad = vad;
        self.vad_config = config;
        if self.vad.is_some() {
            tracing::info!(
                "VAD enabled (threshold {}, min_silence {}ms)",
                self.vad_config.threshold,
                self.vad_config.min_silence_ms
            );
        }
        self
    }

    /// Whether a VAD is attached (silence skipping / VAD endpointing active).
    pub fn has_vad(&self) -> bool {
        self.vad.is_some()
    }

    /// Boot-time VAD config (threshold, min silence, …). Useful when recreating
    /// a per-session endpointer after `configure.min_silence_ms`.
    pub fn vad_config(&self) -> &crate::vad::VadConfig {
        &self.vad_config
    }

    /// Boot-time streaming endpoint mode for new sessions.
    pub fn endpoint_mode(&self) -> EndpointMode {
        self.endpoint_mode
    }

    /// Set the default streaming utterance-end policy for new sessions.
    pub fn with_endpoint_mode(mut self, mode: EndpointMode) -> Self {
        self.endpoint_mode = mode;
        self
    }

    /// Set the max retained streaming encoder window in seconds (default 2.5).
    /// The value is clamped to the supported range (2.4–30 s; see
    /// [`Engine::stream_max_window_samples`]). Longer windows
    /// improve streaming WER on phrases that previously slid at the cap, at a
    /// linear per-stride encoder-cost increase.
    pub fn with_stream_max_window_secs(mut self, secs: f64) -> Self {
        let samples = crate::inference::windows::stream_max_window_samples(secs);
        if samples != (secs * 16000.0).round() as usize {
            tracing::warn!(
                "stream max window {secs}s clamped to {:.1}s",
                samples as f64 / 16000.0
            );
        }
        self.stream_max_window_samples = samples;
        self
    }

    /// Resolved max streaming encoder window (samples @16kHz).
    pub fn stream_max_window_samples(&self) -> usize {
        self.stream_max_window_samples
    }

    /// Enable stable-prefix commits at the streaming window cap: instead of
    /// committing the whole live tail when the window slides, only the prefix
    /// that two consecutive window hypotheses agree on becomes stable (minus a
    /// 0.5 s commit horizon at the window edge); the rest of the tail stays
    /// revisable by later decodes. Bounds long-phrase WER loss at slide
    /// boundaries without widening the window. Off by default.
    pub fn with_stream_stable_prefix(mut self, enabled: bool) -> Self {
        self.stream_stable_prefix = enabled;
        self
    }

    /// Max pooled triplets one file decode may hold for overlapping-window
    /// parallelism (including the caller's already-checked-out slot). `1`
    /// (the default) keeps the serial loop. Values below 1 are clamped to 1.
    pub fn with_file_window_concurrency(mut self, n: usize) -> Self {
        self.file_window_concurrency = n.max(1);
        self
    }

    /// Resolved file-window concurrency cap (`>= 1`).
    pub fn file_window_concurrency(&self) -> usize {
        self.file_window_concurrency.max(1)
    }

    /// Whether stable-prefix commits are enabled.
    pub fn has_stream_stable_prefix(&self) -> bool {
        self.stream_stable_prefix
    }

    /// Size of the BPE vocabulary the loaded tokenizer covers. Exposed so the
    /// REST `/v1/models` handler can report the real value instead of a
    /// hardcoded literal that would drift if the upstream model rev changes.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }
}
