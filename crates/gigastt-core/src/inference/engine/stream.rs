//! `impl Engine` methods — split out of the former god-file.
use super::*;
impl Engine {
    /// Return `true` if a speaker model file was present at boot and diarization
    /// can be requested. The ONNX session may still be unloaded until the first
    /// diarization request (lazy load).
    #[cfg(feature = "diarization")]
    pub fn has_speaker_encoder(&self) -> bool {
        self.speaker_encoder.is_some()
    }

    /// Create a fresh streaming state for a new connection.
    ///
    /// Pass `diarization_enabled = true` to activate speaker diarization for
    /// this session. Without the `diarization` feature or a speaker model file,
    /// the flag is silently ignored (a `warn!` is emitted when the caller asked
    /// for diarization but the build does not support it, so the contract
    /// mismatch is visible in logs). Enabling diarization loads the speaker
    /// encoder on first use if it was only probed at boot.
    pub fn create_state(&self, diarization_enabled: bool) -> StreamingState {
        #[cfg(feature = "diarization")]
        let diarization_state = if diarization_enabled {
            match self.speaker_encoder.as_ref() {
                Some(lazy) => lazy
                    .get_or_load()
                    .and_then(|enc| diarization::open_streaming(&enc)),
                None => {
                    tracing::warn!(
                        "diarization_enabled=true ignored: wespeaker model not present at engine boot"
                    );
                    None
                }
            }
        } else {
            None
        };

        #[cfg(not(feature = "diarization"))]
        if diarization_enabled {
            tracing::warn!(
                "diarization_enabled=true ignored: build lacks the `diarization` feature"
            );
        }

        StreamingState {
            commit_policy: CommitPolicy::Auto,
            abort: None,
            partial: None,
            failed: false,
            decoder: DecoderState::new(self.tokenizer.blank_id()),
            audio_buffer: Vec::new(),
            assembler: TranscriptAssembler::new(),
            window_start_samples: 0,
            context_samples: 0,
            pending_samples: 0,
            resampler: None,
            mel_fft_input: Vec::new(),
            mel_power: Vec::new(),
            mel_output: Vec::new(),
            resample_output_buf: Vec::new(),
            vad_endpointer: self
                .vad
                .as_ref()
                .map(|_| crate::vad::VadEndpointer::new(&self.vad_config)),
            punctuation: None,
            itn: None,
            endpoint_mode: self.endpoint_mode,
            agreed_prefix: 0,
            cap_streak: 0,
            #[cfg(feature = "diarization")]
            diarization_state,
        }
    }

    /// Process a chunk of 16kHz f32 audio samples and return any new transcript segments.
    ///
    /// Returns [`TranscriptSegment`] with `is_final == false` during speech (Partial),
    /// and `is_final == true` only on a **true utterance endpoint**:
    ///
    /// - decoder blank-run (~600 ms) when no VAD and [`EndpointMode::Auto`];
    /// - VAD trailing silence when a VAD is attached (and mode is not `Manual`);
    /// - never on the ~2.5 s encoder window cap (cap commits a stable prefix and
    ///   emits a non-final partial so voice assistants do not treat a slide as
    ///   "command complete").
    ///
    /// Streaming state (LSTM hidden/cell, leftover audio, accumulated text) is
    /// maintained in `state`.
    ///
    /// # Errors
    ///
    /// Returns [`GigasttError::Inference`] if the ONNX runtime fails.
    pub fn process_chunk(
        &self,
        samples: &[f32],
        state: &mut StreamingState,
        triplet: &mut SessionTriplet,
    ) -> Result<Vec<TranscriptSegment>, GigasttError> {
        if state.abort_requested() {
            state.failed = true;
            self.publish_stream_partial(state);
            return Err(GigasttError::Cancelled);
        }
        if samples.is_empty() {
            return Ok(vec![]);
        }

        // Diarization tracks speakers continuously, so feed every chunk's audio
        // even when this chunk doesn't trigger a decode (see the stride gate).
        #[cfg(feature = "diarization")]
        if let Some(dia) = state.diarization_state.as_mut() {
            diarization::feed_chunk(dia, samples);
        }

        // Sliding-window streaming: accumulate audio; the encoder re-runs on the
        // whole retained window so the offline Conformer always has left context
        // (an isolated ~100ms chunk decodes to garbage). Re-decoding is the cost,
        // so we only decode once STREAM_DECODE_STRIDE_SAMPLES of NEW audio have
        // arrived (or the window hit its cap) — this keeps the engine real-time.
        // The window is bounded by `self.stream_max_window_samples` (default
        // 2.5s, configurable at serve time); on endpoint or cap we finalize the
        // tail and slide, retaining STREAM_LEFT_CONTEXT_SAMPLES.
        state.audio_buffer.extend_from_slice(samples);
        state.pending_samples += samples.len();

        // Feed the VAD on every chunk so trailing silence is tracked
        // continuously (independent of the decode stride). A VAD endpoint forces
        // a decode + finalize this chunk even if the stride gate wouldn't fire.
        // VAD is non-blocking: an inference error is logged and ignored, leaving
        // the window cap as the only backstop until the VAD recovers. With no
        // VAD attached `vad_endpoint` is always false and the decoder's
        // blank-run heuristic keeps owning endpointing, byte-for-byte unchanged.
        let mut vad_endpoint = false;
        if let (Some(vad), Some(ep)) = (self.vad.as_ref(), state.vad_endpointer.as_mut()) {
            match ep.push(vad, samples) {
                Ok(fired) => vad_endpoint = fired,
                Err(e) => tracing::warn!("VAD endpoint detection failed: {e:#}"),
            }
        }

        let over_cap = state.audio_buffer.len() >= self.stream_max_window_samples;
        // Stride gate on NEW audio since the last decode (not since the last
        // slide): otherwise a non-finalizing partial would leave the counter
        // high and decode on every subsequent chunk. A VAD endpoint overrides
        // the gate so the utterance finalizes promptly. The cap forces an
        // early decode only in legacy commit mode; with stable-prefix commits
        // an over-cap buffer just waits for the next stride (bounded by the
        // stride itself), avoiding an 8x decode rate during cap saturation.
        let cap_forces_decode = over_cap && !self.stable_prefix_enabled(state);
        if state.pending_samples < STREAM_DECODE_STRIDE_SAMPLES
            && !cap_forces_decode
            && !vad_endpoint
        {
            return Ok(vec![]);
        }
        // Too little audio to extract a frame. Skip — but never when finalizing:
        // a fired VAD endpoint (or cap) must still flush the assembler below,
        // even though `decode_window` will add no new words from a sub-frame
        // buffer. (In practice a VAD endpoint needs ≥ min_silence_ms of trailing
        // audio, so the buffer is always ≫ N_FFT here; this guards the edge.)
        if state.audio_buffer.len() < N_FFT && !vad_endpoint && !over_cap {
            return Ok(vec![]);
        }

        let decoded = self.decode_window(state, triplet);
        self.publish_stream_partial(state);
        if state.abort_requested() {
            state.failed = true;
            return Err(GigasttError::Cancelled);
        }
        let endpoint = decoded.map_err(|e| GigasttError::Inference { source: e.into() })?;
        state.pending_samples = 0;
        let ts = now_timestamp();

        // True utterance endpoints only — never the encoder window cap.
        // Cap used to emit `final` as a "backstop", which voice assistants
        // (Irene) treated as "command complete" mid-phrase; it now commits a
        // stable prefix and emits a non-final partial instead.
        let speech_endpoint = Self::speech_endpoint(
            state.endpoint_mode,
            endpoint,
            vad_endpoint,
            state.vad_endpointer.is_some(),
        );

        if speech_endpoint {
            let reason = if vad_endpoint {
                EndpointReason::Vad
            } else {
                EndpointReason::Blank
            };
            let seg = self.finalize_stream_segment(state, ts, reason);
            Self::slide_streaming_window(state);
            if seg.text.trim().is_empty() {
                return Ok(vec![]);
            }
            return Ok(vec![seg]);
        }

        if over_cap {
            // Encoder cost bound: commit live words so they are not lost when
            // the window slides, but do **not** end the utterance.
            let live = state.assembler.live_word_count();
            let committed = if self.stable_prefix_enabled(state) {
                Self::cap_commit_stable_prefix(state)
            } else {
                state.cap_streak = 0;
                state.assembler.commit_live();
                live
            };
            tracing::debug!(
                committed,
                live,
                agreed = state.agreed_prefix,
                streak = state.cap_streak,
                window_samples = state.audio_buffer.len(),
                stable = self.stable_prefix_enabled(state),
                "stream window cap: committed prefix, sliding"
            );
            if self.stable_prefix_enabled(state) {
                // Slide only when words were actually committed: the anchored
                // slide drops audio before the stable prefix's coverage end,
                // and sliding without a commit would cut audio under the live
                // (uncommitted) tail — the next decode would suppress those
                // words and lose them permanently. With no commit the buffer
                // simply waits for agreement (bounded by the cap streak).
                // Exception: an empty tail (silence) slides as before, or a
                // long silent stream would grow the buffer unboundedly.
                if committed > 0 {
                    let first_live = state.assembler.live_words().first().map(|w| w.start);
                    Self::slide_streaming_window_anchored(state, first_live);
                } else if state.assembler.live_word_count() == 0 {
                    Self::slide_streaming_window(state);
                }
            } else {
                Self::slide_streaming_window(state);
            }
            self.publish_stream_partial(state);
            if state.assembler.is_empty() {
                return Ok(vec![]);
            }
            return Ok(vec![self.stream_partial(state, ts)]);
        }

        if state.assembler.is_empty() {
            return Ok(vec![]);
        }
        Ok(vec![self.stream_partial(state, ts)])
    }

    /// Whether this decode should close the utterance (`final` / `speech_final`).
    ///
    /// Pure helper so endpoint policy is unit-testable without ONNX.
    pub(crate) fn speech_endpoint(
        mode: EndpointMode,
        decoder_blank_endpoint: bool,
        vad_endpoint: bool,
        vad_attached: bool,
    ) -> bool {
        let decoder_endpoint = decoder_blank_endpoint && !vad_attached;
        match mode {
            EndpointMode::Auto => decoder_endpoint || vad_endpoint,
            // Assistant: only VAD silence (when attached). Blank-run alone is too
            // aggressive for multi-word voice commands.
            EndpointMode::Assistant => vad_endpoint,
            EndpointMode::Manual => false,
        }
    }

    /// Slide the streaming audio window, retaining left context for the next decode.
    pub(crate) fn slide_streaming_window(state: &mut StreamingState) {
        let keep = STREAM_LEFT_CONTEXT_SAMPLES.min(state.audio_buffer.len());
        let slide_off = state.audio_buffer.len() - keep;
        if slide_off > 0 {
            audio::consume_audio_buffer(&mut state.audio_buffer, slide_off);
            state.window_start_samples += slide_off;
        }
        state.context_samples = keep;
    }

    /// Stable-prefix commit at the window cap: commit only the prefix that the
    /// last two window hypotheses agreed on, minus a commit horizon at the
    /// window's right edge (words decoded from the edge may still be revised by
    /// the next decode — or truncated mid-word by the buffer edge). Cap hits
    /// with nothing committable are counted; at [`STREAM_CAP_STREAK_MAX`] the
    /// committable prefix is taken even without agreement so the retained
    /// buffer stays bounded — but the horizon is still respected: committing a
    /// word the buffer edge may have truncated locks the truncated form in
    /// permanently. Returns the number of committed words. The caller must NOT
    /// slide the window when this returns 0 — the uncommitted live words'
    /// audio must stay in the window, or the next decode would suppress them
    /// and lose them permanently.
    pub(crate) fn cap_commit_stable_prefix(state: &mut StreamingState) -> usize {
        let live = state.assembler.live_words();
        let edge_s = (state.window_start_samples + state.audio_buffer.len()) as f64 / 16000.0;
        let horizon_s = edge_s - STREAM_COMMIT_HORIZON_SECS;
        // Beyond agreement, never commit a word ending within the horizon of
        // the buffer's right edge — it may be mid-word, decoded from incomplete
        // audio ("разва" instead of "развалины").
        let mut n = state.agreed_prefix.min(live.len());
        while n > 0 && live[n - 1].end > horizon_s {
            n -= 1;
        }
        if n == 0 && !live.is_empty() {
            state.cap_streak += 1;
            if state.cap_streak >= STREAM_CAP_STREAK_MAX {
                // Boundedness fallback: commit the pre-horizon prefix even
                // without agreement. Edge words keep waiting; the moving buffer
                // edge takes them out of the horizon within ~0.5 s of audio.
                while n < live.len() && live[n].end <= horizon_s {
                    n += 1;
                }
                state.cap_streak = 0;
            }
        } else {
            state.cap_streak = 0;
        }
        state.assembler.commit_prefix(n)
    }

    /// Anchored variant of [`Self::slide_streaming_window`] for stable-prefix
    /// commits: the retained window starts `STREAM_LEFT_CONTEXT_SAMPLES` before
    /// the stable prefix's coverage end (not a fixed 1.5 s off the tail), and
    /// never later than the first still-live word's start — starting the window
    /// mid-word makes the decoder emit a truncated first word. The suppression
    /// boundary lands exactly on the committed coverage end. Call only after at
    /// least one word was committed this pass; without a fresh anchor the
    /// buffer must be kept intact (uncommitted live words would otherwise slide
    /// out of the window and be suppressed on the next decode).
    pub(crate) fn slide_streaming_window_anchored(
        state: &mut StreamingState,
        first_live_start_s: Option<f64>,
    ) {
        let Some(end_s) = state.assembler.committed_coverage_end() else {
            return;
        };
        let boundary = (end_s * 16000.0).round() as usize;
        if boundary <= state.window_start_samples {
            return;
        }
        let mut new_start = boundary
            .saturating_sub(STREAM_LEFT_CONTEXT_SAMPLES)
            .max(state.window_start_samples);
        if let Some(start_s) = first_live_start_s {
            let first_live = (start_s * 16000.0).floor() as usize;
            new_start = new_start.min(first_live);
        }
        let slide_off = new_start - state.window_start_samples;
        if slide_off > 0 {
            audio::consume_audio_buffer(&mut state.audio_buffer, slide_off);
            state.window_start_samples += slide_off;
        }
        state.context_samples = boundary - new_start;
    }

    /// Re-decode the whole retained window from a fresh decoder state and update
    /// the assembler with the context-suppressed tail. Returns whether the
    /// decoder detected an endpoint. Shared by [`Engine::process_chunk`]
    /// (strided) and [`Engine::finish_stream`] (forced at end of stream).
    pub(crate) fn decode_window(
        &self,
        state: &mut StreamingState,
        triplet: &mut SessionTriplet,
    ) -> anyhow::Result<bool> {
        if state.abort_requested() {
            state.failed = true;
            return Ok(false);
        }
        let mel_start = std::time::Instant::now();
        let num_frames = self.features.compute_mel(
            &state.audio_buffer,
            &mut state.mel_fft_input,
            &mut state.mel_power,
            &mut state.mel_output,
        );
        tracing::debug!(
            elapsed_us = mel_start.elapsed().as_micros() as u64,
            "mel_compute"
        );
        if num_frames == 0 {
            return Ok(false);
        }

        // Encoder-frame offset of the window start (drift-free: a single division
        // over the cumulative slid-off sample count).
        let frame_offset = state.window_start_samples / (HOP_LENGTH * ENCODER_SUBSAMPLING);

        // The window overlaps the previous one, so persisting the LSTM state
        // would double-condition the prediction network — decode fresh.
        let mut decoder_state = DecoderState::new(self.tokenizer.blank_id());
        // Streaming always uses the engine boot biaser (per-request hotwords
        // apply to file-transcription paths that carry TranscribeOverrides).
        let abort = || {
            state
                .abort
                .as_ref()
                .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Relaxed))
        };
        let (all_words, endpoint) = self.run_inference(
            triplet,
            &state.mel_output[..],
            num_frames,
            &mut decoder_state,
            frame_offset,
            true, // streaming: ANE low-latency pad floor when available
            self.biaser.as_ref(),
            Some(&abort),
        )?;

        // Suppress words inside the already-emitted left context so a slid
        // window does not re-emit committed words. Legacy mode suppresses by
        // word start; stable-prefix mode suppresses only words FULLY covered
        // by the committed region (end <= boundary) — a word whose timestamp
        // straddles the boundary was never committed, and dropping it by start
        // would lose it permanently at the seam. Boundary 0 means nothing was
        // ever committed or slid, so nothing is suppressed.
        let window_start_s = frame_offset as f64 * SECONDS_PER_FRAME;
        let context_boundary_s = window_start_s + state.context_samples as f64 / 16000.0;
        let stable = self.stable_prefix_enabled(state);
        let decoded = all_words.len();
        #[cfg_attr(not(feature = "diarization"), allow(unused_mut))]
        let mut tail: Vec<WordInfo> = all_words
            .into_iter()
            .filter(|w| {
                if stable {
                    context_boundary_s <= 0.0 || w.end > context_boundary_s + f64::EPSILON
                } else {
                    w.start + f64::EPSILON >= context_boundary_s
                }
            })
            .collect();

        // Seam dedup: a committed word whose timing drifted past the boundary
        // in this decode would be re-emitted as a duplicate of the last
        // committed word — drop the leading re-emission (same text, starting
        // within a frame-scale window after the boundary).
        if stable
            && let Some(last) = state.assembler.committed_last()
            && let Some(first) = tail.first()
            && first.word == last.word
            && first.start < context_boundary_s + 0.1
        {
            tail.remove(0);
        }

        // Hypothesis stability: longest text-equal prefix shared by the
        // previous live tail and this fresh hypothesis. Used by the
        // stable-prefix commit at the window cap.
        let agreed = state
            .assembler
            .live_words()
            .iter()
            .zip(tail.iter())
            .take_while(|(a, b)| a.word == b.word)
            .count();
        state.agreed_prefix = agreed;

        // Per-pass visibility for stream-vs-file divergence analysis:
        // decoded = full window hypothesis, suppressed = dropped context words,
        // replaced = previous live tail this hypothesis overwrites.
        tracing::debug!(
            decoded,
            suppressed = decoded - tail.len(),
            replaced = state.assembler.live_word_count(),
            agreed,
            live = tail.len(),
            "stream decode window"
        );

        #[cfg(feature = "diarization")]
        if let Some(dia) = state.diarization_state.as_mut()
            && let Some(speaker) = diarization::last_turn_speaker(dia)
        {
            for w in &mut tail {
                w.speaker = Some(speaker);
            }
        }

        // A cancelled re-decode may cover less audio than the previous live
        // hypothesis. Keep that readable tail instead of replacing it by empty
        // or shorter text. The committed prefix is never modified here.
        if !state.abort_requested() || tail.len() > state.assembler.live_word_count() {
            state.assembler.set_words(tail);
        }
        Ok(endpoint)
    }

    fn publish_stream_partial(&self, state: &StreamingState) {
        if let Some(partial) = &state.partial {
            partial.store(self.stream_partial(state, now_timestamp()));
        }
    }

    /// Decode any audio buffered since the last strided decode, then finalize.
    /// Call when the stream ends (Stop / EOF) so the decode-stride batching does
    /// not drop trailing words. Best-effort: on decode failure, falls back to a
    /// plain flush of whatever the assembler already holds.
    pub fn finish_stream(
        &self,
        state: &mut StreamingState,
        triplet: &mut SessionTriplet,
    ) -> Option<TranscriptSegment> {
        if state.abort_requested() {
            state.failed = true;
            self.publish_stream_partial(state);
            return (!state.assembler.is_empty())
                .then(|| self.stream_partial(state, now_timestamp()));
        }
        let has_pending = state.pending_samples > 0 && state.audio_buffer.len() >= N_FFT;
        if has_pending && let Err(e) = self.decode_window(state, triplet) {
            tracing::warn!("finish_stream decode failed: {e:#}");
        }
        if state.abort_requested() {
            state.failed = true;
            self.publish_stream_partial(state);
            return (!state.assembler.is_empty())
                .then(|| self.stream_partial(state, now_timestamp()));
        }
        self.flush_state(state)
    }

    /// Flush accumulated text as a Final segment (called on Stop/Close).
    pub fn flush_state(&self, state: &mut StreamingState) -> Option<TranscriptSegment> {
        if state.abort_requested() {
            state.failed = true;
            self.publish_stream_partial(state);
            return (!state.assembler.is_empty())
                .then(|| self.stream_partial(state, now_timestamp()));
        }
        if state.assembler.is_empty() {
            return None;
        }
        Some(self.finalize_stream_segment(state, now_timestamp(), EndpointReason::Stop))
    }

    fn stable_prefix_enabled(&self, state: &StreamingState) -> bool {
        self.stream_stable_prefix || state.commit_policy == CommitPolicy::StablePrefix
    }

    fn resolved_commit_policy(&self, state: &StreamingState) -> CommitPolicy {
        match state.commit_policy {
            CommitPolicy::Auto
                if state.itn.unwrap_or(self.itn)
                    || (state.punctuation.unwrap_or(true) && self.has_punctuator()) =>
            {
                CommitPolicy::OnFinalize
            }
            CommitPolicy::Auto => CommitPolicy::StablePrefix,
            policy => policy,
        }
    }

    /// Window slides still retain an internal prefix under ON_FINALIZE; only
    /// its public commitment is deferred, keeping the encoder buffer bounded.
    pub(super) fn stream_partial(
        &self,
        state: &StreamingState,
        timestamp: f64,
    ) -> TranscriptSegment {
        let mut segment = state.assembler.partial(timestamp);
        if self.resolved_commit_policy(state) == CommitPolicy::OnFinalize {
            segment.committed.clear();
            segment.tentative = segment.text.clone();
        }
        segment
    }

    /// Commit the remaining tail at a true endpoint. Post-processing may only
    /// rewrite tentative bytes; AUTO defers the entire utterance when ITN or
    /// punctuation is enabled, preserving the historical final text exactly.
    fn finalize_stream_segment(
        &self,
        state: &mut StreamingState,
        timestamp: f64,
        reason: EndpointReason,
    ) -> TranscriptSegment {
        let partial = self.stream_partial(state, timestamp);
        let mut segment = state.assembler.finalize_with_reason(timestamp, reason);
        let tail = self.apply_text_postprocess(
            partial.tentative.trim_start().to_owned(),
            state.itn.unwrap_or(self.itn),
            state.punctuation.unwrap_or(true),
        );
        segment.text = if partial.committed.is_empty() {
            tail
        } else if tail.is_empty() {
            partial.committed
        } else {
            format!("{} {}", partial.committed, tail)
        };
        segment.committed = segment.text.clone();
        segment.tentative.clear();
        if let Some(snapshot) = &state.partial {
            snapshot.clear();
        }
        segment
    }
}
