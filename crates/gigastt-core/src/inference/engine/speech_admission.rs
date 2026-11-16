//! Optional CTC admission and automatic endpoint confirmation for RNNT streaming.
use super::*;

#[derive(Clone, Copy)]
pub(super) struct CtcObservation {
    pub(super) all_blank: bool,
    pub(super) trailing_blanks: usize,
}

impl Engine {
    /// Attach a cached CPU INT8 multilingual-large CTC engine to RNNT streaming.
    ///
    /// File transcription is unchanged. Both engines must have one interactive
    /// slot and no batch pool. Callers must load both with one encoder thread.
    /// The attachment is opt-in and consumes the sidecar; it performs no download.
    ///
    /// # Errors
    /// Returns [`GigasttError::ModelLoad`] for incompatible engines or an existing
    /// attachment. Validation leaves the recipient unchanged on failure.
    pub fn attach_ctc_speech_admission(&mut self, sidecar: Engine) -> Result<(), GigasttError> {
        let error = |message: &str| GigasttError::ModelLoad {
            path: "CTC streaming attachment".into(),
            source: Some(anyhow::anyhow!("{message}").into()),
        };
        if self.speech_admission.is_some() || sidecar.speech_admission.is_some() {
            return Err(error("CTC streaming attachment already present"));
        }
        if self.variant != ModelVariant::Rnnt || !self.int8 || self.execution_provider != "cpu" {
            return Err(error("CTC streaming confirmation requires CPU INT8 RNNT"));
        }
        if sidecar.variant != ModelVariant::MlCtcLarge
            || !sidecar.int8
            || sidecar.execution_provider != "cpu"
            || sidecar.tokenizer.vocab_size() != 71
            || sidecar.tokenizer.blank_id() != 70
        {
            return Err(error(
                "CTC streaming confirmation requires CPU INT8 ml_ctc_large",
            ));
        }
        if self.pool.total() != 1
            || sidecar.pool.total() != 1
            || self.batch_pool.is_some()
            || sidecar.batch_pool.is_some()
        {
            return Err(error(
                "CTC streaming confirmation requires single-slot engines without batch pools",
            ));
        }
        self.speech_admission = Some(Box::new(sidecar));
        Ok(())
    }
}

impl Engine {
    pub(super) fn speech_admission_attached(&self) -> bool {
        self.speech_admission.is_some()
    }

    pub(super) fn empty_speech_observation(
        &self,
        state: &StreamingState,
        num_frames: usize,
    ) -> anyhow::Result<Option<CtcObservation>> {
        if !state.assembler.is_empty() {
            if self.speech_admission_attached() {
                #[cfg(feature = "__internals")]
                tracing::debug!("speech_admission_nonempty_bypass");
            }
            return Ok(None);
        }
        self.observe_ctc_window(state, num_frames, "admission")
    }

    pub(super) fn observe_ctc_window(
        &self,
        state: &StreamingState,
        num_frames: usize,
        purpose: &'static str,
    ) -> anyhow::Result<Option<CtcObservation>> {
        {
            let _ = purpose;
            let Some(sidecar) = self.speech_admission.as_ref() else {
                return Ok(None);
            };
            let mut session = sidecar
                .pool
                .checkout_blocking()
                .context("CTC admission pool unavailable")?;
            session.encoder_inputs[0].resize_to(Shape::new(vec![1, N_MELS, num_frames]));
            session.encoder_inputs[0]
                .as_f32_mut()
                .context("CTC admission input is not f32")?
                .copy_from_slice(&state.mel_output);
            session.encoder_inputs[1]
                .as_i64_mut()
                .context("CTC admission length is not i64")?[0] = i64::try_from(num_frames)?;
            let outputs = session
                .encoder
                .run_low_latency(&session.encoder_inputs)
                .context("CTC admission inference failed")?;
            anyhow::ensure!(outputs.len() == 2, "CTC admission requires two outputs");
            let length = match outputs[1].view().data() {
                TensorDataView::I64(v) if v.len() == 1 => usize::try_from(v[0])?,
                TensorDataView::I32(v) if v.len() == 1 => usize::try_from(v[0])?,
                _ => anyhow::bail!("CTC admission invalid length tensor"),
            };
            let shape = outputs[0].shape().dims();
            anyhow::ensure!(
                shape.len() == 3
                    && shape[0] == 1
                    && shape[2] == 71
                    && length > 0
                    && length <= shape[1],
                "CTC admission invalid logits shape/length"
            );
            let logits = outputs[0]
                .view()
                .data()
                .as_f32()
                .context("CTC admission logits are not f32")?;
            let valid = logits
                .get(
                    ..length
                        .checked_mul(71)
                        .context("CTC admission length overflow")?,
                )
                .context("CTC admission truncated logits")?;
            anyhow::ensure!(
                valid.iter().all(|v| v.is_finite()),
                "CTC admission nonfinite logits"
            );
            let mut blanks = 0;
            let mut trailing_blanks = 0;
            for row in valid.as_chunks::<71>().0 {
                if decode::argmax(row, 70) == 70 {
                    blanks += 1;
                    trailing_blanks += 1;
                } else {
                    trailing_blanks = 0;
                }
            }
            #[cfg(feature = "__internals")]
            tracing::info!(
                window_start_samples = state.window_start_samples,
                window_samples = state.audio_buffer.len(),
                context_samples = state.context_samples,
                pending_samples = state.pending_samples,
                frames = length,
                blanks,
                purpose,
                trailing_blanks,
                veto = purpose == "admission" && blanks == length,
                "speech_admission"
            );
            Ok(Some(CtcObservation {
                all_blank: blanks == length,
                trailing_blanks,
            }))
        }
    }

    pub(super) fn slide_uncommitted_speech_window(state: &mut StreamingState) {
        let keep = STREAM_LEFT_CONTEXT_SAMPLES.min(state.audio_buffer.len());
        let drop = state.audio_buffer.len() - keep;
        let old_context = state.context_samples;
        audio::consume_audio_buffer(&mut state.audio_buffer, drop);
        state.window_start_samples += drop;
        state.context_samples = old_context.saturating_sub(drop);
        #[cfg(feature = "__internals")]
        tracing::info!(
            drop,
            keep,
            old_context,
            context_samples = state.context_samples,
            window_start_samples = state.window_start_samples,
            "speech_admission_idle_slide"
        );
    }
}
