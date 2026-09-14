//! `impl Engine` methods — split out of the former god-file.
use super::*;
impl Engine {
    /// Run encoder + decode. `low_latency` selects the streaming encoder path
    /// ([`RuntimeSession::run_low_latency`]) so ANE can pad underfilled short
    /// windows; file mode keeps the calibrated 0.5 fill floor. `biaser` is the
    /// effective per-call hotword biaser (boot, request override, or off).
    #[allow(clippy::too_many_arguments)] // encoder/decode call site; bundle later if it grows again
    pub(crate) fn run_inference(
        &self,
        triplet: &mut SessionTriplet,
        features: &[f32],
        num_frames: usize,
        decoder_state: &mut DecoderState,
        frame_offset: usize,
        low_latency: bool,
        biaser: Option<&bias::Biaser>,
        abort: Option<&(dyn Fn() -> bool + Sync)>,
    ) -> anyhow::Result<(Vec<WordInfo>, bool)> {
        if abort.is_some_and(|abort| abort()) {
            return Ok((Vec::new(), false));
        }
        // Reuse the encoder input tensors: resize the signal tensor to the
        // current frame count and overwrite both buffers in place.
        triplet.encoder_inputs[0].resize_to(Shape::new(vec![1, N_MELS, num_frames]));
        triplet.encoder_inputs[0]
            .as_f32_mut()
            .context("encoder signal tensor is not f32")?
            .copy_from_slice(features);
        triplet.encoder_inputs[1]
            .as_i64_mut()
            .context("encoder length tensor is not i64")?[0] = num_frames as i64;

        let enc_start = std::time::Instant::now();
        let encoder_outputs = if low_latency {
            triplet
                .encoder
                .run_low_latency(&triplet.encoder_inputs)
                .context("Encoder inference failed")?
        } else {
            triplet
                .encoder
                .run(&triplet.encoder_inputs)
                .context("Encoder inference failed")?
        };
        tracing::info!(
            elapsed_ms = enc_start.elapsed().as_millis() as u64,
            "encoder_inference"
        );

        let enc_len = match encoder_outputs[1].view().data() {
            TensorDataView::I32(v) => usize::try_from(v[0]).context("Negative encoder length")?,
            TensorDataView::I64(v) => usize::try_from(v[0]).context("Negative encoder length")?,
            _ => anyhow::bail!("Unexpected encoder length tensor type"),
        };

        tracing::debug!("Encoder output: {} frames", enc_len);

        // CTC head: the single encoder emits per-frame class log-probs
        // (`[1, T', 71]`, row-major). Decode them directly — there is no
        // prediction network / joiner, so we return before the RNN-T block
        // borrows `encoder_outputs` for the decode loop.
        //
        // A glossary switches the decode to a prefix beam, which is the only
        // form that can act on one: a per-frame argmax has no continuation
        // state to steer. Without hotwords the greedy path runs untouched, so
        // output for everyone else is byte-for-byte what it was.
        if self.variant.is_ctc() {
            let log_probs = encoder_outputs[0]
                .view()
                .data()
                .as_f32()
                .context("CTC log_probs tensor is not f32")?;
            let tokens = match biaser {
                Some(b) => ctc::ctc_prefix_beam_decode_with_abort(
                    log_probs,
                    enc_len,
                    self.tokenizer.vocab_size(),
                    self.tokenizer.blank_id(),
                    b,
                    abort,
                ),
                None => ctc::ctc_greedy_decode_with_abort(
                    log_probs,
                    enc_len,
                    self.tokenizer.vocab_size(),
                    self.tokenizer.blank_id(),
                    abort,
                ),
            };
            let words = ctc::ctc_tokens_to_words(&self.tokenizer, &tokens, frame_offset);
            return Ok((words, false)); // CTC has no endpoint signal
        }

        // RNN-T greedy decode — the encoder output is borrowed for the decode loop.
        let dec_start = std::time::Instant::now();
        let decoder = triplet
            .decoder
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("RNN-T decoder session missing for a non-CTC head"))?;
        let joiner = triplet
            .joiner
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("RNN-T joiner session missing for a non-CTC head"))?;
        let result = decode::greedy_decode(
            decoder,
            joiner,
            &encoder_outputs[0].view(),
            enc_len,
            self.tokenizer.blank_id(),
            decoder_state,
            biaser,
            abort,
        )?;
        tracing::info!(
            elapsed_ms = dec_start.elapsed().as_millis() as u64,
            "greedy_decode"
        );

        // Convert token infos to words with timestamps
        let words = self.tokens_to_words(&result.tokens, frame_offset);

        tracing::info!(
            tokens = result.tokens.len(),
            words = words.len(),
            duration_ms = dec_start.elapsed().as_millis() as u64,
            "Decoded tokens"
        );

        Ok((words, result.endpoint_detected))
    }

    /// Convert decoded tokens into words with timestamps and confidence.
    pub(crate) fn tokens_to_words(
        &self,
        tokens: &[decode::TokenInfo],
        frame_offset: usize,
    ) -> Vec<WordInfo> {
        TokenFormatter::tokens_to_words(&self.tokenizer, tokens, frame_offset)
    }
}
