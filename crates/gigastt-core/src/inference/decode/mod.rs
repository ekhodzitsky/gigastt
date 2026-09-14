//! RNN-T greedy decoding shared by the GigaAM v3 rnnt and e2e_rnnt heads.

use anyhow::{Context, Result};

use crate::runtime::{
    session::RuntimeSession,
    tensor::{Shape, Tensor, TensorData, TensorView},
};

use super::bias::Biaser;
use super::{DecoderState, PRED_HIDDEN};

const MAX_TOKENS_PER_STEP: usize = 10;

/// How many times per encoder frame hotword biasing may overturn the model's
/// own greedy pick.
///
/// Emitting a non-blank token does not advance the frame, so without a cap a
/// boosted continuation keeps outrunning blank on the same 40 ms slot and the
/// decoder spends its whole [`MAX_TOKENS_PER_STEP`] budget stuttering one
/// hotword prefix. The prefix state survives across frames, so one flip per
/// frame still lets a phrase complete — at the pace speech is actually spoken,
/// which is slower than one token per frame.
const MAX_BIAS_OVERRIDES_PER_STEP: usize = 1;
const ENC_DIM: usize = 768;
/// Number of consecutive blank frames to trigger endpointing (~600ms at 40ms/frame).
/// This is the streaming endpoint signal only when no VAD is attached; with a
/// VAD, `process_chunk` ignores it and the VAD's `min_silence_ms` owns endpointing.
pub(crate) const ENDPOINT_BLANK_THRESHOLD: usize = 15;

/// Token emitted by the decoder with metadata.
#[derive(Debug, Clone)]
pub(crate) struct TokenInfo {
    pub token_id: usize,
    pub frame_index: usize,
    pub confidence: f32,
}

/// Result of greedy decode: tokens + endpointing signal.
#[derive(Debug)]
pub(crate) struct DecodeResult {
    pub tokens: Vec<TokenInfo>,
    pub endpoint_detected: bool,
}

/// Extract encoder frame `t` from channels-first layout [1, ENC_DIM, enc_len].
///
/// Element [0, ch, t] is at index `ch * enc_len + t`.
pub(crate) fn extract_encoder_frame(
    encoded: &[f32],
    encoded_len: usize,
    t: usize,
    enc_frame: &mut [f32],
) {
    for ch in 0..enc_frame.len() {
        enc_frame[ch] = encoded[ch * encoded_len + t];
    }
}

/// Argmax over logits, returning the index of the largest value.
///
/// Returns `blank_id` if logits is empty.
pub(crate) fn argmax(logits: &[f32], blank_id: usize) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_i, a), (_j, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap_or(blank_id)
}

/// Softmax probability `logits` assigns to `token`. Zero for an empty buffer or
/// a token beyond it.
pub(crate) fn token_confidence(logits: &[f32], token: usize) -> f32 {
    let Some(&logit) = logits.get(token) else {
        return 0.0;
    };
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
    (logit - max_logit).exp() / sum_exp
}

/// Argmax with softmax confidence score.
///
/// Returns `(token_id, confidence)` where confidence is the softmax probability.
pub(crate) fn argmax_with_confidence(logits: &[f32], blank_id: usize) -> (usize, f32) {
    if logits.is_empty() {
        return (blank_id, 0.0);
    }
    let token = argmax(logits, blank_id);
    (token, token_confidence(logits, token))
}

/// Decoder call result — owned, reusable buffers for caching across frames.
///
/// During blank runs, decoder inputs (prev_token, h, c) are unchanged, so the
/// output is deterministic and the buffers are reused (read-only) without
/// re-calling the decoder. On a non-blank token the decoder runs again and
/// overwrites these buffers in place ([`copy_from_slice`]), so steady-state
/// decoding allocates nothing per token. The buffers are sized once on the
/// first decode call and stay stable for the rest of the loop.
#[derive(Default)]
pub(crate) struct DecoderOutput {
    /// Decoder output vector [PRED_HIDDEN].
    dec_data: Vec<f32>,
    /// New LSTM hidden state [PRED_HIDDEN] — committed only on non-blank token.
    new_h: Vec<f32>,
    /// New LSTM cell state [PRED_HIDDEN] — committed only on non-blank token.
    new_c: Vec<f32>,
}

impl DecoderOutput {
    /// Overwrite `dst` in place with `src`, resizing only if the length differs
    /// (first call / shape change). Steady-state calls hit the `copy_from_slice`
    /// fast path and allocate nothing.
    fn fill(dst: &mut Vec<f32>, src: &[f32]) {
        if dst.len() != src.len() {
            dst.resize(src.len(), 0.0);
        }
        dst.copy_from_slice(src);
    }
}

/// Reusable input tensors for the decoder and joiner sessions.
///
/// These tensors are allocated once per decode and mutated in place, replacing
/// the previous per-step `Vec::clone`/`to_vec` allocations.
#[derive(Debug)]
pub(crate) struct DecodeBuffers {
    /// Decoder inputs: `[prev_token [1,1], h [1,1,PRED_HIDDEN], c [1,1,PRED_HIDDEN]]`.
    decoder_inputs: Vec<Tensor>,
    /// Joiner inputs: `[enc_frame [1,ENC_DIM,1], dec_data [1,PRED_HIDDEN,1]]`.
    joiner_inputs: Vec<Tensor>,
}

impl DecodeBuffers {
    fn new() -> Self {
        Self {
            decoder_inputs: vec![
                Tensor::new_checked(Shape::new(vec![1, 1]), TensorData::I64(vec![0])),
                Tensor::new_checked(
                    Shape::new(vec![1, 1, PRED_HIDDEN]),
                    TensorData::F32(vec![0.0; PRED_HIDDEN]),
                ),
                Tensor::new_checked(
                    Shape::new(vec![1, 1, PRED_HIDDEN]),
                    TensorData::F32(vec![0.0; PRED_HIDDEN]),
                ),
            ],
            joiner_inputs: vec![
                Tensor::new_checked(
                    Shape::new(vec![1, ENC_DIM, 1]),
                    TensorData::F32(vec![0.0; ENC_DIM]),
                ),
                Tensor::new_checked(
                    Shape::new(vec![1, PRED_HIDDEN, 1]),
                    TensorData::F32(vec![0.0; PRED_HIDDEN]),
                ),
            ],
        }
    }
}

/// Run decoder ONNX session with current state, writing into reusable buffers.
///
/// Input: prev_token [1,1] + h [1,1,PRED_HIDDEN] + c [1,1,PRED_HIDDEN]
/// Output: `out` is overwritten in place with dec_data, new_h, new_c.
fn run_decoder(
    decoder: &dyn RuntimeSession,
    state: &DecoderState,
    out: &mut DecoderOutput,
    bufs: &mut DecodeBuffers,
) -> Result<()> {
    bufs.decoder_inputs[0]
        .as_i64_mut()
        .context("decoder prev_token tensor is not i64")?[0] = state.prev_token;
    bufs.decoder_inputs[1]
        .as_f32_mut()
        .context("decoder h tensor is not f32")?
        .copy_from_slice(&state.h);
    bufs.decoder_inputs[2]
        .as_f32_mut()
        .context("decoder c tensor is not f32")?
        .copy_from_slice(&state.c);

    let decoder_outputs = decoder
        .run(&bufs.decoder_inputs)
        .context("Decoder inference failed")?;

    let dec_data = decoder_outputs[0]
        .view()
        .data()
        .as_f32()
        .context("Failed to extract decoder output")?;
    let new_h_data = decoder_outputs[1]
        .view()
        .data()
        .as_f32()
        .context("Failed to extract decoder h state")?;
    let new_c_data = decoder_outputs[2]
        .view()
        .data()
        .as_f32()
        .context("Failed to extract decoder c state")?;

    DecoderOutput::fill(&mut out.dec_data, dec_data);
    DecoderOutput::fill(&mut out.new_h, new_h_data);
    DecoderOutput::fill(&mut out.new_c, new_c_data);
    Ok(())
}

/// Run joiner ONNX session on a single encoder frame.
///
/// Input: enc [1, ENC_DIM, 1] + dec [1, PRED_HIDDEN, 1]
/// Output: logits [VOCAB_SIZE] (flattened from [1, 1, 1, VOCAB_SIZE]).
fn run_joiner_single(
    joiner: &dyn RuntimeSession,
    enc_frame: &[f32],
    dec_data: &[f32],
    logits_buf: &mut Vec<f32>,
    bufs: &mut DecodeBuffers,
) -> Result<()> {
    bufs.joiner_inputs[0]
        .as_f32_mut()
        .context("joiner enc_frame tensor is not f32")?
        .copy_from_slice(enc_frame);
    bufs.joiner_inputs[1]
        .as_f32_mut()
        .context("joiner dec_data tensor is not f32")?
        .copy_from_slice(dec_data);

    let joiner_outputs = joiner
        .run(&bufs.joiner_inputs)
        .context("Joiner inference failed")?;

    let logits = joiner_outputs[0]
        .view()
        .data()
        .as_f32()
        .context("Failed to extract joiner output")?;

    // Reuse the buffer's capacity: copy in place after a one-time size match,
    // so steady-state joiner calls allocate nothing.
    DecoderOutput::fill(logits_buf, logits);
    Ok(())
}

/// Abstraction over the two ONNX session calls in the RNN-T inner loop, so the
/// decode logic can be unit-tested with a deterministic stub instead of a real
/// runtime session (which requires a model file on disk).
pub(crate) trait DecodeBackend {
    /// Run the prediction network for the current decoder state, overwriting
    /// `out` in place (reused across calls to avoid per-token allocation).
    fn decode_step(
        &mut self,
        state: &DecoderState,
        out: &mut DecoderOutput,
        bufs: &mut DecodeBuffers,
    ) -> Result<()>;
    /// Run the joiner for one encoder frame, writing logits into `logits_buf`.
    fn joiner_step(
        &mut self,
        enc_frame: &[f32],
        dec_data: &[f32],
        logits_buf: &mut Vec<f32>,
        bufs: &mut DecodeBuffers,
    ) -> Result<()>;
}

/// Production backend over the real encoder/joiner runtime sessions.
struct OrtBackend<'a> {
    decoder: &'a dyn RuntimeSession,
    joiner: &'a dyn RuntimeSession,
}

impl DecodeBackend for OrtBackend<'_> {
    fn decode_step(
        &mut self,
        state: &DecoderState,
        out: &mut DecoderOutput,
        bufs: &mut DecodeBuffers,
    ) -> Result<()> {
        run_decoder(self.decoder, state, out, bufs)
    }
    fn joiner_step(
        &mut self,
        enc_frame: &[f32],
        dec_data: &[f32],
        logits_buf: &mut Vec<f32>,
        bufs: &mut DecodeBuffers,
    ) -> Result<()> {
        run_joiner_single(self.joiner, enc_frame, dec_data, logits_buf, bufs)
    }
}

/// Run RNN-T greedy decode on encoder output.
///
/// Encoder output layout: [1, 768, enc_len] (channels-first).
/// Decoder LSTM state is read from and written back to `state`.
///
/// Optimization: during blank runs (consecutive frames where joiner outputs blank),
/// the decoder call is skipped and the cached decoder output is reused, since
/// decoder inputs (prev_token, h, c) are unchanged during blank runs.
/// `biaser` is optional contextual hotword biasing: when `Some`, a fixed boost
/// is added to the joiner logits of token-ids that extend an active hotword
/// prefix, before the argmax. `None` ⇒ the decode is byte-for-byte identical to
/// the un-biased path (zero regression risk when no hotwords are configured).
#[allow(clippy::too_many_arguments)]
pub fn greedy_decode(
    decoder: &dyn RuntimeSession,
    joiner: &dyn RuntimeSession,
    encoded: &TensorView<'_>, // [1, 768, enc_len] — channels-first
    encoded_len: usize,
    blank_id: usize,
    state: &mut DecoderState,
    biaser: Option<&Biaser>,
    abort: Option<&(dyn Fn() -> bool + Sync)>,
) -> Result<DecodeResult> {
    let mut backend = OrtBackend { decoder, joiner };
    greedy_decode_impl_with_abort(
        &mut backend,
        encoded,
        encoded_len,
        blank_id,
        state,
        biaser,
        abort,
    )
}

/// Pick the next token from joiner logits, optionally applying hotword bias.
///
/// The boost is applied to a copy so `logits` keeps the model's own scores:
/// it decides the pick, it does not get to report on it. A pick it flips
/// spends this frame's override budget — see [`MAX_BIAS_OVERRIDES_PER_STEP`].
///
/// Returns `(token, confidence, spent_override)`.
fn select_token(
    logits: &[f32],
    blank_id: usize,
    biaser: Option<&Biaser>,
    bias_state: Option<&super::bias::BiasState>,
    bias_overrides: usize,
    biased_buf: &mut Vec<f32>,
) -> (usize, f32, bool) {
    match (biaser, bias_state) {
        (Some(b), Some(bs)) if bias_overrides < MAX_BIAS_OVERRIDES_PER_STEP => {
            biased_buf.clear();
            biased_buf.extend_from_slice(logits);
            b.boost_logits(bs, biased_buf);
            let boosted = argmax(biased_buf, blank_id);
            let spent = boosted != argmax(logits, blank_id);
            (boosted, token_confidence(logits, boosted), spent)
        }
        _ => {
            let (token, confidence) = argmax_with_confidence(logits, blank_id);
            (token, confidence, false)
        }
    }
}

/// Commit a non-blank token into decoder state and advance hotword prefix.
fn commit_non_blank(
    state: &mut DecoderState,
    decoder_out: &DecoderOutput,
    token: usize,
    biaser: Option<&Biaser>,
    bias_state: Option<&mut super::bias::BiasState>,
) -> Result<()> {
    state.consecutive_blanks = 0;
    state.prev_token = token as i64;
    if decoder_out.new_h.len() != PRED_HIDDEN || decoder_out.new_c.len() != PRED_HIDDEN {
        anyhow::bail!(
            "Unexpected decoder state shape: h={}, c={}, expected {}",
            decoder_out.new_h.len(),
            decoder_out.new_c.len(),
            PRED_HIDDEN
        );
    }
    state.h.copy_from_slice(&decoder_out.new_h);
    state.c.copy_from_slice(&decoder_out.new_c);
    // Advance the hotword prefix automaton on the emitted label. Blank
    // frames emit no label, so a partial hotword survives silence gaps.
    if let (Some(b), Some(bs)) = (biaser, bias_state) {
        b.advance(bs, token);
    }
    Ok(())
}

/// Backend-generic greedy decode loop. Identical behaviour to the production
/// path; extracted so unit tests can drive it with a stub [`DecodeBackend`].
#[cfg(test)]
fn greedy_decode_impl<B: DecodeBackend>(
    backend: &mut B,
    encoded: &TensorView<'_>, // [1, 768, enc_len] — channels-first
    encoded_len: usize,
    blank_id: usize,
    state: &mut DecoderState,
    biaser: Option<&Biaser>,
) -> Result<DecodeResult> {
    greedy_decode_impl_with_abort(backend, encoded, encoded_len, blank_id, state, biaser, None)
}

/// Cancellation returns the tokens decoded so far; the engine publishes them
/// before returning its typed cancellation error. Never interrupt a runtime Run.
fn greedy_decode_impl_with_abort<B: DecodeBackend>(
    backend: &mut B,
    encoded: &TensorView<'_>,
    encoded_len: usize,
    blank_id: usize,
    state: &mut DecoderState,
    biaser: Option<&Biaser>,
    abort: Option<&(dyn Fn() -> bool + Sync)>,
) -> Result<DecodeResult> {
    let encoded = encoded
        .data()
        .as_f32()
        .context("encoder output must be f32")?;

    let mut tokens = Vec::new();
    let mut endpoint_detected = false;

    // Reusable input tensors for decoder/joiner and a reusable logits buffer.
    let mut bufs = DecodeBuffers::new();
    // Pre-allocate buffer for extracting a single encoder frame [768, 1].
    // The data is copied into the reusable joiner input tensor inside
    // `joiner_step`, avoiding a per-step `to_vec` allocation.
    let mut enc_frame = vec![0.0_f32; ENC_DIM];
    // Reusable joiner logits buffer to avoid per-call allocation.
    let mut logits_buf = Vec::new();
    let mut decoder_calls: u32 = 0;
    let mut joiner_calls: u32 = 0;
    let mut skipped_decoder_calls: u32 = 0;

    // Decoder output caching: during blank runs, decoder inputs (prev_token, h, c)
    // are unchanged, so the decoder output is deterministic and can be reused.
    // `decoder_out` is an owned, reusable buffer overwritten in place on every
    // non-blank decode call; `cache_valid` guards reuse during a blank run (the
    // decoder is only ever called when NOT in a blank run, i.e. precisely when
    // these buffers are about to be overwritten — so a blank run always reads a
    // valid, stable cache). Future work (out of scope here): precompute the
    // encoder-projection in the joiner / use ort IoBinding to also avoid the
    // ONNX-side input/output copies.
    let mut decoder_out = DecoderOutput::default();
    let mut cache_valid = false;
    let mut in_blank_run = false;

    // Hotword prefix-tracking state, only when biasing is active. `None` keeps
    // the loop on its exact pre-biasing path.
    let mut bias_state = biaser.map(|b| b.new_state());
    // Boosted copy of the joiner logits, allocated only when biasing is active.
    let mut biased_buf: Vec<f32> = Vec::new();

    anyhow::ensure!(
        encoded.len() >= ENC_DIM * encoded_len,
        "Encoder output size mismatch: got {}, expected >= {}",
        encoded.len(),
        ENC_DIM * encoded_len
    );

    'frames: for t in 0..encoded_len {
        let mut tokens_this_step = 0;
        // Per-frame biasing budget, spent only by picks the boost actually
        // flipped. Reset here so a hotword resumes on the next frame.
        let mut bias_overrides = 0usize;

        extract_encoder_frame(encoded, encoded_len, t, &mut enc_frame);

        loop {
            if abort.is_some_and(|abort| abort()) {
                endpoint_detected = false;
                break 'frames;
            }
            // === DECODER CALL (skip if in blank run) ===
            // During a blank run, prev_token/h/c are unchanged (state mutation
            // at the end of this loop is only reached for non-blank tokens).
            // Therefore run_decoder() with the same inputs produces identical
            // output, so the reusable `decoder_out` buffers are read unchanged.
            if in_blank_run {
                skipped_decoder_calls += 1;
                if !cache_valid {
                    anyhow::bail!("blank run invariant violated: decoder output cache is stale");
                }
            } else {
                decoder_calls += 1;
                // Overwrites `decoder_out` in place — no per-token allocation.
                backend.decode_step(state, &mut decoder_out, &mut bufs)?;
                cache_valid = true;
            }

            // === JOINER CALL ===
            joiner_calls += 1;
            backend.joiner_step(
                &enc_frame,
                &decoder_out.dec_data,
                &mut logits_buf,
                &mut bufs,
            )?;

            // === CONTEXTUAL HOTWORD BIASING (shallow fusion) ===
            let (token, confidence, spent) = select_token(
                &logits_buf,
                blank_id,
                biaser,
                bias_state.as_ref(),
                bias_overrides,
                &mut biased_buf,
            );
            if spent {
                bias_overrides += 1;
            }

            // === TOKEN CLASSIFICATION ===
            if token == blank_id {
                // True blank: decoder state was NOT updated. Safe to cache.
                in_blank_run = true;
                state.consecutive_blanks += 1;
                if state.consecutive_blanks >= ENDPOINT_BLANK_THRESHOLD && !tokens.is_empty() {
                    endpoint_detected = true;
                }
                break;
            }

            if tokens_this_step >= MAX_TOKENS_PER_STEP {
                // Token cap: the joiner emitted MAX_TOKENS_PER_STEP non-blank tokens
                // on this frame — dense speech, NOT silence. It is therefore NOT an
                // endpoint signal, so reset the blank counter (consistent with the
                // non-blank branch). The cached decoder output is stale.
                in_blank_run = false;
                cache_valid = false;
                state.consecutive_blanks = 0;
                break;
            }

            // === NON-BLANK TOKEN: commit state, emit token ===
            in_blank_run = false;
            commit_non_blank(state, &decoder_out, token, biaser, bias_state.as_mut())?;
            tokens.push(TokenInfo {
                token_id: token,
                frame_index: t,
                confidence,
            });
            tokens_this_step += 1;
        }
    }

    tracing::debug!(
        decoder_calls,
        joiner_calls,
        skipped_decoder_calls,
        encoded_len,
        "decode_loop_stats"
    );
    Ok(DecodeResult {
        tokens,
        endpoint_detected,
    })
}

#[cfg(test)]
mod tests;
