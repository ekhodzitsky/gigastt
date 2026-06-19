//! RNN-T greedy decoding for GigaAM v3 e2e_rnnt.

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::TensorRef;

use super::{DecoderState, PRED_HIDDEN};

const MAX_TOKENS_PER_STEP: usize = 10;
const ENC_DIM: usize = 768;
/// Number of consecutive blank frames to trigger endpointing (~600ms at 40ms/frame).
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

/// Argmax with softmax confidence score.
///
/// Returns `(token_id, confidence)` where confidence is the softmax probability.
pub(crate) fn argmax_with_confidence(logits: &[f32], blank_id: usize) -> (usize, f32) {
    if logits.is_empty() {
        return (blank_id, 0.0);
    }
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
    let token = argmax(logits, blank_id);
    let confidence = (logits[token] - max_logit).exp() / sum_exp;
    (token, confidence)
}

/// Decoder call result — owned data for caching across frames.
///
/// During blank runs, decoder inputs (prev_token, h, c) are unchanged,
/// so the output is deterministic and can be reused without re-calling the decoder.
pub(crate) struct DecoderOutput {
    /// Decoder output vector [PRED_HIDDEN].
    dec_data: Vec<f32>,
    /// New LSTM hidden state [PRED_HIDDEN] — committed only on non-blank token.
    new_h: Vec<f32>,
    /// New LSTM cell state [PRED_HIDDEN] — committed only on non-blank token.
    new_c: Vec<f32>,
}

/// Run decoder ONNX session with current state.
///
/// Input: prev_token [1,1] + h [1,1,PRED_HIDDEN] + c [1,1,PRED_HIDDEN]
/// Output: DecoderOutput with dec_data, new_h, new_c (all owned).
fn run_decoder(decoder: &mut Session, state: &DecoderState) -> Result<DecoderOutput> {
    let target_data = [state.prev_token];
    let target_tensor = TensorRef::from_array_view(([1_usize, 1], target_data.as_slice()))?;
    let h_tensor = TensorRef::from_array_view(([1_usize, 1, PRED_HIDDEN], state.h.as_slice()))?;
    let c_tensor = TensorRef::from_array_view(([1_usize, 1, PRED_HIDDEN], state.c.as_slice()))?;

    let decoder_outputs = decoder
        .run(ort::inputs![target_tensor, h_tensor, c_tensor])
        .context("Decoder inference failed")?;

    let (_dec_shape, dec_data) = decoder_outputs[0]
        .try_extract_tensor::<f32>()
        .context("Failed to extract decoder output")?;
    let (_h_shape, new_h_data) = decoder_outputs[1]
        .try_extract_tensor::<f32>()
        .context("Failed to extract decoder h state")?;
    let (_c_shape, new_c_data) = decoder_outputs[2]
        .try_extract_tensor::<f32>()
        .context("Failed to extract decoder c state")?;

    Ok(DecoderOutput {
        dec_data: dec_data.to_vec(),
        new_h: new_h_data.to_vec(),
        new_c: new_c_data.to_vec(),
    })
}

/// Run joiner ONNX session on a single encoder frame.
///
/// Input: enc [1, ENC_DIM, 1] + dec [1, PRED_HIDDEN, 1]
/// Output: logits [VOCAB_SIZE] (flattened from [1, 1, 1, VOCAB_SIZE]).
fn run_joiner_single(
    joiner: &mut Session,
    enc_frame: &[f32],
    dec_data: &[f32],
    logits_buf: &mut Vec<f32>,
) -> Result<()> {
    let enc_tensor = TensorRef::from_array_view(([1_usize, ENC_DIM, 1], enc_frame))?;
    let dec_tensor = TensorRef::from_array_view(([1_usize, PRED_HIDDEN, 1], dec_data))?;

    let joiner_outputs = joiner
        .run(ort::inputs![enc_tensor, dec_tensor])
        .context("Joiner inference failed")?;

    let (_joint_shape, logits) = joiner_outputs[0]
        .try_extract_tensor::<f32>()
        .context("Failed to extract joiner output")?;

    logits_buf.clear();
    logits_buf.extend_from_slice(logits);
    Ok(())
}

/// Abstraction over the two ONNX session calls in the RNN-T inner loop, so the
/// decode logic can be unit-tested with a deterministic stub instead of a real
/// `ort::Session` (which requires a model file on disk).
pub(crate) trait DecodeBackend {
    /// Run the prediction network for the current decoder state.
    fn decode_step(&mut self, state: &DecoderState) -> Result<DecoderOutput>;
    /// Run the joiner for one encoder frame, writing logits into `logits_buf`.
    fn joiner_step(
        &mut self,
        enc_frame: &[f32],
        dec_data: &[f32],
        logits_buf: &mut Vec<f32>,
    ) -> Result<()>;
}

/// Production backend over the real encoder/joiner ONNX sessions.
struct OrtBackend<'a> {
    decoder: &'a mut Session,
    joiner: &'a mut Session,
}

impl DecodeBackend for OrtBackend<'_> {
    fn decode_step(&mut self, state: &DecoderState) -> Result<DecoderOutput> {
        run_decoder(self.decoder, state)
    }
    fn joiner_step(
        &mut self,
        enc_frame: &[f32],
        dec_data: &[f32],
        logits_buf: &mut Vec<f32>,
    ) -> Result<()> {
        run_joiner_single(self.joiner, enc_frame, dec_data, logits_buf)
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
pub fn greedy_decode(
    decoder: &mut Session,
    joiner: &mut Session,
    encoded: &[f32], // [1, 768, enc_len] — channels-first
    encoded_len: usize,
    blank_id: usize,
    state: &mut DecoderState,
) -> Result<DecodeResult> {
    let mut backend = OrtBackend { decoder, joiner };
    greedy_decode_impl(&mut backend, encoded, encoded_len, blank_id, state)
}

/// Backend-generic greedy decode loop. Identical behaviour to the production
/// path; extracted so unit tests can drive it with a stub [`DecodeBackend`].
fn greedy_decode_impl<B: DecodeBackend>(
    backend: &mut B,
    encoded: &[f32], // [1, 768, enc_len] — channels-first
    encoded_len: usize,
    blank_id: usize,
    state: &mut DecoderState,
) -> Result<DecodeResult> {
    let mut tokens = Vec::new();
    let mut endpoint_detected = false;

    // Pre-allocate buffer for extracting a single encoder frame [768, 1]
    let mut enc_frame = vec![0.0_f32; ENC_DIM];
    // Reusable joiner logits buffer to avoid per-call allocation.
    let mut logits_buf = Vec::new();
    let mut decoder_calls: u32 = 0;
    let mut joiner_calls: u32 = 0;
    let mut skipped_decoder_calls: u32 = 0;

    // Decoder output caching: during blank runs, decoder inputs (prev_token, h, c)
    // are unchanged, so the decoder output is deterministic and can be reused.
    let mut cached_decoder_output: Option<DecoderOutput> = None;
    let mut in_blank_run = false;

    anyhow::ensure!(
        encoded.len() >= ENC_DIM * encoded_len,
        "Encoder output size mismatch: got {}, expected >= {}",
        encoded.len(),
        ENC_DIM * encoded_len
    );

    for t in 0..encoded_len {
        let mut tokens_this_step = 0;

        extract_encoder_frame(encoded, encoded_len, t, &mut enc_frame);

        loop {
            // === DECODER CALL (skip if in blank run) ===
            // During a blank run, prev_token/h/c are unchanged (state mutation
            // at the end of this loop is only reached for non-blank tokens).
            // Therefore run_decoder() with the same inputs produces identical output.
            let decoder_out = if in_blank_run {
                skipped_decoder_calls += 1;
                match cached_decoder_output.as_ref() {
                    Some(cache) => cache,
                    None => {
                        anyhow::bail!("blank run invariant violated: cached_decoder_output is None")
                    }
                }
            } else {
                decoder_calls += 1;
                let out = backend.decode_step(state)?;
                cached_decoder_output = Some(out);
                cached_decoder_output
                    .as_ref()
                    .expect("decoder output just cached")
            };

            // === JOINER CALL ===
            joiner_calls += 1;
            backend.joiner_step(&enc_frame, &decoder_out.dec_data, &mut logits_buf)?;

            // Greedy: argmax with confidence over logits
            let (token, confidence) = argmax_with_confidence(&logits_buf, blank_id);

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
                cached_decoder_output = None;
                state.consecutive_blanks = 0;
                break;
            }

            // === NON-BLANK TOKEN: commit state, emit token ===
            in_blank_run = false;
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
mod tests {
    use super::*;

    // --- extract_encoder_frame tests ---

    #[test]
    fn test_extract_encoder_frame_first() {
        // 2 channels, 3 time steps: [ch0: 1,2,3, ch1: 4,5,6]
        let encoded = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut frame = vec![0.0; 2];
        extract_encoder_frame(&encoded, 3, 0, &mut frame);
        assert_eq!(frame, vec![1.0, 4.0]);
    }

    #[test]
    fn test_extract_encoder_frame_last() {
        let encoded = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut frame = vec![0.0; 2];
        extract_encoder_frame(&encoded, 3, 2, &mut frame);
        assert_eq!(frame, vec![3.0, 6.0]);
    }

    #[test]
    fn test_extract_encoder_frame_middle() {
        let encoded = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut frame = vec![0.0; 2];
        extract_encoder_frame(&encoded, 3, 1, &mut frame);
        assert_eq!(frame, vec![2.0, 5.0]);
    }

    // --- argmax tests ---

    #[test]
    fn test_argmax_clear_winner() {
        let logits = vec![0.1, 0.5, 0.9, 0.2];
        assert_eq!(argmax(&logits, 999), 2);
    }

    #[test]
    fn test_argmax_tie_returns_last() {
        // Rust's Iterator::max_by returns the last element on ties
        let logits = vec![1.0, 1.0, 0.5];
        assert_eq!(argmax(&logits, 999), 1);
    }

    #[test]
    fn test_argmax_single_element() {
        let logits = vec![42.0];
        assert_eq!(argmax(&logits, 999), 0);
    }

    #[test]
    fn test_argmax_negative_values() {
        let logits = vec![-3.0, -1.0, -2.0];
        assert_eq!(argmax(&logits, 999), 1);
    }

    #[test]
    fn test_argmax_empty_returns_blank() {
        let logits: Vec<f32> = vec![];
        assert_eq!(argmax(&logits, 1024), 1024);
    }

    #[test]
    fn test_argmax_blank_id_selected() {
        // If blank_id is the argmax, it should be returned
        let logits = vec![0.1, 0.2, 0.9]; // index 2 is max
        assert_eq!(argmax(&logits, 2), 2); // blank_id matches argmax
    }

    // --- greedy_decode tests via a deterministic stub backend (no model) ---

    /// Stub backend: a scripted sequence of token ids drives the joiner's argmax;
    /// decoder/joiner call counts are recorded so the blank-run cache can be checked.
    struct FakeBackend {
        script: std::collections::VecDeque<usize>,
        vocab: usize,
        blank_id: usize,
        decoder_calls: u32,
        joiner_calls: u32,
    }

    impl FakeBackend {
        fn new(script: Vec<usize>, vocab: usize, blank_id: usize) -> Self {
            Self {
                script: script.into(),
                vocab,
                blank_id,
                decoder_calls: 0,
                joiner_calls: 0,
            }
        }
    }

    impl DecodeBackend for FakeBackend {
        fn decode_step(&mut self, _state: &DecoderState) -> Result<DecoderOutput> {
            self.decoder_calls += 1;
            Ok(DecoderOutput {
                dec_data: vec![0.0; PRED_HIDDEN],
                new_h: vec![0.0; PRED_HIDDEN],
                new_c: vec![0.0; PRED_HIDDEN],
            })
        }

        fn joiner_step(
            &mut self,
            _enc_frame: &[f32],
            _dec_data: &[f32],
            logits_buf: &mut Vec<f32>,
        ) -> Result<()> {
            self.joiner_calls += 1;
            // Once the script runs out, return blank so the loop terminates.
            let tok = self.script.pop_front().unwrap_or(self.blank_id);
            logits_buf.clear();
            logits_buf.resize(self.vocab, 0.0);
            logits_buf[tok] = 10.0; // argmax → tok
            Ok(())
        }
    }

    /// Encoder buffer of `frames` zeroed frames (content is irrelevant to the stub).
    fn fake_enc(frames: usize) -> Vec<f32> {
        vec![0.0_f32; ENC_DIM * frames]
    }

    #[test]
    fn test_greedy_decode_happy_path() {
        // vocab=5, blank=4. Frame 0 emits token 1 then blank; frame 1 emits token 2.
        let mut backend = FakeBackend::new(vec![1, 4, 2, 4], 5, 4);
        let mut state = DecoderState::new(4);
        let result = greedy_decode_impl(&mut backend, &fake_enc(2), 2, 4, &mut state).unwrap();

        assert_eq!(result.tokens.len(), 2);
        assert_eq!(result.tokens[0].token_id, 1);
        assert_eq!(result.tokens[0].frame_index, 0);
        assert_eq!(result.tokens[1].token_id, 2);
        assert_eq!(result.tokens[1].frame_index, 1);
        // Last committed token updates prev_token and the LSTM state buffers.
        assert_eq!(state.prev_token, 2);
        assert_eq!(state.h.len(), PRED_HIDDEN);
        assert!(!result.endpoint_detected);
    }

    #[test]
    fn test_greedy_decode_blank_run_skips_decoder() {
        // Frame 0: token then blank (2 decoder calls). Frames 1-3: blank only.
        // The decoder must NOT be called again during the blank run (cache reuse).
        let mut backend = FakeBackend::new(vec![1, 4, 4, 4, 4], 5, 4);
        let mut state = DecoderState::new(4);
        let result = greedy_decode_impl(&mut backend, &fake_enc(4), 4, 4, &mut state).unwrap();

        assert_eq!(result.tokens.len(), 1);
        assert_eq!(
            backend.decoder_calls, 2,
            "decoder must not run during the blank run"
        );
        assert!(backend.joiner_calls >= 5);
    }

    #[test]
    fn test_greedy_decode_endpoint_after_threshold_blanks() {
        // One token, then ENDPOINT_BLANK_THRESHOLD+ blanks → endpoint detected.
        let mut script = vec![1usize];
        script.extend(std::iter::repeat_n(4usize, ENDPOINT_BLANK_THRESHOLD + 1));
        let frames = ENDPOINT_BLANK_THRESHOLD + 2;
        let mut backend = FakeBackend::new(script, 5, 4);
        let mut state = DecoderState::new(4);
        let result =
            greedy_decode_impl(&mut backend, &fake_enc(frames), frames, 4, &mut state).unwrap();

        assert!(
            result.endpoint_detected,
            "{ENDPOINT_BLANK_THRESHOLD}+ blanks after a token must endpoint"
        );
    }

    #[test]
    fn test_greedy_decode_no_endpoint_before_first_token() {
        // All blanks, no token emitted → the !tokens.is_empty() guard blocks endpoint.
        let frames = ENDPOINT_BLANK_THRESHOLD + 5;
        let mut backend = FakeBackend::new(vec![4usize; frames], 5, 4);
        let mut state = DecoderState::new(4);
        let result =
            greedy_decode_impl(&mut backend, &fake_enc(frames), frames, 4, &mut state).unwrap();

        assert!(result.tokens.is_empty());
        assert!(
            !result.endpoint_detected,
            "blanks before any token must not endpoint"
        );
    }

    #[test]
    fn test_greedy_decode_token_cap_does_not_inflate_blanks() {
        // One frame; the joiner returns a non-blank token on every call past the cap.
        // Exactly MAX_TOKENS_PER_STEP tokens are emitted, and the token cap must NOT
        // bump the blank counter or fire an endpoint.
        let mut backend = FakeBackend::new(vec![1usize; MAX_TOKENS_PER_STEP + 1], 5, 4);
        let mut state = DecoderState::new(4);
        let result = greedy_decode_impl(&mut backend, &fake_enc(1), 1, 4, &mut state).unwrap();

        assert_eq!(result.tokens.len(), MAX_TOKENS_PER_STEP);
        assert_eq!(
            state.consecutive_blanks, 0,
            "token cap must not inflate the blank counter"
        );
        assert!(!result.endpoint_detected);
    }

    #[test]
    fn test_argmax_with_confidence_clear_winner() {
        let (tok, conf) = argmax_with_confidence(&[0.1, 5.0, 0.2], 99);
        assert_eq!(tok, 1);
        assert!(
            conf > 0.5 && conf <= 1.0,
            "confidence should be a softmax prob in (0.5, 1], got {conf}"
        );
    }

    #[test]
    fn test_argmax_with_confidence_empty_returns_blank_zero() {
        let (tok, conf) = argmax_with_confidence(&[], 1024);
        assert_eq!(tok, 1024);
        assert_eq!(conf, 0.0);
    }
}
