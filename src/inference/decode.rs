//! RNN-T greedy decoding for GigaAM v3 e2e_rnnt.

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::TensorRef;

use super::{PRED_HIDDEN, DecoderState};

const MAX_TOKENS_PER_STEP: usize = 10;
const ENC_DIM: usize = 768;

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

/// Run RNN-T greedy decode on encoder output.
///
/// Encoder output layout: [1, 768, enc_len] (channels-first).
/// Decoder LSTM state is read from and written back to `state`.
pub fn greedy_decode(
    decoder: &mut Session,
    joiner: &mut Session,
    encoded: &[f32],    // [1, 768, enc_len] — channels-first
    encoded_len: usize,
    blank_id: usize,
    state: &mut DecoderState,
) -> Result<Vec<usize>> {
    let mut tokens = Vec::new();

    // Pre-allocate buffer for extracting a single encoder frame [768, 1]
    let mut enc_frame = vec![0.0_f32; ENC_DIM];

    anyhow::ensure!(
        encoded.len() >= ENC_DIM * encoded_len,
        "Encoder output size mismatch: got {}, expected >= {}",
        encoded.len(), ENC_DIM * encoded_len
    );

    for t in 0..encoded_len {
        let mut tokens_this_step = 0;

        extract_encoder_frame(encoded, encoded_len, t, &mut enc_frame);

        loop {
            // Run decoder: input prev_token [1,1] + hidden state [1,1,320]
            let target_data = [state.prev_token];
            let target_tensor =
                TensorRef::from_array_view(([1_usize, 1], target_data.as_slice()))?;
            let h_tensor =
                TensorRef::from_array_view(([1_usize, 1, PRED_HIDDEN], state.h.as_slice()))?;
            let c_tensor =
                TensorRef::from_array_view(([1_usize, 1, PRED_HIDDEN], state.c.as_slice()))?;

            let decoder_outputs = decoder
                .run(ort::inputs![target_tensor, h_tensor, c_tensor])
                .context("Decoder inference failed")?;

            // Extract decoder output [1, 1, 320] and new states
            let (_dec_shape, dec_data) = decoder_outputs[0]
                .try_extract_tensor::<f32>()
                .context("Failed to extract decoder output")?;
            let (_h_shape, new_h_data) = decoder_outputs[1]
                .try_extract_tensor::<f32>()
                .context("Failed to extract decoder h state")?;
            let (_c_shape, new_c_data) = decoder_outputs[2]
                .try_extract_tensor::<f32>()
                .context("Failed to extract decoder c state")?;

            // Joiner inputs: enc [1, 768, 1] + dec [1, 320, 1] → joint [1, 1, 1, 1025]
            let enc_tensor =
                TensorRef::from_array_view(([1_usize, ENC_DIM, 1], enc_frame.as_slice()))?;
            let dec_tensor =
                TensorRef::from_array_view(([1_usize, PRED_HIDDEN, 1], dec_data))?;

            let joiner_outputs = joiner
                .run(ort::inputs![enc_tensor, dec_tensor])
                .context("Joiner inference failed")?;

            let (_joint_shape, logits) = joiner_outputs[0]
                .try_extract_tensor::<f32>()
                .context("Failed to extract joiner output")?;

            // Greedy: argmax over logits (1025 classes)
            let token = argmax(logits, blank_id);

            if token == blank_id || tokens_this_step >= MAX_TOKENS_PER_STEP {
                break;
            }

            // Non-blank: emit token, update state
            tokens.push(token);
            state.prev_token = token as i64;
            if new_h_data.len() != PRED_HIDDEN || new_c_data.len() != PRED_HIDDEN {
                anyhow::bail!(
                    "Unexpected decoder state shape: h={}, c={}, expected {}",
                    new_h_data.len(), new_c_data.len(), PRED_HIDDEN
                );
            }
            state.h.copy_from_slice(new_h_data);
            state.c.copy_from_slice(new_c_data);
            tokens_this_step += 1;
        }
    }

    Ok(tokens)
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
}
