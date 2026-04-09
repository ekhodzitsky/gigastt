//! RNN-T greedy decoding for GigaAM v3 e2e_rnnt.

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::TensorRef;

const MAX_TOKENS_PER_STEP: usize = 3;
const PRED_HIDDEN: usize = 320;
const ENC_DIM: usize = 768;

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
    state: &mut super::DecoderState,
) -> Result<Vec<usize>> {
    let mut tokens = Vec::new();

    // Pre-allocate buffer for extracting a single encoder frame [768, 1]
    let mut enc_frame = vec![0.0_f32; ENC_DIM];

    for t in 0..encoded_len {
        let mut tokens_this_step = 0;

        // Extract encoder frame t from [1, 768, enc_len] layout:
        // element [0, ch, t] is at index ch * enc_len + t
        for ch in 0..ENC_DIM {
            enc_frame[ch] = encoded[ch * encoded_len + t];
        }

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
            let token = logits
                .iter()
                .enumerate()
                .max_by(|(_i, a), (_j, b)| a.total_cmp(b))
                .map(|(idx, _)| idx)
                .unwrap_or(blank_id);

            if token == blank_id || tokens_this_step >= MAX_TOKENS_PER_STEP {
                break;
            }

            // Non-blank: emit token, update state
            tokens.push(token);
            state.prev_token = token as i64;
            state.h.copy_from_slice(new_h_data);
            state.c.copy_from_slice(new_c_data);
            tokens_this_step += 1;
        }
    }

    Ok(tokens)
}
