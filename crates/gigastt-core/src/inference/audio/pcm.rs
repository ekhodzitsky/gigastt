//! PCM16 frame parsing and streaming audio buffer management.

use super::{HOP_LENGTH, MAX_BUFFER_SAMPLES, N_FFT};

/// Parse PCM16 LE bytes into f32 samples, carrying a trailing odd byte across calls.
///
/// WebSocket clients may split their audio stream on arbitrary byte boundaries.
/// This function maintains a carry byte across frames so that odd-length payloads
/// don't introduce a 1-sample phase shift in the decoded audio.
pub fn parse_pcm16_with_carry(data: &[u8], pending: &mut Option<u8>) -> Vec<f32> {
    let mut out = Vec::new();
    parse_pcm16_with_carry_into(data, pending, &mut out);
    out
}

/// Parse PCM16 LE bytes into f32 samples, writing into a caller-provided buffer.
///
/// Same semantics as [`parse_pcm16_with_carry`] but avoids allocating a new
/// `Vec<f32>` on every call when the caller supplies a reusable buffer.
pub fn parse_pcm16_with_carry_into(data: &[u8], pending: &mut Option<u8>, out: &mut Vec<f32>) {
    out.clear();
    let carry_prev = pending.take();
    let needs_combine = carry_prev.is_some() || !data.len().is_multiple_of(2);

    if needs_combine {
        out.reserve(data.len().div_ceil(2));
        let mut bytes = data.iter().copied();
        if let Some(prev) = carry_prev {
            if let Some(b) = bytes.next() {
                out.push(i16::from_le_bytes([prev, b]) as f32 / 32768.0);
            } else {
                *pending = Some(prev);
                return;
            }
        }
        while let Some(b0) = bytes.next() {
            let b1 = match bytes.next() {
                Some(b) => b,
                None => {
                    *pending = Some(b0);
                    break;
                }
            };
            out.push(i16::from_le_bytes([b0, b1]) as f32 / 32768.0);
        }
    } else {
        out.reserve(data.len() / 2);
        for chunk in data.chunks_exact(2) {
            out.push(i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0);
        }
    }
}

/// Prepare audio buffer for processing: merge new samples with leftover,
/// truncate if too long, split into usable samples and new leftover.
///
/// Returns `Some(usable_samples)` if enough data for at least one frame,
/// `None` if all data was buffered for the next call.
/// Updates `buffer` in-place with leftover samples.
///
/// { true }
/// fn prepare_audio_buffer(new_samples: &[f32], buffer: &mut Vec<f32>) -> Option<usize>
/// { ret.is_none() == (buffer.len() < N_FFT) }
/// Determine how many samples at the front of `buffer` form complete frames.
///
/// Returns `Some(usable)` if enough data for at least one frame, `None` otherwise.
/// The caller should borrow `&buffer[..usable]`, then call
/// [`consume_audio_buffer`] to shift the leftovers.
pub(crate) fn prepare_audio_buffer(new_samples: &[f32], buffer: &mut Vec<f32>) -> Option<usize> {
    buffer.extend_from_slice(new_samples);

    if buffer.len() > MAX_BUFFER_SAMPLES {
        tracing::warn!("Audio buffer exceeded 5s limit, truncating");
        let excess = buffer.len() - MAX_BUFFER_SAMPLES;
        buffer.copy_within(excess.., 0);
        buffer.truncate(MAX_BUFFER_SAMPLES);
    }

    let hop_length = HOP_LENGTH;
    let n_fft = N_FFT;
    if buffer.len() >= n_fft {
        let num_frames = (buffer.len() - n_fft) / hop_length + 1;
        let usable = (num_frames - 1) * hop_length + n_fft;
        Some(usable)
    } else {
        None
    }
}

/// Shift leftover samples in `buffer` forward by `usable` samples and truncate.
pub(crate) fn consume_audio_buffer(buffer: &mut Vec<f32>, usable: usize) {
    buffer.copy_within(usable.., 0);
    buffer.truncate(buffer.len() - usable);
}
