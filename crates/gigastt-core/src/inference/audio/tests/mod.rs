//! Audio unit tests, split by concern. Child modules see the parent `audio`
//! surface via the glob re-export below.

pub(super) use super::*;

use audio_codec::{Decoder, Encoder};

/// `audio-codec` 0.4 keeps allocating `encode`/`decode` behind the `std`
/// feature. Tests drive the heap-free `*_into` API with a sized scratch
/// buffer so we can keep `default-features = false` (no bundled Opus).
fn encode_telephony<E: Encoder>(encoder: &mut E, samples: &[i16]) -> Vec<u8> {
    let n = encoder.max_encode_bytes(samples.len());
    let mut out = vec![0u8; n];
    let written = encoder
        .encode_into(samples, &mut out)
        .unwrap_or_else(|e| panic!("test encode failed: {e}"));
    out.truncate(written);
    out
}

fn decode_telephony_pcm<D: Decoder>(decoder: &mut D, data: &[u8]) -> Vec<i16> {
    let n = decoder.max_decode_samples(data.len());
    let mut pcm = vec![0i16; n];
    let written = decoder
        .decode_into(data, &mut pcm)
        .unwrap_or_else(|e| panic!("test decode failed: {e}"));
    pcm.truncate(written);
    pcm
}

pub(super) fn make_wav_bytes(samples: &[i16], sample_rate: u32) -> Vec<u8> {
    let data_size = (samples.len() * 2) as u32;
    let file_size = 36 + data_size;
    let mut buf = Vec::new();
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&1u16.to_le_bytes()); // mono
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    buf.extend_from_slice(&2u16.to_le_bytes()); // block align
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());
    for &s in samples {
        buf.extend_from_slice(&s.to_le_bytes());
    }
    buf
}

fn make_stereo_wav_from_frames(frames: &[(i16, i16)], sample_rate: u32) -> Vec<u8> {
    let data_size = (frames.len() * 4) as u32; // 2 channels * 2 bytes
    let file_size = 36 + data_size;
    let mut buf = Vec::new();
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&2u16.to_le_bytes()); // stereo
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&(sample_rate * 4).to_le_bytes()); // byte rate
    buf.extend_from_slice(&4u16.to_le_bytes()); // block align
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());
    for &(l, r) in frames {
        buf.extend_from_slice(&l.to_le_bytes());
        buf.extend_from_slice(&r.to_le_bytes());
    }
    buf
}

fn make_stereo_wav_bytes(left: &[i16], right: &[i16], sample_rate: u32) -> Vec<u8> {
    assert_eq!(left.len(), right.len());
    let num_samples = left.len();
    let data_size = (num_samples * 4) as u32; // 2 channels * 2 bytes
    let file_size = 36 + data_size;
    let mut buf = Vec::with_capacity(file_size as usize);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&2u16.to_le_bytes()); // stereo
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&(sample_rate * 4).to_le_bytes()); // byte rate
    buf.extend_from_slice(&4u16.to_le_bytes()); // block align
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());
    for i in 0..num_samples {
        buf.extend_from_slice(&left[i].to_le_bytes());
        buf.extend_from_slice(&right[i].to_le_bytes());
    }
    buf
}

/// Build a WAV buffer with an arbitrary format tag around an encoded
/// payload (mono). The `fmt ` chunk carries the 2-byte `cbSize` extension
/// field (18 bytes total) because symphonia rejects 16-byte `fmt ` chunks
/// for the G.711 tags — and it is what ffmpeg writes for all of these.
fn make_compressed_wav(tag: u16, sample_rate: u32, byte_rate: u32, payload: &[u8]) -> Vec<u8> {
    let data_size = payload.len() as u32;
    let mut buf = Vec::with_capacity(46 + payload.len());
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(38 + data_size).to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&18u32.to_le_bytes()); // fmt chunk size (incl. cbSize)
    buf.extend_from_slice(&tag.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes()); // mono
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes()); // block align
    buf.extend_from_slice(&8u16.to_le_bytes()); // bits per sample
    buf.extend_from_slice(&0u16.to_le_bytes()); // cbSize = 0
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());
    buf.extend_from_slice(payload);
    buf
}

fn test_tone_8k(n_samples: usize) -> Vec<i16> {
    (0..n_samples)
        .map(|i| ((i as f32 * 0.05).sin() * 12000.0) as i16)
        .collect()
}

/// RMSE between two equal-rate signals at the best integer lag within
/// ±`max_lag` samples. Lossy codecs carry an inherent group delay, so a
/// fixed-alignment RMSE would report the delay as distortion.
pub(super) fn best_lag_rmse(a: &[f32], b: &[f32], max_lag: usize) -> f64 {
    let mut best = f64::INFINITY;
    for lag in 0..=max_lag {
        for (a_slice, b_slice) in [
            (a.get(lag..).unwrap_or(&[]), b),
            (a, b.get(lag..).unwrap_or(&[])),
        ] {
            let n = a_slice.len().min(b_slice.len());
            if n < 100 {
                continue;
            }
            let mse = a_slice[..n]
                .iter()
                .zip(&b_slice[..n])
                .map(|(x, y)| {
                    let d = f64::from(x - y);
                    d * d
                })
                .sum::<f64>()
                / n as f64;
            best = best.min(mse.sqrt());
        }
    }
    best
}

#[cfg(feature = "file-decode")]
mod aac;
mod decode;
mod opus;
mod pcm;
mod resample;
mod stream;
mod telephony;

#[cfg(feature = "file-decode")]
mod cancellation;
