//! Sample-rate types and high-quality polyphase FIR resampling (rubato).

use anyhow::Result;
use rubato::Resampler;

/// Sample rate in Hz. Invariant: `rate > 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SampleRate(pub u32);

impl SampleRate {
    /// { rate > 0 }
    /// fn new(rate: u32) -> Result<SampleRate, String>
    /// { ret.as_ref().map(|r| r.0 > 0).unwrap_or(true) }
    pub fn new(rate: u32) -> Result<Self, String> {
        if rate == 0 {
            return Err("sample rate must be > 0".into());
        }
        Ok(SampleRate(rate))
    }

    /// { true }
    /// fn get(self) -> u32
    /// { ret > 0 }
    pub fn get(self) -> u32 {
        self.0
    }
}

/// High-quality polyphase FIR resampler (rubato Async, sinc interpolation).
///
/// Non-finite samples (NaN, infinity) are replaced with `0.0` before resampling.
///
/// ```text
/// { from_rate.0 > 0 && to_rate.0 > 0 }
/// fn resample(samples: &[f32], from_rate: SampleRate, to_rate: SampleRate) -> Result<Vec<f32>>
/// { ret.as_ref().map(|v| !v.is_empty() || samples.is_empty() || from_rate == to_rate).unwrap_or(true) }
/// ```
pub fn resample(samples: &[f32], from_rate: SampleRate, to_rate: SampleRate) -> Result<Vec<f32>> {
    if samples.is_empty() || from_rate.0 == 0 || to_rate.0 == 0 {
        return Ok(Vec::new());
    }
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }

    // Sanitize non-finite values
    let samples: Vec<f32> = samples
        .iter()
        .map(|&s| if s.is_finite() { s } else { 0.0 })
        .collect();

    use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
    use rubato::{
        Async, FixedAsync, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let ratio = to_rate.0 as f64 / from_rate.0 as f64;
    let chunk = samples.len();
    let mut resampler = Async::<f32>::new_sinc(ratio, 2.0, &params, chunk, 1, FixedAsync::Input)
        .map_err(|e| anyhow::anyhow!("Resampler init failed: {e}"))?;

    let input_data = [samples];
    let out_frames = resampler.output_frames_next();
    let mut output_data = [vec![0.0f32; out_frames]];
    {
        let input = SequentialSliceOfVecs::new(&input_data, 1, chunk)
            .map_err(|e| anyhow::anyhow!("Resampler input adapter failed: {e}"))?;
        let mut output = SequentialSliceOfVecs::new_mut(&mut output_data, 1, out_frames)
            .map_err(|e| anyhow::anyhow!("Resampler output adapter failed: {e}"))?;
        resampler
            .process_into_buffer(&input, &mut output, None)
            .map_err(|e| anyhow::anyhow!("Resampling failed: {e}"))?;
    }
    let [out_vec] = output_data;
    Ok(out_vec)
}

/// Lower bound for the cached streaming resampler's chunk capacity.
///
/// The capacity is fixed when the resampler is first created and cannot be
/// raised later without recreating it (which would reset the FIR state). A
/// tiny first frame would otherwise cap every later frame at that size and
/// make the oversized-frame split loop run many times per call; 4096 samples
/// covers ~85 ms at 48 kHz, so realistic frames never split. The value only
/// bounds per-call overhead — correctness holds for any capacity.
const MIN_RESAMPLER_CAPACITY: usize = 4096;

/// Resample audio using an optional cached resampler, writing into a caller-provided buffer.
///
/// The cached resampler is created once on first call and reused for the rest
/// of the session; it is never recreated, so the FIR history and fractional
/// phase survive across frames and no seam discontinuities appear at frame
/// boundaries. Chunk sizes may vary freely between calls: sizes up to the
/// resampler capacity are applied via `set_chunk_size` (which rubato supports
/// without touching filter state), and larger chunks are fed through the same
/// resampler in capacity-sized pieces.
///
/// Non-finite samples are sanitized in-place.
///
/// `samples` is consumed (moved) so that in-place sanitization avoids an
/// extra allocation. Callers that already own the input vector should pass
/// it directly; the buffer is not borrowed after the call.
///
/// ```text
/// { from_rate.0 > 0 && to_rate.0 > 0 }
/// fn resample_with_cache(samples: Vec<f32>, from_rate: SampleRate, to_rate: SampleRate, cache: &mut Option<rubato::Async<f32>>, out_buf: &mut Vec<f32>) -> anyhow::Result<()>
/// { ret.as_ref().map(|v| !v.is_empty() || samples.is_empty() || from_rate == to_rate).unwrap_or(true) }
/// ```
pub fn resample_with_cache(
    mut samples: Vec<f32>,
    from_rate: SampleRate,
    to_rate: SampleRate,
    cache: &mut Option<rubato::Async<f32>>,
    out_buf: &mut Vec<f32>,
) -> anyhow::Result<()> {
    if samples.is_empty() || from_rate.0 == 0 || to_rate.0 == 0 {
        out_buf.clear();
        return Ok(());
    }
    if from_rate == to_rate {
        *out_buf = samples;
        return Ok(());
    }

    // Sanitize non-finite values in-place
    for s in &mut samples {
        if !s.is_finite() {
            *s = 0.0;
        }
    }

    if cache.is_none() {
        use rubato::{
            Async, FixedAsync, SincInterpolationParameters, SincInterpolationType, WindowFunction,
        };
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };
        let ratio = to_rate.0 as f64 / from_rate.0 as f64;
        // Fix the capacity up front: it can never be raised without
        // recreating the resampler and losing the FIR state.
        let capacity = samples.len().max(MIN_RESAMPLER_CAPACITY);
        let r = Async::<f32>::new_sinc(ratio, 2.0, &params, capacity, 1, FixedAsync::Input)
            .map_err(|e| anyhow::anyhow!("Resampler init failed: {e}"))?;
        *cache = Some(r);
    }

    let resampler = match cache.as_mut() {
        Some(r) => r,
        None => anyhow::bail!("Resampler cache is None after initialization"),
    };
    out_buf.clear();
    let max_input = resampler.input_frames_max();
    if samples.len() <= max_input {
        process_cached_chunk(resampler, samples, out_buf)?;
    } else {
        // Frame exceeds the fixed capacity: feed it in capacity-sized pieces
        // through the same resampler so the FIR state carries across pieces.
        let mut piece_out = Vec::new();
        for piece in samples.chunks(max_input) {
            process_cached_chunk(resampler, piece.to_vec(), &mut piece_out)?;
            out_buf.extend_from_slice(&piece_out);
        }
    }
    Ok(())
}

/// Run one chunk through the cached resampler, replacing `dst` with the output.
///
/// `samples.len()` must not exceed `resampler.input_frames_max()`. The chunk
/// size is applied via `set_chunk_size`, which adjusts the required
/// input/output lengths while preserving the FIR history and fractional
/// phase — this is what keeps variable-sized streaming frames seamless.
fn process_cached_chunk(
    resampler: &mut rubato::Async<f32>,
    samples: Vec<f32>,
    dst: &mut Vec<f32>,
) -> anyhow::Result<()> {
    use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;

    let chunk = samples.len();
    resampler
        .set_chunk_size(chunk)
        .map_err(|e| anyhow::anyhow!("Resampler chunk resize failed: {e}"))?;
    let needed = resampler.output_frames_next();
    dst.clear();
    dst.resize(needed, 0.0);

    let input_data = [samples];
    let input = SequentialSliceOfVecs::new(&input_data, 1, chunk)
        .map_err(|e| anyhow::anyhow!("Resampler input adapter failed: {e}"))?;
    let mut output = SequentialSliceOfVecs::new_mut(std::slice::from_mut(dst), 1, needed)
        .map_err(|e| anyhow::anyhow!("Resampler output adapter failed: {e}"))?;
    resampler
        .process_into_buffer(&input, &mut output, None)
        .map_err(|e| anyhow::anyhow!("Resampling failed: {e}"))?;
    Ok(())
}
