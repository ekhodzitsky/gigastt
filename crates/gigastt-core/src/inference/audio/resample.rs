//! Sample-rate types and high-quality polyphase FIR resampling (rubato).

#[cfg(feature = "file-decode")]
use anyhow::Context;
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

/// Source-rate samples staged before each streaming resample flush.
///
/// One second at 48 kHz: big enough that the cached resampler — whose capacity
/// freezes on its first call — never has to split a later chunk, small enough
/// that the staged copy stays a few hundred KiB no matter how long the file is.
/// This is a resample-granularity constant, unrelated to any length limit.
#[cfg(feature = "file-decode")]
pub(super) const RESAMPLE_STAGING_FRAMES: usize = 48_000;

/// Decoded-sample accumulator that resamples to 16 kHz while the file decodes.
///
/// Packet decoders append straight into [`stage`](Self::stage) and call
/// [`flush_full`](Self::flush_full) once per packet. When the source is
/// already 16 kHz the staged buffer *is* the output: no resampler is built and
/// no sample is copied, so that path stays byte-for-byte what the whole-buffer
/// version produced. At any other rate the staging buffer is capped at
/// [`RESAMPLE_STAGING_FRAMES`] source-rate samples and drained through
/// [`resample_with_cache`], whose FIR history and fractional phase carry
/// across flushes, so peak memory stays O(one chunk) instead of O(file) and no
/// seam appears at a flush boundary.
///
/// Against a single whole-buffer [`resample`] call the concatenated output is
/// bit-identical at integer ratios (48/32/8 kHz, where `1/ratio` is exact in
/// f64). At a non-integer ratio the two separate slowly as the input grows,
/// because [`resample`] runs rubato's fractional-position accumulator over the
/// whole file in one pass and loses sub-sample resolution as it grows, while
/// this path restarts it near zero every flush. The staged path is the more
/// accurate of the two — it stays phase-locked to an analytic tone at every
/// length while the whole-buffer reference walks; see the long-input test in
/// `tests.rs` for the measured bounds.
#[cfg(feature = "file-decode")]
pub(super) struct ResampleTo16k {
    from_rate: SampleRate,
    stage: Vec<f32>,
    out: Vec<f32>,
    cache: Option<rubato::Async<f32>>,
    scratch: Vec<f32>,
}

#[cfg(feature = "file-decode")]
impl ResampleTo16k {
    /// `source_frames_hint` is the container's declared frame count (already
    /// bounded by the decode budget); it reserves the 16 kHz output up front.
    pub(super) fn new(from_rate: SampleRate, source_frames_hint: Option<usize>) -> Self {
        if from_rate.0 == 16_000 {
            // Passthrough: the staging buffer is handed back verbatim, so it
            // takes the whole reservation and `out` is never touched.
            return Self {
                from_rate,
                stage: match source_frames_hint {
                    Some(n) => Vec::with_capacity(n),
                    None => Vec::new(),
                },
                out: Vec::new(),
                cache: None,
                scratch: Vec::new(),
            };
        }
        let out = match source_frames_hint {
            Some(n) => {
                Vec::with_capacity((n as u64 * 16_000 / u64::from(from_rate.0.max(1))) as usize)
            }
            None => Vec::new(),
        };
        Self {
            from_rate,
            stage: Vec::with_capacity(RESAMPLE_STAGING_FRAMES),
            out,
            cache: None,
            scratch: Vec::new(),
        }
    }

    /// The buffer decoded packets append to, in SOURCE-rate samples.
    pub(super) fn stage(&mut self) -> &mut Vec<f32> {
        &mut self.stage
    }

    /// Drain the staging buffer once it holds a full chunk.
    pub(super) fn flush_full(&mut self) -> Result<()> {
        if self.from_rate.0 == 16_000 || self.stage.len() < RESAMPLE_STAGING_FRAMES {
            return Ok(());
        }
        self.drain()
    }

    /// Drain whatever is still staged and yield the 16 kHz samples.
    pub(super) fn finish(mut self) -> Result<Vec<f32>> {
        if self.from_rate.0 == 16_000 {
            return Ok(self.stage);
        }
        self.drain()?;
        Ok(self.out)
    }

    /// Move the 16 kHz samples produced so far into `dst`, leaving the
    /// accumulator ready to keep decoding.
    ///
    /// For a resampling rate these are the samples already flushed to `out` (the
    /// partial staging chunk stays until it fills or [`Self::finish_into`] runs);
    /// for the 16 kHz passthrough the staged samples ARE the output, so they move
    /// directly. The windowed streaming source calls this after every packet so
    /// peak memory stays O(one window) rather than O(file); the flat
    /// [`Self::finish`] path never touches it. The concatenation of every
    /// `drain_ready_into` plus a final [`Self::finish_into`] is byte-identical to
    /// one [`Self::finish`], because the staged chunk sequence — and therefore
    /// every cached-resampler call — is unchanged.
    pub(super) fn drain_ready_into(&mut self, dst: &mut Vec<f32>) {
        let ready = if self.from_rate.0 == 16_000 {
            &mut self.stage
        } else {
            &mut self.out
        };
        dst.append(ready);
    }

    /// Flush any staged remainder through the resampler and move all remaining
    /// 16 kHz output into `dst`. The streaming counterpart of [`Self::finish`];
    /// idempotent once drained, so an end-of-stream poll may call it repeatedly.
    pub(super) fn finish_into(&mut self, dst: &mut Vec<f32>) -> Result<()> {
        if self.from_rate.0 == 16_000 {
            dst.append(&mut self.stage);
            return Ok(());
        }
        self.drain()?;
        dst.append(&mut self.out);
        Ok(())
    }

    fn drain(&mut self) -> Result<()> {
        if self.stage.is_empty() {
            return Ok(());
        }
        // `resample_with_cache` consumes its input, so hand it the staging
        // buffer and leave a same-sized empty one in its place.
        let chunk = std::mem::replace(&mut self.stage, Vec::with_capacity(RESAMPLE_STAGING_FRAMES));
        resample_with_cache(
            chunk,
            self.from_rate,
            SampleRate(16_000),
            &mut self.cache,
            &mut self.scratch,
        )
        .context("Resampling failed")?;
        self.out.extend_from_slice(&self.scratch);
        Ok(())
    }
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
