//! Windowed PCM source for the VAD file path.
//!
//! The batch VAD path decodes the whole file, scans every frame for speech, and
//! copies the kept spans into a second whole-file buffer — two O(file)
//! allocations, which is why that path alone still enforced a duration ceiling
//! while the plain file path had none. [`VadWindows`] replaces both with a
//! stream: [`FileWindows`] feeds the container in fixed blocks, [`VadSegmenter`]
//! scores them causally and releases kept audio as soon as its bounded
//! look-ahead allows, and this source hands the result to the decode loop as the
//! same overlapping windows [`SliceWindows`](super::SliceWindows) would yield
//! over the fully compressed buffer.
//!
//! Peak audio memory is one single-pass ceiling of audio plus one stride of
//! look-behind (for the end-anchored trailing window) and well under a second
//! of retained PCM, regardless of file length. The windows themselves are unchanged, so the
//! transcript is too.

use crate::error::GigasttError;
use crate::vad::{SileroVad, VadConfig, VadSegmenter};

use super::stream::{FileWindows, PcmWindow, PcmWindows, WindowCursor, WindowSpec};

/// Block size (@16 kHz) pulled from the container per VAD step. A multiple of
/// the encoder frame stride, large enough to amortize the per-block work and
/// small enough that the retained PCM stays negligible.
const PULL_SAMPLES: usize = 32_000; // 2 s

/// Poll the abort flag every this many pulled blocks (~30 s of audio), matching
/// the batch scan's cadence: interruptible on a multi-hour file without an
/// atomic load per block.
const ABORT_POLL_BLOCKS: usize = 16;

/// [`PcmWindows`] over the silence-free timeline of a streamed container.
pub(crate) struct VadWindows<'a> {
    raw: FileWindows,
    vad: &'a SileroVad,
    seg: VadSegmenter,
    abort: Option<&'a dyn Fn() -> bool>,
    /// Rolling compressed buffer holding `[buf_start_abs, compressed_total)`.
    buf: Vec<f32>,
    /// Absolute sample index (on the compressed timeline) of `buf[0]`.
    buf_start_abs: usize,
    /// Total compressed samples released so far.
    compressed_total: usize,
    /// True once the container is drained and the segmenter has been closed.
    eof: bool,
    /// Set when the VAD model itself failed mid-stream; the caller re-decodes
    /// without VAD rather than returning a truncated transcript.
    vad_failed: bool,
    blocks: usize,
    cursor: WindowCursor,
}

impl<'a> VadWindows<'a> {
    /// Wrap an open container in the VAD stream. `spec` is the decode geometry
    /// applied to the *compressed* timeline — the same one the no-VAD path uses.
    pub(crate) fn new(
        raw: FileWindows,
        vad: &'a SileroVad,
        cfg: &VadConfig,
        spec: WindowSpec,
        abort: Option<&'a dyn Fn() -> bool>,
    ) -> Self {
        Self {
            raw,
            vad,
            seg: VadSegmenter::new(cfg),
            abort,
            buf: Vec::new(),
            buf_start_abs: 0,
            compressed_total: 0,
            eof: false,
            vad_failed: false,
            blocks: 0,
            cursor: WindowCursor::new(spec),
        }
    }

    /// Window geometry for the flat container reads this source performs:
    /// consecutive, non-overlapping blocks covering the stream exactly once.
    pub(crate) fn pull_spec() -> WindowSpec {
        WindowSpec::new(0, PULL_SAMPLES, 0)
    }

    /// Kept speech spans on the **original** timeline, in order. Complete once
    /// the source is drained; used to map decoded word timestamps back off the
    /// compressed timeline.
    pub(crate) fn regions(&self) -> &[(usize, usize)] {
        self.seg.regions()
    }

    /// Total 16 kHz samples read from the container. Exact once drained — this
    /// is the clip's real duration, not the compressed one.
    pub(crate) fn total_16k_samples(&self) -> usize {
        self.raw.total_16k_samples()
    }

    /// True when the caller must re-decode without VAD: either the model failed
    /// mid-stream, or the scan found no speech at all in a non-empty clip (a bad
    /// threshold against continuous speech or a pure tone). Both cases fall back
    /// to the full/chunked decode rather than returning an empty transcript.
    pub(crate) fn needs_fallback(&self) -> bool {
        self.vad_failed || (self.regions().is_empty() && self.total_16k_samples() > 0)
    }

    /// Decode and score until `target` compressed samples are available (or the
    /// stream ends).
    fn fill_to(&mut self, target: usize) -> Result<(), GigasttError> {
        let Self {
            raw,
            vad,
            seg,
            abort,
            buf,
            compressed_total,
            eof,
            vad_failed,
            blocks,
            ..
        } = self;
        while !*eof && *compressed_total < target {
            if let Some(abort) = abort {
                *blocks += 1;
                if *blocks >= ABORT_POLL_BLOCKS {
                    *blocks = 0;
                    if abort() {
                        return Err(GigasttError::Cancelled);
                    }
                }
            }
            let before = buf.len();
            let scanned = match raw.next_window()? {
                Some(w) => seg.push(vad, w.samples, buf),
                None => {
                    *eof = true;
                    seg.finish(vad, raw.total_16k_samples(), buf)
                }
            };
            if let Err(e) = scanned {
                // Same policy as the batch path: a VAD failure never fails the
                // request, it drops VAD for this clip.
                tracing::warn!("VAD failed mid-stream, decoding full audio: {e:#}");
                *vad_failed = true;
                *eof = true;
                buf.truncate(before);
                return Ok(());
            }
            *compressed_total += buf.len() - before;
        }
        Ok(())
    }
}

impl PcmWindows for VadWindows<'_> {
    fn remap_words(&self, words: &mut [crate::inference::WordInfo]) {
        for word in words {
            word.start = crate::vad::remap_compressed_seconds(word.start, self.regions(), 16000.0);
            word.end = crate::vad::remap_compressed_seconds(word.end, self.regions(), 16000.0);
        }
    }

    fn next_window(&mut self) -> Result<Option<PcmWindow<'_>>, GigasttError> {
        if self.cursor.is_done() {
            return Ok(None);
        }

        // Reclaim the consumed prefix; windows only move forward, so everything
        // before `retain_from` is dead.
        let drop = self
            .cursor
            .retain_from()
            .saturating_sub(self.buf_start_abs)
            .min(self.buf.len());
        if drop > 0 {
            self.buf.drain(0..drop);
            self.buf_start_abs += drop;
        }

        self.fill_to(self.cursor.fill_target())?;

        // Nothing survived the VAD: yield no window rather than an empty one.
        // The batch path returns early on empty regions for the same reason —
        // a zero-length encoder run is not a decode.
        if self.eof && self.compressed_total == 0 {
            return Ok(None);
        }

        let Some((start, end)) = self.cursor.take(self.compressed_total, self.eof) else {
            return Ok(None);
        };
        let s = start - self.buf_start_abs;
        let e = end - self.buf_start_abs;
        Ok(Some(PcmWindow {
            start_sample: start,
            samples: &self.buf[s..e],
        }))
    }
}

/// Byte-identity tests against the batch VAD path.
///
/// The Silero session is scripted through the mock runtime, so a whole
/// probability sequence — and therefore a whole speech/silence layout — can be
/// pinned without the model on disk. That makes the comparison that matters run
/// in CI: the same clip, the same config, batch versus streaming, window for
/// window.
#[cfg(test)]
mod tests;
