//! Windowed PCM source for long-form file decode.
//!
//! Long-form decode used to own its `while start < total { … }` loop directly
//! over one `&[f32]` holding the whole file. This module puts that window
//! geometry behind a small source trait so the decode loop no longer assumes
//! the PCM is fully materialized: [`SliceWindows`] is the buffer-backed source
//! used today, and a decoder-backed source can be added later without touching
//! the loop.
//!
//! [`PcmWindows::next_window`] **lends** its window — the returned
//! [`PcmWindow`] borrows the source — so a source can hand out a view into a
//! decoder's own scratch buffer instead of copying. That borrow shape is why
//! this is not an [`Iterator`].

use crate::error::GigasttError;
use crate::inference::{ENCODER_SUBSAMPLING, HOP_LENGTH};

/// Samples per encoder output frame (`HOP_LENGTH * ENCODER_SUBSAMPLING`,
/// 640 @16 kHz). Window starts are multiples of this so each window's frame
/// offset is integral.
const FRAME_SAMPLES: usize = HOP_LENGTH * ENCODER_SUBSAMPLING;

/// Long-form window geometry, all in samples @16 kHz.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WindowSpec {
    single_pass_max: usize,
    window: usize,
    stride: usize,
}

impl WindowSpec {
    /// Build a spec from the single-pass ceiling, the window length, and the
    /// overlap retained between consecutive windows.
    ///
    /// The stride (`window - overlap`) is aligned **down** to an encoder-frame
    /// boundary so every window start maps to an integral frame offset;
    /// otherwise the offset would drift by a sub-frame each hop. It is clamped
    /// to one frame so a mis-specified `overlap >= window` cannot produce a
    /// zero-stride (non-advancing) source.
    pub(crate) fn new(single_pass_max: usize, window: usize, overlap: usize) -> Self {
        let stride =
            (window.saturating_sub(overlap) / FRAME_SAMPLES * FRAME_SAMPLES).max(FRAME_SAMPLES);
        Self {
            single_pass_max,
            window,
            stride,
        }
    }

    /// Window length in samples.
    pub(crate) fn window(&self) -> usize {
        self.window
    }

    /// Frame-aligned distance between consecutive window starts.
    pub(crate) fn stride(&self) -> usize {
        self.stride
    }

    /// Samples shared by two consecutive windows (`window - stride`). Equals the
    /// requested overlap whenever that overlap is already frame-aligned.
    pub(crate) fn overlap(&self) -> usize {
        self.window.saturating_sub(self.stride)
    }

    /// True when `total` samples are short enough for the single-pass (one
    /// encoder Run over the whole buffer) branch.
    pub(crate) fn is_single_pass(&self, total: usize) -> bool {
        total <= self.single_pass_max
    }
}

/// One decode window lent by a [`PcmWindows`] source.
pub(crate) struct PcmWindow<'a> {
    /// Absolute offset of `samples[0]` in the stream, in samples @16 kHz.
    pub(crate) start_sample: usize,
    /// The window's PCM.
    pub(crate) samples: &'a [f32],
}

/// A source of overlapping decode windows.
pub(crate) trait PcmWindows {
    /// The window geometry this source yields.
    fn spec(&self) -> WindowSpec;

    /// Lend the next window, or `Ok(None)` once the stream is exhausted.
    fn next_window(&mut self) -> Result<Option<PcmWindow<'_>>, GigasttError>;
}

/// [`PcmWindows`] over a fully materialized buffer.
///
/// Yields exactly the `(start, end)` sequence of the loop it replaced:
/// `while start < total { end = min(start + window, total); …; if end == total
/// { break } start += stride }`.
pub(crate) struct SliceWindows<'a> {
    samples: &'a [f32],
    spec: WindowSpec,
    next_start: usize,
    done: bool,
}

impl<'a> SliceWindows<'a> {
    pub(crate) fn new(samples: &'a [f32], spec: WindowSpec) -> Self {
        Self {
            samples,
            spec,
            next_start: 0,
            done: false,
        }
    }
}

impl PcmWindows for SliceWindows<'_> {
    fn spec(&self) -> WindowSpec {
        self.spec
    }

    fn next_window(&mut self) -> Result<Option<PcmWindow<'_>>, GigasttError> {
        let total = self.samples.len();
        if self.done || self.next_start >= total {
            return Ok(None);
        }
        let start = self.next_start;
        let end = (start + self.spec.window()).min(total);
        // The replaced loop stopped on `end == total` rather than on the next
        // start passing `total`, so a window that reaches the end is the last
        // one even when `start + stride` is still short of `total`.
        if end == total {
            self.done = true;
        } else {
            self.next_start = start + self.spec.stride();
        }
        Ok(Some(PcmWindow {
            start_sample: start,
            samples: &self.samples[start..end],
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The loop `SliceWindows` replaced, verbatim, as the reference oracle.
    fn legacy_windows(total: usize, window: usize, stride: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut start = 0usize;
        while start < total {
            let end = (start + window).min(total);
            out.push((start, end - start));
            if end == total {
                break;
            }
            start += stride;
        }
        out
    }

    fn observed(total: usize, spec: WindowSpec) -> Vec<(usize, usize)> {
        let samples = vec![0.0f32; total];
        let mut src = SliceWindows::new(&samples, spec);
        let mut out = Vec::new();
        while let Some(w) = src.next_window().expect("slice source never fails") {
            out.push((w.start_sample, w.samples.len()));
        }
        out
    }

    /// The ort long-form geometry, spelled out so this test pins the numbers the
    /// engine feeds rather than following it.
    fn ort_spec() -> WindowSpec {
        WindowSpec::new(16000 * 30, 16000 * 24, 16000 * 2)
    }

    #[test]
    fn test_window_spec_stride_is_frame_aligned() {
        let spec = ort_spec();
        assert_eq!(spec.window(), 384_000);
        assert_eq!(spec.stride(), 352_000);
        assert_eq!(spec.stride() % FRAME_SAMPLES, 0);
        // 2 s overlap is already frame-aligned, so it survives the alignment.
        assert_eq!(spec.overlap(), 32_000);
        // ANE geometry: 30 s window, same 2 s overlap.
        let ane = WindowSpec::new(16000 * 30, 16000 * 30, 16000 * 2);
        assert_eq!(ane.stride(), 448_000);
        assert_eq!(ane.stride() % FRAME_SAMPLES, 0);
        assert_eq!(ane.overlap(), 32_000);
    }

    #[test]
    fn test_window_spec_single_pass_boundary() {
        let spec = ort_spec();
        assert!(spec.is_single_pass(0));
        assert!(spec.is_single_pass(479_999));
        assert!(spec.is_single_pass(480_000)); // exactly 30 s stays single-pass
        assert!(!spec.is_single_pass(480_001));
    }

    #[test]
    fn test_window_spec_degenerate_overlap_still_advances() {
        // overlap >= window would give a zero stride and a non-advancing source.
        let spec = WindowSpec::new(0, 1000, 4000);
        assert_eq!(spec.stride(), FRAME_SAMPLES);
        assert_eq!(observed(5000, spec).len(), 8);
    }

    #[test]
    fn test_slice_windows_matches_legacy_loop_swept() {
        let spec = ort_spec();
        let (window, stride) = (spec.window(), spec.stride());
        let mut lengths: Vec<usize> = Vec::new();
        // Coarse sweep across 0..3x window.
        let mut n = 0usize;
        while n <= 3 * window {
            lengths.push(n);
            n += 4_001; // deliberately coprime with the stride/frame grid
        }
        // Exact boundaries: window/stride multiples ± 1, the single-pass
        // threshold, and the degenerate sub-frame tail band.
        for anchor in [
            0,
            1,
            FRAME_SAMPLES,
            window,
            stride,
            stride + window,
            2 * stride,
            2 * stride + window,
            480_000, // single-pass branch boundary
        ] {
            for d in [-1isize, 0, 1] {
                let v = anchor as isize + d;
                if v >= 0 {
                    lengths.push(v as usize);
                }
            }
        }
        lengths.extend(704_000..=704_320); // degenerate band, every length
        lengths.sort_unstable();
        lengths.dedup();

        for total in lengths {
            assert_eq!(
                observed(total, spec),
                legacy_windows(total, window, stride),
                "window sequence diverged at total={total}"
            );
        }
    }

    #[test]
    fn test_slice_windows_empty_yields_nothing() {
        assert!(observed(0, ort_spec()).is_empty());
    }

    #[test]
    fn test_slice_windows_stop_exactly_at_the_end() {
        let spec = ort_spec();
        let total = 1_440_000; // 90 s
        let seq = observed(total, spec);
        assert!(!seq.is_empty());
        // Exactly one window reaches the end, and it is the last one emitted.
        assert_eq!(
            seq.iter()
                .filter(|(start, len)| start + len == total)
                .count(),
            1
        );
        let (start, len) = seq[seq.len() - 1];
        assert_eq!(start + len, total);
        // Every start is frame-aligned, so the frame offset is integral.
        for (start, _) in &seq {
            assert_eq!(start % FRAME_SAMPLES, 0);
        }
    }
}
