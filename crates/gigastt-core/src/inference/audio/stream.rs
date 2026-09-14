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

    /// Largest total (samples @16 kHz) that stays on the single-pass branch — one
    /// encoder Run over the whole buffer rather than overlapping windows.
    pub(crate) fn single_pass_max(&self) -> usize {
        self.single_pass_max
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

    /// Sentinel geometry for flat (`drain_to_vec`) decode, which never windows.
    /// Its window/stride are never read; only [`FileWindows::drain_to_vec`] uses
    /// a `FileWindows` built with it.
    #[cfg(feature = "file-decode")]
    pub(crate) fn flat() -> Self {
        Self::new(usize::MAX, usize::MAX, 0)
    }
}

/// Window-cursor arithmetic for a source that decodes as it goes.
///
/// Pure and shared, so every streaming source yields the one window sequence
/// [`SliceWindows`] would over the same fully-decoded buffer:
/// [`WindowCursor::fill_target`] says how far the source must decode before the
/// next window can be decided, and [`WindowCursor::take`] turns "here is what
/// is available" into that window.
#[cfg(feature = "file-decode")]
pub(crate) struct WindowCursor {
    spec: WindowSpec,
    /// Absolute start (@16 kHz) of the next window to yield.
    next_start: usize,
    /// True until the first window's single-pass-vs-windowed decision is made.
    first: bool,
    done: bool,
}

#[cfg(feature = "file-decode")]
impl WindowCursor {
    pub(crate) fn new(spec: WindowSpec) -> Self {
        Self {
            spec,
            next_start: 0,
            first: true,
            done: false,
        }
    }

    pub(crate) fn is_done(&self) -> bool {
        self.done
    }

    pub(crate) fn next_start(&self) -> usize {
        self.next_start
    }

    pub(crate) fn spec(&self) -> WindowSpec {
        self.spec
    }

    /// Decode at least this many samples before calling [`WindowCursor::take`]:
    /// one past the window end, so `end == total` is distinguishable from a
    /// mid-stream boundary, and enough for the first window to decide
    /// single-pass vs windowed.
    pub(crate) fn fill_target(&self) -> usize {
        if self.first {
            self.spec
                .single_pass_max()
                .saturating_add(1)
                .max(self.spec.window().saturating_add(1))
        } else {
            self.next_start + self.spec.window() + 1
        }
    }

    /// The next `[start, end)` window given the samples decoded so far
    /// (`avail_end`) and whether the stream is exhausted, or `None` once every
    /// window has been yielded.
    pub(crate) fn take(&mut self, avail_end: usize, eof: bool) -> Option<(usize, usize)> {
        if self.done {
            return None;
        }
        let start = self.next_start;
        if self.first {
            self.first = false;
            if eof && avail_end <= self.spec.single_pass_max() {
                // The whole stream fits the single-pass ceiling: one window over
                // all of it. `decode_words_streaming` then runs one encoder pass
                // with frame offset 0 and stitches onto an empty list —
                // byte-identical to `decode_words`' non-windowed branch.
                self.done = true;
                return Some((start, avail_end));
            }
        }
        if start >= avail_end {
            // Reachable only at EOF, once the last window has been yielded.
            self.done = true;
            return None;
        }
        // Because the caller decoded one sample past `start + window` (or hit
        // EOF), `end == avail_end` holds only when this window reaches the end.
        let end = (start + self.spec.window()).min(avail_end);
        if eof && end == avail_end {
            self.done = true;
        } else {
            self.next_start = start + self.spec.stride();
        }
        Some((start, end))
    }
}

/// Which channel a windowed decode yields.
///
/// The default file pipeline wants the mono mix; `channels=split` wants each
/// channel on its own, and needs it *streamed* — materializing every channel of
/// the whole file is what pinned that path to a duration ceiling.
#[cfg(feature = "file-decode")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChannelSelect {
    /// Mean of every channel present in the packet.
    Mono,
    /// A single channel, by zero-based index.
    One(usize),
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
    /// Convert a provisional hypothesis to the input timeline (VAD sources
    /// remove silence before decoding). Plain sources already use that clock.
    fn remap_words(&self, _words: &mut [crate::inference::WordInfo]) {}

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

#[cfg(feature = "file-decode")]
mod file;
#[cfg(feature = "file-decode")]
pub(crate) use file::FileWindows;

#[cfg(test)]
mod tests;

/// [`FileWindows`] streaming-decode tests: prove that pulling windows from the
/// container yields byte-identical geometry to [`SliceWindows`] over the same
/// fully-decoded buffer — and, below the single-pass ceiling, exactly one window
/// over the whole buffer (matching `Engine::decode_words`' non-windowed branch).
/// No model is required.
#[cfg(all(test, feature = "file-decode"))]
mod file_windows_tests;
