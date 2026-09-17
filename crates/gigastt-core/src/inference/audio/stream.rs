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

/// Window-cursor arithmetic shared by every window source.
///
/// Pure, so [`SliceWindows`] and the streaming sources yield one and the same
/// window sequence over the same audio: [`WindowCursor::fill_target`] says how
/// far a source must decode before the next window can be decided, and
/// [`WindowCursor::take`] turns "here is what is available" into that window.
///
/// No window of a long input is shorter than the full window length. Once the
/// stream length is known, the last window either **absorbs** the remainder
/// (it may run to the single-pass ceiling, 30 s, exactly as a short file is
/// decoded in one pass) or, when the remainder is longer than that allows, is
/// **anchored** to the end: it starts at `len - window` (rounded up to the
/// encoder frame grid) and overlaps the previous window by more than the
/// nominal overlap. A source must therefore keep audio from
/// [`WindowCursor::retain_from`] on.
pub(crate) struct WindowCursor {
    spec: WindowSpec,
    /// Absolute start (@16 kHz) of the next window to yield.
    next_start: usize,
    /// Start of the most recently yielded window, once there is one.
    prev_start: Option<usize>,
    done: bool,
}

impl WindowCursor {
    pub(crate) fn new(spec: WindowSpec) -> Self {
        Self {
            spec,
            next_start: 0,
            prev_start: None,
            done: false,
        }
    }

    /// Longest window the last one may grow to: the single-pass ceiling, and
    /// never less than the window itself (a partition spec has no ceiling).
    fn last_window_max(&self) -> usize {
        self.spec.single_pass_max().max(self.spec.window())
    }

    #[cfg(feature = "file-decode")]
    pub(crate) fn is_done(&self) -> bool {
        self.done
    }

    #[cfg(test)]
    pub(crate) fn next_start(&self) -> usize {
        self.next_start
    }

    /// Earliest absolute sample the source must still hold for the next
    /// [`WindowCursor::take`]: one stride before the nominal next start, because
    /// an end-anchored trailing window can begin anywhere after the previous
    /// window's start.
    #[cfg(feature = "file-decode")]
    pub(crate) fn retain_from(&self) -> usize {
        self.next_start.saturating_sub(self.spec.stride())
    }

    /// Decode at least this many samples before calling [`WindowCursor::take`]:
    /// one past the longest last window that could start at `next_start`, so
    /// `end == total` is distinguishable from a mid-stream boundary and the
    /// absorb-or-anchor decision for the remainder can be made.
    #[cfg(feature = "file-decode")]
    pub(crate) fn fill_target(&self) -> usize {
        self.next_start
            .saturating_add(self.last_window_max())
            .saturating_add(1)
    }

    /// The next `[start, end)` window given the samples decoded so far
    /// (`avail_end`) and whether the stream is exhausted, or `None` once every
    /// window has been yielded.
    pub(crate) fn take(&mut self, avail_end: usize, eof: bool) -> Option<(usize, usize)> {
        if self.done {
            return None;
        }
        let start = self.next_start;
        if start >= avail_end {
            // Reachable only at EOF, once the last window has been yielded.
            self.done = true;
            return None;
        }
        // Because the caller decoded one sample past the longest possible last
        // window (or hit EOF), a remainder within reach makes this window the
        // last one: it absorbs the remainder up to the single-pass ceiling
        // when that keeps it at least one window long (for the first window
        // that is `decode_words`' non-windowed branch, byte-identical). A
        // remainder shorter than one window is re-anchored to the end instead.
        let remaining = avail_end - start;
        let absorb = eof
            && remaining <= self.last_window_max()
            && (remaining >= self.spec.window() || self.prev_start.is_none());
        let (start, end) = if absorb {
            (start, avail_end)
        } else {
            let end = (start + self.spec.window()).min(avail_end);
            if eof && end == avail_end {
                (self.anchored_start(start, avail_end), avail_end)
            } else {
                (start, end)
            }
        };
        if eof && end == avail_end {
            self.done = true;
        } else {
            self.next_start = start + self.spec.stride();
        }
        self.prev_start = Some(start);
        Some((start, end))
    }

    /// Start of an end-anchored trailing window: `len - window` rounded up to
    /// the frame grid, so the window is at most one full window long and its
    /// origin maps to an integral encoder frame. It always lies strictly after
    /// the previous window's start (the previous window did not reach the end)
    /// and at or before `nominal`. The first window keeps the nominal start:
    /// there is nothing to anchor against. A zero-overlap spec is a partition
    /// (the flat container pulls feeding the VAD), not decode context, so it is
    /// never anchored either: re-reading the tail would hand the same audio
    /// out twice.
    fn anchored_start(&self, nominal: usize, len: usize) -> usize {
        let Some(prev_start) = self.prev_start else {
            return nominal;
        };
        if len < self.spec.window() || self.spec.overlap() == 0 {
            return nominal;
        }
        let anchored = (len - self.spec.window()).div_ceil(FRAME_SAMPLES) * FRAME_SAMPLES;
        debug_assert!(anchored > prev_start && anchored <= nominal);
        anchored.max(prev_start + FRAME_SAMPLES).min(nominal)
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

    /// Lend the next window, or `Ok(None)` once the stream is exhausted.
    fn next_window(&mut self) -> Result<Option<PcmWindow<'_>>, GigasttError>;
}

/// [`PcmWindows`] over a fully materialized buffer: the whole buffer is
/// available up front, so the cursor sees every call as end-of-stream.
pub(crate) struct SliceWindows<'a> {
    samples: &'a [f32],
    cursor: WindowCursor,
}

impl<'a> SliceWindows<'a> {
    pub(crate) fn new(samples: &'a [f32], spec: WindowSpec) -> Self {
        Self {
            samples,
            cursor: WindowCursor::new(spec),
        }
    }
}

impl PcmWindows for SliceWindows<'_> {
    fn next_window(&mut self) -> Result<Option<PcmWindow<'_>>, GigasttError> {
        let total = self.samples.len();
        if total == 0 {
            return Ok(None);
        }
        Ok(self.cursor.take(total, true).map(|(start, end)| PcmWindow {
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
