//! Incremental file/bytes PCM decoding, independent of recognition windows.
//!
//! The caller chooses how many absolute 16 kHz samples to make available and
//! when to release consumed samples. The source retains codec packet cadence,
//! channel selection, budget accounting and resampler EOF flushing.

use anyhow::Result;
use symphonia::core::codecs::audio::AudioDecoder;
use symphonia::core::formats::FormatReader;

use super::opus::OpusStream;
use super::resample::ResampleTo16k;
use super::wave::WaveSource;

mod fill;
mod open;
#[cfg(test)]
mod tests;

/// Which channel a PCM source yields.
///
/// The default file pipeline wants the mono mix; `channels=split` wants each
/// channel on its own, and needs it *streamed* — materializing every channel of
/// the whole file is what pinned that path to a duration ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChannelSelect {
    /// Mean of every channel present in the packet.
    Mono,
    /// A single channel, by zero-based index.
    One(usize),
}

/// The decode engine behind [`PcmSource`]: a streaming symphonia loop, a
/// WAVE-family ryf pull, or Opus via `opus-rs`.
enum Source {
    /// Container decoded packet-by-packet, resampled to 16 kHz as it goes.
    Streaming {
        format: Box<dyn FormatReader>,
        decoder: Box<dyn AudioDecoder>,
        track_id: u32,
        /// Source (container) sample rate — the units the length budget counts.
        sample_rate: u32,
        /// Running SOURCE-rate frame count, tracked separately from the 16 kHz
        /// accumulator because the length budget is expressed in source frames.
        source_frames: usize,
        /// Source-rate frame budget. `usize::MAX` when the caller imposed no
        /// limit — the streaming path is O(one window), so length is unbounded
        /// by default; the flat drain and the whole-buffer callers pass a finite
        /// budget (see `max_samples_for_secs`).
        max_samples: usize,
        /// The seconds limit `max_samples` was derived from, echoed verbatim in
        /// [`AudioTooLong`](crate::error::GigasttError::AudioTooLong) on a trip.
        limit_secs: f64,
        /// Boxed: it carries the heavyweight rubato FIR state, so keeping it
        /// behind a pointer keeps the `Streaming` variant small.
        resampler: Box<ResampleTo16k>,
        /// Per-packet interleaved scratch, hoisted out of the decode loop.
        interleaved: Vec<f32>,
    },
    /// OGG/Opus: symphonia demuxes it but ships no decoder, so packets go
    /// through the `opus-rs` fallback ([`OpusStream`]) and are mixed to mono
    /// and resampled as they arrive — same shape as `Streaming`, different
    /// decoder.
    Opus {
        format: Box<dyn FormatReader>,
        track_id: u32,
        /// Boxed: it carries Opus decoder state.
        stream: Box<OpusStream>,
        /// Running decoded frame count at the Opus decode rate (48 kHz), which
        /// is what the budget below counts and what a trip reports.
        decoded_48k: usize,
        max_samples: usize,
        limit_secs: f64,
        resampler: Box<ResampleTo16k>,
        /// Decoded mono samples not yet staged. The whole-buffer path fed the
        /// resampler in exact `RESAMPLE_STAGING_FRAMES` chunks; holding the
        /// remainder here reproduces that chunk sequence packet-by-packet, so
        /// the flush boundaries — and with them the resampled output — do not
        /// move. Bounded by one chunk plus one packet.
        pending: Vec<f32>,
    },
    /// WAVE family (PCM/IEEE, G.711, G.722, ADPCM, RF64/RIFX/Wave64) via ryf.
    Wave(Box<WaveSource>),
}

/// Container-backed rolling 16 kHz PCM buffer.
///
/// Fill requests can overshoot by a codec block. Callers release consumed
/// prefixes as they progress; the source does not impose window geometry.
pub(super) struct PcmSource {
    src: Source,
    eof: bool,
    finished: bool,
    buf: Vec<f32>,
    buf_start_abs: usize,
    decoded_16k_total: usize,
    channel: ChannelSelect,
}

impl PcmSource {
    fn new(src: Source, channel: ChannelSelect) -> Self {
        Self {
            src,
            eof: false,
            finished: false,
            buf: Vec::new(),
            buf_start_abs: 0,
            decoded_16k_total: 0,
            channel,
        }
    }

    /// Exclusive absolute end position of the decoded PCM.
    pub(super) fn position(&self) -> usize {
        self.decoded_16k_total
    }

    /// True once the container has ended and the resampler has flushed.
    pub(super) fn is_eof(&self) -> bool {
        self.finished
    }

    /// Absolute start position and the retained PCM.
    pub(super) fn buffer(&self) -> (usize, &[f32]) {
        (self.buf_start_abs, &self.buf)
    }

    /// Release PCM before `start`, keeping any later samples already decoded.
    pub(super) fn retain_from(&mut self, start: usize) {
        let drop = start.saturating_sub(self.buf_start_abs).min(self.buf.len());
        if drop > 0 {
            self.buf.drain(..drop);
            self.buf_start_abs += drop;
        }
    }

    /// Materialize the remaining PCM, for flat-decode callers with a finite
    /// source-frame budget. Does not need any recognition window geometry.
    pub(super) fn drain_to_vec(
        mut self,
        abort: Option<&(dyn Fn() -> bool + Sync)>,
    ) -> Result<Vec<f32>> {
        self.fill_to(usize::MAX, abort)?;
        Ok(self.buf)
    }
}
