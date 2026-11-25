//! Recognition-window adapter over an incremental PCM source.

use anyhow::Result;
use bytes::Bytes;

use super::super::decode_error;
use super::super::pcm_source::{ChannelSelect, PcmSource};
use super::{PcmWindow, PcmWindows, WindowCursor, WindowSpec};
use crate::error::GigasttError;

/// Pull overlapping recognition windows from a container without retaining the whole file.
/// PCM decoding is independent of the host's window geometry and look-behind.
pub(crate) struct FileWindows {
    src: PcmSource,
    cursor: WindowCursor,
}

impl FileWindows {
    pub(crate) fn open(path: &str, spec: WindowSpec, max_audio_secs: Option<f64>) -> Result<Self> {
        Ok(Self {
            src: PcmSource::open(path, max_audio_secs, ChannelSelect::Mono)?,
            cursor: WindowCursor::new(spec),
        })
    }

    pub(crate) fn from_bytes(
        data: Bytes,
        spec: WindowSpec,
        max_audio_secs: Option<f64>,
    ) -> Result<Self> {
        Ok(Self {
            src: PcmSource::from_bytes(data, max_audio_secs, ChannelSelect::Mono)?,
            cursor: WindowCursor::new(spec),
        })
    }

    /// Select just one channel for the streamed split-channel path.
    pub(crate) fn from_bytes_channel(
        data: Bytes,
        spec: WindowSpec,
        max_audio_secs: Option<f64>,
        channel: usize,
    ) -> Result<Self> {
        Ok(Self {
            src: PcmSource::from_bytes(data, max_audio_secs, ChannelSelect::One(channel))?,
            cursor: WindowCursor::new(spec),
        })
    }

    /// Total 16 kHz samples decoded; exact once the stream is drained.
    pub(crate) fn total_16k_samples(&self) -> usize {
        self.src.position()
    }
}

impl PcmWindows for FileWindows {
    fn next_window(&mut self) -> Result<Option<PcmWindow<'_>>, GigasttError> {
        if self.cursor.is_done() {
            return Ok(None);
        }

        // Retain the possible end-anchored tail; the PCM source knows no
        // recognition geometry and only releases what this cursor requests.
        self.src.retain_from(self.cursor.retain_from());
        self.src
            .fill_to(self.cursor.fill_target(), None)
            .map_err(decode_error)?;

        let Some((start, end)) = self.cursor.take(self.src.position(), self.src.is_eof()) else {
            return Ok(None);
        };
        let (buf_start_abs, buf) = self.src.buffer();
        Ok(Some(PcmWindow {
            start_sample: start,
            samples: &buf[start - buf_start_abs..end - buf_start_abs],
        }))
    }
}
