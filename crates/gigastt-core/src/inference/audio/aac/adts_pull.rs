//! Bounded ADTS pull. Framing only partitions bytes; the codec validates them.
use std::io::{ErrorKind, Read};

use anyhow::Result;
use syom::{AacError, Decoder, Frame};

use super::super::stream::ChannelSelect;
use super::{AacBlock, BATCH_FRAMES, append_plane, map_aac, stream_options};

const READ_BYTES: usize = 4096;
const LOOKAHEAD: usize = 5;
const STASH_BYTES: usize = READ_BYTES + LOOKAHEAD;

pub(super) struct AdtsPull<R> {
    reader: R,
    decoder: Decoder,
    channel: ChannelSelect,
    stash: Vec<u8>,
    scratch: Vec<u8>,
    scan: usize,
    fed: usize,
    remaining: usize,
    input_eof: bool,
    finished: bool,
    batch: Option<AacBlock>,
    ready: Option<AacBlock>,
    error: Option<anyhow::Error>,
    #[cfg(test)]
    forwarded: Option<Vec<u8>>,
}

impl<R: Read> AdtsPull<R> {
    pub(super) fn new(reader: R, channel: ChannelSelect, max_audio_secs: Option<f64>) -> Self {
        Self {
            reader,
            decoder: Decoder::new(stream_options(max_audio_secs)),
            channel,
            stash: Vec::with_capacity(STASH_BYTES),
            scratch: vec![0; READ_BYTES],
            scan: 0,
            fed: 0,
            remaining: 0,
            input_eof: false,
            finished: false,
            batch: None,
            ready: None,
            error: None,
            #[cfg(test)]
            forwarded: None,
        }
    }

    pub(super) fn recv(&mut self) -> Result<Option<AacBlock>> {
        loop {
            if let Some(block) = self.ready.take() {
                return Ok(Some(block));
            }
            if let Some(error) = self.error.take() {
                return Err(error);
            }
            if self.finished {
                return Ok(self.batch.take());
            }
            let piece = match self.next_piece() {
                Ok(piece) => piece,
                Err(error) => {
                    self.fail(error);
                    continue;
                }
            };
            let result = match piece {
                Some((start, end)) => {
                    self.fed = end;
                    #[cfg(test)]
                    if let Some(bytes) = &mut self.forwarded {
                        bytes.extend_from_slice(&self.stash[start..end]);
                    }
                    let (batch, ready, channel) = (&mut self.batch, &mut self.ready, self.channel);
                    self.decoder
                        .feed(&self.stash[start..end], |frame| {
                            accept(batch, ready, channel, frame)
                        })
                        .map(|_| ())
                }
                None => {
                    self.finished = true;
                    let (batch, ready, channel) = (&mut self.batch, &mut self.ready, self.channel);
                    self.decoder
                        .finish(|frame| accept(batch, ready, channel, frame))
                        .map(|_| ())
                }
            };
            if let Err(error) = result {
                self.fail(map_aac(error));
            }
        }
    }

    fn fail(&mut self, error: anyhow::Error) {
        self.finished = true;
        self.batch = None; // Only complete logical blocks precede real errors.
        self.error = Some(error);
    }

    /// All bytes are forwarded exactly once. Already-fed lookahead can frame
    /// a header crossing a read without withholding errors from the codec.
    fn next_piece(&mut self) -> Result<Option<(usize, usize)>> {
        loop {
            let len = self.stash.len();
            if self.remaining > 0 {
                let count = self.remaining.min(len - self.fed);
                if count > 0 {
                    self.remaining -= count;
                    self.scan = self.fed + count;
                    return Ok(Some((self.fed, self.scan)));
                }
                if self.input_eof {
                    return Ok(None);
                }
                self.scan = len;
                self.refill()?;
                continue;
            }
            if self.input_eof && self.fed == len {
                return Ok(None);
            }
            if len - self.scan >= 6 {
                let header =
                    (self.scan..=len - 6).find(|&at| syom::sniff_is_adts(&self.stash[at..]));
                if let Some(at) = header {
                    if at > self.fed {
                        self.scan = at;
                        return Ok(Some((self.fed, at)));
                    }
                    // Public 13-bit framing length only; no AAC/header validation here.
                    let h = &self.stash[at..];
                    let size = (usize::from(h[3] & 3) << 11)
                        | (usize::from(h[4]) << 3)
                        | (usize::from(h[5]) >> 5);
                    let end = at + size;
                    let available_end = end.min(len);
                    anyhow::ensure!(available_end > self.fed, "ADTS framing made no progress");
                    self.remaining = end - available_end;
                    self.scan = available_end;
                    return Ok(Some((self.fed, available_end)));
                }
            }
            // Do not cross back into a previously framed AU when retaining
            // the last five unframed bytes. They have already reached feed().
            self.scan = self.scan.max(len.saturating_sub(LOOKAHEAD));
            if self.fed < len {
                return Ok(Some((self.fed, len)));
            }
            if self.input_eof {
                return Ok(None);
            }
            self.refill()?;
        }
    }

    fn refill(&mut self) -> Result<()> {
        anyhow::ensure!(
            self.fed == self.stash.len(),
            "ADTS read would withhold unfed bytes"
        );
        anyhow::ensure!(
            self.stash.len() - self.scan <= LOOKAHEAD,
            "ADTS compressed lookahead exceeded"
        );
        self.stash.drain(..self.scan);
        self.fed -= self.scan;
        self.scan = 0;
        let count = loop {
            match self.reader.read(&mut self.scratch) {
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                other => break other.map_err(AacError::from).map_err(map_aac)?,
            }
        };
        self.input_eof = count == 0;
        anyhow::ensure!(
            self.stash.len() + count <= STASH_BYTES,
            "ADTS compressed stash exceeded"
        );
        self.stash.extend_from_slice(&self.scratch[..count]);
        Ok(())
    }
}

fn accept(
    batch: &mut Option<AacBlock>,
    ready: &mut Option<AacBlock>,
    channel: ChannelSelect,
    frame: Frame<'_>,
) -> syom::Result<()> {
    if let Some(block) = batch
        && (block.rate, block.channels) != (frame.sample_rate, frame.planar.len())
    {
        return Err(AacError::decode("AAC rate or channel count changed"));
    }
    let block = batch.get_or_insert_with(|| AacBlock {
        frames: 0,
        samples: Vec::with_capacity(BATCH_FRAMES),
        rate: frame.sample_rate,
        channels: frame.planar.len(),
    });
    block.frames += frame.samples;
    append_plane(&mut block.samples, &frame, channel);
    if block.frames >= BATCH_FRAMES {
        if ready.is_some() {
            return Err(AacError::decode("ADTS feed exceeded one ready block"));
        }
        *ready = batch.take();
    }
    Ok(())
}

#[cfg(test)]
#[path = "adts_pull_tests.rs"]
mod tests;
