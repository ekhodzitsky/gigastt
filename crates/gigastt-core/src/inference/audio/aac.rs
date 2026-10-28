//! Bounded AAC ingest. syom handles AAC transports; Symphonia retains AAC in
//! Matroska and non-AAC MP4 tracks. The callback codec feeds windowed consumers
//! in bounded batches, while channel processing stays synchronous and incremental.

use std::sync::mpsc::{self, Receiver};
use std::thread::{self, JoinHandle};

use anyhow::{Context, Result};
use bytes::Bytes;
use syom::{AacError, DecodeOptions, Frame};

use super::resample::{ResampleTo16k, SampleRate};
use super::stream::ChannelSelect;
use super::{MAX_SAMPLE_RATE, resolve_budget};
use crate::error::GigasttError;

mod channels;
mod input;
mod probe;
pub(super) use channels::{try_decode_planes_bytes, try_decode_planes_path, try_scan_bytes};
use input::{Reader, decode_reader};
use probe::sniff_slice;
pub(super) use probe::{ProbeHit, probe_bytes, probe_file, sniff_file};

// Amortize thread handoffs without retaining duration-sized PCM. Two queued
// batches plus the producer's current batch retain PCM independent of duration.
const BATCH_FRAMES: usize = 24 * 1024;

pub(super) struct AacBlock {
    pub(super) frames: usize,
    pub(super) samples: Vec<f32>,
    rate: u32,
    channels: usize,
}

pub(super) enum AacRecv {
    Block(AacBlock),
    Eof,
}

/// Own the worker even during initialization and error exits.
struct Worker {
    rx: Option<Receiver<Result<AacBlock>>>,
    join: Option<JoinHandle<()>>,
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.rx.take(); // Unblock a producer waiting to send before joining it.
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

impl Worker {
    fn recv(&mut self) -> Result<Option<AacBlock>> {
        let Some(rx) = &self.rx else { return Ok(None) };
        match rx.recv() {
            Ok(result) => result.map(Some),
            Err(_) => {
                self.rx.take();
                if let Some(join) = self.join.take() {
                    anyhow::ensure!(join.join().is_ok(), "AAC decode thread panicked");
                }
                Ok(None)
            }
        }
    }
}

pub(super) struct AacSource {
    worker: Worker,
    pending: Option<AacBlock>,
    pub(super) source_frames: usize,
    pub(super) sample_rate: u32,
    pub(super) max_samples: usize,
    pub(super) limit_secs: f64,
    pub(super) resampler: Box<ResampleTo16k>,
}

impl AacSource {
    pub(super) fn recv_block(&mut self) -> Result<AacRecv> {
        let next = match self.pending.take() {
            Some(block) => Some(block),
            None => self.worker.recv()?,
        };
        Ok(match next {
            Some(block) => AacRecv::Block(block),
            None => AacRecv::Eof,
        })
    }
}

pub(super) fn try_open_bytes(
    data: Bytes,
    channel: ChannelSelect,
    max_audio_secs: Option<f64>,
) -> Result<Option<AacSource>> {
    if !sniff_slice(&data) {
        return Ok(None);
    }
    open_input(Reader::bytes(data), channel, max_audio_secs)
}

pub(super) fn try_open_path(
    path: &str,
    channel: ChannelSelect,
    max_audio_secs: Option<f64>,
) -> Result<Option<AacSource>> {
    let mut file =
        std::fs::File::open(path).with_context(|| format!("Failed to open audio file: {path}"))?;
    if !sniff_file(&mut file)? {
        return Ok(None);
    }
    open_input(Reader::File(file), channel, max_audio_secs)
}

fn open_input(
    mut input: Reader,
    channel: ChannelSelect,
    max_audio_secs: Option<f64>,
) -> Result<Option<AacSource>> {
    let (tx, rx) = mpsc::sync_channel(2);
    let join = thread::Builder::new()
        .name("gigastt-aac".into())
        .spawn(move || {
            let mut batch: Option<AacBlock> = None;
            let result = decode_reader(&mut input, stream_options(max_audio_secs), |frame| {
                if let Some(block) = &batch
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
                if block.frames >= BATCH_FRAMES
                    && let Some(block) = batch.take()
                {
                    tx.send(Ok(block))
                        .map_err(|_| AacError::decode("AAC consumer closed"))?;
                }
                Ok(())
            });
            match result {
                Ok(()) => {
                    if let Some(block) = batch {
                        let _ = tx.send(Ok(block));
                    }
                }
                Err(error) => {
                    let _ = tx.send(Err(map_aac(error)));
                }
            }
        })
        .context("Failed to start AAC decode thread")?;
    let mut worker = Worker {
        rx: Some(rx),
        join: Some(join),
    };
    let first = match worker.recv() {
        Ok(Some(block)) => block,
        Ok(None) => anyhow::bail!("No audio track found"),
        Err(error) if matches!(error.downcast_ref::<AacError>(), Some(AacError::NotAac)) => {
            return Ok(None);
        }
        Err(error) => return Err(error),
    };
    anyhow::ensure!(
        first.rate > 0 && first.rate <= MAX_SAMPLE_RATE,
        "Unsupported sample rate: {}Hz",
        first.rate
    );
    let (max_samples, limit_secs) = resolve_budget(max_audio_secs, first.rate);
    tracing::info!(
        "Audio (aac): {}Hz, {}ch (streaming windows)",
        first.rate,
        first.channels
    );
    Ok(Some(AacSource {
        sample_rate: first.rate,
        resampler: Box::new(ResampleTo16k::new(SampleRate(first.rate), None)),
        pending: Some(first),
        worker,
        source_frames: 0,
        max_samples,
        limit_secs,
    }))
}

fn append_plane(out: &mut Vec<f32>, frame: &Frame<'_>, channel: ChannelSelect) {
    match channel {
        ChannelSelect::One(index) => {
            if let Some(plane) = frame.planar.get(index) {
                out.extend_from_slice(plane);
            }
        }
        ChannelSelect::Mono => {
            let Some(first) = frame.planar.first() else {
                return;
            };
            let start = out.len();
            out.extend_from_slice(first);
            for plane in frame.planar.iter().skip(1) {
                for (sample, value) in out[start..].iter_mut().zip(plane.iter()) {
                    *sample += value;
                }
            }
            let count = frame.planar.len() as f32;
            for sample in &mut out[start..] {
                *sample /= count;
            }
        }
    }
}

fn stream_options(max_audio_secs: Option<f64>) -> DecodeOptions {
    let mut opts = DecodeOptions::unbounded().with_max_sample_rate(MAX_SAMPLE_RATE);
    if let Some(secs) = max_audio_secs.filter(|secs| secs.is_finite() && *secs > 0.0) {
        opts = opts.with_max_duration_secs(secs);
    }
    opts
}

fn map_aac(error: AacError) -> anyhow::Error {
    match error {
        AacError::TooLong {
            observed_secs,
            max_secs,
        } => GigasttError::AudioTooLong {
            observed_secs,
            limit_secs: max_secs,
        }
        .into(),
        other => other.into(),
    }
}
