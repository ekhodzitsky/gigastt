//! WAVE family ingest via [`ryf`].
//!
//! PCM / IEEE, G.711, G.722, G.726, GSM 06.10, MS/IMA ADPCM, and the RF64 / RIFX /
//! BW64 / Wave64 containers go through ryf. Other containers stay on the
//! symphonia path. `decode_streaming` is push-based, so the windowed decoder
//! drives it from a dedicated thread with a bounded channel — peak audio
//! memory stays O(one block + one window).

use std::io::{Cursor, Seek as _, SeekFrom};
use std::sync::mpsc::{self, Receiver, RecvError, SyncSender};
use std::thread::{self, JoinHandle};

use anyhow::{Context, Result};
use bytes::Bytes;
use ryf::{ByteSource, ChannelMode, DecodeOptions, StreamBlock, WavError};

use super::resample::{ResampleTo16k, SampleRate};
use super::stream::ChannelSelect;
use super::{MAX_SAMPLE_RATE, audio_too_long_err, resolve_budget};
use crate::error::GigasttError;

/// One pull-parser block, already mixed or picked down to a single plane.
pub(super) struct WaveBlock {
    /// Source-rate frames in this block (budget units).
    pub(super) frames: usize,
    /// Picked/mixed samples; empty when `ChannelSelect::One(k)` is out of range.
    pub(super) samples: Vec<f32>,
}

pub(super) enum WaveErr {
    TooLong { observed_secs: f64, max_secs: f64 },
    Other(String),
}

/// Backpressured WAVE decoder feeding [`super::stream::FileWindows`].
pub(super) struct WaveSource {
    pub(super) rx: Option<Receiver<Result<WaveBlock, WaveErr>>>,
    pub(super) join: Option<JoinHandle<()>>,
    pub(super) source_frames: usize,
    pub(super) sample_rate: u32,
    pub(super) max_samples: usize,
    pub(super) limit_secs: f64,
    pub(super) resampler: Box<ResampleTo16k>,
}

impl Drop for WaveSource {
    fn drop(&mut self) {
        // Drop the receiver first so a blocked send unblocks, then join.
        self.rx.take();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

pub(super) fn decode_options(max_audio_secs: Option<f64>, mode: ChannelMode) -> DecodeOptions {
    let mut opts = DecodeOptions::unbounded()
        .with_channel_mode(mode)
        .with_max_sample_rate(MAX_SAMPLE_RATE);
    if let Some(secs) = max_audio_secs.filter(|s| s.is_finite() && *s > 0.0) {
        opts = opts.with_max_duration_secs(secs);
    }
    opts
}

pub(super) fn channel_mode(channel: ChannelSelect) -> ChannelMode {
    match channel {
        ChannelSelect::Mono => ChannelMode::Mono,
        ChannelSelect::One(_) => ChannelMode::Split,
    }
}

pub(super) fn map_ryf_err(err: WavError) -> anyhow::Error {
    match err {
        WavError::TooLong {
            observed_secs,
            max_secs,
        } => GigasttError::AudioTooLong {
            observed_secs,
            limit_secs: max_secs,
        }
        .into(),
        WavError::UnsupportedSampleRate { rate, .. } => {
            anyhow::anyhow!("Unsupported sample rate: {rate}Hz")
        }
        other => anyhow::anyhow!("{other}"),
    }
}

fn wave_err(err: WavError) -> WaveErr {
    match err {
        WavError::TooLong {
            observed_secs,
            max_secs,
        } => WaveErr::TooLong {
            observed_secs,
            max_secs,
        },
        other => WaveErr::Other(other.to_string()),
    }
}

fn pick_plane(block: &StreamBlock<'_>, channel: ChannelSelect) -> Vec<f32> {
    let plane = match channel {
        ChannelSelect::Mono => block.planar.first(),
        ChannelSelect::One(k) => block.planar.get(k),
    };
    plane.map(|s| s.to_vec()).unwrap_or_default()
}

fn spawn_decode<F>(
    build_source: F,
    opts: DecodeOptions,
    channel: ChannelSelect,
    tx: SyncSender<Result<WaveBlock, WaveErr>>,
) -> Result<JoinHandle<()>>
where
    F: FnOnce() -> Result<ByteSource<'static>, WavError> + Send + 'static,
{
    thread::Builder::new()
        .name("gigastt-wave".into())
        .spawn(move || {
            let run = || -> Result<(), WavError> {
                let mut src = build_source()?;
                ryf::decode_streaming(&mut src, &opts, |block| {
                    let samples = pick_plane(&block, channel);
                    tx.send(Ok(WaveBlock {
                        frames: block.frames,
                        samples,
                    }))
                    .map_err(|_| WavError::format(ryf::FormatKind::InvalidOperation))?;
                    Ok(())
                })?;
                Ok(())
            };
            if let Err(err) = run() {
                let _ = tx.send(Err(wave_err(err)));
            }
        })
        .context("Failed to start WAVE decode thread")
}

fn wave_source_from(
    probe: ryf::WavProbe,
    opts: DecodeOptions,
    channel: ChannelSelect,
    max_audio_secs: Option<f64>,
    build_source: impl FnOnce() -> Result<ByteSource<'static>, WavError> + Send + 'static,
) -> Result<WaveSource> {
    let sample_rate = probe.sample_rate;
    if sample_rate == 0 || sample_rate > MAX_SAMPLE_RATE {
        anyhow::bail!("Unsupported sample rate: {sample_rate}Hz");
    }
    let (max_samples, limit_secs) = resolve_budget(max_audio_secs, sample_rate);
    let (tx, rx) = mpsc::sync_channel(1);
    let join = spawn_decode(build_source, opts, channel, tx)?;
    tracing::info!(
        "Audio (wave): {sample_rate}Hz, {}ch (streaming windows)",
        probe.channels
    );
    Ok(WaveSource {
        rx: Some(rx),
        join: Some(join),
        source_frames: 0,
        sample_rate,
        max_samples,
        limit_secs,
        resampler: Box::new(ResampleTo16k::new(SampleRate(sample_rate), None)),
    })
}

/// Open a WAVE buffer. Returns `Ok(None)` when the bytes are not a WAVE
/// container so the caller can fall through to symphonia.
pub(super) fn try_open_bytes(
    data: Bytes,
    channel: ChannelSelect,
    max_audio_secs: Option<f64>,
) -> Result<Option<WaveSource>> {
    if !ryf::sniff_wav(data.as_ref()) {
        return Ok(None);
    }
    let opts = decode_options(max_audio_secs, channel_mode(channel));
    let probe = {
        let mut probe_src = ByteSource::from_slice(data.as_ref());
        ryf::probe_with(&mut probe_src, &opts).map_err(map_ryf_err)?
    };
    let len = data.len() as u64;
    let source = wave_source_from(probe, opts, channel, max_audio_secs, move || {
        Ok(ByteSource::from_read_seek(Cursor::new(data), Some(len)))
    })?;
    Ok(Some(source))
}

/// Open a WAVE file. Returns `Ok(None)` when the path is not a WAVE container.
pub(super) fn try_open_path(
    path: &str,
    channel: ChannelSelect,
    max_audio_secs: Option<f64>,
) -> Result<Option<WaveSource>> {
    let mut file =
        std::fs::File::open(path).with_context(|| format!("Failed to open audio file: {path}"))?;
    let mut prefix = [0u8; 40];
    let n = std::io::Read::read(&mut file, &mut prefix)
        .with_context(|| format!("Failed to read audio file: {path}"))?;
    if !ryf::sniff_wav(&prefix[..n]) {
        return Ok(None);
    }
    file.seek(SeekFrom::Start(0))
        .with_context(|| format!("Failed to read audio file: {path}"))?;
    let opts = decode_options(max_audio_secs, channel_mode(channel));
    let probe = {
        let mut probe_src = ByteSource::from_file(file);
        ryf::probe_with(&mut probe_src, &opts).map_err(map_ryf_err)?
    };
    let path = path.to_owned();
    let source = wave_source_from(probe, opts, channel, max_audio_secs, move || {
        let file = std::fs::File::open(&path).map_err(WavError::from)?;
        Ok(ByteSource::from_file(file))
    })?;
    Ok(Some(source))
}

pub(super) enum WaveRecv {
    Block(WaveBlock),
    Eof,
}

impl WaveSource {
    pub(super) fn recv_block(&mut self) -> Result<WaveRecv> {
        let Some(rx) = &self.rx else {
            return Ok(WaveRecv::Eof);
        };
        match rx.recv() {
            Ok(Ok(block)) => Ok(WaveRecv::Block(block)),
            Ok(Err(WaveErr::TooLong {
                observed_secs,
                max_secs,
            })) => Err(GigasttError::AudioTooLong {
                observed_secs,
                limit_secs: max_secs,
            }
            .into()),
            Ok(Err(WaveErr::Other(msg))) => Err(anyhow::anyhow!("{msg}")),
            Err(RecvError) => {
                self.rx.take();
                if let Some(join) = self.join.take()
                    && join.join().is_err()
                {
                    anyhow::bail!("WAVE decode thread panicked");
                }
                Ok(WaveRecv::Eof)
            }
        }
    }
}

/// Header-only duration for a sniffed WAVE buffer.
pub(super) fn probe_duration(data: &[u8]) -> Result<Option<f64>> {
    let mut src = ByteSource::from_slice(data);
    probe_duration_src(&mut src)
}

/// Header-only duration for a WAVE file already positioned at offset 0.
pub(super) fn probe_duration_file(file: std::fs::File) -> Result<Option<f64>> {
    let mut src = ByteSource::from_file(file);
    probe_duration_src(&mut src)
}

fn probe_duration_src(src: &mut ByteSource<'_>) -> Result<Option<f64>> {
    let opts = decode_options(None, ChannelMode::Mono);
    let probe = match ryf::probe_with(src, &opts) {
        Ok(p) => p,
        Err(WavError::NotWave) => return Ok(None),
        Err(e) => return Err(map_ryf_err(e)).context("Unsupported audio format"),
    };
    if probe.sample_rate == 0 || probe.sample_rate > MAX_SAMPLE_RATE {
        return Ok(None);
    }
    Ok(probe
        .declared_frames
        .filter(|n| *n > 0)
        .map(|n| n as f64 / f64::from(probe.sample_rate)))
}

/// Take the mixed/first plane out of a ryf decode, or error on empty.
pub(super) fn take_mono(decoded: ryf::DecodedWav) -> Result<(u32, Vec<f32>)> {
    let rate = decoded.sample_rate;
    let mono = decoded
        .channels
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("WAVE data chunk is empty"))?;
    if mono.is_empty() {
        anyhow::bail!("WAVE data chunk is empty");
    }
    Ok((rate, mono))
}

/// Incremental source-rate budget check used by the windowed WAVE loop.
pub(super) fn check_budget(
    source_frames: usize,
    sample_rate: u32,
    max_samples: usize,
    limit_secs: f64,
) -> Result<()> {
    if source_frames > max_samples {
        return Err(audio_too_long_err(source_frames, sample_rate, limit_secs));
    }
    Ok(())
}
