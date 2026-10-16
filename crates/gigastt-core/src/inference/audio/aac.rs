//! AAC ingest via [`syom`].
//!
//! ADTS, LATM/LOAS, M4A, and bounded fragmented MP4 (AAC-LC, HE-AAC, AAC-LD)
//! go through syom. MP3, Vorbis, FLAC, and Matroska stay on symphonia.
//! syom's decode callbacks run to the end of the stream, so the windowed
//! decoder drives them from a dedicated thread with a bounded channel — peak
//! audio memory stays O(one frame + one window).
//!
//! Channels are requested split and mixed here. The mix is the equal-weight
//! mean of every plane, including LFE, matching the other container paths.
//! syom's speech mono skips LFE and caps a stream at two hours; an unbounded
//! windowed decode must not inherit that cap.

use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, RecvError, SyncSender};
use std::thread::{self, JoinHandle};

use anyhow::{Context, Result};
use bytes::Bytes;
use syom::{AacError, DecodeOptions, Decoder, Frame, M4aSeek};

use super::resample::{RESAMPLE_STAGING_FRAMES, ResampleTo16k, SampleRate};
use super::stream::ChannelSelect;
use super::{MAX_SAMPLE_RATE, resolve_budget, whole_buffer_limit_secs};
use crate::error::GigasttError;

/// One pull from the AAC worker, already mixed or picked down to a single plane.
pub(super) struct AacBlock {
    /// Source-rate frames in this block (budget units).
    pub(super) frames: usize,
    /// Picked or mixed samples. Empty when `ChannelSelect::One(k)` is out of range.
    pub(super) samples: Vec<f32>,
    /// Presentation rate. HE-AAC is twice the core rate.
    rate: u32,
    /// Coded plane count, before the mono mix.
    channels: usize,
}

/// What an AAC probe can say without handing the bytes to another decoder.
pub(super) enum ProbeHit {
    /// Not AAC. The caller should keep walking the other sniffers.
    FallThrough,
    /// AAC. `None` is a stream with no usable duration (typical ADTS).
    Duration(Option<f64>),
}

/// Header channel count. `None` means the bytes are not an AAC stream this
/// probe understands, and the caller should fall through.
pub(super) struct AacChannels {
    pub(super) channels: usize,
}

pub(super) enum AacRecv {
    Block(AacBlock),
    Eof,
}

enum First {
    Frame(AacBlock),
    NotAac,
    Failed(AacErr),
    Eof,
}

enum AacErr {
    NotAac,
    TooLong { observed_secs: f64, max_secs: f64 },
    Rate(u32),
    Other(String),
}

enum AacInput {
    Bytes(Bytes),
    Path(PathBuf),
}

enum Reader {
    File(std::fs::File),
    Mem(Cursor<Bytes>),
}

impl Read for Reader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Self::File(file) => file.read(buf),
            Self::Mem(cursor) => cursor.read(buf),
        }
    }
}

impl Seek for Reader {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        match self {
            Self::File(file) => file.seek(pos),
            Self::Mem(cursor) => cursor.seek(pos),
        }
    }
}

/// Backpressured AAC decoder feeding [`super::stream::FileWindows`].
pub(super) struct AacSource {
    rx: Option<Receiver<Result<AacBlock, AacErr>>>,
    join: Option<JoinHandle<()>>,
    pub(super) source_frames: usize,
    pub(super) sample_rate: u32,
    pub(super) max_samples: usize,
    pub(super) limit_secs: f64,
    pub(super) resampler: Box<ResampleTo16k>,
    pending: Vec<f32>,
    pending_frames: usize,
    pending_ready: bool,
}

impl Drop for AacSource {
    fn drop(&mut self) {
        // Drop the receiver first so a blocked send unblocks, then join.
        // Field drops run after this, so both handles are taken here.
        self.rx.take();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

impl AacSource {
    pub(super) fn recv_block(&mut self) -> Result<AacRecv> {
        if self.pending_ready {
            self.pending_ready = false;
            let frames = self.pending_frames;
            self.pending_frames = 0;
            let samples = std::mem::take(&mut self.pending);
            return Ok(AacRecv::Block(AacBlock {
                frames,
                samples,
                rate: self.sample_rate,
                channels: 0,
            }));
        }
        let Some(rx) = &self.rx else {
            return Ok(AacRecv::Eof);
        };
        match rx.recv() {
            Ok(Ok(block)) => Ok(AacRecv::Block(block)),
            Ok(Err(AacErr::NotAac)) => anyhow::bail!("Unsupported audio codec"),
            Ok(Err(AacErr::TooLong {
                observed_secs,
                max_secs,
            })) => Err(GigasttError::AudioTooLong {
                observed_secs,
                limit_secs: max_secs,
            }
            .into()),
            Ok(Err(AacErr::Rate(rate))) => {
                anyhow::bail!("Unsupported sample rate: {rate}Hz")
            }
            Ok(Err(AacErr::Other(msg))) => anyhow::bail!("{msg}"),
            Err(RecvError) => {
                self.rx.take();
                if let Some(join) = self.join.take()
                    && join.join().is_err()
                {
                    anyhow::bail!("AAC decode thread panicked");
                }
                Ok(AacRecv::Eof)
            }
        }
    }
}

/// Open an AAC buffer. `Ok(None)` when the bytes are not AAC, so the caller
/// can fall through to symphonia.
pub(super) fn try_open_bytes(
    data: Bytes,
    channel: ChannelSelect,
    max_audio_secs: Option<f64>,
) -> Result<Option<AacSource>> {
    if !sniff_slice(data.as_ref()) {
        return Ok(None);
    }
    open_input(AacInput::Bytes(data), channel, max_audio_secs)
}

/// Open an AAC file. `Ok(None)` when the path is not AAC.
pub(super) fn try_open_path(
    path: &str,
    channel: ChannelSelect,
    max_audio_secs: Option<f64>,
) -> Result<Option<AacSource>> {
    let mut file =
        std::fs::File::open(path).with_context(|| format!("Failed to open audio file: {path}"))?;
    if !sniff_file(&mut file).with_context(|| format!("Failed to read audio file: {path}"))? {
        return Ok(None);
    }
    open_input(AacInput::Path(PathBuf::from(path)), channel, max_audio_secs)
}

fn open_input(
    input: AacInput,
    channel: ChannelSelect,
    max_audio_secs: Option<f64>,
) -> Result<Option<AacSource>> {
    let opts = stream_options(max_audio_secs);
    let (tx, rx) = mpsc::sync_channel(2);
    let join = thread::Builder::new()
        .name("gigastt-aac".into())
        .spawn(move || worker(input, opts, channel, tx))
        .context("Failed to start AAC decode thread")?;
    let first = match rx.recv() {
        Ok(Ok(block)) => First::Frame(block),
        Ok(Err(AacErr::NotAac)) => First::NotAac,
        Ok(Err(err)) => First::Failed(err),
        Err(RecvError) => First::Eof,
    };
    let block = match first {
        First::Frame(block) => block,
        other => {
            drop(rx);
            let _ = join.join();
            return match other {
                First::NotAac => Ok(None),
                First::Failed(err) => Err(aac_err(err)),
                First::Eof => anyhow::bail!("No audio track found"),
                First::Frame(_) => Ok(None),
            };
        }
    };
    if block.rate == 0 || block.rate > MAX_SAMPLE_RATE {
        drop(rx);
        let _ = join.join();
        anyhow::bail!("Unsupported sample rate: {}Hz", block.rate);
    }
    let (max_samples, limit_secs) = resolve_budget(max_audio_secs, block.rate);
    tracing::info!(
        "Audio (aac): {}Hz, {}ch (streaming windows)",
        block.rate,
        block.channels
    );
    Ok(Some(AacSource {
        rx: Some(rx),
        join: Some(join),
        source_frames: 0,
        sample_rate: block.rate,
        max_samples,
        limit_secs,
        resampler: Box::new(ResampleTo16k::new(SampleRate(block.rate), None)),
        pending: block.samples,
        pending_frames: block.frames,
        pending_ready: true,
    }))
}

/// Whole-buffer split decode. `Ok(None)` when the bytes are not AAC.
pub(super) fn try_decode_planes_bytes(
    data: &[u8],
    max_audio_secs: Option<f64>,
) -> Result<Option<Vec<Vec<f32>>>> {
    if !sniff_slice(data) {
        return Ok(None);
    }
    let limit = whole_buffer_limit_secs(max_audio_secs);
    match syom::decode_with(data, &capped_options(limit)) {
        Ok(decoded) => Ok(Some(resample_planes(
            decoded.sample_rate,
            decoded.channels,
        )?)),
        Err(AacError::NotAac) => Ok(None),
        Err(err) => Err(map_aac(err)),
    }
}

/// Path twin of [`try_decode_planes_bytes`].
pub(super) fn try_decode_planes_path(
    path: &str,
    max_audio_secs: Option<f64>,
) -> Result<Option<Vec<Vec<f32>>>> {
    let mut file =
        std::fs::File::open(path).with_context(|| format!("Failed to open audio file: {path}"))?;
    if !sniff_file(&mut file).with_context(|| format!("Failed to read audio file: {path}"))? {
        return Ok(None);
    }
    let limit = whole_buffer_limit_secs(max_audio_secs);
    match syom::read_with(path, &capped_options(limit)) {
        Ok(decoded) => Ok(Some(resample_planes(
            decoded.sample_rate,
            decoded.channels,
        )?)),
        Err(AacError::NotAac) => Ok(None),
        Err(err) => Err(map_aac(err)),
    }
}

/// Header duration for a buffer already in memory. Does not decode PCM.
pub(super) fn probe_bytes(data: &[u8]) -> Result<ProbeHit> {
    if !sniff_slice(data) {
        return Ok(ProbeHit::FallThrough);
    }
    match syom::probe(data) {
        Ok(info) => Ok(ProbeHit::Duration(duration_of(
            info.meta.output_rate,
            info.duration,
        ))),
        Err(AacError::NotAac) => Ok(ProbeHit::FallThrough),
        Err(AacError::UnsupportedSampleRate { .. }) => Ok(ProbeHit::Duration(None)),
        Err(err) => Err(map_aac(err)).context("Unsupported audio format"),
    }
}

/// Duration for a file positioned at offset 0. M4A reads `moov` plus one
/// frame (for the presentation rate) and stops. ADTS and LATM have no
/// reliable header duration and return `Duration(None)` without a full read.
pub(super) fn probe_file(mut file: std::fs::File) -> Result<ProbeHit> {
    if !sniff_file(&mut file).context("Failed to read audio file")? {
        return Ok(ProbeHit::FallThrough);
    }
    let mut prefix = [0u8; 64];
    let n = std::io::Read::read(&mut file, &mut prefix).context("Failed to read audio file")?;
    file.seek(SeekFrom::Start(0))
        .context("Failed to read audio file")?;
    if !syom::sniff_is_isobmff(&prefix[..n]) {
        return Ok(ProbeHit::Duration(None));
    }
    let opts = stream_options(None);
    let mut seek = match M4aSeek::open(file, opts) {
        Ok(seek) => seek,
        Err(AacError::NotAac) => return Ok(ProbeHit::FallThrough),
        Err(AacError::UnsupportedSampleRate { .. } | AacError::Unsupported(_)) => {
            return Ok(ProbeHit::Duration(None));
        }
        Err(err) => return Err(map_aac(err)).context("Unsupported audio format"),
    };
    let samples = seek.presentation_len();
    let mut rate = None;
    let decoded = seek.decode(|frame| {
        rate = Some(frame.sample_rate);
        Err(AacError::decode("probe"))
    });
    if let Err(err) = decoded {
        let aborted = matches!(err, AacError::Decode(ref msg) if msg == "probe");
        if !aborted || rate.is_none() {
            return match err {
                AacError::NotAac => Ok(ProbeHit::FallThrough),
                AacError::Decode(ref msg) if msg == "probe" => Ok(ProbeHit::Duration(None)),
                other => Err(map_aac(other)).context("Unsupported audio format"),
            };
        }
    }
    let Some(rate) = rate else {
        return Ok(ProbeHit::Duration(None));
    };
    if rate == 0 || rate > MAX_SAMPLE_RATE || samples == 0 {
        return Ok(ProbeHit::Duration(None));
    }
    Ok(ProbeHit::Duration(Some(samples as f64 / f64::from(rate))))
}

/// Coded channel count from the AAC header, with no PCM decode.
/// `Ok(None)` falls through to symphonia (not AAC, or a header with no labels).
pub(super) fn header_channels(data: &[u8]) -> Result<Option<AacChannels>> {
    if !sniff_slice(data) {
        return Ok(None);
    }
    let info = match syom::probe(data) {
        Ok(info) => info,
        Err(AacError::NotAac) => return Ok(None),
        Err(err) => return Err(map_aac(err)),
    };
    let channels = info.meta.labels().count();
    if channels == 0 {
        return Ok(None);
    }
    let rate = info.meta.output_rate;
    if rate == 0 || rate > MAX_SAMPLE_RATE {
        anyhow::bail!("Unsupported sample rate: {rate}Hz");
    }
    Ok(Some(AacChannels { channels }))
}

fn duration_of(rate: u32, duration: syom::ProbeDuration) -> Option<f64> {
    if rate == 0 || rate > MAX_SAMPLE_RATE {
        return None;
    }
    let samples = match duration {
        syom::ProbeDuration::Exact { samples } | syom::ProbeDuration::Estimated { samples }
            if samples > 0 =>
        {
            samples
        }
        _ => return None,
    };
    Some(samples as f64 / f64::from(rate))
}

fn resample_planes(rate: u32, planes: Vec<Vec<f32>>) -> Result<Vec<Vec<f32>>> {
    if rate == 0 || rate > MAX_SAMPLE_RATE {
        anyhow::bail!("Unsupported sample rate: {rate}Hz");
    }
    let mut out = Vec::with_capacity(planes.len());
    for samples in planes {
        let mut chan = ResampleTo16k::new(SampleRate(rate), Some(samples.len()));
        for piece in samples.chunks(RESAMPLE_STAGING_FRAMES) {
            chan.stage().extend_from_slice(piece);
            chan.flush_full()?;
        }
        out.push(chan.finish()?);
    }
    Ok(out)
}

/// AAC, including one leading ID3v2.3/2.4 tag in front of ADTS, LATM, or M4A.
///
/// syom's ADTS encoder writes an `iTunSMPB` tag, and [`syom::sniff_aac`] does
/// not look past it. The tag stays in the buffer: syom peels it while decoding
/// so the priming edit is not lost.
fn sniff_slice(data: &[u8]) -> bool {
    if syom::sniff_aac(data) {
        return true;
    }
    match id3v2_payload_at(data) {
        Some(at) if at < data.len() => syom::sniff_aac(&data[at..]),
        _ => false,
    }
}

/// File twin of [`sniff_slice`]. Rewinds to offset 0 either way. A tag that
/// runs past the first 64 bytes is seeked to, not read in full.
pub(super) fn sniff_file(file: &mut std::fs::File) -> std::io::Result<bool> {
    file.seek(SeekFrom::Start(0))?;
    let mut prefix = [0u8; 64];
    let n = file.read(&mut prefix)?;
    if syom::sniff_aac(&prefix[..n]) {
        file.seek(SeekFrom::Start(0))?;
        return Ok(true);
    }
    let Some(at) = id3v2_payload_at(&prefix[..n]) else {
        file.seek(SeekFrom::Start(0))?;
        return Ok(false);
    };
    let hit = if at < n {
        syom::sniff_aac(&prefix[at..n])
    } else {
        file.seek(SeekFrom::Start(at as u64))?;
        let mut audio = [0u8; 16];
        let m = file.read(&mut audio)?;
        syom::sniff_aac(&audio[..m])
    };
    file.seek(SeekFrom::Start(0))?;
    Ok(hit)
}

/// Offset of the bytes after one ID3v2.3/2.4 header. `None` when those
/// bytes are not that tag. A declared size above 1 GiB is refused.
fn id3v2_payload_at(data: &[u8]) -> Option<usize> {
    if data.len() < 10 || &data[..3] != b"ID3" {
        return None;
    }
    if data[3] != 3 && data[3] != 4 {
        return None;
    }
    if data[6..10].iter().any(|byte| byte & 0x80 != 0) {
        return None;
    }
    let body = (u32::from(data[6]) << 21)
        | (u32::from(data[7]) << 14)
        | (u32::from(data[8]) << 7)
        | u32::from(data[9]);
    let mut total = 10usize.saturating_add(body as usize);
    if data[5] & 0x10 != 0 {
        total = total.saturating_add(10);
    }
    if total > (1 << 30) {
        return None;
    }
    Some(total)
}

/// Windowed decode. `None` stays unbounded: speech defaults would refuse a
/// file longer than two hours, and this path's memory is O(one window).
fn stream_options(max_audio_secs: Option<f64>) -> DecodeOptions {
    let mut opts = DecodeOptions::unbounded().with_max_sample_rate(MAX_SAMPLE_RATE);
    if let Some(secs) = max_audio_secs.filter(|secs| secs.is_finite() && *secs > 0.0) {
        opts = opts.with_max_duration_secs(secs);
    }
    opts
}

fn capped_options(limit_secs: f64) -> DecodeOptions {
    DecodeOptions::unbounded()
        .with_max_sample_rate(MAX_SAMPLE_RATE)
        .with_max_duration_secs(limit_secs)
}

fn worker(
    input: AacInput,
    opts: DecodeOptions,
    channel: ChannelSelect,
    tx: SyncSender<Result<AacBlock, AacErr>>,
) {
    if let Err(err) = decode_input(&input, opts, channel, &tx) {
        if is_stopped(&err) {
            return;
        }
        let _ = tx.send(Err(classify(err)));
    }
}

fn decode_input(
    input: &AacInput,
    opts: DecodeOptions,
    channel: ChannelSelect,
    tx: &SyncSender<Result<AacBlock, AacErr>>,
) -> syom::Result<()> {
    let mut reader = open_reader(input)?;
    let mut prefix = [0u8; 64];
    let n = reader.read(&mut prefix)?;
    reader.seek(SeekFrom::Start(0))?;
    if !syom::sniff_is_isobmff(&prefix[..n]) {
        return pump_feed(&mut reader, opts, channel, tx);
    }
    let first = match M4aSeek::open(reader, opts.clone()) {
        Ok(seek) => return pump_seek(seek, channel, tx),
        Err(err) => err,
    };
    if !matches!(first, AacError::Unsupported(_)) {
        return Err(first);
    }
    let mut reader = open_reader(input)?;
    match pump_feed(&mut reader, opts, channel, tx) {
        Err(AacError::Unsupported(_)) => Err(first),
        other => other,
    }
}

fn pump_seek(
    mut seek: M4aSeek<Reader>,
    channel: ChannelSelect,
    tx: &SyncSender<Result<AacBlock, AacErr>>,
) -> syom::Result<()> {
    let mut emit = Emit::new(channel, tx);
    seek.decode(|frame| emit.on_frame(frame))?;
    Ok(())
}

fn pump_feed<R: Read>(
    reader: &mut R,
    opts: DecodeOptions,
    channel: ChannelSelect,
    tx: &SyncSender<Result<AacBlock, AacErr>>,
) -> syom::Result<()> {
    let mut dec = Decoder::new(opts);
    let mut emit = Emit::new(channel, tx);
    dec.feed_read(reader, |frame| emit.on_frame(frame))?;
    dec.finish(|frame| emit.on_frame(frame))?;
    Ok(())
}

struct Emit<'a> {
    channel: ChannelSelect,
    tx: &'a SyncSender<Result<AacBlock, AacErr>>,
    rate: Option<u32>,
}

impl<'a> Emit<'a> {
    fn new(channel: ChannelSelect, tx: &'a SyncSender<Result<AacBlock, AacErr>>) -> Self {
        Self {
            channel,
            tx,
            rate: None,
        }
    }

    fn on_frame(&mut self, frame: Frame<'_>) -> syom::Result<()> {
        if frame.sample_rate == 0 || frame.sample_rate > MAX_SAMPLE_RATE {
            return Err(AacError::sample_rate(frame.sample_rate, MAX_SAMPLE_RATE));
        }
        if let Some(prev) = self.rate {
            if prev != frame.sample_rate {
                return Err(AacError::decode("sample rate changed"));
            }
        } else {
            self.rate = Some(frame.sample_rate);
        }
        let channels = frame.planar.len();
        let block = AacBlock {
            frames: frame.samples,
            samples: pick_plane(&frame, self.channel),
            rate: frame.sample_rate,
            channels,
        };
        self.tx
            .send(Ok(block))
            .map_err(|_| AacError::decode("stopped"))?;
        Ok(())
    }
}

fn pick_plane(frame: &Frame<'_>, channel: ChannelSelect) -> Vec<f32> {
    let planes = frame.planar;
    match channel {
        ChannelSelect::Mono => {
            let Some(first) = planes.first() else {
                return Vec::new();
            };
            let mut acc = first.to_vec();
            for plane in planes.iter().skip(1) {
                for (dst, src) in acc.iter_mut().zip(plane.iter()) {
                    *dst += *src;
                }
            }
            let n = planes.len() as f32;
            for sample in &mut acc {
                *sample /= n;
            }
            acc
        }
        ChannelSelect::One(index) => planes
            .get(index)
            .map(|plane| plane.to_vec())
            .unwrap_or_default(),
    }
}

fn open_reader(input: &AacInput) -> syom::Result<Reader> {
    match input {
        AacInput::Bytes(data) => Ok(Reader::Mem(Cursor::new(data.clone()))),
        AacInput::Path(path) => std::fs::File::open(path)
            .map(Reader::File)
            .map_err(AacError::from),
    }
}

fn is_stopped(err: &AacError) -> bool {
    matches!(err, AacError::Decode(msg) if msg == "stopped")
}

fn classify(err: AacError) -> AacErr {
    match err {
        AacError::NotAac => AacErr::NotAac,
        AacError::TooLong {
            observed_secs,
            max_secs,
        } => AacErr::TooLong {
            observed_secs,
            max_secs,
        },
        AacError::UnsupportedSampleRate { rate, .. } => AacErr::Rate(rate),
        other => AacErr::Other(other.to_string()),
    }
}

fn aac_err(err: AacErr) -> anyhow::Error {
    match err {
        AacErr::NotAac => anyhow::anyhow!("Unsupported audio codec"),
        AacErr::TooLong {
            observed_secs,
            max_secs,
        } => GigasttError::AudioTooLong {
            observed_secs,
            limit_secs: max_secs,
        }
        .into(),
        AacErr::Rate(rate) => anyhow::anyhow!("Unsupported sample rate: {rate}Hz"),
        AacErr::Other(msg) => anyhow::anyhow!("{msg}"),
    }
}

fn map_aac(err: AacError) -> anyhow::Error {
    aac_err(classify(err))
}
