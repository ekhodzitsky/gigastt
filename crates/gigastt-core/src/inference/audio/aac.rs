//! Bounded AAC ingest. syom handles AAC transports; Symphonia retains AAC in
//! Matroska and non-AAC MP4 tracks. The callback codec feeds windowed consumers
//! in bounded batches, while channel processing stays synchronous and incremental.

use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::{self, JoinHandle};

use anyhow::{Context, Result};
use bytes::Bytes;
use syom::{AacError, DecodeOptions, Frame};

use super::resample::{ResampleTo16k, SampleRate};
use super::stream::ChannelSelect;
use super::{MAX_SAMPLE_RATE, resolve_budget};
use crate::error::GigasttError;

mod adts_pull;
mod buffered;
mod channels;
mod input;
mod probe;
mod pull;
pub(super) use channels::{try_decode_planes_bytes, try_decode_planes_path, try_scan_bytes};
use input::{Reader, decode_reader_with_mode};
pub(super) use probe::{ProbeHit, probe_bytes, probe_file, sniff_file};
use probe::{Transport, classify_file, classify_slice};

// Logical cuts never change: the resampler observes each block individually.
const BATCH_FRAMES: usize = 24 * 1024;
type WorkerPacket = (AacBlock, Option<AacBlock>);
const WORKER_MESSAGE_CAPACITY: usize = 1;

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
    rx: Option<Receiver<Result<WorkerPacket>>>,
    join: Option<JoinHandle<()>>,
    pending: Option<AacBlock>,
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
        if let Some(block) = self.pending.take() {
            return Ok(Some(block));
        }
        let Some(rx) = &self.rx else { return Ok(None) };
        match rx.recv() {
            Ok(Ok((first, second))) => {
                self.pending = second;
                Ok(Some(first))
            }
            Ok(Err(error)) => Err(error),
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

// One queued packet holds at most two blocks. The producer holds at most two
// more (including a blocked send), and the receiver one pending second block.
// Including the actively consumed block, the bound is six logical blocks,
// versus four before packing: two extra blocks, still independent of duration.
struct BlockSender {
    tx: SyncSender<Result<WorkerPacket>>,
    held: Option<AacBlock>,
    first: bool,
}
impl BlockSender {
    fn send(&mut self, block: AacBlock) -> syom::Result<()> {
        if self.first {
            self.first = false;
            self.transmit((block, None))
        } else if let Some(first) = self.held.take() {
            self.transmit((first, Some(block)))
        } else {
            self.held = Some(block);
            Ok(())
        }
    }
    fn transmit(&self, packet: WorkerPacket) -> syom::Result<()> {
        self.tx
            .send(Ok(packet))
            .map_err(|_| AacError::decode("AAC consumer closed"))
    }
    fn flush(&mut self) -> syom::Result<()> {
        if let Some(block) = self.held.take() {
            self.transmit((block, None))?;
        }
        Ok(())
    }
    fn finish(mut self, result: syom::Result<()>, tail: Option<AacBlock>) {
        match result {
            Ok(()) => {
                if let Some(block) = tail
                    && self.send(block).is_err()
                {
                    return;
                }
                let _ = self.flush();
            }
            Err(error) => {
                // Completed logical blocks precede errors; incomplete PCM is discarded.
                if self.flush().is_ok() {
                    let _ = self.tx.send(Err(map_aac(error)));
                }
            }
        }
    }
}

pub(super) struct AacSource {
    producer: Producer,
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
            None => self.producer.recv()?,
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
    let Some(transport) = classify_slice(&data) else {
        return Ok(None);
    };
    open_input(Reader::bytes(data), transport, channel, max_audio_secs)
}

pub(super) fn try_open_path(
    path: &str,
    channel: ChannelSelect,
    max_audio_secs: Option<f64>,
) -> Result<Option<AacSource>> {
    let mut file =
        std::fs::File::open(path).with_context(|| format!("Failed to open audio file: {path}"))?;
    let Some(transport) = classify_file(&mut file)? else {
        return Ok(None);
    };
    open_input(Reader::File(file), transport, channel, max_audio_secs)
}

fn open_input(
    mut input: Reader,
    transport: Transport,
    channel: ChannelSelect,
    max_audio_secs: Option<f64>,
) -> Result<Option<AacSource>> {
    let pull = match if transport == Transport::SeekableM4a {
        pull::try_open(&mut input, channel, max_audio_secs)
    } else {
        Ok(None)
    } {
        Err(error) if matches!(error.downcast_ref::<AacError>(), Some(AacError::NotAac)) => {
            return Ok(None);
        }
        other => other?,
    };
    let mut producer = match pull {
        Some(source) => Producer::Pull(Box::new(source)),
        None if transport == Transport::BareAdts => Producer::Adts(Box::new(
            adts_pull::AdtsPull::new(input, channel, max_audio_secs),
        )),
        None => Producer::Worker(spawn_worker_with_mode(
            input,
            channel,
            max_audio_secs,
            true,
        )?),
    };
    let first = match producer.recv() {
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
        producer,
        source_frames: 0,
        max_samples,
        limit_secs,
    }))
}

// M4A is naturally seekable, so its documented callback pause contract permits
// a synchronous bounded pull. Transports requiring push retain their worker.
enum Producer {
    Worker(Worker),
    Pull(Box<pull::Pull>),
    Adts(Box<adts_pull::AdtsPull<Reader>>),
}
impl Producer {
    fn recv(&mut self) -> Result<Option<AacBlock>> {
        match self {
            Self::Worker(worker) => worker.recv(),
            Self::Pull(source) => source.recv(),
            Self::Adts(source) => source.recv(),
        }
    }
}

#[cfg(test)]
fn spawn_worker(
    input: Reader,
    channel: ChannelSelect,
    max_audio_secs: Option<f64>,
) -> Result<Worker> {
    spawn_worker_with_mode(input, channel, max_audio_secs, false)
}

fn spawn_worker_with_mode(
    mut input: Reader,
    channel: ChannelSelect,
    max_audio_secs: Option<f64>,
    known_push: bool,
) -> Result<Worker> {
    let (tx, rx) = mpsc::sync_channel(WORKER_MESSAGE_CAPACITY);
    let join = thread::Builder::new()
        .name("gigastt-aac".into())
        .spawn(move || {
            let mut sender = BlockSender {
                tx,
                held: None,
                first: true,
            };
            let mut batch: Option<AacBlock> = None;
            let result = decode_reader_with_mode(
                &mut input,
                stream_options(max_audio_secs),
                known_push,
                |frame| {
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
                        sender.send(block)?;
                    }
                    Ok(())
                },
            );
            sender.finish(result, batch);
        })
        .context("Failed to start AAC decode thread")?;
    Ok(Worker {
        rx: Some(rx),
        join: Some(join),
        pending: None,
    })
}

fn append_plane(out: &mut Vec<f32>, frame: &Frame<'_>, channel: ChannelSelect) {
    match channel {
        ChannelSelect::One(index) => {
            if let Some(plane) = frame.planar.get(index) {
                out.extend_from_slice(plane);
            }
        }
        ChannelSelect::Mono => append_mono_planes(out, frame.planar),
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

// Preserve the existing add-then-divide order, with one output pass for stereo.
// Unequal planes retain the defensive historical zip behavior.
fn append_mono_planes(out: &mut Vec<f32>, planes: &[&[f32]]) {
    if let [mono] = planes {
        out.extend_from_slice(mono);
        return;
    }
    if let [left, right] = planes
        && left.len() == right.len()
    {
        out.extend(left.iter().zip(*right).map(|(a, b)| (*a + *b) / 2.0));
        return;
    }
    let Some(first) = planes.first() else { return };
    let start = out.len();
    out.extend_from_slice(first);
    for plane in planes.iter().skip(1) {
        for (sample, value) in out[start..].iter_mut().zip(plane.iter()) {
            *sample += value;
        }
    }
    let count = planes.len() as f32;
    for sample in &mut out[start..] {
        *sample /= count;
    }
}

#[cfg(test)]
mod worker_packet_tests {
    use super::*;

    fn block(id: usize, frames: usize) -> AacBlock {
        AacBlock {
            frames,
            samples: vec![id as f32, -0.0],
            rate: 48_000,
            channels: 1,
        }
    }
    fn worker(count: usize, tail: bool, fail: bool) -> Worker {
        let (tx, rx) = mpsc::sync_channel(WORKER_MESSAGE_CAPACITY);
        let join = thread::spawn(move || {
            let mut sender = BlockSender {
                tx,
                held: None,
                first: true,
            };
            for i in 0..count {
                if sender.send(block(i, BATCH_FRAMES)).is_err() {
                    return;
                }
            }
            let result = if fail {
                Err(AacError::decode("original decoder failure"))
            } else {
                Ok(())
            };
            sender.finish(result, tail.then(|| block(count, 137)));
        });
        Worker {
            rx: Some(rx),
            join: Some(join),
            pending: None,
        }
    }

    #[test]
    fn test_aac_worker_packets_preserve_odd_even_full_partial_eof() {
        for count in 0..6 {
            for tail in [false, true] {
                let mut worker = worker(count, tail, false);
                for i in 0..count + usize::from(tail) {
                    let actual = worker.recv().unwrap().unwrap();
                    assert_eq!(actual.frames, if i < count { BATCH_FRAMES } else { 137 });
                    assert_eq!(
                        actual
                            .samples
                            .iter()
                            .map(|s| s.to_bits())
                            .collect::<Vec<_>>(),
                        [i as f32, -0.0].map(f32::to_bits)
                    );
                }
                assert!(worker.recv().unwrap().is_none());
            }
        }
    }

    #[test]
    fn test_aac_worker_packets_flush_completed_blocks_before_typed_error() {
        for count in 0..6 {
            let mut worker = worker(count, true, true);
            for i in 0..count {
                assert_eq!(worker.recv().unwrap().unwrap().samples[0], i as f32);
            }
            let error = worker.recv().err().unwrap();
            assert!(error.downcast_ref::<AacError>().is_some());
            assert!(error.to_string().contains("original decoder failure"));
            assert!(
                worker.recv().unwrap().is_none(),
                "partial tail must be discarded"
            );
        }
    }

    #[test]
    fn test_aac_worker_packets_keep_first_block_immediate_and_drop_unblocks_sender() {
        let mut worker = worker(1000, true, false);
        assert_eq!(worker.recv().unwrap().unwrap().samples[0], 0.0);
        assert!(worker.pending.is_none(), "the first block is sent alone");
        let (done, completion) = mpsc::channel();
        thread::spawn(move || {
            drop(worker);
            done.send(()).unwrap();
        });
        completion
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
    }

    #[test]
    fn test_aac_worker_packet_queue_has_explicit_two_block_bound() {
        let (tx, _rx) = mpsc::sync_channel::<Result<WorkerPacket>>(WORKER_MESSAGE_CAPACITY);
        assert!(
            tx.try_send(Ok((block(0, BATCH_FRAMES), Some(block(1, BATCH_FRAMES)))))
                .is_ok()
        );
        assert!(matches!(
            tx.try_send(Ok((block(2, BATCH_FRAMES), None))),
            Err(mpsc::TrySendError::Full(_))
        ));
    }
}

#[cfg(test)]
mod mono_plane_tests {
    use super::append_mono_planes;

    fn original(out: &mut Vec<f32>, planes: &[&[f32]]) {
        let Some(first) = planes.first() else { return };
        let start = out.len();
        out.extend_from_slice(first);
        for plane in planes.iter().skip(1) {
            for (sample, value) in out[start..].iter_mut().zip(plane.iter()) {
                *sample += value;
            }
        }
        for sample in &mut out[start..] {
            *sample /= planes.len() as f32;
        }
    }

    #[test]
    fn test_aac_mono_planes_preserve_add_then_divide_bits() {
        let left = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::from_bits(0x7fc0_0001),
            f32::from_bits(0xffc1_2345),
        ];
        let right = [
            -0.0,
            -0.0,
            -0.5,
            0.5,
            -f32::MIN_POSITIVE,
            f32::from_bits(1),
            -f32::MAX,
            f32::NEG_INFINITY,
            f32::INFINITY,
            1.0,
            f32::from_bits(0x7fc0_0001),
            f32::from_bits(0xffc1_2345),
        ];
        for planes in [
            vec![],
            vec![&left[..]],
            vec![&left[..], &right[..]],
            vec![&left[..], &right[..], &left[..]],
            vec![&left[..], &right[..3]],
        ] {
            let mut actual = vec![17.0, -0.0];
            let mut expected = actual.clone();
            original(&mut expected, &planes);
            append_mono_planes(&mut actual, &planes);
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
}
