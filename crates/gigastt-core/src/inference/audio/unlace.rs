//! Optional bounded sequential WebM/Matroska Opus source.
//!
//! This strict path deliberately applies container trim metadata that the
//! legacy path ignores. It is not a transparent decoder replacement.
use std::io::{Read, Seek, SeekFrom};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use anyhow::{Context, Result, bail, ensure};
use unlace::demux::{AudioSelection, DemuxEvent, DemuxLimits, Demuxer};

use super::opus::OpusStream;
use super::stream::ChannelSelect;
use super::{audio_too_long_err, check_decode_abort, resolve_budget};

const RATE: u32 = 48_000;
const MAX_PACKET_FRAMES: usize = 5760;
const MAX_BLOCK_FRAMES: usize = 256 * MAX_PACKET_FRAMES;
const MAX_HEADER_BYTES: u64 = 8 * 1024 * 1024;

type Input = Box<dyn Read + Send + Sync>;

// Both codec identification and strict header parsing share this physical-read
// budget. Once TracksReady is validated, normal sequential reads are enabled.
struct ProbeInput<R> {
    inner: R,
    remaining: usize,
    bounded: Arc<AtomicBool>,
}

impl<R: Read> Read for ProbeInput<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if !self.bounded.load(Ordering::Relaxed) {
            return self.inner.read(buf);
        }
        if buf.is_empty() {
            return Ok(0);
        }
        if self.remaining == 0 {
            return Err(std::io::Error::other("WebM probe read budget exceeded"));
        }
        let limit = buf.len().min(self.remaining);
        let n = self.inner.read(&mut buf[..limit])?;
        self.remaining -= n;
        Ok(n)
    }
}

impl<R: Seek> Seek for ProbeInput<R> {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(pos)
    }
}

impl<R: Read + Seek + Send + Sync> symphonia::core::io::MediaSource for ProbeInput<R> {
    fn is_seekable(&self) -> bool {
        // Preserve legacy codec identification, including bounded lookups of
        // existing SeekHead metadata. The format and its metadata are dropped
        // before sequential strict Opus decoding starts.
        true
    }
    fn byte_len(&self) -> Option<u64> {
        None
    }
}

pub(super) struct UnlaceSource {
    channels: usize,
    demux: Demuxer<Input>,
    decoder: OpusStream,
    block: Vec<f32>,
    pre_skip: usize,
    raw_frames: usize,
    max_frames: usize,
    limit_secs: f64,
    first_timestamp: Option<i128>,
    tick_ns: i128,
    ended: bool,
}

impl UnlaceSource {
    /// Probe without emitting PCM. Non-EBML/non-Opus callers may reopen their
    /// source for the legacy demuxer; recognized malformed input is an error.
    pub(super) fn probe<R: Read + Seek + Send + Sync + 'static>(
        input: R,
        channel: ChannelSelect,
        max_audio_secs: Option<f64>,
    ) -> Result<Option<Self>> {
        use symphonia::core::{
            codecs::audio::well_known::CODEC_ID_OPUS,
            formats::{FormatOptions, TrackType, probe::Hint},
            io::MediaSourceStream,
            meta::MetadataOptions,
        };
        let bounded = Arc::new(AtomicBool::new(true));
        let mut input = ProbeInput {
            inner: input,
            remaining: MAX_HEADER_BYTES as usize,
            bounded: Arc::clone(&bounded),
        };
        let mut signature = [0; 4];
        let signature_result = input.read_exact(&mut signature);
        input.seek(SeekFrom::Start(0))?;
        if let Err(error) = signature_result {
            if error.kind() == std::io::ErrorKind::UnexpectedEof {
                return Ok(None);
            }
            return Err(error.into());
        }
        if signature != [0x1a, 0x45, 0xdf, 0xa3] {
            return Ok(None);
        }
        // Identify the codec using the existing permissive demuxer first.
        // Non-Opus files must not inherit unlace's narrower metadata admission.
        let format = symphonia::default::get_probe()
            .probe(
                &Hint::new(),
                MediaSourceStream::new(Box::new(input), Default::default()),
                FormatOptions::default(),
                MetadataOptions::default(),
            )
            .context("Cannot identify WebM audio codec")?;
        let is_opus = format
            .default_track(TrackType::Audio)
            .and_then(|t| t.codec_params.as_ref())
            .and_then(|p| p.audio())
            .is_some_and(|p| p.codec == CODEC_ID_OPUS);
        if !is_opus {
            return Ok(None);
        }
        let mut input = format.into_inner();
        input.seek(SeekFrom::Start(0))?;
        let mut limits = DemuxLimits::default();
        limits.block.max_block_size = 1024 * 1024;
        limits.block.max_frame_size = 64 * 1024;
        limits.block.max_frames = 256;
        limits.framing.max_payload_size = 1024 * 1024;
        let mut demux = Demuxer::new(Box::new(input) as Input, AudioSelection::First, limits)?;
        loop {
            ensure!(
                demux.position() <= MAX_HEADER_BYTES,
                "WebM header exceeds probe budget"
            );
            match demux.poll().context("Invalid WebM header")? {
                DemuxEvent::TracksReady => {
                    ensure!(
                        demux.position() <= MAX_HEADER_BYTES,
                        "WebM header exceeds probe budget"
                    );
                    break;
                }
                DemuxEvent::Yield => {}
                DemuxEvent::NeedInput => bail!("WebM local source would block"),
                _ => bail!("WebM ended before audio track metadata"),
            }
        }
        let mut tracks = demux
            .tracks()
            .iter()
            .filter(|t| t.enabled && t.track_type == 2);
        let track = tracks.next().context("No enabled WebM audio track")?;
        if track.codec_id != "A_OPUS" {
            return Ok(None);
        }
        ensure!(
            tracks.next().is_none(),
            "Multiple enabled WebM audio tracks are unsupported"
        );
        let head = track.opus_head()?.context("Missing OpusHead")?;
        ensure!(
            head.mapping_family() == 0,
            "Unsupported Opus channel mapping"
        );
        ensure!(
            head.output_gain_q8_db() == 0,
            "Nonzero Opus output gain is unsupported"
        );
        let channels = usize::from(head.channels());
        let decoder = OpusStream::new(channels, channel)?;
        let pre_skip = usize::from(head.pre_skip_samples());
        let tick_ns = i128::from(
            demux
                .info()
                .context("Missing WebM timing metadata")?
                .timestamp_scale_ns,
        );
        let (max_frames, limit_secs) = resolve_budget(max_audio_secs, RATE);
        bounded.store(false, Ordering::Relaxed);
        Ok(Some(Self {
            channels,
            demux,
            decoder,
            block: Vec::new(),
            pre_skip,
            raw_frames: 0,
            max_frames,
            limit_secs,
            first_timestamp: None,
            tick_ns,
            ended: false,
        }))
    }

    pub(super) fn channels(&self) -> usize {
        self.channels
    }

    /// Lends one mono block after whole-block padding and stream pre-skip.
    /// Decoder scratch is bounded to 5760 interleaved stereo frames; retained
    /// mono block is bounded separately, independent of recording duration.
    pub(super) fn next_block(
        &mut self,
        abort: Option<&(dyn Fn() -> bool + Sync)>,
    ) -> Result<Option<&[f32]>> {
        check_decode_abort(abort)?;
        if self.ended {
            return Ok(None);
        }
        self.block.clear();
        let mut origin = None;
        let mut next_lace = 0;
        loop {
            check_decode_abort(abort)?;
            match self
                .demux
                .poll()
                .context("Invalid or truncated WebM audio")?
            {
                DemuxEvent::Packet(packet) => {
                    if next_lace == 0 {
                        origin = Some(packet.block_offset);
                        let first = *self
                            .first_timestamp
                            .get_or_insert(packet.block_timestamp_ns);
                        let delta = packet
                            .block_timestamp_ns
                            .checked_sub(first)
                            .and_then(|n| n.checked_mul(i128::from(RATE)))
                            .and_then(|n| n.checked_sub(self.raw_frames as i128 * 1_000_000_000))
                            .and_then(i128::checked_abs)
                            .context("WebM timestamp arithmetic overflow")?;
                        ensure!(
                            delta <= self.tick_ns * i128::from(RATE),
                            "WebM audio timeline gap or overlap"
                        );
                    }
                    ensure!(
                        origin == Some(packet.block_offset) && packet.lace_index == next_lace,
                        "Discontinuous WebM lace sequence"
                    );
                    ensure!(
                        self.block.len() <= MAX_BLOCK_FRAMES - MAX_PACKET_FRAMES,
                        "Decoded WebM block exceeds budget"
                    );
                    let before = self.block.len();
                    let expected_frames = OpusStream::packet_frames(packet.data)?;
                    let frames = self.decoder.decode_packet(packet.data, &mut self.block)?;
                    ensure!(
                        frames == expected_frames
                            && frames <= MAX_PACKET_FRAMES
                            && self.block.len() - before == frames,
                        "Invalid Opus decoded packet duration"
                    );
                    self.raw_frames = self
                        .raw_frames
                        .checked_add(frames)
                        .context("Opus duration overflow")?;
                    if self.raw_frames > self.max_frames {
                        return Err(audio_too_long_err(self.raw_frames, RATE, self.limit_secs));
                    }
                    next_lace += 1;
                    check_decode_abort(abort)?;
                    if next_lace == packet.lace_count {
                        trim_block(
                            &mut self.block,
                            packet.block_discard_padding_ns,
                            &mut self.pre_skip,
                        )?;
                        return Ok(Some(&self.block));
                    }
                }
                DemuxEvent::End(_) => {
                    ensure!(next_lace == 0, "WebM ended within a laced block");
                    ensure!(self.pre_skip == 0, "Opus pre-skip exceeds decoded audio");
                    self.ended = true;
                    return Ok(None);
                }
                DemuxEvent::Yield => {}
                DemuxEvent::NeedInput => bail!("WebM local source would block"),
                DemuxEvent::TracksReady => bail!("Repeated WebM track metadata"),
            }
        }
    }
}

fn trim_block(pcm: &mut Vec<f32>, padding_ns: i64, pre_skip: &mut usize) -> Result<()> {
    let numerator = i128::from(padding_ns).abs() * i128::from(RATE);
    let frames = (numerator + 500_000_000) / 1_000_000_000;
    ensure!(
        (numerator - frames * 1_000_000_000).abs() < i128::from(RATE),
        "Opus discard padding is not an integral sample count"
    );
    let frames = usize::try_from(frames).context("Opus padding overflow")?;
    ensure!(frames <= pcm.len(), "Opus padding exceeds decoded block");
    if padding_ns < 0 {
        pcm.drain(..frames);
    } else {
        pcm.truncate(pcm.len() - frames);
    }
    let skip = (*pre_skip).min(pcm.len());
    pcm.drain(..skip);
    *pre_skip -= skip;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_padding_spans_laces_before_stream_pre_skip() {
        let mut positive = (0..2880).map(|x| x as f32).collect::<Vec<_>>();
        let mut skip = 312;
        trim_block(&mut positive, 30_000_000, &mut skip).unwrap();
        assert_eq!(positive, (312..1440).map(|x| x as f32).collect::<Vec<_>>());
        let mut negative = (0..2880).map(|x| x as f32).collect::<Vec<_>>();
        let mut skip = 312;
        trim_block(&mut negative, -30_000_000, &mut skip).unwrap();
        assert_eq!(negative, (1752..2880).map(|x| x as f32).collect::<Vec<_>>());
    }
    #[test]
    fn pre_skip_carries_across_blocks_and_impossible_padding_fails() {
        let mut skip = 1200;
        let mut first = vec![1.0; 960];
        trim_block(&mut first, 0, &mut skip).unwrap();
        assert!(first.is_empty());
        assert_eq!(skip, 240);
        let mut second = vec![2.0; 960];
        trim_block(&mut second, 0, &mut skip).unwrap();
        assert_eq!(second.len(), 720);
        assert_eq!(skip, 0);
        assert!(trim_block(&mut second, 1_000_000_000, &mut skip).is_err());
        assert!(trim_block(&mut second, 1, &mut skip).is_err());
    }
    fn element(id: u32, bytes: &[u8]) -> Vec<u8> {
        let ib = id.to_be_bytes();
        let start = ib.iter().position(|b| *b != 0).unwrap();
        let mut out = ib[start..].to_vec();
        let width = (1..=8)
            .find(|n| bytes.len() < ((1usize << (7 * n)) - 1))
            .unwrap();
        let size = (bytes.len() as u64 | (1u64 << (7 * width))).to_be_bytes();
        out.extend_from_slice(&size[8 - width..]);
        out.extend_from_slice(bytes);
        out
    }
    fn num(id: u32, value: u64) -> Vec<u8> {
        element(id, &value.to_be_bytes())
    }
    fn document(blocks: usize, padding: i64, skip: u16, stereo: bool) -> Vec<u8> {
        let mut head = b"OpusHead".to_vec();
        head.extend_from_slice(&[1, if stereo { 2 } else { 1 }]);
        head.extend_from_slice(&skip.to_le_bytes());
        head.extend_from_slice(&48000u32.to_le_bytes());
        head.extend_from_slice(&[0, 0, 0]);
        let mut track = num(0xd7, 1);
        track.extend(num(0x73c5, 1));
        track.extend(num(0x83, 2));
        track.extend(element(0x86, b"A_OPUS"));
        track.extend(element(0x63a2, &head));
        track.extend(num(0x56aa, u64::from(skip) * 1_000_000_000 / 48000));
        track.extend(num(0x56bb, 80_000_000));
        let mut audio = element(0xb5, &48000f64.to_be_bytes());
        audio.extend(num(0x9f, if stereo { 2 } else { 1 }));
        track.extend(element(0xe1, &audio));
        let mut info = num(0x2ad7b1, 1_000_000);
        info.extend(element(0x4d80, b"gigastt tests"));
        info.extend(element(0x5741, b"synthetic fixture"));
        let mut segment = element(0x1549a966, &info);
        segment.extend(element(0x1654ae6b, &element(0xae, &track)));
        for index in 0..blocks {
            let mut cluster = num(0xe7, (index * 60) as u64);
            let mut block = vec![0x81, 0, 0, 4, 2]; // fixed lacing, three 20ms silence packets
            block.extend_from_slice(&[0xf8, 0xff, 0xfe, 0xf8, 0xff, 0xfe, 0xf8, 0xff, 0xfe]);
            let mut group = element(0xa1, &block);
            group.extend(num(0x9b, 60));
            group.extend(element(0x75a2, &padding.to_be_bytes()));
            cluster.extend(element(0xa0, &group));
            segment.extend(element(0x1f43b675, &cluster));
        }
        let mut ebml = element(0x4282, b"webm");
        ebml.extend(num(0x4287, 4));
        ebml.extend(num(0x4285, 2));
        let mut out = element(0x1a45dfa3, &ebml);
        out.extend(element(0x18538067, &segment));
        out
    }

    #[test]
    fn native_filewindows_laced_trim_window_tail_and_channels() {
        use super::super::stream::{FileWindows, PcmWindows, WindowSpec};
        use bytes::Bytes;
        for stereo in [false, true] {
            for padding in [30_000_000, -30_000_000, 0] {
                let data = Bytes::from(document(40, padding, 312, stereo));
                let full = FileWindows::decode_bytes(data.clone(), None).unwrap();
                let expected_48k =
                    40 * (2880 - (padding.unsigned_abs() as usize * 48000 / 1_000_000_000)) - 312;
                assert!((full.len() as isize - (expected_48k / 3) as isize).abs() <= 2);
                let spec = WindowSpec::new(4000, 4000, 1280);
                let mut windows = FileWindows::from_bytes(data, spec, None).unwrap();
                let mut count = 0;
                while let Some(window) = windows.next_window().unwrap() {
                    assert_eq!(
                        window.samples,
                        &full[window.start_sample..window.start_sample + window.samples.len()]
                    );
                    assert!(window.samples.len() <= 4000);
                    count += 1;
                }
                assert!(count > 1);
                assert_eq!(windows.total_16k_samples(), full.len());
            }
        }
    }
    #[test]
    fn native_filewindows_budget_abort_and_truncated_tail_fail() {
        use super::super::stream::{FileWindows, WindowSpec};
        use bytes::Bytes;
        let data = document(50, 0, 312, false);
        assert!(FileWindows::decode_bytes(Bytes::from(data.clone()), Some(0.1)).is_err());
        let windows = FileWindows::from_bytes(
            Bytes::from(data.clone()),
            WindowSpec::new(4000, 4000, 1280),
            None,
        )
        .unwrap();
        assert!(windows.drain_to_vec_with_abort(Some(&|| true)).is_err());
        let polls = std::sync::atomic::AtomicUsize::new(0);
        let abort = || polls.fetch_add(1, std::sync::atomic::Ordering::Relaxed) >= 8;
        let windows = FileWindows::from_bytes(
            Bytes::from(data.clone()),
            WindowSpec::new(4000, 4000, 1280),
            None,
        )
        .unwrap();
        assert!(windows.drain_to_vec_with_abort(Some(&abort)).is_err());
        assert!(polls.load(std::sync::atomic::Ordering::Relaxed) >= 8);
        assert!(
            FileWindows::decode_bytes(Bytes::from(data[..data.len() - 1].to_vec()), None).is_err()
        );
    }
    #[test]
    fn source_has_duration_independent_retention_and_midstream_abort() {
        use std::io::Cursor;
        let mut source = UnlaceSource::probe(
            Cursor::new(document(2000, 0, 312, false)),
            ChannelSelect::Mono,
            None,
        )
        .unwrap()
        .unwrap();
        let mut count = 0;
        while source.next_block(None).unwrap().is_some() {
            assert!(source.block.capacity() <= 5760);
            count += 1;
        }
        assert_eq!(count, 2000);
        let mut source = UnlaceSource::probe(
            Cursor::new(document(3, 0, 312, false)),
            ChannelSelect::Mono,
            None,
        )
        .unwrap()
        .unwrap();
        source.next_block(None).unwrap();
        assert!(source.next_block(Some(&|| true)).is_err());
    }
    #[test]
    fn strict_metadata_rejections_and_all_entrypoints_agree() {
        use super::super::decode::{
            decode_audio_bytes, decode_audio_bytes_shared_channels, decode_audio_file,
        };
        use bytes::Bytes;
        for (data, reason) in [
            (
                &include_bytes!("../../../tests/fixtures/opus/opus_sf16.webm")[..],
                "InputRateMismatch",
            ),
            (
                &include_bytes!("../../../tests/fixtures/opus/opus_tone_webm_live.webm")[..],
                "PaddingExceedsDuration",
            ),
        ] {
            assert!(format!("{:#}", decode_audio_bytes(data).unwrap_err()).contains(reason));
            assert!(
                format!(
                    "{:#}",
                    decode_audio_bytes_shared_channels(Bytes::copy_from_slice(data)).unwrap_err()
                )
                .contains(reason)
            );
            let mut file = tempfile::NamedTempFile::with_suffix(".webm").unwrap();
            std::io::Write::write_all(&mut file, data).unwrap();
            assert!(
                format!(
                    "{:#}",
                    decode_audio_file(file.path().to_str().unwrap()).unwrap_err()
                )
                .contains(reason)
            );
        }
        let data = Bytes::from(document(40, 30_000_000, 312, true));
        let mono = decode_audio_bytes(&data).unwrap();
        let split = decode_audio_bytes_shared_channels(data.clone()).unwrap();
        assert_eq!(split.len(), 2);
        assert!(split.iter().all(|c| c.len() == mono.len()));
        let mut file = tempfile::NamedTempFile::with_suffix(".webm").unwrap();
        std::io::Write::write_all(&mut file, &data).unwrap();
        assert_eq!(
            decode_audio_file(file.path().to_str().unwrap()).unwrap(),
            mono
        );
    }

    #[test]
    fn live_unknown_size_clean_eof_and_truncated_payload_are_distinct() {
        use super::super::decode::decode_audio_bytes;
        let finite = document(4, 0, 312, false);
        let expected = decode_audio_bytes(&finite).unwrap();
        let mut live = finite.clone();
        for id in [[0x18, 0x53, 0x80, 0x67], [0x1f, 0x43, 0xb6, 0x75]] {
            let positions = finite
                .windows(4)
                .enumerate()
                .filter_map(|(i, b)| (b == id).then_some(i + 4))
                .collect::<Vec<_>>();
            for position in positions {
                let width = finite[position].leading_zeros() as usize + 1;
                live[position] = ((1u16 << (9 - width)) - 1) as u8;
                live[position + 1..position + width].fill(0xff);
            }
        }
        assert_eq!(decode_audio_bytes(&live).unwrap(), expected);
        assert!(decode_audio_bytes(&live[..live.len() - 1]).is_err());
    }

    #[test]
    fn extreme_container_timestamps_return_error_without_panicking() {
        use super::super::decode::decode_audio_bytes;
        let mut data = document(2, 0, 312, false);
        let scale = data
            .windows(3)
            .position(|b| b == [0x2a, 0xd7, 0xb1])
            .unwrap()
            + 4;
        data[scale..scale + 8].copy_from_slice(&1_000_000_000_000_000_000u64.to_be_bytes());
        let stamps = data
            .windows(2)
            .enumerate()
            .filter_map(|(i, b)| (b == [0xe7, 0x88]).then_some(i + 2))
            .collect::<Vec<_>>();
        data[stamps[1]..stamps[1] + 8].copy_from_slice(&u64::MAX.to_be_bytes());
        assert!(decode_audio_bytes(&data).is_err());
    }

    #[test]
    fn nonopus_matroska_with_legacy_metadata_uses_legacy_decoder() {
        use super::super::decode::{decode_audio_bytes, decode_audio_file};
        let data = include_bytes!("../../../tests/fixtures/unlace/nonopus-vorbis.mka");
        let mut demux = Demuxer::new(
            std::io::Cursor::new(data),
            AudioSelection::First,
            DemuxLimits::default(),
        )
        .unwrap();
        let mut refused = false;
        loop {
            match demux.poll() {
                Err(_) => {
                    refused = true;
                    break;
                }
                Ok(DemuxEvent::End(_)) => break,
                _ => {}
            }
        }
        assert!(refused, "fixture must exercise an admission difference");
        let pcm = decode_audio_bytes(data).unwrap();
        assert!(!pcm.is_empty());
        let late = include_bytes!("../../../tests/fixtures/unlace/nonopus-late-tracks.mka");
        assert_eq!(decode_audio_bytes(late).unwrap(), pcm);
        let mut file = tempfile::NamedTempFile::with_suffix(".mka").unwrap();
        std::io::Write::write_all(&mut file, data).unwrap();
        assert_eq!(
            decode_audio_file(file.path().to_str().unwrap()).unwrap(),
            pcm
        );
    }

    #[test]
    fn probe_budget_caps_reads_across_seeks_and_handles_short_reads() {
        let enabled = Arc::new(AtomicBool::new(true));
        let mut input = ProbeInput {
            inner: std::io::Cursor::new(vec![7u8; 32]),
            remaining: 5,
            bounded: Arc::clone(&enabled),
        };
        let mut buf = [0u8; 16];
        assert_eq!(input.read(&mut buf).unwrap(), 5);
        input.seek(SeekFrom::Start(0)).unwrap();
        assert!(input.read(&mut buf).is_err());
        enabled.store(false, Ordering::Relaxed);
        assert_eq!(input.read(&mut buf).unwrap(), 16);
        struct ShortReads(std::io::Cursor<Vec<u8>>);
        impl Read for ShortReads {
            fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
                let n = b.len().min(1);
                self.0.read(&mut b[..n])
            }
        }
        impl Seek for ShortReads {
            fn seek(&mut self, p: SeekFrom) -> std::io::Result<u64> {
                self.0.seek(p)
            }
        }
        let mut source = UnlaceSource::probe(
            ShortReads(std::io::Cursor::new(document(2, 0, 312, false))),
            ChannelSelect::Mono,
            None,
        )
        .unwrap()
        .unwrap();
        assert_eq!(source.next_block(None).unwrap().unwrap().len(), 2880 - 312);
    }

    #[test]
    fn unknown_codec_falls_back_before_pcm_but_invalid_opus_does_not() {
        use std::io::Cursor;
        let mut unknown = document(2, 0, 312, false);
        let index = unknown.windows(6).position(|p| p == b"A_OPUS").unwrap();
        unknown[index..index + 6].copy_from_slice(b"A_TEST");
        assert!(
            UnlaceSource::probe(Cursor::new(unknown), ChannelSelect::Mono, None)
                .unwrap()
                .is_none()
        );
        let mut gain = document(2, 0, 312, false);
        let head = gain.windows(8).position(|p| p == b"OpusHead").unwrap();
        gain[head + 16] = 1;
        assert!(UnlaceSource::probe(Cursor::new(gain), ChannelSelect::Mono, None).is_err());
    }
}
