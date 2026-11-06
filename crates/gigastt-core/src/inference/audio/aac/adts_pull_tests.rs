#![allow(clippy::unwrap_used, clippy::expect_used)]
use super::*;
use bytes::Bytes;
use std::io::{Cursor, Error, Write};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

type Block = (usize, u32, usize, Vec<u32>);
type Outcome = (Vec<Block>, Option<String>);
fn block(block: AacBlock) -> Block {
    (
        block.frames,
        block.rate,
        block.channels,
        block.samples.into_iter().map(f32::to_bits).collect(),
    )
}
fn error_tag(error: anyhow::Error) -> String {
    error
        .downcast_ref::<AacError>()
        .map_or_else(|| format!("{error:?}"), |e| format!("{e:?}"))
}
fn collect(mut next: impl FnMut() -> Result<Option<AacBlock>>) -> Outcome {
    let mut blocks = Vec::new();
    loop {
        match next() {
            Ok(Some(value)) => blocks.push(block(value)),
            Ok(None) => return (blocks, None),
            Err(error) => return (blocks, Some(error_tag(error))),
        }
    }
}
fn bare(bytes: Vec<u8>) -> Vec<u8> {
    let at = if bytes.starts_with(b"ID3") {
        10 + bytes[6..10]
            .iter()
            .fold(0usize, |n, b| (n << 7) | usize::from(*b))
            + if bytes[5] & 0x10 != 0 { 10 } else { 0 }
    } else {
        0
    };
    let bytes = bytes[at..].to_vec();
    assert!(syom::sniff_is_adts(&bytes));
    bytes
}
fn fixture(rate: u32, channels: usize, profile: u8) -> Vec<u8> {
    let pcm: Vec<Vec<f32>> = (0..channels)
        .map(|ch| {
            (0..rate * 2 + 13)
                .map(|i| (i as f32 * (0.03 + ch as f32 * 0.01)).sin() * 0.1)
                .collect()
        })
        .collect();
    let options =
        syom::EncodeOptions::adts().with_bitrate_bps(48_000.min(rate * channels as u32 * 4));
    let options = match profile {
        1 => options.with_he(true),
        2 => options.with_he_v2(true).with_bitrate_bps(32_000),
        _ => options,
    };
    let bytes = bare(syom::encode_with(&pcm, rate, &options).unwrap());
    if profile != 0 {
        assert_eq!((bytes[2] >> 2) & 15, 6, "HE core must be24kHz");
    }
    bytes
}
fn boundaries(bytes: &[u8]) -> Vec<usize> {
    let mut result = vec![0];
    let mut at = 0;
    while at < bytes.len() {
        assert!(syom::sniff_is_adts(&bytes[at..]));
        at += (usize::from(bytes[at + 3] & 3) << 11)
            | (usize::from(bytes[at + 4]) << 3)
            | (usize::from(bytes[at + 5]) >> 5);
        assert!(at <= bytes.len());
        result.push(at);
    }
    result
}
fn old(bytes: Vec<u8>) -> Outcome {
    let mut worker = super::super::spawn_worker(
        super::super::input::Reader::bytes(Bytes::from(bytes)),
        ChannelSelect::Mono,
        None,
    )
    .unwrap();
    collect(|| worker.recv())
}
fn pull<R: Read>(reader: R) -> (Outcome, AdtsPull<R>) {
    let mut source = AdtsPull::new(reader, ChannelSelect::Mono, None);
    source.forwarded = Some(Vec::new());
    let result = collect(|| source.recv());
    assert!(source.recv().unwrap().is_none());
    assert!(source.stash.len() <= STASH_BYTES);
    (result, source)
}

#[test]
fn test_aac_bare_adts_pull_native_batches_bits_file_bytes_profiles() {
    let cases = [
        (8_000, 1, 0),
        (8_000, 2, 0),
        (22_050, 1, 0),
        (22_050, 2, 0),
        (48_000, 1, 0),
        (48_000, 2, 0),
        (48_000, 2, 1),
        (48_000, 2, 2),
    ];
    for (rate, channels, profile) in cases {
        let bytes = fixture(rate, channels, profile);
        let expected = old(bytes.clone());
        assert!(expected.1.is_none(), "{expected:?}");
        assert_eq!((expected.0[0].1, expected.0[0].2), (rate, channels));
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&bytes).unwrap();
        let (actual, memory) = pull(Cursor::new(bytes.clone()));
        assert_eq!(actual, expected);
        assert_eq!(memory.forwarded.unwrap(), bytes);
        let (actual, disk) = pull(std::fs::File::open(file.path()).unwrap());
        assert_eq!(actual, expected);
        assert_eq!(disk.forwarded.unwrap(), bytes);
    }
}

struct Chunked {
    bytes: Cursor<Vec<u8>>,
    chunk: usize,
    fail: bool,
    interrupt: bool,
    dropped: Arc<AtomicBool>,
    reads: Arc<AtomicUsize>,
}
impl Chunked {
    fn new(bytes: Vec<u8>, chunk: usize, fail: bool) -> Self {
        Self {
            bytes: Cursor::new(bytes),
            chunk,
            fail,
            interrupt: false,
            dropped: Arc::new(AtomicBool::new(false)),
            reads: Arc::new(AtomicUsize::new(0)),
        }
    }
}
impl Read for Chunked {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        if self.interrupt {
            self.interrupt = false;
            return Err(Error::from(ErrorKind::Interrupted));
        }
        if self.fail && self.bytes.position() as usize == self.bytes.get_ref().len() {
            return Err(Error::from(ErrorKind::PermissionDenied));
        }
        let n = out.len().min(self.chunk);
        self.bytes.read(&mut out[..n])
    }
}
impl Drop for Chunked {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::Relaxed);
    }
}

// The old raw worker's actual feed_read/finish driver, with its same full-block
// emission and incomplete-tail-on-error discard; faults need a generic reader.
fn reference<R: Read>(mut reader: R) -> Outcome {
    let mut decoder = Decoder::new(stream_options(None));
    let mut batch: Option<AacBlock> = None;
    let mut blocks = Vec::new();
    let mut emit = |frame: Frame<'_>| {
        let b = batch.get_or_insert_with(|| AacBlock {
            frames: 0,
            samples: Vec::new(),
            rate: frame.sample_rate,
            channels: frame.planar.len(),
        });
        b.frames += frame.samples;
        append_plane(&mut b.samples, &frame, ChannelSelect::Mono);
        if b.frames >= BATCH_FRAMES {
            blocks.push(block(batch.take().unwrap()));
        }
        Ok(())
    };
    let result = decoder
        .feed_read(&mut reader, &mut emit)
        .and_then(|_| decoder.finish(&mut emit).map(|_| ()));
    match result {
        Ok(()) => {
            if let Some(b) = batch {
                blocks.push(block(b));
            }
            (blocks, None)
        }
        Err(error) => (blocks, Some(error_tag(map_aac(error)))),
    }
}

#[test]
fn test_aac_bare_adts_pull_forwards_junk_and_split_headers_exactly_once() {
    let raw = fixture(48_000, 1, 0);
    let edges = boundaries(&raw);
    let mut bytes = Vec::new();
    let mut seed = 145u32;
    for (i, pair) in edges[..9].windows(2).enumerate() {
        bytes.extend_from_slice(&raw[pair[0]..pair[1]]);
        for _ in 0..i + 1 {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            bytes.push((seed >> 24) as u8 & 0x7f);
        }
    }
    bytes.extend_from_slice(&raw[edges[8]..]);
    for chunk in [1, 2, 5, 6, 7, 13, 4096] {
        let expected = reference(Chunked::new(bytes.clone(), chunk, false));
        assert!(
            expected.1.is_none(),
            "valid resync fixture must decode: {expected:?}"
        );
        let (actual, source) = pull(Chunked::new(bytes.clone(), chunk, false));
        assert_eq!(actual, expected, "chunk={chunk}");
        let forwarded = source.forwarded.unwrap();
        assert_eq!(forwarded, bytes[..forwarded.len()]);
        if actual.1.is_none() {
            assert_eq!(forwarded, bytes);
        }
    }
}

fn header_size(header: &mut [u8], len: usize) {
    header[3] = (header[3] & !3) | ((len >> 11) as u8 & 3);
    header[4] = (len >> 3) as u8;
    header[5] = (header[5] & 31) | (((len & 7) as u8) << 5);
}
fn crc_frame(raw: &[u8]) -> Vec<u8> {
    let end = boundaries(raw)[1];
    let payload = &raw[7..end];
    assert_eq!(payload[0] >> 5, 0);
    assert!(payload.len() * 8 >= 195);
    let mut header = raw[..7].to_vec();
    header[1] &= !1;
    header_size(&mut header, end + 2);
    // This authored mono encoder fixture starts with an SCE exceeding192bodybits.
    // ADTS protects the56headerbits and192SCEbodybits afterthe3-bit elementID.
    let bits = (0..56)
        .map(|i| (header[i / 8] >> (7 - i % 8)) & 1)
        .chain((3..195).map(|i| (payload[i / 8] >> (7 - i % 8)) & 1));
    let mut crc = 0xffffu16;
    for bit in bits {
        let feedback = (crc >> 15) ^ u16::from(bit);
        crc <<= 1;
        if feedback != 0 {
            crc ^= 0x8005;
        }
    }
    header.extend_from_slice(&crc.to_be_bytes());
    header.extend_from_slice(payload);
    header
}

#[test]
fn test_aac_bare_adts_pull_partial_crc_multirdb_and_completed_before_error() {
    let raw = fixture(48_000, 1, 0);
    let edges = boundaries(&raw);
    for n in [1, 5, 6, 7, raw.len() - 1] {
        let bytes = raw[..n].to_vec();
        assert_eq!(pull(Cursor::new(bytes.clone())).0, old(bytes));
    }
    let protected = crc_frame(&raw);
    let expected = old(protected.clone());
    assert!(expected.1.is_none(), "{expected:?}");
    assert_eq!(pull(Cursor::new(protected.clone())).0, expected);
    let mut corrupt = protected;
    corrupt[8] ^= 1;
    assert_eq!(pull(Cursor::new(corrupt.clone())).0, old(corrupt));
    let mut multi = raw[..7].to_vec();
    multi[6] = (multi[6] & !3) | 3;
    let payload = &raw[7..edges[1]];
    header_size(&mut multi, 7 + payload.len() * 4);
    for _ in 0..4 {
        multi.extend_from_slice(payload);
    }
    let expected = old(multi.clone());
    assert!(expected.1.is_none(), "{expected:?}");
    assert_eq!(expected.0[0].0, 4096);
    assert_eq!(pull(Cursor::new(multi)).0, expected);
    for profile in [1, 2] {
        let he = fixture(48_000, 2, profile);
        let first_end = boundaries(&he)[1];
        let payload = &he[7..first_end];
        let mut multi = he[..7].to_vec();
        multi[6] = (multi[6] & !3) | 3;
        header_size(&mut multi, 7 + payload.len() * 4);
        for _ in 0..4 {
            multi.extend_from_slice(payload);
        }
        let expected = old(multi.clone());
        assert!(expected.1.is_none(), "{expected:?}");
        assert_eq!(expected.0[0].0, 8192);
        assert_eq!(pull(Cursor::new(multi)).0, expected);
    }
    let mut broken = raw[..edges[24]].to_vec();
    let mut header = raw[..8].to_vec();
    header[2] = (header[2] & 0xc3) | (13 << 2);
    header_size(&mut header, 8191);
    broken.extend(header);
    let expected = old(broken.clone());
    assert_eq!(expected.0.len(), 1);
    assert!(expected.1.is_some());
    assert_eq!(pull(Cursor::new(broken)).0, expected);
}

#[test]
fn test_aac_bare_adts_pull_preserves_header_io_precedence_and_interrupted_reads() {
    let raw = fixture(48_000, 1, 0);
    let edges = boundaries(&raw);
    for mutation in 0..7 {
        let mut header = raw[..8].to_vec();
        header_size(&mut header, 8191);
        match mutation {
            0 | 1 => header[2] = (header[2] & 0xc3) | ((13 + mutation as u8) << 2),
            2 | 3 => {
                header[1] &= !1;
                header_size(&mut header, 7 + mutation - 2);
            }
            4 => header[1] |= 2,
            5 => header[2] = (header[2] & 0xc3) | (15 << 2),
            _ => header_size(&mut header, 6),
        }
        for padding in [0, 1, 4, 5, 6, 7] {
            let mut bytes = raw[..edges[1]].to_vec();
            bytes.extend(vec![0; padding]);
            bytes.extend_from_slice(&header);
            for chunk in [7, 13, 4096] {
                let expected = reference(Chunked::new(bytes.clone(), chunk, true));
                assert_eq!(
                    pull(Chunked::new(bytes.clone(), chunk, true)).0,
                    expected,
                    "mutation={mutation}padding={padding}chunk={chunk}"
                );
            }
        }
    }
    let mut reader = Chunked::new(raw.clone(), 17, false);
    reader.interrupt = true;
    assert_eq!(pull(reader).0, old(raw.clone()));
    let bytes = raw[..edges[25] - 1].to_vec();
    let expected = reference(Chunked::new(bytes.clone(), 4096, true));
    assert_eq!(expected.0.len(), 1);
    assert!(expected.1.is_some());
    assert_eq!(pull(Chunked::new(bytes, 4096, true)).0, expected);
}

#[test]
fn test_aac_bare_adts_pull_public_budget_and_cancel() {
    let raw = fixture(48_000, 1, 0);
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(&raw).unwrap();
    for result in [
        crate::inference::audio::decode_audio_bytes_shared_bounded(
            Bytes::from(raw.clone()),
            Some(0.1),
        ),
        crate::inference::audio::decode::decode_audio_file_bounded(
            file.path().to_str().unwrap(),
            Some(0.1),
        ),
    ] {
        assert!(matches!(
            result
                .unwrap_err()
                .downcast_ref::<crate::error::GigasttError>(),
            Some(crate::error::GigasttError::AudioTooLong { .. })
        ));
    }
    let calls = AtomicUsize::new(0);
    let abort = || calls.fetch_add(1, Ordering::Relaxed) >= 3;
    let error = crate::inference::audio::decode::decode_audio_bytes_with_abort(
        Bytes::from(raw.clone()),
        None,
        Some(&abort),
    )
    .unwrap_err();
    assert!(matches!(
        error.downcast_ref::<crate::error::GigasttError>(),
        Some(crate::error::GigasttError::Cancelled)
    ));
}

#[test]
fn test_aac_bare_adts_pull_send_and_drop() {
    fn is_send<T: Send>() {}
    is_send::<super::super::Producer>();
    let reader = Chunked::new(fixture(48_000, 1, 0), 4096, false);
    let dropped = reader.dropped.clone();
    let reads = reader.reads.clone();
    let mut source = AdtsPull::new(reader, ChannelSelect::Mono, None);
    assert!(source.recv().unwrap().is_some());
    let before = reads.load(Ordering::Relaxed);
    drop(source);
    assert!(dropped.load(Ordering::Relaxed));
    assert_eq!(reads.load(Ordering::Relaxed), before);
}

#[test]
fn test_aac_bare_adts_pull_he_ps_small_stack() {
    for profile in [0, 1, 2] {
        let raw = fixture(48_000, if profile == 0 { 1 } else { 2 }, profile);
        let check = raw.clone();
        let stack = if cfg!(debug_assertions) {
            512 * 1024
        } else {
            128 * 1024
        };
        let actual = std::thread::Builder::new()
            .stack_size(stack)
            .spawn(move || pull(Cursor::new(raw)).0)
            .unwrap()
            .join()
            .unwrap();
        assert_eq!(actual, old(check));
    }
}
