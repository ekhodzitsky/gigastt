//! Shared callback driver for windowed decode and synchronous channel scans.
use bytes::Bytes;
use std::io::{Cursor, Read, Seek, SeekFrom};
use syom::{AacError, DecodeOptions, Decoder, Frame, M4aSeek};

use super::buffered::SmartBuffered;

pub(super) enum Reader {
    File(std::fs::File),
    Buffered(SmartBuffered<std::fs::File>),
    Mem(Cursor<Bytes>),
}
impl Reader {
    pub(super) fn bytes(data: Bytes) -> Self {
        Self::Mem(Cursor::new(data))
    }

    pub(super) fn clone_for_pull(&self) -> std::io::Result<Self> {
        let mut copy = match self {
            Self::Mem(cursor) => Self::bytes(cursor.get_ref().clone()),
            Self::File(file) => Self::Buffered(SmartBuffered::new(file.try_clone()?)),
            Self::Buffered(file) => Self::Buffered(SmartBuffered::new(file.get_ref().try_clone()?)),
        };
        copy.force_rewind()?;
        Ok(copy)
    }

    // File::try_clone shares the OS cursor. Always issue an absolute rewind,
    // even when a buffer's cached logical position already claims to be zero.
    pub(super) fn force_rewind(&mut self) -> std::io::Result<()> {
        match self {
            Self::Buffered(file) => file.force_rewind(),
            other => other.seek(SeekFrom::Start(0)).map(|_| ()),
        }
    }
}
impl Read for Reader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Self::File(file) => file.read(buf),
            Self::Buffered(file) => file.read(buf),
            Self::Mem(cursor) => cursor.read(buf),
        }
    }
}
impl Seek for Reader {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        match self {
            Self::File(file) => file.seek(pos),
            Self::Buffered(file) => file.seek(pos),
            Self::Mem(cursor) => cursor.seek(pos),
        }
    }
}

pub(super) fn decode_reader<R: Read + Seek>(
    reader: &mut R,
    opts: DecodeOptions,
    emit: impl FnMut(Frame<'_>) -> syom::Result<()>,
) -> syom::Result<()> {
    decode_reader_with_mode(reader, opts, false, emit)
}

pub(super) fn decode_reader_with_mode<R: Read + Seek>(
    reader: &mut R,
    opts: DecodeOptions,
    known_push: bool,
    mut emit: impl FnMut(Frame<'_>) -> syom::Result<()>,
) -> syom::Result<()> {
    if !known_push {
        let mut prefix = [0; 64];
        let n = reader.read(&mut prefix)?;
        reader.seek(SeekFrom::Start(0))?;
        if syom::sniff_is_isobmff(&prefix[..n]) {
            match M4aSeek::open(&mut *reader, opts.clone()) {
                Ok(mut seek) => return seek.decode(&mut emit).map(|_| ()),
                Err(AacError::Unsupported(_)) => {} // Bounded fragmented MP4 uses the push decoder.
                Err(error) => return Err(error),
            }
            reader.seek(SeekFrom::Start(0))?;
        }
    }
    let mut decoder = Decoder::new(opts);
    decoder.feed_read(reader, &mut emit)?;
    decoder.finish(emit).map(|_| ())
}

#[cfg(test)]
mod transport_tests {
    use super::super::probe::{Transport, classify_file, classify_slice};
    use super::*;
    use std::io::Write;

    struct Counted<R> {
        inner: R,
        reads: usize,
        seeks: usize,
    }
    impl<R: Read> Read for Counted<R> {
        fn read(&mut self, data: &mut [u8]) -> std::io::Result<usize> {
            self.reads += 1;
            self.inner.read(data)
        }
    }
    impl<R: Seek> Seek for Counted<R> {
        fn seek(&mut self, at: SeekFrom) -> std::io::Result<u64> {
            self.seeks += 1;
            self.inner.seek(at)
        }
    }
    fn fixture() -> Vec<u8> {
        let pcm: Vec<_> = (0..96_013).map(|i| (i as f32 * 0.03).sin() * 0.1).collect();
        syom::encode(&[&pcm], 48_000).unwrap()
    }
    fn compare_count<R: Read + Seek>(inner: R, bytes: &[u8], file: bool) -> (usize, usize) {
        let mut reader = Counted {
            inner,
            reads: 0,
            seeks: 0,
        };
        let transport = if file {
            classify_file(&mut reader).unwrap()
        } else {
            classify_slice(bytes)
        };
        assert!(matches!(
            transport,
            Some(Transport::Streaming | Transport::BareAdts)
        ));
        let initial = (reader.reads, reader.seeks);
        let mut fast = Vec::new();
        decode_reader_with_mode(&mut reader, DecodeOptions::unbounded(), true, |frame| {
            fast.extend(frame.planar[0].iter().map(|v| v.to_bits()));
            Ok(())
        })
        .unwrap();
        let fast_calls = (reader.reads - initial.0, reader.seeks - initial.1);
        reader.seek(SeekFrom::Start(0)).unwrap();
        reader.reads = 0;
        reader.seeks = 0;
        // Frozen adapter did this M4A pull probe before its worker's own probe.
        let mut prefix = [0; 64];
        assert_eq!(reader.read(&mut prefix).unwrap(), prefix.len());
        reader.seek(SeekFrom::Start(0)).unwrap();
        let mut old = Vec::new();
        decode_reader(&mut reader, DecodeOptions::unbounded(), |frame| {
            old.extend(frame.planar[0].iter().map(|v| v.to_bits()));
            Ok(())
        })
        .unwrap();
        assert_eq!(fast, old);
        assert_eq!(
            (reader.reads - fast_calls.0, reader.seeks - fast_calls.1),
            (2, 2)
        );
        fast_calls
    }

    #[test]
    fn test_aac_known_streaming_file_bytes_id3_remove_duplicate_reads_seeks() {
        let tagged = fixture();
        let raw_at = if tagged.starts_with(b"ID3") {
            10 + tagged[6..10]
                .iter()
                .fold(0usize, |n, b| (n << 7) | usize::from(*b))
        } else {
            0
        };
        let raw = tagged[raw_at..].to_vec();
        let mut long_tag = vec![b'I', b'D', b'3', 4, 0, 0, 0, 0, 2, 44];
        long_tag.resize(310, 0);
        long_tag.extend_from_slice(&raw);
        for bytes in [tagged, raw, long_tag] {
            compare_count(Cursor::new(bytes.clone()), &bytes, false);
            let mut file = tempfile::NamedTempFile::new().unwrap();
            file.write_all(&bytes).unwrap();
            compare_count(std::fs::File::open(file.path()).unwrap(), &bytes, true);
        }
    }

    #[test]
    fn test_aac_classified_unsupported_m4a_rewinds_then_push_preserves_typed_error() {
        let pcm = vec![0.0; 4096];
        let mut bytes = syom::encode_with(&[&pcm], 48_000, &syom::EncodeOptions::m4a()).unwrap();
        let moov = bytes.windows(4).position(|w| w == b"moov").unwrap() - 4;
        let size = u32::from_be_bytes(bytes[moov..moov + 4].try_into().unwrap()) as usize;
        let stsd = moov
            + bytes[moov..moov + size]
                .windows(4)
                .position(|w| w == b"stsd")
                .unwrap();
        bytes[stsd + 8..stsd + 12].copy_from_slice(&2u32.to_be_bytes());
        // Fragmented init with an unsupported sample-description switch.
        // Opening rejects the typed feature before reading any coded payload.
        bytes.splice(
            moov + size..moov + size,
            [0, 0, 0, 8, b'm', b'v', b'e', b'x'],
        );
        bytes[moov..moov + 4].copy_from_slice(&((size + 8) as u32).to_be_bytes());
        bytes.extend_from_slice(&[0, 0, 0, 8, b'm', b'o', b'o', b'f']);
        assert_eq!(classify_slice(&bytes), Some(Transport::SeekableM4a));
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&bytes).unwrap();
        for mut input in [
            Reader::bytes(Bytes::from(bytes)),
            Reader::File(std::fs::File::open(file.path()).unwrap()),
        ] {
            assert!(
                super::super::pull::try_open(
                    &mut input,
                    super::super::super::stream::ChannelSelect::Mono,
                    None
                )
                .unwrap()
                .is_none()
            );
            assert_eq!(input.stream_position().unwrap(), 0);
            let mut reader = Counted {
                inner: input,
                reads: 0,
                seeks: 0,
            };
            let fast =
                decode_reader_with_mode(&mut reader, DecodeOptions::unbounded(), true, |_| Ok(()))
                    .unwrap_err();
            let fast_calls = (reader.reads, reader.seeks);
            reader.seek(SeekFrom::Start(0)).unwrap();
            reader.reads = 0;
            reader.seeks = 0;
            let old =
                decode_reader(&mut reader, DecodeOptions::unbounded(), |_| Ok(())).unwrap_err();
            assert!(matches!(fast, AacError::Unsupported(_)));
            assert_eq!(fast.to_string(), old.to_string());
            assert!(reader.reads > fast_calls.0 && reader.seeks > fast_calls.1);
        }
    }
}
