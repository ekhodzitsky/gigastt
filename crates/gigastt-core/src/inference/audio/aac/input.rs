//! Shared callback driver for windowed decode and synchronous channel scans.
use bytes::Bytes;
use std::io::{Cursor, Read, Seek, SeekFrom};
use syom::{AacError, DecodeOptions, Decoder, Frame, M4aSeek};

pub(super) enum Reader {
    File(std::fs::File),
    Mem(Cursor<Bytes>),
}
impl Reader {
    pub(super) fn bytes(data: Bytes) -> Self {
        Self::Mem(Cursor::new(data))
    }
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

pub(super) fn decode_reader(
    reader: &mut Reader,
    opts: DecodeOptions,
    mut emit: impl FnMut(Frame<'_>) -> syom::Result<()>,
) -> syom::Result<()> {
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
    let mut decoder = Decoder::new(opts);
    decoder.feed_read(reader, &mut emit)?;
    decoder.finish(emit).map(|_| ())
}
