//! Bounded file buffering that preserves cached bytes across absolute AU seeks.

use std::io::{self, BufReader, Read, Seek, SeekFrom};

const CAPACITY: usize = 64 * 1024;

pub(super) struct SmartBuffered<R> {
    inner: BufReader<R>,
    // Unknown initially and after any I/O error. Never base a relative seek on
    // a position that an unsuccessful BufReader operation may have invalidated.
    position: Option<u64>,
}

impl<R: Read + Seek> SmartBuffered<R> {
    pub(super) fn new(inner: R) -> Self {
        Self {
            inner: BufReader::with_capacity(CAPACITY, inner),
            position: None,
        }
    }

    pub(super) fn get_ref(&self) -> &R {
        self.inner.get_ref()
    }

    /// Drop cached bytes and reset the underlying reader absolutely. File
    /// clones share a physical offset: a clone or fallback must not reuse a
    /// stale cached position after another handle has moved that offset.
    pub(super) fn force_rewind(&mut self) -> io::Result<()> {
        let result = self.inner.seek(SeekFrom::Start(0));
        self.position = result.as_ref().ok().copied();
        result.map(|_| ())
    }
}

impl<R: Read + Seek> Read for SmartBuffered<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self.inner.read(buf) {
            Ok(count) => {
                self.position = self
                    .position
                    .and_then(|position| position.checked_add(count as u64));
                Ok(count)
            }
            Err(error) => {
                self.position = None;
                Err(error)
            }
        }
    }
}

impl<R: Read + Seek> Seek for SmartBuffered<R> {
    fn seek(&mut self, request: SeekFrom) -> io::Result<u64> {
        let result = match (request, self.position) {
            (SeekFrom::Start(target), Some(position)) => {
                let distance = i128::from(target) - i128::from(position);
                match i64::try_from(distance) {
                    Ok(distance) => self.inner.seek_relative(distance).map(|()| target),
                    Err(_) => self.inner.seek(request),
                }
            }
            _ => self.inner.seek(request),
        };
        self.position = result.as_ref().ok().copied();
        result
    }

    fn stream_position(&mut self) -> io::Result<u64> {
        if let Some(position) = self.position {
            return Ok(position);
        }
        let result = self.inner.stream_position();
        self.position = result.as_ref().ok().copied();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::io::Cursor;
    use std::rc::Rc;

    #[derive(Default)]
    struct Counts {
        reads: Cell<usize>,
        seeks: Cell<usize>,
    }

    struct Source {
        cursor: Cursor<Vec<u8>>,
        counts: Rc<Counts>,
        read_error: bool,
        seek_error: bool,
    }

    impl Read for Source {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            self.counts.reads.set(self.counts.reads.get() + 1);
            if std::mem::take(&mut self.read_error) {
                return Err(io::Error::other("injected read failure"));
            }
            self.cursor.read(buf)
        }
    }

    impl Seek for Source {
        fn seek(&mut self, request: SeekFrom) -> io::Result<u64> {
            self.counts.seeks.set(self.counts.seeks.get() + 1);
            if std::mem::take(&mut self.seek_error) {
                return Err(io::Error::other("injected seek failure"));
            }
            self.cursor.seek(request)
        }
    }

    fn source() -> (Source, Rc<Counts>) {
        let counts = Rc::new(Counts::default());
        (
            Source {
                cursor: Cursor::new((0..200).collect()),
                counts: counts.clone(),
                read_error: false,
                seek_error: false,
            },
            counts,
        )
    }

    #[test]
    fn test_absolute_au_seeks_reuse_buffer_without_changing_bytes() {
        let (source, counts) = source();
        let mut reader = SmartBuffered::new(source);
        reader.force_rewind().expect("rewind");
        let mut bytes = [0; 4];
        reader.read_exact(&mut bytes).expect("first bytes");
        assert_eq!(bytes, [0, 1, 2, 3]);
        let seeks = counts.seeks.get();
        reader.seek(SeekFrom::Start(20)).expect("forward AU");
        reader.read_exact(&mut bytes).expect("forward bytes");
        assert_eq!(bytes, [20, 21, 22, 23]);
        reader.seek(SeekFrom::Start(3)).expect("backward AU");
        reader.read_exact(&mut bytes).expect("backward bytes");
        assert_eq!(bytes, [3, 4, 5, 6]);
        assert_eq!(reader.stream_position().expect("position"), 7);
        assert_eq!(counts.reads.get(), 1);
        assert_eq!(counts.seeks.get(), seeks);
    }

    #[test]
    fn test_current_end_and_eof_match_unbuffered_reader() {
        let (source, _) = source();
        let mut reader = SmartBuffered::new(source);
        let mut reference = Cursor::new((0..200).collect::<Vec<u8>>());
        for request in [
            SeekFrom::Start(4),
            SeekFrom::Current(-2),
            SeekFrom::End(-12),
            SeekFrom::Start(199),
            SeekFrom::Start(400),
            SeekFrom::Start(0),
        ] {
            assert_eq!(
                reader.seek(request).expect("buffered seek"),
                reference.seek(request).expect("reference seek")
            );
            let (mut actual, mut expected) = ([0; 8], [0; 8]);
            assert_eq!(
                reader.read(&mut actual).expect("buffered read"),
                reference.read(&mut expected).expect("reference read")
            );
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_read_and_seek_failures_invalidate_relative_position() {
        let (mut source, _) = source();
        source.read_error = true;
        let mut reader = SmartBuffered::new(source);
        assert!(reader.read(&mut [0; 1]).is_err());
        assert!(reader.position.is_none());
        reader.seek(SeekFrom::Start(4)).expect("recover read");
        let mut bytes = [0; 2];
        reader
            .read_exact(&mut bytes)
            .expect("bytes after read error");
        assert_eq!(bytes, [4, 5]);
        reader.inner.get_mut().seek_error = true;
        assert!(reader.seek(SeekFrom::End(-1)).is_err());
        assert!(reader.position.is_none());
        reader.seek(SeekFrom::Start(4)).expect("recover seek");
        reader
            .read_exact(&mut bytes)
            .expect("bytes after seek error");
        assert_eq!(bytes, [4, 5]);
    }

    #[test]
    fn test_force_rewind_invalidates_cache_after_shared_offset_moves() {
        let (source, _) = source();
        let mut reader = SmartBuffered::new(source);
        reader.force_rewind().expect("rewind");
        reader.read_exact(&mut [0; 4]).expect("fill buffer");
        // This models File::try_clone: another handle moved the physical
        // offset while this wrapper still retained cached bytes.
        reader.inner.get_mut().cursor.set_position(123);
        reader.force_rewind().expect("absolute reset");
        let mut bytes = [0; 4];
        reader.read_exact(&mut bytes).expect("reset bytes");
        assert_eq!(bytes, [0, 1, 2, 3]);
        assert_eq!(reader.stream_position().expect("reset position"), 4);
    }
}
