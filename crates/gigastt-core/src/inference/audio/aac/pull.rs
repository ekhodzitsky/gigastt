//! Bounded synchronous M4A pull, preserving the existing logical batch cadence.
use anyhow::Result;
use syom::{AacError, M4aSeek};

use super::super::stream::ChannelSelect;
use super::{AacBlock, BATCH_FRAMES, append_plane, input::Reader, map_aac, stream_options};

pub(super) struct Pull {
    source: M4aSeek<Reader>,
    channel: ChannelSelect,
    finished: bool,
}

/// The caller has already classified a bare, seekable M4A prefix.
pub(super) fn try_open(
    input: &mut Reader,
    channel: ChannelSelect,
    max_audio_secs: Option<f64>,
) -> Result<Option<Pull>> {
    let candidate = input.clone_for_pull()?;
    match M4aSeek::open(candidate, stream_options(max_audio_secs)) {
        Ok(source) => Ok(Some(Pull {
            source,
            channel,
            finished: false,
        })),
        Err(AacError::Unsupported(_)) => {
            // A failed open consumed the cloned descriptor's shared OS cursor.
            // Restore the original reader before the existing push fallback.
            input.force_rewind()?;
            Ok(None)
        }
        Err(error) => Err(map_aac(error)),
    }
}

impl Pull {
    pub(super) fn recv(&mut self) -> Result<Option<AacBlock>> {
        if self.finished {
            return Ok(None);
        }
        let mut block: Option<AacBlock> = None;
        let mut yielded = false;
        let result = self.source.decode(|frame| {
            if let Some(block) = &block
                && (block.rate, block.channels) != (frame.sample_rate, frame.planar.len())
            {
                return Err(AacError::decode("AAC rate or channel count changed"));
            }
            let block = block.get_or_insert_with(|| AacBlock {
                frames: 0,
                samples: Vec::with_capacity(BATCH_FRAMES),
                rate: frame.sample_rate,
                channels: frame.planar.len(),
            });
            block.frames += frame.samples;
            append_plane(&mut block.samples, &frame, self.channel);
            if block.frames >= BATCH_FRAMES {
                yielded = true;
                return Err(AacError::decode("AAC logical batch boundary"));
            }
            Ok(())
        });
        match result {
            Err(_) if yielded => Ok(block),
            Err(error) => {
                self.finished = true;
                Err(map_aac(error))
            }
            Ok(_) => {
                self.finished = true;
                Ok(block)
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use std::io::{Read, Seek, SeekFrom, Write};

    fn fixture(rate: u32, channels: usize, he: bool) -> Vec<u8> {
        let n = rate as usize * 2 + 13;
        let planes: Vec<Vec<f32>> = (0..channels)
            .map(|ch| {
                (0..n)
                    .map(|i| 0.2 * (i as f32 * (0.07 + ch as f32 * 0.03)).sin())
                    .collect()
            })
            .collect();
        syom::encode_with(
            &planes,
            rate,
            &syom::EncodeOptions::m4a()
                .with_pns(true)
                .with_he(he)
                .with_bitrate_bps(48_000.min(rate * channels as u32 * 4)),
        )
        .unwrap()
    }

    fn collect(
        mut next: impl FnMut() -> Result<Option<AacBlock>>,
    ) -> Vec<(usize, u32, usize, Vec<u32>)> {
        let mut result = Vec::new();
        while let Some(block) = next().unwrap() {
            result.push((
                block.frames,
                block.rate,
                block.channels,
                block.samples.iter().map(|v| v.to_bits()).collect(),
            ));
        }
        result
    }

    fn compare(bytes: Vec<u8>) {
        let mut worker = super::super::spawn_worker(
            Reader::bytes(Bytes::from(bytes.clone())),
            ChannelSelect::Mono,
            None,
        )
        .unwrap();
        let expected = collect(|| worker.recv());
        let mut input = Reader::bytes(Bytes::from(bytes.clone()));
        let mut pull = try_open(&mut input, ChannelSelect::Mono, None)
            .unwrap()
            .unwrap();
        assert_eq!(collect(|| pull.recv()), expected);
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&bytes).unwrap();
        let mut input = Reader::File(std::fs::File::open(file.path()).unwrap());
        let mut pull = try_open(&mut input, ChannelSelect::Mono, None)
            .unwrap()
            .unwrap();
        assert_eq!(collect(|| pull.recv()), expected);
    }

    #[test]
    fn test_aac_m4a_pull_preserves_worker_batches_and_native_bits() {
        for rate in [8_000, 22_050, 48_000] {
            for channels in [1, 2] {
                compare(fixture(rate, channels, false));
            }
        }
        compare(fixture(48_000, 2, true));
        let mut partial = fixture(48_000, 2, false);
        let at = partial.windows(4).position(|w| w == b"elst").unwrap();
        let skip = u32::from_be_bytes(partial[at + 16..at + 20].try_into().unwrap());
        let play = u32::from_be_bytes(partial[at + 12..at + 16].try_into().unwrap());
        partial[at + 16..at + 20].copy_from_slice(&(skip + 17).to_be_bytes());
        partial[at + 12..at + 16].copy_from_slice(&(play - 17).to_be_bytes());
        compare(partial);
    }

    #[test]
    fn test_aac_transport_classification_preserves_public_raw_id3_file_bytes_batches() {
        let pcm: Vec<_> = (0..96_013).map(|i| (i as f32 * 0.03).sin() * 0.1).collect();
        let tagged = syom::encode(&[&pcm], 48_000).unwrap();
        let at = if tagged.starts_with(b"ID3") {
            10 + tagged[6..10]
                .iter()
                .fold(0usize, |n, b| (n << 7) | usize::from(*b))
        } else {
            0
        };
        for bytes in [tagged[at..].to_vec(), tagged] {
            let mut worker = super::super::spawn_worker(
                Reader::bytes(Bytes::from(bytes.clone())),
                ChannelSelect::Mono,
                None,
            )
            .unwrap();
            let expected = collect(|| worker.recv());
            let mut file = tempfile::NamedTempFile::new().unwrap();
            file.write_all(&bytes).unwrap();
            let sources = [
                super::super::try_open_bytes(Bytes::from(bytes), ChannelSelect::Mono, None)
                    .unwrap()
                    .unwrap(),
                super::super::try_open_path(
                    file.path().to_str().unwrap(),
                    ChannelSelect::Mono,
                    None,
                )
                .unwrap()
                .unwrap(),
            ];
            for mut source in sources {
                let actual = collect(|| {
                    source.recv_block().map(|b| match b {
                        super::super::AacRecv::Block(block) => Some(block),
                        super::super::AacRecv::Eof => None,
                    })
                });
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn test_aac_transport_classification_keeps_non_aac_mp4_fallthrough() {
        let mut bytes = fixture(48_000, 1, false);
        let at = bytes.windows(4).position(|w| w == b"mp4a").unwrap();
        bytes[at..at + 4].copy_from_slice(b"avc1");
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&bytes).unwrap();
        assert!(
            super::super::try_open_bytes(Bytes::from(bytes), ChannelSelect::Mono, None)
                .unwrap()
                .is_none()
        );
        assert!(
            super::super::try_open_path(file.path().to_str().unwrap(), ChannelSelect::Mono, None)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn test_aac_m4a_pull_refuses_real_completion_errors_and_budget_excess() {
        let mut bytes = fixture(48_000, 1, false);
        // Opening only prepares the bounded source. The public windowed
        // consumer enforces the source-frame budget before publishing output.
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&bytes).unwrap();
        for result in [
            crate::inference::audio::decode_audio_bytes_shared_bounded(
                Bytes::from(bytes.clone()),
                Some(0.1),
            ),
            crate::inference::audio::decode::decode_audio_file_bounded(
                file.path().to_str().unwrap(),
                Some(0.1),
            ),
        ] {
            let error = result.expect_err("a 100 ms budget cannot return two seconds of PCM");
            assert!(
                matches!(error.downcast_ref::<crate::error::GigasttError>(),
                Some(crate::error::GigasttError::AudioTooLong { observed_secs, limit_secs })
                    if *observed_secs > *limit_secs && *limit_secs <= 0.1),
                "{error:#}"
            );
        }
        let at = bytes.windows(4).position(|w| w == b"elst").unwrap();
        bytes[at + 12..at + 16].copy_from_slice(&480_000u32.to_be_bytes());
        let mut reader = Reader::bytes(Bytes::from(bytes));
        let mut pull = try_open(&mut reader, ChannelSelect::Mono, None)
            .unwrap()
            .unwrap();
        let mut got_error = false;
        for _ in 0..32 {
            match pull.recv() {
                Err(_) => {
                    got_error = true;
                    break;
                }
                Ok(None) => break,
                Ok(Some(_)) => {}
            }
        }
        assert!(
            got_error,
            "intentional batch pauses must not swallow real EOF failures"
        );
        assert!(
            pull.recv().unwrap().is_none(),
            "real errors terminate the source"
        );
    }

    #[test]
    fn test_aac_m4a_pull_fallback_rewinds_shared_file_cursor() {
        let bytes = fixture(48_000, 1, false);
        let mut reader = Reader::bytes(Bytes::from(bytes.clone()));
        let mut clone = reader.clone_for_pull().unwrap();
        clone.seek(SeekFrom::End(0)).unwrap();
        reader.force_rewind().unwrap();
        let mut head = [0; 8];
        reader.read_exact(&mut head).unwrap();
        assert_eq!(head, bytes[..8]);
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&bytes).unwrap();
        let mut reader = Reader::File(std::fs::File::open(file.path()).unwrap());
        let mut clone = reader.clone_for_pull().unwrap();
        clone.seek(SeekFrom::End(0)).unwrap();
        reader.force_rewind().unwrap();
        reader.read_exact(&mut head).unwrap();
        assert_eq!(head, bytes[..8]);
    }

    #[test]
    fn test_aac_m4a_pull_restores_on_small_stack_without_worker() {
        let bytes = fixture(48_000, 2, false);
        let stack = if cfg!(debug_assertions) {
            512 * 1024
        } else {
            128 * 1024
        };
        std::thread::Builder::new()
            .stack_size(stack)
            .spawn(move || {
                let mut reader = Reader::bytes(Bytes::from(bytes));
                let mut pull = try_open(&mut reader, ChannelSelect::Mono, None)
                    .unwrap()
                    .unwrap();
                let count: usize = collect(|| pull.recv()).into_iter().map(|row| row.0).sum();
                assert_eq!(count, 96_013);
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
