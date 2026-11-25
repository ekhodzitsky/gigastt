use super::*;
use crate::error::GigasttError;
use crate::inference::audio::{decode_audio_bytes_shared, encode_wav_pcm16};
use bytes::Bytes;

fn signal(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * 0.03).sin() * 0.5).collect()
}

/// Consume small, non-frame-aligned requests, even when the decoder returns
/// substantially more than asked. EOF must include the final resampler flush.
fn collect(mut src: PcmSource) -> Vec<f32> {
    let mut out = Vec::new();
    loop {
        src.fill_to(out.len() + 137, None).unwrap();
        let (start, pcm) = src.buffer();
        assert_eq!(start, out.len());
        assert_eq!(start + pcm.len(), src.position());
        let take = pcm.len().min(137);
        out.extend_from_slice(&pcm[..take]);
        src.retain_from(out.len());
        if src.is_eof() && src.position() == out.len() {
            break;
        }
    }
    src.fill_to(usize::MAX, None).unwrap();
    assert!(src.is_eof());
    assert_eq!(src.buffer(), (out.len(), &[][..]));
    out
}

#[test]
fn test_incremental_pcm_short_and_final_blocks() {
    let lengths: &[usize] = if cfg!(miri) {
        &[1, 136, 137, 138]
    } else {
        &[1, 136, 137, 138, 40_003]
    };
    for &n in lengths {
        let data = Bytes::from(encode_wav_pcm16(&signal(n), 16_000));
        let expected = decode_audio_bytes_shared(data.clone()).unwrap();
        assert_eq!(expected.len(), n);
        assert_eq!(
            collect(PcmSource::from_bytes(data, None, ChannelSelect::Mono).unwrap()),
            expected
        );
    }
}

#[test]
#[cfg_attr(miri, ignore = "file I/O and resampling are tested natively")]
fn test_path_matches_bytes_through_resampler() {
    let data = Bytes::from(encode_wav_pcm16(&signal(48_005), 48_000));
    let expected = decode_audio_bytes_shared(data.clone()).unwrap();
    assert!(expected.len().abs_diff(16_002) <= 1);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("audio.wav");
    std::fs::write(&path, &data).unwrap();
    let src = PcmSource::open(path.to_str().unwrap(), None, ChannelSelect::Mono).unwrap();
    assert_eq!(collect(src), expected);
    assert_eq!(
        collect(PcmSource::from_bytes(data, None, ChannelSelect::Mono).unwrap()),
        expected
    );
}

#[test]
#[cfg_attr(miri, ignore = "large WAV exercises multiple decoder blocks natively")]
fn test_fill_exposes_cancellation_at_boundary() {
    let n = 400_000;
    let data = Bytes::from(encode_wav_pcm16(&signal(n), 16_000));
    let mut src = PcmSource::from_bytes(data, None, ChannelSelect::Mono).unwrap();
    src.fill_to(137, None).unwrap();
    let before = src.position();
    assert!(before >= 137 && before < n);
    assert!(!src.is_eof());
    let error = src.fill_to(usize::MAX, Some(&|| true)).unwrap_err();
    assert!(matches!(
        error.downcast_ref::<GigasttError>(),
        Some(GigasttError::Cancelled)
    ));
    assert_eq!(src.position(), before);
}

#[test]
#[cfg_attr(miri, ignore = "Opus resampling is tested natively")]
fn test_source_frame_limit_uses_opus_decode_rate() {
    // Same Opus packets, different WebM SamplingFrequency. The actual PCM
    // rate must determine the limit in either case.
    for data in [
        include_bytes!("../../../../tests/fixtures/opus/opus_rate48.webm").as_slice(),
        include_bytes!("../../../../tests/fixtures/opus/opus_sf16.webm").as_slice(),
    ] {
        let limit_secs = 0.05;
        let mut src = PcmSource::from_bytes(
            Bytes::from_static(data),
            Some(limit_secs),
            ChannelSelect::Mono,
        )
        .unwrap();
        let error = src.fill_to(usize::MAX, None).unwrap_err();
        match error.downcast_ref::<GigasttError>() {
            Some(GigasttError::AudioTooLong {
                observed_secs,
                limit_secs: actual_limit,
            }) => {
                assert_eq!(*actual_limit, limit_secs);
                assert_eq!(*observed_secs, 0.06);
            }
            other => panic!("expected typed duration error, got {other:?}"),
        }
    }
}
