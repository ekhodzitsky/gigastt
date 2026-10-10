//! AAC via syom. Fixtures are encoded in the test: the roundtrip proves the
//! gigastt wiring (resample, budget, probe, split). Bitstream goldens live in syom.

use super::*;

use crate::error::GigasttError;

fn sine(n: usize, step: f32) -> Vec<f32> {
    (0..n).map(|i| 0.3 * (i as f32 * step).sin()).collect()
}

fn mean_square(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|s| f64::from(*s) * f64::from(*s)).sum();
    sum / samples.len() as f64
}

#[test]
#[cfg_attr(miri, ignore)]
fn test_adts_sine_survives_resample() {
    let pcm = sine(48_000, 0.05);
    let adts = syom::encode(&[&pcm], 48_000).expect("encode adts");
    let decoded = decode_audio_bytes(&adts).expect("decode adts");
    let expected = pcm.len() * 16_000 / 48_000;
    assert!(
        (expected / 2..expected + expected / 5).contains(&decoded.len()),
        "len {} vs about {expected}",
        decoded.len()
    );
    assert!(
        mean_square(&decoded) > 1e-3,
        "sine energy collapsed: {}",
        mean_square(&decoded)
    );
    // The encoder's iTunSMPB tag names the length. Raw ADTS does not.
    let tagged = probe_duration_bytes(bytes::Bytes::from(adts.clone()))
        .expect("probe tagged")
        .expect("iTunSMPB declares a duration");
    assert!((0.8..1.3).contains(&tagged), "tagged duration {tagged}");
    let raw = strip_id3(&adts);
    assert!(syom::sniff_aac(raw), "tag did not peel back to ADTS");
    assert!(
        probe_duration_bytes(bytes::Bytes::copy_from_slice(raw))
            .expect("probe raw")
            .is_none(),
        "untagged ADTS must not invent a duration"
    );
}

/// Drop one leading ID3v2.3/2.4 tag. The test fixture's ADTS starts there.
fn strip_id3(data: &[u8]) -> &[u8] {
    if data.len() < 10 || &data[..3] != b"ID3" || (data[3] != 3 && data[3] != 4) {
        return data;
    }
    if data[6..10].iter().any(|byte| byte & 0x80 != 0) {
        return data;
    }
    let body = (u32::from(data[6]) << 21)
        | (u32::from(data[7]) << 14)
        | (u32::from(data[8]) << 7)
        | u32::from(data[9]);
    let mut total = 10usize.saturating_add(body as usize);
    if data[5] & 0x10 != 0 {
        total = total.saturating_add(10);
    }
    data.get(total..).unwrap_or(data)
}

#[test]
#[cfg_attr(miri, ignore)]
fn test_m4a_file_matches_bytes_and_reports_duration() {
    let pcm = sine(48_000, 0.05);
    let m4a = syom::encode_with(&[&pcm], 48_000, &syom::EncodeOptions::m4a()).expect("encode m4a");
    let from_bytes = decode_audio_bytes(&m4a).expect("decode bytes");
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("tone.m4a");
    std::fs::write(&path, &m4a).expect("write m4a");
    let path = path.to_str().expect("utf8 path");
    let from_file = decode_audio_file(path).expect("decode file");
    assert_eq!(from_file, from_bytes);
    let secs = probe_duration_file(path)
        .expect("probe file")
        .expect("m4a declares a duration");
    assert!((0.8..1.3).contains(&secs), "duration {secs}");
    let from_buf = probe_duration_bytes(bytes::Bytes::from(m4a))
        .expect("probe bytes")
        .expect("m4a bytes declare a duration");
    assert!((secs - from_buf).abs() < 0.05, "{secs} vs {from_buf}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn test_stereo_split_keeps_distinct_channels() {
    let left = sine(48_000, 0.05);
    let right = sine(48_000, 0.13);
    let adts = syom::encode(&[&left, &right], 48_000).expect("encode stereo");
    let planes = decode_audio_bytes_shared_channels(bytes::Bytes::from(adts)).expect("split");
    assert_eq!(planes.len(), 2);
    assert_ne!(planes[0], planes[1]);
    assert!(mean_square(&planes[0]) > 1e-3);
    assert!(mean_square(&planes[1]) > 1e-3);
    let again = syom::encode(&[&left, &right], 48_000).expect("encode stereo again");
    let scan = scan_channels(bytes::Bytes::from(again), None).expect("scan");
    assert_eq!(scan.channels, 2);
    assert!(!scan.dual_mono);
}

#[test]
#[cfg_attr(miri, ignore)]
fn test_short_budget_is_audio_too_long() {
    let pcm = sine(48_000, 0.05);
    let adts = syom::encode(&[&pcm], 48_000).expect("encode adts");
    let err = decode_audio_bytes_shared_bounded(bytes::Bytes::from(adts), Some(0.05))
        .expect_err("50 ms cannot hold a second of audio");
    assert!(
        matches!(
            err.downcast_ref::<GigasttError>(),
            Some(GigasttError::AudioTooLong { .. })
        ),
        "{err:#}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn test_truncated_adts_errors() {
    let pcm = sine(48_000, 0.05);
    let adts = syom::encode(&[&pcm], 48_000).expect("encode adts");
    let cut = &adts[..adts.len().min(24)];
    assert!(decode_audio_bytes(cut).is_err());
}

#[test]
fn test_wav_and_id3_are_not_aac() {
    assert!(!syom::sniff_aac(b"ID3\x04\x00\x00\x00\x00\x00\x00"));
    let wav = make_wav_bytes(&[1000, -1000, 500, -500, 250, -250], 16_000);
    assert!(!syom::sniff_aac(&wav));
    let decoded = decode_audio_bytes(&wav).expect("wav still decodes");
    assert!(!decoded.is_empty());
}

#[test]
#[cfg_attr(miri, ignore)]
fn test_dropping_aac_windows_joins_the_decoder() {
    let pcm = sine(48_000, 0.05);
    let adts = syom::encode(&[&pcm], 48_000).expect("encode adts");
    let windows = crate::inference::audio::FileWindows::from_bytes(
        bytes::Bytes::from(adts),
        crate::inference::audio::WindowSpec::flat(),
        None,
    )
    .expect("open");
    drop(windows);
}
