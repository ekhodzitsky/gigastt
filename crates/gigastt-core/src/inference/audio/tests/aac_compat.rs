//! Existing transports must work with and without extended-aac.
use super::*;
use crate::error::GigasttError;

#[test]
fn test_aac_matroska_remains_supported() {
    let data = include_bytes!("../../../../tests/fixtures/aac/speech.mka");
    let decoded = decode_audio_bytes(data).expect("AAC in Matroska must retain its decoder");
    assert!(decoded.len() > 3 * 16_000);
}

#[test]
fn test_mp3_in_mp4_remains_supported() {
    let data = include_bytes!("../../../../tests/fixtures/aac/mp3.mp4");
    let decoded = decode_audio_bytes(data).expect("MP4 demux must retain non-AAC audio");
    assert!(
        probe_duration_bytes(bytes::Bytes::from_static(data))
            .unwrap()
            .is_some()
    );
    assert!(decoded.len() > 3 * 16_000);
    let shared = bytes::Bytes::from_static(data);
    assert_eq!(
        scan_channels(shared.clone(), None)
            .expect("scan MP3 in MP4")
            .channels,
        1
    );
    assert_eq!(
        decode_audio_bytes_shared_channels(shared)
            .expect("split MP3 in MP4")
            .len(),
        1
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("speech.mp4");
    std::fs::write(&path, data).expect("write fixture");
    assert!(
        probe_duration_file(path.to_str().expect("utf8 path"))
            .expect("probe MP3 file")
            .is_some()
    );
}

#[test]
fn test_aac_split_checks_cancellation_between_frames() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let checks = AtomicUsize::new(0);
    let abort = || checks.fetch_add(1, Ordering::Relaxed) >= 3;
    let data =
        bytes::Bytes::from_static(include_bytes!("../../../../tests/fixtures/aac/speech.m4a"));
    let error = decode_audio_bytes_shared_channels_bounded_with_abort(data, None, Some(&abort))
        .map(|_| ())
        .expect_err(
            "cancellation must be observed during AAC decode, not only after the whole file",
        );
    assert!(matches!(decode_error(error), GigasttError::Cancelled));
    assert_eq!(checks.load(Ordering::Relaxed), 4);
}
