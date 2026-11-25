use super::super::pcm_source::{ChannelSelect, PcmSource};
use super::*;
use bytes::Bytes;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn test_cancelled_scan_stops_before_probe() {
    let error =
        scan_channels_with_abort(Bytes::from_static(b"invalid audio"), None, Some(&|| true))
            .unwrap_err();
    assert!(matches!(
        decode_error(error),
        crate::error::GigasttError::Cancelled
    ));
}

#[test]
#[cfg_attr(miri, ignore = "Opus decoding is too slow under Miri")]
fn test_channel_scans_stop_at_cooperative_boundaries() {
    let left: Vec<_> = (0..48000).map(|i| (i as f32 * 0.1).sin()).collect();
    let right: Vec<_> = (0..48000).map(|i| (i as f32 * 0.2).sin()).collect();
    for bytes in [
        super::stream::stereo_wav(&left, &right, 48000),
        Bytes::from_static(include_bytes!(
            "../../../../tests/fixtures/opus/late_stereo.ogg"
        )),
    ] {
        for retain in [false, true] {
            let checks = AtomicUsize::new(0);
            let abort = || checks.fetch_add(1, Ordering::Relaxed) >= 12;
            let result = if retain {
                prepare_channels_for_vad_with_abort(bytes.clone(), None, Some(&abort))
                    .map(|p| p.scan)
            } else {
                scan_channels_with_abort(bytes.clone(), None, Some(&abort))
            };
            assert!(matches!(
                decode_error(result.unwrap_err()),
                crate::error::GigasttError::Cancelled
            ));
            assert_eq!(checks.load(Ordering::Relaxed), 13);
        }
    }
}

#[test]
fn test_flat_decode_cancels_between_packets_without_changing_samples() {
    let samples: Vec<_> = (0..48000).map(|i| (i as f32 * 0.1).sin()).collect();
    for data in [
        super::stream::stereo_wav(&samples, &samples, 48000),
        Bytes::from_static(include_bytes!(
            "../../../../tests/fixtures/opus/late_stereo.ogg"
        )),
    ] {
        let decode = || PcmSource::from_bytes(data.clone(), None, ChannelSelect::Mono).unwrap();
        let expected = decode().drain_to_vec(None).unwrap();
        assert_eq!(expected, decode().drain_to_vec(Some(&|| false)).unwrap());
        let checks = AtomicUsize::new(0);
        let abort = || checks.fetch_add(1, Ordering::Relaxed) >= 3;
        let error = decode().drain_to_vec(Some(&abort)).unwrap_err();
        assert!(matches!(
            decode_error(error),
            crate::error::GigasttError::Cancelled
        ));
        assert_eq!(checks.load(Ordering::Relaxed), 4);
    }
}
