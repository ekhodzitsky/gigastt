use super::*;

// --- Opus (OGG container, pure-Rust opus-rs fallback decoder) ---

/// The whole-buffer Opus pipeline as it stood before the streaming source:
/// decode every channel of the whole file, mix to mono, then feed the resampler
/// in exact `RESAMPLE_STAGING_FRAMES` chunks. Kept here verbatim as the oracle
/// the streaming decode must reproduce sample for sample.
#[cfg(feature = "file-decode")]
fn eager_opus_reference(bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::formats::{FormatOptions, TrackType};
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;

    let source = BytesMediaSource::new(bytes::Bytes::copy_from_slice(bytes));
    let mss = MediaSourceStream::new(Box::new(source), Default::default());
    let mut format = symphonia::default::get_probe().probe(
        &Hint::new(),
        mss,
        FormatOptions::default(),
        MetadataOptions::default(),
    )?;
    let (track_id, channels) = {
        let track = format
            .default_track(TrackType::Audio)
            .ok_or_else(|| anyhow::anyhow!("no audio track"))?;
        let p = track
            .codec_params
            .as_ref()
            .and_then(|p| p.audio())
            .ok_or_else(|| anyhow::anyhow!("no audio params"))?;
        (
            track.id,
            p.channels.as_ref().map(|c| c.count()).unwrap_or(1),
        )
    };
    let mono = mix_channels_to_mono(&crate::inference::audio::opus::decode_opus_channels(
        &mut *format,
        track_id,
        channels,
        usize::MAX,
        f64::INFINITY,
    )?);
    // The decoder emits 48 kHz even when the container tag says otherwise.
    // The oracle has to resample from that rate, matching the production path.
    let mut resampler = crate::inference::audio::resample::ResampleTo16k::new(
        SampleRate(crate::inference::audio::opus::OPUS_DECODE_RATE),
        None,
    );
    for piece in mono.chunks(crate::inference::audio::resample::RESAMPLE_STAGING_FRAMES) {
        resampler.stage().extend_from_slice(piece);
        resampler.flush_full()?;
    }
    let mut out = Vec::new();
    resampler.finish_into(&mut out)?;
    Ok(out)
}

#[test]
fn test_opus_streaming_decode_matches_whole_buffer() {
    // Opus used to be the one container that could not stream: the fallback
    // decoder accumulated every channel of the whole file before anything was
    // mixed or resampled, which is what kept it on the duration ceiling. It
    // decodes packet-by-packet now — and must yield the same samples, not
    // merely close ones.
    for (name, bytes) in [
        (
            "opus_tone.ogg",
            &include_bytes!("../../../../tests/fixtures/opus/opus_tone.ogg")[..],
        ),
        (
            "opus_tone_no_eos.ogg",
            &include_bytes!("../../../../tests/fixtures/opus/opus_tone_no_eos.ogg")[..],
        ),
        // Multi-frame packets: the per-packet decode now splits three frames
        // where the other fixtures carry one, so both paths must still agree.
        (
            "opus_tone_60ms.ogg",
            &include_bytes!("../../../../tests/fixtures/opus/opus_tone_60ms.ogg")[..],
        ),
    ] {
        let streamed = decode_audio_bytes(bytes).expect("streaming decode");
        let eager = eager_opus_reference(bytes).expect("whole-buffer decode");
        assert!(!streamed.is_empty(), "{name} decoded to nothing");
        assert_eq!(streamed, eager, "{name}: streaming decode diverged");
    }
}

#[test]
fn test_opus_streaming_windows_match_slice_over_flat_decode() {
    // The windowed source over an Opus stream must yield exactly what
    // `SliceWindows` yields over the same flat decode — the same guarantee the
    // symphonia-backed source carries.
    let bytes = bytes::Bytes::from_static(include_bytes!(
        "../../../../tests/fixtures/opus/opus_tone.ogg"
    ));
    // A window shorter than the clip so the overlapping geometry is exercised.
    let spec = WindowSpec::new(16_000, 16_000, 3_200);
    let flat = FileWindows::from_bytes(bytes.clone(), WindowSpec::flat(), None)
        .expect("open flat")
        .drain_to_vec()
        .expect("drain");
    let mut src = FileWindows::from_bytes(bytes, spec, None).expect("open windows");
    let mut got = Vec::new();
    while let Some(w) = src.next_window().expect("window") {
        got.push((w.start_sample, w.samples.to_vec()));
    }
    let mut want = Vec::new();
    let mut sw = SliceWindows::new(&flat, spec);
    while let Some(w) = sw.next_window().expect("slice window") {
        want.push((w.start_sample, w.samples.to_vec()));
    }
    assert!(
        got.len() > 1,
        "expected the windowed regime, got {}",
        got.len()
    );
    assert_eq!(got, want);
    assert_eq!(src.total_16k_samples(), flat.len());
}

#[test]
fn test_push_mono_mix_matches_mix_channels_to_mono() {
    // The streaming decode mixes per packet; the whole-buffer one mixed the
    // finished per-channel buffers. Same arithmetic, pinned.
    for channels in [1usize, 2] {
        let frames = 97;
        let pcm: Vec<f32> = (0..frames * channels)
            .map(|i| ((i as f32) * 0.37).sin() * 0.8 - 0.13)
            .collect();
        let mut got = Vec::new();
        crate::inference::audio::opus::push_mono_mix(&pcm, channels, frames, &mut got);

        let per_channel: Vec<Vec<f32>> = (0..channels)
            .map(|c| (0..frames).map(|f| pcm[f * channels + c]).collect())
            .collect();
        assert_eq!(
            got,
            mix_channels_to_mono(&per_channel),
            "channels={channels}"
        );
    }
}

#[test]
fn test_is_recoverable_packet_eof_matches_unexpected_eof_only() {
    use std::io::{Error as IoError, ErrorKind};
    use symphonia::core::errors::Error as SymError;

    let eof = SymError::IoError(IoError::new(
        ErrorKind::UnexpectedEof,
        "unexpected end of file",
    ));
    assert!(is_recoverable_packet_eof(&eof));

    let other_io = SymError::IoError(IoError::other("disk full"));
    assert!(!is_recoverable_packet_eof(&other_io));

    let decode = SymError::DecodeError("bad page");
    assert!(!is_recoverable_packet_eof(&decode));

    let unsupported = SymError::Unsupported("codec");
    assert!(!is_recoverable_packet_eof(&unsupported));
}

#[test]
fn test_decode_audio_bytes_opus_ogg_missing_eos_succeeds() {
    // Telegram Android (and some MediaRecorder paths) write Ogg/Opus without
    // the EOS flag on the final page. Symphonia then ends the demux with
    // UnexpectedEof instead of Ok(None). Soft-EOF must still return audio
    // (issue #217). Fixture is opus_tone.ogg with the EOS bit cleared and
    // the page CRC recomputed.
    let no_eos = include_bytes!("../../../../tests/fixtures/opus/opus_tone_no_eos.ogg");
    let with_eos = include_bytes!("../../../../tests/fixtures/opus/opus_tone.ogg");
    let decoded_no_eos = decode_audio_bytes(no_eos).expect("OGG/Opus without EOS must decode");
    let decoded_with_eos =
        decode_audio_bytes(with_eos).expect("OGG/Opus with EOS must still decode");
    assert!(
        !decoded_no_eos.is_empty(),
        "missing-EOS stream must yield non-empty PCM"
    );
    // Same content; length must match the EOS sibling within a sample or two.
    let delta = (decoded_no_eos.len() as i64 - decoded_with_eos.len() as i64).unsigned_abs();
    assert!(
        delta <= 2,
        "no-EOS length {} diverged from with-EOS length {}",
        decoded_no_eos.len(),
        decoded_with_eos.len()
    );
    // Spot-check a stretch of samples (pre-skip / tone body) for identity.
    let start = decoded_no_eos.len().min(decoded_with_eos.len()) / 4;
    let end = start + 1000;
    for (a, b) in decoded_no_eos[start..end]
        .iter()
        .zip(decoded_with_eos[start..end].iter())
    {
        assert!((a - b).abs() < f32::EPSILON);
    }
}

#[test]
fn test_decode_audio_file_opus_missing_eos_matches_bytes() {
    let no_eos = include_bytes!("../../../../tests/fixtures/opus/opus_tone_no_eos.ogg");
    let mut tmp = tempfile::NamedTempFile::with_suffix(".ogg").expect("temp file");
    std::io::Write::write_all(&mut tmp, no_eos).expect("write temp file");
    let via_file = decode_audio_file(tmp.path().to_str().expect("utf-8 path"))
        .expect("missing-EOS OGG/Opus file must decode");
    let via_bytes = decode_audio_bytes(no_eos).expect("missing-EOS bytes must decode");
    assert_eq!(via_file.len(), via_bytes.len());
    for (a, b) in via_file.iter().zip(via_bytes.iter()) {
        assert!((a - b).abs() < f32::EPSILON);
    }
}

#[test]
fn test_decode_audio_bytes_truncated_opus_headers_only_still_errors() {
    // Truncate after OpusHead+OpusTags pages so demux may open the track but
    // no audio packets arrive. Soft-EOF must NOT turn this into silence.
    let full = include_bytes!("../../../../tests/fixtures/opus/opus_tone.ogg");
    // First two Ogg pages only (~header); keep under 200 bytes of safety.
    // Find end of page 1 (seq 1) more carefully: second page ends before first audio.
    let mut pages = Vec::new();
    let mut i = 0usize;
    let data = full;
    while i + 27 <= data.len() {
        if &data[i..i + 4] != b"OggS" {
            break;
        }
        let nseg = data[i + 26] as usize;
        let body: usize = data[i + 27..i + 27 + nseg]
            .iter()
            .map(|&s| s as usize)
            .sum();
        let page_end = i + 27 + nseg + body;
        pages.push(page_end);
        i = page_end;
        if pages.len() == 2 {
            break;
        }
    }
    assert!(pages.len() >= 2, "fixture must have header pages");
    let headers_only = &data[..pages[1]];
    let err = decode_audio_bytes(headers_only).expect_err("headers-only Opus must fail");
    let msg = format!("{err:#}");
    // Must not succeed with empty/near-empty PCM via soft-EOF.
    assert!(
        msg.contains("packet")
            || msg.contains("end of file")
            || msg.contains("audio")
            || msg.contains("Decode")
            || msg.contains("Unsupported")
            || msg.contains("malformed")
            || msg.contains("Opus")
            || msg.contains("track")
            || msg.contains("empty")
            || msg.contains("No "),
        "unexpected error for headers-only: {msg}"
    );
}

#[test]
fn test_decode_audio_bytes_random_bytes_still_errors() {
    let junk = [0u8; 64];
    assert!(decode_audio_bytes(&junk).is_err());
}

#[test]
fn test_decode_audio_bytes_opus_ogg_matches_ffmpeg_reference() {
    // Independent-reference verification: `opus_tone.ogg` was ENCODED by
    // ffmpeg (libopus) and `opus_tone_ffmpeg.pcm` is ffmpeg's own DECODE
    // of it resampled to 16 kHz mono (see
    // scripts/generate_opus_fixtures.sh). Our opus-rs decode is compared
    // against libopus, so the pure-Rust port is validated against a
    // second implementation rather than against itself. We do not trim
    // the OpusHead pre-skip (ffmpeg does), so the comparison runs at the
    // best lag. Tolerance: RMSE below 2% of full scale.
    let ogg = include_bytes!("../../../../tests/fixtures/opus/opus_tone.ogg");
    let reference_pcm = include_bytes!("../../../../tests/fixtures/opus/opus_tone_ffmpeg.pcm");
    let ours = decode_audio_bytes(ogg).expect("OGG/Opus must decode");
    let reference: Vec<f32> = reference_pcm
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from(i16::from_le_bytes(*c)) / 32768.0)
        .collect();
    // 3 s of tone at 16 kHz; the untrimmed pre-skip on our side and the
    // resampler's FIR delay shift the exact count by a few hundred.
    assert!(
        ours.len() > 46_000 && ours.len() < 50_000,
        "unexpected decoded length {}",
        ours.len()
    );
    let rmse = best_lag_rmse(&ours, &reference, 1024);
    assert!(
        rmse < 0.02,
        "Opus decode diverged from ffmpeg reference: RMSE {rmse}"
    );
}

#[test]
fn test_decode_audio_bytes_opus_code3_multiframe_matches_ffmpeg_reference() {
    // `opus_tone_60ms.ogg` carries 60 ms packets — three 20 ms CELT frames per
    // packet, code 3 CBR — which is what Chromium's MediaRecorder emits and no
    // other fixture exercises (`opus_tone.ogg` is code 0 throughout). Verified
    // against ffmpeg's own decode of the same file, so a packet split that
    // silently mis-slices the frames fails here rather than merely not erroring.
    let ogg = include_bytes!("../../../../tests/fixtures/opus/opus_tone_60ms.ogg");
    let reference_pcm = include_bytes!("../../../../tests/fixtures/opus/opus_tone_60ms_ffmpeg.pcm");
    let ours = decode_audio_bytes(ogg).expect("multi-frame OGG/Opus must decode");
    let reference: Vec<f32> = reference_pcm
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from(i16::from_le_bytes(*c)) / 32768.0)
        .collect();
    assert!(
        ours.len() > 46_000 && ours.len() < 50_000,
        "unexpected decoded length {}",
        ours.len()
    );
    let rmse = best_lag_rmse(&ours, &reference, 1024);
    assert!(
        rmse < 0.02,
        "multi-frame Opus decode diverged from ffmpeg reference: RMSE {rmse}"
    );
}

#[test]
fn test_decode_audio_bytes_webm_opus_live_matches_ffmpeg_reference() {
    // A browser's MediaRecorder writes a *live* WebM: the Segment and every
    // Cluster carry an unknown size, because the length is not known while
    // recording. Nothing ffmpeg writes has that shape, so the fixture is built
    // by rewriting each Cluster's size to the unknown-size vint (see
    // scripts/generate_opus_fixtures.sh) — the byte layout a browser produces.
    // Verified against ffmpeg's own decode of the same file.
    let webm = include_bytes!("../../../../tests/fixtures/opus/opus_tone_webm_live.webm");
    let reference_pcm =
        include_bytes!("../../../../tests/fixtures/opus/opus_tone_webm_live_ffmpeg.pcm");
    let ours = decode_audio_bytes(webm).expect("live WebM/Opus must decode");
    let reference: Vec<f32> = reference_pcm
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from(i16::from_le_bytes(*c)) / 32768.0)
        .collect();
    assert!(
        ours.len() > 46_000 && ours.len() < 50_000,
        "unexpected decoded length {}",
        ours.len()
    );
    let rmse = best_lag_rmse(&ours, &reference, 1024);
    assert!(
        rmse < 0.02,
        "WebM/Opus decode diverged from ffmpeg reference: RMSE {rmse}"
    );
}

#[test]
fn test_decode_audio_file_webm_extension_matches_bytes() {
    // Uploads arrive as bytes and are sniffed by content; the CLI goes through
    // the path-based probe with a `.webm` hint. Both must reach the same
    // Matroska demuxer and produce the same samples.
    let webm = include_bytes!("../../../../tests/fixtures/opus/opus_tone_webm_live.webm");
    let mut tmp = tempfile::NamedTempFile::with_suffix(".webm").expect("temp file");
    std::io::Write::write_all(&mut tmp, webm).expect("write temp file");
    let via_file =
        decode_audio_file(tmp.path().to_str().expect("utf-8 path")).expect("WebM file must decode");
    let via_bytes = decode_audio_bytes(webm).expect("WebM bytes must decode");
    assert_eq!(via_file.len(), via_bytes.len());
    for (a, b) in via_file.iter().zip(via_bytes.iter()) {
        assert!((a - b).abs() < f32::EPSILON);
    }
}

#[test]
fn test_opus_container_rate_tag_does_not_change_decoded_pcm() {
    // libopus always emits 48 kHz PCM (RFC 7845). These four files are one
    // encode of tone_src.wav; the 16 kHz copies only rewrite OpusHead's input
    // sample rate or the WebM SamplingFrequency (see
    // scripts/generate_opus_fixtures.sh). A resampler built from the container
    // tag treats that 48 kHz PCM as 16 kHz and returns ~3× as many samples,
    // which is what a browser MediaRecorder WebM does. 3 s at 16 kHz.
    let cases: &[(&str, &[u8])] = &[
        (
            "ogg input 48 kHz",
            include_bytes!("../../../../tests/fixtures/opus/opus_rate48.ogg"),
        ),
        (
            "ogg input 16 kHz",
            include_bytes!("../../../../tests/fixtures/opus/opus_head16.ogg"),
        ),
        (
            "webm sampling 48 kHz",
            include_bytes!("../../../../tests/fixtures/opus/opus_rate48.webm"),
        ),
        (
            "webm sampling 16 kHz",
            include_bytes!("../../../../tests/fixtures/opus/opus_sf16.webm"),
        ),
    ];
    let mut flat = Vec::with_capacity(cases.len());
    for (name, bytes) in cases {
        let pcm = decode_audio_bytes(bytes).unwrap_or_else(|e| panic!("{name} flat decode: {e:#}"));
        assert!(
            (46_000..50_000).contains(&pcm.len()),
            "{name}: decoded {} samples, expected ~48_000 (3 s at 16 kHz)",
            pcm.len()
        );
        // Lazy windowed path (FileWindows), not only the flat drain REST uses.
        let spec = WindowSpec::new(8_000, 8_000, 1_600);
        let mut windows = FileWindows::from_bytes(bytes::Bytes::copy_from_slice(bytes), spec, None)
            .unwrap_or_else(|e| panic!("{name} windows: {e:#}"));
        let mut n_windows = 0usize;
        while windows
            .next_window()
            .unwrap_or_else(|e| panic!("{name} next window: {e}"))
            .is_some()
        {
            n_windows += 1;
        }
        assert!(n_windows > 1, "{name}: expected more than one window");
        assert_eq!(
            windows.total_16k_samples(),
            pcm.len(),
            "{name}: windowed decode length diverged from flat"
        );
        // channels=split holds the whole Opus buffer, then resamples.
        let split = decode_audio_bytes_shared_channels(bytes::Bytes::copy_from_slice(bytes))
            .unwrap_or_else(|e| panic!("{name} split: {e:#}"));
        assert_eq!(split.len(), 1, "{name}: expected mono");
        assert_eq!(
            split[0].len(),
            pcm.len(),
            "{name}: split-channel length diverged from flat"
        );
        flat.push(pcm);
    }
    // Identical packets: a rate tag must not move the samples.
    assert_eq!(flat[0], flat[1], "OpusHead input rate changed the PCM");
    assert_eq!(flat[2], flat[3], "WebM SamplingFrequency changed the PCM");
}

#[test]
fn test_decode_audio_file_opus_extension_matches_bytes() {
    // The file path probes with an `.opus` extension hint; the bytes path
    // sniffs content only. Both must decode the same OGG/Opus stream
    // identically (the CLI transcribes via `decode_audio_file`).
    let ogg = include_bytes!("../../../../tests/fixtures/opus/opus_tone.ogg");
    let mut tmp = tempfile::NamedTempFile::with_suffix(".opus").expect("temp file");
    std::io::Write::write_all(&mut tmp, ogg).expect("write temp file");
    let via_file = decode_audio_file(tmp.path().to_str().expect("utf-8 path"))
        .expect("OGG/Opus file must decode");
    let via_bytes = decode_audio_bytes(ogg).expect("OGG/Opus bytes must decode");
    assert_eq!(via_file.len(), via_bytes.len());
    for (a, b) in via_file.iter().zip(via_bytes.iter()) {
        assert!((a - b).abs() < f32::EPSILON);
    }
}
