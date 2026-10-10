//! Duration probe. WAVE and the symphonia containers are header-only.
//! M4A reads `moov` plus one frame so the presentation rate (HE-AAC is
//! twice the core rate) is known, then drops that frame.

use std::io::{Seek as _, SeekFrom};

use anyhow::{Context, Result};
use bytes::Bytes;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use super::super::MAX_SAMPLE_RATE;
use super::super::{aac, wave};
use super::BytesMediaSource;

/// Read an audio container's declared duration (seconds) from its header
/// WITHOUT decoding a single audio packet.
///
/// WAVE files go through ryf's header probe, AAC through syom, and other
/// containers use symphonia's format probe. Each is an O(header) read, not the O(T)
/// decode `decode_audio_bytes_shared` performs. Returns:
/// - `Ok(Some(secs))` when the container declares a positive frame count and a
///   plausible sample rate (WAV, FLAC, M4A, and OGG usually do);
/// - `Ok(None)` when the stream is probeable but declares no usable duration
///   (a raw MP3 stream typically does not, and neither does a crafted header);
/// - `Err(_)` when the bytes cannot be probed as a supported container at all.
///
/// Use this to size a long job's progress bar without paying for a full decode
/// whose samples are immediately discarded; fall back to a real decode on
/// anything other than `Ok(Some(_))`.
///
/// ```text
/// { true }
/// fn probe_duration_bytes(data: Bytes) -> Result<Option<f64>>
/// { true }
/// ```
pub fn probe_duration_bytes(data: Bytes) -> Result<Option<f64>> {
    if ryf::sniff_wav(data.as_ref()) {
        return wave::probe_duration(data.as_ref());
    }
    if let aac::ProbeHit::Duration(duration) = aac::probe_bytes(data.as_ref())? {
        return Ok(duration);
    }
    let source = BytesMediaSource::new(data);
    let mss = MediaSourceStream::new(Box::new(source), Default::default());
    probe_duration_inner(mss, Hint::new())
}

/// Path twin of [`probe_duration_bytes`]: read a file's declared duration from
/// its header without decoding. Seeds the probe hint from the extension exactly
/// as [`decode_audio_file`](super::decode_audio_file) does.
pub fn probe_duration_file(path: &str) -> Result<Option<f64>> {
    let mut file =
        std::fs::File::open(path).with_context(|| format!("Failed to open audio file: {path}"))?;
    let mut prefix = [0u8; 40];
    let n = std::io::Read::read(&mut file, &mut prefix)
        .with_context(|| format!("Failed to read audio file: {path}"))?;
    if ryf::sniff_wav(&prefix[..n]) {
        file.seek(SeekFrom::Start(0))
            .with_context(|| format!("Failed to read audio file: {path}"))?;
        return wave::probe_duration_file(file);
    }
    if aac::sniff_file(&mut file).with_context(|| format!("Failed to read audio file: {path}"))? {
        match aac::probe_file(file)? {
            aac::ProbeHit::Duration(duration) => return Ok(duration),
            aac::ProbeHit::FallThrough => {
                file = std::fs::File::open(path)
                    .with_context(|| format!("Failed to open audio file: {path}"))?;
            }
        }
    }
    file.seek(SeekFrom::Start(0))
        .with_context(|| format!("Failed to read audio file: {path}"))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
    {
        hint.with_extension(ext);
    }
    probe_duration_inner(mss, hint)
}

/// Shared probe: identify the container and read the default audio track's
/// declared frame count / sample rate, deriving duration = frames / rate. No
/// packet is ever decoded, so a container that does not declare its length
/// yields `Ok(None)` rather than a scanned duration.
fn probe_duration_inner(mss: MediaSourceStream<'_>, hint: Hint) -> Result<Option<f64>> {
    let format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .context("Unsupported audio format")?;

    let Some(track) = format.default_track(TrackType::Audio) else {
        return Ok(None);
    };
    let Some(audio_params) = track.codec_params.as_ref().and_then(|p| p.audio()) else {
        return Ok(None);
    };
    let Some(sample_rate) = audio_params.sample_rate else {
        return Ok(None);
    };
    // A crafted / implausible header rate can't be trusted to scale a duration:
    // report "unknown" and let the caller fall back to a real decode, which
    // clamps the rate and enforces the length budget incrementally.
    if sample_rate == 0 || sample_rate > MAX_SAMPLE_RATE {
        return Ok(None);
    }
    match track.num_frames {
        Some(n) if n > 0 => Ok(Some(n as f64 / sample_rate as f64)),
        _ => Ok(None),
    }
}
