//! AAC metadata and transport detection.
use super::{MAX_SAMPLE_RATE, map_aac, stream_options};
use anyhow::{Context, Result};
use std::io::{Read, Seek, SeekFrom};
use syom::{AacError, M4aSeek};
pub(crate) enum ProbeHit {
    FallThrough,
    Duration(Option<f64>),
}
/// Header duration for a buffer already in memory. Does not decode PCM.
pub(crate) fn probe_bytes(data: &[u8]) -> Result<ProbeHit> {
    if !sniff_slice(data) {
        return Ok(ProbeHit::FallThrough);
    }
    match syom::probe(data) {
        Ok(info) => Ok(ProbeHit::Duration(duration_of(
            info.meta.output_rate,
            info.duration,
        ))),
        Err(AacError::NotAac) => Ok(ProbeHit::FallThrough),
        Err(AacError::Format(_)) if syom::sniff_is_isobmff(data) => Ok(ProbeHit::FallThrough),
        Err(AacError::UnsupportedSampleRate { .. }) => Ok(ProbeHit::Duration(None)),
        Err(err) => Err(map_aac(err)).context("Unsupported audio format"),
    }
}

/// Duration for a file positioned at offset 0. M4A reads `moov` plus one
/// frame (for the presentation rate) and stops. ADTS and LATM have no
/// reliable header duration and return `Duration(None)` without a full read.
pub(crate) fn probe_file(mut file: std::fs::File) -> Result<ProbeHit> {
    if !sniff_file(&mut file).context("Failed to read audio file")? {
        return Ok(ProbeHit::FallThrough);
    }
    let mut prefix = [0u8; 64];
    let n = std::io::Read::read(&mut file, &mut prefix).context("Failed to read audio file")?;
    file.seek(SeekFrom::Start(0))
        .context("Failed to read audio file")?;
    if !syom::sniff_is_isobmff(&prefix[..n]) {
        return Ok(ProbeHit::Duration(None));
    }
    let opts = stream_options(None);
    let mut seek = match M4aSeek::open(file, opts) {
        Ok(seek) => seek,
        Err(AacError::NotAac) => return Ok(ProbeHit::FallThrough),
        Err(AacError::UnsupportedSampleRate { .. } | AacError::Unsupported(_)) => {
            return Ok(ProbeHit::Duration(None));
        }
        Err(err) => return Err(map_aac(err)).context("Unsupported audio format"),
    };
    let samples = seek.presentation_len();
    let mut rate = None;
    let decoded = seek.decode(|frame| {
        rate = Some(frame.sample_rate);
        Err(AacError::decode("probe"))
    });
    if let Err(err) = decoded {
        let aborted = matches!(err, AacError::Decode(ref msg) if msg == "probe");
        if !aborted || rate.is_none() {
            return match err {
                AacError::NotAac => Ok(ProbeHit::FallThrough),
                AacError::Decode(ref msg) if msg == "probe" => Ok(ProbeHit::Duration(None)),
                other => Err(map_aac(other)).context("Unsupported audio format"),
            };
        }
    }
    let Some(rate) = rate else {
        return Ok(ProbeHit::Duration(None));
    };
    if rate == 0 || rate > MAX_SAMPLE_RATE || samples == 0 {
        return Ok(ProbeHit::Duration(None));
    }
    Ok(ProbeHit::Duration(Some(samples as f64 / f64::from(rate))))
}

fn duration_of(rate: u32, duration: syom::ProbeDuration) -> Option<f64> {
    if rate == 0 || rate > MAX_SAMPLE_RATE {
        return None;
    }
    let samples = match duration {
        syom::ProbeDuration::Exact { samples } | syom::ProbeDuration::Estimated { samples }
            if samples > 0 =>
        {
            samples
        }
        _ => return None,
    };
    Some(samples as f64 / f64::from(rate))
}

/// AAC, including one leading ID3v2.3/2.4 tag in front of ADTS, LATM, or M4A.
///
/// syom's ADTS encoder writes an `iTunSMPB` tag, and [`syom::sniff_aac`] does
/// not look past it. The tag stays in the buffer: syom peels it while decoding
/// so the priming edit is not lost.
pub(super) fn sniff_slice(data: &[u8]) -> bool {
    if syom::sniff_aac(data) {
        return true;
    }
    match id3v2_payload_at(data) {
        Some(at) if at < data.len() => syom::sniff_aac(&data[at..]),
        _ => false,
    }
}

/// File twin of [`sniff_slice`]. Rewinds to offset 0 either way. A tag that
/// runs past the first 64 bytes is seeked to, not read in full.
pub(crate) fn sniff_file(file: &mut std::fs::File) -> std::io::Result<bool> {
    file.seek(SeekFrom::Start(0))?;
    let mut prefix = [0u8; 64];
    let n = file.read(&mut prefix)?;
    if syom::sniff_aac(&prefix[..n]) {
        file.seek(SeekFrom::Start(0))?;
        return Ok(true);
    }
    let Some(at) = id3v2_payload_at(&prefix[..n]) else {
        file.seek(SeekFrom::Start(0))?;
        return Ok(false);
    };
    let hit = if at < n {
        syom::sniff_aac(&prefix[at..n])
    } else {
        file.seek(SeekFrom::Start(at as u64))?;
        let mut audio = [0u8; 16];
        let m = file.read(&mut audio)?;
        syom::sniff_aac(&audio[..m])
    };
    file.seek(SeekFrom::Start(0))?;
    Ok(hit)
}

/// Offset of the bytes after one ID3v2.3/2.4 header. `None` when those
/// bytes are not that tag. A declared size above 1 GiB is refused.
fn id3v2_payload_at(data: &[u8]) -> Option<usize> {
    if data.len() < 10 || &data[..3] != b"ID3" {
        return None;
    }
    if data[3] != 3 && data[3] != 4 {
        return None;
    }
    if data[6..10].iter().any(|byte| byte & 0x80 != 0) {
        return None;
    }
    let body = (u32::from(data[6]) << 21)
        | (u32::from(data[7]) << 14)
        | (u32::from(data[8]) << 7)
        | u32::from(data[9]);
    let mut total = 10usize.saturating_add(body as usize);
    if data[5] & 0x10 != 0 {
        total = total.saturating_add(10);
    }
    if total > (1 << 30) {
        return None;
    }
    Some(total)
}
