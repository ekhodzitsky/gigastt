//! Ogg/Opus packet decode and soft-EOF demux helpers.

#[cfg(feature = "file-decode")]
use anyhow::Result;
#[cfg(feature = "file-decode")]
use symphonia::core::formats::FormatReader;

#[cfg(feature = "file-decode")]
use super::audio_too_long_err;

/// Maximum decoded samples per channel for one Opus packet: 120 ms at 48 kHz
/// (RFC 6716 §3.2.5). A packet claiming more is malformed.
#[cfg(feature = "file-decode")]
const OPUS_MAX_PACKET_SAMPLES: usize = 5760;

/// Total decoded samples per channel for an Opus packet at 48 kHz, parsed
/// from the TOC byte (RFC 6716 §3.1): the 5-bit configuration selects the
/// per-frame duration and the 2 low bits the frame count (code 3 reads it
/// from the second byte). The `opus-rs` decoder API takes the exact packet
/// duration rather than an output-buffer capacity, so it is computed here
/// instead of trusting demuxer timestamps.
#[cfg(feature = "file-decode")]
pub(crate) fn opus_packet_frame_size(packet: &[u8]) -> Option<usize> {
    // Per-frame duration in 48 kHz samples for each of the 32 TOC
    // configurations (RFC 6716 Table 2): SILK 10/20/40/60 ms, hybrid 10/20
    // ms, CELT 2.5/5/10/20 ms.
    #[rustfmt::skip]
    const FRAME_DURATION_48K: [usize; 32] = [
        480, 960, 1920, 2880, // SILK narrowband
        480, 960, 1920, 2880, // SILK mediumband
        480, 960, 1920, 2880, // SILK wideband
        480, 960,             // hybrid super-wideband
        480, 960,             // hybrid fullband
        120, 240, 480, 960,   // CELT narrowband
        120, 240, 480, 960,   // CELT wideband
        120, 240, 480, 960,   // CELT super-wideband
        120, 240, 480, 960,   // CELT fullband
    ];
    let toc = *packet.first()?;
    let frames = match toc & 0b11 {
        0 => 1,
        1 | 2 => 2,
        _ => usize::from(packet.get(1)? & 0x3F),
    };
    if frames == 0 {
        return None;
    }
    let size = FRAME_DURATION_48K[(toc >> 3) as usize] * frames;
    (size <= OPUS_MAX_PACKET_SAMPLES).then_some(size)
}

/// True when a demuxer `next_packet` failure is a recoverable end-of-stream.
///
/// Symphonia surfaces a missing Ogg EOS page as `IoError(UnexpectedEof)` rather
/// than `Ok(None)`. Real-world producers (notably Android Telegram voice notes)
/// often omit EOS; if any PCM has already been decoded, treat that EOF as a
/// clean stream end so the upload still transcribes (see issue #217).
#[cfg(feature = "file-decode")]
pub(crate) fn is_recoverable_packet_eof(err: &symphonia::core::errors::Error) -> bool {
    matches!(
        err,
        symphonia::core::errors::Error::IoError(ioe)
            if ioe.kind() == std::io::ErrorKind::UnexpectedEof
    )
}

/// Pull the next demux packet, treating UnexpectedEof after successful PCM as EOS.
///
/// Returns `Ok(Some(packet))` to decode, `Ok(None)` to end the loop, or `Err`
/// for non-recoverable demux failures (and EOF with no audio yet).
#[cfg(feature = "file-decode")]
pub(super) fn next_demux_packet(
    format: &mut dyn FormatReader,
    have_pcm: bool,
) -> Result<Option<symphonia::core::packet::Packet>> {
    match format.next_packet() {
        Ok(Some(p)) => Ok(Some(p)),
        Ok(None) => Ok(None),
        Err(e) if is_recoverable_packet_eof(&e) && have_pcm => {
            tracing::debug!(
                "Demux UnexpectedEof after PCM already decoded; treating as end of stream"
            );
            Ok(None)
        }
        Err(e) => Err(anyhow::anyhow!("Error reading packet: {e}")),
    }
}

/// Decode the packets of an Opus track (OGG container) to per-channel f32
/// samples at 48 kHz.
///
/// Symphonia's OGG demuxer recognizes Opus (`CODEC_ID_OPUS`) but ships no
/// Opus decoder, so packets are decoded here with the pure-Rust `opus-rs`
/// libopus port (decoder only). Per RFC 7845 the decode rate is always
/// 48 kHz — the rate symphonia's mapper reports — and callers resample to
/// 16 kHz like for every other format. Only mono and stereo are supported,
/// which covers Telegram voice notes, browser MediaRecorder captures, and
/// `.opus` files; multistream (>2ch) OGG/Opus is rejected. `max_samples` is
/// the per-channel (48 kHz) sample budget, enforced incrementally as in the
/// symphonia decode loops; `limit_secs` is the seconds figure reported on a
/// trip.
#[cfg(feature = "file-decode")]
pub(super) fn decode_opus_channels(
    format: &mut dyn FormatReader,
    track_id: u32,
    channels: usize,
    max_samples: usize,
    limit_secs: f64,
) -> Result<Vec<Vec<f32>>> {
    if !(1..=2).contains(&channels) {
        anyhow::bail!("Opus with {channels} channels is not supported (mono/stereo only)");
    }
    let mut decoder = opus_rs::OpusDecoder::new(48_000, channels)
        .map_err(|e| anyhow::anyhow!("Opus decoder init failed: {e}"))?;
    let mut per_channel: Vec<Vec<f32>> = (0..channels).map(|_| Vec::new()).collect();
    let mut pcm: Vec<f32> = Vec::new();
    loop {
        let have_pcm = per_channel.first().is_some_and(|c| !c.is_empty());
        let Some(packet) = next_demux_packet(format, have_pcm)? else {
            break;
        };
        if packet.track_id != track_id {
            continue;
        }
        let frame_size = opus_packet_frame_size(&packet.data)
            .ok_or_else(|| anyhow::anyhow!("Malformed Opus packet"))?;
        pcm.resize(frame_size * channels, 0.0);
        let decoded = decoder
            .decode(&packet.data, frame_size, &mut pcm)
            .map_err(|e| anyhow::anyhow!("Opus decode error: {e}"))?
            .min(frame_size);
        if channels == 1 {
            per_channel[0].extend_from_slice(&pcm[..decoded]);
        } else {
            for frame in 0..decoded {
                for (c, buf) in per_channel.iter_mut().enumerate() {
                    buf.push(pcm[frame * channels + c]);
                }
            }
        }
        // Incremental length budget, same as the symphonia decode loops.
        let decoded_len = per_channel.first().map(|v| v.len()).unwrap_or(0);
        if decoded_len > max_samples {
            return Err(audio_too_long_err(decoded_len, 48_000, limit_secs));
        }
    }
    Ok(per_channel)
}
