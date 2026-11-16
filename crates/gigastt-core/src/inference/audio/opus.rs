//! Ogg/Opus packet decode and soft-EOF demux helpers.

#[cfg(feature = "file-decode")]
use anyhow::Result;
#[cfg(feature = "file-decode")]
use symphonia::core::formats::FormatReader;

#[cfg(feature = "file-decode")]
use super::audio_too_long_err;
#[cfg(feature = "file-decode")]
use super::stream::ChannelSelect;

/// Opus always decodes at 48 kHz regardless of the container's declared input
/// rate (RFC 7845 §5.1), so this — not the header rate — is the unit the length
/// budget counts and a duration trip reports.
#[cfg(feature = "file-decode")]
pub(super) const OPUS_DECODE_RATE: u32 = 48_000;

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

/// Reject channel layouts the `opus-rs` fallback cannot decode. Mono/stereo
/// covers Telegram voice notes, browser MediaRecorder captures, and `.opus`
/// files; multistream (>2ch) OGG/Opus is out.
#[cfg(feature = "file-decode")]
fn check_opus_channels(channels: usize) -> Result<()> {
    if !(1..=2).contains(&channels) {
        anyhow::bail!("Opus with {channels} channels is not supported (mono/stereo only)");
    }
    Ok(())
}

/// Per RFC 7845 the decode rate is always 48 kHz, whatever the container's
/// declared input rate.
#[cfg(feature = "file-decode")]
fn new_opus_decoder(channels: usize) -> Result<opus_rs::OpusDecoder> {
    opus_rs::OpusDecoder::new(OPUS_DECODE_RATE as i32, channels)
        .map_err(|e| anyhow::anyhow!("Opus decoder init failed: {e}"))
}

/// Decode one packet into `pcm` (interleaved, `channels` deep), returning the
/// per-channel frame count. The single place that knows the packet contract —
/// the `opus-rs` API takes the exact frame duration rather than a buffer
/// capacity — shared by the whole-buffer and streaming decoders so they cannot
/// drift apart.
///
/// Each frame is handed over as its own code 0 packet (`scratch` holds the
/// rewritten TOC plus the frame). The decoder carries its state across the
/// calls exactly as it would across the frames of one packet, so this is the
/// same decode — only with [`split_opus_packet`] doing the framing instead of
/// the `opus-rs` parser it cannot be trusted with.
#[cfg(feature = "file-decode")]
fn decode_packet_interleaved(
    decoder: &mut opus_rs::OpusDecoder,
    channels: usize,
    data: &[u8],
    pcm: &mut Vec<f32>,
    scratch: &mut Vec<u8>,
) -> Result<usize> {
    let framing = split_opus_packet(data)?;
    let frame_size = framing.samples_per_frame;
    pcm.resize(framing.packet_samples() * channels, 0.0);
    let mut decoded = 0usize;
    for frame in framing.frames() {
        scratch.clear();
        scratch.reserve(frame.len() + 1);
        // Clearing the two code bits turns the shared TOC into a code 0
        // (single-frame) one; the rest of the configuration is untouched.
        scratch.push(framing.toc & 0xFC);
        scratch.extend_from_slice(frame);
        let written = decoder
            .decode(scratch, frame_size, &mut pcm[decoded * channels..])
            .map_err(|e| anyhow::anyhow!("Opus decode error: {e}"))?
            .min(frame_size);
        decoded += written;
    }
    Ok(decoded)
}

/// Packet-at-a-time Opus decode, mixed down to mono as it goes.
///
/// The streaming counterpart of [`decode_opus_channels_with_abort`], which has to hold
/// every channel of the whole file before anything can be mixed or resampled —
/// the reason the Opus path stayed on the whole-buffer duration ceiling while
/// every other container streamed. This one holds a single packet.
///
/// The mix is the same mean
/// [`mix_channels_to_mono`](super::decode::mix_channels_to_mono) computes, in
/// the same summation order, so the samples are identical rather than merely
/// equivalent.
#[cfg(feature = "file-decode")]
pub(super) struct OpusStream {
    decoder: opus_rs::OpusDecoder,
    channels: usize,
    /// Mix down, or keep one channel — `channels=split` needs the latter, and
    /// without it a per-channel read of an `.opus` would silently get the mix.
    channel: ChannelSelect,
    /// Interleaved per-packet scratch, hoisted out of the decode loop.
    pcm: Vec<f32>,
    /// Per-frame code 0 packet scratch, hoisted out of the decode loop.
    frame: Vec<u8>,
}

#[cfg(feature = "file-decode")]
impl OpusStream {
    #[cfg(feature = "unlace")]
    pub(super) fn packet_frames(data: &[u8]) -> Result<usize> {
        Ok(split_opus_packet(data)?.packet_samples())
    }

    pub(super) fn new(channels: usize, channel: ChannelSelect) -> Result<Self> {
        check_opus_channels(channels)?;
        Ok(Self {
            decoder: new_opus_decoder(channels)?,
            channels,
            channel,
            pcm: Vec::new(),
            frame: Vec::new(),
        })
    }

    /// Decode one packet, appending its mono 48 kHz samples to `out`. Returns
    /// the number of frames appended.
    pub(super) fn decode_packet(&mut self, data: &[u8], out: &mut Vec<f32>) -> Result<usize> {
        let frames = decode_packet_interleaved(
            &mut self.decoder,
            self.channels,
            data,
            &mut self.pcm,
            &mut self.frame,
        )?;
        match self.channel {
            ChannelSelect::Mono => push_mono_mix(&self.pcm, self.channels, frames, out),
            ChannelSelect::One(k) if k < self.channels => {
                for frame in 0..frames {
                    out.push(self.pcm[frame * self.channels + k]);
                }
            }
            ChannelSelect::One(_) => {}
        }
        Ok(frames)
    }
}

/// Append the mono mix of the first `frames` interleaved frames of `pcm`.
///
/// Deliberately the same arithmetic as
/// [`mix_channels_to_mono`](super::decode::mix_channels_to_mono) over the same
/// samples de-interleaved — same summation order, same divisor — so mixing
/// per packet cannot drift from mixing the whole file at once. Pinned against
/// it in `tests.rs`.
#[cfg(feature = "file-decode")]
pub(super) fn push_mono_mix(pcm: &[f32], channels: usize, frames: usize, out: &mut Vec<f32>) {
    let ch = channels as f32;
    for frame in 0..frames {
        let base = frame * channels;
        out.push(pcm[base..base + channels].iter().sum::<f32>() / ch);
    }
}

#[cfg(feature = "file-decode")]
#[cfg(test)]
thread_local! {
    pub(super) static CHANNEL_DECODE_PASSES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
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
#[cfg(all(test, feature = "file-decode"))]
pub(super) fn decode_opus_channels(
    format: &mut dyn FormatReader,
    track_id: u32,
    channels: usize,
    max_samples: usize,
    limit_secs: f64,
) -> Result<Vec<Vec<f32>>> {
    decode_opus_channels_with_abort(format, track_id, channels, max_samples, limit_secs, None)
}

#[cfg(feature = "file-decode")]
pub(super) fn decode_opus_channels_with_abort(
    format: &mut dyn FormatReader,
    track_id: u32,
    channels: usize,
    max_samples: usize,
    limit_secs: f64,
    abort: Option<&(dyn Fn() -> bool + Sync)>,
) -> Result<Vec<Vec<f32>>> {
    super::check_decode_abort(abort)?;
    #[cfg(test)]
    CHANNEL_DECODE_PASSES.with(|count| count.set(count.get() + 1));
    check_opus_channels(channels)?;
    let mut decoder = new_opus_decoder(channels)?;
    let mut per_channel: Vec<Vec<f32>> = (0..channels).map(|_| Vec::new()).collect();
    let mut pcm: Vec<f32> = Vec::new();
    let mut frame: Vec<u8> = Vec::new();
    loop {
        super::check_decode_abort(abort)?;
        let have_pcm = per_channel.first().is_some_and(|c| !c.is_empty());
        let Some(packet) = next_demux_packet(format, have_pcm)? else {
            break;
        };
        if packet.track_id != track_id {
            continue;
        }
        let decoded =
            decode_packet_interleaved(&mut decoder, channels, &packet.data, &mut pcm, &mut frame)?;
        super::check_decode_abort(abort)?;
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
            return Err(audio_too_long_err(
                decoded_len,
                OPUS_DECODE_RATE,
                limit_secs,
            ));
        }
    }
    super::check_decode_abort(abort)?;
    Ok(per_channel)
}

#[cfg(feature = "file-decode")]
mod framing;
#[cfg(feature = "file-decode")]
use framing::split_opus_packet;
