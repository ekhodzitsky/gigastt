//! Per-channel (non-mixing) container decode.

use anyhow::{Context, Result};
use bytes::Bytes;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::codecs::audio::well_known::CODEC_ID_OPUS;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use super::super::MAX_SAMPLE_RATE;
use super::super::opus::{OPUS_DECODE_RATE, decode_opus_channels, next_demux_packet};
use super::super::resample::{RESAMPLE_STAGING_FRAMES, ResampleTo16k, SampleRate};
use super::super::{audio_too_long_err, resolve_budget, whole_buffer_limit_secs};
use super::BytesMediaSource;

/// Decode an audio file to one f32 sample vector per channel at 16 kHz.
///
/// Same probe/decode/resample pipeline as [`super::decode_audio_file`], but keeps
/// channels separate. The mono mix path remains unchanged.
#[cfg(feature = "file-decode")]
pub fn load_audio_channels(path: &str) -> Result<Vec<Vec<f32>>> {
    load_audio_channels_bounded(path, None)
}

/// `channels=split` decode with an explicit operator length budget. This path
/// holds every channel's decoded buffer in RAM at once, so it is bounded by at
/// least the whole-buffer safety ceiling; the server threads the request's
/// `--max-audio-secs` here, the public wrapper passes `None` (ceiling only).
#[cfg(feature = "file-decode")]
pub(crate) fn load_audio_channels_bounded(
    path: &str,
    max_audio_secs: Option<f64>,
) -> Result<Vec<Vec<f32>>> {
    let file =
        std::fs::File::open(path).with_context(|| format!("Failed to open audio file: {path}"))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
    {
        hint.with_extension(ext);
    }

    let source_label = format!(
        "format={}",
        std::path::Path::new(path)
            .extension()
            .unwrap_or_default()
            .to_string_lossy()
    );

    decode_audio_inner_channels(mss, hint, &source_label, max_audio_secs)
}

/// Decode raw audio bytes to one f32 sample vector per channel at 16 kHz.
#[cfg(feature = "file-decode")]
pub fn decode_audio_bytes_shared_channels(data: Bytes) -> Result<Vec<Vec<f32>>> {
    decode_audio_bytes_shared_channels_bounded(data, None)
}

/// `channels=split` byte decode with an explicit operator length budget — the
/// bytes twin of the crate-internal path decoder. `None` behaves exactly like
/// [`decode_audio_bytes_shared_channels`] (the whole-buffer safety ceiling); a
/// `Some(secs)` from the server's `--max-audio-secs` lowers it. Public so the
/// server can thread the operator limit into its own `channels=split` decode.
#[cfg(feature = "file-decode")]
pub fn decode_audio_bytes_shared_channels_bounded(
    data: Bytes,
    max_audio_secs: Option<f64>,
) -> Result<Vec<Vec<f32>>> {
    let source = BytesMediaSource::new(data);
    let mss = MediaSourceStream::new(Box::new(source), Default::default());
    decode_audio_inner_channels(mss, Hint::new(), "bytes", max_audio_secs)
}

/// Shared non-mixing decode: probe → format → decode → per-channel resample.
#[cfg(feature = "file-decode")]
fn decode_audio_inner_channels<'s>(
    mss: MediaSourceStream<'s>,
    hint: Hint,
    source_label: &str,
    max_audio_secs: Option<f64>,
) -> Result<Vec<Vec<f32>>> {
    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .context("Unsupported audio format")?;

    let track = format
        .default_track(TrackType::Audio)
        .context("No audio track found")?;
    let track_id = track.id;
    let audio_params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .context("No audio codec parameters")?;
    let sample_rate = audio_params.sample_rate.context("Unknown sample rate")?;
    if sample_rate == 0 || sample_rate > MAX_SAMPLE_RATE {
        anyhow::bail!("Unsupported sample rate: {sample_rate}Hz");
    }
    let channels = audio_params
        .channels
        .as_ref()
        .map(|c| c.count())
        .unwrap_or(1);
    let n_frames_hint = track.num_frames;
    // Opus packets decode at 48 kHz regardless of the container tag. The
    // length budget and the resampler have to use that rate, or a WebM
    // SamplingFrequency of 16 kHz both trips the ceiling early and leaves
    // the 48 kHz PCM unresampled.
    let is_opus = audio_params.codec == CODEC_ID_OPUS;
    let pcm_rate = if is_opus {
        OPUS_DECODE_RATE
    } else {
        sample_rate
    };
    let (max_samples, limit_secs) =
        resolve_budget(Some(whole_buffer_limit_secs(max_audio_secs)), pcm_rate);

    tracing::info!("Audio ({source_label}): {pcm_rate}Hz, {channels}ch (split)");

    // Each channel is an independent stream, so each gets its own staging
    // buffer and cached resampler; none of them ever holds more than one
    // chunk of source-rate audio.
    let hint = match n_frames_hint {
        Some(n) if n > 0 && n <= max_samples as u64 => Some(n as usize),
        _ => None,
    };
    // Source-rate frame count of the first channel, tracked separately because
    // the accumulators now hold 16 kHz samples while the length budget is
    // expressed in source-rate frames.
    let mut source_frames: usize = 0;

    // Symphonia demuxes OGG/Opus but ships no Opus decoder, so Opus packets
    // go through the `opus-rs` fallback and rejoin the shared resample tail.
    let acc: Vec<ResampleTo16k> = if is_opus {
        let decoded =
            decode_opus_channels(&mut *format, track_id, channels, max_samples, limit_secs)?;
        source_frames = decoded.first().map(|v| v.len()).unwrap_or(0);
        let mut acc = Vec::with_capacity(decoded.len());
        for samples in decoded {
            let mut chan = ResampleTo16k::new(SampleRate(pcm_rate), Some(samples.len()));
            for piece in samples.chunks(RESAMPLE_STAGING_FRAMES) {
                chan.stage().extend_from_slice(piece);
                chan.flush_full()?;
            }
            acc.push(chan);
        }
        acc
    } else {
        let mut decoder = symphonia::default::get_codecs()
            .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
            .context("Unsupported audio codec")?;

        let mut acc: Vec<ResampleTo16k> = (0..channels)
            .map(|_| ResampleTo16k::new(SampleRate(sample_rate), hint))
            .collect();

        loop {
            let have_pcm = source_frames > 0;
            let Some(packet) = next_demux_packet(&mut *format, have_pcm)? else {
                break;
            };

            if packet.track_id != track_id {
                continue;
            }

            let decoded = decoder.decode(&packet).context("Decode error")?;
            let spec = decoded.spec().clone();
            let num_frames = decoded.frames();
            let ch = spec.channels().count();

            if ch > acc.len() {
                // A channel that appears mid-stream starts from this packet,
                // so it gets no length hint.
                acc.resize_with(ch, || ResampleTo16k::new(SampleRate(sample_rate), None));
            }

            if ch > 1 {
                let mut interleaved: Vec<f32> = Vec::with_capacity(num_frames * ch);
                decoded.copy_to_vec_interleaved(&mut interleaved);
                for frame in 0..num_frames {
                    for c in 0..ch {
                        acc[c].stage().push(interleaved[frame * ch + c]);
                    }
                }
            } else if !acc.is_empty() {
                let stage = acc[0].stage();
                let offset = stage.len();
                stage.resize(offset + num_frames, 0.0);
                decoded.copy_to_slice_interleaved(&mut stage[offset..]);
            }
            // Both branches above grow the first channel by `num_frames`; the
            // `ch <= 1` branch is skipped entirely when there is no channel.
            if !acc.is_empty() {
                source_frames += num_frames;
            }

            if source_frames > max_samples {
                return Err(audio_too_long_err(source_frames, sample_rate, limit_secs));
            }

            for chan in &mut acc {
                chan.flush_full()?;
            }
        }

        acc
    };

    let duration_s = source_frames as f64 / pcm_rate as f64;
    tracing::info!(
        "Decoded {} channel(s), first channel {} samples at {}Hz ({:.1}s)",
        acc.len(),
        source_frames,
        pcm_rate,
        duration_s
    );

    let channel_count = acc.len();
    let per_channel = acc
        .into_iter()
        .map(ResampleTo16k::finish)
        .collect::<Result<Vec<_>>>()?;
    if pcm_rate != 16000 {
        tracing::info!("Resampled {channel_count} channel(s) to 16kHz");
    }

    Ok(per_channel)
}
