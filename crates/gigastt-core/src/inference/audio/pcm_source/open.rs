use anyhow::{Context, Result};
use bytes::Bytes;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::codecs::audio::well_known::CODEC_ID_OPUS;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use super::super::decode::BytesMediaSource;
use super::super::opus::{OPUS_DECODE_RATE, OpusStream};
use super::super::resample::{RESAMPLE_STAGING_FRAMES, ResampleTo16k, SampleRate};
use super::super::wave::{try_open_bytes, try_open_path};
use super::super::{MAX_SAMPLE_RATE, resolve_budget};
use super::{ChannelSelect, PcmSource, Source};

impl PcmSource {
    /// Open a file: WAVE containers go through ryf; others through symphonia.
    pub(in super::super) fn open(
        path: &str,
        max_audio_secs: Option<f64>,
        channel: ChannelSelect,
    ) -> Result<Self> {
        if let Some(wave) = try_open_path(path, channel, max_audio_secs)? {
            return Ok(Self::new(Source::Wave(Box::new(wave)), channel));
        }
        let file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open audio file: {path}"))?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());
        let mut hint = Hint::new();
        if let Some(ext) = std::path::Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
        {
            hint.with_extension(ext);
        }
        Self::from_mss(mss, hint, max_audio_secs, channel)
    }

    /// Open a shared, seekable bytes buffer without copying the container.
    pub(in super::super) fn from_bytes(
        data: Bytes,
        max_audio_secs: Option<f64>,
        channel: ChannelSelect,
    ) -> Result<Self> {
        if let Some(wave) = try_open_bytes(data.clone(), channel, max_audio_secs)? {
            return Ok(Self::new(Source::Wave(Box::new(wave)), channel));
        }
        let source = BytesMediaSource::new(data);
        let mss = MediaSourceStream::new(Box::new(source), Default::default());
        Self::from_mss(mss, Hint::new(), max_audio_secs, channel)
    }

    /// Probe a non-WAVE container and set up the streaming decoder (or the
    /// Opus `opus-rs` fallback). Scalar params are copied out (and the non-Opus
    /// decoder built) inside one borrow scope so the `FormatReader` is free to
    /// move or be driven afterwards — the same shape as the whole-buffer
    /// `decode_audio_inner`.
    fn from_mss(
        mss: MediaSourceStream<'static>,
        hint: Hint,
        max_audio_secs: Option<f64>,
        channel: ChannelSelect,
    ) -> Result<Self> {
        let format = symphonia::default::get_probe()
            .probe(
                &hint,
                mss,
                FormatOptions::default(),
                MetadataOptions::default(),
            )
            .context("Unsupported audio format")?;

        let (track_id, sample_rate, channels, decoder_opt) = {
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
            // Opus is demuxed by symphonia but has no symphonia decoder; the
            // `opus-rs` fallback needs the `FormatReader`, so build no decoder
            // here and take the packet-wise Opus branch below.
            let decoder_opt = if audio_params.codec == CODEC_ID_OPUS {
                None
            } else {
                Some(
                    symphonia::default::get_codecs()
                        .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
                        .context("Unsupported audio codec")?,
                )
            };
            (track_id, sample_rate, channels, decoder_opt)
        };

        let (max_samples, limit_secs) = resolve_budget(max_audio_secs, sample_rate);

        match decoder_opt {
            None => {
                // The budget and the resampler both count at the Opus decode
                // rate (48 kHz). That is what the decoder emits. The container
                // rate is the capture rate (OpusHead input rate, or a WebM
                // SamplingFrequency a browser sets to the AudioContext rate)
                // and is not the PCM rate (RFC 7845). Feeding it to the
                // resampler leaves 48 kHz audio unresampled.
                let (max_samples, limit_secs) = resolve_budget(max_audio_secs, OPUS_DECODE_RATE);
                tracing::info!(
                    "Audio (opus): container {sample_rate}Hz, decode {OPUS_DECODE_RATE}Hz, {channels}ch (streaming PCM)"
                );
                Ok(Self::new(
                    Source::Opus {
                        format,
                        track_id,
                        stream: Box::new(OpusStream::new(channels, channel)?),
                        decoded_48k: 0,
                        max_samples,
                        limit_secs,
                        resampler: Box::new(ResampleTo16k::new(SampleRate(OPUS_DECODE_RATE), None)),
                        pending: Vec::with_capacity(RESAMPLE_STAGING_FRAMES),
                    },
                    channel,
                ))
            }
            Some(decoder) => {
                tracing::info!("Audio: {sample_rate}Hz, {channels}ch (streaming PCM)");
                Ok(Self::new(
                    Source::Streaming {
                        format,
                        decoder,
                        track_id,
                        sample_rate,
                        source_frames: 0,
                        max_samples,
                        limit_secs,
                        // No length hint: the windowed path never materializes the
                        // whole 16 kHz stream, so it must not reserve for it.
                        resampler: Box::new(ResampleTo16k::new(SampleRate(sample_rate), None)),
                        interleaved: Vec::new(),
                    },
                    channel,
                ))
            }
        }
    }
}
