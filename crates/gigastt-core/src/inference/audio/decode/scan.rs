//! One-pass channel scan for `channels=split` without materializing the file.

use anyhow::{Context, Result};
use bytes::Bytes;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::codecs::audio::well_known::CODEC_ID_OPUS;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use super::super::MAX_SAMPLE_RATE;
use super::super::opus::next_demux_packet;
use super::super::resample::{ResampleTo16k, SampleRate};
use super::super::{audio_too_long_err, check_decode_abort, resolve_budget};
use super::{
    BytesMediaSource, DualMonoDetector, decode_audio_bytes_shared_channels_bounded_with_abort,
    is_dual_mono,
};

/// What a one-pass scan of a container's channels found.
#[cfg(feature = "file-decode")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelScan {
    /// Channels the container declares.
    pub channels: usize,
    /// True only when there are exactly two and they are near-identical — a PBX
    /// that recorded the same mix to both. Transcribing those as two speakers
    /// would duplicate every word.
    pub dual_mono: bool,
}

#[cfg(feature = "file-decode")]
impl ChannelScan {
    /// Whether `channels=split` should fall back to the mono mix, and why.
    /// `None` means genuine stereo: split it.
    pub fn mono_fallback_reason(&self) -> Option<&'static str> {
        match self.channels {
            0 => Some("no channels"),
            1 => Some("mono audio"),
            2 if self.dual_mono => Some("dual-mono audio"),
            2 => None,
            _ => Some("more than two channels"),
        }
    }
}

/// Decide how `channels=split` should treat `data` **without materializing it**.
///
/// The split path used to answer this by decoding every channel of the whole
/// file and correlating the two in full, which is what pinned it to a duration
/// ceiling. Almost all of the answer is in the header: anything that is not
/// exactly two channels falls back to the mono mix, and that needs no decode at
/// all. Only the two-channel case has to look at the audio, and
/// [`DualMonoDetector`] does that in one pass over six accumulators.
///
/// The correlation is taken on the **16 kHz resampled** channels, on the same
/// per-packet staging cadence the whole-buffer decode used, so it is the same
/// statistic on the same numbers — the verdict does not move.
///
/// OGG/Opus and stereo AAC are exceptions: the scan materializes their channels
/// before correlation, retaining the whole-buffer decode ceiling.
#[cfg(feature = "file-decode")]
pub fn scan_channels(data: Bytes, max_audio_secs: Option<f64>) -> Result<ChannelScan> {
    scan_channels_inner(data, max_audio_secs, false, None).map(|prepared| prepared.scan)
}

fn scan_channels_inner(
    data: Bytes,
    max_audio_secs: Option<f64>,
    retain_decoded: bool,
    abort: Option<&(dyn Fn() -> bool + Sync)>,
) -> Result<PreparedChannels> {
    check_decode_abort(abort)?;
    if let Some(hint) = super::super::aac::header_channels(data.as_ref())? {
        check_decode_abort(abort)?;
        if hint.channels != 2 {
            return Ok(PreparedChannels {
                scan: ChannelScan {
                    channels: hint.channels,
                    dual_mono: false,
                },
                decoded: None,
            });
        }
        let decoded =
            decode_audio_bytes_shared_channels_bounded_with_abort(data, max_audio_secs, abort)?;
        check_decode_abort(abort)?;
        let scan = ChannelScan {
            channels: decoded.len(),
            dual_mono: is_dual_mono(&decoded),
        };
        check_decode_abort(abort)?;
        let decoded = if retain_decoded && scan.mono_fallback_reason().is_none() {
            Some(decoded)
        } else {
            None
        };
        return Ok(PreparedChannels { scan, decoded });
    }
    let source = BytesMediaSource::new(data.clone());
    let mss = MediaSourceStream::new(Box::new(source), Default::default());
    let mut format = symphonia::default::get_probe()
        .probe(
            &Hint::new(),
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .context("Unsupported audio format")?;

    check_decode_abort(abort)?;
    let (track_id, sample_rate, channels, is_opus) = {
        let track = format
            .default_track(TrackType::Audio)
            .context("No audio track found")?;
        let params = track
            .codec_params
            .as_ref()
            .and_then(|p| p.audio())
            .context("No audio codec parameters")?;
        let sample_rate = params.sample_rate.context("Unknown sample rate")?;
        if sample_rate == 0 || sample_rate > MAX_SAMPLE_RATE {
            anyhow::bail!("Unsupported sample rate: {sample_rate}Hz");
        }
        (
            track.id,
            sample_rate,
            params.channels.as_ref().map(|c| c.count()).unwrap_or(1),
            params.codec == CODEC_ID_OPUS,
        )
    };

    // Header-only verdict: no decode, whatever the file's length.
    if channels != 2 {
        return Ok(PreparedChannels {
            scan: ChannelScan {
                channels,
                dual_mono: false,
            },
            decoded: None,
        });
    }

    if is_opus {
        let decoded =
            decode_audio_bytes_shared_channels_bounded_with_abort(data, max_audio_secs, abort)?;
        check_decode_abort(abort)?;
        let scan = ChannelScan {
            channels: decoded.len(),
            dual_mono: is_dual_mono(&decoded),
        };
        check_decode_abort(abort)?;
        let decoded = if retain_decoded && scan.mono_fallback_reason().is_none() {
            Some(decoded)
        } else {
            None
        };
        return Ok(PreparedChannels { scan, decoded });
    }

    let (max_samples, limit_secs) = resolve_budget(max_audio_secs, sample_rate);
    let mut decoder = {
        let track = format
            .default_track(TrackType::Audio)
            .context("No audio track found")?;
        let params = track
            .codec_params
            .as_ref()
            .and_then(|p| p.audio())
            .context("No audio codec parameters")?;
        symphonia::default::get_codecs()
            .make_audio_decoder(params, &AudioDecoderOptions::default())
            .context("Unsupported audio codec")?
    };

    // One resampler per channel, fed on the same cadence the whole-buffer
    // decode used, so the 16 kHz samples reaching the detector are the ones it
    // would have correlated.
    let mut acc: Vec<ResampleTo16k> = (0..2)
        .map(|_| ResampleTo16k::new(SampleRate(sample_rate), None))
        .collect();
    let mut detector = DualMonoDetector::new();
    let mut ready: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
    let mut interleaved: Vec<f32> = Vec::new();
    let mut source_frames: usize = 0;

    loop {
        check_decode_abort(abort)?;
        let Some(packet) = next_demux_packet(&mut *format, source_frames > 0)? else {
            break;
        };
        if packet.track_id != track_id {
            continue;
        }
        let decoded = decoder.decode(&packet).context("Decode error")?;
        check_decode_abort(abort)?;
        let num_frames = decoded.frames();
        let ch = decoded.spec().channels().count();
        if ch < 2 {
            // A mid-stream drop to mono makes the pair meaningless; treat the
            // stream as not dual-mono and let the split path decide per channel.
            break;
        }
        interleaved.clear();
        decoded.copy_to_vec_interleaved(&mut interleaved);
        for (c, chan) in acc.iter_mut().enumerate() {
            let stage = chan.stage();
            for frame in 0..num_frames {
                stage.push(interleaved[frame * ch + c]);
            }
        }
        source_frames += num_frames;
        if source_frames > max_samples {
            return Err(audio_too_long_err(source_frames, sample_rate, limit_secs));
        }
        for (c, chan) in acc.iter_mut().enumerate() {
            check_decode_abort(abort)?;
            chan.flush_full()?;
            chan.drain_ready_into(&mut ready[c]);
        }
        let (left, right) = ready.split_at_mut(1);
        detector.push(&left[0], &right[0]);
        ready[0].clear();
        ready[1].clear();
    }

    for (c, chan) in acc.into_iter().enumerate() {
        check_decode_abort(abort)?;
        let mut tail = Vec::new();
        let mut chan = chan;
        chan.finish_into(&mut tail)?;
        ready[c] = tail;
    }
    let (left, right) = ready.split_at_mut(1);
    detector.push(&left[0], &right[0]);
    check_decode_abort(abort)?;

    Ok(PreparedChannels {
        scan: ChannelScan {
            channels: 2,
            dual_mono: detector.is_dual_mono(),
        },
        decoded: None,
    })
}

/// Channel decision with reusable PCM when the scan already decoded it in full.
#[derive(Debug)]
pub struct PreparedChannels {
    /// Full-recording channel decision.
    pub scan: ChannelScan,
    /// Existing scan PCM for genuine stereo VAD; absent for streaming scans and mono fallback.
    pub decoded: Option<Vec<Vec<f32>>>,
}

/// Prepare the channel decision for a VAD-enabled split request.
/// Retains genuine stereo Opus PCM that the scan already materializes, under
/// the same whole-buffer duration ceiling. Other formats keep their streaming
/// scan and return no PCM. Mono fallback always uses the original container,
/// preserving its mix-before-resample behavior.
///
/// # Errors
/// Returns the same probe, decode and duration-limit errors as [`scan_channels`].
pub fn prepare_channels_for_vad(
    data: Bytes,
    max_audio_secs: Option<f64>,
) -> Result<PreparedChannels> {
    scan_channels_inner(data, max_audio_secs, true, None)
}

/// Scan channels with cooperative cancellation between decode steps.
/// `None` preserves [`scan_channels`] behavior. A true predicate returns typed
/// [`GigasttError::Cancelled`](crate::error::GigasttError::Cancelled); active
/// codec/resampler calls and full-buffer Opus correlation are not interrupted.
pub fn scan_channels_with_abort(
    data: Bytes,
    max_audio_secs: Option<f64>,
    abort: Option<&(dyn Fn() -> bool + Sync)>,
) -> Result<ChannelScan> {
    scan_channels_inner(data, max_audio_secs, false, abort).map(|prepared| prepared.scan)
}

/// Prepare VAD channels with the cancellation contract of [`scan_channels_with_abort`].
/// `None` preserves [`prepare_channels_for_vad`] behavior.
pub fn prepare_channels_for_vad_with_abort(
    data: Bytes,
    max_audio_secs: Option<f64>,
    abort: Option<&(dyn Fn() -> bool + Sync)>,
) -> Result<PreparedChannels> {
    scan_channels_inner(data, max_audio_secs, true, abort)
}
