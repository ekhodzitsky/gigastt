//! Shared file-transcription core for REST, OpenAI, and jobs.
//!
//! Surfaces keep their own HTTP / progress envelopes; the blocking
//! channels / diarization / hotwords routing lives here so those paths
//! cannot drift.

use axum::body::Bytes;
use gigastt_core::error::GigasttError;
use gigastt_core::inference::{
    Engine, HotwordOverride, OwnedReservation, SessionTriplet, TranscribeOverrides,
    TranscribeRequest, TranscribeResult, TranscribeSource,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Options for a single file-transcription run after request validation.
#[derive(Clone, Default)]
pub(crate) struct FileTranscribeOpts {
    pub overrides: TranscribeOverrides,
    pub hotwords: Option<HotwordOverride>,
    pub split_channels: bool,
    pub diarization: bool,
    /// When set, decode the body as raw telephony with legacy PCM16 precision
    /// using bounded PCM for short clips and compact WAV for longer ones.
    /// REST-only today; jobs do not expose `?codec=`.
    pub raw_codec: Option<(gigastt_core::inference::audio::TelephonyCodec, u32)>,
    /// Cooperative-cancellation flag threaded into the engine's token/frame
    /// decode loop. Flipping it (client disconnect, `DELETE /v1/jobs/{id}`,
    /// shutdown, or the no-progress watchdog) ends the run at the next decode step.
    pub abort: Option<Arc<AtomicBool>>,
    pub partial: Option<Arc<gigastt_core::inference::TranscriptSnapshot>>,
    /// Progress sink the engine advances after each window with the cumulative
    /// count of processed 16 kHz samples. The server watchdog reads it to reset
    /// its no-progress deadline and to drive a real job progress bar.
    pub progress: Option<Arc<AtomicU64>>,
    /// Actual number of independently decoded channels (one for mono fallback).
    /// Jobs divide sample work by this count for completion presentation; the
    /// watchdog reads the unscaled sample counter so no work increments vanish.
    pub progress_channels: Option<Arc<AtomicU64>>,
    /// Write-once sink for the offline-diarization outcome. When set (with
    /// `diarization = true`), the engine records why speakers were or were not
    /// labeled so a surface can turn a `?diarization=true` request that produced
    /// no labels into a capability notice instead of an all-empty-speaker
    /// transcript. `None` records nothing.
    pub diarization_outcome:
        Option<Arc<std::sync::OnceLock<gigastt_core::inference::DiarizationOutcome>>>,
    /// Opt-in operator length limit from `--max-audio-secs` (`None` = unlimited).
    /// Threaded into the engine's per-window decode and the `channels=split`
    /// decode; the whole-buffer decoders clamp it to their fixed safety ceiling.
    pub max_audio_secs: Option<f64>,
    pub file_zero_context: bool,
}

impl FileTranscribeOpts {
    /// Shared engine request context for every audio-routing branch.
    fn request<'a>(&'a self, source: TranscribeSource<'a>) -> TranscribeRequest<'a> {
        // Channel speaker labels take precedence over offline diarization,
        // including when the split request falls back to mono.
        let diarization = self.diarization && !self.split_channels;
        TranscribeRequest::new(source)
            .with_overrides(self.overrides)
            .with_hotwords(self.hotwords.as_ref())
            .with_diarization(diarization)
            .with_diarization_outcome(if self.split_channels {
                None
            } else {
                self.diarization_outcome.clone()
            })
            .with_abort(self.abort.clone())
            .with_partial(self.partial.clone())
            .with_progress(self.progress.clone())
            .with_max_audio_secs(self.max_audio_secs)
            .with_file_zero_context(self.file_zero_context)
    }
}

/// Map a core decode `anyhow::Error` to a typed [`GigasttError`], preserving a
/// typed `AudioTooLong` (so the HTTP layer answers 413 `audio_too_long` instead
/// of a generic 422) rather than flattening every decode failure into
/// `InvalidAudio`. Mirrors the core-internal seam the engine uses.
pub(crate) fn map_decode_error(e: anyhow::Error) -> GigasttError {
    match e.downcast::<GigasttError>() {
        Ok(g) => g,
        Err(e) => GigasttError::InvalidAudio {
            reason: format!("{e:#}"),
        },
    }
}

// Limit direct PCM to 480,000 samples (1.92 MB of data, excluding spare capacity).
// Longer clips retain PCM16
// WAV (2 bytes/sample) and use the existing bounded window decoder.
const RAW_PCM_MAX_SAMPLES: usize = 30 * 16_000;

#[derive(Debug)]
pub(crate) enum PreparedRawAudio {
    Samples(Vec<f32>),
    Wav(Vec<u8>),
}

/// Preserve the former WAV precision without doubling long-clip retained storage.
pub(crate) fn prepare_raw_audio(
    body: &[u8],
    codec: gigastt_core::inference::audio::TelephonyCodec,
    sample_rate: u32,
    max_audio_secs: Option<f64>,
    abort: Option<&(dyn Fn() -> bool + Sync)>,
) -> Result<PreparedRawAudio, GigasttError> {
    let mut samples = gigastt_core::inference::audio::decode_telephony_raw_bounded_with_abort(
        body,
        codec,
        sample_rate,
        max_audio_secs,
        abort,
    )
    .map_err(map_decode_error)?;
    if samples.len() > RAW_PCM_MAX_SAMPLES {
        if abort.is_some_and(|abort| abort()) {
            return Err(GigasttError::Cancelled);
        }
        // Encode original samples exactly once; quantizing before encoding
        // would apply the legacy 32767 multiplier twice.
        let wav = gigastt_core::inference::audio::encode_wav_pcm16(&samples, 16000);
        if abort.is_some_and(|abort| abort()) {
            return Err(GigasttError::Cancelled);
        }
        return Ok(PreparedRawAudio::Wav(wav));
    }
    for chunk in samples.chunks_mut(4096) {
        if abort.is_some_and(|abort| abort()) {
            return Err(GigasttError::Cancelled);
        }
        gigastt_core::inference::audio::quantize_wav_pcm16_in_place(chunk);
    }
    if abort.is_some_and(|abort| abort()) {
        return Err(GigasttError::Cancelled);
    }
    Ok(PreparedRawAudio::Samples(samples))
}

/// Blocking file transcription against a reserved triplet.
///
/// Handles raw-codec preparation, `channels=split` with mono fallback, diarization,
/// and the default mono path. Callers own pool checkout, panic wrapping, and
/// timeout policy.
pub(crate) fn run_file_transcribe_blocking(
    engine: &Engine,
    body: Bytes,
    reservation: &mut OwnedReservation<SessionTriplet>,
    opts: &FileTranscribeOpts,
) -> Result<TranscribeResult, GigasttError> {
    with_file_transcribe_request(body, opts, engine.has_vad(), |request| {
        engine.transcribe_request(request, reservation)
    })
}

/// Route the audio while keeping the request visible at the engine boundary.
fn with_file_transcribe_request<T>(
    body: Bytes,
    opts: &FileTranscribeOpts,
    has_vad: bool,
    transcribe: impl FnOnce(TranscribeRequest<'_>) -> Result<T, GigasttError>,
) -> Result<T, GigasttError> {
    let cancelled = || {
        opts.abort
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
    };
    let check_abort = || {
        if cancelled() {
            Err(GigasttError::Cancelled)
        } else {
            Ok(())
        }
    };
    check_abort()?;
    let abort: Option<&(dyn Fn() -> bool + Sync)> = opts.abort.as_ref().map(|_| &cancelled as _);
    // Check the boundary even for mono fallback and already prepared channels.
    let transcribe = |request| {
        check_abort()?;
        transcribe(request)
    };
    if let Some(channels) = &opts.progress_channels {
        channels.store(1, Ordering::Relaxed);
    }
    // Keep upload admission alive even if raw-codec or channel decoding replaces
    // the encoded buffer with prepared audio inside this detached worker.
    let _upload_lifetime = body.clone();
    if let Some((codec, rate)) = opts.raw_codec {
        let prepared = prepare_raw_audio(&body, codec, rate, opts.max_audio_secs, abort)?;
        if opts.split_channels {
            tracing::warn!(
                "channels=split requested for raw mono audio; falling back to mono transcription"
            );
        }
        return match prepared {
            PreparedRawAudio::Samples(samples) => {
                transcribe(opts.request(TranscribeSource::Samples(&samples)))
            }
            PreparedRawAudio::Wav(wav) => {
                transcribe(opts.request(TranscribeSource::Bytes(wav.into())))
            }
        };
    }

    check_abort()?;
    if opts.split_channels {
        // Deciding whether to split used to mean decoding every channel of the
        // whole file and correlating two of them, which is what pinned this
        // path to a duration ceiling. The scan answers it in one pass — from
        // the header alone unless the file is exactly stereo.
        let use_vad = has_vad && opts.overrides.vad.unwrap_or(true);
        let prepared = if use_vad {
            gigastt_core::inference::audio::prepare_channels_for_vad_with_abort(
                body.clone(),
                opts.max_audio_secs,
                abort,
            )
            .map_err(map_decode_error)?
        } else {
            gigastt_core::inference::audio::PreparedChannels {
                scan: gigastt_core::inference::audio::scan_channels_with_abort(
                    body.clone(),
                    opts.max_audio_secs,
                    abort,
                )
                .map_err(map_decode_error)?,
                decoded: None,
            }
        };
        let scan = prepared.scan;
        if let Some(channels) = &opts.progress_channels {
            let count = if scan.mono_fallback_reason().is_some() {
                1
            } else {
                scan.channels
            };
            channels.store(count as u64, Ordering::Relaxed);
        }
        if let Some(reason) = scan.mono_fallback_reason() {
            tracing::warn!(
                "channels=split requested but {reason} detected; falling back to mono transcription"
            );
            transcribe(opts.request(TranscribeSource::Bytes(body)))
        } else if use_vad {
            // The per-channel VAD pass still needs each channel resident, so a
            // VAD-enabled server keeps the whole-buffer split (and its ceiling)
            // until the VAD itself runs inside the window loop.
            let channels = match prepared.decoded {
                Some(channels) => channels,
                None => gigastt_core::inference::audio::decode_audio_bytes_shared_channels_bounded_with_abort(
                    body,
                    opts.max_audio_secs,
                    abort,
                )
                .map_err(map_decode_error)?,
            };
            transcribe(opts.request(TranscribeSource::Channels(&channels)))
        } else {
            transcribe(opts.request(TranscribeSource::ChannelStreams {
                data: body,
                channels: scan.channels,
            }))
        }
    } else {
        transcribe(opts.request(TranscribeSource::Bytes(body)))
    }
}

#[cfg(test)]
mod tests;
