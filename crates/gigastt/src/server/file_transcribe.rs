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

/// Sets a shared abort flag when dropped. Held in the REST handler's async
/// scope so that a client disconnect — which drops the handler future before it
/// returns — flips the flag, and the detached blocking decode observes it at the
/// next decode step and releases its pooled triplet. On the normal return
/// path the flag is set after the result is already in hand, so it is a no-op.
pub(crate) struct AbortOnDrop(pub Arc<AtomicBool>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// Link a blocking SSE producer to shutdown and receiver disconnect. Dropping
/// the returned guard after the producer finishes also retires the watcher.
pub(crate) fn stream_abort<T: Send + 'static>(
    tx: &tokio::sync::mpsc::Sender<T>,
    shutdown: tokio_util::sync::CancellationToken,
    tracker: &tokio_util::task::TaskTracker,
) -> (Arc<AtomicBool>, tokio_util::sync::DropGuard) {
    let abort = Arc::new(AtomicBool::new(false));
    let flag = abort.clone();
    let finished = tokio_util::sync::CancellationToken::new();
    let complete = finished.clone();
    let tx = tx.clone();
    tracker.spawn(async move {
        tokio::select! {
            biased;
            _ = complete.cancelled() => return,
            _ = shutdown.cancelled() => {},
            _ = tx.closed() => {},
        }
        flag.store(true, Ordering::Relaxed);
    });
    (abort, finished.drop_guard())
}

/// Outcome of awaiting a blocking transcription under the no-progress watchdog.
pub(crate) enum WatchdogOutcome<T> {
    /// The blocking task finished; carries the raw join result.
    Joined(Result<T, tokio::task::JoinError>),
    /// No window completed within the inference timeout. `abort` was set so the
    /// run cancels at its next decode step and frees the triplet.
    TimedOut,
}

/// Await a blocking transcription `handle`, redefining `inference_timeout_secs`
/// from a total wall-clock cap into a **no-progress watchdog**: the deadline
/// resets every time `progress` advances (i.e. a window completes), so a long
/// file that keeps making progress never trips, while a stalled run exhausts
/// its no-progress budget. `timeout_secs == 0` disables the
/// watchdog. Progress is sampled every 100 ms: with prompt runtime scheduling,
/// a stall is detected between `timeout` and `timeout + 100 ms` after the last
/// increment. Atomic counters carry no write timestamp, so the deadline uses
/// the observation time. A fired `shutdown` also flips `abort`, so SIGTERM
/// cancels the run at its next window instead of draining the whole file.
pub(crate) async fn await_transcription_watchdog<T>(
    mut handle: tokio::task::JoinHandle<T>,
    progress: &AtomicU64,
    abort: &AtomicBool,
    timeout_secs: u64,
    shutdown: &tokio_util::sync::CancellationToken,
) -> WatchdogOutcome<T> {
    let mut shutdown_fired = false;

    if timeout_secs == 0 {
        // No watchdog: still link shutdown -> abort so a drain cancels promptly.
        loop {
            tokio::select! {
                joined = &mut handle => return WatchdogOutcome::Joined(joined),
                _ = shutdown.cancelled(), if !shutdown_fired => {
                    shutdown_fired = true;
                    abort.store(true, Ordering::Relaxed);
                }
            }
        }
    }

    let timeout = std::time::Duration::from_secs(timeout_secs);
    let mut last_progress = progress.load(Ordering::Relaxed);
    let mut deadline = tokio::time::Instant::now() + timeout;
    let poll_interval = std::time::Duration::from_millis(100);
    loop {
        let next_poll = deadline.min(tokio::time::Instant::now() + poll_interval);
        tokio::select! {
            joined = &mut handle => return WatchdogOutcome::Joined(joined),
            _ = shutdown.cancelled(), if !shutdown_fired => {
                shutdown_fired = true;
                abort.store(true, Ordering::Relaxed);
            }
            _ = tokio::time::sleep_until(next_poll) => {
                let cur = progress.load(Ordering::Relaxed);
                if cur > last_progress {
                    // A window completed since the last check: reset the deadline.
                    last_progress = cur;
                    deadline = tokio::time::Instant::now() + timeout;
                } else if tokio::time::Instant::now() >= deadline {
                    abort.store(true, Ordering::Relaxed);
                    return WatchdogOutcome::TimedOut;
                }
            }
        }
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
mod tests {
    use super::*;

    fn channel_fixture(channels: u16, dual_mono: bool) -> Bytes {
        let frames = 1600u32;
        let data_bytes = frames * u32::from(channels) * 2;
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&channels.to_le_bytes());
        wav.extend_from_slice(&16000u32.to_le_bytes());
        wav.extend_from_slice(&(16000 * u32::from(channels) * 2).to_le_bytes());
        wav.extend_from_slice(&(channels * 2).to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_bytes.to_le_bytes());
        for frame in 0..frames {
            for channel in 0..channels {
                let frequency = if channel == 0 || dual_mono {
                    440.0
                } else {
                    710.0
                };
                let sample = ((frame as f32 * frequency * std::f32::consts::TAU / 16000.0).sin()
                    * 10000.0) as i16;
                wav.extend_from_slice(&sample.to_le_bytes());
            }
        }
        Bytes::from(wav)
    }

    #[test]
    fn test_cancelled_preparation_never_decodes_or_transcribes() {
        for split_channels in [false, true] {
            for raw_codec in [
                None,
                Some((gigastt_core::inference::audio::TelephonyCodec::Pcmu, 8000)),
            ] {
                let opts = FileTranscribeOpts {
                    split_channels,
                    raw_codec,
                    abort: Some(Arc::new(AtomicBool::new(true))),
                    ..Default::default()
                };
                let result = with_file_transcribe_request(Bytes::new(), &opts, true, |_| {
                    panic!("cancelled preparation reached inference")
                });
                assert!(matches!(result, Err::<(), _>(GigasttError::Cancelled)));
            }
        }
    }

    #[test]
    fn test_channel_routing_preserves_request_context() {
        for (fixture, channels, dual_mono) in [
            (channel_fixture(1, false), 1, false),
            (channel_fixture(2, true), 2, true),
            (channel_fixture(2, false), 2, false),
            (
                Bytes::from_static(include_bytes!(
                    "../../../gigastt-core/tests/fixtures/opus/dual.ogg"
                )),
                2,
                true,
            ),
            (
                Bytes::from_static(include_bytes!(
                    "../../../gigastt-core/tests/fixtures/opus/late_stereo.ogg"
                )),
                2,
                false,
            ),
        ] {
            for has_vad in [false, true] {
                for vad in [None, Some(false), Some(true)] {
                    for enabled in [false, true] {
                        for split_channels in [false, true] {
                            let opts = FileTranscribeOpts {
                                overrides: TranscribeOverrides {
                                    punctuation: Some(enabled),
                                    itn: Some(!enabled),
                                    vad,
                                },
                                hotwords: Some(HotwordOverride::new(
                                    vec!["тест".into()],
                                    Some(3.0),
                                )),
                                split_channels,
                                diarization: true,
                                abort: Some(Arc::new(AtomicBool::new(false))),
                                partial: Some(Arc::new(Default::default())),
                                progress: Some(Arc::new(AtomicU64::new(0))),
                                progress_channels: Some(Arc::new(AtomicU64::new(0))),
                                diarization_outcome: Some(Arc::new(Default::default())),
                                max_audio_secs: Some(3.0),
                                file_zero_context: false,
                                ..Default::default()
                            };
                            with_file_transcribe_request(
                                fixture.clone(),
                                &opts,
                                has_vad,
                                |request| {
                                    let split = split_channels && channels == 2 && !dual_mono;
                                    assert_eq!(
                                        opts.progress_channels
                                            .as_ref()
                                            .unwrap()
                                            .load(Ordering::Relaxed),
                                        if split { 2 } else { 1 },
                                    );
                                    match &request.source {
                                        TranscribeSource::Bytes(_) => assert!(!split),
                                        TranscribeSource::Channels(audio) => {
                                            assert!(split && has_vad && vad.unwrap_or(true));
                                            assert_eq!(audio.len(), 2);
                                        }
                                        TranscribeSource::ChannelStreams { channels, .. } => {
                                            assert!(split && !(has_vad && vad.unwrap_or(true)));
                                            assert_eq!(*channels, 2);
                                        }
                                        _ => panic!("unexpected source"),
                                    }
                                    assert_eq!(
                                        request.overrides.punctuation,
                                        opts.overrides.punctuation
                                    );
                                    assert_eq!(request.overrides.itn, opts.overrides.itn);
                                    assert_eq!(request.overrides.vad, opts.overrides.vad);
                                    assert!(std::ptr::eq(
                                        request.hotwords.unwrap(),
                                        opts.hotwords.as_ref().unwrap()
                                    ));
                                    assert!(Arc::ptr_eq(
                                        request.abort.as_ref().unwrap(),
                                        opts.abort.as_ref().unwrap()
                                    ));
                                    assert!(Arc::ptr_eq(
                                        request.partial.as_ref().unwrap(),
                                        opts.partial.as_ref().unwrap()
                                    ));
                                    assert!(Arc::ptr_eq(
                                        request.progress.as_ref().unwrap(),
                                        opts.progress.as_ref().unwrap()
                                    ));
                                    assert_eq!(request.max_audio_secs, opts.max_audio_secs);
                                    // Channel speaker labels take precedence over offline diarization,
                                    // including mono fallback, as on the existing split path.
                                    assert_eq!(request.diarization, !split_channels);
                                    if split_channels {
                                        assert!(request.diarization_outcome.is_none());
                                    } else {
                                        assert!(Arc::ptr_eq(
                                            request.diarization_outcome.as_ref().unwrap(),
                                            opts.diarization_outcome.as_ref().unwrap()
                                        ));
                                    }
                                    Ok(())
                                },
                            )
                            .unwrap();
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_raw_pcm_routing_preserves_legacy_samples_and_request_context() {
        use gigastt_core::inference::audio::{
            TelephonyCodec, decode_audio_bytes, decode_telephony_raw, encode_wav_pcm16,
        };
        let body: Bytes = (0..8000).map(|i| i as u8).collect::<Vec<_>>().into();
        for codec in [
            TelephonyCodec::Pcmu,
            TelephonyCodec::Pcma,
            TelephonyCodec::G722,
        ] {
            let decoded = decode_telephony_raw(&body, codec, 8000).unwrap();
            let expected = decode_audio_bytes(&encode_wav_pcm16(&decoded, 16000)).unwrap();
            for split_channels in [false, true] {
                let opts = FileTranscribeOpts {
                    raw_codec: Some((codec, 8000)),
                    split_channels,
                    diarization: true,
                    overrides: TranscribeOverrides {
                        punctuation: Some(false),
                        itn: Some(true),
                        vad: Some(true),
                    },
                    hotwords: Some(HotwordOverride::new(vec!["тест".into()], Some(3.0))),
                    abort: Some(Arc::new(AtomicBool::new(false))),
                    partial: Some(Arc::new(Default::default())),
                    progress: Some(Arc::new(AtomicU64::new(17))),
                    progress_channels: Some(Arc::new(AtomicU64::new(0))),
                    diarization_outcome: Some(Arc::new(Default::default())),
                    max_audio_secs: Some(2.0),
                    file_zero_context: false,
                };
                with_file_transcribe_request(body.clone(), &opts, true, |request| {
                    let TranscribeSource::Samples(samples) = request.source else {
                        panic!("raw audio should bypass the WAV container");
                    };
                    assert_eq!(
                        samples.iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|s| s.to_bits()).collect::<Vec<_>>()
                    );
                    assert_eq!(request.overrides.punctuation, Some(false));
                    assert_eq!(request.overrides.itn, Some(true));
                    assert_eq!(request.overrides.vad, Some(true));
                    assert!(std::ptr::eq(
                        request.hotwords.unwrap(),
                        opts.hotwords.as_ref().unwrap()
                    ));
                    assert!(Arc::ptr_eq(
                        request.abort.as_ref().unwrap(),
                        opts.abort.as_ref().unwrap()
                    ));
                    assert!(Arc::ptr_eq(
                        request.partial.as_ref().unwrap(),
                        opts.partial.as_ref().unwrap()
                    ));
                    assert!(Arc::ptr_eq(
                        request.progress.as_ref().unwrap(),
                        opts.progress.as_ref().unwrap()
                    ));
                    assert_eq!(request.max_audio_secs, opts.max_audio_secs);
                    assert_eq!(request.diarization, !split_channels);
                    assert_eq!(request.diarization_outcome.is_some(), !split_channels);
                    assert_eq!(
                        opts.progress_channels
                            .as_ref()
                            .unwrap()
                            .load(Ordering::Relaxed),
                        1
                    );
                    Ok(())
                })
                .unwrap();
            }
        }
    }

    #[test]
    fn test_raw_routing_threshold_and_one_sample_over_preserve_precision() {
        use gigastt_core::inference::audio::{
            TelephonyCodec, decode_audio_bytes, decode_telephony_raw, encode_wav_pcm16,
        };
        for samples in [RAW_PCM_MAX_SAMPLES, RAW_PCM_MAX_SAMPLES + 1] {
            // G.711 at 16 kHz decodes one byte to exactly one output sample.
            let body: Bytes = (0..samples).map(|i| i as u8).collect::<Vec<_>>().into();
            let expected = encode_wav_pcm16(
                &decode_telephony_raw(&body, TelephonyCodec::Pcma, 16000).unwrap(),
                16000,
            );
            let opts = FileTranscribeOpts {
                raw_codec: Some((TelephonyCodec::Pcma, 16000)),
                ..Default::default()
            };
            with_file_transcribe_request(body, &opts, false, |request| {
                match request.source {
                    TranscribeSource::Samples(pcm) => {
                        assert_eq!(samples, RAW_PCM_MAX_SAMPLES);
                        assert_eq!(pcm, decode_audio_bytes(&expected).unwrap());
                    }
                    TranscribeSource::Bytes(wav) => {
                        assert_eq!(samples, RAW_PCM_MAX_SAMPLES + 1);
                        assert_eq!(wav.as_ref(), expected);
                    }
                    _ => panic!("unexpected raw source"),
                }
                Ok(())
            })
            .unwrap();
        }
    }

    #[test]
    fn test_raw_preparation_checks_cancellation_after_both_representations() {
        use gigastt_core::inference::audio::TelephonyCodec;
        use std::sync::atomic::AtomicUsize;
        for samples in [160, RAW_PCM_MAX_SAMPLES + 1] {
            let body = vec![0xff; samples];
            let checks = AtomicUsize::new(0);
            prepare_raw_audio(
                &body,
                TelephonyCodec::Pcmu,
                16000,
                None,
                Some(&|| {
                    checks.fetch_add(1, Ordering::Relaxed);
                    false
                }),
            )
            .unwrap();
            let final_check = checks.load(Ordering::Relaxed) - 1;
            checks.store(0, Ordering::Relaxed);
            let result = prepare_raw_audio(
                &body,
                TelephonyCodec::Pcmu,
                16000,
                None,
                Some(&|| checks.fetch_add(1, Ordering::Relaxed) == final_check),
            );
            assert!(matches!(result, Err(GigasttError::Cancelled)));
        }
    }

    #[test]
    fn test_raw_pcm_duration_budget_rejects_before_engine_routing() {
        use gigastt_core::inference::audio::TelephonyCodec;
        for codec in [
            TelephonyCodec::Pcmu,
            TelephonyCodec::Pcma,
            TelephonyCodec::G722,
        ] {
            let opts = FileTranscribeOpts {
                raw_codec: Some((codec, 8000)),
                max_audio_secs: Some(0.01),
                ..Default::default()
            };
            let result =
                with_file_transcribe_request(Bytes::from(vec![0xff; 8000]), &opts, false, |_| {
                    Ok(())
                });
            assert!(matches!(result, Err(GigasttError::AudioTooLong { .. })));
        }
    }

    #[test]
    fn test_file_transcribe_opts_default() {
        let opts = FileTranscribeOpts::default();
        assert!(!opts.split_channels);
        assert!(!opts.diarization);
        assert!(opts.hotwords.is_none());
        assert!(opts.raw_codec.is_none());
        assert!(opts.overrides.punctuation.is_none());
        assert!(opts.overrides.itn.is_none());
        assert!(opts.overrides.vad.is_none());
    }

    #[test]
    fn test_prepare_raw_audio_rejects_empty_pcmu() {
        // Empty PCMU body is invalid for telephony decode.
        let err = prepare_raw_audio(
            &[],
            gigastt_core::inference::audio::TelephonyCodec::Pcmu,
            8000,
            None,
            None,
        )
        .unwrap_err();
        match err {
            GigasttError::InvalidAudio { .. } => {}
            other => panic!("expected InvalidAudio, got {other:?}"),
        }
    }

    #[test]
    fn test_abort_on_drop_sets_flag() {
        let flag = Arc::new(AtomicBool::new(false));
        {
            let _guard = AbortOnDrop(flag.clone());
            assert!(!flag.load(Ordering::Relaxed), "flag is clear while held");
        }
        assert!(
            flag.load(Ordering::Relaxed),
            "drop must flip the abort flag"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn test_watchdog_times_out_and_aborts_without_progress() {
        // A run that never advances `progress` past the timeout must trip and
        // flip `abort` (so the real decode would cancel at its next window).
        let progress = Arc::new(AtomicU64::new(0));
        let abort = Arc::new(AtomicBool::new(false));
        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = tokio::task::spawn(async {
            tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
            0u32
        });
        let start = tokio::time::Instant::now();
        let outcome = await_transcription_watchdog(handle, &progress, &abort, 1, &shutdown).await;
        assert!(matches!(outcome, WatchdogOutcome::TimedOut));
        assert_eq!(start.elapsed(), std::time::Duration::from_secs(1));
        assert!(abort.load(Ordering::Relaxed), "a trip must flip abort");
    }

    #[tokio::test(start_paused = true)]
    async fn test_watchdog_resets_on_progress_and_joins() {
        // A run that reports a window every 0.5 s never exhausts the 1 s
        // no-progress budget, so it joins normally and is not aborted.
        let progress = Arc::new(AtomicU64::new(0));
        let abort = Arc::new(AtomicBool::new(false));
        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = tokio::task::spawn({
            let progress = progress.clone();
            async move {
                for window in 1..=4u64 {
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                    progress.store(window * 16_000, Ordering::Relaxed);
                }
                42u32
            }
        });
        let outcome = await_transcription_watchdog(handle, &progress, &abort, 1, &shutdown).await;
        match outcome {
            WatchdogOutcome::Joined(Ok(v)) => assert_eq!(v, 42),
            other => panic!("expected Joined(Ok(42)), got {}", label(&other)),
        }
        assert!(
            !abort.load(Ordering::Relaxed),
            "a steadily-progressing run must not be aborted"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn test_watchdog_disabled_never_times_out() {
        // `timeout_secs == 0` disables the watchdog: even a long, silent run
        // joins without tripping and without abort.
        let progress = Arc::new(AtomicU64::new(0));
        let abort = Arc::new(AtomicBool::new(false));
        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = tokio::task::spawn(async {
            tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
            5u32
        });
        let outcome = await_transcription_watchdog(handle, &progress, &abort, 0, &shutdown).await;
        assert!(matches!(outcome, WatchdogOutcome::Joined(Ok(5))));
        assert!(!abort.load(Ordering::Relaxed));
    }

    #[tokio::test(start_paused = true)]
    async fn test_watchdog_shutdown_flips_abort() {
        // A fired shutdown must flip `abort` so the run cancels at its next
        // window; the watchdog then joins the (now-cancelled) task.
        let progress = Arc::new(AtomicU64::new(0));
        let abort = Arc::new(AtomicBool::new(false));
        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = tokio::task::spawn({
            let abort = abort.clone();
            async move {
                loop {
                    if abort.load(Ordering::Relaxed) {
                        break 9u32;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                }
            }
        });
        shutdown.cancel();
        // A generous timeout that must not be what ends the run.
        let outcome = await_transcription_watchdog(handle, &progress, &abort, 600, &shutdown).await;
        assert!(matches!(outcome, WatchdogOutcome::Joined(Ok(9))));
        assert!(abort.load(Ordering::Relaxed), "shutdown must flip abort");
    }

    #[tokio::test(start_paused = true)]
    async fn test_watchdog_stall_detection_is_bounded_after_early_progress() {
        let progress = Arc::new(AtomicU64::new(0));
        let abort = AtomicBool::new(false);
        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = tokio::spawn({
            let progress = progress.clone();
            async move {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                progress.store(1, Ordering::Relaxed);
                tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
            }
        });
        let start = tokio::time::Instant::now();
        let outcome = await_transcription_watchdog(handle, &progress, &abort, 1, &shutdown).await;
        assert!(matches!(outcome, WatchdogOutcome::TimedOut));
        let elapsed = start.elapsed();
        assert!(
            elapsed >= std::time::Duration::from_millis(1010),
            "{elapsed:?}"
        );
        assert!(
            elapsed <= std::time::Duration::from_millis(1110),
            "{elapsed:?}"
        );
        assert!(abort.load(Ordering::Relaxed));
    }

    #[tokio::test(start_paused = true)]
    async fn test_watchdog_channel_transition_preserves_liveness() {
        let progress = Arc::new(AtomicU64::new(0));
        let abort = AtomicBool::new(false);
        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = tokio::spawn({
            let progress = progress.clone();
            async move {
                for channel in 0..2 {
                    for local_samples in 1..=3 {
                        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                        progress.store(channel * 3 + local_samples, Ordering::Relaxed);
                    }
                }
            }
        });
        let start = tokio::time::Instant::now();
        let result = await_transcription_watchdog(handle, &progress, &abort, 1, &shutdown).await;
        assert!(matches!(result, WatchdogOutcome::Joined(Ok(()))));
        assert_eq!(start.elapsed(), std::time::Duration::from_secs(3));
        assert!(!abort.load(Ordering::Relaxed));
    }

    #[tokio::test(start_paused = true)]
    async fn test_watchdog_disabled_still_cancels_on_shutdown() {
        let progress = AtomicU64::new(0);
        let abort = Arc::new(AtomicBool::new(false));
        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = tokio::spawn({
            let abort = abort.clone();
            let shutdown = shutdown.clone();
            async move {
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                shutdown.cancel();
                while !abort.load(Ordering::Relaxed) {
                    tokio::task::yield_now().await;
                }
            }
        });
        let start = tokio::time::Instant::now();
        let result = await_transcription_watchdog(handle, &progress, &abort, 0, &shutdown).await;
        assert!(matches!(result, WatchdogOutcome::Joined(Ok(()))));
        assert_eq!(start.elapsed(), std::time::Duration::from_secs(5));
        assert!(abort.load(Ordering::Relaxed));
    }

    /// Human-readable tag for a `WatchdogOutcome` in test panics.
    fn label<T>(outcome: &WatchdogOutcome<T>) -> &'static str {
        match outcome {
            WatchdogOutcome::Joined(Ok(_)) => "Joined(Ok)",
            WatchdogOutcome::Joined(Err(_)) => "Joined(Err)",
            WatchdogOutcome::TimedOut => "TimedOut",
        }
    }

    #[test]
    fn test_file_zero_context_routes_default_off_and_opt_in_to_native_request() {
        for enabled in [false, true] {
            let opts = FileTranscribeOpts {
                file_zero_context: enabled,
                ..Default::default()
            };
            with_file_transcribe_request(Bytes::from_static(b"file"), &opts, false, |request| {
                assert!(matches!(request.source, TranscribeSource::Bytes(_)));
                assert_eq!(request.file_zero_context, enabled);
                Ok(())
            })
            .unwrap();
        }
    }
}
