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
    /// When set, decode the body as a raw telephony stream and re-wrap as WAV
    /// before the engine path. REST-only today; jobs do not expose `?codec=`.
    pub raw_codec: Option<(gigastt_core::inference::audio::TelephonyCodec, u32)>,
    /// Cooperative-cancellation flag threaded into the engine's per-window
    /// decode loop. Flipping it (client disconnect, `DELETE /v1/jobs/{id}`,
    /// shutdown, or the no-progress watchdog) ends the run at the next window.
    pub abort: Option<Arc<AtomicBool>>,
    /// Progress sink the engine advances after each window with the cumulative
    /// count of processed 16 kHz samples. The server watchdog reads it to reset
    /// its no-progress deadline and to drive a real job progress bar.
    pub progress: Option<Arc<AtomicU64>>,
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
}

/// Sets a shared abort flag when dropped. Held in the REST handler's async
/// scope so that a client disconnect — which drops the handler future before it
/// returns — flips the flag, and the detached blocking decode observes it at the
/// next window boundary and releases its pooled triplet. On the normal return
/// path the flag is set after the result is already in hand, so it is a no-op.
pub(crate) struct AbortOnDrop(pub Arc<AtomicBool>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// Outcome of awaiting a blocking transcription under the no-progress watchdog.
pub(crate) enum WatchdogOutcome<T> {
    /// The blocking task finished; carries the raw join result.
    Joined(Result<T, tokio::task::JoinError>),
    /// No window completed within the inference timeout. `abort` was set so the
    /// run cancels at its next window boundary and frees the triplet.
    TimedOut,
}

/// Await a blocking transcription `handle`, redefining `inference_timeout_secs`
/// from a total wall-clock cap into a **no-progress watchdog**: the deadline
/// resets every time `progress` advances (i.e. a window completes), so a long
/// file that keeps making progress never trips, while a genuinely stalled run
/// trips at the same moment it always did. `timeout_secs == 0` disables the
/// watchdog. A fired `shutdown` also flips `abort`, so SIGTERM cancels the run
/// at its next window instead of blocking the drain for the whole file.
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
    loop {
        tokio::select! {
            joined = &mut handle => return WatchdogOutcome::Joined(joined),
            _ = shutdown.cancelled(), if !shutdown_fired => {
                shutdown_fired = true;
                abort.store(true, Ordering::Relaxed);
            }
            _ = tokio::time::sleep_until(deadline) => {
                let cur = progress.load(Ordering::Relaxed);
                if cur > last_progress {
                    // A window completed since the last check: reset the deadline.
                    last_progress = cur;
                    deadline = tokio::time::Instant::now() + timeout;
                } else {
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

/// Decode raw telephony bytes to an in-memory PCM16 WAV for engine paths.
pub(crate) fn raw_codec_to_wav(
    body: &[u8],
    codec: gigastt_core::inference::audio::TelephonyCodec,
    sample_rate: u32,
) -> Result<Bytes, GigasttError> {
    let samples = gigastt_core::inference::audio::decode_telephony_raw(body, codec, sample_rate)
        .map_err(map_decode_error)?;
    Ok(Bytes::from(
        gigastt_core::inference::audio::encode_wav_pcm16(&samples, 16000),
    ))
}

/// Blocking file transcription against a reserved triplet.
///
/// Handles raw-codec rewrap, `channels=split` with mono fallback, diarization,
/// and the default mono path. Callers own pool checkout, panic wrapping, and
/// timeout policy.
pub(crate) fn run_file_transcribe_blocking(
    engine: &Engine,
    body: Bytes,
    reservation: &mut OwnedReservation<SessionTriplet>,
    opts: &FileTranscribeOpts,
) -> Result<TranscribeResult, GigasttError> {
    let body = match opts.raw_codec {
        Some((codec, rate)) => raw_codec_to_wav(&body, codec, rate)?,
        None => body,
    };

    if opts.split_channels {
        let channels = gigastt_core::inference::audio::decode_audio_bytes_shared_channels_bounded(
            body.clone(),
            opts.max_audio_secs,
        )
        .map_err(map_decode_error)?;
        let fallback_reason = match channels.len() {
            0 => Some("no channels"),
            1 => Some("mono audio"),
            2 if gigastt_core::inference::audio::is_dual_mono(&channels) => Some("dual-mono audio"),
            n if n > 2 => Some("more than two channels"),
            _ => None,
        };
        if let Some(reason) = fallback_reason {
            // The mono path re-decodes from `body`; release the split channels
            // first so both copies are never resident at once.
            drop(channels);
            tracing::warn!(
                "channels=split requested but {reason} detected; falling back to mono transcription"
            );
            engine.transcribe_request(
                TranscribeRequest::new(TranscribeSource::Bytes(body))
                    .with_abort(opts.abort.clone())
                    .with_progress(opts.progress.clone())
                    .with_max_audio_secs(opts.max_audio_secs),
                reservation,
            )
        } else {
            engine.transcribe_request(
                TranscribeRequest::new(TranscribeSource::Channels(&channels))
                    .with_abort(opts.abort.clone())
                    .with_max_audio_secs(opts.max_audio_secs),
                reservation,
            )
        }
    } else {
        // Mono path (optional diarization) via unified Engine request API.
        engine.transcribe_request(
            TranscribeRequest::new(TranscribeSource::Bytes(body))
                .with_overrides(opts.overrides)
                .with_hotwords(opts.hotwords.as_ref())
                .with_diarization(opts.diarization)
                .with_diarization_outcome(opts.diarization_outcome.clone())
                .with_abort(opts.abort.clone())
                .with_progress(opts.progress.clone())
                .with_max_audio_secs(opts.max_audio_secs),
            reservation,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_raw_codec_to_wav_rejects_empty_pcmu() {
        // Empty PCMU body is invalid for telephony decode.
        let err = raw_codec_to_wav(
            &[],
            gigastt_core::inference::audio::TelephonyCodec::Pcmu,
            8000,
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
        let outcome = await_transcription_watchdog(handle, &progress, &abort, 1, &shutdown).await;
        assert!(matches!(outcome, WatchdogOutcome::TimedOut));
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

    /// Human-readable tag for a `WatchdogOutcome` in test panics.
    fn label<T>(outcome: &WatchdogOutcome<T>) -> &'static str {
        match outcome {
            WatchdogOutcome::Joined(Ok(_)) => "Joined(Ok)",
            WatchdogOutcome::Joined(Err(_)) => "Joined(Err)",
            WatchdogOutcome::TimedOut => "TimedOut",
        }
    }
}
