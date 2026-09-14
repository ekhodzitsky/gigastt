//! Production job executor (decode + inference + progress).

use arc_swap::ArcSwap;
use axum::body::Bytes;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::super::config::RuntimeLimits;
use super::super::http::ExportParams;
use super::queue::{JobExecution, broadcast_event};
use super::store::{JobEvent, JobStatus, JobStore};

/// Production executor: decodes audio to size the job, runs inference against
/// the batch pool, and emits timer-based progress events.
#[derive(Clone)]
pub struct RealJobExecutor {
    engine: Arc<ArcSwap<gigastt_core::inference::Engine>>,
    limits: Arc<ArcSwap<RuntimeLimits>>,
    /// Server shutdown signal. A fired token flips the per-run abort flag so an
    /// in-flight job releases its triplet at the next window during a drain.
    shutdown: tokio_util::sync::CancellationToken,
}

impl RealJobExecutor {
    /// Create an executor bound to the live engine, runtime limits, and the
    /// server shutdown token.
    pub fn new(
        engine: Arc<ArcSwap<gigastt_core::inference::Engine>>,
        limits: Arc<ArcSwap<RuntimeLimits>>,
        shutdown: tokio_util::sync::CancellationToken,
    ) -> Self {
        Self {
            engine,
            limits,
            shutdown,
        }
    }
}

impl JobExecution for RealJobExecutor {
    async fn execute(
        &self,
        id: &str,
        store: Arc<dyn JobStore>,
        body: Bytes,
        params: ExportParams,
    ) -> anyhow::Result<gigastt_core::inference::TranscribeResult> {
        let engine = self.engine.load_full();
        let limits = self.limits.load();

        // Size the progress bar from the container header when it declares a
        // duration (WAV / FLAC / M4A / OGG do). A probe reads O(header) bytes
        // and decodes nothing, so we skip the full O(T) decode this used to run
        // purely to throw the samples away — the engine re-decodes `body` for
        // the actual transcript below. That eliminates one wasted full decode
        // per job: its wall time and its transient f32 buffer (measured ~0.9 s
        // and ~73 MB on a 20-min 48 kHz file). Peak RSS is roughly unchanged —
        // the old pre-decode buffer was already freed before the engine decode,
        // so the two never coexisted — but the redundant CPU pass is gone.
        const TARGET_SAMPLE_RATE: f64 = 16_000.0;
        let probed = tokio::task::spawn_blocking({
            let body = body.clone();
            move || gigastt_core::inference::audio::probe_duration_bytes(body)
        })
        .await
        .map_err(|e| anyhow::anyhow!("audio probe task panicked: {e}"))?
        .ok()
        .flatten();

        // Unknown duration (raw MP3, headerless streams): do **not** expand
        // the whole file to f32 just to size a progress bar. That used to run
        // before pool checkout and could OOM a `--enable-jobs` host on a
        // compressed upload. Percent stays 0 until the job finishes; the
        // engine still decodes after the slot is reserved.
        let total_seconds = probed.unwrap_or(0.0);
        let _ = store
            .update(id, Box::new(move |j| j.total_seconds = total_seconds))
            .await;

        // Validate per-request variant / knob overrides before holding a triplet.
        if let Some(requested) = params.variant.as_deref() {
            let matches = gigastt_core::model::ModelVariant::from_str(requested)
                .map(|v| v == engine.variant())
                .unwrap_or(false);
            if !matches {
                return Err(anyhow::anyhow!("Requested model variant is not loaded"));
            }
        }
        let overrides = super::super::http::overrides_from_export_params(&params);
        if let Err(e) = engine.validate_overrides(&overrides) {
            return Err(anyhow::anyhow!("Invalid input: {}", e.message()));
        }
        let hotwords = super::super::http::hotwords_from_export_params(&params);
        if let Some(ref hw) = hotwords
            && let Err(e) = engine.validate_hotwords(hw)
        {
            return Err(anyhow::anyhow!("Invalid input: {}", e.message()));
        }

        // Cooperative cancellation + real progress. `abort` is flipped by
        // `DELETE /v1/jobs/{id}` or a shutdown drain; `progress` carries the
        // engine's cumulative processed 16 kHz sample count out of the blocking
        // decode. Register `abort` on the job so the cancel handler can reach it.
        let abort = Arc::new(AtomicBool::new(false));
        let partial = Arc::new(gigastt_core::inference::TranscriptSnapshot::default());
        let progress = Arc::new(AtomicU64::new(0));
        let _ = store
            .update(id, {
                let abort = abort.clone();
                let partial = partial.clone();
                Box::new(move |j| {
                    // Honour a cancel that raced in between the worker marking
                    // this job Processing and this registration: seed the flag
                    // from the current status so the run still stops at its first
                    // window instead of ignoring the already-recorded cancel.
                    if matches!(j.status, JobStatus::Cancelled) {
                        abort.store(true, Ordering::Relaxed);
                    }
                    j.abort = Some(abort);
                    j.partial = Some(partial);
                })
            })
            .await;
        let shutdown = self.shutdown.clone();

        // Real per-window progress updater: mirrors the engine's processed-sample
        // count into the store and SSE stream every 500 ms. No RTF guess — the
        // bar tracks audio actually decoded, monotonically. Same interval and
        // `JobEvent` shape as before, so the SSE wire contract is unchanged.
        let progress_cancel = tokio_util::sync::CancellationToken::new();
        let progress_handle = {
            let store = store.clone();
            let id = id.to_string();
            let cancel = progress_cancel.clone();
            let total = total_seconds;
            let progress = progress.clone();
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(std::time::Duration::from_millis(500));
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                loop {
                    interval.tick().await;
                    if cancel.is_cancelled() {
                        break;
                    }
                    let processed =
                        (progress.load(Ordering::Relaxed) as f64 / TARGET_SAMPLE_RATE).min(total);
                    let percent = if total > 0.0 {
                        ((processed / total) * 100.0) as u32
                    } else {
                        0
                    };
                    let _ = store
                        .update(&id, Box::new(move |j| j.processed_seconds = processed))
                        .await;
                    broadcast_event(
                        &*store,
                        &id,
                        JobEvent::Progress {
                            percent,
                            processed_seconds: processed,
                        },
                    )
                    .await;
                }
            })
        };

        // Write-once sink for *why* speakers were or were not labeled. Armed
        // only when diarization was requested, matching the synchronous path:
        // an unarmed sink makes the engine record nothing.
        let diar_sink: Option<
            Arc<std::sync::OnceLock<gigastt_core::inference::DiarizationOutcome>>,
        > = (params.diarization == Some(true)).then(|| Arc::new(std::sync::OnceLock::new()));

        // Check out a triplet from the batch pool and run inference.
        // This is wrapped in its own async block so the progress updater is
        // always cancelled and awaited before the function returns, even on
        // the early-return error paths below.
        let inference_result: anyhow::Result<gigastt_core::inference::TranscribeResult> = async {
            let guard = tokio::time::timeout(
                std::time::Duration::from_secs(limits.pool_checkout_timeout_secs),
                engine.pool_for_batch().checkout(),
            )
            .await
            .map_err(|_| anyhow::anyhow!("pool checkout timed out"))?
            .map_err(|_| anyhow::anyhow!("pool closed"))?;
            let mut reservation = guard.into_owned();

            let inference_timeout_secs = limits.inference_timeout_secs;
            let engine_for_inference = engine.clone();
            let body_for_inference = body.clone();
            let span = tracing::Span::current();
            let file_opts = super::super::file_transcribe::FileTranscribeOpts {
                overrides,
                hotwords,
                split_channels: params.channels.as_deref() == Some("split"),
                diarization: params.diarization == Some(true),
                raw_codec: None,
                abort: Some(abort.clone()),
                partial: Some(partial.clone()),
                progress: Some(progress.clone()),
                // Same sink the synchronous endpoint uses, so an async job that
                // produces no speaker labels can say why instead of returning a
                // transcript with silently empty speaker fields. Only armed when
                // diarization was actually requested; otherwise the engine
                // records nothing and the response shape is untouched.
                diarization_outcome: diar_sink.clone(),
                max_audio_secs: limits.max_audio_secs_opt(),
            };
            let handle = tokio::task::spawn_blocking(move || {
                let _enter = span.enter();
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    super::super::file_transcribe::run_file_transcribe_blocking(
                        &engine_for_inference,
                        body_for_inference,
                        &mut reservation,
                        &file_opts,
                    )
                }));
                match r {
                    Ok(result) => result,
                    Err(_) => Err(gigastt_core::error::GigasttError::Inference {
                        source: anyhow::anyhow!("Inference thread panicked").into(),
                    }),
                }
            });

            // The inference timeout is a no-progress watchdog (shared with the
            // REST path): the deadline resets on each completed window, so a long
            // file making steady progress never trips, while a genuinely stalled
            // run still fails with the deterministic `inference_timeout` marker.
            // A fired shutdown also flips `abort`, freeing the triplet mid-drain.
            match super::super::file_transcribe::await_transcription_watchdog(
                handle,
                &progress,
                &abort,
                inference_timeout_secs,
                &shutdown,
            )
            .await
            {
                super::super::file_transcribe::WatchdogOutcome::Joined(r) => r
                    .map_err(|e| anyhow::anyhow!("spawn_blocking join error: {e}"))?
                    .map_err(anyhow::Error::from),
                super::super::file_transcribe::WatchdogOutcome::TimedOut => {
                    Err(anyhow::anyhow!("inference_timeout"))
                }
            }
        }
        .await;

        progress_cancel.cancel();
        let _ = progress_handle.await;

        // Record the outcome on the job before returning: the worker stores the
        // transcript after this call, and `GET /v1/jobs/{id}/result` reads both
        // together. Written even on a failed run — the sink is empty then, so
        // this is a no-op rather than a stale value.
        if let Some(outcome) = diar_sink.as_ref().and_then(|s| s.get().copied()) {
            let _ = store
                .update(id, Box::new(move |j| j.diarization = Some(outcome)))
                .await;
        }

        inference_result
    }
}
