//! Job queue, workers, and shared helpers (broadcast / sanitize / retry).

use axum::body::Bytes;
use std::sync::Arc;

use super::super::http::ExportParams;
use super::store::{JobEvent, JobStatus, JobStore, JobTransition, TransitionOutcome};

pub trait JobExecution: Send + Sync {
    /// Run one transcription attempt. The executor may update progress via the
    /// store and should return an error for retryable failures.
    fn execute(
        &self,
        id: &str,
        store: Arc<dyn JobStore>,
        body: Bytes,
        params: ExportParams,
    ) -> impl std::future::Future<Output = anyhow::Result<gigastt_core::inference::TranscribeResult>>
    + Send;
}

/// In-memory FIFO job queue. Spawns `concurrency` workers that pull from the
/// store, run the executor, and retry up to `max_retries` on transient failures.
pub struct JobQueue {
    store: Arc<dyn JobStore>,
    semaphore: Arc<tokio::sync::Semaphore>,
    max_retries: u32,
    shutdown: tokio_util::sync::CancellationToken,
}

impl JobQueue {
    /// Create a new queue. `concurrency` is clamped to at least 1.
    pub fn new(
        store: Arc<dyn JobStore>,
        concurrency: usize,
        max_retries: u32,
        shutdown: tokio_util::sync::CancellationToken,
    ) -> Arc<Self> {
        Arc::new(Self {
            store,
            semaphore: Arc::new(tokio::sync::Semaphore::new(concurrency.max(1))),
            max_retries,
            shutdown,
        })
    }

    /// Spawn worker tasks onto the server drain [`tokio_util::task::TaskTracker`] so graceful
    /// shutdown waits for in-flight jobs (same lane as WS / SSE / REST).
    /// Each worker owns a clone of `executor`. Call once after constructing
    /// the queue.
    pub fn spawn<E>(&self, executor: E, tracker: &tokio_util::task::TaskTracker)
    where
        E: JobExecution + Clone + Send + Sync + 'static,
    {
        let permits = self.semaphore.available_permits();
        for _ in 0..permits {
            let worker = JobWorker {
                store: self.store.clone(),
                semaphore: self.semaphore.clone(),
                max_retries: self.max_retries,
                shutdown: self.shutdown.clone(),
                executor: executor.clone(),
            };
            tracker.spawn(worker.run());
        }
    }

    /// Mark all queued jobs as cancelled. Called during graceful shutdown.
    pub async fn cancel_all_queued(&self) {
        loop {
            let Ok(Some(id)) = self.store.next_queued().await else {
                break;
            };
            let _ = self.store.transition(&id, JobTransition::Cancel).await;
        }
    }
}

struct JobWorker<E> {
    store: Arc<dyn JobStore>,
    semaphore: Arc<tokio::sync::Semaphore>,
    max_retries: u32,
    shutdown: tokio_util::sync::CancellationToken,
    executor: E,
}

impl<E: JobExecution + 'static> JobWorker<E> {
    async fn run(self) {
        loop {
            if self.shutdown.is_cancelled() {
                break;
            }
            let permit = match tokio::time::timeout(
                std::time::Duration::from_secs(1),
                self.semaphore.clone().acquire_owned(),
            )
            .await
            {
                Ok(Ok(p)) => p,
                _ => continue,
            };
            let Some(id) = self.store.next_queued().await.unwrap_or(None) else {
                drop(permit);
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                continue;
            };

            if !matches!(
                self.store.transition(&id, JobTransition::Start).await,
                Ok(TransitionOutcome::Applied)
            ) {
                drop(permit);
                continue;
            }

            let store = self.store.clone();
            let (body, params) = match self.store.get(&id).await {
                Ok(Some(job)) if job.status == JobStatus::Processing => (job.body, job.params),
                _ => {
                    drop(permit);
                    continue;
                }
            };

            let result = self
                .executor
                .execute(&id, store.clone(), body, params)
                .await;
            drop(permit);

            match result {
                Ok(res) => {
                    let _ = self
                        .store
                        .transition(&id, JobTransition::Complete(res))
                        .await;
                }
                Err(e) => {
                    let attempts = self
                        .store
                        .get(&id)
                        .await
                        .ok()
                        .flatten()
                        .map(|j| j.attempts)
                        .unwrap_or(0);
                    let transition = if is_retryable_error(&e) && attempts <= self.max_retries {
                        JobTransition::Retry
                    } else {
                        JobTransition::Fail(sanitize_job_error(&e))
                    };
                    let _ = self.store.transition(&id, transition).await;
                }
            }
        }
    }
}

/// Send an event to all active listeners, pruning dead channels. Terminal
/// events are fire-and-forget: the channel is dropped afterwards so the SSE
/// stream ends naturally.
pub(crate) async fn broadcast_event(store: &dyn JobStore, id: &str, event: JobEvent) {
    let _ = store
        .update(id, Box::new(move |job| job.broadcast(event)))
        .await;
}

pub(crate) fn is_retryable_error(e: &anyhow::Error) -> bool {
    // Only a panic in the inference thread is transient: the triplet is
    // recovered via catch_unwind and the same input may well succeed on a fresh
    // worker, so it is worth another attempt.
    //
    // An `inference_timeout` is deliberately NOT retryable. A file that is too
    // slow for the limit will be exactly as slow next time, so retrying it just
    // burns (max_retries + 1) × the timeout on a job that can never pass while
    // the rest of the queue waits behind it. Fail it once instead.
    let s = format!("{e:#}");
    s.contains("panicked")
}

pub(crate) fn sanitize_job_error(e: &anyhow::Error) -> String {
    // Preserve the typed length rejection so a client can tell "too long" (raise
    // `--max-audio-secs` or split) from "corrupt" — the same distinction the REST
    // path answers with 413. The blocking result reaches us via
    // `anyhow::Error::from`, so the concrete variant survives the downcast.
    if let Some(err) = e.downcast_ref::<gigastt_core::error::GigasttError>() {
        match err {
            gigastt_core::error::GigasttError::Cancelled => {
                return "Transcription cancelled.".into();
            }
            gigastt_core::error::GigasttError::AudioTooLong {
                observed_secs,
                limit_secs,
            } => {
                return format!(
                    "Audio too long: {observed_secs:.0}s exceeds the maximum of {limit_secs:.0}s."
                );
            }
            gigastt_core::error::GigasttError::InvalidInput { message }
                if message == super::executor::FILE_ZERO_CONTEXT_PARAMS_ERROR =>
            {
                return super::executor::FILE_ZERO_CONTEXT_PARAMS_ERROR.into();
            }
            gigastt_core::error::GigasttError::InvalidAudio { .. } => {
                return "Failed to decode audio file. Check format.".into();
            }
            _ => {}
        }
    }
    let msg = format!("{e:#}");
    let lower = msg.to_ascii_lowercase();
    if lower.contains("inference_timeout") {
        "Inference timed out.".into()
    } else if lower.contains("invalid audio") {
        "Failed to decode audio file. Check format.".into()
    } else {
        "Transcription failed.".into()
    }
}
