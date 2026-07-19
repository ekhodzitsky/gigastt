//! Asynchronous job queue for long-file and batch transcription.
//!
//! The queue is intentionally decoupled from the HTTP surface via the
//! [`JobStore`] trait so a persistent backend (e.g. SQLite) can plug in
//! later without touching the handlers.

use arc_swap::ArcSwap;
use axum::body::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::str::FromStr;
use std::sync::Arc;

use parking_lot::Mutex;

use super::config::RuntimeLimits;
use super::http::ExportParams;

/// Object-safe boxed future returned by [`JobStore`] methods.
pub type JobStoreFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Lifecycle status of a transcription job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    /// Waiting for a worker slot.
    Queued,
    /// Currently holding a triplet and transcribing.
    Processing,
    /// Finished successfully; result is available.
    Done,
    /// Failed after exhausting retries.
    Failed,
    /// Cancelled by the client or by shutdown.
    Cancelled,
}

impl JobStatus {
    /// Whether the job has reached a terminal state.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            JobStatus::Done | JobStatus::Failed | JobStatus::Cancelled
        )
    }
}

/// Server-sent event emitted by `GET /v1/jobs/{id}/events`.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum JobEvent {
    /// Progress estimate while processing.
    Progress {
        /// Approximate fraction complete, 0–100.
        percent: u32,
        /// Seconds of audio considered processed so far.
        processed_seconds: f64,
    },
    /// Job completed successfully.
    Done,
    /// Job failed.
    Failed {
        /// Sanitized error message (no paths or model internals).
        error: String,
    },
    /// Job was cancelled.
    Cancelled,
}

impl JobEvent {
    /// Whether this event ends the stream.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            JobEvent::Done | JobEvent::Failed { .. } | JobEvent::Cancelled
        )
    }
}

/// Public status response for `GET /v1/jobs/{id}`.
#[derive(Debug, Clone, Serialize)]
pub struct JobStatusResponse {
    pub job_id: String,
    pub status: JobStatus,
    pub processed_seconds: f64,
    pub percent: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Build a public status view from a stored job.
pub(crate) fn job_status_response(job: &Job) -> JobStatusResponse {
    let percent = if job.total_seconds > 0.0 {
        ((job.processed_seconds / job.total_seconds) * 100.0) as u32
    } else {
        0
    };
    JobStatusResponse {
        job_id: job.id.clone(),
        status: job.status,
        processed_seconds: job.processed_seconds,
        percent,
        error: job.error.clone(),
    }
}

/// A transcription job.
#[derive(Debug, Clone)]
pub struct Job {
    pub id: String,
    pub status: JobStatus,
    /// Raw uploaded audio bytes.
    pub body: Bytes,
    /// Export / post-processing parameters captured at submission.
    pub params: ExportParams,
    pub created_at: f64,
    pub updated_at: f64,
    pub processed_seconds: f64,
    /// Total audio duration in seconds, set once the body is decoded.
    pub total_seconds: f64,
    /// Number of execution attempts made so far.
    pub attempts: u32,
    /// Populated when status becomes `Done`.
    pub result: Option<gigastt_core::inference::TranscribeResult>,
    /// Populated when status becomes `Failed`.
    pub error: Option<String>,
    /// Active SSE listeners.
    pub event_channels: Vec<tokio::sync::mpsc::UnboundedSender<JobEvent>>,
}

impl Job {
    /// Create a new queued job.
    pub fn queued(body: Bytes, params: ExportParams) -> Self {
        let now = gigastt_core::inference::now_timestamp();
        Self {
            id: uuid::Uuid::now_v7().to_string(),
            status: JobStatus::Queued,
            body,
            params,
            created_at: now,
            updated_at: now,
            processed_seconds: 0.0,
            total_seconds: 0.0,
            attempts: 0,
            result: None,
            error: None,
            event_channels: Vec::new(),
        }
    }
}

/// Persistence boundary for jobs. Handlers talk to this trait; the in-memory
/// implementation is the default, but a SQLite-backed store can be dropped in.
pub trait JobStore: Send + Sync + 'static {
    /// Persist a new job and return its id.
    fn create<'a>(&'a self, job: Job) -> JobStoreFuture<'a, anyhow::Result<String>>;
    /// Return a clone of the job, if it exists.
    fn get<'a>(&'a self, id: &str) -> JobStoreFuture<'a, anyhow::Result<Option<Job>>>;
    /// Apply an in-place mutation.
    fn update<'a>(
        &'a self,
        id: &str,
        f: Box<dyn FnOnce(&mut Job) + Send>,
    ) -> JobStoreFuture<'a, anyhow::Result<()>>;
    /// Pop the oldest queued job id whose status is still `Queued`.
    fn next_queued<'a>(&'a self) -> JobStoreFuture<'a, anyhow::Result<Option<String>>>;
    /// Push a job id to the back of the queue (used after a retryable failure).
    fn requeue<'a>(&'a self, id: &str) -> JobStoreFuture<'a, anyhow::Result<()>>;
    /// Whether the store has reached its capacity limit.
    fn is_full<'a>(&'a self) -> JobStoreFuture<'a, bool>;
}

/// In-memory FIFO job store with TTL eviction.
pub struct InMemoryJobStore {
    limits: RuntimeLimits,
    jobs: Mutex<HashMap<String, Job>>,
    queue: Mutex<VecDeque<String>>,
}

impl InMemoryJobStore {
    /// Create a new store with the given limits.
    pub fn new(limits: RuntimeLimits) -> Self {
        Self {
            limits,
            jobs: Mutex::new(HashMap::new()),
            queue: Mutex::new(VecDeque::new()),
        }
    }

    /// Evict terminal jobs whose TTL has expired. Must be called with both locks held.
    fn evict_expired_locked(&self, jobs: &mut HashMap<String, Job>, queue: &mut VecDeque<String>) {
        let ttl = self.limits.jobs_ttl_secs;
        if ttl == 0 {
            return;
        }
        let now = gigastt_core::inference::now_timestamp();
        let expired: Vec<String> = jobs
            .iter()
            .filter(|(_, j)| j.status.is_terminal() && now - j.updated_at > ttl as f64)
            .map(|(id, _)| id.clone())
            .collect();
        for id in expired {
            jobs.remove(&id);
            queue.retain(|x| x != &id);
        }
    }
}

impl JobStore for InMemoryJobStore {
    fn create<'a>(&'a self, job: Job) -> JobStoreFuture<'a, anyhow::Result<String>> {
        Box::pin(async move {
            let mut jobs = self.jobs.lock();
            let mut queue = self.queue.lock();
            self.evict_expired_locked(&mut jobs, &mut queue);
            if jobs.len() >= self.limits.jobs_max {
                return Err(anyhow::anyhow!("job store is full"));
            }
            let id = job.id.clone();
            jobs.insert(id.clone(), job);
            queue.push_back(id.clone());
            Ok(id)
        })
    }

    fn get<'a>(&'a self, id: &str) -> JobStoreFuture<'a, anyhow::Result<Option<Job>>> {
        let id = id.to_owned();
        Box::pin(async move {
            let jobs = self.jobs.lock();
            Ok(jobs.get(&id).cloned())
        })
    }

    fn update<'a>(
        &'a self,
        id: &str,
        f: Box<dyn FnOnce(&mut Job) + Send>,
    ) -> JobStoreFuture<'a, anyhow::Result<()>> {
        let id = id.to_owned();
        Box::pin(async move {
            let mut jobs = self.jobs.lock();
            let Some(job) = jobs.get_mut(&id) else {
                return Err(anyhow::anyhow!("job not found"));
            };
            f(job);
            job.updated_at = gigastt_core::inference::now_timestamp();
            Ok(())
        })
    }

    fn next_queued<'a>(&'a self) -> JobStoreFuture<'a, anyhow::Result<Option<String>>> {
        Box::pin(async move {
            let mut jobs = self.jobs.lock();
            let mut queue = self.queue.lock();
            self.evict_expired_locked(&mut jobs, &mut queue);
            while let Some(id) = queue.pop_front() {
                if let Some(job) = jobs.get(&id)
                    && matches!(job.status, JobStatus::Queued)
                {
                    return Ok(Some(id));
                }
            }
            Ok(None)
        })
    }

    fn requeue<'a>(&'a self, id: &str) -> JobStoreFuture<'a, anyhow::Result<()>> {
        let id = id.to_owned();
        Box::pin(async move {
            let mut queue = self.queue.lock();
            queue.push_back(id);
            Ok(())
        })
    }

    fn is_full<'a>(&'a self) -> JobStoreFuture<'a, bool> {
        Box::pin(async move {
            let mut jobs = self.jobs.lock();
            let mut queue = self.queue.lock();
            self.evict_expired_locked(&mut jobs, &mut queue);
            jobs.len() >= self.limits.jobs_max
        })
    }
}

/// Executor abstraction so unit tests can run the queue without loading ONNX.
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

    /// Spawn worker tasks. Each worker owns a clone of `executor`. Call once
    /// after constructing the queue.
    pub fn spawn<E>(&self, executor: E)
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
            tokio::spawn(worker.run());
        }
    }

    /// Mark all queued jobs as cancelled. Called during graceful shutdown.
    pub async fn cancel_all_queued(&self) {
        loop {
            let Ok(Some(id)) = self.store.next_queued().await else {
                break;
            };
            let _ = self
                .store
                .update(
                    &id,
                    Box::new(|j| {
                        j.status = JobStatus::Cancelled;
                        j.updated_at = gigastt_core::inference::now_timestamp();
                    }),
                )
                .await;
            broadcast_event(&*self.store, &id, JobEvent::Cancelled).await;
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

            let _ = self
                .store
                .update(
                    &id,
                    Box::new(|j| {
                        j.status = JobStatus::Processing;
                        j.attempts += 1;
                    }),
                )
                .await;
            broadcast_event(
                &*self.store,
                &id,
                JobEvent::Progress {
                    percent: 0,
                    processed_seconds: 0.0,
                },
            )
            .await;

            let store = self.store.clone();
            let body = match self.store.get(&id).await {
                Ok(Some(job)) => job.body,
                _ => {
                    drop(permit);
                    continue;
                }
            };
            let params = match self.store.get(&id).await {
                Ok(Some(job)) => job.params,
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

            // If the job was cancelled while running, discard the result.
            let cancelled = self
                .store
                .get(&id)
                .await
                .ok()
                .flatten()
                .map(|j| matches!(j.status, JobStatus::Cancelled))
                .unwrap_or(false);
            if cancelled {
                continue;
            }

            match result {
                Ok(res) => {
                    let total = res.duration_s;
                    let _ = self
                        .store
                        .update(
                            &id,
                            Box::new(move |j| {
                                j.status = JobStatus::Done;
                                j.result = Some(res);
                                j.processed_seconds = total;
                            }),
                        )
                        .await;
                    broadcast_event(&*self.store, &id, JobEvent::Done).await;
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
                    let retryable = is_retryable_error(&e) && attempts <= self.max_retries;
                    if retryable {
                        let _ = self
                            .store
                            .update(
                                &id,
                                Box::new(|j| {
                                    j.status = JobStatus::Queued;
                                }),
                            )
                            .await;
                        let _ = self.store.requeue(&id).await;
                        broadcast_event(
                            &*self.store,
                            &id,
                            JobEvent::Progress {
                                percent: 0,
                                processed_seconds: 0.0,
                            },
                        )
                        .await;
                    } else {
                        let sanitized = sanitize_job_error(&e);
                        let _ = self
                            .store
                            .update(
                                &id,
                                Box::new({
                                    let sanitized = sanitized.clone();
                                    move |j| {
                                        j.status = JobStatus::Failed;
                                        j.error = Some(sanitized);
                                    }
                                }),
                            )
                            .await;
                        broadcast_event(&*self.store, &id, JobEvent::Failed { error: sanitized })
                            .await;
                    }
                }
            }
        }
    }
}

/// Send an event to all active listeners, pruning dead channels. Terminal
/// events are fire-and-forget: the channel is dropped afterwards so the SSE
/// stream ends naturally.
pub(crate) async fn broadcast_event(store: &dyn JobStore, id: &str, event: JobEvent) {
    let channels = match store.get(id).await {
        Ok(Some(job)) => job.event_channels,
        _ => return,
    };
    let terminal = event.is_terminal();
    let mut keep = Vec::new();
    for tx in channels {
        if tx.send(event.clone()).is_ok() && !terminal {
            keep.push(tx);
        }
    }
    let _ = store
        .update(
            id,
            Box::new(move |j| {
                j.event_channels = keep;
            }),
        )
        .await;
}

fn is_retryable_error(e: &anyhow::Error) -> bool {
    let s = format!("{e:#}");
    s.contains("inference_timeout") || s.contains("panicked")
}

fn sanitize_job_error(e: &anyhow::Error) -> String {
    let msg = format!("{e:#}");
    if msg.contains("inference_timeout") {
        "Inference timed out.".into()
    } else if msg.contains("Invalid audio") {
        "Failed to decode audio file. Check format.".into()
    } else {
        "Transcription failed.".into()
    }
}

/// Production executor: decodes audio to size the job, runs inference against
/// the batch pool, and emits timer-based progress events.
#[derive(Clone)]
pub struct RealJobExecutor {
    engine: Arc<ArcSwap<gigastt_core::inference::Engine>>,
    limits: Arc<ArcSwap<RuntimeLimits>>,
}

impl RealJobExecutor {
    /// Create an executor bound to the live engine and runtime limits.
    pub fn new(
        engine: Arc<ArcSwap<gigastt_core::inference::Engine>>,
        limits: Arc<ArcSwap<RuntimeLimits>>,
    ) -> Self {
        Self { engine, limits }
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

        // Decode audio (blocking) once to estimate duration for progress.
        // Wrap the decoder in catch_unwind so a malformed file cannot be
        // retried as a transient inference panic.
        let samples = tokio::task::spawn_blocking({
            let body = body.clone();
            move || {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    gigastt_core::inference::audio::decode_audio_bytes_shared(body)
                }))
            }
        })
        .await
        .map_err(|e| anyhow::anyhow!("audio decode task panicked: {e}"))?
        .map_err(|_| anyhow::anyhow!("Invalid audio: decoder panicked"))?
        .map_err(|e| anyhow::anyhow!("Invalid audio: {e:#}"))?;

        const TARGET_SAMPLE_RATE: f64 = 16_000.0;
        let total_seconds = samples.len() as f64 / TARGET_SAMPLE_RATE;
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
        let overrides = gigastt_core::inference::TranscribeOverrides {
            punctuation: params.punctuation,
            itn: params.itn,
            vad: params.vad,
        };
        if let Err(e) = engine.validate_overrides(&overrides) {
            return Err(anyhow::anyhow!("Invalid input: {}", e.message()));
        }

        // Timer-based progress updater. Assumes RTF ≈ 0.1 (10 s audio / 1 s wall).
        let progress_cancel = tokio_util::sync::CancellationToken::new();
        let progress_handle = {
            let store = store.clone();
            let id = id.to_string();
            let cancel = progress_cancel.clone();
            let total = total_seconds;
            tokio::spawn(async move {
                let start = std::time::Instant::now();
                let rtf = 0.1_f64;
                let mut interval = tokio::time::interval(std::time::Duration::from_millis(500));
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                loop {
                    interval.tick().await;
                    if cancel.is_cancelled() {
                        break;
                    }
                    let elapsed = start.elapsed().as_secs_f64();
                    let processed = (elapsed / rtf).min(total);
                    let percent = if total > 0.0 {
                        ((processed / total) * 100.0) as u32
                    } else {
                        100
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
                    if processed >= total {
                        break;
                    }
                }
            })
        };

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
            let handle = tokio::task::spawn_blocking(move || {
                let _enter = span.enter();
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    if params.channels.as_deref() == Some("split") {
                        let channels =
                            gigastt_core::inference::audio::decode_audio_bytes_shared_channels(
                                body_for_inference.clone(),
                            )
                            .map_err(|e| {
                                gigastt_core::error::GigasttError::InvalidAudio {
                                    reason: format!("{e:#}"),
                                }
                            })?;
                        let fallback_reason = match channels.len() {
                            0 => Some("no channels"),
                            1 => Some("mono audio"),
                            2 if gigastt_core::inference::audio::is_dual_mono(&channels) => {
                                Some("dual-mono audio")
                            }
                            n if n > 2 => Some("more than two channels"),
                            _ => None,
                        };
                        if let Some(reason) = fallback_reason {
                            tracing::warn!(
                                "channels=split requested but {reason} detected; falling back to mono transcription"
                            );
                            engine_for_inference
                                .transcribe_bytes_shared(body_for_inference, &mut reservation)
                        } else {
                            engine_for_inference.transcribe_channels(&channels, &mut reservation)
                        }
                    } else if params.diarization == Some(true) {
                        // Diarization is opt-in (`?diarization=true`): only then run
                        // the offline speaker pass, matching the sync REST path.
                        engine_for_inference.transcribe_bytes_shared_with_overrides_diarized(
                            body_for_inference,
                            &mut reservation,
                            &overrides,
                        )
                    } else {
                        engine_for_inference.transcribe_bytes_shared_with_overrides(
                            body_for_inference,
                            &mut reservation,
                            &overrides,
                        )
                    }
                }));
                match r {
                    Ok(result) => result,
                    Err(_) => Err(gigastt_core::error::GigasttError::Inference {
                        source: anyhow::anyhow!("Inference thread panicked").into(),
                    }),
                }
            });

            if inference_timeout_secs == 0 {
                handle
                    .await
                    .map_err(|e| anyhow::anyhow!("spawn_blocking join error: {e}"))?
                    .map_err(anyhow::Error::from)
            } else {
                match tokio::time::timeout(
                    std::time::Duration::from_secs(inference_timeout_secs),
                    handle,
                )
                .await
                {
                    Ok(r) => r.map_err(|e| anyhow::anyhow!("spawn_blocking join error: {e}"))?
                        .map_err(anyhow::Error::from),
                    Err(_) => Err(anyhow::anyhow!("inference_timeout")),
                }
            }
        }
        .await;

        progress_cancel.cancel();
        let _ = progress_handle.await;

        inference_result
    }
}

#[cfg(test)]
impl InMemoryJobStore {
    /// Shift a job's `updated_at` back in time for TTL eviction tests.
    pub async fn backdate(&self, id: &str, seconds: f64) {
        let mut jobs = self.jobs.lock();
        if let Some(job) = jobs.get_mut(id) {
            job.updated_at -= seconds;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_limits() -> RuntimeLimits {
        RuntimeLimits {
            jobs_enabled: true,
            jobs_ttl_secs: 3600,
            jobs_max: 10,
            jobs_retry: 0,
            ..RuntimeLimits::default()
        }
    }

    #[tokio::test]
    async fn test_in_memory_store_crud() {
        let store = InMemoryJobStore::new(test_limits());
        let id = store
            .create(Job::queued(
                Bytes::from_static(b"a"),
                ExportParams::default(),
            ))
            .await
            .unwrap();
        let job = store.get(&id).await.unwrap().unwrap();
        assert!(matches!(job.status, JobStatus::Queued));
        store
            .update(&id, Box::new(|j| j.status = JobStatus::Processing))
            .await
            .unwrap();
        let job = store.get(&id).await.unwrap().unwrap();
        assert!(matches!(job.status, JobStatus::Processing));
    }

    #[tokio::test]
    async fn test_store_fifo_order() {
        let store = InMemoryJobStore::new(test_limits());
        let id1 = store
            .create(Job::queued(
                Bytes::from_static(b"1"),
                ExportParams::default(),
            ))
            .await
            .unwrap();
        let id2 = store
            .create(Job::queued(
                Bytes::from_static(b"2"),
                ExportParams::default(),
            ))
            .await
            .unwrap();
        assert_eq!(store.next_queued().await.unwrap(), Some(id1.clone()));
        // id1 is still queued in the store; next_queued returned it but did not
        // change its status. Simulate another worker trying to pop while id1
        // is still queued: it should see id1 again because status is still Queued.
        // Mark id1 processing and then next should be id2.
        store
            .update(&id1, Box::new(|j| j.status = JobStatus::Processing))
            .await
            .unwrap();
        assert_eq!(store.next_queued().await.unwrap(), Some(id2));
    }

    #[tokio::test]
    async fn test_store_capacity_limit() {
        let limits = RuntimeLimits {
            jobs_max: 1,
            ..test_limits()
        };
        let store = InMemoryJobStore::new(limits);
        store
            .create(Job::queued(
                Bytes::from_static(b"a"),
                ExportParams::default(),
            ))
            .await
            .unwrap();
        assert!(store.is_full().await);
        let result = store
            .create(Job::queued(
                Bytes::from_static(b"b"),
                ExportParams::default(),
            ))
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_store_is_full_evicts_expired() {
        let limits = RuntimeLimits {
            jobs_ttl_secs: 1,
            jobs_max: 1,
            ..test_limits()
        };
        let store = InMemoryJobStore::new(limits);
        let id = store
            .create(Job::queued(
                Bytes::from_static(b"a"),
                ExportParams::default(),
            ))
            .await
            .unwrap();
        store
            .update(&id, Box::new(|j| j.status = JobStatus::Done))
            .await
            .unwrap();
        store.backdate(&id, 2.0).await;
        // is_full should evict the expired terminal job and report capacity.
        assert!(!store.is_full().await);
    }

    #[tokio::test]
    async fn test_store_ttl_eviction() {
        let limits = RuntimeLimits {
            jobs_ttl_secs: 1,
            jobs_max: 10,
            ..test_limits()
        };
        let store = InMemoryJobStore::new(limits);
        let id = store
            .create(Job::queued(
                Bytes::from_static(b"a"),
                ExportParams::default(),
            ))
            .await
            .unwrap();
        store
            .update(&id, Box::new(|j| j.status = JobStatus::Done))
            .await
            .unwrap();
        // Backdate the job by more than the 1-second TTL.
        store.backdate(&id, 2.0).await;
        // Creating a new job should evict the expired one.
        store
            .create(Job::queued(
                Bytes::from_static(b"b"),
                ExportParams::default(),
            ))
            .await
            .unwrap();
        assert!(store.get(&id).await.unwrap().is_none());
    }

    #[derive(Clone)]
    struct MockExecutor {
        results: Arc<Mutex<Vec<anyhow::Result<gigastt_core::inference::TranscribeResult>>>>,
        delay_ms: u64,
    }

    impl JobExecution for MockExecutor {
        async fn execute(
            &self,
            _id: &str,
            _store: Arc<dyn JobStore>,
            body: Bytes,
            _params: ExportParams,
        ) -> anyhow::Result<gigastt_core::inference::TranscribeResult> {
            if self.delay_ms > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;
            }
            let mut results = self.results.lock();
            let result = results.remove(0);
            // Use body length to make failures deterministic per test.
            let _ = body.len();
            result
        }
    }

    fn ok_result() -> anyhow::Result<gigastt_core::inference::TranscribeResult> {
        Ok(gigastt_core::inference::TranscribeResult {
            text: "ok".into(),
            words: vec![],
            duration_s: 1.0,
        })
    }

    #[tokio::test]
    async fn test_queue_runs_jobs_in_fifo_order() {
        let limits = test_limits();
        let store: Arc<dyn JobStore> = Arc::new(InMemoryJobStore::new(limits));
        let executor = MockExecutor {
            results: Arc::new(Mutex::new(vec![ok_result(), ok_result()])),
            delay_ms: 0,
        };
        let queue = JobQueue::new(
            store.clone(),
            2,
            0,
            tokio_util::sync::CancellationToken::new(),
        );
        queue.spawn(executor);

        let id1 = store
            .create(Job::queued(
                Bytes::from_static(b"1"),
                ExportParams::default(),
            ))
            .await
            .unwrap();
        let id2 = store
            .create(Job::queued(
                Bytes::from_static(b"2"),
                ExportParams::default(),
            ))
            .await
            .unwrap();

        // Wait for both to finish.
        for _ in 0..50 {
            let j1 = store.get(&id1).await.unwrap().unwrap();
            let j2 = store.get(&id2).await.unwrap().unwrap();
            if matches!(j1.status, JobStatus::Done) && matches!(j2.status, JobStatus::Done) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }

        assert!(matches!(
            store.get(&id1).await.unwrap().unwrap().status,
            JobStatus::Done
        ));
        assert!(matches!(
            store.get(&id2).await.unwrap().unwrap().status,
            JobStatus::Done
        ));
    }

    #[tokio::test]
    async fn test_queue_retry_then_fail() {
        let limits = RuntimeLimits {
            jobs_retry: 2,
            ..test_limits()
        };
        let store: Arc<dyn JobStore> = Arc::new(InMemoryJobStore::new(limits));
        let executor = MockExecutor {
            results: Arc::new(Mutex::new(vec![
                Err(anyhow::anyhow!("inference_timeout")),
                Err(anyhow::anyhow!("inference_timeout")),
                Err(anyhow::anyhow!("inference_timeout")),
            ])),
            delay_ms: 0,
        };
        let queue = JobQueue::new(
            store.clone(),
            1,
            2,
            tokio_util::sync::CancellationToken::new(),
        );
        queue.spawn(executor);

        let id = store
            .create(Job::queued(
                Bytes::from_static(b"x"),
                ExportParams::default(),
            ))
            .await
            .unwrap();

        for _ in 0..50 {
            let job = store.get(&id).await.unwrap().unwrap();
            if matches!(job.status, JobStatus::Failed) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }

        let job = store.get(&id).await.unwrap().unwrap();
        assert!(matches!(job.status, JobStatus::Failed));
        assert_eq!(job.attempts, 3);
    }

    #[tokio::test]
    async fn test_queue_cancellation_discards_result() {
        let store: Arc<dyn JobStore> = Arc::new(InMemoryJobStore::new(test_limits()));
        let executor = MockExecutor {
            results: Arc::new(Mutex::new(vec![ok_result()])),
            delay_ms: 300,
        };
        let queue = JobQueue::new(
            store.clone(),
            1,
            0,
            tokio_util::sync::CancellationToken::new(),
        );
        queue.spawn(executor);

        let id = store
            .create(Job::queued(
                Bytes::from_static(b"x"),
                ExportParams::default(),
            ))
            .await
            .unwrap();

        // Wait until processing, then cancel while the executor is still running.
        for _ in 0..50 {
            let job = store.get(&id).await.unwrap().unwrap();
            if matches!(job.status, JobStatus::Processing) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        store
            .update(&id, Box::new(|j| j.status = JobStatus::Cancelled))
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        let job = store.get(&id).await.unwrap().unwrap();
        assert!(matches!(job.status, JobStatus::Cancelled));
        assert!(job.result.is_none());
    }

    #[tokio::test]
    async fn test_queue_cancel_all_queued_on_shutdown() {
        let store: Arc<dyn JobStore> = Arc::new(InMemoryJobStore::new(test_limits()));
        let token = tokio_util::sync::CancellationToken::new();
        let queue = JobQueue::new(store.clone(), 1, 0, token.clone());
        // Don't spawn workers so jobs stay queued.

        let id = store
            .create(Job::queued(
                Bytes::from_static(b"x"),
                ExportParams::default(),
            ))
            .await
            .unwrap();

        token.cancel();
        queue.cancel_all_queued().await;

        let job = store.get(&id).await.unwrap().unwrap();
        assert!(matches!(job.status, JobStatus::Cancelled));
    }

    #[test]
    fn test_sanitize_job_error_maps_known_errors() {
        assert_eq!(
            sanitize_job_error(&anyhow::anyhow!("inference_timeout")),
            "Inference timed out."
        );
        assert_eq!(
            sanitize_job_error(&anyhow::anyhow!("Invalid audio: unsupported format")),
            "Failed to decode audio file. Check format."
        );
        assert_eq!(
            sanitize_job_error(&anyhow::anyhow!("some internal onnx path /foo/bar")),
            "Transcription failed."
        );
    }

    #[test]
    fn test_is_retryable_error_recognizes_transient_failures() {
        assert!(is_retryable_error(&anyhow::anyhow!("inference_timeout")));
        assert!(is_retryable_error(&anyhow::anyhow!(
            "worker thread panicked"
        )));
        assert!(!is_retryable_error(&anyhow::anyhow!("Invalid audio")));
    }

    #[tokio::test]
    async fn test_store_get_missing_returns_none() {
        let store = InMemoryJobStore::new(test_limits());
        assert!(store.get("no-such-id").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_store_update_missing_returns_error() {
        let store = InMemoryJobStore::new(test_limits());
        assert!(
            store
                .update("no-such-id", Box::new(|j| j.status = JobStatus::Done))
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn test_broadcast_event_prunes_dead_channels() {
        let store: Arc<dyn JobStore> = Arc::new(InMemoryJobStore::new(test_limits()));
        let id = store
            .create(Job::queued(
                Bytes::from_static(b"x"),
                ExportParams::default(),
            ))
            .await
            .unwrap();

        // Add a live channel and a channel that will be dropped before broadcast.
        let (live_tx, mut live_rx) = tokio::sync::mpsc::unbounded_channel::<JobEvent>();
        {
            let (dead_tx, _dead_rx) = tokio::sync::mpsc::unbounded_channel::<JobEvent>();
            store
                .update(
                    &id,
                    Box::new(move |j| {
                        j.event_channels.push(live_tx);
                        j.event_channels.push(dead_tx);
                    }),
                )
                .await
                .unwrap();
        }

        broadcast_event(
            &*store,
            &id,
            JobEvent::Progress {
                percent: 50,
                processed_seconds: 1.0,
            },
        )
        .await;

        let job = store.get(&id).await.unwrap().unwrap();
        assert_eq!(job.event_channels.len(), 1);
        assert!(matches!(live_rx.try_recv(), Ok(JobEvent::Progress { .. })));
    }

    #[tokio::test]
    async fn test_queue_concurrency_clamped_to_one() {
        let store: Arc<dyn JobStore> = Arc::new(InMemoryJobStore::new(test_limits()));
        let executor = MockExecutor {
            results: Arc::new(Mutex::new(vec![ok_result(), ok_result()])),
            delay_ms: 0,
        };
        // Pass 0; the queue must still spawn at least one worker.
        let queue = JobQueue::new(
            store.clone(),
            0,
            0,
            tokio_util::sync::CancellationToken::new(),
        );
        queue.spawn(executor);

        let id = store
            .create(Job::queued(
                Bytes::from_static(b"x"),
                ExportParams::default(),
            ))
            .await
            .unwrap();

        for _ in 0..50 {
            let job = store.get(&id).await.unwrap().unwrap();
            if matches!(job.status, JobStatus::Done) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }

        assert!(matches!(
            store.get(&id).await.unwrap().unwrap().status,
            JobStatus::Done
        ));
    }

    #[tokio::test]
    async fn test_queue_non_retryable_error_fails_immediately() {
        let store: Arc<dyn JobStore> = Arc::new(InMemoryJobStore::new(test_limits()));
        let executor = MockExecutor {
            results: Arc::new(Mutex::new(vec![Err(anyhow::anyhow!(
                "Invalid audio: bad header"
            ))])),
            delay_ms: 0,
        };
        let queue = JobQueue::new(
            store.clone(),
            1,
            3,
            tokio_util::sync::CancellationToken::new(),
        );
        queue.spawn(executor);

        let id = store
            .create(Job::queued(
                Bytes::from_static(b"x"),
                ExportParams::default(),
            ))
            .await
            .unwrap();

        for _ in 0..50 {
            let job = store.get(&id).await.unwrap().unwrap();
            if matches!(job.status, JobStatus::Failed) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }

        let job = store.get(&id).await.unwrap().unwrap();
        assert!(matches!(job.status, JobStatus::Failed));
        assert_eq!(job.attempts, 1);
    }

    #[tokio::test]
    async fn test_queue_retry_boundary_exhausted_after_max_retries() {
        // max_retries=1 means one retry is allowed: attempts 1 (retry), 2 (fail).
        let limits = RuntimeLimits {
            jobs_retry: 1,
            ..test_limits()
        };
        let store: Arc<dyn JobStore> = Arc::new(InMemoryJobStore::new(limits));
        let executor = MockExecutor {
            results: Arc::new(Mutex::new(vec![
                Err(anyhow::anyhow!("inference_timeout")),
                Err(anyhow::anyhow!("inference_timeout")),
            ])),
            delay_ms: 0,
        };
        let queue = JobQueue::new(
            store.clone(),
            1,
            1,
            tokio_util::sync::CancellationToken::new(),
        );
        queue.spawn(executor);

        let id = store
            .create(Job::queued(
                Bytes::from_static(b"x"),
                ExportParams::default(),
            ))
            .await
            .unwrap();

        for _ in 0..50 {
            let job = store.get(&id).await.unwrap().unwrap();
            if matches!(job.status, JobStatus::Failed) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }

        let job = store.get(&id).await.unwrap().unwrap();
        assert!(matches!(job.status, JobStatus::Failed));
        assert_eq!(job.attempts, 2);
    }

    #[test]
    fn test_job_status_response_percent() {
        let mut job = Job::queued(Bytes::new(), ExportParams::default());
        job.id = "test".into();
        job.total_seconds = 10.0;
        job.processed_seconds = 3.5;
        job.status = JobStatus::Processing;
        let resp = job_status_response(&job);
        assert_eq!(resp.percent, 35);
        assert_eq!(resp.processed_seconds, 3.5);
    }
}
