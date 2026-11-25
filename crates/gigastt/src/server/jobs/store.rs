//! Job types and in-memory store for the async transcription queue.

use axum::body::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use parking_lot::Mutex;

use super::super::config::RuntimeLimits;
use super::super::http::ExportParams;
use super::events::{JobEvent, JobEventSender};

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

/// Public status response for `GET /v1/jobs/{id}`.
#[derive(Debug, Clone, Serialize)]
pub struct JobStatusResponse {
    pub job_id: String,
    pub status: JobStatus,
    pub processed_seconds: f64,
    pub percent: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Last provisional text, retained after cancellation or failure.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partial: Option<gigastt_core::inference::TranscriptSegment>,
}

/// Build a public status view from a stored job.
pub(crate) fn job_status_response(job: &Job) -> JobStatusResponse {
    let mut response = job_status_metadata(job);
    response.partial = job.partial.as_ref().and_then(|partial| partial.get());
    response
}

fn job_status_metadata(job: &Job) -> JobStatusResponse {
    let percent = if job.total_seconds > 0.0 {
        (((job.processed_seconds / job.total_seconds) * 100.0) as u32).min(100)
    } else {
        0
    };
    JobStatusResponse {
        job_id: job.id.clone(),
        status: job.status,
        processed_seconds: job.processed_seconds,
        percent,
        error: job.error.clone(),
        partial: None,
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
    /// Why speakers were or were not labeled, recorded when the run finishes and
    /// only when `?diarization=true` was submitted. `GET /v1/jobs/{id}/result`
    /// turns it into the same capability notice the synchronous endpoint
    /// attaches, so an async job cannot answer with silently empty speaker
    /// fields either. `None` when diarization was not requested.
    pub diarization: Option<gigastt_core::inference::DiarizationOutcome>,
    /// Populated when status becomes `Failed`.
    pub error: Option<String>,
    /// Active SSE listeners.
    pub event_channels: Vec<JobEventSender>,
    /// Cooperative-cancellation flag for the in-flight run. The executor sets it
    /// when it begins inference; `DELETE /v1/jobs/{id}` flips it so the engine
    /// releases its pooled triplet after the current runtime call instead of transcribing the
    /// whole file. `None` while queued and after a terminal state. Purely
    /// in-memory (never serialized): it is a live handle, not job metadata.
    pub abort: Option<Arc<AtomicBool>>,
    /// Shared with a blocking run, so late cancellation can still publish its
    /// final partial after the watchdog has already returned to the worker.
    pub partial: Option<Arc<gigastt_core::inference::TranscriptSnapshot>>,
}

/// Upper bound on simultaneous SSE listeners per job. Dead channels are
/// pruned on subscribe and on broadcast, but broadcasts can be ≥500 ms apart
/// (and rarer for a queued job), so a connect/disconnect flood could otherwise
/// accumulate listeners faster than they get cleaned up.
pub(crate) const MAX_JOB_EVENT_SUBSCRIBERS: usize = 32;

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
            diarization: None,
            error: None,
            event_channels: Vec::new(),
            abort: None,
            partial: None,
        }
    }

    /// Apply metadata and publish the corresponding event while the store
    /// owns its mutation lock. Queue bookkeeping belongs to the store.
    fn transition(&mut self, transition: JobTransition) -> TransitionOutcome {
        let allowed = match &transition {
            JobTransition::Start => self.status == JobStatus::Queued,
            JobTransition::Cancel => !self.status.is_terminal(),
            _ => self.status == JobStatus::Processing,
        };
        if !allowed {
            return TransitionOutcome::Rejected(self.status);
        }
        match transition {
            JobTransition::Start => {
                self.status = JobStatus::Processing;
                self.attempts += 1;
            }
            JobTransition::Complete(result) => {
                self.status = JobStatus::Done;
                self.processed_seconds = result.duration_s;
                self.result = Some(result);
                self.partial = None;
            }
            JobTransition::Fail(error) => {
                self.status = JobStatus::Failed;
                self.error = Some(error);
            }
            JobTransition::Retry => {
                self.status = JobStatus::Queued;
                self.abort = None;
            }
            JobTransition::Cancel => {
                self.status = JobStatus::Cancelled;
                if let Some(abort) = &self.abort {
                    abort.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
        self.updated_at = gigastt_core::inference::now_timestamp();
        if self.status.is_terminal() {
            self.body = Bytes::new();
            self.abort = None;
        }
        let event = self.terminal_event().unwrap_or(JobEvent::Progress {
            percent: 0,
            processed_seconds: 0.0,
        });
        self.broadcast(event);
        TransitionOutcome::Applied
    }

    fn terminal_event(&self) -> Option<JobEvent> {
        match self.status {
            JobStatus::Done => Some(JobEvent::Done),
            JobStatus::Failed => Some(JobEvent::Failed {
                error: self
                    .error
                    .clone()
                    .unwrap_or_else(|| "Transcription failed.".into()),
            }),
            JobStatus::Cancelled => Some(JobEvent::Cancelled),
            _ => None,
        }
    }

    /// Publish while holding the store mutation lock; never restore a stale
    /// subscriber list or send progress after a terminal transition.
    pub(crate) fn broadcast(&mut self, event: JobEvent) {
        let compatible = matches!(
            (&event, self.status),
            (JobEvent::Done, JobStatus::Done)
                | (JobEvent::Failed { .. }, JobStatus::Failed)
                | (JobEvent::Cancelled, JobStatus::Cancelled)
        ) || (!event.is_terminal() && !self.status.is_terminal());
        if !compatible {
            return;
        }
        self.event_channels
            .retain(|tx| tx.publish(event.clone()) && !event.is_terminal());
    }

    /// Register an SSE listener, pruning closed channels first. When the
    /// subscriber cap is reached, the oldest listener is evicted (its stream
    /// ends) so a connect/disconnect flood cannot grow the list without bound.
    pub(crate) fn subscribe(&mut self, tx: JobEventSender) {
        if let Some(event) = self.terminal_event() {
            let _ = tx.publish(event);
            return;
        }
        self.event_channels.retain(|tx| !tx.is_closed());
        if self.event_channels.len() >= MAX_JOB_EVENT_SUBSCRIBERS {
            self.event_channels.remove(0);
        }
        self.event_channels.push(tx);
    }
}

/// Guarded metadata transitions. Runtime handles are registered separately.
pub enum JobTransition {
    /// Claim a queued job and count this attempt.
    Start,
    /// Finish a processing job successfully.
    Complete(gigastt_core::inference::TranscribeResult),
    /// Fail a processing job with a sanitized public error.
    Fail(String),
    /// Requeue a processing job without releasing its upload.
    Retry,
    /// Cancel a queued or processing job and signal its runtime handle.
    Cancel,
}

/// Result observed under the same lock as the attempted transition.
#[derive(Debug, PartialEq, Eq)]
pub enum TransitionOutcome {
    Applied,
    Rejected(JobStatus),
    Missing,
}

/// Atomic admission rejected the upload because the queue is full.
#[derive(Debug)]
pub struct JobStoreFull;

impl std::fmt::Display for JobStoreFull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("job store is full")
    }
}
impl std::error::Error for JobStoreFull {}

/// Persistence boundary for jobs. Handlers talk to this trait; the in-memory
/// implementation is the default, but a SQLite-backed store can be dropped in.
pub trait JobStore: Send + Sync + 'static {
    /// Persist a new job and return its id.
    fn create<'a>(&'a self, job: Job) -> JobStoreFuture<'a, anyhow::Result<String>>;
    /// Return a clone of the job, if it exists.
    fn get<'a>(&'a self, id: &str) -> JobStoreFuture<'a, anyhow::Result<Option<Job>>>;
    /// Read status metadata without requiring a completed transcript. The default
    /// preserves existing stores; implementations can avoid cloning the result.
    fn status<'a>(
        &'a self,
        id: &str,
    ) -> JobStoreFuture<'a, anyhow::Result<Option<JobStatusResponse>>> {
        let id = id.to_owned();
        Box::pin(async move { Ok(self.get(&id).await?.as_ref().map(job_status_response)) })
    }
    /// Register a live listener or send the terminal event under one lock.
    /// Construct subscriptions with [`super::job_event_channel`] and publish
    /// through [`JobEventSender::publish`] to preserve the terminal slot.
    /// Returns false if the job does not exist.
    fn subscribe<'a>(
        &'a self,
        id: &str,
        tx: JobEventSender,
    ) -> JobStoreFuture<'a, anyhow::Result<bool>> {
        let id = id.to_owned();
        Box::pin(async move {
            if self.get(&id).await?.is_none() {
                return Ok(false);
            }
            self.update(&id, Box::new(move |job| job.subscribe(tx)))
                .await?;
            Ok(true)
        })
    }
    /// Apply a guarded lifecycle transition and publish its event atomically.
    /// The default uses `update`; stores may override to also enqueue retries
    /// in the same transaction. Legacy store lookup errors remain errors.
    fn transition<'a>(
        &'a self,
        id: &str,
        transition: JobTransition,
    ) -> JobStoreFuture<'a, anyhow::Result<TransitionOutcome>> {
        let id = id.to_owned();
        Box::pin(async move {
            if self.get(&id).await?.is_none() {
                return Ok(TransitionOutcome::Missing);
            }
            let retry = matches!(transition, JobTransition::Retry);
            let (tx, rx) = tokio::sync::oneshot::channel();
            self.update(
                &id,
                Box::new(move |job| {
                    let _ = tx.send(job.transition(transition));
                }),
            )
            .await?;
            let outcome = rx.await?;
            if retry && outcome == TransitionOutcome::Applied {
                self.requeue(&id).await?;
            }
            Ok(outcome)
        })
    }
    /// Apply an in-place runtime or metadata mutation. Lifecycle callers must
    /// use `transition` so state and event publication remain indivisible.
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

    /// Total bytes of buffered uploads currently resident across all jobs.
    /// Terminal jobs release their body (`Bytes::new()`), so this is the live
    /// upload footprint — the quantity `jobs_max_bytes` bounds. O(jobs), and
    /// `jobs_max` already caps the count, so this is a short walk.
    fn resident_body_bytes(jobs: &HashMap<String, Job>) -> usize {
        jobs.values().map(|j| j.body.len()).sum()
    }

    /// Whether the store is at capacity by count OR by resident upload bytes.
    /// Folding the byte budget in here means an over-budget queue produces the
    /// exact same 429 + `Retry-After` backpressure as a count-full one, with no
    /// new error type. Bounds live upload RAM to `jobs_max_bytes` plus at most
    /// one `body_limit_bytes` (the job that crosses the line is admitted, the
    /// next is refused), mirroring how the count cap admits exactly `jobs_max`.
    fn at_capacity(&self, jobs: &HashMap<String, Job>) -> bool {
        jobs.len() >= self.limits.jobs_max
            || Self::resident_body_bytes(jobs) >= self.limits.jobs_max_bytes
    }
}

impl JobStore for InMemoryJobStore {
    fn create<'a>(&'a self, job: Job) -> JobStoreFuture<'a, anyhow::Result<String>> {
        Box::pin(async move {
            let mut jobs = self.jobs.lock();
            let mut queue = self.queue.lock();
            self.evict_expired_locked(&mut jobs, &mut queue);
            // Admission and insertion share the lock: concurrent uploads see
            // the same count/byte budget and receive typed backpressure.
            if self.at_capacity(&jobs) {
                return Err(JobStoreFull.into());
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

    fn status<'a>(
        &'a self,
        id: &str,
    ) -> JobStoreFuture<'a, anyhow::Result<Option<JobStatusResponse>>> {
        let id = id.to_owned();
        Box::pin(async move {
            let snapshot = {
                let jobs = self.jobs.lock();
                jobs.get(&id)
                    .map(|job| (job_status_metadata(job), job.partial.clone()))
            };
            // A snapshot can be updated by a blocking inference worker. Resolve
            // its text after releasing the store lock, including after eviction.
            Ok(snapshot.map(|(mut response, partial)| {
                response.partial = partial.and_then(|partial| partial.get());
                response
            }))
        })
    }

    fn subscribe<'a>(
        &'a self,
        id: &str,
        tx: JobEventSender,
    ) -> JobStoreFuture<'a, anyhow::Result<bool>> {
        let id = id.to_owned();
        Box::pin(async move {
            let mut jobs = self.jobs.lock();
            let Some(job) = jobs.get_mut(&id) else {
                return Ok(false);
            };
            job.subscribe(tx);
            Ok(true)
        })
    }

    fn transition<'a>(
        &'a self,
        id: &str,
        transition: JobTransition,
    ) -> JobStoreFuture<'a, anyhow::Result<TransitionOutcome>> {
        let id = id.to_owned();
        Box::pin(async move {
            let mut jobs = self.jobs.lock();
            let Some(job) = jobs.get_mut(&id) else {
                return Ok(TransitionOutcome::Missing);
            };
            let retry = matches!(transition, JobTransition::Retry);
            let outcome = job.transition(transition);
            if retry && outcome == TransitionOutcome::Applied {
                self.queue.lock().push_back(id);
            }
            Ok(outcome)
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
            self.at_capacity(&jobs)
        })
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
mod polling_benchmark {
    use super::*;

    #[test]
    #[ignore = "synthetic lock-hold benchmark; run explicitly with --nocapture"]
    fn benchmark_completed_status_lock_hold() {
        let store = InMemoryJobStore::new(RuntimeLimits::default());
        let mut job = Job::queued(Bytes::new(), ExportParams::default());
        job.status = JobStatus::Done;
        job.result = Some(gigastt_core::inference::TranscribeResult {
            text: "x".repeat(8 * 1024 * 1024),
            words: vec![],
            duration_s: 1.0,
            confidence: None,
        });
        let id = job.id.clone();
        store.jobs.lock().insert(id.clone(), job);
        for legacy in [true, false] {
            let mut elapsed = std::time::Duration::ZERO;
            for _ in 0..200 {
                let jobs = store.jobs.lock();
                let started = std::time::Instant::now();
                let job = jobs.get(&id).unwrap();
                if legacy {
                    let snapshot = std::hint::black_box(job.clone());
                    elapsed += started.elapsed();
                    drop(jobs);
                    drop(snapshot);
                } else {
                    let snapshot =
                        std::hint::black_box((job_status_metadata(job), job.partial.clone()));
                    elapsed += started.elapsed();
                    drop(jobs);
                    drop(snapshot);
                }
            }
            eprintln!(
                "legacy={legacy} mean_lock_hold_ns={}",
                elapsed.as_nanos() / 200
            );
        }
    }
}
