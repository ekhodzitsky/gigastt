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
    /// Last provisional text, retained after cancellation or failure.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partial: Option<gigastt_core::inference::TranscriptSegment>,
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
        partial: job.partial.as_ref().and_then(|partial| partial.get()),
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
    pub event_channels: Vec<tokio::sync::mpsc::UnboundedSender<JobEvent>>,
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
/// accumulate `UnboundedSender`s faster than they get cleaned up.
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

    /// Register an SSE listener, pruning closed channels first. When the
    /// subscriber cap is reached, the oldest listener is evicted (its stream
    /// ends) so a connect/disconnect flood cannot grow the list without bound.
    pub(crate) fn subscribe(&mut self, tx: tokio::sync::mpsc::UnboundedSender<JobEvent>) {
        self.event_channels.retain(|tx| !tx.is_closed());
        if self.event_channels.len() >= MAX_JOB_EVENT_SUBSCRIBERS {
            self.event_channels.remove(0);
        }
        self.event_channels.push(tx);
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
            // Race backstop behind the `is_full` gate in `submit_job` (which
            // returns 429 + Retry-After): rejects on either the count or the
            // byte budget so two concurrent submits can't both slip past the gate.
            if self.at_capacity(&jobs) {
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
