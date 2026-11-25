//! Unit tests for the job queue, store, and helpers.

pub(super) use super::super::config::RuntimeLimits;
pub(super) use super::super::http::ExportParams;
pub(super) use super::queue::{is_retryable_error, sanitize_job_error};
pub(super) use super::store::MAX_JOB_EVENT_SUBSCRIBERS;
pub(super) use super::*;
pub(super) use axum::body::Bytes;
pub(super) use parking_lot::Mutex;
pub(super) use std::sync::Arc;

pub(super) fn test_limits() -> RuntimeLimits {
    RuntimeLimits {
        jobs_enabled: true,
        jobs_ttl_secs: 3600,
        jobs_max: 10,
        jobs_retry: 0,
        ..RuntimeLimits::default()
    }
}

#[derive(Clone)]
pub(super) struct MockExecutor {
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

pub(super) fn ok_result() -> anyhow::Result<gigastt_core::inference::TranscribeResult> {
    Ok(gigastt_core::inference::TranscribeResult {
        text: "ok".into(),
        words: vec![],
        duration_s: 1.0,
        confidence: None,
    })
}

/// Records the body length seen by each attempt so tests can verify the
/// upload survives retries and is only released at a terminal state.
#[derive(Clone)]
pub(super) struct BodyLenRecorder {
    lens: Arc<Mutex<Vec<usize>>>,
}

impl JobExecution for BodyLenRecorder {
    async fn execute(
        &self,
        _id: &str,
        _store: Arc<dyn JobStore>,
        body: Bytes,
        _params: ExportParams,
    ) -> anyhow::Result<gigastt_core::inference::TranscribeResult> {
        self.lens.lock().push(body.len());
        // Retryable failure class, so the body must survive to the next
        // attempt and only be released at the terminal state.
        Err(anyhow::anyhow!("worker thread panicked"))
    }
}

mod bounded_events;
mod events;
mod executor;
mod queue;
mod store;
