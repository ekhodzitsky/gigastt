//! Job HTTP handlers — model-free via the mock engine + in-memory store.

use super::*;
use crate::server::http::{
    JobServerState, cancel_job, get_job, get_job_result, job_events, submit_job,
};
use crate::server::jobs::{InMemoryJobStore, Job, JobQueue, JobStatus, JobStore};
use axum::extract::Path;
use axum::response::IntoResponse;

fn jobs_state(engine: Arc<Engine>, limits: RuntimeLimits) -> Arc<AppState> {
    let store: Arc<dyn crate::server::jobs::JobStore> =
        Arc::new(InMemoryJobStore::new(limits.clone()));
    let shutdown = tokio_util::sync::CancellationToken::new();
    let queue = JobQueue::new(store.clone(), 1, 0, shutdown.clone());
    Arc::new(AppState {
        engine: engine_swap(engine),
        limits: Arc::new(ArcSwap::from_pointee(limits)),
        metrics_registry: None,
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        shutdown,
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: Some(JobServerState { store, queue }),
    })
}

#[tokio::test]
async fn test_submit_job_disabled_is_404() {
    let state = Arc::new(AppState {
        engine: engine_swap(test_engine()),
        limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
        metrics_registry: None,
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        shutdown: tokio_util::sync::CancellationToken::new(),
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: None,
    });
    let err = submit_job(State(state), Query(ExportParams::default()), short_wav())
        .await
        .expect_err("jobs disabled");
    assert_eq!(err.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_submit_job_empty_body_is_400() {
    let state = jobs_state(test_engine(), RuntimeLimits::default());
    let err = submit_job(State(state), Query(ExportParams::default()), Bytes::new())
        .await
        .expect_err("empty");
    assert_eq!(err.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_submit_job_payload_too_large() {
    let state = jobs_state(
        test_engine(),
        RuntimeLimits {
            body_limit_bytes: 4,
            ..RuntimeLimits::default()
        },
    );
    let err = submit_job(State(state), Query(ExportParams::default()), short_wav())
        .await
        .expect_err("too large");
    assert_eq!(err.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn test_submit_job_conflict_split_and_diarization() {
    let state = jobs_state(test_engine(), RuntimeLimits::default());
    let params = ExportParams {
        channels: Some("split".into()),
        diarization: Some(true),
        ..ExportParams::default()
    };
    let err = submit_job(State(state), Query(params), short_wav())
        .await
        .expect_err("conflict");
    assert_eq!(err.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_submit_get_cancel_job_round_trip() {
    let state = jobs_state(test_engine(), RuntimeLimits::default());
    let resp = submit_job(
        State(state.clone()),
        Query(ExportParams::default()),
        short_wav(),
    )
    .await
    .expect("submit");
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    let bytes = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let id = v["job_id"].as_str().unwrap().to_string();

    let resp = get_job(State(state.clone()), Path(id.clone()))
        .await
        .expect("get");
    assert_eq!(resp.status(), StatusCode::OK);

    let err = get_job_result(State(state.clone()), Path(id.clone()))
        .await
        .expect_err("not finished");
    assert_eq!(err.status(), StatusCode::CONFLICT);

    let resp = cancel_job(State(state.clone()), Path(id.clone()))
        .await
        .expect("cancel");
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);

    let err = get_job(State(state.clone()), Path("missing".into()))
        .await
        .expect_err("missing");
    assert_eq!(err.status(), StatusCode::NOT_FOUND);

    let events = job_events(State(state), Path(id)).await.expect("events");
    drop(events);
}

#[tokio::test]
async fn test_get_job_result_returns_json_when_done() {
    let limits = RuntimeLimits::default();
    let store = Arc::new(InMemoryJobStore::new(limits.clone()));
    let mut job = Job::queued(short_wav(), ExportParams::default());
    job.status = JobStatus::Done;
    job.result = Some(sample_export_result());
    let id = store.create(job).await.expect("create");

    let shutdown = tokio_util::sync::CancellationToken::new();
    let queue = JobQueue::new(store.clone(), 1, 0, shutdown.clone());
    let state = Arc::new(AppState {
        engine: engine_swap(test_engine()),
        limits: Arc::new(ArcSwap::from_pointee(limits)),
        metrics_registry: None,
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        shutdown,
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: Some(JobServerState { store, queue }),
    });
    let resp = get_job_result(State(state), Path(id))
        .await
        .expect("result");
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 16)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["text"], "привет мир");
}

#[tokio::test]
async fn test_submit_job_capacity_returns_retryable_backpressure() {
    let state = jobs_state(
        test_engine(),
        RuntimeLimits {
            jobs_max: 1,
            ..RuntimeLimits::default()
        },
    );
    submit_job(
        State(state.clone()),
        Query(ExportParams::default()),
        short_wav(),
    )
    .await
    .unwrap();
    let error = submit_job(State(state), Query(ExportParams::default()), short_wav())
        .await
        .unwrap_err();
    let response = error.into_response();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert!(
        response
            .headers()
            .contains_key(axum::http::header::RETRY_AFTER)
    );
    let bytes = axum::body::to_bytes(response.into_body(), 1024)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["code"], "queue_full");
    assert!(json["retry_after_ms"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn test_cancel_completed_job_preserves_terminal_event() {
    use crate::server::jobs::{JobEvent, JobTransition};
    let state = jobs_state(test_engine(), RuntimeLimits::default());
    let store = &state.jobs.as_ref().unwrap().store;
    let id = store
        .create(Job::queued(short_wav(), ExportParams::default()))
        .await
        .unwrap();
    store.transition(&id, JobTransition::Start).await.unwrap();
    let (tx, mut rx) = crate::server::jobs::job_event_channel();
    store.subscribe(&id, tx).await.unwrap();
    store
        .transition(&id, JobTransition::Complete(sample_export_result()))
        .await
        .unwrap();
    let error = cancel_job(State(state.clone()), Path(id.clone()))
        .await
        .unwrap_err();
    assert_eq!(error.status(), StatusCode::CONFLICT);
    assert_eq!(
        store.get(&id).await.unwrap().unwrap().status,
        JobStatus::Done
    );
    assert!(matches!(rx.try_recv(), Ok(JobEvent::Done)));
    assert!(matches!(
        rx.try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Disconnected)
    ));
}

struct AdmissionStore {
    inner: InMemoryJobStore,
    advisory_full: bool,
}

impl JobStore for AdmissionStore {
    fn create<'a>(
        &'a self,
        _job: Job,
    ) -> crate::server::jobs::JobStoreFuture<'a, anyhow::Result<String>> {
        Box::pin(async move {
            assert!(
                !self.advisory_full,
                "legacy full stores must be rejected before create"
            );
            Err(crate::server::jobs::JobStoreFull.into())
        })
    }
    fn get<'a>(
        &'a self,
        id: &str,
    ) -> crate::server::jobs::JobStoreFuture<'a, anyhow::Result<Option<Job>>> {
        self.inner.get(id)
    }
    fn update<'a>(
        &'a self,
        id: &str,
        f: Box<dyn FnOnce(&mut Job) + Send>,
    ) -> crate::server::jobs::JobStoreFuture<'a, anyhow::Result<()>> {
        self.inner.update(id, f)
    }
    fn next_queued<'a>(
        &'a self,
    ) -> crate::server::jobs::JobStoreFuture<'a, anyhow::Result<Option<String>>> {
        self.inner.next_queued()
    }
    fn requeue<'a>(
        &'a self,
        id: &str,
    ) -> crate::server::jobs::JobStoreFuture<'a, anyhow::Result<()>> {
        self.inner.requeue(id)
    }
    fn is_full<'a>(&'a self) -> crate::server::jobs::JobStoreFuture<'a, bool> {
        Box::pin(async move { self.advisory_full })
    }
}

#[tokio::test]
async fn test_submit_job_handles_advisory_and_atomic_capacity_rejections() {
    for advisory_full in [false, true] {
        let mut state = jobs_state(test_engine(), RuntimeLimits::default());
        Arc::get_mut(&mut state)
            .unwrap()
            .jobs
            .as_mut()
            .unwrap()
            .store = Arc::new(AdmissionStore {
            inner: InMemoryJobStore::new(RuntimeLimits::default()),
            advisory_full,
        });
        let error = submit_job(State(state), Query(ExportParams::default()), short_wav())
            .await
            .unwrap_err();
        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(
            response
                .headers()
                .contains_key(axum::http::header::RETRY_AFTER)
        );
        let bytes = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["code"], "queue_full");
    }
}

#[tokio::test]
async fn test_job_sse_ends_after_terminal_even_with_job_snapshot() {
    let state = jobs_state(test_engine(), RuntimeLimits::default());
    let store = &state.jobs.as_ref().unwrap().store;
    let id = store
        .create(Job::queued(Bytes::new(), ExportParams::default()))
        .await
        .unwrap();
    let response = job_events(State(state.clone()), Path(id.clone()))
        .await
        .unwrap()
        .into_response();
    // A cloned job can retain a sender independently of the store.
    let snapshot = store.get(&id).await.unwrap().unwrap();
    assert_eq!(snapshot.event_channels.len(), 1);
    store
        .transition(&id, crate::server::jobs::JobTransition::Cancel)
        .await
        .unwrap();
    let bytes = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        axum::body::to_bytes(response.into_body(), 1024),
    )
    .await
    .expect("terminal event ends the response")
    .unwrap();
    assert_eq!(
        std::str::from_utf8(&bytes).unwrap(),
        "data: {\"type\":\"cancelled\"}\n\n"
    );
    drop(snapshot);
}
