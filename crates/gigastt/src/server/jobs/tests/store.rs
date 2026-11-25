use super::*;

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
async fn test_store_byte_budget_backpressures_under_count_limit() {
    // Count limit is generous (10) but the byte budget is tiny: a store
    // holding one upload already at/over the byte budget must report full
    // and reject the next submission, even though it holds 1 << 10 jobs.
    let limits = RuntimeLimits {
        jobs_max: 10,
        jobs_max_bytes: 8,
        ..test_limits()
    };
    let store = InMemoryJobStore::new(limits);
    // First upload (11 bytes) is admitted: like the count cap admitting the
    // Nth job, the budget is checked against the bytes already resident (0).
    store
        .create(Job::queued(
            Bytes::from_static(b"audio-bytes"),
            ExportParams::default(),
        ))
        .await
        .unwrap();
    // 11 resident bytes now exceed the 8-byte budget, so the store is full
    // by bytes despite being far below jobs_max, and the next create fails.
    assert!(store.is_full().await);
    let result = store
        .create(Job::queued(
            Bytes::from_static(b"more"),
            ExportParams::default(),
        ))
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_store_byte_budget_released_when_job_terminal() {
    // A terminal job releases its body, so those bytes stop counting against
    // the budget and the queue accepts new work again.
    let limits = RuntimeLimits {
        jobs_max: 10,
        jobs_max_bytes: 8,
        ..test_limits()
    };
    let store = InMemoryJobStore::new(limits);
    let id = store
        .create(Job::queued(
            Bytes::from_static(b"audio-bytes"),
            ExportParams::default(),
        ))
        .await
        .unwrap();
    assert!(store.is_full().await);
    // Release the body the way the worker does at a terminal state.
    store
        .update(
            &id,
            Box::new(|j| {
                j.status = JobStatus::Done;
                j.body = Bytes::new();
            }),
        )
        .await
        .unwrap();
    assert!(!store.is_full().await);
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

/// Implements only the original store surface, as an external backend would.
struct LegacyStore(InMemoryJobStore);

impl JobStore for LegacyStore {
    fn create<'a>(&'a self, job: Job) -> JobStoreFuture<'a, anyhow::Result<String>> {
        self.0.create(job)
    }
    fn get<'a>(&'a self, id: &str) -> JobStoreFuture<'a, anyhow::Result<Option<Job>>> {
        self.0.get(id)
    }
    fn update<'a>(
        &'a self,
        id: &str,
        f: Box<dyn FnOnce(&mut Job) + Send>,
    ) -> JobStoreFuture<'a, anyhow::Result<()>> {
        self.0.update(id, f)
    }
    fn next_queued<'a>(&'a self) -> JobStoreFuture<'a, anyhow::Result<Option<String>>> {
        self.0.next_queued()
    }
    fn requeue<'a>(&'a self, id: &str) -> JobStoreFuture<'a, anyhow::Result<()>> {
        self.0.requeue(id)
    }
    fn is_full<'a>(&'a self) -> JobStoreFuture<'a, bool> {
        self.0.is_full()
    }
}

#[tokio::test]
async fn test_legacy_store_defaults_preserve_lifecycle_and_retry() {
    let store = LegacyStore(InMemoryJobStore::new(test_limits()));
    assert!(store.status("missing").await.unwrap().is_none());
    assert_eq!(
        store
            .transition("missing", JobTransition::Cancel)
            .await
            .unwrap(),
        TransitionOutcome::Missing
    );
    let (tx, _rx) = job_event_channel();
    assert!(!store.subscribe("missing", tx).await.unwrap());
    let id = store
        .create(Job::queued(
            Bytes::from_static(b"audio"),
            ExportParams::default(),
        ))
        .await
        .unwrap();
    assert_eq!(store.next_queued().await.unwrap(), Some(id.clone()));
    assert_eq!(
        store.transition(&id, JobTransition::Start).await.unwrap(),
        TransitionOutcome::Applied
    );
    assert_eq!(
        store.transition(&id, JobTransition::Retry).await.unwrap(),
        TransitionOutcome::Applied
    );
    assert_eq!(store.next_queued().await.unwrap(), Some(id.clone()));
    store.transition(&id, JobTransition::Start).await.unwrap();
    let (tx, mut rx) = job_event_channel();
    assert!(store.subscribe(&id, tx).await.unwrap());
    store
        .transition(&id, JobTransition::Fail("sanitized failure".into()))
        .await
        .unwrap();
    assert!(
        matches!(rx.try_recv(), Ok(JobEvent::Failed { error }) if error == "sanitized failure")
    );
    assert_eq!(
        store.transition(&id, JobTransition::Cancel).await.unwrap(),
        TransitionOutcome::Rejected(JobStatus::Failed)
    );
    let (tx, mut rx) = job_event_channel();
    assert!(store.subscribe(&id, tx).await.unwrap());
    assert!(matches!(rx.try_recv(), Ok(JobEvent::Failed { .. })));
    let owned = store.get(&id).await.unwrap().unwrap();
    assert_eq!(
        serde_json::to_value(store.status(&id).await.unwrap().unwrap()).unwrap(),
        serde_json::to_value(job_status_response(&owned)).unwrap(),
    );
}

#[tokio::test]
async fn test_status_matches_job_projection_and_retains_result_ownership() {
    let store = Arc::new(InMemoryJobStore::new(test_limits()));
    for status in [
        JobStatus::Queued,
        JobStatus::Processing,
        JobStatus::Done,
        JobStatus::Failed,
        JobStatus::Cancelled,
    ] {
        let mut job = Job::queued(Bytes::new(), ExportParams::default());
        job.status = status;
        job.processed_seconds = 2.5;
        job.total_seconds = 10.0;
        job.error = Some("sanitized error".into());
        job.result = Some(ok_result().unwrap());
        let id = store.create(job).await.unwrap();
        let owned = store.get(&id).await.unwrap().unwrap();
        let status = store.status(&id).await.unwrap().unwrap();
        assert_eq!(
            serde_json::to_value(status).unwrap(),
            serde_json::to_value(job_status_response(&owned)).unwrap()
        );
        if owned.status.is_terminal() {
            store.backdate(&id, 7200.0).await;
            store.is_full().await;
            assert!(store.status(&id).await.unwrap().is_none());
            assert_eq!(owned.result.unwrap().text, "ok");
        }
    }
    assert!(store.status("missing").await.unwrap().is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_result_snapshot_survives_concurrent_poll_cancel_and_eviction() {
    let store = Arc::new(InMemoryJobStore::new(test_limits()));
    let mut job = Job::queued(Bytes::new(), ExportParams::default());
    job.status = JobStatus::Done;
    let mut result = ok_result().unwrap();
    result.text = "transcript ".repeat(100_000);
    job.result = Some(result);
    let id = store.create(job).await.unwrap();
    let owned = store.get(&id).await.unwrap().unwrap();
    let reader = {
        let store = store.clone();
        let id = id.clone();
        tokio::spawn(async move {
            for _ in 0..100 {
                if let Some(status) = store.status(&id).await.unwrap() {
                    assert_eq!(status.status, JobStatus::Done);
                }
                if let Some(job) = store.get(&id).await.unwrap() {
                    assert_eq!(job.result.unwrap().text.len(), 1_100_000);
                }
                tokio::task::yield_now().await;
            }
        })
    };
    assert_eq!(
        store.transition(&id, JobTransition::Cancel).await.unwrap(),
        TransitionOutcome::Rejected(JobStatus::Done)
    );
    store.backdate(&id, 7200.0).await;
    store.is_full().await;
    reader.await.unwrap();
    assert!(store.get(&id).await.unwrap().is_none());
    assert_eq!(owned.result.unwrap().text, "transcript ".repeat(100_000));
}
