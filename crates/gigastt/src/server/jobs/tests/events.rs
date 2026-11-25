use super::*;

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
    // Typed InvalidAudio displays as lowercase "invalid audio: …".
    let decode: anyhow::Error = gigastt_core::error::GigasttError::InvalidAudio {
        reason: "unsupported format".into(),
    }
    .into();
    assert_eq!(
        sanitize_job_error(&decode),
        "Failed to decode audio file. Check format."
    );
    assert_eq!(
        sanitize_job_error(&anyhow::anyhow!("some internal onnx path /foo/bar")),
        "Transcription failed."
    );
    // A typed AudioTooLong (arrives via `anyhow::Error::from`) surfaces the
    // observed/limit seconds so a client can tell "too long" from "corrupt".
    let too_long: anyhow::Error = gigastt_core::error::GigasttError::AudioTooLong {
        observed_secs: 4000.0,
        limit_secs: 1800.0,
    }
    .into();
    assert_eq!(
        sanitize_job_error(&too_long),
        "Audio too long: 4000s exceeds the maximum of 1800s."
    );
}

#[test]
fn test_is_retryable_error_recognizes_transient_failures() {
    // A panic is transient — a fresh worker may succeed. An inference_timeout
    // is deterministic, so it is NOT retryable. A decode error never is.
    assert!(is_retryable_error(&anyhow::anyhow!(
        "worker thread panicked"
    )));
    assert!(!is_retryable_error(&anyhow::anyhow!("inference_timeout")));
    assert!(!is_retryable_error(&anyhow::anyhow!("Invalid audio")));
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
    let (live_tx, mut live_rx) = job_event_channel();
    {
        let (dead_tx, _dead_rx) = job_event_channel();
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

#[test]
fn test_subscribe_prunes_closed_channels() {
    let mut job = Job::queued(Bytes::new(), ExportParams::default());
    let (dead_tx, dead_rx) = job_event_channel();
    job.event_channels.push(dead_tx);
    drop(dead_rx);

    let (live_tx, _live_rx) = job_event_channel();
    job.subscribe(live_tx);

    assert_eq!(job.event_channels.len(), 1);
}

#[test]
fn test_subscribe_evicts_oldest_at_cap() {
    let mut job = Job::queued(Bytes::new(), ExportParams::default());
    let mut rxs = Vec::new();
    for _ in 0..MAX_JOB_EVENT_SUBSCRIBERS {
        let (tx, rx) = job_event_channel();
        job.subscribe(tx);
        rxs.push(rx);
    }
    assert_eq!(job.event_channels.len(), MAX_JOB_EVENT_SUBSCRIBERS);

    let (tx, rx) = job_event_channel();
    job.subscribe(tx);
    rxs.push(rx);

    // The list stays capped; the oldest listener was evicted, so its
    // stream is disconnected.
    assert_eq!(job.event_channels.len(), MAX_JOB_EVENT_SUBSCRIBERS);
    assert!(matches!(
        rxs[0].try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Disconnected)
    ));
}

#[test]
fn test_job_status_response_bounds_percent_for_inexact_duration() {
    let mut job = Job::queued(Bytes::new(), ExportParams::default());
    job.status = JobStatus::Processing;
    job.total_seconds = 10.0;
    job.processed_seconds = 10.5;
    assert_eq!(job_status_response(&job).percent, 100);
    job.total_seconds = 0.0;
    let response = job_status_response(&job);
    assert_eq!(response.percent, 0);
    assert_eq!(response.processed_seconds, 10.5);
}

#[tokio::test]
async fn test_terminal_transition_and_subscription_in_both_orders() {
    use crate::server::jobs::store::JobTransition;
    for subscribe_first in [false, true] {
        let store = InMemoryJobStore::new(test_limits());
        let id = store
            .create(Job::queued(Bytes::new(), ExportParams::default()))
            .await
            .unwrap();
        let (tx, mut rx) = job_event_channel();
        if subscribe_first {
            store.subscribe(&id, tx).await.unwrap();
            store.transition(&id, JobTransition::Cancel).await.unwrap();
        } else {
            store.transition(&id, JobTransition::Cancel).await.unwrap();
            store.subscribe(&id, tx).await.unwrap();
        }
        assert!(matches!(rx.try_recv(), Ok(JobEvent::Cancelled)));
        assert!(matches!(
            rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected)
        ));
    }
}

#[tokio::test]
async fn test_completion_and_cancellation_publish_only_winning_transition() {
    use crate::server::jobs::store::{JobTransition, TransitionOutcome};
    for cancel_first in [false, true] {
        let store = InMemoryJobStore::new(test_limits());
        let id = store
            .create(Job::queued(Bytes::new(), ExportParams::default()))
            .await
            .unwrap();
        store.transition(&id, JobTransition::Start).await.unwrap();
        let (tx, mut rx) = job_event_channel();
        store.subscribe(&id, tx).await.unwrap();
        let completed = JobTransition::Complete(ok_result().unwrap());
        let (first, second) = if cancel_first {
            (JobTransition::Cancel, completed)
        } else {
            (completed, JobTransition::Cancel)
        };
        assert!(matches!(
            store.transition(&id, first).await.unwrap(),
            TransitionOutcome::Applied
        ));
        assert!(matches!(
            store.transition(&id, second).await.unwrap(),
            TransitionOutcome::Rejected(_)
        ));
        let event = rx.try_recv().unwrap();
        assert_eq!(matches!(event, JobEvent::Cancelled), cancel_first);
        assert!(matches!(
            rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected)
        ));
    }
}

#[tokio::test]
async fn test_broadcast_and_subscription_in_both_orders_retain_terminal_delivery() {
    for subscribe_first in [false, true] {
        let store = InMemoryJobStore::new(test_limits());
        let id = store
            .create(Job::queued(Bytes::new(), ExportParams::default()))
            .await
            .unwrap();
        let (tx, mut rx) = job_event_channel();
        let event = JobEvent::Progress {
            percent: 1,
            processed_seconds: 0.1,
        };
        if subscribe_first {
            store.subscribe(&id, tx).await.unwrap();
            broadcast_event(&store, &id, event).await;
            assert!(matches!(rx.try_recv(), Ok(JobEvent::Progress { .. })));
        } else {
            broadcast_event(&store, &id, event).await;
            store.subscribe(&id, tx).await.unwrap();
        }
        store.transition(&id, JobTransition::Cancel).await.unwrap();
        broadcast_event(&store, &id, JobEvent::Done).await;
        assert!(matches!(rx.try_recv(), Ok(JobEvent::Cancelled)));
        assert!(matches!(
            rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected)
        ));
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_concurrent_admission_reports_typed_backpressure() {
    let store = Arc::new(InMemoryJobStore::new(RuntimeLimits {
        jobs_max: 1,
        ..test_limits()
    }));
    let barrier = Arc::new(tokio::sync::Barrier::new(2));
    let mut tasks = Vec::new();
    for _ in 0..2 {
        let store = store.clone();
        let barrier = barrier.clone();
        tasks.push(tokio::spawn(async move {
            assert!(!store.is_full().await);
            barrier.wait().await;
            store
                .create(Job::queued(Bytes::new(), ExportParams::default()))
                .await
        }));
    }
    let mut admitted = 0;
    let mut rejected = 0;
    for task in tasks {
        match task.await.unwrap() {
            Ok(_) => admitted += 1,
            Err(error) => {
                assert!(error.is::<JobStoreFull>());
                rejected += 1;
            }
        }
    }
    assert_eq!((admitted, rejected), (1, 1));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_concurrent_subscribe_broadcast_and_completion_deliver_terminal() {
    let store = Arc::new(InMemoryJobStore::new(test_limits()));
    let id = store
        .create(Job::queued(Bytes::new(), ExportParams::default()))
        .await
        .unwrap();
    store.transition(&id, JobTransition::Start).await.unwrap();
    let barrier = Arc::new(tokio::sync::Barrier::new(3));
    let (tx, mut rx) = job_event_channel();
    let subscriber = tokio::spawn({
        let (store, id, barrier) = (store.clone(), id.clone(), barrier.clone());
        async move {
            barrier.wait().await;
            store.subscribe(&id, tx).await.unwrap()
        }
    });
    let broadcaster = tokio::spawn({
        let (store, id, barrier) = (store.clone(), id.clone(), barrier.clone());
        async move {
            barrier.wait().await;
            broadcast_event(
                &*store,
                &id,
                JobEvent::Progress {
                    percent: 50,
                    processed_seconds: 0.5,
                },
            )
            .await;
        }
    });
    barrier.wait().await;
    store
        .transition(&id, JobTransition::Complete(ok_result().unwrap()))
        .await
        .unwrap();
    assert!(subscriber.await.unwrap());
    broadcaster.await.unwrap();
    let mut events = Vec::new();
    while let Ok(event) = rx.try_recv() {
        events.push(event);
    }
    assert!(matches!(events.last(), Some(JobEvent::Done)));
    assert_eq!(events.iter().filter(|event| event.is_terminal()).count(), 1);
    assert!(matches!(
        rx.try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Disconnected)
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_concurrent_cancel_and_complete_agree_with_final_state() {
    let store = Arc::new(InMemoryJobStore::new(test_limits()));
    let id = store
        .create(Job::queued(
            Bytes::from_static(b"audio"),
            ExportParams::default(),
        ))
        .await
        .unwrap();
    store.transition(&id, JobTransition::Start).await.unwrap();
    let abort = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let partial = Arc::new(gigastt_core::inference::TranscriptSnapshot::default());
    store
        .update(
            &id,
            Box::new({
                let (abort, partial) = (abort.clone(), partial.clone());
                move |j| {
                    j.abort = Some(abort);
                    j.partial = Some(partial);
                }
            }),
        )
        .await
        .unwrap();
    let (tx, mut rx) = job_event_channel();
    store.subscribe(&id, tx).await.unwrap();
    let barrier = Arc::new(tokio::sync::Barrier::new(2));
    let cancel = tokio::spawn({
        let (store, id, barrier) = (store.clone(), id.clone(), barrier.clone());
        async move {
            barrier.wait().await;
            store.transition(&id, JobTransition::Cancel).await.unwrap()
        }
    });
    barrier.wait().await;
    let completion = store
        .transition(&id, JobTransition::Complete(ok_result().unwrap()))
        .await
        .unwrap();
    let cancellation = cancel.await.unwrap();
    let job = store.get(&id).await.unwrap().unwrap();
    let event = rx.try_recv().unwrap();
    match (completion, cancellation) {
        (TransitionOutcome::Applied, TransitionOutcome::Rejected(JobStatus::Done)) => {
            assert_eq!(job.status, JobStatus::Done);
            assert!(matches!(event, JobEvent::Done));
            assert!(job.result.is_some());
        }
        (TransitionOutcome::Rejected(JobStatus::Cancelled), TransitionOutcome::Applied) => {
            assert_eq!(job.status, JobStatus::Cancelled);
            assert!(matches!(event, JobEvent::Cancelled));
            assert!(job.result.is_none());
            assert!(abort.load(std::sync::atomic::Ordering::Relaxed));
            assert!(Arc::ptr_eq(job.partial.as_ref().unwrap(), &partial));
        }
        outcomes => panic!("inconsistent outcomes: {outcomes:?}"),
    }
    assert!(job.body.is_empty());
    assert!(job.abort.is_none());
    assert!(matches!(
        rx.try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Disconnected)
    ));
}
