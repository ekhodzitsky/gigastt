//! Cross-mode contracts not established by routing metadata alone.
use super::*;
use crate::server::jobs::{
    InMemoryJobStore, Job, JobEvent, JobExecution, JobStatus, JobStore, JobTransition,
    RealJobExecutor, TransitionOutcome,
};
use gigastt_core::inference::TranscribeResult;

fn number_engine() -> (
    Arc<Engine>,
    gigastt_core::test_support::FailingStreamFactory,
) {
    let dir = tempfile::tempdir().unwrap();
    gigastt_core::test_support::write_rnnt_layout(dir.path()).unwrap();
    std::fs::write(dir.path().join("v3_vocab.txt"), "▁двадцать\n<blk>\n").unwrap();
    // Unarmed, this existing runtime emits the number word on each frame.
    let factory = gigastt_core::test_support::FailingStreamFactory::new(false, false);
    let engine = gigastt_core::test_support::load_rnnt_engine_with_factory(
        dir.path(),
        1,
        Box::new(factory.clone()),
    )
    .unwrap()
    .with_itn(true);
    (Arc::new(engine), factory)
}

fn contract_state(engine: Arc<Engine>) -> Arc<AppState> {
    Arc::new(AppState {
        engine: engine_swap(engine),
        limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
        metrics_registry: None,
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        shutdown: tokio_util::sync::CancellationToken::new(),
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: None,
    })
}

fn channel_fixture(channels: u16, dual: bool) -> Bytes {
    match (channels, dual) {
        (1, _) => Bytes::from(gigastt_core::test_support::pcm16_wav(&[500; 1600], 16000)),
        (2, true) => Bytes::from_static(include_bytes!(
            "../../../../../gigastt-core/tests/fixtures/opus/dual.ogg"
        )),
        (2, false) => Bytes::from_static(include_bytes!(
            "../../../../../gigastt-core/tests/fixtures/opus/late_stereo.ogg"
        )),
        _ => unreachable!("test fixture channel layout"),
    }
}

async fn rest_result(state: Arc<AppState>, params: ExportParams, body: Bytes) -> serde_json::Value {
    let response = transcribe(State(state), Query(params), body).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    serde_json::from_slice(&body).unwrap()
}

#[tokio::test]
async fn test_rest_and_job_split_modes_preserve_observable_request_override() {
    let (engine, _) = number_engine();
    let state = contract_state(engine.clone());
    for (channels, dual) in [(1, false), (2, true), (2, false)] {
        let body = channel_fixture(channels, dual);
        let params = ExportParams {
            channels: Some("split".into()),
            itn: Some(false),
            vad: Some(false),
            punctuation: Some(false),
            hotwords: Some(String::new()),
            ..Default::default()
        };
        let mut inherited = params.clone();
        inherited.itn = None;
        let baseline = rest_result(state.clone(), inherited, body.clone()).await;
        let rest = rest_result(state.clone(), params.clone(), body.clone()).await;
        assert!(rest["text"].as_str().unwrap().contains("двадцать"));
        assert_ne!(
            rest["text"], baseline["text"],
            "fixture must expose dropped ITN override"
        );

        let store = Arc::new(InMemoryJobStore::new(RuntimeLimits::default()));
        let id = store
            .create(Job::queued(body.clone(), params.clone()))
            .await
            .unwrap();
        store.transition(&id, JobTransition::Start).await.unwrap();
        let (tx, mut rx) = crate::server::jobs::job_event_channel();
        store.subscribe(&id, tx).await.unwrap();
        let executor = RealJobExecutor::new(
            state.engine.clone(),
            state.limits.clone(),
            state.shutdown.clone(),
        );
        let job = executor
            .execute(&id, store.clone(), body, params)
            .await
            .unwrap();
        assert_eq!(rest["text"], job.text);
        assert_eq!(
            rest["words"],
            serde_json::from_slice::<serde_json::Value>(&serde_json::to_vec(&job.words).unwrap())
                .unwrap()
        );
        assert_eq!(rest["duration"], job.duration_s);
        assert_eq!(
            rest["confidence"],
            serde_json::from_slice::<serde_json::Value>(
                &serde_json::to_vec(&job.confidence).unwrap()
            )
            .unwrap()
        );
        let speakers: std::collections::BTreeSet<_> =
            job.words.iter().map(|word| word.speaker).collect();
        assert_eq!(
            speakers,
            if channels == 2 && !dual {
                [Some(0), Some(1)].into()
            } else {
                [None].into()
            }
        );
        assert!(matches!(
            store
                .transition(&id, JobTransition::Complete(job))
                .await
                .unwrap(),
            TransitionOutcome::Applied
        ));
        let stored = store.get(&id).await.unwrap().unwrap();
        assert_eq!(stored.status, JobStatus::Done);
        assert_eq!(
            stored.processed_seconds,
            stored.result.as_ref().unwrap().duration_s
        );
        let mut terminal = Vec::new();
        while let Ok(event) = rx.try_recv() {
            if event.is_terminal() {
                terminal.push(event);
            }
        }
        assert!(matches!(terminal.as_slice(), [JobEvent::Done]));
        assert_eq!(engine.pool.available(), engine.pool.total());
    }
}

#[tokio::test]
async fn test_cancel_before_executor_registration_preserves_terminal_subscription() {
    let (engine, factory) = number_engine();
    let state = contract_state(engine.clone());
    factory.arm(1); // Any accidental recognition returns a distinguishable inference error.
    let body = channel_fixture(2, false);
    let params = ExportParams {
        channels: Some("split".into()),
        ..Default::default()
    };
    let store = Arc::new(InMemoryJobStore::new(RuntimeLimits::default()));
    let id = store
        .create(Job::queued(body.clone(), params.clone()))
        .await
        .unwrap();
    store.transition(&id, JobTransition::Start).await.unwrap();
    let (tx, mut rx) = crate::server::jobs::job_event_channel();
    store.subscribe(&id, tx).await.unwrap();
    store.transition(&id, JobTransition::Cancel).await.unwrap();
    let executor = RealJobExecutor::new(
        state.engine.clone(),
        state.limits.clone(),
        state.shutdown.clone(),
    );
    let error = executor
        .execute(&id, store.clone(), body, params)
        .await
        .unwrap_err();
    assert!(matches!(
        error.downcast_ref::<gigastt_core::error::GigasttError>(),
        Some(gigastt_core::error::GigasttError::Cancelled)
    ));
    let late = TranscribeResult {
        text: "late result".into(),
        words: Vec::new(),
        duration_s: 0.1,
        confidence: None,
    };
    assert!(matches!(
        store
            .transition(&id, JobTransition::Complete(late))
            .await
            .unwrap(),
        TransitionOutcome::Rejected(_)
    ));
    assert!(matches!(rx.try_recv(), Ok(JobEvent::Cancelled)));
    assert!(matches!(
        rx.try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Disconnected)
    ));
    let stored = store.get(&id).await.unwrap().unwrap();
    assert_eq!(stored.status, JobStatus::Cancelled);
    assert!(stored.result.is_none());
    assert!(stored.abort.is_none());
    assert_eq!(engine.pool.available(), engine.pool.total());
}
