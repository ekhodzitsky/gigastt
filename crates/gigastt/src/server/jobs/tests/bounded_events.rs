use super::*;

#[tokio::test]
async fn test_slow_reader_has_bounded_progress_and_receives_terminal() {
    // Exercise each terminal transition with one stalled reader and one
    // reader consuming progress normally. A slow reader must not hold up
    // the store or another reader.
    for transition in [
        JobTransition::Complete(ok_result().unwrap()),
        JobTransition::Fail("Transcription failed.".into()),
        JobTransition::Cancel,
    ] {
        let store = InMemoryJobStore::new(test_limits());
        let id = store
            .create(Job::queued(Bytes::new(), ExportParams::default()))
            .await
            .unwrap();
        store.transition(&id, JobTransition::Start).await.unwrap();
        let (slow_tx, mut slow_rx) = job_event_channel();
        let (live_tx, mut live_rx) = job_event_channel();
        store.subscribe(&id, slow_tx).await.unwrap();
        store.subscribe(&id, live_tx).await.unwrap();

        let cap = u32::try_from(MAX_JOB_EVENT_PROGRESS).unwrap();
        for percent in 1..=cap + 4 {
            broadcast_event(
                &store,
                &id,
                JobEvent::Progress {
                    percent,
                    processed_seconds: f64::from(percent),
                },
            )
            .await;
            assert!(matches!(
                live_rx.try_recv(),
                Ok(JobEvent::Progress { percent: seen, .. }) if seen == percent
            ));
        }
        // Taking one progress item frees room for a new update. The skipped
        // estimates stay skipped, and both listeners can continue normally.
        assert!(matches!(
            slow_rx.try_recv(),
            Ok(JobEvent::Progress { percent: 1, .. })
        ));
        let latest = cap + 5;
        broadcast_event(
            &store,
            &id,
            JobEvent::Progress {
                percent: latest,
                processed_seconds: f64::from(latest),
            },
        )
        .await;
        assert!(matches!(
            live_rx.try_recv(),
            Ok(JobEvent::Progress { percent, .. }) if percent == latest
        ));

        let expected = match &transition {
            JobTransition::Complete(_) => JobStatus::Done,
            JobTransition::Fail(_) => JobStatus::Failed,
            JobTransition::Cancel => JobStatus::Cancelled,
            _ => unreachable!("terminal transition required"),
        };
        store.transition(&id, transition).await.unwrap();
        assert!(terminal_matches(live_rx.try_recv().unwrap(), expected));

        for percent in (2..=cap).chain(std::iter::once(latest)) {
            assert!(matches!(
                slow_rx.try_recv(),
                Ok(JobEvent::Progress { percent: seen, .. }) if seen == percent
            ));
        }
        assert!(terminal_matches(slow_rx.try_recv().unwrap(), expected));
        assert!(
            store
                .get(&id)
                .await
                .unwrap()
                .unwrap()
                .event_channels
                .is_empty()
        );
        for result in [slow_rx.try_recv(), live_rx.try_recv()] {
            assert!(matches!(
                result,
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected)
            ));
        }
    }
}

fn terminal_matches(event: JobEvent, status: JobStatus) -> bool {
    matches!(
        (event, status),
        (JobEvent::Done, JobStatus::Done)
            | (JobEvent::Failed { .. }, JobStatus::Failed)
            | (JobEvent::Cancelled, JobStatus::Cancelled)
    )
}
