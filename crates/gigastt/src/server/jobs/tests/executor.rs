use super::*;
use crate::server::jobs::RealJobExecutor;
use arc_swap::ArcSwap;
use gigastt_core::inference::Engine;

fn mock_engine() -> (Arc<Engine>, tempfile::TempDir) {
    let tmp = tempfile::tempdir().expect("tempdir");
    gigastt_core::test_support::write_rnnt_layout(tmp.path()).expect("layout");
    let engine = gigastt_core::test_support::load_rnnt_engine(tmp.path(), 1).expect("engine");
    (Arc::new(engine), tmp)
}

#[tokio::test]
async fn test_real_executor_transcribes_short_wav() {
    let (engine, _tmp) = mock_engine();
    let limits = test_limits();
    let store: Arc<dyn JobStore> = Arc::new(InMemoryJobStore::new(limits.clone()));
    let job = Job::queued(
        Bytes::from(gigastt_core::test_support::pcm16_wav(&[0; 320], 16_000)),
        ExportParams::default(),
    );
    let id = store.create(job).await.expect("create");

    let executor = RealJobExecutor::new(
        Arc::new(ArcSwap::new(engine)),
        Arc::new(ArcSwap::from_pointee(limits)),
        tokio_util::sync::CancellationToken::new(),
    );
    let result = executor
        .execute(
            &id,
            store.clone(),
            Bytes::from(gigastt_core::test_support::pcm16_wav(&[0; 320], 16_000)),
            ExportParams::default(),
        )
        .await
        .expect("executor");
    assert!(result.text.is_empty());
    assert!(result.duration_s > 0.0);
}

#[tokio::test]
async fn test_real_executor_rejects_unknown_variant() {
    let (engine, _tmp) = mock_engine();
    let limits = test_limits();
    let store: Arc<dyn JobStore> = Arc::new(InMemoryJobStore::new(limits.clone()));
    let params = ExportParams {
        variant: Some("does-not-exist".into()),
        ..ExportParams::default()
    };
    let job = Job::queued(Bytes::from(vec![1, 2, 3]), params.clone());
    let id = store.create(job).await.expect("create");

    let executor = RealJobExecutor::new(
        Arc::new(ArcSwap::new(engine)),
        Arc::new(ArcSwap::from_pointee(limits)),
        tokio_util::sync::CancellationToken::new(),
    );
    let err = executor
        .execute(&id, store, Bytes::from(vec![1, 2, 3]), params)
        .await
        .expect_err("unknown variant");
    assert!(
        format!("{err:#}").contains("not loaded"),
        "unexpected error: {err:#}"
    );
}

#[test]
fn test_jobs_file_zero_context_rejects_conflicting_modes_with_explicit_error() {
    use crate::server::jobs::executor::validate_file_zero_context_params;
    let mut options = ExportParams::default();
    assert!(validate_file_zero_context_params(&options).is_ok());
    options.channels = Some("split".into());
    let error = validate_file_zero_context_params(&options).unwrap_err();
    assert!(super::sanitize_job_error(&error).contains("File zero context"));
    assert!(!super::is_retryable_error(&error));
    options.channels = None;
    options.diarization = Some(true);
    assert!(validate_file_zero_context_params(&options).is_err());
    options.diarization = None;
    options.vad = Some(true);
    assert!(validate_file_zero_context_params(&options).is_err());
    options.vad = Some(false);
    options.hotwords = Some("".into());
    assert!(validate_file_zero_context_params(&options).is_ok());
    options.hotwords = Some("тест".into());
    assert!(validate_file_zero_context_params(&options).is_err());
}
