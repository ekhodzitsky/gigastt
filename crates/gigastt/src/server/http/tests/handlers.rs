use super::*;

#[tokio::test]
async fn test_readiness_when_shutdown_cancelled() {
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
    state.shutdown.cancel();
    let resp = readiness(State(state)).await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let bytes = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["status"], "not_ready");
    assert_eq!(v["reason"], "shutting_down");
}

#[tokio::test]
async fn test_readiness_when_pool_exhausted() {
    let engine = fresh_engine();
    let _guards: Vec<_> = (0..engine.pool.total())
        .map(|_| engine.pool.checkout_blocking().unwrap())
        .collect();
    let state = Arc::new(AppState {
        engine: engine_swap(engine),
        limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
        metrics_registry: None,
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        shutdown: tokio_util::sync::CancellationToken::new(),
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: None,
    });
    let resp = readiness(State(state)).await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let bytes = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["status"], "not_ready");
    assert_eq!(v["reason"], "pool_exhausted");
}

#[tokio::test]
async fn test_transcribe_payload_too_large() {
    let state = Arc::new(AppState {
        engine: engine_swap(test_engine()),
        limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits {
            body_limit_bytes: 10,
            ..RuntimeLimits::default()
        })),
        metrics_registry: None,
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        shutdown: tokio_util::sync::CancellationToken::new(),
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: None,
    });
    let body = Bytes::from(vec![0u8; 100]);
    let result = transcribe(State(state), Query(ExportParams::default()), body).await;
    match result {
        Err(resp) => assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE),
        Ok(_) => panic!("expected payload_too_large error"),
    }
}

#[tokio::test]
async fn test_transcribe_channels_split_diarization_conflict_returns_400() {
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
    let params = ExportParams {
        channels: Some("split".into()),
        diarization: Some(true),
        ..ExportParams::default()
    };
    let resp = transcribe(State(state), Query(params), minimal_wav())
        .await
        .unwrap_err();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["code"], "conflicting_modes");
}

#[tokio::test]
async fn test_models_with_metrics() {
    let state = Arc::new(AppState {
        engine: engine_swap(test_engine()),
        limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
        metrics_registry: Some(Arc::new(MetricsRegistry::new())),
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        shutdown: tokio_util::sync::CancellationToken::new(),
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: None,
    });
    let resp = models(State(state)).await;
    let json = serde_json::to_value(&*resp).unwrap();
    // The id reflects the head actually loaded on disk (rnnt or e2e_rnnt),
    // not a hardcoded literal, so assert the stable shape instead.
    let id = json["id"].as_str().unwrap();
    assert!(
        id == "gigaam-v3-rnnt" || id == "gigaam-v3-e2e-rnnt",
        "unexpected model id: {id}"
    );
    assert_eq!(
        json["variant"],
        if id.contains("e2e") {
            "e2e_rnnt"
        } else {
            "rnnt"
        }
    );
}

#[tokio::test]
async fn test_readiness_with_metrics() {
    let state = Arc::new(AppState {
        engine: engine_swap(fresh_engine()),
        limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
        metrics_registry: Some(Arc::new(MetricsRegistry::new())),
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        shutdown: tokio_util::sync::CancellationToken::new(),
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: None,
    });
    let resp = readiness(State(state)).await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_transcribe_pool_closed() {
    let engine = fresh_engine();
    engine.pool.close();
    let state = Arc::new(AppState {
        engine: engine_swap(engine),
        limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
        metrics_registry: None,
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        shutdown: tokio_util::sync::CancellationToken::new(),
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: None,
    });
    let body = Bytes::from(vec![0u8; 100]);
    let result = transcribe(State(state), Query(ExportParams::default()), body).await;
    match result {
        Err(resp) => assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE),
        Ok(_) => panic!("expected pool_closed error"),
    }
}

#[tokio::test]
async fn test_transcribe_stream_invalid_audio() {
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
    let body = Bytes::from(vec![0u8; 100]);
    let result =
        transcribe_stream(State(state), axum::extract::Query(Default::default()), body).await;
    match result {
        Err(resp) => assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY),
        Ok(_) => panic!("expected invalid_audio error"),
    }
}

#[tokio::test]
async fn test_transcribe_stream_invalid_commit_policy_is_bad_request() {
    let query = serde_json::from_value(serde_json::json!({"commit_policy":"unknown"})).unwrap();
    let Err(error) = transcribe_stream(
        State(bare_state(test_engine())),
        axum::extract::Query(query),
        Bytes::new(),
    )
    .await
    else {
        panic!("invalid policy must be rejected before audio decoding");
    };
    assert_eq!(error.status(), StatusCode::BAD_REQUEST);
    let body = axum::body::to_bytes(error.into_body(), usize::MAX)
        .await
        .unwrap();
    let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(payload["code"], "invalid_commit_policy");
}

#[tokio::test]
async fn test_transcribe_stream_payload_too_large() {
    let state = Arc::new(AppState {
        engine: engine_swap(test_engine()),
        limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits {
            body_limit_bytes: 10,
            ..RuntimeLimits::default()
        })),
        metrics_registry: None,
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        shutdown: tokio_util::sync::CancellationToken::new(),
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: None,
    });
    let body = Bytes::from(vec![0u8; 100]);
    let result =
        transcribe_stream(State(state), axum::extract::Query(Default::default()), body).await;
    match result {
        Err(resp) => assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE),
        Ok(_) => panic!("expected payload_too_large error"),
    }
}

#[tokio::test]
async fn test_transcribe_stream_pool_closed() {
    let engine = fresh_engine();
    engine.pool.close();
    let state = Arc::new(AppState {
        engine: engine_swap(engine),
        limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
        metrics_registry: None,
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        shutdown: tokio_util::sync::CancellationToken::new(),
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: None,
    });
    let body = minimal_wav();
    let result =
        transcribe_stream(State(state), axum::extract::Query(Default::default()), body).await;
    match result {
        Err(resp) => assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE),
        Ok(_) => panic!("expected pool_closed error"),
    }
}

#[tokio::test]
async fn test_transcribe_with_metrics() {
    let state = Arc::new(AppState {
        engine: engine_swap(test_engine()),
        limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
        metrics_registry: Some(Arc::new(MetricsRegistry::new())),
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        shutdown: tokio_util::sync::CancellationToken::new(),
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: None,
    });
    let body = short_wav();
    match transcribe(State(state), Query(ExportParams::default()), body).await {
        Ok(_) => {}
        Err(_) => panic!("transcribe with metrics failed"),
    }
}

#[tokio::test]
async fn test_transcribe_stream_with_metrics() {
    let state = Arc::new(AppState {
        engine: engine_swap(test_engine()),
        limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
        metrics_registry: Some(Arc::new(MetricsRegistry::new())),
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        shutdown: tokio_util::sync::CancellationToken::new(),
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: None,
    });
    let body = short_wav();
    match transcribe_stream(State(state), axum::extract::Query(Default::default()), body).await {
        Ok(_) => {}
        Err(_) => panic!("transcribe_stream with metrics failed"),
    }
}

#[tokio::test]
async fn test_transcribe_segments_json() {
    // `?segments=true` on the default JSON response adds a `segments` array
    // with sane start/end ordering and per-segment words, while keeping the
    // top-level text/words/duration contract.
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
    let params = ExportParams {
        segments: Some(true),
        ..ExportParams::default()
    };
    let resp = transcribe(State(state), Query(params), short_wav())
        .await
        .expect("transcribe with segments should succeed");
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    // Top-level contract is preserved.
    assert!(v.get("text").is_some());
    assert!(v.get("words").is_some());
    assert!(v.get("duration").is_some());
    // The segments array is present and every segment has monotonic timing.
    let segments = v["segments"].as_array().expect("segments array present");
    for seg in segments {
        let start = seg["start"].as_f64().unwrap();
        let end = seg["end"].as_f64().unwrap();
        assert!(end >= start, "segment end {end} < start {start}");
        assert!(seg["words"].is_array());
    }
}

fn bare_state(engine: Arc<Engine>) -> Arc<AppState> {
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

#[tokio::test]
async fn test_health_reports_mock_rnnt_identity() {
    let resp = health(State(bare_state(test_engine()))).await;
    let json = serde_json::to_value(&*resp).unwrap();
    assert_eq!(json["status"], "ok");
    assert_eq!(json["variant"], "rnnt");
    assert_eq!(json["model"], "gigaam-v3-rnnt");
    assert_eq!(json["version"], env!("CARGO_PKG_VERSION"));
}

#[tokio::test]
async fn test_metrics_disabled_returns_404() {
    let resp = metrics(State(bare_state(test_engine()))).await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_metrics_enabled_returns_prometheus_text() {
    let registry = Arc::new(MetricsRegistry::new());
    registry.counter_inc("gigastt_http_requests_total", &[], 1);
    let state = Arc::new(AppState {
        engine: engine_swap(test_engine()),
        limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
        metrics_registry: Some(registry),
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        shutdown: tokio_util::sync::CancellationToken::new(),
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: None,
    });
    let resp = metrics(State(state)).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(
        text.contains("gigastt_http_requests_total"),
        "prometheus body should include the recorded counter, got {text:?}"
    );
}

#[tokio::test]
async fn test_readiness_ready_when_pool_has_a_slot() {
    let resp = readiness(State(bare_state(fresh_engine()))).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["status"], "ready");
    assert!(v["pool_available"].as_u64().unwrap() >= 1);
}

#[tokio::test]
async fn test_transcribe_stream_empty_body_is_bad_request() {
    let result = transcribe_stream(
        State(bare_state(test_engine())),
        axum::extract::Query(Default::default()),
        Bytes::new(),
    )
    .await;
    let resp = result.expect_err("empty body");
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_transcribe_empty_body_is_bad_request() {
    let result = transcribe(
        State(bare_state(test_engine())),
        Query(ExportParams::default()),
        Bytes::new(),
    )
    .await;
    let resp = result.expect_err("empty body");
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}
