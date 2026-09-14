use super::*;

#[tokio::test]
async fn test_timeout_error_keeps_partial_without_reporting_success() {
    let mut assembler = gigastt_core::inference::TranscriptAssembler::new();
    assembler.append(vec![gigastt_core::inference::WordInfo::new(
        "readable", 0.0, 0.5, 0.9, None,
    )]);
    let response = api_inference_timeout_error(Some(assembler.partial(1.0)));
    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    let bytes = axum::body::to_bytes(response.into_body(), 4096)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["code"], "inference_timeout");
    assert_eq!(body["partial"]["text"], "readable");
    assert_eq!(body["partial"]["is_final"], false);
}

#[tokio::test]
async fn test_api_error_basic() {
    let resp = api_error(StatusCode::BAD_REQUEST, "bad request", "bad_request");
    let (parts, body) = resp.into_parts();
    assert_eq!(parts.status, StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(body, 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["error"], "bad request");
    assert_eq!(v["code"], "bad_request");
}

#[tokio::test]
async fn test_override_conflict_error_mapping() {
    // The per-request-knob 409s reuse the shared `api_error` machinery, so
    // they must carry StatusCode::CONFLICT and the stable code an operator's
    // client keys off. Drive it via the same `OverrideError::{code,message}`
    // the handler maps, plus the standalone `variant_not_loaded` guard.
    use gigastt_core::inference::OverrideError;
    for err in [
        OverrideError::VadNotLoaded,
        OverrideError::PunctuationNotAvailable,
    ] {
        let resp = api_error(StatusCode::CONFLICT, err.message(), err.code());
        let (parts, body) = resp.into_parts();
        assert_eq!(parts.status, StatusCode::CONFLICT);
        let bytes = axum::body::to_bytes(body, 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["code"], err.code());
        assert!(v["error"].as_str().is_some_and(|s| !s.is_empty()));
    }
    assert_eq!(OverrideError::VadNotLoaded.code(), "vad_not_loaded");
    assert_eq!(
        OverrideError::PunctuationNotAvailable.code(),
        "punctuation_not_available"
    );

    // Hotword DoS limit violations map to 400 (not 409).
    use gigastt_core::inference::HotwordError;
    for err in [HotwordError::TooManyHotwords, HotwordError::PhraseTooLong] {
        let resp = api_error(StatusCode::BAD_REQUEST, err.message(), err.code());
        let (parts, body) = resp.into_parts();
        assert_eq!(parts.status, StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(body, 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["code"], err.code());
    }

    // The variant guard is a standalone literal (no engine needed to check
    // the code/status contract it emits).
    let resp = api_error(
        StatusCode::CONFLICT,
        "Requested model variant is not loaded",
        "variant_not_loaded",
    );
    let (parts, body) = resp.into_parts();
    assert_eq!(parts.status, StatusCode::CONFLICT);
    let bytes = axum::body::to_bytes(body, 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["code"], "variant_not_loaded");
}

#[tokio::test]
async fn test_api_timeout_error_includes_retry_after() {
    let limits = RuntimeLimits::default();
    let resp = api_timeout_error(&limits);
    let (parts, body) = resp.into_parts();
    assert_eq!(parts.status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        parts
            .headers
            .get(header::RETRY_AFTER)
            .unwrap()
            .to_str()
            .unwrap(),
        pool_retry_after_secs(&limits).to_string()
    );
    let bytes = axum::body::to_bytes(body, 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["code"], "timeout");
    assert_eq!(v["retry_after_ms"], pool_retry_after_ms(&limits));
}

#[tokio::test]
async fn test_api_pool_closed_error_no_retry() {
    let resp = api_pool_closed_error();
    let (parts, body) = resp.into_parts();
    assert_eq!(parts.status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(parts.headers.get(header::RETRY_AFTER).is_none());
    let bytes = axum::body::to_bytes(body, 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["code"], "pool_closed");
    assert!(v.get("retry_after_ms").is_none());
}

#[tokio::test]
async fn test_api_inference_timeout_error_is_504() {
    let resp = api_inference_timeout_error(None);
    let (parts, body) = resp.into_parts();
    assert_eq!(parts.status, StatusCode::GATEWAY_TIMEOUT);
    // A wedged run would just time out again, so no Retry-After hint.
    assert!(parts.headers.get(header::RETRY_AFTER).is_none());
    let bytes = axum::body::to_bytes(body, 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["code"], "inference_timeout");
}

#[tokio::test]
async fn test_api_inference_timeout_error_body_message() {
    // The 504 inference-timeout body should not leak internals, just the
    // stable code + a sanitized message.
    let resp = api_inference_timeout_error(None);
    let bytes = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["code"], "inference_timeout");
    assert_eq!(v["error"], "Inference timed out.");
}

#[tokio::test]
async fn test_api_pool_closed_error_status_and_message() {
    // pool_closed is a 503 with a sanitized "shutting down" message and no
    // retry hint (the pool is not coming back).
    let resp = api_pool_closed_error();
    let (parts, body) = resp.into_parts();
    assert_eq!(parts.status, StatusCode::SERVICE_UNAVAILABLE);
    let bytes = axum::body::to_bytes(body, 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["error"], "Server is shutting down");
    assert_eq!(v["code"], "pool_closed");
}
