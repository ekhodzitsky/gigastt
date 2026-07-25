//! REST API error helpers shared by HTTP handlers.

use axum::http::StatusCode;
use axum::http::header;
use axum::response::{IntoResponse, Json, Response};

use super::super::config::{RuntimeLimits, pool_retry_after_ms, pool_retry_after_secs};

/// Error response produced by the REST handlers. Using `Response` directly
/// (rather than a `(StatusCode, Json<_>)` tuple) lets timeout paths attach
/// a `Retry-After` header without changing the handler signatures.
pub(super) type ApiError = Response;

pub(super) fn api_error(status: StatusCode, msg: &str, code: &str) -> ApiError {
    (
        status,
        Json(serde_json::json!({"error": msg, "code": code})),
    )
        .into_response()
}

/// 503 response for pool-saturation backpressure: carries both the standard
/// `Retry-After` header (seconds, per RFC 9110 §10.2.3) and a machine-readable
/// `retry_after_ms` field in the JSON body so clients on either surface can
/// back off with the same hint.
pub(super) fn api_timeout_error(limits: &RuntimeLimits) -> ApiError {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [(
            header::RETRY_AFTER,
            pool_retry_after_secs(limits).to_string(),
        )],
        Json(serde_json::json!({
            "error": "Server busy, try again later",
            "code": "timeout",
            "retry_after_ms": pool_retry_after_ms(limits),
        })),
    )
        .into_response()
}

/// 503 response for the case where the pool was closed (graceful shutdown
/// in progress). Distinct from `timeout` so clients can decide whether to
/// retry: a closed pool is not coming back, so no `retry_after_ms` hint.
pub(super) fn api_pool_closed_error() -> ApiError {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(serde_json::json!({
            "error": "Server is shutting down",
            "code": "pool_closed",
        })),
    )
        .into_response()
}

/// 504 response for a single inference run that exceeded the per-request
/// inference timeout (`--inference-timeout-secs`). Distinct from the pool
/// `timeout` (503): the slot was free, the *run* itself was too slow / wedged,
/// so there is no `Retry-After` — retrying the same payload would time out
/// again. Extracted (mirroring [`api_timeout_error`]) so the status + code can
/// be asserted without a model.
pub(super) fn api_inference_timeout_error() -> ApiError {
    api_error(
        StatusCode::GATEWAY_TIMEOUT,
        "Inference timed out.",
        "inference_timeout",
    )
}
