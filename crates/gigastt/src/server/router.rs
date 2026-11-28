//! Protected `/v1` route table (REST + WebSocket + optional jobs).

use super::upload::{UploadAdmission, UploadPermit};
use axum::body::Bytes;
use axum::http::StatusCode;
use axum::routing::{delete, get, options, post};
use axum::{Extension, Router};
use std::sync::Arc;

use super::http;
use super::ws;

/// New explicit v4 request selectors are reserved, never interpreted as
/// legacy aliases. Other historic OpenAI model aliases remain compatible.
pub(crate) fn is_v4_model_id(id: &str) -> bool {
    id.starts_with("gigaam-v4-")
}
pub(crate) fn is_v4_selector(key: &str) -> bool {
    matches!(
        key,
        "model_family"
            | "model_head"
            | "transcript_style"
            | "model_precision"
            | "v4_mkl_runtime_dir"
    )
}
fn requests_v4(query: &[(String, String)]) -> bool {
    query.iter().any(|(key, id)| {
        is_v4_selector(key) || (matches!(key.as_str(), "variant" | "model") && is_v4_model_id(id))
    })
}
async fn reject_v4_on_legacy(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if request.method() == axum::http::Method::POST
        && matches!(
            request.uri().path(),
            "/v1/transcribe" | "/v1/transcribe/stream" | "/v1/audio/transcriptions" | "/v1/jobs"
        )
        && let Ok(axum::extract::Query(query)) =
            axum::extract::Query::<Vec<(String, String)>>::try_from_uri(request.uri())
        && requests_v4(&query)
    {
        return (StatusCode::CONFLICT,axum::Json(serde_json::json!({"error":"Explicit v4 model selection requires a v4 file server","code":"variant_not_loaded"}))).into_response();
    }
    next.run(request).await
}

/// Routes gated by origin middleware (and the per-IP limiter when enabled).
/// `/metrics` is intentionally absent — it lives on the loopback listener.
pub(crate) fn protected_v1_router(
    jobs_enabled: bool,
    admission: UploadAdmission,
) -> Router<Arc<http::AppState>> {
    let protected = Router::new()
        .route("/v1/models", get(http::models))
        .route("/v1/models", options(|| async { StatusCode::NO_CONTENT }))
        .route(
            "/v1/transcribe",
            post(
                |state, query, Extension(permit): Extension<UploadPermit>, body: Bytes| async move {
                    http::transcribe(state, query, permit.retain(body)).await
                },
            ),
        )
        .route(
            "/v1/transcribe",
            options(|| async { StatusCode::NO_CONTENT }),
        )
        .route(
            "/v1/transcribe/stream",
            post(
                |state, query, Extension(permit): Extension<UploadPermit>, body: Bytes| async move {
                    http::transcribe_stream(state, query, permit.retain(body)).await
                },
            ),
        )
        .route(
            "/v1/transcribe/stream",
            options(|| async { StatusCode::NO_CONTENT }),
        )
        // OpenAI-compatible alias for clients (llama-swap, Hermes Agent, SDKs
        // with a custom base_url) that POST multipart `file` + `model`.
        .route(
            "/v1/audio/transcriptions",
            post(http::openai_transcriptions_admitted),
        )
        .route(
            "/v1/audio/transcriptions",
            options(|| async { StatusCode::NO_CONTENT }),
        )
        // /v1/ws is the canonical WebSocket path (versioned, aligned with REST).
        .route("/v1/ws", get(ws::ws_handler))
        .route("/v1/ws", options(|| async { StatusCode::NO_CONTENT }))
        // Admin: hot-reload the model without a restart. Registered inside the
        // protected router so it inherits `origin_middleware`, but the handler
        // additionally enforces a strict loopback peer check (see `http::reload`)
        // so it stays local even under `--bind-all` / `--cors-allow-any`.
        .route("/v1/admin/reload", post(http::reload))
        .route(
            "/v1/admin/reload",
            options(|| async { StatusCode::NO_CONTENT }),
        );

    // Asynchronous job API routes. Only registered when `--enable-jobs` is set;
    // without the flag the paths fall through to axum's default 404.
    let protected = if jobs_enabled {
        protected
            .route("/v1/jobs", post(http::submit_job))
            .route("/v1/jobs", options(|| async { StatusCode::NO_CONTENT }))
            .route("/v1/jobs/{id}", get(http::get_job))
            .route("/v1/jobs/{id}", delete(http::cancel_job))
            .route(
                "/v1/jobs/{id}",
                options(|| async { StatusCode::NO_CONTENT }),
            )
            .route("/v1/jobs/{id}/result", get(http::get_job_result))
            .route(
                "/v1/jobs/{id}/result",
                options(|| async { StatusCode::NO_CONTENT }),
            )
            .route("/v1/jobs/{id}/events", get(http::job_events))
    } else {
        protected
    };
    protected
        .route_layer(axum::middleware::from_fn_with_state(
            admission,
            super::upload::admit,
        ))
        .route_layer(axum::middleware::from_fn(reject_v4_on_legacy))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_explicit_v4_profile_cannot_fall_through_to_legacy() {
        for query in [
            "model_family=gigaam_v4",
            "model_precision=fp32",
            "transcript_style=raw",
            "model_head=rnnt",
            "variant=gigaam-v4-rnnt-e2e-fp32",
            "model=gigaam-v4-unqualified",
            "model=gigaam-v4-rnnt-raw-fp32&model=whisper-1",
        ] {
            let uri = format!("/v1/transcribe?{query}").parse().unwrap();
            let axum::extract::Query(fields) =
                axum::extract::Query::<Vec<(String, String)>>::try_from_uri(&uri).unwrap();
            assert!(requests_v4(&fields), "{query}");
        }
        for query in [
            "variant=rnnt",
            "model=whisper-1",
            "unknown_legacy=ok",
            "format=json",
        ] {
            let uri = format!("/v1/transcribe?{query}").parse().unwrap();
            let axum::extract::Query(fields) =
                axum::extract::Query::<Vec<(String, String)>>::try_from_uri(&uri).unwrap();
            assert!(!requests_v4(&fields), "{query}");
        }
    }

    #[test]
    fn test_protected_v1_router_builds_with_and_without_jobs() {
        // Both branches must construct: jobs routes are only registered when
        // the flag is on, and a panic here would be a boot-time outage.
        let _ = protected_v1_router(false, UploadAdmission::new(1, 30));
        let _ = protected_v1_router(true, UploadAdmission::new(1, 30));
    }
}
