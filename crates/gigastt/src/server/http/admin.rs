//! Administrative endpoints (model hot-reload).

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use std::sync::Arc;

use super::state::AppState;

/// Whether the reload endpoint should accept a request from `peer`.
///
/// Model reload is an administrative, machine-local action: it must stay
/// reachable only from the loopback interface even under `--bind-all` /
/// `--cors-allow-any`, which would otherwise widen `origin_middleware` (the only
/// other gate). Pure so the loopback decision can be unit-tested without a model
/// or a live socket.
pub(super) fn peer_is_loopback(peer: &std::net::SocketAddr) -> bool {
    peer.ip().is_loopback()
}

/// POST /v1/admin/reload — rebuild the inference engine from the boot recipe and
/// atomically swap it in without a restart.
///
/// Strictly loopback-only (checked here, not just via the origin middleware),
/// serialized by a mutex so two reloads can't race, and fail-safe: a build error
/// leaves the currently-serving engine untouched. The new engine is warmed
/// before the swap so the first post-swap request doesn't pay the cold cost.
/// In-flight requests keep the engine they started on and finish against its
/// pool; the old engine drops when its last reference goes.
pub async fn reload(
    axum::extract::ConnectInfo(peer): axum::extract::ConnectInfo<std::net::SocketAddr>,
    State(state): State<Arc<AppState>>,
) -> Response {
    // Gotcha #2: enforce loopback here so reload stays local even when
    // `origin_middleware` has been widened by `--bind-all` / `--cors-allow-any`
    // or a caller omits the Origin header.
    if !peer_is_loopback(&peer) {
        tracing::warn!(peer = %peer, "Rejecting non-loopback model reload request");
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "error": "Model reload is only available from loopback",
                "code": "loopback_only",
            })),
        )
            .into_response();
    }

    let Some(builder) = state.engine_builder.clone() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": "Model reload is not available on this server",
                "code": "reload_unsupported",
            })),
        )
            .into_response();
    };

    // Serialize reloads: the loser of the race gets 409 rather than queueing, so
    // an operator hammering the endpoint can't stack up concurrent rebuilds.
    let _reload_guard = match state.reload_lock.try_lock() {
        Ok(guard) => guard,
        Err(_) => {
            return (
                StatusCode::CONFLICT,
                Json(serde_json::json!({
                    "error": "A model reload is already in progress",
                    "code": "reload_in_progress",
                })),
            )
                .into_response();
        }
    };

    tracing::info!(peer = %peer, "Model reload requested — rebuilding engine");

    // Build the new engine off the request path (ONNX session load is blocking).
    let build = tokio::task::spawn_blocking(move || builder()).await;

    let new_engine = match build {
        Ok(Ok(engine)) => engine,
        Ok(Err(e)) => {
            // Keep the old engine untouched. Log the detail, return a sanitized
            // message (no path / model leakage) matching the internal-error policy.
            tracing::error!("Model reload build failed: {e:#}");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "Model reload failed; the previous model is still serving",
                    "code": "reload_failed",
                })),
            )
                .into_response();
        }
        Err(join_err) => {
            tracing::error!("Model reload build task panicked: {join_err}");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "Model reload failed; the previous model is still serving",
                    "code": "reload_failed",
                })),
            )
                .into_response();
        }
    };

    // Warm the fresh engine BEFORE swapping so the first post-swap request
    // doesn't pay the EP-compile / first-allocation cost. A warmup failure is
    // non-fatal (mirrors boot): the engine already fell back to CPU internally.
    let new_engine = match tokio::task::spawn_blocking(move || {
        if let Err(e) = new_engine.warmup() {
            tracing::warn!("Reloaded engine warmup failed (swapping anyway): {e:#}");
        }
        new_engine
    })
    .await
    {
        Ok(engine) => engine,
        Err(join_err) => {
            tracing::error!("Model reload warmup task panicked: {join_err}");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "Model reload failed; the previous model is still serving",
                    "code": "reload_failed",
                })),
            )
                .into_response();
        }
    };

    let variant = new_engine.variant();
    let encoder = if new_engine.is_int8() { "int8" } else { "fp32" };

    // Atomic swap. In-flight requests holding the old `Arc<Engine>` finish
    // against the old pool; it drops when its last reference goes. We do NOT
    // close the old pool — that would strand in-flight work.
    state.engine.store(Arc::new(new_engine));
    tracing::info!(
        variant = variant.as_str(),
        encoder,
        "Model reloaded and swapped"
    );

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "reloaded": true,
            "variant": variant.as_str(),
            "encoder": encoder,
        })),
    )
        .into_response()
}
