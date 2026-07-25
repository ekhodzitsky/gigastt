//! Health, readiness, metrics, and model capability endpoints.

use axum::extract::State;
use axum::http::StatusCode;
use axum::http::header;
use axum::response::{IntoResponse, Json, Response};
use serde::Serialize;
use std::sync::Arc;

use super::super::metrics::MetricsRegistry;
use gigastt_core::inference::Engine;

use super::state::AppState;

/// GET /metrics — Prometheus text-format exposition. Returns 404 when the
/// server was started without `--metrics`.
pub async fn metrics(State(state): State<Arc<AppState>>) -> Response {
    match &state.metrics_registry {
        Some(registry) => (
            StatusCode::OK,
            [(
                header::CONTENT_TYPE,
                "text/plain; version=0.0.4; charset=utf-8",
            )],
            registry.render_prometheus(),
        )
            .into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "metrics endpoint disabled",
                "code": "metrics_disabled",
            })),
        )
            .into_response(),
    }
}

/// Health check response.
#[derive(Serialize)]
pub struct HealthResponse {
    /// Always `"ok"` when the server is running.
    pub status: String,
    /// Stable model identifier for the head actually loaded
    /// (`"gigaam-v3-rnnt"` or `"gigaam-v3-e2e-rnnt"`).
    pub model: String,
    /// Recognition head in use: `"rnnt"` or `"e2e_rnnt"`. Added so a client can
    /// tell from a single `/health` call which head (and therefore which output
    /// style) is producing transcripts.
    pub variant: String,
    /// Server version from `CARGO_PKG_VERSION`.
    pub version: String,
    /// Whether the punctuation / casing restoration pass is active for this
    /// server (the effective `--punctuation` policy).
    pub punctuation: bool,
    /// Whether inverse text normalization (numbers → digits) is active for this
    /// server (the effective `--itn` policy).
    pub itn: bool,
}

/// Model info response.
#[derive(Serialize)]
pub struct ModelInfo {
    /// Stable model identifier for the head actually loaded
    /// (`"gigaam-v3-rnnt"` or `"gigaam-v3-e2e-rnnt"`).
    pub id: String,
    /// Human-readable model name.
    pub name: String,
    /// Recognition head in use: `"rnnt"` or `"e2e_rnnt"`.
    pub variant: String,
    /// Server version from `CARGO_PKG_VERSION`.
    pub version: String,
    /// Encoder precision in use: `"int8"` or `"fp32"`.
    pub encoder: String,
    /// Number of tokens in the BPE vocabulary.
    pub vocab_size: usize,
    /// Native sample rate the model operates at (always 16000 Hz).
    pub sample_rate: u32,
    /// Total number of session triplets in the pool.
    pub pool_size: usize,
    /// Number of session triplets currently available for checkout.
    pub pool_available: usize,
    /// Audio container formats accepted by `/v1/transcribe`.
    pub supported_formats: Vec<String>,
    /// PCM sample rates accepted by the WebSocket endpoint.
    pub supported_rates: Vec<u32>,
    /// Whether the punctuation / casing restoration pass is active (effective
    /// `--punctuation` policy for the loaded head).
    pub punctuation: bool,
    /// Whether inverse text normalization (numbers → digits) is active
    /// (effective `--itn` policy for the loaded head).
    pub itn: bool,
    /// Whether speaker diarization is available (feature-gated build + model loaded).
    /// Added in v0.7.0 so clients can probe capabilities via REST instead of
    /// opening a WebSocket just to read the `Ready` frame.
    pub diarization: bool,
}

/// Readiness probe response.
#[derive(Serialize)]
pub struct ReadinessResponse {
    /// `"ready"` when the server can accept requests, `"not_ready"` otherwise.
    pub status: String,
    /// Number of session triplets currently available for checkout.
    pub pool_available: usize,
    /// Total number of session triplets in the pool.
    pub pool_total: usize,
    /// Machine-readable reason code when not ready (e.g. `"pool_exhausted"`, `"shutting_down"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// GET /health — liveness check for monitoring and Docker HEALTHCHECK.
///
/// Liveness: stays 200 while the process is alive. It reads only the engine's
/// static identity (loaded head + effective punctuation/ITN policy) — a cheap,
/// infallible field read, no pool checkout or I/O — so a client can confirm
/// *which* model is serving from the same probe it already makes. Pool /
/// shutdown readiness remains the separate `/ready` probe (see [`readiness`]).
///
/// During first-run model download / quantization the listener is served by a
/// minimal bootstrap responder (see [`crate::server::run_with_config_loading`]) that
/// reports `model: "loading"`; this handler only runs once the engine is ready.
pub async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let engine = state.engine.load_full();
    let variant = engine.variant();
    Json(HealthResponse {
        status: "ok".into(),
        model: variant.model_id().into(),
        variant: variant.as_str().into(),
        version: env!("CARGO_PKG_VERSION").into(),
        punctuation: engine.has_punctuator(),
        itn: engine.has_itn(),
    })
}

/// Sample the dedicated batch pool's availability / waiters when one exists
/// (`--batch-pool-size > 0`). The batch pool has its own FIFO queue, so it can
/// be saturated while the interactive pool reads healthy; exporting it under
/// distinct gauges keeps batch-pool exhaustion observable instead of hidden.
/// No-op when no batch pool was split off.
pub(crate) fn sample_batch_pool_gauges(reg: &MetricsRegistry, engine: &Engine) {
    if let Some(ref batch) = engine.batch_pool {
        reg.gauge_set(
            "gigastt_batch_pool_available",
            &[],
            batch.available() as i64,
        );
        reg.gauge_set("gigastt_batch_pool_waiters", &[], batch.waiters() as i64);
    }
}

/// GET /ready — readiness probe for k8s and orchestrators.
pub async fn readiness(State(state): State<Arc<AppState>>) -> Response {
    let engine = state.engine.load_full();
    if state.shutdown.is_cancelled() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ReadinessResponse {
                status: "not_ready".into(),
                pool_available: 0,
                pool_total: engine.pool.total(),
                reason: Some("shutting_down".into()),
            }),
        )
            .into_response();
    }
    let available = engine.pool.available();
    if let Some(ref reg) = state.metrics_registry {
        reg.gauge_set("gigastt_pool_available", &[], available as i64);
        reg.gauge_set("gigastt_pool_waiters", &[], engine.pool.waiters() as i64);
        sample_batch_pool_gauges(reg, &engine);
    }
    if available == 0 {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ReadinessResponse {
                status: "not_ready".into(),
                pool_available: 0,
                pool_total: engine.pool.total(),
                reason: Some("pool_exhausted".into()),
            }),
        )
            .into_response();
    }
    Json(ReadinessResponse {
        status: "ready".into(),
        pool_available: available,
        pool_total: engine.pool.total(),
        reason: None,
    })
    .into_response()
}

/// GET /v1/models — list loaded models and capabilities.
pub async fn models(State(state): State<Arc<AppState>>) -> Json<ModelInfo> {
    let engine = state.engine.load_full();
    #[cfg(feature = "diarization")]
    let diarization = engine.has_speaker_encoder();
    #[cfg(not(feature = "diarization"))]
    let diarization = false;
    if let Some(ref reg) = state.metrics_registry {
        reg.gauge_set(
            "gigastt_pool_available",
            &[],
            engine.pool.available() as i64,
        );
        reg.gauge_set("gigastt_pool_waiters", &[], engine.pool.waiters() as i64);
        sample_batch_pool_gauges(reg, &engine);
    }
    let variant = engine.variant();
    Json(ModelInfo {
        id: variant.model_id().into(),
        name: variant.display_name().into(),
        variant: variant.as_str().into(),
        version: env!("CARGO_PKG_VERSION").into(),
        encoder: if engine.is_int8() {
            "int8".into()
        } else {
            "fp32".into()
        },
        vocab_size: engine.vocab_size(),
        sample_rate: 16000,
        pool_size: engine.pool.total(),
        pool_available: engine.pool.available(),
        supported_formats: vec![
            "wav".into(),
            "mp3".into(),
            "m4a".into(),
            "ogg".into(),
            "flac".into(),
        ],
        supported_rates: super::super::config::SUPPORTED_RATES.to_vec(),
        punctuation: engine.has_punctuator(),
        itn: engine.has_itn(),
        diarization,
    })
}
