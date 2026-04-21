//! HTTP handlers for REST API endpoints.

use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::http::header;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use futures_util::StreamExt;
use futures_util::stream::Stream;
use serde::Serialize;
use std::sync::Arc;

use super::metrics::MetricsRegistry;
use super::{POOL_RETRY_AFTER_MS, POOL_RETRY_AFTER_SECS, RuntimeLimits};
use crate::inference::Engine;

/// Shared application state for all handlers. Carries runtime limits so the
/// WebSocket path can enforce configurable frame / idle bounds without
/// re-threading every CLI arg through each handler, plus an optional
/// in-tree `MetricsRegistry` backing the `/metrics` endpoint.
///
/// Also carries a shutdown `CancellationToken` and a `TaskTracker` used to
/// drain in-flight WebSocket / SSE tasks on SIGTERM (V1-03). `axum::serve`'s
/// built-in `with_graceful_shutdown` only tracks the HTTP router; upgraded
/// WebSocket handlers and `spawn_blocking` SSE tasks fall outside that lane
/// and must be drained explicitly.
pub struct AppState {
    pub engine: Arc<Engine>,
    pub limits: RuntimeLimits,
    pub metrics_registry: Option<Arc<MetricsRegistry>>,
    pub shutdown: tokio_util::sync::CancellationToken,
    pub tracker: tokio_util::task::TaskTracker,
}

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
    pub status: String,
    pub model: String,
    pub version: String,
}

/// Model info response.
#[derive(Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub version: String,
    pub encoder: String,
    pub vocab_size: usize,
    pub sample_rate: u32,
    pub pool_size: usize,
    pub pool_available: usize,
    pub supported_formats: Vec<String>,
    pub supported_rates: Vec<u32>,
    /// Whether speaker diarization is available (feature-gated build + model loaded).
    /// Added in v0.7.0 so clients can probe capabilities via REST instead of
    /// opening a WebSocket just to read the `Ready` frame.
    pub diarization: bool,
}

/// Transcription response.
#[derive(Serialize)]
pub struct TranscribeResponse {
    pub text: String,
    pub words: Vec<crate::inference::WordInfo>,
    pub duration: f64,
}

/// Error response produced by the REST handlers. Using `Response` directly
/// (rather than a `(StatusCode, Json<_>)` tuple) lets timeout paths attach
/// a `Retry-After` header without changing the handler signatures.
type ApiError = Response;

fn api_error(status: StatusCode, msg: &str, code: &str) -> ApiError {
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
fn api_timeout_error() -> ApiError {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [(header::RETRY_AFTER, POOL_RETRY_AFTER_SECS.to_string())],
        Json(serde_json::json!({
            "error": "Server busy, try again later",
            "code": "timeout",
            "retry_after_ms": POOL_RETRY_AFTER_MS,
        })),
    )
        .into_response()
}

/// 503 response for the case where the pool was closed (graceful shutdown
/// in progress). Distinct from `timeout` so clients can decide whether to
/// retry: a closed pool is not coming back, so no `retry_after_ms` hint.
fn api_pool_closed_error() -> ApiError {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(serde_json::json!({
            "error": "Server is shutting down",
            "code": "pool_closed",
        })),
    )
        .into_response()
}

/// GET /health — health check for monitoring and Docker HEALTHCHECK.
pub async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let _ = &state.engine;
    Json(HealthResponse {
        status: "ok".into(),
        model: "gigaam-v3-e2e-rnnt".into(),
        version: env!("CARGO_PKG_VERSION").into(),
    })
}

/// GET /v1/models — list loaded models and capabilities.
pub async fn models(State(state): State<Arc<AppState>>) -> Json<ModelInfo> {
    let engine = &state.engine;
    #[cfg(feature = "diarization")]
    let diarization = engine.has_speaker_encoder();
    #[cfg(not(feature = "diarization"))]
    let diarization = false;
    Json(ModelInfo {
        id: "gigaam-v3-e2e-rnnt".into(),
        name: "GigaAM v3 RNN-T".into(),
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
        supported_rates: super::SUPPORTED_RATES.to_vec(),
        diarization,
    })
}

/// POST /v1/transcribe — upload audio file, get full transcript.
///
/// Accepts raw audio body. Supported formats: WAV, MP3, M4A/AAC, OGG, FLAC.
/// Max body size enforced by the axum `DefaultBodyLimit` layer configured
/// from [`RuntimeLimits::body_limit_bytes`] (default 50 MiB).
pub async fn transcribe(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Result<Json<TranscribeResponse>, ApiError> {
    if body.is_empty() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "Empty request body",
            "empty_body",
        ));
    }

    // Defence-in-depth: `DefaultBodyLimit` already rejects oversized bodies
    // before they reach this handler, but a mis-ordered middleware stack or
    // a `Content-Length`-spoofing client could still deliver too many bytes.
    // The explicit 413 keeps the REST contract honest and gives clients a
    // machine-readable `payload_too_large` code alongside the spec-conformant
    // status. Cheap: `Bytes::len()` is a load, not a walk.
    if body.len() > state.limits.body_limit_bytes {
        return Err(api_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "Request body exceeds the configured size limit",
            "payload_too_large",
        ));
    }

    // Checkout a session triplet from the pool (blocks if none available).
    // The guard's lifetime is stripped via `into_owned` so the triplet can
    // travel through `spawn_blocking`; the reservation handles checkin.
    let guard = match tokio::time::timeout(
        std::time::Duration::from_secs(30),
        state.engine.pool.checkout(),
    )
    .await
    {
        Ok(Ok(guard)) => guard,
        Ok(Err(_pool_closed)) => return Err(api_pool_closed_error()),
        Err(_timeout) => return Err(api_timeout_error()),
    };
    let (triplet, reservation) = guard.into_owned();

    let engine = state.engine.clone();

    let result = tokio::task::spawn_blocking(move || {
        let mut triplet = triplet;
        // catch_unwind ensures triplet is returned to pool even on panic
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // `body` is an `axum::body::Bytes` (re-export of `bytes::Bytes`):
            // `clone()` is a refcount bump, not a data copy, so the decode
            // path shares the original upload buffer.
            engine.transcribe_bytes_shared(body, &mut triplet)
        }));
        match r {
            Ok(inference_result) => (inference_result, triplet),
            Err(_) => {
                tracing::error!("Panic in REST transcribe — triplet recovered");
                (
                    Err(crate::error::GigasttError::Inference(
                        "Inference thread panicked".into(),
                    )),
                    triplet,
                )
            }
        }
    })
    .await;

    match result {
        Ok((Ok(result), triplet)) => {
            reservation.checkin(triplet);
            Ok(Json(TranscribeResponse {
                text: result.text,
                words: result.words,
                duration: result.duration_s,
            }))
        }
        Ok((Err(e), triplet)) => {
            reservation.checkin(triplet);
            tracing::error!("Transcription error: {e}");
            Err(api_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "Transcription failed. Check audio format.",
                "transcription_error",
            ))
        }
        Err(e) => {
            // spawn_blocking task itself failed (e.g., runtime shutdown).
            // Triplet is lost in this branch; reservation is dropped without
            // sending. The pool degrades by one slot.
            tracing::error!("spawn_blocking join error: {e}");
            Err(api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal server error",
                "internal",
            ))
        }
    }
}

/// POST /v1/transcribe/stream — upload audio file, get SSE stream of partial/final results.
///
/// Real streaming: audio is processed chunk-by-chunk inside `spawn_blocking`,
/// and segments are sent to the SSE stream via an mpsc channel as they are produced.
pub async fn transcribe_stream(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Result<Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>>, ApiError> {
    if body.is_empty() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "Empty request body",
            "empty_body",
        ));
    }

    // Defence-in-depth early reject; matches `/v1/transcribe` — see that
    // handler for the rationale.
    if body.len() > state.limits.body_limit_bytes {
        return Err(api_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "Request body exceeds the configured size limit",
            "payload_too_large",
        ));
    }

    // Decode audio first (in spawn_blocking since symphonia is blocking).
    // `body` is `axum::body::Bytes`, so the move into the blocking closure is
    // a refcount bump and `decode_audio_bytes_shared` reads the upload
    // buffer in place.
    let samples = tokio::task::spawn_blocking(move || {
        crate::inference::audio::decode_audio_bytes_shared(body)
    })
    .await
    .map_err(|e| {
        tracing::error!("spawn_blocking join error: {e}");
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Internal server error",
            "internal",
        )
    })?
    .map_err(|e| {
        tracing::error!("Audio decode error: {e:#}");
        api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "Failed to decode audio file. Check format (WAV, MP3, M4A, OGG, FLAC supported).",
            "invalid_audio",
        )
    })?;

    // Checkout a session triplet from the pool. Strip the lifetime via
    // `into_owned` so the triplet can travel through `spawn_blocking`.
    let guard = match tokio::time::timeout(
        std::time::Duration::from_secs(30),
        state.engine.pool.checkout(),
    )
    .await
    {
        Ok(Ok(guard)) => guard,
        Ok(Err(_pool_closed)) => return Err(api_pool_closed_error()),
        Err(_timeout) => return Err(api_timeout_error()),
    };
    let (triplet, reservation) = guard.into_owned();

    // Create mpsc channel for streaming segments from spawn_blocking to SSE
    let (tx, rx) =
        tokio::sync::mpsc::channel::<Result<crate::inference::TranscriptSegment, String>>(16);

    let engine = state.engine.clone();
    // V1-03: the axum handler future has already returned by the time the
    // SSE stream starts flowing, so `with_graceful_shutdown` can't observe
    // this task. Clone the shutdown token and check it before every chunk
    // so SIGTERM during a long transcription drops cleanly.
    let cancel = state.shutdown.clone();
    let tracker = state.tracker.clone();
    tracker.spawn_blocking(move || {
        let mut triplet = triplet;

        // catch_unwind ensures triplet is returned to pool even on panic
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut stream_state = engine.create_state(false);
            let chunk_size = 16000; // 1 second at 16kHz

            for chunk in samples.chunks(chunk_size) {
                if cancel.is_cancelled() {
                    tracing::info!("SSE transcription cancelled by shutdown");
                    return;
                }
                match engine.process_chunk(chunk, &mut stream_state, &mut triplet) {
                    Ok(segs) => {
                        for seg in segs {
                            if tx.blocking_send(Ok(seg)).is_err() {
                                // Receiver dropped (client disconnected)
                                return;
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.blocking_send(Err(format!("{e}")));
                        return;
                    }
                }
            }

            // Flush final segment — best-effort; skipped on cancel.
            if !cancel.is_cancelled()
                && let Some(seg) = engine.flush_state(&mut stream_state)
            {
                let _ = tx.blocking_send(Ok(seg));
            }
        }));

        if result.is_err() {
            tracing::error!("Panic in SSE inference task — triplet recovered");
        }

        // Always return triplet to pool (even after panic). Sync `try_send`
        // is safe from a blocking thread; if the pool was closed in the
        // interim the triplet is silently dropped.
        reservation.checkin(triplet);
    });

    // Convert receiver to SSE stream
    let stream = tokio_stream::wrappers::ReceiverStream::new(rx)
        .map(|result| {
            let event = match result {
                Ok(seg) => {
                    let msg = if seg.is_final {
                        serde_json::json!({"type": "final", "text": seg.text, "timestamp": seg.timestamp, "words": seg.words})
                    } else {
                        serde_json::json!({"type": "partial", "text": seg.text, "timestamp": seg.timestamp, "words": seg.words})
                    };
                    Event::default().data(msg.to_string())
                }
                Err(_) => {
                    let msg = serde_json::json!({"type": "error", "message": "Transcription failed.", "code": "inference_error"});
                    Event::default().data(msg.to_string())
                }
            };
            Ok(event)
        });

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_response_serialization() {
        let resp = HealthResponse {
            status: "ok".into(),
            model: "test".into(),
            version: "0.3.0".into(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["status"], "ok");
        assert_eq!(v["model"], "test");
    }

    #[test]
    fn test_transcribe_response_serialization() {
        let resp = TranscribeResponse {
            text: "hello".into(),
            words: vec![],
            duration: 1.5,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["text"], "hello");
        assert_eq!(v["duration"], 1.5);
    }
}
