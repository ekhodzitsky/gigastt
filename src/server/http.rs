//! HTTP handlers for REST API endpoints.

use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::Json;
use futures_util::stream::Stream;
use serde::Serialize;
use std::sync::Arc;

use crate::inference::Engine;

/// Shared application state for all handlers.
pub struct AppState {
    pub engine: Arc<Engine>,
    pub semaphore: Arc<tokio::sync::Semaphore>,
}

/// Health check response.
#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub model: String,
    pub version: String,
}

/// Transcription response.
#[derive(Serialize)]
pub struct TranscribeResponse {
    pub text: String,
    pub words: Vec<crate::inference::WordInfo>,
    pub duration: f64,
}

type ApiError = (StatusCode, Json<serde_json::Value>);

fn api_error(status: StatusCode, msg: &str, code: &str) -> ApiError {
    (status, Json(serde_json::json!({"error": msg, "code": code})))
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

/// POST /v1/transcribe — upload audio file, get full transcript.
///
/// Accepts raw audio body. Supported formats: WAV, MP3, M4A/AAC, OGG, FLAC.
/// Max body size enforced by tower-http layer (50MB).
pub async fn transcribe(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Result<Json<TranscribeResponse>, ApiError> {
    if body.is_empty() {
        return Err(api_error(StatusCode::BAD_REQUEST, "Empty request body", "empty_body"));
    }

    // C2: timeout on semaphore acquisition
    let _permit = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        state.semaphore.acquire(),
    )
    .await
    .map_err(|_| api_error(StatusCode::SERVICE_UNAVAILABLE, "Server busy, try again later", "timeout"))?
    .map_err(|_| api_error(StatusCode::SERVICE_UNAVAILABLE, "Server shutting down", "busy"))?;

    let body_bytes = body.to_vec();
    // H7: drop body after copying
    drop(body);

    let engine = state.engine.clone();

    let result = tokio::task::spawn_blocking(move || engine.transcribe_bytes(&body_bytes)).await;

    match result {
        Ok(Ok(result)) => Ok(Json(TranscribeResponse {
            text: result.text,
            words: result.words,
            duration: result.duration_s,
        })),
        Ok(Err(e)) => {
            tracing::error!("Transcription error: {e}");
            Err(api_error(StatusCode::UNPROCESSABLE_ENTITY, "Transcription failed. Check audio format.", "transcription_error"))
        }
        Err(e) => {
            tracing::error!("spawn_blocking join error: {e}");
            Err(api_error(StatusCode::INTERNAL_SERVER_ERROR, "Internal server error", "internal"))
        }
    }
}

/// POST /v1/transcribe/stream — upload audio file, get SSE stream of partial/final results.
pub async fn transcribe_stream(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Result<Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>>, ApiError> {
    if body.is_empty() {
        return Err(api_error(StatusCode::BAD_REQUEST, "Empty request body", "empty_body"));
    }

    // C2: timeout on semaphore acquisition; H1: acquire before writing, move into stream
    let semaphore = state.semaphore.clone();
    let permit = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        semaphore.acquire_owned(),
    )
    .await
    .map_err(|_| api_error(StatusCode::SERVICE_UNAVAILABLE, "Server busy, try again later", "timeout"))?
    .map_err(|_| api_error(StatusCode::SERVICE_UNAVAILABLE, "Server shutting down", "busy"))?;

    let body_bytes = body.to_vec();
    // H7: drop body after copying
    drop(body);

    let samples = crate::inference::audio::decode_audio_bytes(&body_bytes)
        .map_err(|e| {
            tracing::error!("Audio decode error: {e:#}");
            api_error(StatusCode::UNPROCESSABLE_ENTITY, "Failed to decode audio file. Check format (WAV, MP3, M4A, OGG, FLAC supported).", "invalid_audio")
        })?;

    let engine = state.engine.clone();

    // H2: run all blocking inference in spawn_blocking, collect segments upfront
    let engine_clone = engine.clone();
    let segments_result = tokio::task::spawn_blocking(move || {
        let mut stream_state = engine_clone.create_state(
            #[cfg(feature = "diarization")]
            false,
        );
        let chunk_size = 16000; // 1 second at 16kHz
        let mut all_segments = Vec::new();
        let mut had_error = None;

        for chunk in samples.chunks(chunk_size) {
            match engine_clone.process_chunk(chunk, &mut stream_state) {
                Ok(segs) => all_segments.extend(segs),
                Err(e) => {
                    had_error = Some(format!("{e}"));
                    break;
                }
            }
        }

        if had_error.is_none() && let Some(seg) = engine_clone.flush_state(&mut stream_state) {
            all_segments.push(seg);
        }

        (all_segments, had_error)
    })
    .await;

    let (all_segments, inference_error) = match segments_result {
        Ok(pair) => pair,
        Err(e) => {
            tracing::error!("spawn_blocking join error: {e}");
            return Err(api_error(StatusCode::INTERNAL_SERVER_ERROR, "Internal server error", "internal"));
        }
    };

    // Build events from collected segments
    let mut events: Vec<Result<Event, std::convert::Infallible>> = Vec::new();
    if let Some(err) = inference_error {
        tracing::error!("SSE inference error: {err}");
        let msg = serde_json::json!({"type": "error", "message": "Transcription failed. Check audio format.", "code": "inference_error"});
        events.push(Ok(Event::default().data(msg.to_string())));
    } else {
        for seg in all_segments {
            let msg = if seg.is_final {
                serde_json::json!({"type": "final", "text": seg.text, "timestamp": seg.timestamp, "words": seg.words})
            } else {
                serde_json::json!({"type": "partial", "text": seg.text, "timestamp": seg.timestamp, "words": seg.words})
            };
            events.push(Ok(Event::default().data(msg.to_string())));
        }
    }

    // Stream events with permit held until stream is exhausted
    let stream = futures_util::stream::unfold(
        (events.into_iter(), permit),
        |(mut iter, permit)| async move {
            iter.next().map(|event| (event, (iter, permit)))
        },
    );

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
