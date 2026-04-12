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

    let _permit = state.semaphore.acquire().await
        .map_err(|_| api_error(StatusCode::SERVICE_UNAVAILABLE, "Server busy", "busy"))?;

    let tmp = tempfile::NamedTempFile::new()
        .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e}"), "internal"))?;
    std::fs::write(tmp.path(), &body)
        .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e}"), "internal"))?;

    let path = tmp.path().to_string_lossy().to_string();
    let engine = state.engine.clone();

    let result = tokio::task::spawn_blocking(move || engine.transcribe_file(&path)).await;

    match result {
        Ok(Ok(text)) => Ok(Json(TranscribeResponse {
            text,
            words: vec![],
            duration: 0.0, // not available from transcribe_file
        })),
        Ok(Err(e)) => Err(api_error(StatusCode::UNPROCESSABLE_ENTITY, &format!("{e}"), "transcription_error")),
        Err(e) => Err(api_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e}"), "internal")),
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

    let _permit = state.semaphore.acquire().await
        .map_err(|_| api_error(StatusCode::SERVICE_UNAVAILABLE, "Server busy", "busy"))?;

    let tmp = tempfile::NamedTempFile::new()
        .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e}"), "internal"))?;
    std::fs::write(tmp.path(), &body)
        .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e}"), "internal"))?;

    let path = tmp.path().to_string_lossy().to_string();
    let engine = state.engine.clone();

    let samples = crate::inference::audio::decode_audio_file(&path)
        .map_err(|e| api_error(StatusCode::UNPROCESSABLE_ENTITY, &format!("{e}"), "invalid_audio"))?;

    let stream = async_stream::stream! {
        let mut stream_state = engine.create_state(
            #[cfg(feature = "diarization")]
            false,
        );
        let chunk_size = 16000; // 1 second at 16kHz

        for chunk in samples.chunks(chunk_size) {
            match engine.process_chunk(chunk, &mut stream_state) {
                Ok(segments) => {
                    for seg in segments {
                        let msg = if seg.is_final {
                            serde_json::json!({"type": "final", "text": seg.text, "timestamp": seg.timestamp, "words": seg.words})
                        } else {
                            serde_json::json!({"type": "partial", "text": seg.text, "timestamp": seg.timestamp, "words": seg.words})
                        };
                        yield Ok(Event::default().data(msg.to_string()));
                    }
                }
                Err(e) => {
                    let msg = serde_json::json!({"type": "error", "message": format!("{e}"), "code": "inference_error"});
                    yield Ok(Event::default().data(msg.to_string()));
                    break;
                }
            }
        }

        if let Some(seg) = engine.flush_state(&mut stream_state) {
            let msg = serde_json::json!({"type": "final", "text": seg.text, "timestamp": seg.timestamp, "words": seg.words});
            yield Ok(Event::default().data(msg.to_string()));
        }
    };

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
