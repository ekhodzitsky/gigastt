//! HTTP handlers for REST API endpoints.

use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::Json;
use futures_util::stream::Stream;
use futures_util::StreamExt;
use serde::Serialize;
use std::sync::Arc;

use crate::inference::Engine;

/// Shared application state for all handlers.
pub struct AppState {
    pub engine: Arc<Engine>,
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

/// GET /v1/models — list loaded models and capabilities.
pub async fn models(State(state): State<Arc<AppState>>) -> Json<ModelInfo> {
    let engine = &state.engine;
    Json(ModelInfo {
        id: "gigaam-v3-e2e-rnnt".into(),
        name: "GigaAM v3 RNN-T".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        encoder: if engine.is_int8() { "int8".into() } else { "fp32".into() },
        vocab_size: 1025,
        sample_rate: 16000,
        pool_size: engine.pool.total(),
        pool_available: engine.pool.available(),
        supported_formats: vec![
            "wav".into(), "mp3".into(), "m4a".into(),
            "ogg".into(), "flac".into(),
        ],
        supported_rates: vec![8000, 16000, 24000, 44100, 48000],
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

    // Checkout a session triplet from the pool (blocks if none available)
    let triplet = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        state.engine.pool.checkout(),
    )
    .await
    .map_err(|_| api_error(StatusCode::SERVICE_UNAVAILABLE, "Server busy, try again later", "timeout"))?;

    let body_bytes = body.to_vec();
    drop(body);

    let engine = state.engine.clone();

    let result = tokio::task::spawn_blocking(move || {
        let mut triplet = triplet;
        let r = engine.transcribe_bytes(&body_bytes, &mut triplet);
        (r, triplet)
    }).await;

    match result {
        Ok((Ok(result), triplet)) => {
            state.engine.pool.checkin(triplet).await;
            Ok(Json(TranscribeResponse {
                text: result.text,
                words: result.words,
                duration: result.duration_s,
            }))
        }
        Ok((Err(e), triplet)) => {
            state.engine.pool.checkin(triplet).await;
            tracing::error!("Transcription error: {e}");
            Err(api_error(StatusCode::UNPROCESSABLE_ENTITY, "Transcription failed. Check audio format.", "transcription_error"))
        }
        Err(e) => {
            // spawn_blocking panicked — triplet is lost
            tracing::error!("spawn_blocking join error: {e}");
            Err(api_error(StatusCode::INTERNAL_SERVER_ERROR, "Internal server error", "internal"))
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
        return Err(api_error(StatusCode::BAD_REQUEST, "Empty request body", "empty_body"));
    }

    let body_bytes = body.to_vec();
    drop(body);

    // Decode audio first (in spawn_blocking since symphonia is blocking)
    let samples = {
        let bytes = body_bytes;
        tokio::task::spawn_blocking(move || {
            crate::inference::audio::decode_audio_bytes(&bytes)
        })
        .await
        .map_err(|e| {
            tracing::error!("spawn_blocking join error: {e}");
            api_error(StatusCode::INTERNAL_SERVER_ERROR, "Internal server error", "internal")
        })?
        .map_err(|e| {
            tracing::error!("Audio decode error: {e:#}");
            api_error(StatusCode::UNPROCESSABLE_ENTITY, "Failed to decode audio file. Check format (WAV, MP3, M4A, OGG, FLAC supported).", "invalid_audio")
        })?
    };

    // Checkout a session triplet from the pool
    let triplet = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        state.engine.pool.checkout(),
    )
    .await
    .map_err(|_| api_error(StatusCode::SERVICE_UNAVAILABLE, "Server busy, try again later", "timeout"))?;

    // Create mpsc channel for streaming segments from spawn_blocking to SSE
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<crate::inference::TranscriptSegment, String>>(16);

    let engine = state.engine.clone();
    tokio::task::spawn_blocking(move || {
        let mut triplet = triplet;

        // catch_unwind ensures triplet is returned to pool even on panic
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut stream_state = engine.create_state(
                #[cfg(feature = "diarization")]
                false,
            );
            let chunk_size = 16000; // 1 second at 16kHz

            for chunk in samples.chunks(chunk_size) {
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

            // Flush final segment
            if let Some(seg) = engine.flush_state(&mut stream_state) {
                let _ = tx.blocking_send(Ok(seg));
            }
        }));

        if result.is_err() {
            tracing::error!("Panic in SSE inference task — triplet recovered");
        }

        // Always return triplet to pool (even after panic)
        engine.pool.blocking_checkin(triplet);
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
