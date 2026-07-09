//! HTTP handlers for REST API endpoints.

use axum::body::Bytes;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::http::header;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use futures_util::StreamExt;
use futures_util::stream::Stream;
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use std::sync::Arc;

use arc_swap::ArcSwap;

use super::config::{RuntimeLimits, pool_retry_after_ms, pool_retry_after_secs};
use super::metrics::MetricsRegistry;
use gigastt_core::export::{ExportFormat, RenderOpts, Segment};
use gigastt_core::inference::Engine;

/// Shared application state for all handlers. Carries runtime limits so the
/// WebSocket path can enforce configurable frame / idle bounds without
/// re-threading every CLI arg through each handler, plus an optional
/// in-tree `MetricsRegistry` backing the `/metrics` endpoint.
///
/// Also carries a shutdown `CancellationToken` and a `TaskTracker` used to
/// drain in-flight WebSocket / SSE tasks on SIGTERM. `axum::serve`'s
/// built-in `with_graceful_shutdown` only tracks the HTTP router; upgraded
/// WebSocket handlers and `spawn_blocking` SSE tasks fall outside that lane
/// and must be drained explicitly.
pub struct AppState {
    /// The live inference engine, held behind an [`ArcSwap`] so it can be
    /// atomically replaced by the model hot-reload endpoint without a restart.
    /// Handlers `load_full()` the current `Arc<Engine>` at request start and
    /// use it for the whole request; a concurrent swap only affects requests
    /// that start after it, so in-flight work always finishes against the
    /// engine (and pool) it began with.
    pub engine: Arc<ArcSwap<Engine>>,
    /// Rebuilds the engine from the exact boot recipe (model dir, pool sizes,
    /// threads, punctuation / ITN / VAD / hotwords). `Some` on the real server
    /// path; `None` for the thin `run`/`run_with_shutdown` test entry points
    /// and unit tests, where the reload endpoint reports `reload_unsupported`.
    pub engine_builder: Option<EngineBuilder>,
    /// Serializes model reloads so two concurrent `POST /v1/admin/reload`
    /// calls can't both rebuild + swap; the loser gets `409 reload_in_progress`.
    pub reload_lock: Arc<tokio::sync::Mutex<()>>,
    pub limits: Arc<ArcSwap<RuntimeLimits>>,
    pub metrics_registry: Option<Arc<MetricsRegistry>>,
    pub shutdown: tokio_util::sync::CancellationToken,
    pub tracker: tokio_util::task::TaskTracker,
}

/// Recipe that rebuilds a fully-configured [`Engine`] from the operator's boot
/// options. Stored in [`AppState`] so `POST /v1/admin/reload` can produce a
/// fresh engine that re-applies the punctuation / ITN / VAD / hotword chain —
/// a bare `Engine::load_*` starts with all of those set to `None`.
pub type EngineBuilder = Arc<dyn Fn() -> anyhow::Result<Engine> + Send + Sync>;

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

/// Transcription response.
#[derive(Serialize)]
pub struct TranscribeResponse {
    /// Full recognized transcript text.
    pub text: String,
    /// Word-level timing, confidence, and optional speaker annotations.
    pub words: Vec<gigastt_core::inference::WordInfo>,
    /// Duration of the submitted audio in seconds.
    pub duration: f64,
    /// Cue-sized segments, present only when the caller passed `?segments=true`.
    /// Additive: absent from the default response, so existing clients that read
    /// `text` / `words` / `duration` are unaffected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<Segment>>,
}

/// Query parameters for `/v1/transcribe` export formatting.
#[derive(Debug, Default, Deserialize)]
pub struct ExportParams {
    /// Export format: `json` (default), `txt`, `srt`, `vtt`, `md`.
    pub format: Option<String>,
    /// When set, the response carries `Content-Disposition: attachment` with this
    /// filename (or `transcript.<ext>` if the value is empty).
    pub download: Option<String>,
    /// Maximum characters per subtitle/caption line. `0` = unlimited.
    #[serde(default)]
    pub max_chars_per_line: Option<usize>,
    /// Maximum words per subtitle/caption line. `0` = unlimited.
    #[serde(default)]
    pub max_words_per_line: Option<usize>,
    /// Include per-word timestamps in Markdown output.
    #[serde(default)]
    pub word_timestamps: Option<bool>,
    /// Return cue-sized segments. In the default (JSON) response this adds a
    /// `segments` array; combined with `format=md` it switches Markdown to
    /// `### [mm:ss]` segment headers. Ignored for `txt`/`srt`/`vtt` (those are
    /// already flat / cue-based).
    #[serde(default)]
    pub segments: Option<bool>,
}

/// Render a transcription result into the requested export format.
///
/// Returns `None` when the caller explicitly requested the default JSON
/// response, so the handler can keep serving the existing `TranscribeResponse`
/// contract unchanged.
#[allow(clippy::result_large_err)]
fn render_export_response(
    result: &gigastt_core::inference::TranscribeResult,
    params: &ExportParams,
) -> Result<Option<Response>, ApiError> {
    let format_str = params.format.as_deref().unwrap_or("json");
    if format_str.eq_ignore_ascii_case("json") {
        return Ok(None);
    }

    let format = ExportFormat::from_str(format_str)
        .map_err(|e| api_error(StatusCode::BAD_REQUEST, &format!("{e}"), "invalid_format"))?;

    let opts = RenderOpts {
        max_chars_per_line: params.max_chars_per_line.unwrap_or(80),
        max_words_per_line: params.max_words_per_line.unwrap_or(14),
        include_word_timestamps: params.word_timestamps.unwrap_or(false),
    };

    // Precedence for `format` × `segments`: only Markdown composes with
    // `segments=true`, switching to `### [mm:ss]` section headers over the same
    // cue boundaries as SRT/VTT. `txt`/`srt`/`vtt` ignore `segments` (flat /
    // already cue-based); plain `format=md` keeps the frontmatter + `# Transcript`
    // blob unchanged.
    let body = if format == ExportFormat::Md && params.segments.unwrap_or(false) {
        gigastt_core::export::to_md_segments(
            result,
            opts.max_chars_per_line,
            opts.max_words_per_line,
        )
    } else {
        format.render(result, &opts)
    };
    let mut builder = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, format.content_type());

    if let Some(filename) = &params.download {
        let filename = if filename.is_empty() {
            format!("transcript.{}", format.extension())
        } else {
            filename.clone()
        };
        // The filename is user-controlled (query param), so build the header value
        // defensively: a filename with control characters would be an invalid
        // header value and otherwise panic when the response is built below. Fall
        // back to the safe default name when the requested value isn't valid.
        let disposition =
            header::HeaderValue::from_str(&format!("attachment; filename=\"{filename}\""))
                .unwrap_or_else(|_| {
                    header::HeaderValue::from_str(&format!(
                        "attachment; filename=\"transcript.{}\"",
                        format.extension()
                    ))
                    .expect("static content-disposition is always a valid header value")
                });
        builder = builder.header(header::CONTENT_DISPOSITION, disposition);
    }

    let response = builder.body(axum::body::Body::from(body)).map_err(|e| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("failed to build response: {e}"),
            "internal_error",
        )
    })?;
    Ok(Some(response))
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
fn api_timeout_error(limits: &RuntimeLimits) -> ApiError {
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

/// 504 response for a single inference run that exceeded the per-request
/// inference timeout (`--inference-timeout-secs`). Distinct from the pool
/// `timeout` (503): the slot was free, the *run* itself was too slow / wedged,
/// so there is no `Retry-After` — retrying the same payload would time out
/// again. Extracted (mirroring [`api_timeout_error`]) so the status + code can
/// be asserted without a model.
fn api_inference_timeout_error() -> ApiError {
    api_error(
        StatusCode::GATEWAY_TIMEOUT,
        "Inference timed out.",
        "inference_timeout",
    )
}

/// Per-segment error carried over the SSE channel: a stable machine-readable
/// code plus a sanitized message, mirroring the WebSocket error contract so
/// SSE clients get the same codes (`inference_error`, `inference_panic`,
/// `inference_timeout`, …) instead of one generic string.
struct StreamError {
    code: &'static str,
    message: String,
}

/// Render one SSE segment-or-error result to the JSON payload string sent in
/// the `data:` field. Pure (no I/O) so the per-variant error `code`, the
/// `inference_panic` / `inference_timeout` events, and the partial/final
/// framing can be unit-tested without a model.
fn sse_data_payload(
    result: &Result<gigastt_core::inference::TranscriptSegment, StreamError>,
) -> String {
    match result {
        Ok(seg) => {
            let ty = if seg.is_final { "final" } else { "partial" };
            serde_json::json!({
                "type": ty,
                "text": seg.text,
                "timestamp": seg.timestamp,
                "words": seg.words,
            })
            .to_string()
        }
        Err(err) => serde_json::json!({
            "type": "error",
            "message": err.message,
            "code": err.code,
        })
        .to_string(),
    }
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
/// minimal bootstrap responder (see [`super::run_with_config_loading`]) that
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
        supported_rates: super::config::SUPPORTED_RATES.to_vec(),
        punctuation: engine.has_punctuator(),
        itn: engine.has_itn(),
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
    Query(params): Query<ExportParams>,
    body: Bytes,
) -> Result<Response, ApiError> {
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
    let limits = state.limits.load();
    if body.len() > limits.body_limit_bytes {
        return Err(api_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "Request body exceeds the configured size limit",
            "payload_too_large",
        ));
    }

    // Snapshot the current engine once at request start; a concurrent hot-reload
    // swaps the `ArcSwap`, but this request keeps the engine (and its pool) it
    // began with for its whole lifetime.
    let engine = state.engine.load_full();

    // Checkout a session triplet from the batch pool (blocks if none
    // available) — this is a long file-transcription job, so it draws from the
    // dedicated batch pool when one exists (falling back to the interactive
    // pool otherwise) to avoid starving WebSocket / SSE streaming. The guard's
    // lifetime is stripped via `into_owned` so the triplet can travel through
    // `spawn_blocking`; the reservation handles checkin.
    let checkout_start = std::time::Instant::now();
    let guard = match tokio::time::timeout(
        std::time::Duration::from_secs(limits.pool_checkout_timeout_secs),
        engine.pool_for_batch().checkout(),
    )
    .await
    {
        Ok(Ok(guard)) => guard,
        Ok(Err(_pool_closed)) => return Err(api_pool_closed_error()),
        Err(_timeout) => {
            if let Some(ref reg) = state.metrics_registry {
                reg.counter_inc("gigastt_pool_timeouts_total", &[], 1);
                reg.histogram_record(
                    "gigastt_pool_checkout_duration_seconds",
                    &[],
                    checkout_start.elapsed().as_secs_f64(),
                );
            }
            return Err(api_timeout_error(&limits));
        }
    };
    if let Some(ref reg) = state.metrics_registry {
        reg.histogram_record(
            "gigastt_pool_checkout_duration_seconds",
            &[],
            checkout_start.elapsed().as_secs_f64(),
        );
    }
    let mut reservation = guard.into_owned();

    let inference_start = std::time::Instant::now();
    let span = tracing::Span::current();
    let handle = tokio::task::spawn_blocking(move || {
        let _enter = span.enter();
        // catch_unwind ensures triplet is returned to pool even on panic
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // `body` is an `axum::body::Bytes` (re-export of `bytes::Bytes`):
            // `clone()` is a refcount bump, not a data copy, so the decode
            // path shares the original upload buffer.
            engine.transcribe_bytes_shared(body, &mut reservation)
        }));
        match r {
            Ok(inference_result) => inference_result,
            Err(_) => {
                tracing::error!("Panic in REST transcribe — triplet recovered");
                Err(gigastt_core::error::GigasttError::Inference {
                    source: anyhow::anyhow!("Inference thread panicked").into(),
                })
            }
        }
        // reservation dropped here automatically returns the triplet to the pool
    });

    // Guard the blocking ONNX run with the per-request inference timeout
    // (`0` disables). `spawn_blocking` can't be cancelled, so the detached task
    // keeps the triplet and returns the slot to the pool only when the run
    // finishes; the client gets a typed `inference_timeout` (504) immediately.
    let inference_timeout_secs = limits.inference_timeout_secs;
    let result = if inference_timeout_secs == 0 {
        handle.await
    } else {
        match tokio::time::timeout(
            std::time::Duration::from_secs(inference_timeout_secs),
            handle,
        )
        .await
        {
            Ok(r) => r,
            Err(_elapsed) => {
                if let Some(ref reg) = state.metrics_registry {
                    reg.counter_inc("gigastt_inference_timeouts_total", &[], 1);
                }
                tracing::error!("REST inference exceeded {inference_timeout_secs}s — aborting");
                return Err(api_inference_timeout_error());
            }
        }
    };
    if let Some(ref reg) = state.metrics_registry {
        reg.histogram_record(
            "gigastt_inference_duration_seconds",
            &[],
            inference_start.elapsed().as_secs_f64(),
        );
    }

    match result {
        Ok(Ok(result)) => {
            if let Some(rendered) = render_export_response(&result, &params)? {
                Ok(rendered)
            } else {
                // Default JSON response. `?segments=true` adds a cue-grouped
                // `segments` array (same boundaries as SRT/VTT) alongside the
                // unchanged top-level `text` / `words` / `duration`; absent
                // otherwise so existing clients see the exact same shape.
                let segments = if params.segments.unwrap_or(false) {
                    Some(gigastt_core::export::to_segments(
                        &result.words,
                        params.max_chars_per_line.unwrap_or(80),
                        params.max_words_per_line.unwrap_or(14),
                    ))
                } else {
                    None
                };
                Ok(Json(TranscribeResponse {
                    text: result.text,
                    words: result.words,
                    duration: result.duration_s,
                    segments,
                })
                .into_response())
            }
        }
        Ok(Err(e)) => {
            tracing::error!("Transcription error: {e}");
            Err(api_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "Transcription failed. Check audio format.",
                "transcription_error",
            ))
        }
        Err(e) => {
            // spawn_blocking task itself failed (e.g., runtime shutdown).
            // The reservation was dropped inside the closure and the triplet
            // was returned to the pool automatically.
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
    let limits = state.limits.load();
    if body.len() > limits.body_limit_bytes {
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
        // catch_unwind mirrors the REST handler: a panic inside the blocking
        // decode (e.g. a crafted container that trips an upstream arithmetic
        // panic) is absorbed and surfaced as a normal decode error instead of a
        // `JoinError`, so the SSE path returns a clean 422 rather than a 500.
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gigastt_core::inference::audio::decode_audio_bytes_shared(body)
        })) {
            Ok(inner) => inner,
            Err(_) => {
                tracing::error!("Panic in SSE audio decode — treated as decode error");
                Err(anyhow::anyhow!("Audio decode thread panicked"))
            }
        }
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

    // Checkout a session triplet from the batch pool — SSE file transcription
    // decodes and transcribes the *entire* upload (holding the triplet for the
    // whole file), so it is a batch workload, not interactive streaming. Draw
    // from the dedicated batch pool when one exists (falling back to the
    // interactive pool otherwise) so it can't starve real-time WebSocket
    // streaming, matching `/v1/transcribe`. Strip the lifetime via `into_owned`
    // so the triplet can travel through `spawn_blocking`.
    // Snapshot the current engine once; a concurrent hot-reload swap only
    // affects later requests, so this stream rides the pool it started on.
    let engine = state.engine.load_full();
    let checkout_start = std::time::Instant::now();
    let guard = match tokio::time::timeout(
        std::time::Duration::from_secs(limits.pool_checkout_timeout_secs),
        engine.pool_for_batch().checkout(),
    )
    .await
    {
        Ok(Ok(guard)) => guard,
        Ok(Err(_pool_closed)) => return Err(api_pool_closed_error()),
        Err(_timeout) => {
            if let Some(ref reg) = state.metrics_registry {
                reg.counter_inc("gigastt_pool_timeouts_total", &[], 1);
                reg.histogram_record(
                    "gigastt_pool_checkout_duration_seconds",
                    &[],
                    checkout_start.elapsed().as_secs_f64(),
                );
            }
            return Err(api_timeout_error(&limits));
        }
    };
    if let Some(ref reg) = state.metrics_registry {
        reg.histogram_record(
            "gigastt_pool_checkout_duration_seconds",
            &[],
            checkout_start.elapsed().as_secs_f64(),
        );
    }
    let mut reservation = guard.into_owned();

    // Create mpsc channel for streaming segments from the inference task to SSE.
    let (tx, rx) = tokio::sync::mpsc::channel::<
        Result<gigastt_core::inference::TranscriptSegment, StreamError>,
    >(16);

    // The axum handler future has already returned by the time the SSE stream
    // starts flowing, so `with_graceful_shutdown` can't observe this task. Clone
    // the shutdown token and check it before every chunk so SIGTERM during a
    // long transcription drops cleanly.
    //
    // The whole file is transcribed in one blocking task, streaming each 1 s
    // chunk's segments out as they are produced. Each `process_chunk` is a small
    // bounded unit of work, so unlike the single-shot REST path it is not
    // wrapped by the per-request inference timeout; liveness on shutdown is
    // handled by the per-chunk cancellation check.
    let cancel = state.shutdown.clone();
    let tracker = state.tracker.clone();
    let span = tracing::Span::current();
    tracker.spawn_blocking(move || {
        let _enter = span.enter();
        // catch_unwind ensures the triplet is returned to the pool even on panic.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut stream_state = engine.create_state(false);
            let chunk_size = 16000; // 1 second at 16 kHz

            for chunk in samples.chunks(chunk_size) {
                if cancel.is_cancelled() {
                    tracing::info!("SSE transcription cancelled by shutdown");
                    return;
                }
                match engine.process_chunk(chunk, &mut stream_state, &mut reservation) {
                    Ok(segs) => {
                        for seg in segs {
                            if tx.blocking_send(Ok(seg)).is_err() {
                                // Receiver dropped (client disconnected).
                                return;
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.blocking_send(Err(StreamError {
                            code: e.code(),
                            message: "Transcription failed. Please check audio format.".into(),
                        }));
                        return;
                    }
                }
            }

            // Final decode of the sub-stride remainder, then flush — best-effort;
            // always emit so SSE clients receive a clean end-of-stream marker.
            if let Some(seg) = engine.finish_stream(&mut stream_state, &mut reservation) {
                let _ = tx.blocking_send(Ok(seg));
            }
        }));

        if result.is_err() {
            tracing::error!("Panic in SSE inference task — triplet recovered");
            // Mirror the WebSocket contract: surface a distinct `inference_panic`
            // code instead of ending the stream silently.
            let _ = tx.blocking_send(Err(StreamError {
                code: "inference_panic",
                message: "Inference failed unexpectedly.".into(),
            }));
        }
        // reservation dropped here automatically returns the triplet to the pool
    });

    // Convert receiver to SSE stream.
    let stream = tokio_stream::wrappers::ReceiverStream::new(rx)
        .map(|result| Ok(Event::default().data(sse_data_payload(&result))));

    // Explicit keep-alive: send a comment (`: \n\n`) every 15 s so nginx / ALB
    // do not close the connection during long transcriptions.
    Ok(Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text(""),
    ))
}

/// Whether the reload endpoint should accept a request from `peer`.
///
/// Model reload is an administrative, machine-local action: it must stay
/// reachable only from the loopback interface even under `--bind-all` /
/// `--cors-allow-any`, which would otherwise widen `origin_middleware` (the only
/// other gate). Pure so the loopback decision can be unit-tested without a model
/// or a live socket.
fn peer_is_loopback(peer: &std::net::SocketAddr) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_response_serialization() {
        let resp = HealthResponse {
            status: "ok".into(),
            model: "gigaam-v3-rnnt".into(),
            variant: "rnnt".into(),
            version: "0.3.0".into(),
            punctuation: true,
            itn: true,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["status"], "ok");
        assert_eq!(v["model"], "gigaam-v3-rnnt");
        assert_eq!(v["variant"], "rnnt");
        assert_eq!(v["punctuation"], true);
        assert_eq!(v["itn"], true);
    }

    #[test]
    fn test_transcribe_response_serialization() {
        let resp = TranscribeResponse {
            text: "hello".into(),
            words: vec![],

            duration: 1.5,
            segments: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["text"], "hello");
        assert_eq!(v["duration"], 1.5);
    }

    #[test]
    fn test_readiness_response_ready_serialization() {
        let resp = ReadinessResponse {
            status: "ready".into(),
            pool_available: 3,
            pool_total: 4,
            reason: None,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["status"], "ready");
        assert_eq!(json["pool_available"], 3);
        assert_eq!(json["pool_total"], 4);
        assert!(json.get("reason").is_none() || json["reason"].is_null());
    }

    #[test]
    fn test_readiness_response_not_ready_serialization() {
        let resp = ReadinessResponse {
            status: "not_ready".into(),
            pool_available: 0,
            pool_total: 4,
            reason: Some("pool_exhausted".into()),
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["status"], "not_ready");
        assert_eq!(json["reason"], "pool_exhausted");
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
        let resp = api_inference_timeout_error();
        let (parts, body) = resp.into_parts();
        assert_eq!(parts.status, StatusCode::GATEWAY_TIMEOUT);
        // A wedged run would just time out again, so no Retry-After hint.
        assert!(parts.headers.get(header::RETRY_AFTER).is_none());
        let bytes = axum::body::to_bytes(body, 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["code"], "inference_timeout");
    }

    #[test]
    fn test_sse_data_payload_preserves_error_codes() {
        // Per-variant code is preserved (not collapsed to a generic string),
        // including the distinct inference_panic / inference_timeout events.
        for code in [
            "invalid_audio",
            "inference_error",
            "inference_panic",
            "inference_timeout",
        ] {
            let payload = sse_data_payload(&Err(StreamError {
                code,
                message: "sanitized".into(),
            }));
            let v: serde_json::Value = serde_json::from_str(&payload).unwrap();
            assert_eq!(v["type"], "error");
            assert_eq!(v["code"], code);
            assert_eq!(v["message"], "sanitized");
        }
    }

    #[test]
    fn test_sse_data_payload_segment_framing() {
        // A final segment renders as type "final"; a non-final one as "partial".
        let seg = gigastt_core::inference::TranscriptSegment::empty_final();
        let final_payload = sse_data_payload(&Ok(seg));
        let v: serde_json::Value = serde_json::from_str(&final_payload).unwrap();
        assert_eq!(v["type"], "final");

        let mut partial = gigastt_core::inference::TranscriptSegment::empty_final();
        partial.is_final = false;
        let partial_payload = sse_data_payload(&Ok(partial));
        let v: serde_json::Value = serde_json::from_str(&partial_payload).unwrap();
        assert_eq!(v["type"], "partial");
    }

    #[tokio::test]
    #[ignore = "requires model"]
    async fn test_readiness_when_shutdown_cancelled() {
        let state = Arc::new(AppState {
            engine: engine_swap(test_engine()),
            limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
            metrics_registry: None,
            engine_builder: None,
            reload_lock: Arc::new(tokio::sync::Mutex::new(())),
            shutdown: tokio_util::sync::CancellationToken::new(),
            tracker: tokio_util::task::TaskTracker::new(),
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
    #[ignore = "requires model"]
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
        });
        let resp = readiness(State(state)).await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let bytes = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["status"], "not_ready");
        assert_eq!(v["reason"], "pool_exhausted");
    }

    #[tokio::test]
    #[ignore = "requires model"]
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
        });
        let body = Bytes::from(vec![0u8; 100]);
        let result = transcribe(State(state), Query(ExportParams::default()), body).await;
        match result {
            Err(resp) => assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE),
            Ok(_) => panic!("expected payload_too_large error"),
        }
    }

    #[tokio::test]
    #[ignore = "requires model"]
    async fn test_models_with_metrics() {
        let state = Arc::new(AppState {
            engine: engine_swap(test_engine()),
            limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
            metrics_registry: Some(Arc::new(MetricsRegistry::new())),
            engine_builder: None,
            reload_lock: Arc::new(tokio::sync::Mutex::new(())),
            shutdown: tokio_util::sync::CancellationToken::new(),
            tracker: tokio_util::task::TaskTracker::new(),
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
    #[ignore = "requires model"]
    async fn test_readiness_with_metrics() {
        let state = Arc::new(AppState {
            engine: engine_swap(fresh_engine()),
            limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
            metrics_registry: Some(Arc::new(MetricsRegistry::new())),
            engine_builder: None,
            reload_lock: Arc::new(tokio::sync::Mutex::new(())),
            shutdown: tokio_util::sync::CancellationToken::new(),
            tracker: tokio_util::task::TaskTracker::new(),
        });
        let resp = readiness(State(state)).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    #[ignore = "requires model"]
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
        });
        let body = Bytes::from(vec![0u8; 100]);
        let result = transcribe(State(state), Query(ExportParams::default()), body).await;
        match result {
            Err(resp) => assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE),
            Ok(_) => panic!("expected pool_closed error"),
        }
    }

    #[tokio::test]
    #[ignore = "requires model"]
    async fn test_transcribe_stream_invalid_audio() {
        let state = Arc::new(AppState {
            engine: engine_swap(test_engine()),
            limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
            metrics_registry: None,
            engine_builder: None,
            reload_lock: Arc::new(tokio::sync::Mutex::new(())),
            shutdown: tokio_util::sync::CancellationToken::new(),
            tracker: tokio_util::task::TaskTracker::new(),
        });
        let body = Bytes::from(vec![0u8; 100]);
        let result = transcribe_stream(State(state), body).await;
        match result {
            Err(resp) => assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY),
            Ok(_) => panic!("expected invalid_audio error"),
        }
    }

    #[tokio::test]
    #[ignore = "requires model"]
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
        });
        let body = Bytes::from(vec![0u8; 100]);
        let result = transcribe_stream(State(state), body).await;
        match result {
            Err(resp) => assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE),
            Ok(_) => panic!("expected payload_too_large error"),
        }
    }

    #[tokio::test]
    #[ignore = "requires model"]
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
        });
        let body = minimal_wav();
        let result = transcribe_stream(State(state), body).await;
        match result {
            Err(resp) => assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE),
            Ok(_) => panic!("expected pool_closed error"),
        }
    }

    #[tokio::test]
    #[ignore = "requires model"]
    async fn test_transcribe_with_metrics() {
        let state = Arc::new(AppState {
            engine: engine_swap(test_engine()),
            limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
            metrics_registry: Some(Arc::new(MetricsRegistry::new())),
            engine_builder: None,
            reload_lock: Arc::new(tokio::sync::Mutex::new(())),
            shutdown: tokio_util::sync::CancellationToken::new(),
            tracker: tokio_util::task::TaskTracker::new(),
        });
        let body = short_wav();
        match transcribe(State(state), Query(ExportParams::default()), body).await {
            Ok(_) => {}
            Err(_) => panic!("transcribe with metrics failed"),
        }
    }

    #[tokio::test]
    #[ignore = "requires model"]
    async fn test_transcribe_stream_with_metrics() {
        let state = Arc::new(AppState {
            engine: engine_swap(test_engine()),
            limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
            metrics_registry: Some(Arc::new(MetricsRegistry::new())),
            engine_builder: None,
            reload_lock: Arc::new(tokio::sync::Mutex::new(())),
            shutdown: tokio_util::sync::CancellationToken::new(),
            tracker: tokio_util::task::TaskTracker::new(),
        });
        let body = short_wav();
        match transcribe_stream(State(state), body).await {
            Ok(_) => {}
            Err(_) => panic!("transcribe_stream with metrics failed"),
        }
    }

    #[tokio::test]
    #[ignore = "requires model"]
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

    fn sample_export_result() -> gigastt_core::inference::TranscribeResult {
        use gigastt_core::inference::WordInfo;
        gigastt_core::inference::TranscribeResult {
            text: "привет мир".into(),
            words: vec![
                WordInfo::new("привет", 0.0, 0.5, 0.98, Some(0)),
                WordInfo::new("мир", 0.6, 1.0, 0.97, Some(0)),
            ],
            duration_s: 1.0,
        }
    }

    #[tokio::test]
    async fn test_render_export_default_returns_none() {
        let result = sample_export_result();
        let params = ExportParams::default();
        assert!(render_export_response(&result, &params).unwrap().is_none());
    }

    #[tokio::test]
    async fn test_render_export_txt() {
        let result = sample_export_result();
        let params = ExportParams {
            format: Some("txt".into()),
            ..ExportParams::default()
        };
        let resp = render_export_response(&result, &params).unwrap().unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
        assert_eq!(body, "привет мир");
    }

    #[tokio::test]
    async fn test_render_export_srt_content_type() {
        let result = sample_export_result();
        let params = ExportParams {
            format: Some("srt".into()),
            ..ExportParams::default()
        };
        let resp = render_export_response(&result, &params).unwrap().unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "application/x-subrip; charset=utf-8"
        );
        let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("[SPEAKER_0] привет мир"));
    }

    #[tokio::test]
    async fn test_render_export_vtt_download_header() {
        let result = sample_export_result();
        let params = ExportParams {
            format: Some("vtt".into()),
            download: Some("recording.vtt".into()),
            ..ExportParams::default()
        };
        let resp = render_export_response(&result, &params).unwrap().unwrap();
        assert_eq!(
            resp.headers().get(header::CONTENT_DISPOSITION).unwrap(),
            "attachment; filename=\"recording.vtt\""
        );
    }

    #[tokio::test]
    async fn test_render_export_download_filename_with_control_char_does_not_panic() {
        // The download filename is user-controlled; a control character must not
        // produce an invalid header value / panic — it falls back to the default.
        let result = sample_export_result();
        let params = ExportParams {
            format: Some("srt".into()),
            download: Some("evil\r\nInjected: x".into()),
            ..ExportParams::default()
        };
        let resp = render_export_response(&result, &params).unwrap().unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get(header::CONTENT_DISPOSITION).unwrap(),
            "attachment; filename=\"transcript.srt\""
        );
    }

    #[tokio::test]
    async fn test_render_export_invalid_format() {
        let result = sample_export_result();
        let params = ExportParams {
            format: Some("docx".into()),
            ..ExportParams::default()
        };
        let err = render_export_response(&result, &params).unwrap_err();
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_render_export_invalid_format_body_code() {
        // The invalid-format error carries the machine-readable `invalid_format`
        // code so clients can distinguish it from other 400s.
        let result = sample_export_result();
        let params = ExportParams {
            format: Some("xml".into()),
            ..ExportParams::default()
        };
        let err = render_export_response(&result, &params).unwrap_err();
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(err.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["code"], "invalid_format");
    }

    #[tokio::test]
    async fn test_render_export_uppercase_json_returns_none() {
        // Format negotiation is case-insensitive: an explicit (any-case) "json"
        // means "keep the default TranscribeResponse contract", so the helper
        // returns None instead of building a Response.
        let result = sample_export_result();
        let params = ExportParams {
            format: Some("JSON".into()),
            ..ExportParams::default()
        };
        assert!(render_export_response(&result, &params).unwrap().is_none());
    }

    #[tokio::test]
    async fn test_render_export_uppercase_format_renders() {
        // Non-JSON format strings are also case-insensitive (parsed via
        // ExportFormat::from_str), so "SRT" still renders subtitles.
        let result = sample_export_result();
        let params = ExportParams {
            format: Some("SRT".into()),
            ..ExportParams::default()
        };
        let resp = render_export_response(&result, &params).unwrap().unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "application/x-subrip; charset=utf-8"
        );
    }

    #[tokio::test]
    async fn test_render_export_empty_download_uses_default_name() {
        // An empty `download` value still requests an attachment; the helper
        // synthesizes the default `transcript.<ext>` filename.
        let result = sample_export_result();
        let params = ExportParams {
            format: Some("vtt".into()),
            download: Some(String::new()),
            ..ExportParams::default()
        };
        let resp = render_export_response(&result, &params).unwrap().unwrap();
        assert_eq!(
            resp.headers().get(header::CONTENT_DISPOSITION).unwrap(),
            "attachment; filename=\"transcript.vtt\""
        );
    }

    #[tokio::test]
    async fn test_render_export_md_includes_word_timestamps() {
        // The Markdown path honours `word_timestamps` and renders the per-word
        // table; the content type is text/markdown.
        let result = sample_export_result();
        let params = ExportParams {
            format: Some("md".into()),
            word_timestamps: Some(true),
            ..ExportParams::default()
        };
        let resp = render_export_response(&result, &params).unwrap().unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "text/markdown; charset=utf-8"
        );
        let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("# Transcript"));
        assert!(text.contains("| Word | Start | End |"));
    }

    #[tokio::test]
    async fn test_render_export_line_break_opts_passed_through() {
        // Tight per-line caps must be threaded into RenderOpts so the rendered
        // subtitles actually break — proving the params override the defaults.
        let result = sample_export_result();
        let loose = ExportParams {
            format: Some("srt".into()),
            ..ExportParams::default()
        };
        let tight = ExportParams {
            format: Some("srt".into()),
            max_words_per_line: Some(1),
            ..ExportParams::default()
        };
        let loose_resp = render_export_response(&result, &loose).unwrap().unwrap();
        let tight_resp = render_export_response(&result, &tight).unwrap().unwrap();
        let loose_body = axum::body::to_bytes(loose_resp.into_body(), 4096)
            .await
            .unwrap();
        let tight_body = axum::body::to_bytes(tight_resp.into_body(), 4096)
            .await
            .unwrap();
        let loose_text = String::from_utf8(loose_body.to_vec()).unwrap();
        let tight_text = String::from_utf8(tight_body.to_vec()).unwrap();
        // One word per line yields one cue per word (more "-->" arrows) than the
        // default 14-words-per-line grouping.
        let loose_cues = loose_text.matches("-->").count();
        let tight_cues = tight_text.matches("-->").count();
        assert!(
            tight_cues > loose_cues,
            "tight={tight_cues} should exceed loose={loose_cues}"
        );
    }

    #[tokio::test]
    async fn test_render_export_md_segments_emits_headers() {
        // `format=md` + `segments=true` switches Markdown to `### [mm:ss]`
        // section headers over the cue boundaries, dropping the flat
        // `# Transcript` blob.
        let result = sample_export_result();
        let params = ExportParams {
            format: Some("md".into()),
            segments: Some(true),
            ..ExportParams::default()
        };
        let resp = render_export_response(&result, &params).unwrap().unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "text/markdown; charset=utf-8"
        );
        let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("### [00:00]"));
        assert!(text.contains("[SPEAKER_0] привет мир"));
        // Segment mode replaces the flat transcript blob.
        assert!(!text.contains("# Transcript"));
    }

    #[tokio::test]
    async fn test_render_export_md_without_segments_unchanged() {
        // Plain `format=md` (no segments) keeps the existing frontmatter +
        // `# Transcript` layout — segment mode is strictly opt-in.
        let result = sample_export_result();
        let params = ExportParams {
            format: Some("md".into()),
            ..ExportParams::default()
        };
        let resp = render_export_response(&result, &params).unwrap().unwrap();
        let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("# Transcript"));
        assert!(!text.contains("### ["));
    }

    #[tokio::test]
    async fn test_render_export_segments_ignored_for_srt() {
        // `segments=true` is a JSON/Markdown affordance; SRT is already
        // cue-based and must render identically with or without the flag.
        let result = sample_export_result();
        let plain = ExportParams {
            format: Some("srt".into()),
            ..ExportParams::default()
        };
        let with_segments = ExportParams {
            format: Some("srt".into()),
            segments: Some(true),
            ..ExportParams::default()
        };
        let a = render_export_response(&result, &plain).unwrap().unwrap();
        let b = render_export_response(&result, &with_segments)
            .unwrap()
            .unwrap();
        let a_body = axum::body::to_bytes(a.into_body(), 4096).await.unwrap();
        let b_body = axum::body::to_bytes(b.into_body(), 4096).await.unwrap();
        assert_eq!(a_body, b_body);
    }

    #[test]
    fn test_transcribe_response_omits_segments_when_none() {
        // The default response must be byte-identical to the pre-feature shape:
        // no `segments` key when the caller didn't ask for it.
        let resp = TranscribeResponse {
            text: "hello".into(),
            words: vec![],
            duration: 1.5,
            segments: None,
        };
        let v = serde_json::to_value(&resp).unwrap();
        assert!(v.get("segments").is_none());
        assert_eq!(v["text"], "hello");
        assert_eq!(v["duration"], 1.5);
    }

    #[test]
    fn test_transcribe_response_includes_segments_when_present() {
        use gigastt_core::export::to_segments;
        use gigastt_core::inference::WordInfo;
        let words = vec![
            WordInfo::new("привет", 0.0, 0.5, 0.98, None),
            WordInfo::new("мир", 0.6, 1.0, 0.97, None),
        ];
        let resp = TranscribeResponse {
            text: "привет мир".into(),
            words: words.clone(),
            duration: 1.0,
            segments: Some(to_segments(&words, 80, 14)),
        };
        let v = serde_json::to_value(&resp).unwrap();
        let segments = v["segments"].as_array().unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0]["start"], 0.0);
        assert_eq!(segments[0]["end"], 1.0);
        assert_eq!(segments[0]["text"], "привет мир");
        assert_eq!(segments[0]["words"][0]["word"], "привет");
    }

    #[test]
    fn test_export_params_deserialize_from_query() {
        // The query-param shape drives format negotiation; confirm axum's Query
        // extractor maps every field so the handler sees the caller's choices.
        let uri: axum::http::Uri = "http://x/?format=srt&download=out.srt&max_chars_per_line=20&max_words_per_line=3&word_timestamps=true&segments=true"
            .parse()
            .unwrap();
        let Query(params): Query<ExportParams> = Query::try_from_uri(&uri).unwrap();
        assert_eq!(params.format.as_deref(), Some("srt"));
        assert_eq!(params.download.as_deref(), Some("out.srt"));
        assert_eq!(params.max_chars_per_line, Some(20));
        assert_eq!(params.max_words_per_line, Some(3));
        assert_eq!(params.word_timestamps, Some(true));
        assert_eq!(params.segments, Some(true));
    }

    #[test]
    fn test_export_params_default_empty_query() {
        // No query params -> all None, which the handler maps to JSON defaults.
        let uri: axum::http::Uri = "http://x/".parse().unwrap();
        let Query(params): Query<ExportParams> = Query::try_from_uri(&uri).unwrap();
        assert!(params.format.is_none());
        assert!(params.download.is_none());
        assert!(params.max_chars_per_line.is_none());
    }

    #[test]
    fn test_model_info_serialization_shape() {
        // ModelInfo is the /v1/models contract; assert the field names/values
        // clients depend on are present and correctly typed.
        let info = ModelInfo {
            id: "gigaam-v3-rnnt".into(),
            name: "GigaAM v3 RNN-T".into(),
            variant: "rnnt".into(),
            version: "0.9.0".into(),
            encoder: "int8".into(),
            vocab_size: 34,
            sample_rate: 16000,
            pool_size: 4,
            pool_available: 3,
            supported_formats: vec!["wav".into(), "mp3".into()],
            supported_rates: vec![16000, 48000],
            punctuation: true,
            itn: true,
            diarization: false,
        };
        let v = serde_json::to_value(&info).unwrap();
        assert_eq!(v["id"], "gigaam-v3-rnnt");
        assert_eq!(v["variant"], "rnnt");
        assert_eq!(v["encoder"], "int8");
        assert_eq!(v["vocab_size"], 34);
        assert_eq!(v["sample_rate"], 16000);
        assert_eq!(v["punctuation"], true);
        assert_eq!(v["itn"], true);
        assert_eq!(v["diarization"], false);
        assert_eq!(v["supported_rates"][1], 48000);
    }

    #[tokio::test]
    async fn test_api_inference_timeout_error_body_message() {
        // The 504 inference-timeout body should not leak internals, just the
        // stable code + a sanitized message.
        let resp = api_inference_timeout_error();
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

    #[test]
    fn test_peer_is_loopback_guard() {
        use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr};
        // IPv4 + IPv6 loopback are accepted regardless of source port.
        assert!(peer_is_loopback(&SocketAddr::from((
            Ipv4Addr::LOCALHOST,
            5000
        ))));
        assert!(peer_is_loopback(&SocketAddr::from((
            Ipv6Addr::LOCALHOST,
            5000
        ))));
        // A non-loopback peer (LAN / public) is rejected — reload must stay local
        // even under --bind-all / --cors-allow-any.
        assert!(!peer_is_loopback(&SocketAddr::from((
            Ipv4Addr::new(192, 168, 1, 10),
            9876
        ))));
        assert!(!peer_is_loopback(&SocketAddr::from((
            Ipv4Addr::new(10, 0, 0, 1),
            9876
        ))));
        assert!(!peer_is_loopback(&SocketAddr::from((
            Ipv4Addr::new(8, 8, 8, 8),
            443
        ))));
    }

    #[tokio::test]
    #[ignore = "requires model"]
    async fn test_reload_rejects_non_loopback_peer() {
        // The loopback guard fires before any engine work: a non-loopback
        // ConnectInfo yields 403 `loopback_only` even with a builder present.
        // Model-gated only because `AppState` needs a concrete `Engine`; the
        // pure guard logic is covered model-free by `test_peer_is_loopback_guard`.
        use std::net::{Ipv4Addr, SocketAddr};
        let state = Arc::new(AppState {
            engine: engine_swap(test_engine()),
            engine_builder: Some(Arc::new(|| {
                anyhow::bail!("builder must not run for a rejected peer")
            })),
            reload_lock: Arc::new(tokio::sync::Mutex::new(())),
            limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
            metrics_registry: None,
            shutdown: tokio_util::sync::CancellationToken::new(),
            tracker: tokio_util::task::TaskTracker::new(),
        });
        let peer = SocketAddr::from((Ipv4Addr::new(203, 0, 113, 7), 40000));
        let resp = reload(axum::extract::ConnectInfo(peer), State(state)).await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let bytes = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["code"], "loopback_only");
    }

    #[tokio::test]
    #[ignore = "requires model"]
    async fn test_reload_unsupported_when_no_builder() {
        // A loopback peer with no stored builder (the thin `run_with_shutdown` /
        // test path) gets `reload_unsupported`, not a swap.
        use std::net::{Ipv4Addr, SocketAddr};
        let state = Arc::new(AppState {
            engine: engine_swap(test_engine()),
            engine_builder: None,
            reload_lock: Arc::new(tokio::sync::Mutex::new(())),
            limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
            metrics_registry: None,
            shutdown: tokio_util::sync::CancellationToken::new(),
            tracker: tokio_util::task::TaskTracker::new(),
        });
        let peer = SocketAddr::from((Ipv4Addr::LOCALHOST, 40000));
        let resp = reload(axum::extract::ConnectInfo(peer), State(state)).await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let bytes = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["code"], "reload_unsupported");
    }

    #[test]
    fn test_sse_data_payload_includes_words_and_timestamp() {
        // A successful segment carries text, timestamp and words through
        // unchanged so SSE clients can render word-level UI.
        use gigastt_core::inference::WordInfo;
        let mut seg = gigastt_core::inference::TranscriptSegment::empty_final();
        seg.text = "привет".into();
        seg.timestamp = 1.25;
        seg.words = vec![WordInfo::new("привет", 0.0, 0.5, 0.99, Some(0))];
        let payload = sse_data_payload(&Ok(seg));
        let v: serde_json::Value = serde_json::from_str(&payload).unwrap();
        assert_eq!(v["type"], "final");
        assert_eq!(v["text"], "привет");
        assert_eq!(v["timestamp"], 1.25);
        assert_eq!(v["words"][0]["word"], "привет");
    }

    fn test_engine() -> Arc<Engine> {
        use std::sync::OnceLock;
        static ENGINE: OnceLock<Arc<Engine>> = OnceLock::new();
        ENGINE
            .get_or_init(|| {
                Arc::new(
                    Engine::load_with_pool_size(&gigastt_core::model::default_model_dir(), 1)
                        .unwrap(),
                )
            })
            .clone()
    }

    fn fresh_engine() -> Arc<Engine> {
        Arc::new(Engine::load_with_pool_size(&gigastt_core::model::default_model_dir(), 1).unwrap())
    }

    /// Wrap an engine handle in the `ArcSwap` the `AppState` now holds. Keeps
    /// the model-gated test constructors terse after the hot-reload swap change.
    fn engine_swap(engine: Arc<Engine>) -> Arc<ArcSwap<Engine>> {
        Arc::new(ArcSwap::new(engine))
    }

    fn minimal_wav() -> Bytes {
        let data_size = 4u32;
        let file_size = 44 + data_size - 8;
        let mut wav = vec![];
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&file_size.to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&16000u32.to_le_bytes());
        wav.extend_from_slice(&(16000u32 * 2).to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());
        wav.extend_from_slice(&0i16.to_le_bytes());
        wav.extend_from_slice(&0i16.to_le_bytes());
        Bytes::from(wav)
    }

    fn short_wav() -> Bytes {
        let sample_rate = 16000u32;
        let duration_s = 0.1f32;
        let num_samples = (sample_rate as f32 * duration_s) as u32;
        let data_size = num_samples * 2;
        let file_size = 44 + data_size - 8;
        let mut wav = vec![];
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&file_size.to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());
        for _ in 0..num_samples {
            wav.extend_from_slice(&0i16.to_le_bytes());
        }
        Bytes::from(wav)
    }
}
