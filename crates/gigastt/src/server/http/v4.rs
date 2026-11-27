//! Explicit full-context file server; legacy interactive Engine is not involved.
use super::{
    error::{ApiError, api_error, api_inference_timeout_error, api_timeout_error},
    export::{ExportParams, render_export_response},
    state::JobServerState,
    transcribe::TranscribeResponse,
};
use crate::server::{
    RuntimeLimits, ServerConfig, config,
    jobs::{InMemoryJobStore, JobExecution, JobQueue, JobStatus, JobStore},
    middleware, rate_limit,
    upload::{UploadAdmission, UploadPermit},
};
use axum::{
    Extension, Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Multipart, Path, Query, State},
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use gigastt_core::{
    error::GigasttError,
    inference::{
        TranscribeResult, TranscriptSegment,
        v4::{V4Engine, V4Transcription},
    },
    model::TranscriptStyle,
};
use serde::Deserialize;
use serde_json::json;
use std::{
    convert::Infallible,
    net::SocketAddr,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use tokio::sync::Semaphore;
use tokio_util::{sync::CancellationToken, task::TaskTracker};

/// The legacy response envelope is a transport shape, not a legacy inference
/// path. Preserve the complete SentencePiece text and author word groups.
pub fn v4_result(result: V4Transcription) -> TranscribeResult {
    TranscribeResult {
        text: result.text,
        words: result.words,
        duration_s: result.duration_s,
        confidence: result.confidence,
    }
}

type NativeRun = Arc<
    dyn Fn(Bytes, Option<f64>, &AtomicBool) -> Result<TranscribeResult, GigasttError> + Send + Sync,
>;
#[derive(Clone)]
struct FileRunner {
    decode: NativeRun,
    style: TranscriptStyle,
    slot: Arc<Semaphore>,
    limits: Arc<RuntimeLimits>,
    shutdown: CancellationToken,
    tracker: TaskTracker,
}
struct FileState {
    runner: FileRunner,
    jobs: Option<JobServerState>,
}

fn model_id(style: TranscriptStyle) -> String {
    format!("gigaam-v4-rnnt-{}-fp32", style.as_str())
}
fn style_name(style: TranscriptStyle) -> &'static str {
    style.as_str()
}

#[derive(Default, Deserialize)]
struct FileSelectors {
    model_family: Option<String>,
    model_head: Option<String>,
    transcript_style: Option<String>,
    model_precision: Option<String>,
    language: Option<String>,
    prompt: Option<String>,
}
#[derive(Default)]
struct FileParams {
    export: ExportParams,
    selector: FileSelectors,
    unknown: bool,
}
impl FileParams {
    fn parse(uri: &axum::http::Uri) -> Result<Self, ApiError> {
        let error = |_| {
            api_error(
                StatusCode::BAD_REQUEST,
                "Invalid v4 query options",
                "invalid_query",
            )
        };
        let Query(export) = Query::<ExportParams>::try_from_uri(uri).map_err(error)?;
        let Query(selector) = Query::<FileSelectors>::try_from_uri(uri).map_err(error)?;
        let Query(raw) = Query::<std::collections::BTreeMap<String, String>>::try_from_uri(uri)
            .map_err(error)?;
        let unknown = raw.keys().any(|k| {
            !matches!(
                k.as_str(),
                "model_family"
                    | "model_head"
                    | "transcript_style"
                    | "model_precision"
                    | "language"
                    | "prompt"
                    | "format"
                    | "download"
                    | "max_chars_per_line"
                    | "max_words_per_line"
                    | "word_timestamps"
                    | "segments"
                    | "punctuation"
                    | "itn"
                    | "vad"
                    | "hotwords"
                    | "hotwords_boost"
                    | "variant"
                    | "channels"
                    | "diarization"
                    | "codec"
                    | "sample_rate"
            )
        });
        Ok(Self {
            export,
            selector,
            unknown,
        })
    }
}
fn unsupported(message: &str) -> ApiError {
    api_error(StatusCode::BAD_REQUEST, message, "unsupported_v4_option")
}
fn validate_params(params: &FileParams, style: TranscriptStyle) -> Result<(), ApiError> {
    let e = &params.export;
    if params
        .selector
        .model_family
        .as_deref()
        .is_some_and(|x| x != "gigaam_v4")
        || params
            .selector
            .model_head
            .as_deref()
            .is_some_and(|x| x != "rnnt")
        || params
            .selector
            .transcript_style
            .as_deref()
            .is_some_and(|x| x != style_name(style))
        || params
            .selector
            .model_precision
            .as_deref()
            .is_some_and(|x| x != "fp32")
        || e.variant.as_deref().is_some_and(|x| x != model_id(style))
    {
        return Err(api_error(
            StatusCode::CONFLICT,
            "Requested model profile is not loaded",
            "variant_not_loaded",
        ));
    }
    if e.punctuation == Some(true)
        || e.itn == Some(true)
        || e.vad == Some(true)
        || e.diarization == Some(true)
        || e.hotwords.is_some()
        || e.hotwords_boost.is_some()
        || e.channels.as_deref().is_some_and(|x| x != "mono")
        || e.codec.is_some()
        || e.sample_rate.is_some()
        || params.selector.language.is_some()
        || params.selector.prompt.is_some()
        || params.unknown
    {
        return Err(unsupported(
            "v4 requires full-context mono file input without extra normalization, language hints, hotwords, diarization, VAD, split channels or headerless codecs",
        ));
    }
    if let Some(f) = &e.format {
        use std::str::FromStr;
        gigastt_core::export::ExportFormat::from_str(f).map_err(|_| {
            api_error(
                StatusCode::BAD_REQUEST,
                "Invalid export format",
                "invalid_format",
            )
        })?;
    }
    Ok(())
}
fn validate_body(limits: &RuntimeLimits, body: &Bytes) -> Result<(), ApiError> {
    if body.is_empty() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "Empty request body",
            "empty_body",
        ));
    }
    if body.len() > limits.body_limit_bytes {
        return Err(api_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "Request body exceeds configured size limit",
            "payload_too_large",
        ));
    }
    Ok(())
}

#[derive(Debug)]
enum RunError {
    PoolTimeout,
    Timeout,
    Core(GigasttError),
    Join,
}
impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PoolTimeout => f.write_str("pool checkout timed out"),
            Self::Timeout => f.write_str("inference_timeout"),
            Self::Core(e) => e.fmt(f),
            Self::Join => f.write_str("native inference task failed"),
        }
    }
}
impl std::error::Error for RunError {}
impl From<GigasttError> for RunError {
    fn from(e: GigasttError) -> Self {
        Self::Core(e)
    }
}
fn api_run_error(error: RunError, limits: &RuntimeLimits) -> ApiError {
    match error {
        RunError::PoolTimeout => api_timeout_error(limits),
        RunError::Timeout => api_inference_timeout_error(None),
        RunError::Core(GigasttError::Cancelled) => api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Transcription cancelled",
            "cancelled",
        ),
        RunError::Core(e @ GigasttError::AudioTooLong { .. }) => {
            api_error(StatusCode::PAYLOAD_TOO_LARGE, &e.to_string(), e.code())
        }
        RunError::Core(e @ GigasttError::InvalidAudio { .. }) => {
            tracing::warn!("v4 audio rejected: {e:#}");
            api_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "Invalid audio: provide a supported file containing 20ms to 30s of mono audio",
                e.code(),
            )
        }
        other => {
            tracing::error!("v4 inference failed: {other:#}");
            api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Transcription failed",
                "internal",
            )
        }
    }
}
struct AbortOnDrop(Arc<AtomicBool>);
impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}
async fn cancelled(flag: &AtomicBool, shutdown: &CancellationToken) {
    loop {
        if flag.load(Ordering::Relaxed) {
            return;
        }
        tokio::select! {_ = shutdown.cancelled()=>return,_ = tokio::time::sleep(Duration::from_millis(10))=>{}}
    }
}
async fn deadline(seconds: u64) {
    if seconds == 0 {
        std::future::pending::<()>().await;
    } else {
        tokio::time::sleep(Duration::from_secs(seconds)).await;
    }
}
impl FileRunner {
    async fn run(&self, body: Bytes, abort: Arc<AtomicBool>) -> Result<TranscribeResult, RunError> {
        let _abort = AbortOnDrop(abort.clone());
        let permit = tokio::select! {
            biased;
            _=cancelled(&abort,&self.shutdown)=>return Err(GigasttError::Cancelled.into()),
            _=deadline(self.limits.pool_checkout_timeout_secs.max(1))=>return Err(RunError::PoolTimeout),
            p=self.slot.clone().acquire_owned()=>p.map_err(|_|GigasttError::Cancelled)?,
        };
        let decode = self.decode.clone();
        let limit = self.limits.max_audio_secs_opt();
        let run_abort = abort.clone();
        let task = self.tracker.spawn_blocking(move || {
            // Keep memory/admission and the request slot through the actual
            // native completion, even if the client timed out/disconnected.
            let _permit = permit;
            decode(body, limit, &run_abort)
        });
        tokio::select! {
            biased;
            _=cancelled(&abort,&self.shutdown)=>Err(GigasttError::Cancelled.into()),
            _=deadline(self.limits.inference_timeout_secs)=>Err(RunError::Timeout),
            result=task=>result.map_err(|_|RunError::Join)?.map_err(Into::into),
        }
    }
}

fn normal_response(result: TranscribeResult, params: &ExportParams) -> Result<Response, ApiError> {
    if let Some(response) = render_export_response(&result, params)? {
        return Ok(response);
    }
    let segments = params
        .segments
        .unwrap_or(false)
        .then(|| gigastt_core::export::to_transcript_segments(&result.words));
    Ok(Json(TranscribeResponse {
        text: result.text,
        words: result.words,
        duration: result.duration_s,
        confidence: result.confidence,
        segments,
        diarization: None,
    })
    .into_response())
}
async fn transcribe(
    State(state): State<Arc<FileState>>,
    uri: axum::http::Uri,
    Extension(permit): Extension<UploadPermit>,
    body: Bytes,
) -> Result<Response, ApiError> {
    let params = FileParams::parse(&uri)?;
    validate_params(&params, state.runner.style)?;
    validate_body(&state.runner.limits, &body)?;
    let result = state
        .runner
        .run(permit.retain(body), Arc::new(AtomicBool::new(false)))
        .await
        .map_err(|e| api_run_error(e, &state.runner.limits))?;
    normal_response(result, &params.export)
}
async fn stream(
    State(state): State<Arc<FileState>>,
    uri: axum::http::Uri,
    Extension(permit): Extension<UploadPermit>,
    body: Bytes,
) -> Result<Response, ApiError> {
    let params = FileParams::parse(&uri)?;
    validate_params(&params, state.runner.style)?;
    validate_body(&state.runner.limits, &body)?;
    if params.export.format.is_some()
        || params.export.download.is_some()
        || params.export.segments.is_some()
        || params.export.max_chars_per_line.is_some()
        || params.export.max_words_per_line.is_some()
        || params.export.word_timestamps.is_some()
    {
        return Err(unsupported(
            "v4 SSE returns one final transcript event, not an export attachment",
        ));
    }
    let body = permit.retain(body);
    let stream = futures_util::stream::once(async move {
        let payload = match state
            .runner
            .run(body, Arc::new(AtomicBool::new(false)))
            .await
        {
            Ok(result) => {
                let mut seg = TranscriptSegment::empty_final();
                seg.text = result.text;
                seg.committed = seg.text.clone();
                seg.words = result.words;
                seg.confidence = result.confidence;
                super::stream::sse_data_payload(&Ok(seg))
            }
            Err(error) => {
                let response = api_run_error(error, &state.runner.limits).into_response();
                let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
                    .await
                    .unwrap_or_default();
                let mut payload: serde_json::Value = serde_json::from_slice(&bytes)
                    .unwrap_or_else(|_| json!({"error":"Transcription failed","code":"internal"}));
                payload["type"] = json!("error");
                payload["message"] = payload["error"].clone();
                payload.to_string()
            }
        };
        Ok::<_, Infallible>(Event::default().data(payload))
    });
    Ok(Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)).text(""))
        .into_response())
}
async fn openai(
    State(state): State<Arc<FileState>>,
    Extension(permit): Extension<UploadPermit>,
    multipart: Multipart,
) -> Result<Response, ApiError> {
    let req = crate::server::openai::parse_v4_openai_multipart(multipart).await?;
    if req.options.model.as_deref() != Some(model_id(state.runner.style).as_str()) {
        return Err(api_error(
            StatusCode::CONFLICT,
            "Requested model is not loaded",
            "variant_not_loaded",
        ));
    }
    if req.options.stream
        || req.options.response_format == crate::server::openai::OpenAIResponseFormat::VerboseJson
        || req.options.include_words
        || req.options.include_segments
    {
        return Err(unsupported(
            "v4 OpenAI route supports json, text, srt and vtt; use /v1/transcribe/stream for one final SSE event or /v1/transcribe for word metadata",
        ));
    }
    validate_body(&state.runner.limits, &req.file)?;
    let result = state
        .runner
        .run(permit.retain(req.file), Arc::new(AtomicBool::new(false)))
        .await
        .map_err(|e| api_run_error(e, &state.runner.limits))?;
    Ok(crate::server::openai::render_openai_response(
        &result,
        &req.options,
    ))
}

impl JobExecution for FileRunner {
    async fn execute(
        &self,
        id: &str,
        store: Arc<dyn JobStore>,
        body: Bytes,
        params: ExportParams,
    ) -> anyhow::Result<TranscribeResult> {
        let abort = Arc::new(AtomicBool::new(false));
        store
            .update(id, {
                let abort = abort.clone();
                Box::new(move |j| {
                    if j.status == JobStatus::Cancelled {
                        abort.store(true, Ordering::Relaxed);
                    }
                    if j.status == JobStatus::Processing {
                        j.abort = Some(abort);
                    }
                })
            })
            .await?;
        validate_params(
            &FileParams {
                export: params,
                ..Default::default()
            },
            self.style,
        )
        .map_err(|_| GigasttError::InvalidInput {
            message: "Unsupported v4 file job options".into(),
        })?;
        let result = self.run(body, abort).await.map_err(|e| match e {
            RunError::Core(e) => anyhow::Error::new(e),
            other => anyhow::Error::new(other),
        })?;
        let duration = result.duration_s;
        store
            .update(
                id,
                Box::new(move |j| {
                    if j.status == JobStatus::Processing {
                        j.total_seconds = duration;
                    }
                }),
            )
            .await?;
        Ok(result)
    }
}
fn require_jobs(state: &FileState) -> Result<&JobServerState, ApiError> {
    state.jobs.as_ref().ok_or_else(|| {
        api_error(
            StatusCode::NOT_FOUND,
            "Job API is not enabled",
            "jobs_disabled",
        )
    })
}
async fn submit_job(
    State(state): State<Arc<FileState>>,
    uri: axum::http::Uri,
    body: Bytes,
) -> Result<Response, ApiError> {
    let params = FileParams::parse(&uri)?;
    validate_params(&params, state.runner.style)?;
    super::jobs_api::submit_job_for(
        require_jobs(&state)?,
        &state.runner.limits,
        params.export,
        body,
    )
    .await
}
async fn job_status(
    State(state): State<Arc<FileState>>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    super::jobs_api::get_job_for(require_jobs(&state)?, id).await
}
async fn job_result(
    State(state): State<Arc<FileState>>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    super::jobs_api::get_job_result_for(require_jobs(&state)?, id).await
}
async fn job_cancel(
    State(state): State<Arc<FileState>>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    super::jobs_api::cancel_job_for(require_jobs(&state)?, id).await
}
async fn job_events(
    State(state): State<Arc<FileState>>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    Ok(super::jobs_api::job_events_for(require_jobs(&state)?, id)
        .await?
        .into_response())
}

async fn health(State(state): State<Arc<FileState>>) -> Json<serde_json::Value> {
    Json(
        json!({"status":"ok","model":model_id(state.runner.style),"variant":model_id(state.runner.style),"family":"gigaam_v4","head":"rnnt","style":style_name(state.runner.style),"precision":"fp32","version":env!("CARGO_PKG_VERSION"),"punctuation":false,"itn":false,"streaming":false}),
    )
}
async fn models(State(state): State<Arc<FileState>>) -> Json<serde_json::Value> {
    Json(
        json!({"id":model_id(state.runner.style),"name":"GigaAM v4 RNNT FP32","variant":model_id(state.runner.style),"family":"gigaam_v4","head":"rnnt","style":style_name(state.runner.style),"precision":"fp32","encoder":"fp32","mode":"full_context","version":env!("CARGO_PKG_VERSION"),"sample_rate":16000,"vocab_size":1025,"pool_size":1,"pool_available":state.runner.slot.available_permits(),"max_audio_seconds":state.runner.limits.max_audio_secs_opt().unwrap_or(30.0).min(30.0),"supported_formats":["wav","mp3","m4a","ogg","flac","opus","webm"],"supported_rates":[],"punctuation":false,"itn":false,"diarization":false,"execution_provider":"cpu","streaming":false}),
    )
}
async fn ready(State(state): State<Arc<FileState>>) -> Response {
    let available = if state.runner.shutdown.is_cancelled() {
        0
    } else {
        state.runner.slot.available_permits()
    };
    let reason = if state.runner.shutdown.is_cancelled() {
        Some("shutting_down")
    } else if available == 0 {
        Some("pool_exhausted")
    } else {
        None
    };
    (
        if reason.is_none() {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        },
        Json(super::ReadinessResponse {
            status: if reason.is_none() {
                "ready"
            } else {
                "not_ready"
            }
            .into(),
            pool_available: available,
            pool_total: 1,
            reason: reason.map(str::to_owned),
        }),
    )
        .into_response()
}
async fn ws_unsupported() -> ApiError {
    api_error(
        StatusCode::CONFLICT,
        "v4 file profile does not support native WebSocket streaming",
        "streaming_unsupported",
    )
}
async fn reload_unsupported() -> ApiError {
    api_error(
        StatusCode::CONFLICT,
        "v4 file profile does not support hot reload",
        "reload_unsupported",
    )
}

/// Start an explicit, full-context v4 file-only server. No warmup, download,
/// model hot reload, legacy Engine, or automatic streaming fallback.
pub async fn run_v4_files(
    engine: V4Engine,
    style: TranscriptStyle,
    config: ServerConfig,
    shutdown: Option<tokio::sync::oneshot::Receiver<()>>,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        !config.metrics_enabled
            && config.config_path.is_none()
            && config.batch_pool_size == 0
            && config.limits.max_audio_secs <= 30,
        "v4 requires an audio limit no greater than 30 seconds, without metrics, config reload or a separate batch pool"
    );
    let addr: SocketAddr = format!("{}:{}", config.host, config.port).parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let shutdown_root = CancellationToken::new();
    let tracker = TaskTracker::new();
    let runner = FileRunner {
        decode: Arc::new(move |body, limit, flag| {
            engine
                .transcribe_bytes_with_limit(body, style, Some(flag), limit)
                .map(v4_result)
        }),
        style,
        slot: Arc::new(Semaphore::new(1)),
        limits: Arc::new(config.limits.clone()),
        shutdown: shutdown_root.clone(),
        tracker: tracker.clone(),
    };
    let jobs = if config.limits.jobs_enabled {
        let store: Arc<dyn JobStore> = Arc::new(InMemoryJobStore::new(config.limits.clone()));
        let queue = JobQueue::new(store.clone(), 1, 0, shutdown_root.clone());
        queue.spawn(runner.clone(), &tracker);
        Some(JobServerState { store, queue })
    } else {
        None
    };
    let state = Arc::new(FileState {
        runner,
        jobs: jobs.clone(),
    });
    let mut protected = Router::new()
        .route(
            "/v1/models",
            get(models).options(|| async { StatusCode::NO_CONTENT }),
        )
        .route(
            "/v1/transcribe",
            post(transcribe).options(|| async { StatusCode::NO_CONTENT }),
        )
        .route(
            "/v1/transcribe/stream",
            post(stream).options(|| async { StatusCode::NO_CONTENT }),
        )
        .route(
            "/v1/audio/transcriptions",
            post(openai).options(|| async { StatusCode::NO_CONTENT }),
        )
        .route(
            "/v1/ws",
            get(ws_unsupported).options(|| async { StatusCode::NO_CONTENT }),
        )
        .route(
            "/v1/admin/reload",
            post(reload_unsupported).options(|| async { StatusCode::NO_CONTENT }),
        );
    if jobs.is_some() {
        protected = protected
            .route(
                "/v1/jobs",
                post(submit_job).options(|| async { StatusCode::NO_CONTENT }),
            )
            .route(
                "/v1/jobs/{id}",
                get(job_status)
                    .delete(job_cancel)
                    .options(|| async { StatusCode::NO_CONTENT }),
            )
            .route(
                "/v1/jobs/{id}/result",
                get(job_result).options(|| async { StatusCode::NO_CONTENT }),
            )
            .route("/v1/jobs/{id}/events", get(job_events));
    }
    protected = protected.route_layer(axum::middleware::from_fn_with_state(
        UploadAdmission::new(1, config::pool_retry_after_secs(&config.limits)),
        crate::server::upload::admit,
    ));
    if config.limits.rate_limit_per_minute > 0 {
        let limiter = Arc::new(rate_limit::RateLimiter::new(
            config.limits.rate_limit_per_minute,
            config.limits.rate_limit_burst,
        ));
        let evict = limiter.clone();
        let cancel = shutdown_root.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {_=cancel.cancelled()=>break,_=tokio::time::sleep(Duration::from_secs(60))=>evict.evict_stale(Duration::from_secs(300))}
            }
        });
        let trust = config.trust_proxy;
        protected = protected.layer(axum::middleware::from_fn(move |req, next| {
            let limiter = limiter.clone();
            async move { rate_limit::rate_limit_middleware(limiter, trust, None, req, next).await }
        }));
    }
    let policy = Arc::new(config.origin_policy);
    let app = Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready))
        .merge(protected)
        .layer(DefaultBodyLimit::max(config.limits.body_limit_bytes))
        .layer(axum::middleware::from_fn(move |req, next| {
            let p = policy.clone();
            async move { middleware::origin_middleware(p, req, next).await }
        }))
        .layer(axum::middleware::from_fn(middleware::request_id_middleware))
        .with_state(state.clone());
    tracing::info!(
        "gigastt v4 full-context file server listening on http://{addr}, style {}",
        style_name(style)
    );
    let stop = shutdown_root.clone();
    let slot = state.runner.slot.clone();
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(async move {
        crate::server::listen::wait_for_shutdown(shutdown).await;
        stop.cancel();
        slot.close();
        if let Some(jobs) = jobs {
            jobs.queue.cancel_all_queued().await;
        }
    })
    .await?;
    tracker.close();
    if tokio::time::timeout(
        Duration::from_secs(config.limits.shutdown_drain_secs.max(1)),
        tracker.wait(),
    )
    .await
    .is_err()
    {
        tracing::warn!("v4 file server drain timeout; native work may still be finishing");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn test_v4_query_profile_and_unsupported_options() {
        let parse =
            |q: &str| FileParams::parse(&format!("/v1/transcribe?{q}").parse().unwrap()).unwrap();
        let p = parse(
            "model_family=gigaam_v4&model_head=rnnt&transcript_style=raw&model_precision=fp32&format=json&segments=true&punctuation=false",
        );
        assert!(validate_params(&p, TranscriptStyle::Raw).is_ok());
        assert_eq!(p.export.segments, Some(true));
        assert_eq!(p.export.punctuation, Some(false));
        for query in [
            "language=ru",
            "hotwords=foo",
            "vad=true",
            "diarization=true",
            "itn=true",
            "channels=split",
            "codec=pcmu",
            "unknown_option=yes",
            "model_precision=int8",
            "transcript_style=e2e",
            "variant=rnnt",
        ] {
            assert!(
                validate_params(&parse(query), TranscriptStyle::Raw).is_err(),
                "{query}"
            );
        }
    }

    struct ReleaseOnDrop(Arc<AtomicBool>);
    impl Drop for ReleaseOnDrop {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Relaxed);
        }
    }
    fn fake_runner(decode: NativeRun) -> FileRunner {
        FileRunner {
            decode,
            style: TranscriptStyle::Raw,
            slot: Arc::new(Semaphore::new(1)),
            limits: Arc::new(RuntimeLimits::default()),
            shutdown: CancellationToken::new(),
            tracker: TaskTracker::new(),
        }
    }

    #[test]
    fn test_v4_timeout_marker_matches_job_error_contract() {
        assert_eq!(RunError::Timeout.to_string(), "inference_timeout");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_v4_cancel_running_keeps_permit_until_native_completion() {
        let entered = Arc::new(tokio::sync::Notify::new());
        let released = Arc::new(AtomicBool::new(false));
        let _release_on_drop = ReleaseOnDrop(released.clone());
        let observed = Arc::new(AtomicBool::new(false));
        let runner = fake_runner(Arc::new({
            let entered = entered.clone();
            let released = released.clone();
            let observed = observed.clone();
            move |_, _, abort| {
                entered.notify_one();
                while !released.load(Ordering::Relaxed) {
                    if abort.load(Ordering::Relaxed) {
                        observed.store(true, Ordering::Relaxed);
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(GigasttError::Cancelled)
            }
        }));
        let abort = Arc::new(AtomicBool::new(false));
        let task = tokio::spawn({
            let runner = runner.clone();
            let abort = abort.clone();
            async move { runner.run(Bytes::from_static(b"fixture"), abort).await }
        });
        tokio::time::timeout(Duration::from_secs(2), entered.notified())
            .await
            .unwrap();
        abort.store(true, Ordering::Relaxed);
        assert!(matches!(
            tokio::time::timeout(Duration::from_secs(2), task)
                .await
                .unwrap()
                .unwrap(),
            Err(RunError::Core(GigasttError::Cancelled))
        ));
        assert_eq!(runner.slot.available_permits(), 0);
        // Let the native fake acknowledge the abort before actually returning.
        tokio::time::timeout(Duration::from_secs(2), async {
            while !observed.load(Ordering::Relaxed) {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        released.store(true, Ordering::Relaxed);
        runner.tracker.close();
        tokio::time::timeout(Duration::from_secs(2), runner.tracker.wait())
            .await
            .unwrap();
        assert_eq!(runner.slot.available_permits(), 1);
    }

    #[tokio::test]
    async fn test_v4_cancelled_waiter_never_decodes() {
        let calls = Arc::new(AtomicUsize::new(0));
        let runner = fake_runner(Arc::new({
            let calls = calls.clone();
            move |_, _, _| {
                calls.fetch_add(1, Ordering::Relaxed);
                Err(GigasttError::Cancelled)
            }
        }));
        let _occupied = runner.slot.clone().acquire_owned().await.unwrap();
        let flag = Arc::new(AtomicBool::new(true));
        assert!(matches!(
            runner.run(Bytes::from_static(b"fixture"), flag).await,
            Err(RunError::Core(GigasttError::Cancelled))
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_v4_timeout_and_disconnect_abort_without_releasing_native_slot() {
        for disconnect in [false, true] {
            let entered = Arc::new(tokio::sync::Notify::new());
            let release = Arc::new(AtomicBool::new(false));
            let _release_on_drop = ReleaseOnDrop(release.clone());
            let observed = Arc::new(AtomicBool::new(false));
            let mut runner = fake_runner(Arc::new({
                let entered = entered.clone();
                let release = release.clone();
                let observed = observed.clone();
                move |_, limit, abort| {
                    assert_eq!(limit, Some(3.0));
                    entered.notify_one();
                    while !release.load(Ordering::Relaxed) {
                        if abort.load(Ordering::Relaxed) {
                            observed.store(true, Ordering::Relaxed);
                        }
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    Err(GigasttError::Cancelled)
                }
            }));
            let limits = RuntimeLimits {
                max_audio_secs: 3,
                inference_timeout_secs: 1,
                ..Default::default()
            };
            runner.limits = Arc::new(limits);
            let task = tokio::spawn({
                let runner = runner.clone();
                async move {
                    runner
                        .run(
                            Bytes::from_static(b"fixture"),
                            Arc::new(AtomicBool::new(false)),
                        )
                        .await
                }
            });
            tokio::time::timeout(Duration::from_secs(2), entered.notified())
                .await
                .unwrap();
            if disconnect {
                task.abort();
                assert!(task.await.unwrap_err().is_cancelled());
            } else {
                assert!(matches!(
                    tokio::time::timeout(Duration::from_secs(3), task)
                        .await
                        .unwrap()
                        .unwrap(),
                    Err(RunError::Timeout)
                ));
            }
            assert_eq!(runner.slot.available_permits(), 0);
            tokio::time::timeout(Duration::from_secs(2), async {
                while !observed.load(Ordering::Relaxed) {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            })
            .await
            .unwrap();
            release.store(true, Ordering::Relaxed);
            runner.tracker.close();
            tokio::time::timeout(Duration::from_secs(2), runner.tracker.wait())
                .await
                .unwrap();
            assert_eq!(runner.slot.available_permits(), 1);
        }
    }
}
