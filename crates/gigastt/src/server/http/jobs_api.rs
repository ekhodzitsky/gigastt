//! Asynchronous job API handlers (`/v1/jobs`).

use axum::body::Bytes;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::http::header;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use futures_util::StreamExt;
use futures_util::stream::Stream;
use serde::Serialize;
use std::sync::Arc;

use super::super::config::{pool_retry_after_ms, pool_retry_after_secs};
use super::super::jobs::{JobEvent, JobStatus, JobStore};

use super::error::{ApiError, api_error};
use super::export::{ExportParams, render_export_response};
use super::state::{AppState, JobServerState};
use super::transcribe::TranscribeResponse;

/// POST /v1/jobs response.
#[derive(Serialize)]
pub struct JobSubmitResponse {
    pub job_id: String,
    pub status: &'static str,
    pub created_at: f64,
}

/// Return the job server state or a 404 if the async job API is disabled.
#[allow(clippy::result_large_err)]
fn require_jobs(state: &AppState) -> Result<&JobServerState, ApiError> {
    state.jobs.as_ref().ok_or_else(|| {
        api_error(
            StatusCode::NOT_FOUND,
            "Job API is not enabled",
            "jobs_disabled",
        )
    })
}

/// Fetch a job by id, mapping store errors to the standard HTTP responses.
async fn load_job(store: &dyn JobStore, id: &str) -> Result<super::super::jobs::Job, ApiError> {
    match store.get(id).await {
        Ok(Some(job)) => Ok(job),
        Ok(None) => Err(api_error(
            StatusCode::NOT_FOUND,
            "Job not found",
            "job_not_found",
        )),
        Err(e) => {
            tracing::error!("Failed to get job {id}: {e:#}");
            Err(api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to read job status",
                "internal",
            ))
        }
    }
}

/// POST /v1/jobs — enqueue a long audio file for asynchronous transcription.
///
/// Accepts the same body and query parameters as `/v1/transcribe`. Returns 202
/// with the job id; use `GET /v1/jobs/{id}` to poll and
/// `GET /v1/jobs/{id}/result` to fetch the finished transcript.
pub async fn submit_job(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ExportParams>,
    body: Bytes,
) -> Result<Response, ApiError> {
    let jobs = require_jobs(&state)?;
    let limits = state.limits.load();
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
            "Request body exceeds the configured size limit",
            "payload_too_large",
        ));
    }
    if jobs.store.is_full().await {
        return Err((
            StatusCode::TOO_MANY_REQUESTS,
            [(
                header::RETRY_AFTER,
                pool_retry_after_secs(&limits).to_string(),
            )],
            Json(serde_json::json!({
                "error": "Job queue is full",
                "code": "queue_full",
                "retry_after_ms": pool_retry_after_ms(&limits),
            })),
        )
            .into_response());
    }
    let split_channels = params.channels.as_deref() == Some("split");
    let request_diarization = params.diarization == Some(true);
    if split_channels && request_diarization {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "channels=split and diarization=true are mutually exclusive",
            "conflicting_modes",
        ));
    }

    let job = super::super::jobs::Job::queued(body, params);
    let created_at = job.created_at;
    let id = match jobs.store.create(job).await {
        Ok(id) => id,
        Err(e) => {
            tracing::error!("Failed to create job: {e:#}");
            return Err(api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to enqueue job",
                "internal",
            ));
        }
    };
    Ok((
        StatusCode::ACCEPTED,
        Json(JobSubmitResponse {
            job_id: id,
            status: "queued",
            created_at,
        }),
    )
        .into_response())
}

/// GET /v1/jobs/{id} — poll job status and progress.
pub async fn get_job(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Response, ApiError> {
    let jobs = require_jobs(&state)?;
    let job = load_job(&*jobs.store, &id).await?;
    Ok(Json(super::super::jobs::job_status_response(&job)).into_response())
}

/// GET /v1/jobs/{id}/result — fetch the finished transcription.
pub async fn get_job_result(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Response, ApiError> {
    let jobs = require_jobs(&state)?;
    let job = load_job(&*jobs.store, &id).await?;
    if !matches!(job.status, JobStatus::Done) {
        return Err(api_error(
            StatusCode::CONFLICT,
            "Job is not finished",
            "job_not_finished",
        ));
    }
    let Some(result) = job.result else {
        tracing::error!(job_id = %id, "Done job is missing result");
        return Err(api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Job result is missing",
            "internal",
        ));
    };
    if let Some(rendered) = render_export_response(&result, &job.params)? {
        Ok(rendered)
    } else {
        let segments = if job.params.segments.unwrap_or(false) {
            Some(gigastt_core::export::to_transcript_segments(&result.words))
        } else {
            None
        };
        Ok(Json(TranscribeResponse {
            text: result.text,
            words: result.words,
            duration: result.duration_s,
            confidence: result.confidence,
            segments,
        })
        .into_response())
    }
}

/// DELETE /v1/jobs/{id} — cancel a queued or processing job.
pub async fn cancel_job(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Response, ApiError> {
    let jobs = require_jobs(&state)?;
    let job = load_job(&*jobs.store, &id).await?;
    if !matches!(job.status, JobStatus::Queued | JobStatus::Processing) {
        return Err(api_error(
            StatusCode::CONFLICT,
            "Job cannot be cancelled",
            "job_not_cancellable",
        ));
    }
    let _ = jobs
        .store
        .update(
            &id,
            Box::new(|j| {
                if matches!(j.status, JobStatus::Queued | JobStatus::Processing) {
                    j.status = JobStatus::Cancelled;
                }
            }),
        )
        .await;
    super::super::jobs::broadcast_event(&*jobs.store, &id, JobEvent::Cancelled).await;
    Ok(StatusCode::NO_CONTENT.into_response())
}

/// GET /v1/jobs/{id}/events — SSE stream of progress / done / failed / cancelled.
pub async fn job_events(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>>, ApiError> {
    let jobs = require_jobs(&state)?;
    let job = load_job(&*jobs.store, &id).await?;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<JobEvent>();
    if job.status.is_terminal() {
        let event = match job.status {
            JobStatus::Done => JobEvent::Done,
            JobStatus::Failed => JobEvent::Failed {
                error: job
                    .error
                    .clone()
                    .unwrap_or_else(|| "Transcription failed.".into()),
            },
            JobStatus::Cancelled => JobEvent::Cancelled,
            _ => unreachable!(),
        };
        let _ = tx.send(event);
    } else {
        let _ = jobs
            .store
            .update(
                &id,
                Box::new(move |j| {
                    j.subscribe(tx);
                }),
            )
            .await;
    }

    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
        .map(|event| Ok(Event::default().data(serde_json::to_string(&event).unwrap_or_default())));

    Ok(Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text(""),
    ))
}
