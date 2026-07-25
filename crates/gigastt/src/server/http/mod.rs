//! HTTP handlers for REST API endpoints.

mod admin;
mod error;
mod export;
mod health;
mod jobs_api;
mod openai_api;
mod state;
mod stream;
mod transcribe;

#[cfg(test)]
mod tests;

// --- state ---
pub use state::{AppState, EngineBuilder, JobServerState};

// --- export / params ---
pub use export::ExportParams;
#[cfg(test)]
pub(crate) use export::parse_hotwords_query;
pub(crate) use export::{hotwords_from_export_params, overrides_from_export_params};

// --- health / models / metrics ---
pub(crate) use health::sample_batch_pool_gauges;
pub use health::{
    HealthResponse, ModelInfo, ReadinessResponse, health, metrics, models, readiness,
};

// --- transcribe ---
pub use transcribe::{TranscribeResponse, transcribe};

// --- stream ---
pub use stream::transcribe_stream;

// --- openai ---
pub use openai_api::openai_transcriptions;

// --- admin ---
pub use admin::reload;

// --- jobs ---
pub use jobs_api::{
    JobSubmitResponse, cancel_job, get_job, get_job_result, job_events, submit_job,
};
