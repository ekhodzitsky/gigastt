//! HTTP + WebSocket server that accepts audio and streams transcripts.
//!
//! Single port serves both REST API (health, transcribe, SSE) and WebSocket.

mod bootstrap;
pub mod config;
mod file_transcribe;
pub mod http;
pub mod jobs;
pub mod metrics;
pub(crate) mod middleware;
pub(crate) mod openai;
pub mod rate_limit;
mod router;
mod upload;
mod ws;

pub use config::{OriginPolicy, RuntimeLimits, ServerConfig};
pub use http::EngineBuilder;
#[cfg(all(
    feature = "v4-mkl",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
))]
pub use http::v4::{run_v4_files, run_v4_files_with_streaming, v4_result};
#[cfg(all(
    feature = "v4-mkl",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
))]
pub use ws::native_v4::NativeV4Streaming;

use anyhow::Result;

/// Serialize a server message to JSON with a safe fallback on error.
pub(crate) fn json_text(msg: &impl serde::Serialize) -> String {
    serde_json::to_string(msg).unwrap_or_else(|e| {
        tracing::error!("Failed to serialize server message: {e}");
        r#"{"type":"error","message":"Internal serialization error","code":"internal"}"#.into()
    })
}

/// Start the HTTP + WebSocket STT server on the given host and port.
///
/// Serves REST API endpoints and WebSocket on a single port:
/// - `GET /health` — health check
/// - `POST /v1/transcribe` — file transcription
/// - `POST /v1/transcribe/stream` — SSE streaming transcription
/// - `POST /v1/audio/transcriptions` — OpenAI-compatible file transcription
/// - `GET /v1/ws` — WebSocket streaming protocol
///
/// Runs until `Ctrl-C` is received.
pub async fn run(engine: gigastt_core::inference::Engine, port: u16, host: &str) -> Result<()> {
    run_with_shutdown(engine, port, host, None).await
}

/// Start server with an optional programmatic shutdown signal.
///
/// When `shutdown` is `Some`, the server stops when the sender fires (or is dropped).
/// When `None`, the server stops on Ctrl-C. Used by tests for clean teardown.
pub async fn run_with_shutdown(
    engine: gigastt_core::inference::Engine,
    port: u16,
    host: &str,
    shutdown: Option<tokio::sync::oneshot::Receiver<()>>,
) -> Result<()> {
    let config = ServerConfig {
        port,
        host: host.to_string(),
        origin_policy: OriginPolicy::loopback_only(),
        limits: RuntimeLimits::default(),
        metrics_enabled: false,
        metrics_listen: config::default_metrics_listen(),
        trust_proxy: false,
        config_path: None,
        batch_pool_size: 0,
    };
    run_with_config(engine, config, shutdown).await
}

mod listen;
pub use listen::{
    run_with_config, run_with_config_listener, run_with_config_listener_reloadable,
    run_with_config_loading, run_with_config_loading_reloadable,
    run_with_config_loading_reloadable_and_job_context,
};

#[cfg(test)]
mod tests;
