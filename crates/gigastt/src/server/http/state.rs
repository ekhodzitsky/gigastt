//! Shared application state for HTTP / WebSocket handlers.

use arc_swap::ArcSwap;
use std::sync::Arc;

use super::super::config::RuntimeLimits;
use super::super::jobs::{JobQueue, JobStore};
use super::super::metrics::MetricsRegistry;
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
    /// In-memory job store and queue. `Some` only when `--enable-jobs` is set;
    /// handlers for `/v1/jobs` are registered conditionally and expect this to
    /// be populated, but the shared state is `Option` so non-job builds compile.
    pub jobs: Option<JobServerState>,
}

/// State shared by the `/v1/jobs` handlers. Kept behind `Arc` so the executor,
/// queue, and handlers all reference the same store.
#[derive(Clone)]
pub struct JobServerState {
    pub store: Arc<dyn JobStore>,
    pub queue: Arc<JobQueue>,
}

/// Recipe that rebuilds a fully-configured [`Engine`] from the operator's boot
/// options. Stored in [`AppState`] so `POST /v1/admin/reload` can produce a
/// fresh engine that re-applies the punctuation / ITN / VAD / hotword chain —
/// a bare `Engine::load_*` starts with all of those set to `None`.
pub type EngineBuilder = Arc<dyn Fn() -> anyhow::Result<Engine> + Send + Sync>;
