//! Bound-listener serve, drain, and SIGHUP reload loop.

use super::ServerConfig;
use super::{bootstrap, config, http, jobs, metrics, middleware, rate_limit, router};
use anyhow::{Context, Result};
use arc_swap::ArcSwap;
use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::get;
use std::net::SocketAddr;
use std::sync::Arc;

/// Start server with a full [`ServerConfig`] and optional programmatic
/// shutdown signal. This is the canonical entry point — the other `run_*`
/// helpers construct a default `ServerConfig` and dispatch here.
pub async fn run_with_config(
    engine: gigastt_core::inference::Engine,
    config: ServerConfig,
    shutdown: Option<tokio::sync::oneshot::Receiver<()>>,
) -> Result<()> {
    let addr: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .context("Invalid host:port")?;
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    run_with_config_listener(engine, config, shutdown, listener).await
}

/// Await the external shutdown source during the loading phase: the caller's
/// oneshot when present, otherwise Ctrl-C / SIGTERM. Mirrors the signal handling
/// in [`run_with_config_listener`] so a stuck first-run model download can still
/// be interrupted before the engine is ready.
async fn wait_for_shutdown(shutdown: Option<tokio::sync::oneshot::Receiver<()>>) {
    match shutdown {
        Some(rx) => {
            let _ = rx.await;
        }
        None => {
            #[cfg(unix)]
            {
                use tokio::signal::unix::{SignalKind, signal};
                match signal(SignalKind::terminate()) {
                    Ok(mut sigterm) => {
                        tokio::select! {
                            _ = tokio::signal::ctrl_c() => {}
                            _ = sigterm.recv() => {}
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to register SIGTERM handler during boot: {e}");
                        let _ = tokio::signal::ctrl_c().await;
                    }
                }
            }
            #[cfg(not(unix))]
            {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
}

/// Start the server with non-blocking first-run boot.
///
/// Binds the listener immediately and serves a minimal bootstrap responder
/// (`/health` → 200 with `model: "loading"`, `/ready` → 503 `initializing`)
/// while `load` builds the engine in the background — so health probes and
/// Docker `HEALTHCHECK` never see connection-refused during the first-run model
/// download + INT8 quantization (which can take minutes). Once the engine is
/// ready the *same* bound socket is handed to [`run_with_config_listener`] with
/// no rebind / no gap. A shutdown signal that arrives before the engine finishes
/// loading aborts cleanly without ever serving real traffic.
///
/// The `load` future is expected to wrap its heavy synchronous work (quantize,
/// ONNX session load) in `spawn_blocking` so the bootstrap responder stays
/// responsive while it runs.
pub async fn run_with_config_loading<Fut>(
    config: ServerConfig,
    shutdown: Option<tokio::sync::oneshot::Receiver<()>>,
    load: Fut,
) -> Result<()>
where
    Fut: std::future::Future<Output = Result<gigastt_core::inference::Engine>> + Send + 'static,
{
    run_with_config_loading_reloadable(config, shutdown, load, None).await
}

/// Like [`run_with_config_loading`], but also carries the [`http::EngineBuilder`]
/// recipe so the server that starts after the model loads exposes a working
/// `POST /v1/admin/reload`. The `load` future typically calls the *same*
/// builder once to produce the boot engine, so boot and reload share one recipe.
pub async fn run_with_config_loading_reloadable<Fut>(
    config: ServerConfig,
    shutdown: Option<tokio::sync::oneshot::Receiver<()>>,
    load: Fut,
    engine_builder: Option<http::EngineBuilder>,
) -> Result<()>
where
    Fut: std::future::Future<Output = Result<gigastt_core::inference::Engine>> + Send + 'static,
{
    run_with_config_loading_reloadable_and_job_context(
        config,
        shutdown,
        load,
        engine_builder,
        false,
    )
    .await
}

/// Like the reloadable loading entry point, with optional zero context limited
/// to asynchronous file jobs. The previous entry point remains default-off.
pub async fn run_with_config_loading_reloadable_and_job_context<Fut>(
    config: ServerConfig,
    shutdown: Option<tokio::sync::oneshot::Receiver<()>>,
    load: Fut,
    engine_builder: Option<http::EngineBuilder>,
    jobs_file_zero_context: bool,
) -> Result<()>
where
    Fut: std::future::Future<Output = Result<gigastt_core::inference::Engine>> + Send + 'static,
{
    anyhow::ensure!(
        !jobs_file_zero_context || config.limits.jobs_enabled,
        "--jobs-file-zero-context requires --enable-jobs"
    );
    let addr: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .context("Invalid host:port")?;
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("gigastt bootstrapping on http://{addr} — loading model, /health is up");

    // Normalize the external shutdown source into a cancellation token (observed
    // during boot) plus a fresh oneshot forwarded to the real server, so a
    // signal works in both phases without registering the OS handler twice.
    let shutdown_token = tokio_util::sync::CancellationToken::new();
    let (real_tx, real_rx) = tokio::sync::oneshot::channel();
    {
        let token = shutdown_token.clone();
        tokio::spawn(async move {
            wait_for_shutdown(shutdown).await;
            token.cancel();
            let _ = real_tx.send(());
        });
    }

    // Load the engine concurrently with the bootstrap responder.
    let mut load_task = tokio::spawn(load);

    // The bootstrap accept loop owns the listener and returns it on cancellation,
    // so the real server can reuse the same bound socket without a rebind window.
    let boot_cancel = tokio_util::sync::CancellationToken::new();
    let boot_task = {
        let boot_cancel = boot_cancel.clone();
        tokio::spawn(async move {
            let version = env!("CARGO_PKG_VERSION");
            loop {
                tokio::select! {
                    biased;
                    _ = boot_cancel.cancelled() => break,
                    accepted = listener.accept() => match accepted {
                        Ok((stream, _peer)) => {
                            tokio::spawn(bootstrap::handle_bootstrap_conn(stream, version));
                        }
                        Err(e) => tracing::warn!("bootstrap accept error: {e}"),
                    },
                }
            }
            listener
        })
    };

    enum Outcome {
        Shutdown,
        // Boxed: the `Engine` payload is large, so keeping it inline would make
        // the `Shutdown` variant pay for it too (clippy::large_enum_variant).
        Loaded(
            Box<
                std::result::Result<
                    Result<gigastt_core::inference::Engine>,
                    tokio::task::JoinError,
                >,
            >,
        ),
    }

    let outcome = tokio::select! {
        biased;
        _ = shutdown_token.cancelled() => Outcome::Shutdown,
        res = &mut load_task => Outcome::Loaded(Box::new(res)),
    };

    // Stop accepting bootstrap connections and reclaim the bound listener.
    boot_cancel.cancel();
    let listener = boot_task.await.context("bootstrap task panicked")?;

    match outcome {
        Outcome::Shutdown => {
            load_task.abort();
            tracing::info!("Shutdown requested during model load — exiting before serving");
            Ok(())
        }
        Outcome::Loaded(res) => {
            let engine = (*res).context("engine load task panicked")??;
            tracing::info!("Model ready — starting full server");
            run_with_config_listener_and_job_context(
                engine,
                config,
                Some(real_rx),
                listener,
                engine_builder,
                jobs_file_zero_context,
            )
            .await
        }
    }
}

/// Start server with a full [`ServerConfig`], an optional shutdown signal,
/// and an already-bound TCP listener. Used by tests to eliminate the TOCTOU
/// race between `free_port()` and server startup.
pub async fn run_with_config_listener(
    engine: gigastt_core::inference::Engine,
    config: ServerConfig,
    shutdown: Option<tokio::sync::oneshot::Receiver<()>>,
    listener: tokio::net::TcpListener,
) -> Result<()> {
    run_with_config_listener_reloadable(engine, config, shutdown, listener, None).await
}

/// Like [`run_with_config_listener`], but also accepts the [`http::EngineBuilder`]
/// recipe that `POST /v1/admin/reload` uses to rebuild the engine in place.
/// `None` disables the reload endpoint (`reload_unsupported`) — the thin
/// `run` / `run_with_shutdown` and test entry points take that path.
pub async fn run_with_config_listener_reloadable(
    engine: gigastt_core::inference::Engine,
    config: ServerConfig,
    shutdown: Option<tokio::sync::oneshot::Receiver<()>>,
    listener: tokio::net::TcpListener,
    engine_builder: Option<http::EngineBuilder>,
) -> Result<()> {
    run_with_config_listener_and_job_context(
        engine,
        config,
        shutdown,
        listener,
        engine_builder,
        false,
    )
    .await
}

fn checked_job_context_builder(
    builder: Option<http::EngineBuilder>,
    enabled: bool,
) -> Option<http::EngineBuilder> {
    if !enabled {
        return builder;
    }
    builder.map(|builder| {
        let checked: http::EngineBuilder = Arc::new(move || {
            let engine = builder()?;
            engine
                .validate_file_zero_context()
                .context("--jobs-file-zero-context")?;
            Ok(engine)
        });
        checked
    })
}

async fn run_with_config_listener_and_job_context(
    engine: gigastt_core::inference::Engine,
    mut config: ServerConfig,
    shutdown: Option<tokio::sync::oneshot::Receiver<()>>,
    listener: tokio::net::TcpListener,
    engine_builder: Option<http::EngineBuilder>,
    jobs_file_zero_context: bool,
) -> Result<()> {
    // Validate the actual resolved engine (including model auto-detection and
    // execution-provider fallback), and revalidate before any reload can swap.
    if jobs_file_zero_context {
        engine
            .validate_file_zero_context()
            .context("--jobs-file-zero-context")?;
        tracing::info!("Experimental file zero context enabled for asynchronous jobs");
    }
    let engine_builder = checked_job_context_builder(engine_builder, jobs_file_zero_context);
    if config.limits.pool_checkout_timeout_secs == 0 {
        tracing::warn!("pool_checkout_timeout_secs=0 would make the pool unusable; clamping to 1");
        config.limits.pool_checkout_timeout_secs = 1;
    }

    // Warm every pooled session triplet before accepting traffic so
    // the first real request doesn't pay the EP compile / first-allocation
    // cost. Inference is blocking work — keep it off the async runtime. A
    // warmup failure is logged but not fatal: under `coreml` the engine has
    // already fallen back to the CPU EP inside `Engine::load`.
    let engine = tokio::task::spawn_blocking(move || {
        let started = std::time::Instant::now();
        match engine.warmup() {
            Ok(()) => tracing::info!(
                elapsed_ms = started.elapsed().as_millis() as u64,
                "Engine warmup complete"
            ),
            Err(e) => tracing::warn!("Engine warmup failed (serving anyway): {e:#}"),
        }
        engine
    })
    .await
    .context("engine warmup task panicked")?;

    let addr: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .context("Invalid host:port")?;

    // Stand up our in-tree metrics registry when the operator asked for it.
    // Unlike the old `PrometheusBuilder::install_recorder()` path this is
    // per-`run_with_config` rather than process-global — restarting the
    // server in tests cannot collide with itself, so we do not need the
    // "already installed" warning fallback the old stack needed.
    let metrics_registry = if config.metrics_enabled {
        let reg = std::sync::Arc::new(metrics::register_server_metrics());
        tracing::info!("Prometheus /metrics endpoint enabled");
        Some(reg)
    } else {
        None
    };

    // Sanity check: an `idle_timeout` larger than `max_session_secs`
    // is usually a misconfiguration — the cap fires before the idle timeout
    // can ever apply, which is surprising. Warn without rejecting so
    // operators who intentionally want both can keep the behaviour.
    if config.limits.max_session_secs != 0
        && config.limits.max_session_secs < config.limits.idle_timeout_secs
    {
        tracing::warn!(
            max_session_secs = config.limits.max_session_secs,
            idle_timeout_secs = config.limits.idle_timeout_secs,
            "max_session_secs < idle_timeout_secs — sessions will be capped before \
             the idle timer can fire; this is probably not what you want"
        );
    }

    // Shutdown lane: `shutdown_root` is cancelled when the caller's
    // oneshot fires (or Ctrl-C is received). Every WS / SSE handler gets a
    // clone so a SIGTERM propagates without racing `axum::serve`'s own
    // graceful shutdown.
    let shutdown_root = tokio_util::sync::CancellationToken::new();
    let tracker = tokio_util::task::TaskTracker::new();

    let engine_swap = Arc::new(ArcSwap::from_pointee(engine));
    let limits_swap = Arc::new(ArcSwap::from_pointee(config.limits.clone()));

    // Stand up the asynchronous job queue when the operator enabled it.
    // We build it before `AppState` so the handlers can hold a clone.
    let jobs_state = if config.limits.jobs_enabled {
        let store: Arc<dyn jobs::JobStore> =
            Arc::new(jobs::InMemoryJobStore::new(config.limits.clone()));
        let concurrency = config.batch_pool_size.max(1);
        let max_retries = config.limits.jobs_retry;
        let queue = jobs::JobQueue::new(
            store.clone(),
            concurrency,
            max_retries,
            shutdown_root.clone(),
        );
        let executor = jobs::RealJobExecutor::new(
            engine_swap.clone(),
            limits_swap.clone(),
            shutdown_root.clone(),
        )
        .with_file_zero_context(jobs_file_zero_context);
        // Spawn onto the same TaskTracker as WS / SSE so `--shutdown-drain-secs`
        // waits for in-flight jobs (not only interactive sessions).
        queue.spawn(executor, &tracker);
        tracing::info!(
            concurrency,
            max_retries,
            "asynchronous /v1/jobs API enabled"
        );
        Some(http::JobServerState { store, queue })
    } else {
        None
    };

    let state = Arc::new(http::AppState {
        engine: engine_swap,
        engine_builder: engine_builder.clone(),
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        limits: limits_swap,
        metrics_registry: metrics_registry.clone(),
        shutdown: shutdown_root.clone(),
        tracker: tracker.clone(),
        jobs: jobs_state.clone(),
    });

    let rate_limiter_swap = if config.limits.rate_limit_per_minute > 0 {
        Some(Arc::new(ArcSwap::from(Arc::new(
            rate_limit::RateLimiter::new(
                config.limits.rate_limit_per_minute,
                config.limits.rate_limit_burst,
            ),
        ))))
    } else {
        None
    };

    #[cfg(unix)]
    {
        let reload_state = state.clone();
        let reload_path = config.config_path.clone();
        let reload_shutdown = shutdown_root.clone();
        let reload_limiter = rate_limiter_swap.clone();
        tokio::spawn(async move {
            use tokio::signal::unix::{SignalKind, signal};
            let mut sig = match signal(SignalKind::hangup()) {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!("Failed to register SIGHUP handler: {e}");
                    return;
                }
            };
            loop {
                tokio::select! {
                    biased;
                    _ = reload_shutdown.cancelled() => break,
                    _ = sig.recv() => {
                        let Some(ref path) = reload_path else {
                            tracing::info!("No config file specified, ignoring SIGHUP");
                            continue;
                        };
                        match config::load_config_file(path) {
                            Ok(new_limits) => {
                                if jobs_file_zero_context && !new_limits.jobs_enabled {
                                    tracing::error!("Cannot disable jobs while jobs file zero context is enabled; keeping previous runtime limits");
                                    continue;
                                }
                                let old = reload_state.limits.load();
                                tracing::info!(
                                    "RuntimeLimits reloaded from {}: idle_timeout_secs {} → {}, rate_limit_per_minute {} → {}",
                                    path.display(),
                                    old.idle_timeout_secs, new_limits.idle_timeout_secs,
                                    old.rate_limit_per_minute, new_limits.rate_limit_per_minute,
                                );
                                if let Some(ref rl) = reload_limiter
                                    && (old.rate_limit_per_minute
                                        != new_limits.rate_limit_per_minute
                                        || old.rate_limit_burst != new_limits.rate_limit_burst)
                                    && new_limits.rate_limit_per_minute > 0
                                {
                                    rl.store(Arc::new(rate_limit::RateLimiter::new(
                                        new_limits.rate_limit_per_minute,
                                        new_limits.rate_limit_burst,
                                    )));
                                    tracing::info!(
                                        "Rate limiter recreated: rpm {} → {}, burst {} → {}",
                                        old.rate_limit_per_minute,
                                        new_limits.rate_limit_per_minute,
                                        old.rate_limit_burst,
                                        new_limits.rate_limit_burst,
                                    );
                                }
                                reload_state.limits.store(Arc::new(new_limits));
                            }
                            Err(e) => {
                                tracing::error!("Failed to reload config on SIGHUP: {e:#}");
                            }
                        }
                    }
                }
            }
        });
    }

    let policy = Arc::new(config.origin_policy.clone());

    let origin_layer = {
        let policy = policy.clone();
        axum::middleware::from_fn(move |req, next| {
            let policy = policy.clone();
            async move { middleware::origin_middleware(policy, req, next).await }
        })
    };

    // Protected sub-router: /v1/* and the /v1/ws WebSocket — all subject to
    // the origin allowlist and (when enabled) the per-IP rate limiter.
    // `/metrics` is intentionally NOT here: it lives on its own loopback
    // listener (see below) so telemetry is never exposed to allowlisted
    // browser origins nor throttled by the per-IP limiter.
    let upload_capacity = state.engine.load().pool_for_batch().total();
    let admission = super::upload::UploadAdmission::new(
        upload_capacity,
        config::pool_retry_after_secs(&config.limits),
    );
    tracing::info!(
        upload_capacity,
        body_limit_bytes = config.limits.body_limit_bytes,
        "HTTP upload admission enabled (boot-time limits)"
    );
    let protected = router::protected_v1_router(config.limits.jobs_enabled, admission);

    let protected = protected
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            middleware::http_metrics_middleware,
        ))
        .with_state(state.clone());

    let protected = if let Some(ref limiter_swap) = rate_limiter_swap {
        let interval_ms = limiter_swap.load().interval_ms();

        let evict_limiter = limiter_swap.clone();
        let evict_cancel = shutdown_root.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(std::time::Duration::from_secs(60));
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            ticker.tick().await;
            loop {
                tokio::select! {
                    biased;
                    _ = evict_cancel.cancelled() => break,
                    _ = ticker.tick() => {
                        evict_limiter.load().evict_stale(std::time::Duration::from_secs(300));
                    }
                }
            }
        });

        tracing::info!(
            rpm = config.limits.rate_limit_per_minute,
            interval_ms,
            burst = config.limits.rate_limit_burst,
            "per-IP rate limiting enabled"
        );
        let layer_limiter = limiter_swap.clone();
        let layer_trust_proxy = config.trust_proxy;
        let layer_metrics = metrics_registry.clone();
        protected.layer(axum::middleware::from_fn(move |req, next| {
            let limiter = layer_limiter.load_full();
            let metrics = layer_metrics.clone();
            async move {
                rate_limit::rate_limit_middleware(limiter, layer_trust_proxy, metrics, req, next)
                    .await
            }
        }))
    } else {
        protected
    };

    // Clone the *swap handle* (not a snapshot) before `state` is consumed by
    // `with_state` so the shutdown closure can close the pools of whichever
    // engine is live at shutdown time. Capturing a pre-swap snapshot here would
    // close the boot engine's pools even after a hot-reload swapped in a new
    // one, stranding the new pool's waiters (they'd never get `PoolError::Closed`).
    let shutdown_engine = state.engine.clone();

    // Clone the state for the separate metrics listener before `state` is moved
    // into the primary router's `with_state`. `None` when metrics are disabled.
    let metrics_state = config.metrics_enabled.then(|| state.clone());

    let request_id_layer = axum::middleware::from_fn(middleware::request_id_middleware);

    let app = Router::new()
        .route("/health", get(http::health))
        .route("/ready", get(http::readiness))
        .merge(protected)
        .layer(DefaultBodyLimit::max(config.limits.body_limit_bytes))
        .layer(origin_layer)
        .layer(request_id_layer)
        .with_state(state);

    tracing::info!("gigastt server listening on http://{addr}");
    tracing::info!("  WebSocket: ws://{addr}/v1/ws");
    tracing::info!(
        "  REST API:  http://{addr}/health, /ready, /v1/transcribe, /v1/transcribe/stream, /v1/audio/transcriptions"
    );
    if config.origin_policy.allow_any {
        tracing::warn!(
            "CORS allow-any is ON: any cross-origin page can call this server. \
             Only use with trusted callers."
        );
    } else if !config.origin_policy.allowed_origins.is_empty() {
        tracing::info!(
            "CORS allowlist (in addition to loopback): {:?}",
            config.origin_policy.allowed_origins
        );
    }

    // Prometheus `/metrics` on its own loopback listener (Prometheus
    // convention): off the primary CORS allowlist + rate limiter, and exposed
    // deliberately by the operator. Shuts down with the same cancellation
    // token as the main server.
    let metrics_server = if let Some(metrics_state) = metrics_state {
        let metrics_app = Router::new()
            .route("/metrics", get(http::metrics))
            .with_state(metrics_state);
        let metrics_listener = tokio::net::TcpListener::bind(config.metrics_listen)
            .await
            .with_context(|| {
                format!(
                    "Failed to bind metrics listener on {}",
                    config.metrics_listen
                )
            })?;
        tracing::info!("  Metrics:   http://{}/metrics", config.metrics_listen);
        let metrics_cancel = shutdown_root.clone();
        Some(tokio::spawn(async move {
            let serve = axum::serve(metrics_listener, metrics_app)
                .with_graceful_shutdown(async move { metrics_cancel.cancelled().await });
            if let Err(e) = serve.await {
                tracing::warn!("metrics listener error: {e}");
            }
        }))
    } else {
        None
    };

    let shutdown_drain_secs = config.limits.shutdown_drain_secs.max(1);

    let shutdown_fut = {
        let shutdown_root = shutdown_root.clone();
        let jobs_state = jobs_state.clone();
        async move {
            match shutdown {
                Some(rx) => {
                    // A dropped sender also triggers shutdown (the usual
                    // programmatic path), but a RecvError is worth surfacing
                    // rather than swallowing silently.
                    if let Err(e) = rx.await {
                        tracing::warn!("Shutdown sender dropped before firing: {e}");
                    }
                }
                None => {
                    #[cfg(unix)]
                    {
                        use tokio::signal::unix::{SignalKind, signal};
                        let mut sigterm = match signal(SignalKind::terminate()) {
                            Ok(s) => s,
                            Err(e) => {
                                tracing::error!("Failed to register SIGTERM handler: {e}");
                                tokio::signal::ctrl_c().await.ok();
                                return;
                            }
                        };
                        tokio::select! {
                            _ = tokio::signal::ctrl_c() => {},
                            _ = sigterm.recv() => {
                                tracing::info!("Received SIGTERM, starting graceful shutdown");
                            }
                        }
                    }
                    #[cfg(not(unix))]
                    {
                        tokio::signal::ctrl_c().await.ok();
                    }
                }
            }
            tracing::info!("Shutting down server");
            // Cancel the per-handler token FIRST so WS / SSE tasks start
            // draining while axum is still completing the in-flight HTTP
            // futures.
            shutdown_root.cancel();
            // Mark every queued job as cancelled so workers don't start new work
            // during the drain window. In-flight jobs are allowed to finish within
            // `shutdown_drain_secs` (their triplet is returned when they complete).
            if let Some(ref jobs) = jobs_state {
                jobs.queue.cancel_all_queued().await;
            }
            // Wake every waiter still blocked on `pool.checkout()` with
            // PoolError::Closed so they fall through to a 503 / `pool_closed`
            // response instead of being stranded for the full checkout timeout.
            // Load the engine that is live *now* (a hot-reload may have swapped
            // it since boot) so we close the pool waiters are actually parked on.
            // Idempotent — safe even if the pool was already closed.
            shutdown_engine.load_full().close_pools();
        }
    };

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_fut)
    .await?;

    // Drain window: give WS / SSE / REST / job-worker tasks
    // `shutdown_drain_secs` to finish cleanly. TaskTracker::wait() returns
    // when every tracked future completes; we close() first so no new
    // futures can be added after shutdown.
    tracker.close();
    match tokio::time::timeout(
        std::time::Duration::from_secs(shutdown_drain_secs),
        tracker.wait(),
    )
    .await
    {
        Ok(()) => tracing::info!("Drain complete: all tracked WS/SSE/job tasks finished"),
        Err(_) => tracing::warn!(
            drain_secs = shutdown_drain_secs,
            pending = tracker.len(),
            "Drain window expired with tracked tasks still running — forcing exit"
        ),
    }

    // The metrics listener drains on the same cancellation token; wait for it
    // to finish so the process doesn't exit with the socket still bound.
    if let Some(handle) = metrics_server {
        let _ = handle.await;
    }

    Ok(())
}

#[cfg(test)]
mod job_context_tests {
    use super::*;

    #[tokio::test]
    async fn test_jobs_file_zero_context_startup_requires_jobs() {
        let result = run_with_config_loading_reloadable_and_job_context(
            ServerConfig::local(0),
            None,
            async { panic!("disabled jobs must reject before model load") },
            None,
            true,
        )
        .await;
        assert!(format!("{:#}", result.unwrap_err()).contains("requires --enable-jobs"));
    }

    #[test]
    fn test_jobs_file_zero_context_reload_checks_actual_engine() {
        use gigastt_core::test_support;
        let tmp = tempfile::tempdir().unwrap();
        test_support::write_rnnt_layout(tmp.path()).unwrap();
        let dir = tmp.path().to_owned();
        let incompatible: http::EngineBuilder = Arc::new(move || {
            Ok(test_support::load_rnnt_engine(&dir, 1)?.with_file_window_concurrency(2))
        });
        assert!(checked_job_context_builder(Some(incompatible.clone()), false).unwrap()().is_ok());
        assert!(checked_job_context_builder(Some(incompatible), true).unwrap()().is_err());
        let dir = tmp.path().to_owned();
        let compatible: http::EngineBuilder =
            Arc::new(move || Ok(test_support::load_rnnt_engine(&dir, 1)?));
        assert!(checked_job_context_builder(Some(compatible), true).unwrap()().is_ok());
    }
}
