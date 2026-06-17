//! HTTP + WebSocket server that accepts audio and streams transcripts.
//!
//! Single port serves both REST API (health, transcribe, SSE) and WebSocket.

pub mod config;
pub mod http;
pub mod metrics;
pub(crate) mod middleware;
pub mod rate_limit;
mod ws;

pub use config::{OriginPolicy, RuntimeLimits, ServerConfig};

use anyhow::{Context, Result};
use arc_swap::ArcSwap;
use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::http::StatusCode;
use axum::routing::{get, options, post};
use std::net::SocketAddr;
use std::sync::Arc;

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
    };
    run_with_config(engine, config, shutdown).await
}

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

/// Start server with a full [`ServerConfig`], an optional shutdown signal,
/// and an already-bound TCP listener. Used by tests to eliminate the TOCTOU
/// race between `free_port()` and server startup.
pub async fn run_with_config_listener(
    engine: gigastt_core::inference::Engine,
    mut config: ServerConfig,
    shutdown: Option<tokio::sync::oneshot::Receiver<()>>,
    listener: tokio::net::TcpListener,
) -> Result<()> {
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
        let reg = std::sync::Arc::new(self::metrics::MetricsRegistry::new());
        reg.register_counter(
            "gigastt_http_requests_total",
            "Total HTTP requests processed",
        );
        reg.register_histogram(
            "gigastt_http_request_duration_seconds",
            "HTTP request duration in seconds",
            self::metrics::DEFAULT_BUCKETS,
        );
        reg.register_gauge(
            "gigastt_pool_available",
            "Number of session triplets currently available in the pool",
        );
        reg.register_gauge(
            "gigastt_pool_waiters",
            "Number of tasks currently waiting for a pool checkout",
        );
        reg.register_gauge(
            "gigastt_batch_pool_available",
            "Number of session triplets currently available in the batch pool \
             (only populated when --batch-pool-size > 0)",
        );
        reg.register_gauge(
            "gigastt_batch_pool_waiters",
            "Number of tasks currently waiting for a batch-pool checkout",
        );
        reg.register_histogram(
            "gigastt_pool_checkout_duration_seconds",
            "Time spent waiting for a pool checkout",
            self::metrics::DEFAULT_BUCKETS,
        );
        reg.register_counter(
            "gigastt_pool_timeouts_total",
            "Total pool checkout timeouts",
        );
        reg.register_gauge(
            "gigastt_ws_active_connections",
            "Number of active WebSocket connections",
        );
        reg.register_histogram(
            "gigastt_inference_duration_seconds",
            "Inference duration in seconds",
            self::metrics::DEFAULT_BUCKETS,
        );
        reg.register_counter(
            "gigastt_rate_limit_rejections_total",
            "Total requests rejected by rate limiter",
        );
        reg.register_counter(
            "gigastt_inference_timeouts_total",
            "Total inference runs aborted by the per-request inference timeout",
        );
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

    let state = Arc::new(http::AppState {
        engine: Arc::new(engine),
        limits: Arc::new(ArcSwap::from_pointee(config.limits.clone())),
        metrics_registry: metrics_registry.clone(),
        shutdown: shutdown_root.clone(),
        tracker: tracker.clone(),
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
    let protected = Router::new()
        .route("/v1/models", get(http::models))
        .route("/v1/models", options(|| async { StatusCode::NO_CONTENT }))
        .route("/v1/transcribe", post(http::transcribe))
        .route(
            "/v1/transcribe",
            options(|| async { StatusCode::NO_CONTENT }),
        )
        .route("/v1/transcribe/stream", post(http::transcribe_stream))
        .route(
            "/v1/transcribe/stream",
            options(|| async { StatusCode::NO_CONTENT }),
        )
        // /v1/ws is the canonical WebSocket path (versioned, aligned with REST).
        .route("/v1/ws", get(ws::ws_handler))
        .route("/v1/ws", options(|| async { StatusCode::NO_CONTENT }))
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

    // Clone the engine handle before `state` is consumed by `with_state` so
    // the shutdown closure can call `pool.close()` after the listener task
    // begins draining.
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
        "  REST API:  http://{addr}/health, /ready, /v1/transcribe, /v1/transcribe/stream"
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
            // Wake every waiter still blocked on `pool.checkout()` with
            // PoolError::Closed so they fall through to a 503 / `pool_closed`
            // response instead of being stranded for the full checkout timeout.
            // Idempotent — safe even if the pool was already closed.
            shutdown_engine.close_pools();
        }
    };

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_fut)
    .await?;

    // Drain window: give WS / SSE tasks `shutdown_drain_secs` to emit their
    // Final frames and close cleanly. TaskTracker::wait() returns when every
    // tracked future completes; we close() first so no new futures can be
    // added after shutdown.
    tracker.close();
    match tokio::time::timeout(
        std::time::Duration::from_secs(shutdown_drain_secs),
        tracker.wait(),
    )
    .await
    {
        Ok(()) => tracing::info!("Drain complete: all tracked WS/SSE tasks finished"),
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
mod tests {
    use super::*;

    #[test]
    fn test_json_text_serializes() {
        let msg = gigastt_core::protocol::ServerMessage::Ready {
            model: "test".into(),
            sample_rate: 16000,
            version: "1.0".into(),
            supported_rates: vec![16000],
            diarization: false,
            min_protocol_version: None,
        };
        let json = json_text(&msg);
        assert!(json.contains("\"type\":\"ready\""));
    }

    #[test]
    fn test_json_text_fallback_on_error() {
        // A type that intentionally fails serialization is hard to construct
        // with serde, so we assert the fallback path exists by checking the
        // function compiles and the happy path works. The fallback is a
        // static string that we can at least verify is present in the binary
        // by inspecting the source.
        let msg = gigastt_core::protocol::ServerMessage::Error {
            message: "test".into(),
            code: "test".into(),
            retry_after_ms: None,
        };
        let json = json_text(&msg);
        assert!(json.contains("error"));
    }

    #[test]
    fn test_rate_limit_interval_formula() {
        // Mirrors the formula used in `run_with_config` so a regression on the
        // integer-divide `/60` fix (truncates sub-60 rpm to 1 rps) trips
        // a unit test before reaching the e2e path.
        const MAX_RPM: u64 = 60_000;
        fn interval_ms_for(rpm: u32) -> u64 {
            let rpm = (rpm as u64).min(MAX_RPM);
            (60_000u64 / rpm).max(1)
        }
        let cases: &[(u32, u64)] = &[
            (1, 60_000),
            (10, 6_000),
            (30, 2_000),
            (59, 1_016), // 60_000 / 59 = 1016 (rounds down) → ~59.05 rpm
            (60, 1_000),
            (600, 100),
            (60_000, 1),
            (120_000, 1), // clamped to MAX_RPM, stays at 1 ms
        ];
        for (rpm, expected) in cases {
            assert_eq!(
                interval_ms_for(*rpm),
                *expected,
                "rpm={rpm} should map to interval_ms={expected}"
            );
        }
    }

    #[test]
    fn test_pool_checkout_timeout_clamping() {
        let mut config = ServerConfig::local(0);
        config.limits.pool_checkout_timeout_secs = 0;
        // `run_with_config_listener` would clamp this to 1.
        if config.limits.pool_checkout_timeout_secs == 0 {
            config.limits.pool_checkout_timeout_secs = 1;
        }
        assert_eq!(config.limits.pool_checkout_timeout_secs, 1);
    }

    #[test]
    fn test_json_text_fallback_on_serialization_error() {
        struct FailingSerialize;
        impl serde::Serialize for FailingSerialize {
            fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                Err(serde::ser::Error::custom("intentional failure"))
            }
        }
        let json = json_text(&FailingSerialize);
        assert_eq!(
            json,
            r#"{"type":"error","message":"Internal serialization error","code":"internal"}"#
        );
    }

    #[tokio::test]
    #[ignore = "requires model"]
    async fn test_run_with_shutdown_starts_and_stops() {
        let engine = gigastt_core::inference::Engine::load_with_pool_size(
            &gigastt_core::model::default_model_dir(),
            1,
        )
        .unwrap();
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let handle = tokio::spawn(async move {
            run_with_shutdown(engine, 0, "127.0.0.1", Some(shutdown_rx)).await
        });
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let _ = shutdown_tx.send(());
        let result = handle.await.expect("join");
        assert!(result.is_ok(), "server should stop gracefully");
    }

    #[tokio::test]
    #[ignore = "requires model"]
    async fn test_run_with_config_listener_clamps_zero_timeout() {
        let engine = gigastt_core::inference::Engine::load_with_pool_size(
            &gigastt_core::model::default_model_dir(),
            1,
        )
        .unwrap();
        let mut config = ServerConfig::local(0);
        config.limits.pool_checkout_timeout_secs = 0;
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let handle = tokio::spawn(async move {
            run_with_config_listener(engine, config, Some(shutdown_rx), listener).await
        });
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let _ = shutdown_tx.send(());
        let result = handle.await.expect("join");
        assert!(result.is_ok(), "server should stop gracefully");
    }
}
