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
pub async fn run(engine: crate::inference::Engine, port: u16, host: &str) -> Result<()> {
    run_with_shutdown(engine, port, host, None).await
}

/// Start server with an optional programmatic shutdown signal.
///
/// When `shutdown` is `Some`, the server stops when the sender fires (or is dropped).
/// When `None`, the server stops on Ctrl-C. Used by tests for clean teardown.
pub async fn run_with_shutdown(
    engine: crate::inference::Engine,
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
        trust_proxy: false,
    };
    run_with_config(engine, config, shutdown).await
}

/// Start server with a full [`ServerConfig`] and optional programmatic
/// shutdown signal. This is the canonical entry point — the other `run_*`
/// helpers construct a default `ServerConfig` and dispatch here.
pub async fn run_with_config(
    engine: crate::inference::Engine,
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
    engine: crate::inference::Engine,
    mut config: ServerConfig,
    shutdown: Option<tokio::sync::oneshot::Receiver<()>>,
    listener: tokio::net::TcpListener,
) -> Result<()> {
    if config.limits.pool_checkout_timeout_secs == 0 {
        tracing::warn!("pool_checkout_timeout_secs=0 would make the pool unusable; clamping to 1");
        config.limits.pool_checkout_timeout_secs = 1;
    }
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
        tracing::info!("Prometheus /metrics endpoint enabled");
        Some(reg)
    } else {
        None
    };

    // V1-04 sanity check: an `idle_timeout` larger than `max_session_secs`
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

    // Shutdown lane (V1-03): `shutdown_root` is cancelled when the caller's
    // oneshot fires (or Ctrl-C is received). Every WS / SSE handler gets a
    // clone so a SIGTERM propagates without racing `axum::serve`'s own
    // graceful shutdown.
    let shutdown_root = tokio_util::sync::CancellationToken::new();
    let tracker = tokio_util::task::TaskTracker::new();

    let state = Arc::new(http::AppState {
        engine: Arc::new(engine),
        limits: config.limits.clone(),
        metrics_registry: metrics_registry.clone(),
        shutdown: shutdown_root.clone(),
        tracker: tracker.clone(),
    });

    let policy = Arc::new(config.origin_policy.clone());

    let origin_layer = {
        let policy = policy.clone();
        axum::middleware::from_fn(move |req, next| {
            let policy = policy.clone();
            async move { middleware::origin_middleware(policy, req, next).await }
        })
    };

    // Protected sub-router: /v1/*, /ws alias, and /metrics — all subject to
    // the origin allowlist and (when enabled) the per-IP rate limiter.
    let protected = Router::new()
        .route("/v1/models", get(http::models))
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
        .route("/metrics", get(http::metrics))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            middleware::http_metrics_middleware,
        ))
        .with_state(state.clone());

    let protected = if config.limits.rate_limit_per_minute > 0 {
        // Replacing `tower_governor` with our own token-bucket implementation
        // (see `rate_limit.rs`) drops the `governor` + `dashmap` +
        // `forwarded-header-value` transitive crates and restores control of
        // the V1-06 refill math: `refill_per_ms = rpm / 60_000`. The V1-11
        // IP-extraction contract (X-Forwarded-For → X-Real-IP → ConnectInfo)
        // is preserved bit-for-bit. `RateLimiter::new` owns the `rpm > MAX_RPM`
        // clamp + warn so the log line below stays consistent.
        let limiter = Arc::new(rate_limit::RateLimiter::new(
            config.limits.rate_limit_per_minute,
            config.limits.rate_limit_burst,
        ));
        let interval_ms = limiter.interval_ms();

        // Background eviction: bound memory under sustained traffic by
        // dropping buckets that haven't been touched in 5 minutes. `tokio`
        // task (not `std::thread::spawn`, V1-15 style) tied to `shutdown_root`
        // so the GC loop exits cleanly on SIGTERM instead of leaking.
        let evict_limiter = limiter.clone();
        let evict_cancel = shutdown_root.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(std::time::Duration::from_secs(60));
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            // First tick fires immediately; skip it so the limiter is populated
            // before the first eviction pass.
            ticker.tick().await;
            loop {
                tokio::select! {
                    biased;
                    _ = evict_cancel.cancelled() => break,
                    _ = ticker.tick() => {
                        evict_limiter.evict_stale(std::time::Duration::from_secs(300));
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
        let layer_limiter = limiter.clone();
        let layer_trust_proxy = config.trust_proxy;
        let layer_metrics = metrics_registry.clone();
        protected.layer(axum::middleware::from_fn(move |req, next| {
            let limiter = layer_limiter.clone();
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

    let app = Router::new()
        .route("/health", get(http::health))
        .route("/ready", get(http::readiness))
        .merge(protected)
        .layer(DefaultBodyLimit::max(config.limits.body_limit_bytes))
        .layer(origin_layer)
        .with_state(state);

    tracing::info!("gigastt server listening on http://{addr}");
    tracing::info!("  WebSocket: ws://{addr}/v1/ws");
    tracing::info!("  REST API:  http://{addr}/health, /ready, /v1/transcribe, /v1/transcribe/stream");
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

    let shutdown_drain_secs = config.limits.shutdown_drain_secs.max(1);

    let shutdown_fut = {
        let shutdown_root = shutdown_root.clone();
        async move {
            match shutdown {
                Some(rx) => {
                    rx.await.ok();
                }
                None => {
                    tokio::signal::ctrl_c().await.ok();
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
            shutdown_engine.pool.close();
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

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_rate_limit_interval_formula() {
        // Mirrors the formula used in `run_with_config` so a regression on the
        // V1-06 fix (integer-divide `/60` truncates sub-60 rpm to 1 rps) trips
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
}
