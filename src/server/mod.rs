//! HTTP + WebSocket server that accepts audio and streams transcripts.
//!
//! Single port serves both REST API (health, transcribe, SSE) and WebSocket.

pub mod http;
pub mod metrics;
pub mod rate_limit;

use crate::inference::{Engine, SessionTriplet};
use crate::protocol::{ClientMessage, ServerMessage};
use anyhow::{Context, Result};
use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::extract::State;
use axum::extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade};
use axum::http::StatusCode;
use axum::response::Response;
use axum::routing::{get, options, post};
use futures_util::{SinkExt, StreamExt};
use std::net::SocketAddr;
use std::sync::Arc;

/// Serialize a server message to JSON with a safe fallback on error.
fn json_text(msg: &impl serde::Serialize) -> String {
    serde_json::to_string(msg).unwrap_or_else(|e| {
        tracing::error!("Failed to serialize server message: {e}");
        r#"{"type":"error","message":"Internal serialization error","code":"internal"}"#.into()
    })
}

/// Supported input sample rates (Hz). Default is 48000 for backward
/// compatibility. Single source of truth for both the WebSocket `Ready`
/// payload and the REST `/v1/models` capabilities response.
pub(crate) const SUPPORTED_RATES: &[u32] = &[8000, 16000, 24000, 44100, 48000];
const DEFAULT_SAMPLE_RATE: u32 = 48000;

/// Derive the pool-backpressure retry hint from the configured checkout
/// timeout so the `Retry-After` header / `retry_after_ms` JSON field stay
/// consistent with the actual wait window.
pub(crate) fn pool_retry_after_ms(limits: &RuntimeLimits) -> u32 {
    limits
        .pool_checkout_timeout_secs
        .saturating_mul(1000)
        .min(u32::MAX as u64) as u32
}
pub(crate) fn pool_retry_after_secs(limits: &RuntimeLimits) -> u64 {
    limits.pool_checkout_timeout_secs
}

/// Origin policy for CORS + cross-origin deny middleware.
///
/// gigastt is a privacy-first local server: by default we deny cross-origin
/// requests outright so a malicious page cannot trigger transcription from a
/// logged-in user's microphone via a drive-by WebSocket. Loopback origins
/// (`localhost`, `127.0.0.1`, `[::1]`) are always permitted; additional origins
/// must be listed explicitly via `--allow-origin`, and the wildcard `*`
/// behavior is opt-in via `--cors-allow-any`.
#[derive(Debug, Clone, Default)]
pub struct OriginPolicy {
    /// When true, the server accepts ANY `Origin` and echoes `*` in the CORS
    /// response — matches the old v0.5.x behavior. Dangerous default-off.
    pub allow_any: bool,
    /// Exact-match allowlist (e.g. `https://app.example.com`). Case-insensitive.
    pub allowed_origins: Vec<String>,
}

impl OriginPolicy {
    /// Loopback-only default policy: cross-origin requests from non-local
    /// pages are denied until the operator adds explicit allowlist entries.
    pub fn loopback_only() -> Self {
        Self::default()
    }
}

#[derive(Debug)]
enum OriginVerdict {
    /// No `Origin` header or opaque `null` — treat as non-browser client,
    /// no CORS echo required.
    AllowedNoEcho,
    /// Origin matches the policy; echo the exact string (or `*` if
    /// `allow_any` is on).
    Allowed(String),
    /// Origin present but not allowed — respond 403 before reaching the
    /// handler.
    Denied,
}

fn is_loopback_origin(origin: &str) -> bool {
    // Normalize once; compare lowercase prefixes. The prefix must be followed
    // by a port separator (`:`), a path (`/`), or end-of-string — otherwise
    // `http://localhost.evil.com` would be accepted as a DNS continuation of
    // the loopback hostname.
    let lowered = origin.to_ascii_lowercase();
    const HOST_PREFIXES: &[&str] = &[
        "http://localhost",
        "https://localhost",
        "http://127.0.0.1",
        "https://127.0.0.1",
        "http://[::1]",
        "https://[::1]",
    ];
    HOST_PREFIXES.iter().any(|p| match lowered.strip_prefix(p) {
        None => false,
        Some(rest) => rest.is_empty() || rest.starts_with(':') || rest.starts_with('/'),
    })
}

impl OriginPolicy {
    fn evaluate(&self, origin: Option<&str>) -> OriginVerdict {
        let Some(origin) = origin else {
            return OriginVerdict::AllowedNoEcho;
        };
        if origin.eq_ignore_ascii_case("null") {
            return OriginVerdict::AllowedNoEcho;
        }
        if self.allow_any || is_loopback_origin(origin) {
            return OriginVerdict::Allowed(origin.to_string());
        }
        if self
            .allowed_origins
            .iter()
            .any(|a| a.eq_ignore_ascii_case(origin))
        {
            return OriginVerdict::Allowed(origin.to_string());
        }
        OriginVerdict::Denied
    }
}

/// Runtime limits surfaced to per-request handlers. Separate from `ServerConfig`
/// because it needs to travel through `http::AppState` to the WebSocket handler.
#[derive(Debug, Clone)]
pub struct RuntimeLimits {
    /// WebSocket idle timeout. If no frame arrives within this window the
    /// server closes the connection. Default: 300s.
    pub idle_timeout_secs: u64,
    /// Maximum WebSocket frame / message size in bytes. Default: 512 KiB.
    pub ws_frame_max_bytes: usize,
    /// Maximum REST request body in bytes. Default: 50 MiB.
    pub body_limit_bytes: usize,
    /// Per-IP rate limit: requests-per-minute. `0` disables the limiter
    /// (default). Applies to /v1/* and /v1/ws; /health is exempt.
    pub rate_limit_per_minute: u32,
    /// Max burst size before the token bucket starts refilling.
    pub rate_limit_burst: u32,
    /// Maximum wall-clock duration of a single WebSocket session (seconds).
    /// `0` disables the cap entirely (not recommended — a silence-streaming
    /// client would hold a triplet forever). Default: 3600 (1 hour).
    pub max_session_secs: u64,
    /// Grace window (seconds) after the shutdown signal during which in-flight
    /// WebSocket / SSE tasks may emit their final frames and close cleanly.
    /// Values of `0` are clamped to `1` to avoid a no-op drain. Default: 10.
    pub shutdown_drain_secs: u64,
    /// Pool checkout timeout (seconds). REST and WebSocket handlers wait this
    /// long for a free session triplet before returning 503 / `timeout`.
    /// The `Retry-After` hint echoes the same value. Default: 30.
    pub pool_checkout_timeout_secs: u64,
}

impl Default for RuntimeLimits {
    fn default() -> Self {
        Self {
            idle_timeout_secs: 300,
            ws_frame_max_bytes: 512 * 1024,
            body_limit_bytes: 50 * 1024 * 1024,
            rate_limit_per_minute: 0,
            rate_limit_burst: 10,
            max_session_secs: 3600,
            shutdown_drain_secs: 10,
            pool_checkout_timeout_secs: 30,
        }
    }
}

/// Server runtime configuration. `run_with_config` is the canonical entry
/// point; `run` / `run_with_shutdown` remain as thin wrappers for callers
/// that only need the pre-0.6 positional parameters.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub port: u16,
    pub host: String,
    pub origin_policy: OriginPolicy,
    pub limits: RuntimeLimits,
    /// Expose Prometheus metrics at `GET /metrics`. Off by default — keeps
    /// the server quiet for single-user local installs. When on, a
    /// `PrometheusHandle` is attached to `AppState` and the endpoint is
    /// added to the protected router so the Origin allowlist still applies.
    pub metrics_enabled: bool,
    /// Trust `X-Forwarded-For` / `X-Real-IP` for rate-limit IP extraction.
    pub trust_proxy: bool,
}

impl ServerConfig {
    /// Sensible local-only default: listen on `127.0.0.1:9876`, deny
    /// non-loopback origins, default runtime limits, metrics off.
    pub fn local(port: u16) -> Self {
        Self {
            port,
            host: "127.0.0.1".to_string(),
            origin_policy: OriginPolicy::loopback_only(),
            limits: RuntimeLimits::default(),
            metrics_enabled: false,
            trust_proxy: false,
        }
    }
}

/// Start the HTTP + WebSocket STT server on the given host and port.
///
/// Serves REST API endpoints and WebSocket on a single port:
/// - `GET /health` — health check
/// - `POST /v1/transcribe` — file transcription
/// - `POST /v1/transcribe/stream` — SSE streaming transcription
/// - `GET /ws` — WebSocket streaming protocol
///
/// Runs until `Ctrl-C` is received.
pub async fn run(engine: Engine, port: u16, host: &str) -> Result<()> {
    run_with_shutdown(engine, port, host, None).await
}

/// Start server with an optional programmatic shutdown signal.
///
/// When `shutdown` is `Some`, the server stops when the sender fires (or is dropped).
/// When `None`, the server stops on Ctrl-C. Used by tests for clean teardown.
pub async fn run_with_shutdown(
    engine: Engine,
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
    engine: Engine,
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
    engine: Engine,
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
            async move { origin_middleware(policy, req, next).await }
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
        // `/v1/ws` is the canonical path (versioned, aligned with REST); `/ws`
        // remains as an alias for existing clients and logs a deprecation
        // warning on each upgrade. Planned for removal in a future major version.
        .route("/v1/ws", get(ws_handler))
        .route("/ws", get(ws_handler_legacy))
        .route("/metrics", get(http::metrics))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            http_metrics_middleware,
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
        .merge(protected)
        .layer(DefaultBodyLimit::max(config.limits.body_limit_bytes))
        .layer(origin_layer)
        .with_state(state);

    tracing::info!("gigastt server listening on http://{addr}");
    tracing::info!("  WebSocket: ws://{addr}/v1/ws (legacy alias: ws://{addr}/ws)");
    tracing::info!("  REST API:  http://{addr}/health, /v1/transcribe, /v1/transcribe/stream");
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

/// Instrumentation middleware: records HTTP request counters and a duration
/// histogram under the `gigastt_http_*` namespace. Takes the whole
/// `AppState` so we can reach `metrics_registry` — when the operator did
/// not pass `--metrics` the registry is `None` and the middleware
/// degrades to a single `Instant::now()` per request.
async fn http_metrics_middleware(
    State(state): State<Arc<http::AppState>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let Some(registry) = state.metrics_registry.clone() else {
        return next.run(req).await;
    };
    let method = req.method().as_str().to_string();
    let path = req.uri().path().to_string();
    let start = std::time::Instant::now();
    // Sample pool availability on every request.
    registry.gauge_set(
        "gigastt_pool_available",
        vec![],
        state.engine.pool.available() as i64,
    );
    let response = next.run(req).await;
    let elapsed = start.elapsed().as_secs_f64();
    let status = response.status().as_u16().to_string();
    registry.counter_inc(
        "gigastt_http_requests_total",
        vec![
            ("method".into(), method.clone()),
            ("path".into(), path.clone()),
            ("status".into(), status),
        ],
        1,
    );
    registry.histogram_record(
        "gigastt_http_request_duration_seconds",
        vec![("method".into(), method), ("path".into(), path)],
        elapsed,
    );
    response
}

async fn origin_middleware(
    policy: Arc<OriginPolicy>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    use axum::http::{StatusCode, header};
    use axum::response::IntoResponse;

    // `/health` is a liveness probe for container orchestrators and monitoring
    // tools that don't send Origin — let it through unconditionally.
    if req.uri().path() == "/health" {
        return next.run(req).await;
    }

    let origin = req
        .headers()
        .get("origin")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);

    match policy.evaluate(origin.as_deref()) {
        OriginVerdict::AllowedNoEcho => next.run(req).await,
        OriginVerdict::Allowed(echo) => {
            let mut response = next.run(req).await;
            let headers = response.headers_mut();
            let value = if policy.allow_any { "*".into() } else { echo };
            if let Ok(v) = axum::http::HeaderValue::from_str(&value) {
                headers.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, v);
            }
            headers.insert(
                header::ACCESS_CONTROL_ALLOW_METHODS,
                axum::http::HeaderValue::from_static("GET, POST, OPTIONS"),
            );
            headers.insert(
                header::ACCESS_CONTROL_ALLOW_HEADERS,
                axum::http::HeaderValue::from_static("*"),
            );
            headers.insert(header::VARY, axum::http::HeaderValue::from_static("origin"));
            response
        }
        OriginVerdict::Denied => {
            let origin_str = origin.as_deref().unwrap_or("");
            let path = req.uri().path().to_string();
            tracing::warn!(
                origin = %origin_str,
                path = %path,
                "cross-origin request denied by default policy"
            );
            (
                StatusCode::FORBIDDEN,
                axum::response::Json(serde_json::json!({
                    "error": "Origin not allowed",
                    "code": "origin_denied",
                })),
            )
                .into_response()
        }
    }
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    axum::extract::ConnectInfo(peer): axum::extract::ConnectInfo<SocketAddr>,
    State(state): State<Arc<http::AppState>>,
) -> Response {
    // Origin allowlist is enforced by `origin_middleware` before the request
    // reaches this handler; anything that arrives here has already been cleared.
    //
    // V1-03: if shutdown has already been requested, refuse the upgrade
    // instead of handing the client a socket we're about to drain. Returning
    // a plain 503 with the `shutting_down` error code keeps the surface
    // consistent with the pool-saturation 503.
    if state.shutdown.is_cancelled() {
        use axum::http::StatusCode;
        use axum::response::IntoResponse;
        tracing::warn!(peer = %peer, "Rejecting WS upgrade after shutdown");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            axum::response::Json(serde_json::json!({
                "error": "Server shutting down",
                "code": "shutting_down",
            })),
        )
            .into_response();
    }
    let max_bytes = state.limits.ws_frame_max_bytes;
    let state_cloned = state.clone();
    ws.max_message_size(max_bytes)
        .max_frame_size(max_bytes)
        .on_upgrade(move |socket| async move {
            // Track every upgraded handler on the shared TaskTracker so
            // `run_with_config` can wait for in-flight sessions to drain
            // before the process exits. `track_future` returns a wrapper
            // that decrements the tracker when dropped.
            state_cloned
                .tracker
                .clone()
                .track_future(handle_ws(socket, peer, state_cloned.clone()))
                .await
        })
}

/// Deprecated WebSocket endpoint at `/ws`. Identical behaviour to `/v1/ws`
/// but emits a warn-level log on every upgrade and adds RFC 8594 `Deprecation`
/// plus `Link: </v1/ws>; rel="successor-version"` headers on the upgrade
/// response so client libraries can surface the migration warning to users
/// before a future major version drops the alias.
async fn ws_handler_legacy(
    ws: WebSocketUpgrade,
    axum::extract::ConnectInfo(peer): axum::extract::ConnectInfo<SocketAddr>,
    State(state): State<Arc<http::AppState>>,
) -> Response {
    tracing::warn!(
        peer = %peer,
        "WebSocket client connected to deprecated /ws path — switch to /v1/ws"
    );
    if state.shutdown.is_cancelled() {
        use axum::http::StatusCode;
        use axum::response::IntoResponse;
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            axum::response::Json(serde_json::json!({
                "error": "Server shutting down",
                "code": "shutting_down",
            })),
        )
            .into_response();
    }
    let max_bytes = state.limits.ws_frame_max_bytes;
    let state_cloned = state.clone();
    let mut response = ws
        .max_message_size(max_bytes)
        .max_frame_size(max_bytes)
        .on_upgrade(move |socket| async move {
            state_cloned
                .tracker
                .clone()
                .track_future(handle_ws(socket, peer, state_cloned.clone()))
                .await
        });
    let headers = response.headers_mut();
    headers.insert("deprecation", axum::http::HeaderValue::from_static("true"));
    headers.insert(
        "link",
        axum::http::HeaderValue::from_static("</v1/ws>; rel=\"successor-version\""),
    );
    response
}

async fn handle_ws(socket: WebSocket, peer: SocketAddr, state: Arc<http::AppState>) {
    if let Some(ref reg) = state.metrics_registry {
        reg.gauge_inc("gigastt_ws_active_connections", vec![], 1);
    }
    struct WsMetricsGuard(Arc<http::AppState>);
    impl Drop for WsMetricsGuard {
        fn drop(&mut self) {
            if let Some(ref reg) = self.0.metrics_registry {
                reg.gauge_inc("gigastt_ws_active_connections", vec![], -1);
            }
        }
    }
    let _ws_guard = WsMetricsGuard(state.clone());
    let checkout_start = std::time::Instant::now();
    // `select!` the pool checkout against the shutdown token so SIGTERM
    // during pool saturation returns immediately instead of waiting the full
    // checkout window. `biased;` keeps cancel priority over progress.
    let guard = tokio::select! {
        biased;
        _ = state.shutdown.cancelled() => {
            tracing::info!(peer = %peer, "Shutdown requested before pool checkout");
            let (mut sink, _) = socket.split();
            let _ = sink
                .send(WsMessage::Close(Some(axum::extract::ws::CloseFrame {
                    code: 1001,
                    reason: "server shutdown".into(),
                })))
                .await;
            return;
        }
        res = tokio::time::timeout(
            std::time::Duration::from_secs(state.limits.pool_checkout_timeout_secs),
            state.engine.pool.checkout(),
        ) => match res {
            Ok(Ok(guard)) => guard,
            Ok(Err(_pool_closed)) => {
                tracing::info!("WebSocket pool closed for {peer} — server is shutting down");
                let (mut sink, _) = socket.split();
                let err = ServerMessage::Error {
                    message: "Server is shutting down".into(),
                    code: "pool_closed".into(),
                    retry_after_ms: None,
                };
                let _ = sink.send(WsMessage::Text(json_text(&err).into())).await;
                return;
            }
            Err(_) => {
                tracing::warn!("WebSocket pool checkout timeout for {peer}");
                if let Some(ref reg) = state.metrics_registry {
                    reg.counter_inc("gigastt_pool_timeouts_total", vec![], 1);
                    reg.histogram_record("gigastt_pool_checkout_duration_seconds", vec![], checkout_start.elapsed().as_secs_f64());
                }
                let (mut sink, _) = socket.split();
                let err = ServerMessage::Error {
                    message: "Server busy, try again later".into(),
                    code: "timeout".into(),
                    retry_after_ms: Some(pool_retry_after_ms(&state.limits)),
                };
                let _ = sink.send(WsMessage::Text(json_text(&err).into())).await;
                return;
            }
        }
    };

    if let Some(ref reg) = state.metrics_registry {
        reg.histogram_record(
            "gigastt_pool_checkout_duration_seconds",
            vec![],
            checkout_start.elapsed().as_secs_f64(),
        );
    }
    // Strip the lifetime so the triplet can travel through `handle_ws_inner`,
    // which currently owns it directly. The reservation handles checkin on
    // the way back; if the inner loop loses the triplet to a spawn_blocking
    // panic, the reservation is dropped without sending and the pool
    // degrades gracefully — matches the pre-rewrite contract.
    let (triplet, reservation) = guard.into_owned();

    let (triplet_opt, result) = handle_ws_inner(
        socket,
        peer,
        &state.engine,
        &state.limits,
        triplet,
        state.shutdown.clone(),
    )
    .await;
    if let Err(e) = result {
        tracing::error!("WebSocket error from {peer}: {e}");
    }

    if let Some(triplet) = triplet_opt {
        reservation.checkin(triplet);
    }
    // If triplet_opt is None, the triplet was lost due to a spawn_blocking panic.
    // The pool degrades gracefully with fewer available slots.
}

/// Outcome returned by per-frame handlers. Keeps `handle_ws_inner` a thin
/// orchestration loop instead of a 250-line one-big-match.
enum FrameOutcome {
    /// Continue consuming frames.
    Continue,
    /// Clean break — client asked to stop (Stop message) or the socket closed.
    Break,
}

type WsSink = futures_util::stream::SplitSink<WebSocket, WsMessage>;

/// Send a serialized ServerMessage over the WebSocket sink. `?`-friendly so
/// handlers can delegate error propagation without duplicating the sink dance.
async fn send_server_message(sink: &mut WsSink, msg: &ServerMessage) -> Result<()> {
    sink.send(WsMessage::Text(json_text(msg).into()))
        .await
        .map_err(Into::into)
}

/// Handle a single PCM16 audio frame: resample if needed, run inference in a
/// `spawn_blocking` guarded by `catch_unwind`, and emit partial/final/error
/// payloads. Always returns the triplet to `triplet_opt` (or recovers a fresh
/// state after an inference panic) so the connection can keep serving.
#[allow(clippy::too_many_arguments)]
async fn handle_binary_frame(
    sink: &mut WsSink,
    engine: &Arc<Engine>,
    state_opt: &mut Option<crate::inference::StreamingState>,
    triplet_opt: &mut Option<SessionTriplet>,
    audio_received: &mut bool,
    client_sample_rate: u32,
    pending_byte: &mut Option<u8>,
    peer: SocketAddr,
    data: axum::body::Bytes,
) -> Result<FrameOutcome> {
    if data.is_empty() {
        tracing::debug!("Empty binary frame from {peer}, skipping");
        return Ok(FrameOutcome::Continue);
    }
    *audio_received = true;

    // V1-25: PCM16 samples are 2 bytes each. Previously we called
    // `chunks_exact(2)` directly and silently dropped a trailing odd byte,
    // which introduced a 1-sample phase shift for every subsequent frame —
    // subtle on the decode side, hard to diagnose. Carry the odd byte
    // across frames: prepend the one left over from the previous frame
    // (if any), then save the new remainder for next time. Observation-only
    // `warn!` remains so server-side logs still flag misaligned streams.
    let carry_prev = pending_byte.take();
    let samples_f32: Vec<f32> = if carry_prev.is_some() || !data.len().is_multiple_of(2) {
        let mut combined = Vec::with_capacity(data.len() + 1);
        if let Some(prev) = carry_prev {
            combined.push(prev);
        }
        combined.extend_from_slice(&data);
        if !combined.len().is_multiple_of(2) {
            tracing::warn!(
                "Odd-length PCM stream from {peer}: {} bytes incl. carry, deferring 1 byte",
                combined.len()
            );
            *pending_byte = combined.pop();
        }
        combined
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0)
            .collect()
    } else {
        data.chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0)
            .collect()
    };
    let samples_16k = if client_sample_rate == 16000 {
        samples_f32
    } else {
        let state_ref = state_opt
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Streaming state lost"))?;
        crate::inference::audio::resample_with_cache(
            &samples_f32,
            crate::inference::audio::SampleRate(client_sample_rate),
            crate::inference::audio::SampleRate(16000),
            &mut state_ref.resampler,
        )?
    };

    let state = state_opt
        .take()
        .ok_or_else(|| anyhow::anyhow!("Streaming state lost"))?;
    let triplet = triplet_opt.take().ok_or_else(|| {
        tracing::error!("Triplet unexpectedly missing for {peer}");
        anyhow::anyhow!("Triplet lost")
    })?;

    let eng = engine.clone();
    let join_result = tokio::task::spawn_blocking(move || {
        // Move ownership into the closure so state and triplet come back
        // unconditionally, including after a panic inside `process_chunk`.
        // Mirrors the pattern in src/server/http.rs.
        let mut state = state;
        let mut triplet = triplet;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            eng.process_chunk(&samples_16k, &mut state, &mut triplet)
        }));
        (r, state, triplet)
    })
    .await;

    match join_result {
        Ok((Ok(Ok(segments)), state_back, triplet_back)) => {
            *state_opt = Some(state_back);
            *triplet_opt = Some(triplet_back);
            for seg in segments {
                let msg = if seg.is_final {
                    ServerMessage::Final(seg)
                } else {
                    ServerMessage::Partial(seg)
                };
                send_server_message(sink, &msg).await?;
            }
            Ok(FrameOutcome::Continue)
        }
        Ok((Ok(Err(e)), state_back, triplet_back)) => {
            *state_opt = Some(state_back);
            *triplet_opt = Some(triplet_back);
            tracing::error!("Inference error for {peer}: {e:#}");
            send_server_message(
                sink,
                &ServerMessage::Error {
                    message: "Inference failed. Please check audio format.".into(),
                    code: "inference_error".into(),
                    retry_after_ms: None,
                },
            )
            .await?;
            Ok(FrameOutcome::Continue)
        }
        Ok((Err(_panic), _state_back, triplet_back)) => {
            // Inference panicked: triplet is recovered, but the streaming
            // state (LSTM h/c buffers) may be mid-update and unsafe to reuse.
            // Drop it and install a fresh state so the session continues.
            tracing::error!(
                "Panic in WS inference for {peer} — triplet recovered, streaming state reset"
            );
            *triplet_opt = Some(triplet_back);
            *state_opt = Some(engine.create_state(false));
            send_server_message(
                sink,
                &ServerMessage::Error {
                    message: "Inference failed unexpectedly. Session reset.".into(),
                    code: "inference_panic".into(),
                    retry_after_ms: None,
                },
            )
            .await?;
            Ok(FrameOutcome::Continue)
        }
        Err(e) => {
            // spawn_blocking itself failed (runtime shutdown or cancellation).
            // Triplet is truly lost in this branch; bail out.
            tracing::error!("spawn_blocking join error for {peer}: {e}");
            Err(anyhow::anyhow!("Blocking task join failed"))
        }
    }
}

/// Handle `{"type":"configure",…}`. Rejects configure-after-first-audio,
/// validates sample rate against `SUPPORTED_RATES`, and (with diarization
/// feature) recreates the streaming state.
#[allow(clippy::too_many_arguments)]
async fn handle_configure_message(
    sink: &mut WsSink,
    engine: &Arc<Engine>,
    state_opt: &mut Option<crate::inference::StreamingState>,
    client_sample_rate: &mut u32,
    audio_received: bool,
    sample_rate: Option<u32>,
    diarization: Option<bool>,
    protocol_version: Option<String>,
    peer: SocketAddr,
) -> Result<FrameOutcome> {
    if audio_received {
        send_server_message(
            sink,
            &ServerMessage::Error {
                message: "Configure must be sent before first audio frame".into(),
                code: "configure_too_late".into(),
                retry_after_ms: None,
            },
        )
        .await?;
        return Ok(FrameOutcome::Continue);
    }
    if let Some(ref ver) = protocol_version
        && ver != crate::protocol::PROTOCOL_VERSION
    {
        send_server_message(
            sink,
            &ServerMessage::Error {
                message: format!(
                    "Unsupported protocol version: {ver}. Supported: {}",
                    crate::protocol::PROTOCOL_VERSION
                ),
                code: "unsupported_protocol_version".into(),
                retry_after_ms: None,
            },
        )
        .await?;
        return Ok(FrameOutcome::Break);
    }
    if let Some(rate) = sample_rate {
        if SUPPORTED_RATES.contains(&rate) {
            *client_sample_rate = rate;
            tracing::info!("Client {peer} configured sample rate: {rate}Hz");
        } else {
            send_server_message(
                sink,
                &ServerMessage::Error {
                    message: format!(
                        "Unsupported sample rate: {rate}Hz. Supported: {SUPPORTED_RATES:?}"
                    ),
                    code: "invalid_sample_rate".into(),
                    retry_after_ms: None,
                },
            )
            .await?;
        }
    }
    #[cfg(feature = "diarization")]
    if let Some(enable_dia) = diarization {
        tracing::info!("Client {peer} configured diarization: {enable_dia}");
        *state_opt = Some(engine.create_state(enable_dia));
    }
    #[cfg(not(feature = "diarization"))]
    {
        let _ = (engine, state_opt, diarization);
    }
    Ok(FrameOutcome::Continue)
}

/// Handle `{"type":"stop"}`. Flushes the streaming state, sends a final
/// segment (empty if there was nothing pending), and signals clean break.
async fn handle_stop_message(
    sink: &mut WsSink,
    engine: &Arc<Engine>,
    state_opt: &mut Option<crate::inference::StreamingState>,
    peer: SocketAddr,
) -> Result<FrameOutcome> {
    tracing::info!("Stop received from {peer}, finalizing");
    let Some(mut state) = state_opt.take() else {
        return Ok(FrameOutcome::Break);
    };
    let flush_seg = engine.flush_state(&mut state);
    drop(state);
    let final_msg = if let Some(seg) = flush_seg {
        ServerMessage::Final(seg)
    } else {
        ServerMessage::Final(crate::inference::TranscriptSegment {
            text: String::new(),
            timestamp: crate::inference::now_timestamp(),
            words: vec![],
            is_final: true,
        })
    };
    send_server_message(sink, &final_msg).await?;
    Ok(FrameOutcome::Break)
}

/// Flush any pending streaming state and emit a `Final` frame (even an empty
/// one) so e2e tests and clients can reliably assert that every session ends
/// with a Final before the Close. Used by the cancel and session-cap branches
/// of `handle_ws_inner`.
async fn flush_and_final(
    sink: &mut WsSink,
    engine: &Arc<Engine>,
    state_opt: &mut Option<crate::inference::StreamingState>,
) -> Result<()> {
    let flush_seg = state_opt
        .as_mut()
        .and_then(|state| engine.flush_state(state));
    let final_msg = match flush_seg {
        Some(seg) => ServerMessage::Final(seg),
        None => ServerMessage::Final(crate::inference::TranscriptSegment {
            text: String::new(),
            timestamp: crate::inference::now_timestamp(),
            words: vec![],
            is_final: true,
        }),
    };
    send_server_message(sink, &final_msg).await
}

/// Runs the WebSocket session loop. Always tries to return the triplet so the
/// caller can check it back into the pool. Returns `None` only if the triplet
/// was lost due to a thread panic inside `spawn_blocking`.
async fn handle_ws_inner(
    socket: WebSocket,
    peer: SocketAddr,
    engine: &Arc<Engine>,
    limits: &RuntimeLimits,
    triplet: SessionTriplet,
    cancel: tokio_util::sync::CancellationToken,
) -> (Option<SessionTriplet>, Result<()>) {
    let (mut sink, mut source) = socket.split();
    tracing::info!("Client connected: {peer}");

    #[cfg(feature = "diarization")]
    let diarization_available = engine.has_speaker_encoder();
    #[cfg(not(feature = "diarization"))]
    let diarization_available = false;

    let ready = ServerMessage::Ready {
        model: "gigaam-v3-e2e-rnnt".into(),
        sample_rate: DEFAULT_SAMPLE_RATE,
        version: crate::protocol::PROTOCOL_VERSION.into(),
        supported_rates: SUPPORTED_RATES.to_vec(),
        diarization: diarization_available,
        min_protocol_version: None,
    };
    if let Err(e) = send_server_message(&mut sink, &ready).await {
        return (Some(triplet), Err(e));
    }

    let mut state_opt = Some(engine.create_state(false));
    let mut triplet_opt = Some(triplet);
    let mut client_sample_rate: u32 = DEFAULT_SAMPLE_RATE;
    let mut audio_received = false;
    // V1-25: carries the trailing odd byte across PCM16 frames so clients
    // that split their streams on odd boundaries don't accumulate a
    // 1-sample phase shift in the decoded audio.
    let mut pending_byte: Option<u8> = None;

    let idle_timeout = std::time::Duration::from_secs(limits.idle_timeout_secs);

    // V1-04: wall-clock deadline independent of `idle_timeout`. Setting
    // `max_session_secs = 0` disables the cap by parking the deadline far in
    // the future (u64::MAX / 2 ≈ 292 billion years) so `sleep_until` never
    // fires — callers who deliberately want unlimited sessions don't pay for
    // an additional branch in the select.
    let session_deadline = if limits.max_session_secs == 0 {
        tokio::time::Instant::now() + std::time::Duration::from_secs(u64::MAX / 2)
    } else {
        tokio::time::Instant::now() + std::time::Duration::from_secs(limits.max_session_secs)
    };

    let result: Result<()> = loop {
        // Fast-path deadline / cancel check: if a client streams frames
        // continuously (e.g. 20 ms silence every 100 ms) the `source.next()`
        // arm is always ready when we re-enter `select!`, and with `biased;`
        // the runtime still polls cancel / sleep_until first — but only if
        // they have a registered waker. `sleep_until` registers its waker
        // correctly, yet a subtle race on fast CI runners can let the frame
        // arm fire before the timer's waker is installed. A cheap
        // pre-check here guarantees the deadline / cancel wins.
        if cancel.is_cancelled() {
            tracing::info!(peer = %peer, "Shutdown signalled — flushing WS session");
            let _ = flush_and_final(&mut sink, engine, &mut state_opt).await;
            let _ = sink
                .send(WsMessage::Close(Some(axum::extract::ws::CloseFrame {
                    code: 1001,
                    reason: "server shutdown".into(),
                })))
                .await;
            break Ok(());
        }
        if tokio::time::Instant::now() >= session_deadline {
            tracing::warn!(
                peer = %peer,
                max_session_secs = limits.max_session_secs,
                "Session cap reached — closing WS"
            );
            let _ = send_server_message(
                &mut sink,
                &ServerMessage::Error {
                    message: "Maximum session duration exceeded".into(),
                    code: "max_session_duration_exceeded".into(),
                    retry_after_ms: None,
                },
            )
            .await;
            let _ = flush_and_final(&mut sink, engine, &mut state_opt).await;
            let _ = sink
                .send(WsMessage::Close(Some(axum::extract::ws::CloseFrame {
                    code: 1008,
                    reason: "max session duration".into(),
                })))
                .await;
            break Ok(());
        }

        tokio::select! {
            // `biased;` — cancel > deadline > frame. Guarantees that a
            // SIGTERM always wins a race against a pending frame, so the
            // drain path is deterministic.
            biased;

            _ = cancel.cancelled() => {
                tracing::info!(peer = %peer, "Shutdown signalled — flushing WS session");
                // Best-effort: the socket may already be dead if the peer
                // closed first, so every send is swallowed.
                let _ = flush_and_final(&mut sink, engine, &mut state_opt).await;
                let _ = sink
                    .send(WsMessage::Close(Some(axum::extract::ws::CloseFrame {
                        code: 1001,
                        reason: "server shutdown".into(),
                    })))
                    .await;
                break Ok(());
            }

            _ = tokio::time::sleep_until(session_deadline) => {
                tracing::warn!(
                    peer = %peer,
                    max_session_secs = limits.max_session_secs,
                    "Session cap reached — closing WS"
                );
                let _ = send_server_message(
                    &mut sink,
                    &ServerMessage::Error {
                        message: "Maximum session duration exceeded".into(),
                        code: "max_session_duration_exceeded".into(),
                        retry_after_ms: None,
                    },
                )
                .await;
                let _ = flush_and_final(&mut sink, engine, &mut state_opt).await;
                let _ = sink
                    .send(WsMessage::Close(Some(axum::extract::ws::CloseFrame {
                        code: 1008,
                        reason: "max session duration".into(),
                    })))
                    .await;
                break Ok(());
            }

            maybe_msg = tokio::time::timeout(idle_timeout, source.next()) => {
                let msg = match maybe_msg {
                    Ok(Some(Ok(msg))) => msg,
                    Ok(Some(Err(e))) => break Err(e.into()),
                    Ok(None) => break Ok(()),
                    Err(_) => {
                        tracing::info!(
                            "Client {peer} idle timeout ({}s)",
                            limits.idle_timeout_secs
                        );
                        break Ok(());
                    }
                };

                let outcome = match msg {
                    WsMessage::Binary(data) => {
                        handle_binary_frame(
                            &mut sink,
                            engine,
                            &mut state_opt,
                            &mut triplet_opt,
                            &mut audio_received,
                            client_sample_rate,
                            &mut pending_byte,
                            peer,
                            data,
                        )
                        .await
                    }
                    WsMessage::Text(text) => match serde_json::from_str::<ClientMessage>(&text) {
                        Ok(ClientMessage::Configure {
                            sample_rate,
                            diarization,
                            protocol_version,
                        }) => {
                            handle_configure_message(
                                &mut sink,
                                engine,
                                &mut state_opt,
                                &mut client_sample_rate,
                                audio_received,
                                sample_rate,
                                diarization,
                                protocol_version,
                                peer,
                            )
                            .await
                        }
                        Ok(ClientMessage::Stop) => {
                            handle_stop_message(&mut sink, engine, &mut state_opt, peer).await
                        }
                        Err(_) => {
                            tracing::debug!(
                                "Unrecognized text message from {peer}: {}",
                                &text[..text.len().min(100)]
                            );
                            Ok(FrameOutcome::Continue)
                        }
                    },
                    WsMessage::Close(_) => Ok(FrameOutcome::Break),
                    _ => Ok(FrameOutcome::Continue), // ignore ping/pong
                };

                match outcome {
                    Ok(FrameOutcome::Continue) => continue,
                    Ok(FrameOutcome::Break) => break Ok(()),
                    Err(e) => break Err(e),
                }
            }
        }
    };

    tracing::info!("Client disconnected: {peer}");
    (triplet_opt, result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_limits_default_rate_limit_disabled() {
        let limits = RuntimeLimits::default();
        assert_eq!(
            limits.rate_limit_per_minute, 0,
            "rate limiting must be off by default (privacy-first)"
        );
        assert_eq!(limits.rate_limit_burst, 10, "default burst size must be 10");
    }

    #[test]
    fn test_runtime_limits_default_session_and_drain() {
        // V1-03 / V1-04: locks in the documented defaults so a silent change
        // can't quietly disable the shutdown drain or the session cap.
        let limits = RuntimeLimits::default();
        assert_eq!(
            limits.max_session_secs, 3600,
            "default session cap must be 1 hour to stop silence-streamers from \
             holding a triplet forever"
        );
        assert_eq!(
            limits.shutdown_drain_secs, 10,
            "default shutdown drain must be 10 s — comfortably inside the usual \
             k8s terminationGracePeriodSeconds = 30"
        );
    }

    #[test]
    fn test_supported_rates_contains_common() {
        assert!(
            SUPPORTED_RATES.contains(&8000),
            "SUPPORTED_RATES must include 8000 Hz"
        );
        assert!(
            SUPPORTED_RATES.contains(&16000),
            "SUPPORTED_RATES must include 16000 Hz"
        );
        assert!(
            SUPPORTED_RATES.contains(&48000),
            "SUPPORTED_RATES must include 48000 Hz"
        );
    }

    #[test]
    fn test_default_sample_rate_in_supported() {
        assert!(
            SUPPORTED_RATES.contains(&DEFAULT_SAMPLE_RATE),
            "DEFAULT_SAMPLE_RATE ({DEFAULT_SAMPLE_RATE}) must be present in SUPPORTED_RATES"
        );
    }

    #[test]
    fn test_loopback_origin_matcher() {
        assert!(is_loopback_origin("http://localhost"));
        assert!(is_loopback_origin("https://localhost:3000"));
        assert!(is_loopback_origin("http://127.0.0.1:9876"));
        assert!(is_loopback_origin("HTTPS://127.0.0.1")); // case-insensitive
        assert!(is_loopback_origin("http://[::1]:9876"));
        assert!(!is_loopback_origin("https://evil.example.com"));
        assert!(!is_loopback_origin("http://192.168.1.10"));
        // Foiled prefix spoof: host must be exactly localhost / 127.0.0.1 / [::1]
        assert!(!is_loopback_origin("http://localhost.evil.example.com"));
    }

    #[test]
    fn test_origin_policy_default_denies_third_party() {
        let policy = OriginPolicy::loopback_only();
        assert!(matches!(
            policy.evaluate(Some("https://evil.example.com")),
            OriginVerdict::Denied
        ));
    }

    #[test]
    fn test_origin_policy_allows_loopback_by_default() {
        let policy = OriginPolicy::loopback_only();
        assert!(matches!(
            policy.evaluate(Some("http://localhost:3000")),
            OriginVerdict::Allowed(_)
        ));
    }

    #[test]
    fn test_origin_policy_allows_listed_origin() {
        let policy = OriginPolicy {
            allow_any: false,
            allowed_origins: vec!["https://app.example.com".into()],
        };
        assert!(matches!(
            policy.evaluate(Some("https://app.example.com")),
            OriginVerdict::Allowed(_)
        ));
        // Trailing-path mutations are not a match — allowlist is exact origin only.
        assert!(matches!(
            policy.evaluate(Some("https://app.example.com.evil.com")),
            OriginVerdict::Denied
        ));
    }

    #[test]
    fn test_origin_policy_allow_any_short_circuits() {
        let policy = OriginPolicy {
            allow_any: true,
            allowed_origins: vec![],
        };
        assert!(matches!(
            policy.evaluate(Some("https://anything.example.com")),
            OriginVerdict::Allowed(_)
        ));
    }

    #[test]
    fn test_origin_policy_no_header_allowed() {
        let policy = OriginPolicy::loopback_only();
        assert!(matches!(
            policy.evaluate(None),
            OriginVerdict::AllowedNoEcho
        ));
        assert!(matches!(
            policy.evaluate(Some("null")),
            OriginVerdict::AllowedNoEcho
        ));
    }

    #[tokio::test]
    async fn test_origin_middleware_integration() {
        // End-to-end check of the origin_middleware layer on a minimal
        // router. Uses real axum::serve + reqwest to catch regressions that
        // unit tests on `OriginPolicy` alone would miss — e.g. the middleware
        // attaching to the wrong routes, or `/health` accidentally being
        // guarded.
        use axum::Router;
        use axum::routing::get;

        let policy = Arc::new(OriginPolicy::loopback_only());
        let origin_layer = {
            let policy = policy.clone();
            axum::middleware::from_fn(move |req, next| {
                let policy = policy.clone();
                async move { origin_middleware(policy, req, next).await }
            })
        };
        let app = Router::new()
            .route("/health", get(|| async { "ok" }))
            .route("/v1/transcribe", get(|| async { "ok" }))
            .layer(origin_layer);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        // Allow the server to bind.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let client = reqwest::Client::new();
        let base = format!("http://127.0.0.1:{port}");

        // /health is exempt — monitoring probes work even when Origin is set.
        let r = client
            .get(format!("{base}/health"))
            .header("Origin", "https://evil.example.com")
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "/health must skip the Origin guard");

        // Cross-origin request must be denied on /v1/*.
        let r = client
            .get(format!("{base}/v1/transcribe"))
            .header("Origin", "https://evil.example.com")
            .send()
            .await
            .unwrap();
        assert_eq!(
            r.status(),
            403,
            "non-loopback Origin must receive 403 Forbidden"
        );
        let text = r.text().await.unwrap();
        let body: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(body["code"], "origin_denied");

        // Loopback origin is always allowed.
        let r = client
            .get(format!("{base}/v1/transcribe"))
            .header("Origin", "http://localhost:3000")
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "loopback Origin must be allowed");
        assert_eq!(
            r.headers()
                .get("access-control-allow-origin")
                .and_then(|v| v.to_str().ok()),
            Some("http://localhost:3000"),
            "CORS echo must mirror the incoming Origin (no wildcard by default)",
        );

        // No Origin header (curl, CLI, server-to-server) — policy allows
        // through without a CORS echo.
        let r = client
            .get(format!("{base}/v1/transcribe"))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "requests without Origin must pass");

        // Attacker trying DNS continuation on the loopback prefix must be denied.
        let r = client
            .get(format!("{base}/v1/transcribe"))
            .header("Origin", "http://localhost.evil.example.com")
            .send()
            .await
            .unwrap();
        assert_eq!(
            r.status(),
            403,
            "localhost.* DNS continuation must not impersonate loopback"
        );
    }

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

    #[test]
    fn test_catch_unwind_preserves_ownership_across_panic() {
        // Locks in the ownership contract used by `handle_ws_inner`'s spawn_blocking
        // block: moving captured values into the closure and wrapping the inner
        // computation in `catch_unwind(AssertUnwindSafe(_))` guarantees that the
        // values are observable after a panic, so the triplet can be returned to the
        // pool and the streaming state can be reset.
        use std::panic::{AssertUnwindSafe, catch_unwind};

        let mut state = 42u32;
        let mut triplet_marker = String::from("pool_slot");

        let result = catch_unwind(AssertUnwindSafe(|| {
            state = 99;
            triplet_marker.push_str("/taken");
            panic!("simulated inference panic");
        }));

        assert!(result.is_err(), "catch_unwind must report the panic");
        assert_eq!(state, 99, "state must remain accessible after panic");
        assert_eq!(
            triplet_marker, "pool_slot/taken",
            "triplet marker must survive panic"
        );
    }
}
