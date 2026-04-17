//! HTTP + WebSocket server that accepts audio and streams transcripts.
//!
//! Single port serves both REST API (health, transcribe, SSE) and WebSocket.

pub mod http;

use crate::inference::{Engine, SessionTriplet};
use crate::protocol::{ClientMessage, ServerMessage};
use anyhow::{Context, Result};
use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::extract::State;
use axum::extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade};
use axum::response::Response;
use axum::routing::{get, post};
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

/// Supported input sample rates (Hz). Default is 48000 for backward compatibility.
const SUPPORTED_RATES: &[u32] = &[8000, 16000, 24000, 44100, 48000];
const DEFAULT_SAMPLE_RATE: u32 = 48000;

/// Hint (milliseconds) returned to clients that hit pool backpressure —
/// matches the `Retry-After` header emitted by the REST handlers and keeps
/// transient 503 / WebSocket error payloads consistent with the 30 s
/// checkout timeout used throughout the server.
pub(crate) const POOL_RETRY_AFTER_MS: u32 = 30_000;
pub(crate) const POOL_RETRY_AFTER_SECS: u64 = 30;

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

/// Server runtime configuration. `run_with_config` is the canonical entry
/// point; `run` / `run_with_shutdown` remain as thin wrappers for callers
/// that only need the pre-0.6 positional parameters.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub port: u16,
    pub host: String,
    pub origin_policy: OriginPolicy,
}

impl ServerConfig {
    /// Sensible local-only default: listen on `127.0.0.1:9876`, deny
    /// non-loopback origins.
    pub fn local(port: u16) -> Self {
        Self {
            port,
            host: "127.0.0.1".to_string(),
            origin_policy: OriginPolicy::loopback_only(),
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

    let state = Arc::new(http::AppState {
        engine: Arc::new(engine),
    });

    let policy = Arc::new(config.origin_policy.clone());

    let origin_layer = {
        let policy = policy.clone();
        axum::middleware::from_fn(move |req, next| {
            let policy = policy.clone();
            async move { origin_middleware(policy, req, next).await }
        })
    };

    let app = Router::new()
        .route("/health", get(http::health))
        .route("/v1/models", get(http::models))
        .route("/v1/transcribe", post(http::transcribe))
        .route("/v1/transcribe/stream", post(http::transcribe_stream))
        .route("/ws", get(ws_handler))
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024)) // 50MB
        .layer(origin_layer)
        .with_state(state);

    tracing::info!("gigastt server listening on http://{addr}");
    tracing::info!("  WebSocket: ws://{addr}/ws");
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

    let listener = tokio::net::TcpListener::bind(&addr).await?;

    let shutdown_fut = async {
        match shutdown {
            Some(rx) => {
                rx.await.ok();
            }
            None => {
                tokio::signal::ctrl_c().await.ok();
            }
        }
        tracing::info!("Shutting down server");
    };

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_fut)
    .await?;

    Ok(())
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
    ws.max_message_size(512 * 1024)
        .max_frame_size(512 * 1024)
        .on_upgrade(move |socket| handle_ws(socket, peer, state))
}

async fn handle_ws(socket: WebSocket, peer: SocketAddr, state: Arc<http::AppState>) {
    let triplet = match tokio::time::timeout(
        std::time::Duration::from_secs(30),
        state.engine.pool.checkout(),
    )
    .await
    {
        Ok(triplet) => triplet,
        Err(_) => {
            tracing::warn!("WebSocket pool checkout timeout for {peer}");
            let (mut sink, _) = socket.split();
            let err = ServerMessage::Error {
                message: "Server busy, try again later".into(),
                code: "timeout".into(),
                retry_after_ms: Some(POOL_RETRY_AFTER_MS),
            };
            let _ = sink.send(WsMessage::Text(json_text(&err).into())).await;
            return;
        }
    };

    let (triplet_opt, result) = handle_ws_inner(socket, peer, &state.engine, triplet).await;
    if let Err(e) = result {
        tracing::error!("WebSocket error from {peer}: {e}");
    }

    if let Some(triplet) = triplet_opt {
        state.engine.pool.checkin(triplet).await;
    }
    // If triplet_opt is None, the triplet was lost due to a spawn_blocking panic.
    // The pool degrades gracefully with fewer available slots.
}

/// Runs the WebSocket session loop. Always tries to return the triplet so the
/// caller can check it back into the pool. Returns `None` only if the triplet
/// was lost due to a thread panic inside `spawn_blocking`.
async fn handle_ws_inner(
    socket: WebSocket,
    peer: SocketAddr,
    engine: &Arc<Engine>,
    triplet: SessionTriplet,
) -> (Option<SessionTriplet>, Result<()>) {
    let (mut sink, mut source) = socket.split();

    tracing::info!("Client connected: {peer}");

    // Send ready message
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
    };
    if let Err(e) = sink.send(WsMessage::Text(json_text(&ready).into())).await {
        return (Some(triplet), Err(e.into()));
    }

    let mut state_opt = Some(engine.create_state(
        #[cfg(feature = "diarization")]
        false,
    ));
    let mut triplet_opt = Some(triplet);
    let mut client_sample_rate: u32 = DEFAULT_SAMPLE_RATE;
    let mut audio_received = false;

    let result: Result<()> = 'outer: {
        loop {
            let msg = match tokio::time::timeout(std::time::Duration::from_secs(300), source.next())
                .await
            {
                Ok(Some(Ok(msg))) => msg,
                Ok(Some(Err(e))) => break 'outer Err(e.into()),
                Ok(None) => break,
                Err(_) => {
                    tracing::info!("Client {peer} idle timeout (300s)");
                    break;
                }
            };
            match msg {
                WsMessage::Binary(data) if data.is_empty() => {
                    tracing::debug!("Empty binary frame from {peer}, skipping");
                }
                WsMessage::Binary(data) => {
                    audio_received = true;
                    if data.len() % 2 != 0 {
                        tracing::warn!(
                            "Odd-length PCM frame ({} bytes) from {peer}, dropping last byte",
                            data.len()
                        );
                    }
                    let samples_f32: Vec<f32> = data
                        .chunks_exact(2)
                        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0)
                        .collect();

                    let samples_16k = if client_sample_rate == 16000 {
                        samples_f32
                    } else {
                        match crate::inference::audio::resample(
                            &samples_f32,
                            client_sample_rate,
                            16000,
                        ) {
                            Ok(s) => s,
                            Err(e) => break 'outer Err(e),
                        }
                    };

                    let state = match state_opt.take() {
                        Some(s) => s,
                        None => break 'outer Err(anyhow::anyhow!("Streaming state lost")),
                    };
                    let triplet = match triplet_opt.take() {
                        Some(t) => t,
                        None => {
                            tracing::error!("Triplet unexpectedly missing for {peer}");
                            break 'outer Err(anyhow::anyhow!("Triplet lost"));
                        }
                    };
                    let eng = engine.clone();
                    let join_result = tokio::task::spawn_blocking(move || {
                        // Move ownership into the closure so state and triplet are
                        // returned unconditionally — including after a panic inside
                        // `process_chunk`. Mirrors the pattern in src/server/http.rs.
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
                            state_opt = Some(state_back);
                            triplet_opt = Some(triplet_back);
                            for seg in segments {
                                let msg = if seg.is_final {
                                    ServerMessage::Final {
                                        text: seg.text,
                                        timestamp: seg.timestamp,
                                        words: seg.words,
                                    }
                                } else {
                                    ServerMessage::Partial {
                                        text: seg.text,
                                        timestamp: seg.timestamp,
                                        words: seg.words,
                                    }
                                };
                                if let Err(e) =
                                    sink.send(WsMessage::Text(json_text(&msg).into())).await
                                {
                                    break 'outer Err(e.into());
                                }
                            }
                        }
                        Ok((Ok(Err(e)), state_back, triplet_back)) => {
                            state_opt = Some(state_back);
                            triplet_opt = Some(triplet_back);
                            tracing::error!("Inference error for {peer}: {e:#}");
                            let err = ServerMessage::Error {
                                message: "Inference failed. Please check audio format.".into(),
                                code: "inference_error".into(),
                                retry_after_ms: None,
                            };
                            if let Err(e) = sink.send(WsMessage::Text(json_text(&err).into())).await
                            {
                                break 'outer Err(e.into());
                            }
                        }
                        Ok((Err(_panic), _state_back, triplet_back)) => {
                            // Inference panicked: triplet is recovered, but the streaming
                            // state (LSTM h/c buffers) may be mid-update and unsafe to
                            // reuse. Drop it and install a fresh state so the session can
                            // continue instead of tearing down the connection.
                            tracing::error!(
                                "Panic in WS inference for {peer} — triplet recovered, \
                                 streaming state reset"
                            );
                            triplet_opt = Some(triplet_back);
                            state_opt = Some(engine.create_state(
                                #[cfg(feature = "diarization")]
                                false,
                            ));
                            let err = ServerMessage::Error {
                                message: "Inference failed unexpectedly. Session reset.".into(),
                                code: "inference_panic".into(),
                                retry_after_ms: None,
                            };
                            if let Err(e) = sink.send(WsMessage::Text(json_text(&err).into())).await
                            {
                                break 'outer Err(e.into());
                            }
                        }
                        Err(e) => {
                            // spawn_blocking itself failed (runtime shutdown or cancellation).
                            // Triplet is truly lost in this branch; bail out so the outer
                            // loop does not retry with invalid state.
                            tracing::error!("spawn_blocking join error for {peer}: {e}");
                            break 'outer Err(anyhow::anyhow!("Blocking task join failed"));
                        }
                    }
                }
                WsMessage::Text(text) => {
                    match serde_json::from_str::<ClientMessage>(&text) {
                        Ok(ClientMessage::Configure {
                            sample_rate,
                            diarization,
                        }) => {
                            if audio_received {
                                let err = ServerMessage::Error {
                                    message: "Configure must be sent before first audio frame"
                                        .into(),
                                    code: "configure_too_late".into(),
                                    retry_after_ms: None,
                                };
                                if let Err(e) =
                                    sink.send(WsMessage::Text(json_text(&err).into())).await
                                {
                                    break 'outer Err(e.into());
                                }
                                continue;
                            }
                            if let Some(rate) = sample_rate {
                                if SUPPORTED_RATES.contains(&rate) {
                                    client_sample_rate = rate;
                                    tracing::info!(
                                        "Client {peer} configured sample rate: {rate}Hz"
                                    );
                                } else {
                                    let err = ServerMessage::Error {
                                        message: format!(
                                            "Unsupported sample rate: {rate}Hz. Supported: {SUPPORTED_RATES:?}"
                                        ),
                                        code: "invalid_sample_rate".into(),
                                        retry_after_ms: None,
                                    };
                                    if let Err(e) =
                                        sink.send(WsMessage::Text(json_text(&err).into())).await
                                    {
                                        break 'outer Err(e.into());
                                    }
                                }
                            }

                            // Re-create state if diarization preference changes
                            #[cfg(feature = "diarization")]
                            if let Some(enable_dia) = diarization {
                                tracing::info!(
                                    "Client {peer} configured diarization: {enable_dia}"
                                );
                                state_opt = Some(engine.create_state(enable_dia));
                            }
                            #[cfg(not(feature = "diarization"))]
                            let _ = diarization;
                        }
                        Ok(ClientMessage::Stop) => {
                            tracing::info!("Stop received from {peer}, finalizing");
                            let mut state = match state_opt.take() {
                                Some(s) => s,
                                None => break,
                            };
                            let flush_seg = engine.flush_state(&mut state);
                            drop(state);
                            let final_msg = if let Some(seg) = flush_seg {
                                ServerMessage::Final {
                                    text: seg.text,
                                    timestamp: seg.timestamp,
                                    words: seg.words,
                                }
                            } else {
                                ServerMessage::Final {
                                    text: String::new(),
                                    timestamp: crate::inference::now_timestamp(),
                                    words: vec![],
                                }
                            };
                            if let Err(e) = sink
                                .send(WsMessage::Text(json_text(&final_msg).into()))
                                .await
                            {
                                break 'outer Err(e.into());
                            }
                            break;
                        }
                        Err(_) => {
                            tracing::debug!(
                                "Unrecognized text message from {peer}: {}",
                                &text[..text.len().min(100)]
                            );
                        }
                    }
                }
                WsMessage::Close(_) => break,
                _ => {} // Ignore ping/pong
            }
        }
        Ok(())
    };

    tracing::info!("Client disconnected: {peer}");
    (triplet_opt, result)
}

#[cfg(test)]
mod tests {
    use super::*;

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
