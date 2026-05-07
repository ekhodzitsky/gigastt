//! Server configuration types, origin policy, and runtime limits.

/// Supported input sample rates (Hz). Default is 48000 for backward
/// compatibility. Single source of truth for both the WebSocket `Ready`
/// payload and the REST `/v1/models` capabilities response.
pub(crate) const SUPPORTED_RATES: &[u32] = &[8000, 16000, 24000, 44100, 48000];
pub(crate) const DEFAULT_SAMPLE_RATE: u32 = 48000;

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
pub(crate) enum OriginVerdict {
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
    pub(crate) fn evaluate(&self, origin: Option<&str>) -> OriginVerdict {
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
}
