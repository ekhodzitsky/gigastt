//! Per-IP token-bucket rate limiter.
//!
//! Replaces the `tower_governor` crate (which pulled `governor`, `dashmap`,
//! `quanta`, `parking_lot`, and `forwarded-header-value`) with a focused
//! ~150-line implementation tailored to gigastt's single middleware hook.
//!
//! Semantics match the V1-06 formula: `refill_per_ms = rpm / 60_000.0`, so
//! `--rate-limit-per-minute 30` allows one token every 2 s with a configurable
//! burst. When the bucket is empty the caller gets a 429 with `Retry-After: 60`.
//!
//! IP extraction mirrors the old `SmartIpKeyExtractor`:
//! - first hop of `X-Forwarded-For` (trimmed), then
//! - `X-Real-IP`, then
//! - `ConnectInfo<SocketAddr>::ip()`.
//!
//! The rate-limiter & X-Forwarded-For trust boundary is documented in
//! `docs/deployment.md` (V1-11) — the reverse proxy must **overwrite** the
//! header with the real peer address, never append.

use axum::extract::{ConnectInfo, Request};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Json, Response};
use dashmap::DashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Requests per minute. Invariant: `0 < rpm <= MAX_RPM`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rpm(u32);

impl Rpm {
    /// { rpm > 0 && rpm <= MAX_RPM }
    /// fn new(rpm: u32) -> Result<Rpm, String>
    /// { ret.as_ref().map(|r| r.0 > 0 && r.0 <= MAX_RPM).unwrap_or(true) }
    pub fn new(rpm: u32) -> Result<Self, String> {
        if rpm == 0 {
            return Err("rpm must be > 0".into());
        }
        if rpm > MAX_RPM {
            return Err(format!("rpm must be <= {MAX_RPM}"));
        }
        Ok(Rpm(rpm))
    }

    /// { true }
    /// fn get(self) -> u32
    /// { ret > 0 }
    pub fn get(self) -> u32 {
        self.0
    }

    /// Construct without validation. Caller must guarantee `0 < rpm <= MAX_RPM`.
    ///
    /// { rpm > 0 && rpm <= MAX_RPM }
    /// fn from_raw(rpm: u32) -> Rpm
    /// { ret.0 > 0 && ret.0 <= MAX_RPM }
    pub(crate) fn from_raw(rpm: u32) -> Self {
        debug_assert!(rpm > 0 && rpm <= MAX_RPM);
        Rpm(rpm)
    }
}

/// Burst size (max concurrent tokens). Invariant: `burst >= 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Burst(u32);

impl Burst {
    /// { burst >= 1 }
    /// fn new(burst: u32) -> Result<Burst, String>
    /// { ret.as_ref().map(|b| b.0 >= 1).unwrap_or(true) }
    pub fn new(burst: u32) -> Result<Self, String> {
        if burst < 1 {
            return Err("burst must be >= 1".into());
        }
        Ok(Burst(burst))
    }

    /// { true }
    /// fn get(self) -> u32
    /// { ret >= 1 }
    pub fn get(self) -> u32 {
        self.0
    }

    /// Construct without validation. Caller must guarantee `burst >= 1`.
    ///
    /// { burst >= 1 }
    /// fn from_raw(burst: u32) -> Burst
    /// { ret.0 >= 1 }
    pub(crate) fn from_raw(burst: u32) -> Self {
        debug_assert!(burst >= 1);
        Burst(burst)
    }
}

/// Single per-IP bucket. Fractional tokens (`f64`) let us express arbitrary
/// refill rates below 1 token/ms without losing precision — matches the
/// `per_millisecond(60_000 / rpm)` semantics of `tower_governor` 0.7.
#[derive(Debug)]
pub struct TokenBucket {
    capacity: f64,
    refill_per_ms: f64,
    tokens: f64,
    last_refill: Instant,
    /// Wall-clock timestamp of the last refill (milliseconds since the epoch)
    /// used by `RateLimiter::evict_stale` to bound memory. Stored as a plain
    /// `u64` rather than a second `Instant` because eviction is driven off a
    /// single global "now" without needing per-bucket monotonic comparison.
    last_seen_ms: u64,
}

impl TokenBucket {
    /// { refill_per_ms >= 0.0 }
    /// fn new(capacity: u32, refill_per_ms: f64, now: Instant, now_ms: u64) -> TokenBucket
    /// { ret.tokens == ret.capacity && ret.capacity == capacity as f64 && ret.refill_per_ms == refill_per_ms }
    pub fn new(capacity: u32, refill_per_ms: f64, now: Instant, now_ms: u64) -> Self {
        Self {
            capacity: capacity as f64,
            refill_per_ms,
            tokens: capacity as f64,
            last_refill: now,
            last_seen_ms: now_ms,
        }
    }

    /// Refill the bucket based on elapsed time and try to consume one token.
    /// Returns `true` when the request is allowed.
    ///
    /// { refill_per_ms >= 0.0 }
    /// fn try_consume(&mut self, now: Instant, now_ms: u64) -> bool
    /// { ret == (self.tokens >= 1.0) }
    pub fn try_consume(&mut self, now: Instant, now_ms: u64) -> bool {
        let elapsed_ms = now
            .saturating_duration_since(self.last_refill)
            .as_secs_f64()
            * 1000.0;
        if elapsed_ms > 0.0 {
            self.tokens = (self.tokens + elapsed_ms * self.refill_per_ms).min(self.capacity);
            self.last_refill = now;
        }
        self.last_seen_ms = now_ms;
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

/// Upper bound on `rpm` accepted by [`RateLimiter::new`]. Beyond this the
/// 1 ms refill interval would truncate to zero and the bucket would saturate.
pub const MAX_RPM: u32 = 60_000;

/// Concurrent map of per-IP buckets. `DashMap` gives us lock-free reads on
/// the common path; writes only lock a single shard.
pub struct RateLimiter {
    buckets: DashMap<IpAddr, TokenBucket>,
    capacity: Burst,
    refill_per_ms: f64,
    effective_rpm: Rpm,
}

impl RateLimiter {
    /// Construct from the same `(rpm, burst)` pair the CLI exposes.
    ///
    /// `rpm` is clamped to the [`MAX_RPM`] maximum documented in V1-06 (the
    /// interval hits 1 ms precision there; anything higher would truncate to
    /// zero and saturate the bucket). Emits a `warn!` once when clamping.
    ///
    /// { rpm > 0 }
    /// fn new(rpm: u32, burst: u32) -> RateLimiter
    /// { ret.effective_rpm.0 > 0 && ret.capacity.0 >= 1 }
    pub fn new(rpm: u32, burst: u32) -> Self {
        if rpm > MAX_RPM {
            tracing::warn!(
                rpm,
                max_rpm = MAX_RPM,
                "rate_limit_per_minute exceeds {MAX_RPM}; clamped to {MAX_RPM} (1 ms minimum interval)"
            );
        }
        let effective_rpm = rpm.clamp(1, MAX_RPM);
        let refill_per_ms = effective_rpm as f64 / 60_000.0;
        Self {
            buckets: DashMap::new(),
            capacity: Burst::from_raw(burst.max(1)),
            refill_per_ms,
            effective_rpm: Rpm::from_raw(effective_rpm),
        }
    }

    /// Minimum interval between successful requests for the effective (clamped)
    /// rpm, in milliseconds. Used for the startup log line.
    ///
    /// { self.effective_rpm.0 > 0 }
    /// fn interval_ms(&self) -> u64
    /// { ret >= 1 }
    pub fn interval_ms(&self) -> u64 {
        (60_000u64 / self.effective_rpm.0.max(1) as u64).max(1)
    }

    /// Check a request from `ip`. Returns `true` when the bucket had a token,
    /// `false` when the caller should be 429'd. Inserts a fresh bucket for
    /// first-time callers.
    ///
    /// { self.capacity.0 >= 1 }
    /// fn check(&self, ip: IpAddr) -> bool
    /// { ret == (self.buckets[&ip].tokens >= 1.0 after refill) }
    pub fn check(&self, ip: IpAddr) -> bool {
        let now = Instant::now();
        let now_ms = unix_ms();
        let mut entry = self
            .buckets
            .entry(ip)
            .or_insert_with(|| TokenBucket::new(self.capacity.0, self.refill_per_ms, now, now_ms));
        entry.try_consume(now, now_ms)
    }

    /// Drop buckets whose `last_seen_ms` is older than `older_than`. Called
    /// from the background tokio task in `run_with_config` to bound memory
    /// under sustained single-visitor traffic.
    ///
    /// { true }
    /// fn evict_stale(&self, older_than: Duration)
    /// { self.buckets.len() <= old(self.buckets.len()) }
    pub fn evict_stale(&self, older_than: Duration) {
        let cutoff = unix_ms().saturating_sub(older_than.as_millis() as u64);
        self.buckets
            .retain(|_, bucket| bucket.last_seen_ms >= cutoff);
    }

    #[cfg(test)]
    #[allow(clippy::len_without_is_empty)]
    /// { true }
    /// fn len(&self) -> usize
    /// { ret == self.buckets.len() }
    pub fn len(&self) -> usize {
        self.buckets.len()
    }
}

/// { true }
/// fn unix_ms() -> u64
/// { ret >= 0 }
fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// Extract the client IP from `X-Forwarded-For` (first hop), `X-Real-IP`, or
/// the TCP `ConnectInfo`, in that order. Mirrors `SmartIpKeyExtractor` from
/// `tower_governor`. The proxy must overwrite (not append) `X-Forwarded-For`
/// — see `docs/deployment.md` (V1-11).
///
/// When `trust_proxy` is `false`, forwarded headers are ignored entirely and
/// only `ConnectInfo` is used. When `true`, the headers are consulted only
/// if the direct peer IP is loopback or RFC1918.
///
/// { true }
/// fn extract_client_ip(req: &Request, trust_proxy: bool) -> Option<IpAddr>
/// { ret.is_some() == (!trust_proxy || req.extensions().get::<ConnectInfo<SocketAddr>>().is_some() || has_forwarded_headers) }
pub fn extract_client_ip(req: &Request, trust_proxy: bool) -> Option<IpAddr> {
    let direct_ip = req
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ci| ci.0.ip());

    if !trust_proxy {
        return direct_ip;
    }

    // Trust proxy mode: only read forwarded headers when the direct peer
    // is a known private proxy subnet.
    if let Some(connect_ip) = direct_ip
        && !connect_ip.is_loopback()
        && !is_rfc1918(connect_ip)
    {
        return Some(connect_ip);
    }

    let headers = req.headers();
    if let Some(value) = headers.get("x-forwarded-for")
        && let Ok(s) = value.to_str()
    {
        let first = s.split(',').next().unwrap_or("").trim();
        if let Ok(ip) = first.parse::<IpAddr>() {
            return Some(ip);
        }
    }
    if let Some(value) = headers.get("x-real-ip")
        && let Ok(s) = value.to_str()
        && let Ok(ip) = s.trim().parse::<IpAddr>()
    {
        return Some(ip);
    }
    direct_ip
}

/// Return true for IPv4 addresses in RFC1918 space:
/// 10/8, 172.16/12, 192.168/16.
fn is_rfc1918(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let o = v4.octets();
            o[0] == 10 || (o[0] == 172 && (o[1] & 0xF0) == 16) || (o[0] == 192 && o[1] == 168)
        }
        IpAddr::V6(_) => false,
    }
}

/// Build a per-request middleware that consults `limiter` before forwarding
/// to the next layer. Emits the same `429 Too Many Requests` +
/// `Retry-After: 60` contract the previous `tower_governor` layer produced.
///
/// { true }
/// async fn rate_limit_middleware(limiter: Arc<RateLimiter>, trust_proxy: bool, req: Request, next: Next) -> Response
/// { ret.status() == 429 || ret.status() == next.run(req).await.status() }
pub async fn rate_limit_middleware(
    limiter: Arc<RateLimiter>,
    trust_proxy: bool,
    req: Request,
    next: Next,
) -> Response {
    let Some(ip) = extract_client_ip(&req, trust_proxy) else {
        tracing::debug!("rate limit: could not determine client IP");
        return next.run(req).await;
    };
    if limiter.check(ip) {
        next.run(req).await
    } else {
        tracing::debug!(client_ip = %ip, "rate limit rejected request");
        (
            StatusCode::TOO_MANY_REQUESTS,
            [(axum::http::header::RETRY_AFTER, "60")],
            Json(serde_json::json!({
                "error": "Too many requests",
                "code": "rate_limited",
            })),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{HeaderValue, Request as HttpRequest};
    use std::net::Ipv4Addr;

    #[test]
    fn test_token_bucket_allows_within_capacity() {
        // Burst = 5, refill irrelevant for this test — we consume under the
        // cap without waiting, every call must succeed.
        let now = Instant::now();
        let mut bucket = TokenBucket::new(5, 0.0, now, unix_ms());
        for i in 0..5 {
            assert!(bucket.try_consume(now, unix_ms()), "call {i} must succeed");
        }
        // 6th consumption without refill must fail — bucket is empty.
        assert!(
            !bucket.try_consume(now, unix_ms()),
            "6th call must be rate-limited"
        );
    }

    #[test]
    fn test_token_bucket_refills_over_time() {
        // Refill rate = 1 token / ms. Drain the capacity, advance the clock,
        // verify the bucket refills.
        let start = Instant::now();
        let mut bucket = TokenBucket::new(2, 1.0, start, unix_ms());
        assert!(bucket.try_consume(start, unix_ms()));
        assert!(bucket.try_consume(start, unix_ms()));
        assert!(
            !bucket.try_consume(start, unix_ms()),
            "should be drained after 2 consumes"
        );
        let later = start + Duration::from_millis(3);
        assert!(
            bucket.try_consume(later, unix_ms()),
            "should refill after 3 ms"
        );
    }

    #[test]
    fn test_rate_limiter_per_ip_isolation() {
        // Two IPs each with a burst of 1 — one consuming must not drain the
        // other.
        let limiter = RateLimiter::new(1, 1);
        let a: IpAddr = "10.0.0.1".parse().unwrap();
        let b: IpAddr = "10.0.0.2".parse().unwrap();
        assert!(limiter.check(a), "A first call allowed");
        assert!(
            limiter.check(b),
            "B first call allowed (independent bucket)"
        );
        assert!(!limiter.check(a), "A second call rate-limited");
        assert!(!limiter.check(b), "B second call rate-limited");
    }

    #[test]
    fn test_rate_limiter_refill_formula_matches_v1_06() {
        // Mirrors `test_rate_limit_interval_formula` in src/server/mod.rs:
        // `refill_per_ms = rpm / 60_000` must equal `1 / interval_ms` for every
        // `interval_ms = (60_000 / rpm).max(1)` pairing. Concretely: draining
        // the bucket then waiting `interval_ms` must refill exactly 1 token.
        for &rpm in &[1u32, 10, 30, 60, 600, 60_000] {
            let limiter = RateLimiter::new(rpm, 1);
            let ip: IpAddr = "10.0.0.3".parse().unwrap();
            assert!(limiter.check(ip), "rpm={rpm}: initial burst allowed");
            assert!(
                !limiter.check(ip),
                "rpm={rpm}: second immediate call blocked"
            );
            // Advance the bucket's last_refill manually by draining, waiting,
            // and re-checking. Real tests use sleeps; here we inject the
            // refill via `try_consume` with a later instant.
            let mut guard = limiter.buckets.get_mut(&ip).expect("bucket exists");
            let interval_ms = (60_000u64 / rpm as u64).max(1);
            let later = guard.last_refill + Duration::from_millis(interval_ms);
            // `later - last_refill = interval_ms`, so the refill should be
            // `elapsed_ms * refill_per_ms = interval_ms * (rpm / 60_000) >= 1`.
            assert!(
                guard.try_consume(later, unix_ms()),
                "rpm={rpm}: 1 token must refill after {interval_ms} ms",
            );
        }
    }

    #[test]
    fn test_extract_ip_prefers_forwarded_for_when_trusted() {
        // First hop (trimmed) wins over X-Real-IP and ConnectInfo.
        let mut req = HttpRequest::builder()
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();
        req.headers_mut().insert(
            "x-forwarded-for",
            HeaderValue::from_static("203.0.113.42 , 10.0.0.1"),
        );
        req.headers_mut()
            .insert("x-real-ip", HeaderValue::from_static("198.51.100.7"));
        req.extensions_mut().insert(ConnectInfo(SocketAddr::new(
            IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
            12345,
        )));
        let got = extract_client_ip(&req, true).expect("XFF must be parsed");
        assert_eq!(got, "203.0.113.42".parse::<IpAddr>().unwrap());
    }

    #[test]
    fn test_extract_ip_ignores_forwarded_when_not_trusted() {
        // trust_proxy=false: headers ignored, ConnectInfo wins.
        let mut req = HttpRequest::builder()
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();
        req.headers_mut().insert(
            "x-forwarded-for",
            HeaderValue::from_static("203.0.113.42 , 10.0.0.1"),
        );
        req.extensions_mut().insert(ConnectInfo(SocketAddr::new(
            IpAddr::V4(Ipv4Addr::new(198, 51, 100, 7)),
            12345,
        )));
        let got = extract_client_ip(&req, false).expect("ConnectInfo must be used");
        assert_eq!(got, IpAddr::V4(Ipv4Addr::new(198, 51, 100, 7)));
    }

    #[test]
    fn test_extract_ip_falls_back_to_connect_info() {
        // No proxy headers — must fall back to the ConnectInfo peer.
        let mut req = HttpRequest::builder()
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();
        req.extensions_mut().insert(ConnectInfo(SocketAddr::new(
            IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
            55555,
        )));
        let got = extract_client_ip(&req, true).expect("ConnectInfo fallback");
        assert_eq!(got, IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)));
    }

    #[test]
    fn test_extract_ip_uses_real_ip_when_forwarded_for_garbage() {
        // X-Forwarded-For is unparseable; X-Real-IP wins.
        let mut req = HttpRequest::builder()
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();
        req.headers_mut()
            .insert("x-forwarded-for", HeaderValue::from_static("not-an-ip"));
        req.headers_mut()
            .insert("x-real-ip", HeaderValue::from_static("198.51.100.7"));
        req.extensions_mut().insert(ConnectInfo(SocketAddr::new(
            IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
            12345,
        )));
        let got = extract_client_ip(&req, true).expect("X-Real-IP fallback");
        assert_eq!(got, "198.51.100.7".parse::<IpAddr>().unwrap());
    }

    #[test]
    fn test_extract_ip_skips_headers_when_direct_peer_is_public() {
        // trust_proxy=true but ConnectInfo is a public IP → ignore headers.
        let mut req = HttpRequest::builder()
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();
        req.headers_mut()
            .insert("x-forwarded-for", HeaderValue::from_static("203.0.113.42"));
        req.extensions_mut().insert(ConnectInfo(SocketAddr::new(
            IpAddr::V4(Ipv4Addr::new(198, 51, 100, 7)),
            12345,
        )));
        let got = extract_client_ip(&req, true).expect("ConnectInfo used");
        assert_eq!(got, IpAddr::V4(Ipv4Addr::new(198, 51, 100, 7)));
    }

    #[test]
    fn test_eviction_removes_stale() {
        // Populate the limiter with two IPs, artificially age one, confirm
        // eviction drops only the stale bucket.
        let limiter = RateLimiter::new(60, 1);
        let fresh: IpAddr = "10.0.0.4".parse().unwrap();
        let stale: IpAddr = "10.0.0.5".parse().unwrap();
        assert!(limiter.check(fresh));
        assert!(limiter.check(stale));
        // Hand-roll an "old" last_seen_ms on the stale bucket.
        {
            let mut guard = limiter.buckets.get_mut(&stale).expect("stale bucket");
            guard.last_seen_ms = unix_ms().saturating_sub(10 * 60_000); // 10 min old
        }
        limiter.evict_stale(Duration::from_secs(60));
        assert_eq!(limiter.len(), 1, "stale bucket should be evicted");
        assert!(
            limiter.buckets.contains_key(&fresh),
            "fresh bucket must survive eviction"
        );
    }

    #[test]
    fn test_rpm_new_rejects_zero() {
        assert!(Rpm::new(0).is_err());
    }

    #[test]
    fn test_rpm_new_rejects_too_high() {
        assert!(Rpm::new(MAX_RPM + 1).is_err());
    }

    #[test]
    fn test_rpm_new_accepts_valid() {
        let r = Rpm::new(30).unwrap();
        assert_eq!(r.get(), 30);
    }

    #[test]
    fn test_burst_new_rejects_zero() {
        assert!(Burst::new(0).is_err());
    }

    #[test]
    fn test_burst_new_accepts_valid() {
        let b = Burst::new(5).unwrap();
        assert_eq!(b.get(), 5);
    }
}
