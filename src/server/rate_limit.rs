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
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

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

/// Concurrent map of per-IP buckets. `DashMap` gives us lock-free reads on
/// the common path; writes only lock a single shard.
pub struct RateLimiter {
    buckets: DashMap<IpAddr, TokenBucket>,
    capacity: u32,
    refill_per_ms: f64,
    /// Unix-ms timestamp of the most recent `evict_stale` scan. Used to make
    /// the background GC tokio task observable and idempotent.
    last_evict_ms: AtomicU64,
}

impl RateLimiter {
    /// Construct from the same `(rpm, burst)` pair the CLI exposes.
    ///
    /// `rpm` is clamped to the 60 000 rpm maximum documented in V1-06 (the
    /// interval hits 1 ms precision there; anything higher would truncate to
    /// zero and saturate the bucket).
    pub fn new(rpm: u32, burst: u32) -> Self {
        const MAX_RPM: u32 = 60_000;
        let effective_rpm = rpm.min(MAX_RPM);
        let refill_per_ms = effective_rpm as f64 / 60_000.0;
        Self {
            buckets: DashMap::new(),
            capacity: burst.max(1),
            refill_per_ms,
            last_evict_ms: AtomicU64::new(unix_ms()),
        }
    }

    /// Check a request from `ip`. Returns `true` when the bucket had a token,
    /// `false` when the caller should be 429'd. Inserts a fresh bucket for
    /// first-time callers.
    pub fn check(&self, ip: IpAddr) -> bool {
        let now = Instant::now();
        let now_ms = unix_ms();
        let mut entry = self
            .buckets
            .entry(ip)
            .or_insert_with(|| TokenBucket::new(self.capacity, self.refill_per_ms, now, now_ms));
        entry.try_consume(now, now_ms)
    }

    /// Drop buckets whose `last_seen_ms` is older than `older_than`. Called
    /// from the background tokio task in `run_with_config` to bound memory
    /// under sustained single-visitor traffic.
    pub fn evict_stale(&self, older_than: Duration) {
        let cutoff = unix_ms().saturating_sub(older_than.as_millis() as u64);
        self.buckets
            .retain(|_, bucket| bucket.last_seen_ms >= cutoff);
        self.last_evict_ms.store(unix_ms(), Ordering::Relaxed);
    }

    #[cfg(test)]
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.buckets.len()
    }
}

fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Extract the client IP from `X-Forwarded-For` (first hop), `X-Real-IP`, or
/// the TCP `ConnectInfo`, in that order. Mirrors `SmartIpKeyExtractor` from
/// `tower_governor`. The proxy must overwrite (not append) `X-Forwarded-For`
/// — see `docs/deployment.md` (V1-11).
pub fn extract_client_ip(req: &Request) -> Option<IpAddr> {
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
    req.extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ci| ci.0.ip())
}

/// Build a per-request middleware that consults `limiter` before forwarding
/// to the next layer. Emits the same `429 Too Many Requests` +
/// `Retry-After: 60` contract the previous `tower_governor` layer produced.
pub async fn rate_limit_middleware(
    limiter: Arc<RateLimiter>,
    req: Request,
    next: Next,
) -> Response {
    let ip = extract_client_ip(&req).unwrap_or(IpAddr::V4(Ipv4Addr::UNSPECIFIED));
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
    fn test_extract_ip_prefers_forwarded_for() {
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
        let got = extract_client_ip(&req).expect("XFF must be parsed");
        assert_eq!(got, "203.0.113.42".parse::<IpAddr>().unwrap());
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
        let got = extract_client_ip(&req).expect("ConnectInfo fallback");
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
        let got = extract_client_ip(&req).expect("X-Real-IP fallback");
        assert_eq!(got, "198.51.100.7".parse::<IpAddr>().unwrap());
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
}
