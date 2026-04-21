# Design: V1-06 + V1-07 (+ V1-21) — rate-limiter math & pool primitive

Owner: architect-3 (2026-04-20). Bundles three tightly-coupled findings:
- **V1-06**: rate-limiter `/60` integer division (`src/server/mod.rs:278`)
- **V1-07**: `Mutex<mpsc::Receiver>` + `.expect("Pool sender dropped")`
- **V1-21**: `SessionPool` Drop-guard (naturally part of the V1-07 rewrite)

Target tag: `v0.9.0`.

## Part A — V1-06 rate-limiter

### A.1 Fix
Replace `(rpm / 60).max(1)` with millisecond-precision interval:
```rust
let interval_ms = (60_000u64 / rpm as u64).max(1);
GovernorConfigBuilder::default()
    .per_millisecond(interval_ms)
    .burst_size(config.limits.rate_limit_burst)
    .key_extractor(SmartIpKeyExtractor)
    .finish()
```
`tower_governor` 0.7 exposes `per_millisecond`; same internal
`Nanoseconds` representation, clearer than `per_nanosecond`.

### A.2 Edge cases

| `rpm` | before fix | after fix |
|-------|-----------|-----------|
| 0 | layer not built | unchanged |
| 1 | 60 rpm (bug) | `per_ms(60_000)` = 1 rpm |
| 30 | **60 rpm (bug)** | `per_ms(2_000)` = 30 rpm |
| 59 | **60 rpm (bug)** | `per_ms(1_016)` ≈ 59.05 rpm |
| 60 | 60 rpm | `per_ms(1_000)` = 60 rpm |
| 600 | 600 rpm | `per_ms(100)` = 600 rpm |
| 60_000 | 60_000 rpm | `per_ms(1)` = 60_000 rpm |
| > 60_000 | silently clamped | clamp + `warn!` |

`burst_size` semantics unchanged; only the refill rate is corrected.

### A.3 CLI validation
Leave clap permissive (`u32`, default 0); add runtime `warn!` when `rpm >
60_000`. A hard `value_parser!(…).range(0..=60_000)` would kill the process
on a typo — too aggressive.

### A.4 Tests
- Unit: table-driven check of the interval formula.
- `tests/e2e_rate_limit.rs` (new, `#[ignore]`): server with `rpm=30,
  burst=1`; expect 2nd request 429, 3rd request (after 2 s) 200.

### A.5 Docs
- `docs/deployment.md`: add a row warning operators who set `rpm < 60`
  before v0.9.0 — actual protection was 60 rpm.
- `CHANGELOG.md [Unreleased] / Fixed`.
- Log line should include both `rpm` and `interval_ms` for diagnostics.

## Part B — V1-07 + V1-21 pool rewrite

### B.1 Trade-off analysis

| Axis | A: `async_channel` | B: `Semaphore + Mutex<VecDeque>` | C: `crossbeam + sync→async` |
|---|---|---|---|
| FIFO | intrinsic (concurrent-queue) | semaphore FIFO, mutex best-effort | FIFO via crossbeam |
| Close semantic | `close()` → `recv() → Err` | manual `AtomicBool` + `Notify` | `Sender::drop` |
| Deps | `async-channel 2.x` (~3 trans deps already in graph) | none | `crossbeam-channel` + `spawn_blocking` wrapper |
| Complexity | single primitive | two primitives, race-prone | async-sync impedance |
| Perf | ~100–300 ns | ~400–600 ns | µs (deal-breaker) |

**Recommendation:** A (`async_channel`). Simpler contract, close semantics
map onto graceful shutdown, dependency graph already contains the
transitive pieces (`concurrent-queue`, `event-listener`). **Fallback:** B —
if, in the first draft, the Semaphore+VecDeque variant stays under ~30
extra lines, prefer it for zero supply-chain surface.

### B.2 `PoolGuard` contract (V1-21)

```rust
#[derive(Debug, thiserror::Error)]
pub enum PoolError {
    #[error("session pool is closed")]
    Closed,
}

pub struct SessionPool {
    sender: async_channel::Sender<SessionTriplet>,
    receiver: async_channel::Receiver<SessionTriplet>,
    total: usize,
    // available: AtomicUsize REMOVED — receiver.len() is O(1)
}

impl SessionPool {
    pub async fn checkout(&self) -> Result<PoolGuard<'_>, PoolError>;
    pub fn close(&self);
    pub fn total(&self) -> usize { self.total }
    pub fn available(&self) -> usize { self.receiver.len() }
}

pub struct PoolGuard<'a> { pool: &'a SessionPool, triplet: Option<SessionTriplet> }

impl PoolGuard<'_> {
    /// Strip the lifetime for spawn_blocking ownership.
    pub fn into_owned(mut self) -> (SessionTriplet, OwnedReservation);
}

impl Deref / DerefMut for PoolGuard<'_> { /* triplet */ }

impl Drop for PoolGuard<'_> {
    fn drop(&mut self) {
        if let Some(t) = self.triplet.take() {
            // Best-effort checkin: try_send; on Closed, drop the triplet.
        }
    }
}

pub struct OwnedReservation { sender: async_channel::Sender<SessionTriplet> }
impl OwnedReservation {
    pub fn checkin(self, triplet: SessionTriplet) { let _ = self.sender.try_send(triplet); }
}
```

Guarantees: automatic checkin on panic unwind, FIFO under contention,
`pool.close()` converts all waiters into `PoolError::Closed`, `blocking_checkin`
is gone (replaced by `OwnedReservation`).

### B.3 Precise edits

`src/inference/mod.rs:70-134` — rewrite `SessionPool`, introduce
`PoolGuard`, `OwnedReservation`, `PoolError`. Delete `available:
AtomicUsize`, delete `blocking_checkin`.

Call sites (seven total):
| Site | Before | After |
|---|---|---|
| `src/server/mod.rs:476-504` WS | checkout+checkin around `handle_ws_inner` | checkout returns guard; guard drop handles checkin |
| `src/server/http.rs:176-237` REST | checkout + spawn_blocking + checkin two-branch | `guard.into_owned()` → move both; `reservation.checkin(triplet)` |
| `src/server/http.rs:276-328` SSE | checkout + spawn_blocking + `blocking_checkin` | same `into_owned` + `reservation.checkin` |
| `src/main.rs:347-349` CLI `transcribe` | explicit checkout/checkin | guard scope |
| `tests/benchmark.rs:307` | `rt.block_on(pool.checkout())` | `let guard = rt.block_on(pool.checkout())?;` |

New error paths: 503 `pool_closed` (REST), `ServerMessage::Error { code:
"pool_closed", retry_after_ms: None }` (WS). Distinct from `timeout`
(`retry_after_ms: Some(30_000)`).

### B.4 Metrics (bridge to V1-30)
Add now: `gigastt_pool_available` gauge, `gigastt_pool_closed_total`
counter. Defer the full RED suite to V1-30.

### B.5 Migration

Atomic rewrite; no feature flag. Rationale: internal primitive, 7 call
sites, two commits:
1. Introduce `async_channel`, `PoolGuard`, `PoolError`; adapt all call
   sites; adapt tests.
2. Wire `engine.pool.close()` into the graceful-shutdown hook from
   V1-03/V1-04.

`CHANGELOG.md [Unreleased] / Changed`: "`SessionPool::{checkout, checkin,
blocking_checkin}` replaced by `SessionPool::checkout() -> Result<PoolGuard,
PoolError>`. The guard `Deref`s to `SessionTriplet` and auto-checks-in
on drop."

### B.6 Edge cases

| Scenario | Before | After |
|---|---|---|
| Panic between checkout and scope end | triplet lost unless `catch_unwind` wrapped the call site | `Drop` on `PoolGuard` restores capacity; `catch_unwind` is kept only for inference panics, which is its real purpose |
| Shutdown while pool saturated | `expect("Pool sender dropped")` → 500 cascade | `sender.close()` → waiters resolve to `PoolError::Closed` → 503 `pool_closed` |
| All triplets out + sender dropped | waiters wait forever | `close()` releases them |
| Inference panic inside triplet (broken LSTM state) | triplet back in pool, possibly carrying garbage next call | Same — orthogonal to V1-07. Future work: mark "poisoned" and reload. |

### B.7 Tests

Unit (`src/inference/mod.rs`):
- `test_pool_guard_returns_triplet_on_normal_drop`
- `test_pool_guard_returns_triplet_on_panic_unwind`
- `test_pool_close_wakes_waiters_with_closed`
- `test_pool_fifo_under_contention`
- `test_into_owned_for_spawn_blocking`

Stress (`tests/load_test.rs` or new `tests/pool_stress.rs`):
- `pool_stress_10x_concurrent_no_leak` — 10 × `pool_size` tasks, assert
  `pool.available() == pool.total()` after completion.

Shutdown (`tests/e2e_shutdown.rs`):
- `test_shutdown_during_pool_saturation_returns_503_not_500` — occupy the
  pool, queue a waiter, fire shutdown → waiter receives 503
  `pool_closed`, no 500s in logs.

### B.8 Perf

Micro-bench optional. Expectation: `async_channel` ~100–300 ns vs
`Mutex<mpsc>` ~400–1 000 ns per checkout/checkin cycle. Correctness is the
primary goal; performance is a bonus.

### B.9 Rollback

`git revert` is the plan; fire a `0.9.1` patch. If extra safety desired,
ship `0.9.0-rc.1` with the same changes and a 2-week soak.

### B.10 Effort

| # | Scope | Hours |
|---|---|---|
| 1 | Add `async_channel`, rewrite `SessionPool` / `PoolGuard`; adapt call sites | 3–4 |
| 2 | Unit + stress + shutdown tests; wire `pool.close()` into shutdown hook | 2–3 |
| 3 | CHANGELOG, `CLAUDE.md` update, `cargo fmt`/`clippy`/`test` | 1 |
| 4 (V1-06) | Rate-limiter formula + unit test + e2e regression + CHANGELOG | 1–2 |

Total ≈ 7–10 h; single PR day.

### B.11 Open questions

1. Atomic V1-07 + V1-21 vs. separate PRs? Combined recommended — `Guard`
   *is* the mechanism of V1-07.
2. `async-channel` vs. `Semaphore + VecDeque`? Prototype both first 1–2 h;
   pick the cleaner implementation.
3. `Engine::pool: pub` stays `pub`; internal-embedder breakage is
   acceptable and documented.
4. Wire `pool.close()` into `run_with_config` shutdown: after
   `shutdown_fut` resolves but before `axum::serve` returns (or as part of
   the new `shutdown_signal` from the V1-03 design).

## References

- `src/server/mod.rs:277-304` — rate-limiter build (V1-06) + thread leak
  (V1-15, separate item).
- `src/server/mod.rs:475-507` — WS checkout/checkin.
- `src/inference/mod.rs:70-134` — `SessionPool`, `Mutex<Receiver>`,
  `AtomicUsize available`, three `.expect("Pool sender dropped")`.
- `src/server/http.rs:176-327` — REST + SSE call sites.
- `src/main.rs:74-76` — clap definition for `--rate-limit-per-minute`.
- `src/main.rs:347-349` — CLI `transcribe` direct checkout/checkin.
- `tests/benchmark.rs:307` — `rt.block_on(pool.checkout())`.
- `Cargo.toml:62` — `tower_governor 0.7` (`per_millisecond` available).
- `Cargo.lock` — `async-channel` not present; transitive pieces are.
- `specs/prod-readiness-v1.0.md` — V1-06, V1-07, V1-21.
- `specs/design-v1.0-ws-lifecycle.md` — graceful shutdown plan (consumes
  `pool.close()`).
