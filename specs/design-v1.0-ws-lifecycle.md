# Design: V1-03 + V1-04 — WebSocket lifecycle (graceful drain + session cap)

Owner: architect-1 (2026-04-20). Covers the two P0 findings that share the
WebSocket handler. Target tag: `v0.9.0`. Implements items **V1-03** (graceful
WS drain on SIGTERM) and **V1-04** (max session duration cap) from
`specs/prod-readiness-v1.0.md`.

## 1. Architecture

**Core idea.** `axum::serve(...).with_graceful_shutdown(fut)` only tracks
the HTTP router; WebSocket upgrades become untracked background tasks. We
need to (a) own a shutdown signal ourselves, (b) deliver it to every active
`handle_ws_inner`, and (c) bound every session by a wall-clock deadline
independent of `idle_timeout`.

### Additions / changes

1. **`RuntimeLimits`** (`src/server/mod.rs:124-138`): two new fields.
   ```rust
   pub max_session_secs: u64,     // default 3600; 0 disables
   pub shutdown_drain_secs: u64,  // default 10
   ```
2. **`AppState`** (`src/server/http.rs:22-26`): carries the shutdown token
   and a tracker for spawned WS/SSE tasks.
   ```rust
   pub shutdown: tokio_util::sync::CancellationToken,
   pub tracker:  tokio_util::task::TaskTracker,
   ```
3. **`Cargo.toml`**: add `tokio-util = { version = "0.7", features = ["rt"] }`.
4. **`StreamingState`** unchanged; `session_deadline` is a local in
   `handle_ws_inner`.

### Signal flow

```
Ctrl-C / oneshot<()>        <- run_with_config
   shutdown_root: CancellationToken   <- AppState
       every ws_handler / SSE handler  <- cancel_token clone
           handle_ws_inner loop: tokio::select! on cancel + deadline + frame
```

After `axum::serve(...).await`:
`tokio::time::timeout(shutdown_drain_secs, tracker.wait())` — drain window
for still-live WS/SSE tasks.

## 2. File-level edits

### `Cargo.toml`
Add `tokio-util = { version = "0.7", features = ["rt"] }`.

### `src/server/mod.rs`
- **`RuntimeLimits`** (124-150): two new fields + defaults above.
- **`run_with_config`** (218-353): build `shutdown_root` + `tracker`;
  inject into `AppState`; new `shutdown_signal` closes them; after axum
  returns, `tokio::time::timeout(drain_secs.max(1), tracker.wait())`; warn
  on expiry.
- **`ws_handler` / `ws_handler_legacy`** (444-473): clone `cancel` from
  state; if already cancelled, return 503 `shutting_down` without upgrade;
  wrap `handle_ws` in `tracker.track_future(...)`.
- **`handle_ws`** (475-507): `select!` cancel vs. `pool.checkout()` so a
  shutdown does not wait for the existing 30 s checkout timeout.
- **`handle_ws_inner`** (745-846): rewritten `select!` loop — see §3.

### `src/server/http.rs`
- **`AppState`**: two new fields above.
- **`transcribe_stream`** (243-351): pass `cancel` into the
  `spawn_blocking` closure; check `cancel.is_cancelled()` before each
  `blocking_send`.

### `src/main.rs`
- **`Commands::Serve`**: two new CLI flags.
  ```rust
  /// 0 = no cap (not recommended).
  #[arg(long, env = "GIGASTT_MAX_SESSION_SECS", default_value_t = 3600)]
  max_session_secs: u64,
  #[arg(long, env = "GIGASTT_SHUTDOWN_DRAIN_SECS", default_value_t = 10)]
  shutdown_drain_secs: u64,
  ```
- Plumb them into `RuntimeLimits` at 283-289.

### `src/protocol/mod.rs`
No new `ServerMessage` variants. Two new **string** error codes:
`"max_session_duration_exceeded"`, `"shutting_down"`. AsyncAPI update is
tracked separately in V1-33.

## 3. `handle_ws_inner` skeleton

```rust
let idle_timeout = Duration::from_secs(limits.idle_timeout_secs);
let session_deadline = if limits.max_session_secs == 0 {
    tokio::time::Instant::now() + Duration::from_secs(u64::MAX / 2)
} else {
    tokio::time::Instant::now() + Duration::from_secs(limits.max_session_secs)
};

let result: Result<()> = loop {
    tokio::select! {
        biased; // cancel > deadline > frame

        _ = cancel.cancelled() => {
            let _ = flush_and_final(&mut sink, engine, &mut state_opt).await;
            let _ = sink.send(WsMessage::Close(Some(CloseFrame {
                code: 1001, reason: "server shutdown".into(),
            }))).await;
            break Ok(());
        }

        _ = tokio::time::sleep_until(session_deadline) => {
            let _ = send_server_message(&mut sink, &ServerMessage::Error {
                message: "Maximum session duration exceeded".into(),
                code:    "max_session_duration_exceeded".into(),
                retry_after_ms: None,
            }).await;
            let _ = flush_and_final(&mut sink, engine, &mut state_opt).await;
            let _ = sink.send(WsMessage::Close(Some(CloseFrame {
                code: 1008, reason: "max session duration".into(),
            }))).await;
            break Ok(());
        }

        maybe_msg = tokio::time::timeout(idle_timeout, source.next()) => {
            // existing frame dispatch unchanged
        }
    }
};
(triplet_opt, result)
```

`flush_and_final` always emits a `Final` (empty if nothing accumulated) so
tests can assert it. `biased;` guarantees cancel wins any race.

## 4. Edge cases & invariants

| Scenario | Behaviour |
|---|---|
| Cancel arrives while `spawn_blocking` holds the triplet | Current chunk finishes (< 1 s typical). Next loop iteration hits the cancel branch. Triplet returns to pool via the normal `handle_ws` path. Drain window (10 s) ≫ single chunk. |
| Client already closed socket at shutdown | `sink.send` → `Err` is swallowed (`let _ =`); `biased` keeps cancel priority. |
| Panic in `handle_binary_frame` | `catch_unwind` path unchanged; loop continues; cancel still works. |
| SSE `/v1/transcribe/stream` | axum returns handler future early → must poll `cancel.is_cancelled()` before each `blocking_send` inside the `spawn_blocking`. |
| `max_session_secs < idle_timeout_secs` | Not rejected; `run_with_config` emits a `warn!`. |
| Client streaming mid-speech on shutdown | No extra grace — shared `shutdown_drain_secs` only. |
| `shutdown_drain_secs = 0` | Clamp to 1 s (`.max(1)`). |
| Mid-flight inference at session cap | Overshoot ≤ 500 ms; acceptable for v1.0. |

## 5. Tests

### Re-enable `tests/e2e_shutdown.rs` in CI
- Existing liveness tests stay.
- New `test_shutdown_ws_emits_final_and_close` — asserts `Final` arrives and
  `Close(1001)` frame is sent before socket EOF.
- New `test_shutdown_sse_stream_terminates_cleanly` — SSE ends within drain
  window, not by timeout.
- Update `.github/workflows/ci.yml` main-push e2e job:
  ```yaml
  - run: cargo test --test e2e_rest --test e2e_ws --test e2e_errors --test e2e_shutdown -- --ignored --test-threads=1
  ```

### New session-cap test
`test_max_session_duration_cap` — server with `max_session_secs=3`, client
streams 20 ms silence every 100 ms for 5 s; expects `error{code:
max_session_duration_exceeded}` + `Close(1008)` within `cap + 2 s`.

Requires a new helper:
```rust
pub async fn start_server_with_limits(model_dir: &str, limits: RuntimeLimits)
    -> (u16, oneshot::Sender<()>);
```

## 6. Migration of existing tests
`e2e_ws.rs`, `e2e_errors.rs`, `e2e_rest.rs` — default `max_session_secs=3600`
covers them. `soak_test.rs` rotates WS clients, each short — unaffected.
`server_integration.rs` is legacy (V1-23, slated for deletion). Any literal
`RuntimeLimits { … }` in the codebase must be updated with the two new
fields; internal call sites are in `src/main.rs` and `tests/common/mod.rs`.

## 7. Rollback plan

Tiered:
1. **Env kill switch.** `GIGASTT_SHUTDOWN_DRAIN=legacy` skips
   `tracker.track_future`, the cancel branch, and the drain wait;
   `--max-session-secs 0` disables V1-04 separately.
2. **CLI override.** `--shutdown-drain-secs 0` + `--max-session-secs 0`.
3. **Git revert** — one focused PR reverts cleanly.

Document in `docs/runbook.md`: "If rollout breaks WS clients, set both
values to 0 and restart".

## 8. Effort breakdown (commits)

| # | Scope | Hours |
|---|---|---|
| 1 | `RuntimeLimits`/`AppState`/CLI scaffolding + `tokio-util` dep | 1.5 |
| 2 | V1-03 graceful drain (handler, tracker, drain wait, SSE cancel check) + re-enable `e2e_shutdown.rs` in CI | 3–4 |
| 3 | V1-04 session cap (deadline branch, new test, warn on config pitfall) | 1.5 |
| 4 | Docs: `CHANGELOG.md`, `docs/runbook.md`, `docs/deployment.md` (`terminationGracePeriodSeconds`) | 0.5 |

Total ≈ 6–7 h pure coding; one PR day.

## 9. Open questions for the user

1. `shutdown_drain_secs = 10` s default — raise to 20 s so the k8s default
   `terminationGracePeriodSeconds = 30` comfortably covers drain + buffer?
2. Close codes: 1001 (Going Away) for shutdown, 1008 (Policy Violation) for
   session cap — confirm semantics.
3. `RuntimeLimits` gains two `pub` fields → 0.8.1 → 0.9.0 minor bump
   (acceptable pre-1.0) vs. hiding behind a builder.
4. Should `ws_handler_legacy` (`/ws`) also be tracked by `TaskTracker`?
   Yes — otherwise legacy clients lose their `Final` on shutdown.
5. Metric `gigastt_ws_cancelled_total{reason="shutdown|session_cap"}` —
   add here or defer to V1-30?

## 10. References

- `src/server/mod.rs:331-353` — current shutdown path (HTTP only).
- `src/server/mod.rs:475-507` — `handle_ws` spawn, presently untracked.
- `src/server/mod.rs:745-846` — `handle_ws_inner` loop (rewrite target).
- `src/server/mod.rs:779-789` — `timeout(idle_timeout, source.next())` —
  root cause of V1-04.
- `src/server/http.rs:22-26` — `AppState`.
- `src/server/http.rs:243-328` — SSE `transcribe_stream` (needs cancel
  check in `spawn_blocking`).
- `src/protocol/mod.rs:52-62` — `ServerMessage::Error` (no new variants
  required).
- `src/main.rs:246-292` — `Commands::Serve` (two new flags).
- `tests/e2e_shutdown.rs` — currently excluded from CI; reactivate + extend.
- `tests/common/mod.rs:52-66` — add `start_server_with_limits`.
- `specs/prod-readiness-v1.0.md` — V1-03, V1-04.
