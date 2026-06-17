# Runbook

Operator-facing guidance for gigastt in production: graceful shutdown, session caps, pool exhaustion / backpressure, inference timeouts, model-download failures, and out-of-memory — with the knobs and escape hatches for each.

## At a glance

| Symptom | First check | Escape hatch |
|---|---|---|
| Clients lose `Final` on deploy | Drain window too short: check `shutdown_drain_secs` vs your orchestrator's grace period | Increase `GIGASTT_SHUTDOWN_DRAIN_SECS` OR disable WS tracking via `--shutdown-drain-secs 0` (clamped to 1 s) |
| Clients receive spurious `max_session_duration_exceeded` | Legitimate long sessions | Raise `GIGASTT_MAX_SESSION_SECS` (default 3600) or set `0` to disable |
| SIGTERM takes 30+ seconds to exit | In-flight spawn_blocking inferences can't be cancelled mid-chunk | Wait or lower `GIGASTT_SHUTDOWN_DRAIN_SECS`; process will still finish the current chunk |
| `Close(1008 Policy Violation)` unexpected | session-duration cap fired | Double check `max_session_secs` is set high enough for your use case |
| `Close(1001 Going Away)` seen by clients | Expected on SIGTERM — not a bug | None — clients should reconnect |
| REST `503` `timeout` / WS error `timeout` (`retry_after_ms`) | Pool saturated — every triplet busy | Raise `--pool-size`; isolate batch with `--batch-pool-size`; see [Pool exhaustion](#pool-exhaustion--backpressure) |
| `inference_timeout` (REST `504` / WS close) | A single run exceeded `--inference-timeout-secs` | Raise it for genuinely long files; investigate a wedged ONNX run |
| Server won't start, model errors | Missing / corrupt model files | See [Model download failures](#model-download-failures) |
| OOM / pod killed | Pool RSS exceeds the box | Lower `--pool-size`, use the INT8 encoder, `--pool-min-size` to boot degraded — see [Out-of-memory](#out-of-memory-oom) |

## Graceful drain (SIGTERM)

When the server receives `SIGTERM` (or the `run_with_shutdown` oneshot fires):

1. A process-wide `CancellationToken` is cancelled.
2. Every live `handle_ws_inner` session sees `cancel.cancelled()` in its `biased;` select loop, flushes its streaming state, emits a (possibly empty) `Final`, and closes with `Close(1001 Going Away)`.
3. SSE `/v1/transcribe/stream` tasks check the token between chunks and drop the channel sender, which terminates the SSE stream from the client's perspective.
4. After `axum::serve` returns, the main task waits up to `shutdown_drain_secs` seconds for the `TaskTracker` to report all tracked WS / SSE futures complete.
5. If the drain window expires with tracked tasks still running, a WARN is emitted (`Drain window expired with tracked tasks still running`) and the process exits anyway.

### Rollback: disable graceful drain

If v0.9.0 rollout breaks WS clients, the runtime supports a tiered rollback:

1. **Shrink the drain window to 1 s** (effectively disabling the wait):
   ```sh
   gigastt serve --shutdown-drain-secs 0
   # or: GIGASTT_SHUTDOWN_DRAIN_SECS=0 gigastt serve
   ```
   Note: `0` is internally clamped to `1` second. The cancel + Final path still fires, but the process won't wait longer than 1 s before exiting.

2. **Disable the session cap independently** (see the section below).

3. **Git revert** — v0.9.0's WS-lifecycle work lives in one PR and reverts cleanly. Only use if options 1-2 are insufficient; you'll need to re-cut the release.

## Max session duration

`idle_timeout` is reset on every frame, so a silence-streaming client could hold a `SessionTriplet` forever. `max_session_secs` is a *wall-clock* deadline that fires regardless of frame activity.

On cap expiry the server sends:
1. `ServerMessage::Error { message: "Maximum session duration exceeded", code: "max_session_duration_exceeded" }`
2. A best-effort `Final` frame (empty if no text accumulated).
3. `Close(1008 Policy Violation)`.

Overshoot ≤ 500 ms in the common case — a chunk that was already in flight when the deadline expired finishes first, then the loop hits the deadline branch on the next iteration.

### Rollback: disable the session cap

```sh
gigastt serve --max-session-secs 0
# or: GIGASTT_MAX_SESSION_SECS=0 gigastt serve
```

`0` parks the deadline at `u64::MAX / 2`, so `sleep_until` never fires. The session then runs as long as the idle timeout allows (default 300 s of silence).

### Config pitfalls

- If you set `--max-session-secs` *below* `--idle-timeout-secs`, the cap will always fire before the idle timer can apply. The server emits a `warn` at startup flagging this as a likely misconfiguration but does not refuse to start.
- Caps smaller than your typical transcription window will produce noisy `max_session_duration_exceeded` errors for legitimate clients.

## Pool exhaustion & backpressure

Each concurrent inference holds one `SessionTriplet` from a pool sized by
`--pool-size` (default 4). When all triplets are busy, callers wait up to
`--pool-checkout-timeout-secs` (default 30) for one to free up, then get
backpressure:

- **REST** `/v1/transcribe` and `/v1/transcribe/stream` → `503` with
  `Retry-After: <secs>` and `{"code":"timeout","retry_after_ms":…}`.
- **WebSocket** → `ServerMessage::Error { code: "timeout", retry_after_ms }`.

A single wedged inference run is bounded by `--inference-timeout-secs`
(default 60): the client gets `inference_timeout` (REST `504`, WS error +
close). Caveat — `spawn_blocking` can't be cancelled, so a truly hung run keeps
its triplet until it finishes (or restart); the timeout frees the *client*,
not the slot.

**Knobs**
- `--pool-size N` — total triplets (more concurrency, more RAM).
- `--batch-pool-size N` — reserve N triplets for long REST file jobs so they
  can't starve interactive WebSocket / SSE streaming (default 0 = shared pool).
- `--pool-checkout-timeout-secs` — how long callers wait before backpressure.
- `--inference-timeout-secs` — per-run ceiling; `0` disables.

**Metrics** (with `--metrics`)
- `gigastt_pool_available` / `gigastt_pool_waiters` — free triplets vs queued
  callers. Sustained `available == 0` with rising `waiters` = saturation.
- `gigastt_pool_timeouts_total` — checkout timeouts (backpressure events).
- `gigastt_inference_timeouts_total` — runs aborted by the inference timeout
  (a non-zero rate points at wedged runs or an over-tight timeout).

**Triage**
1. If streaming is being starved by batch uploads, set `--batch-pool-size 1+`.
2. If `gigastt_inference_timeouts_total` is climbing, capture a stuck run's
   input and check for an adversarial / huge file; raise the timeout only if the
   inputs are legitimately long.
3. If saturation is steady, scale `--pool-size` (watch RSS) or add replicas.

## Model download failures

On first run `gigastt download` / `gigastt serve` fetches the model from
HuggingFace into `~/.gigastt/models/`, streaming each file to a `.partial`
path, verifying SHA-256, then atomically renaming. Concurrent processes
coordinate via an advisory `flock`; downloads use connect/read timeouts and a
bounded redirect policy.

**Symptoms & recovery**
- *SHA-256 mismatch* — a corrupt or tampered mirror. The `.partial` is deleted
  and nothing is promoted; just re-run. Persistent mismatches mean a bad mirror
  or a stale pinned checksum.
- *Hang / timeout mid-download* — network or HuggingFace issue; re-run (the
  `.partial` is re-fetched, not resumed).
- *"Model not found" after a crash* — a crash before rename leaves only a
  `.partial`; `model_files_exist()` ignores it, so re-running re-downloads.
- *Air-gapped / repeatable deploys* — bake the model into the image
  (`GIGASTT_BAKE_MODEL=1`, see `docs/deployment.md`) or pre-populate
  `~/.gigastt/models/` from a trusted copy.

To force a clean re-download, remove `~/.gigastt/models/` and re-run.

## Out-of-memory (OOM)

RSS scales with `--pool-size` (each triplet owns its ONNX sessions) plus ORT's
per-request scratch (a few minutes of 16 kHz audio allocates ~90+ MiB in the
encoder by itself). A default pool of 4 with the INT8 encoder sits around
~560 MiB; the FP32 encoder is much larger.

**Reduce footprint**
- Use the **INT8 encoder** (the default — auto-quantized on first run; don't
  pass `--skip-quantize`). It is ~4× smaller than FP32.
- Lower `--pool-size` (e.g. `1`–`2` on a 4 GB box).
- `--pool-min-size 1` lets the server **boot on a degraded pool** if some
  triplets fail to load under memory pressure, instead of failing outright.
- The REST upload path is zero-copy (`bytes::Bytes` end-to-end), so concurrent
  large uploads no longer multiply the body in RAM — but the decoded PCM and
  encoder scratch still scale with audio length and `--pool-size`.

**Triage**
1. Check `terminationGracePeriodSeconds` isn't masking an OOM-kill as a slow
   shutdown.
2. Confirm the INT8 encoder is in use (`/v1/models` reports `"encoder":"int8"`).
3. Cap concurrency: `--pool-size` × (per-triplet RSS + peak scratch) must fit
   the box with headroom.

## Metrics

`gigastt_http_requests_total{path="/v1/ws",status="503"}` with code `shutting_down` in the body is the signal that upgrades are being rejected because shutdown was already in flight. Usually correlated with `terminationGracePeriodSeconds` being shorter than `shutdown_drain_secs`.

(Counter for cancelled-WS by reason is tracked separately — see `specs/prod-readiness-v1.0.md`.)

## On-call triage checklist

1. Pull a WS trace from the affected client. Confirm presence (or absence) of `Final` and the `Close` code.
2. Check server logs for `Shutdown signalled`, `Session cap reached`, or `Drain window expired`.
3. Confirm orchestrator `terminationGracePeriodSeconds` ≥ `shutdown_drain_secs + 5` (see `docs/deployment.md`).
4. If clients are seeing unexpected 503 `shutting_down`, the proxy LB may still be routing traffic after the pod started draining — add a `preStop` sleep to the k8s manifest so the LB deregisters the pod before the app sees `SIGTERM`.
5. If the cap is firing for legitimate long sessions, raise it — there's no correctness downside to `max_session_secs = 14400` (4 h), only a weaker guarantee against wedged sessions.
