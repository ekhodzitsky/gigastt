# Runbook

Operator-facing guidance for gigastt in production: graceful shutdown, session caps, pool exhaustion / backpressure, inference timeouts, model-download failures, and out-of-memory — with the knobs and escape hatches for each.

## At a glance

| Symptom | First check | Escape hatch |
|---|---|---|
| Clients lose `Final` on deploy | Drain window too short: check `shutdown_drain_secs` vs your orchestrator's grace period | Increase `GIGASTT_SHUTDOWN_DRAIN_SECS` OR disable WS tracking via `--shutdown-drain-secs 0` (clamped to 1 s) |
| Clients receive spurious `max_session_duration_exceeded` | Legitimate long sessions | Raise `GIGASTT_MAX_SESSION_SECS` (default 3600) or set `0` to disable |
| SIGTERM takes 30+ seconds to exit | A native encoder call may still be running | Cancellation is checked before encoding and between decode steps; an encoder call already in progress must return first |
| `Close(1008 Policy Violation)` unexpected | session-duration cap fired | Double check `max_session_secs` is set high enough for your use case |
| `Close(1001 Going Away)` seen by clients | Expected on SIGTERM — not a bug | None — clients should reconnect |
| WARN `optimized graph cache directory cannot be created / is not writable` | Cache dir missing or unwritable (e.g. read-only model dir) | Service still works — see [ORT cache warnings](#ort-optimized-graph-cache-warnings) |
| REST `503` `timeout` / WS error `timeout` (`retry_after_ms`) | Pool saturated — every triplet busy | Raise `--pool-size`; isolate batch with `--batch-pool-size`; see [Pool exhaustion](#pool-exhaustion--backpressure) |
| `inference_timeout` (REST `504` / WS close) | A run made no progress for `--inference-timeout-secs` (default 600 s) | Not a length limit — the deadline resets on every decode window, so long files never trip it. Investigate a wedged ONNX run |
| Server won't start, model errors | Missing / corrupt model files | See [Model download failures](#model-download-failures) |
| OOM / pod killed | Pool RSS exceeds the box | Lower `--pool-size`, use the INT8 encoder, `--pool-min-size` to boot degraded — see [Out-of-memory](#out-of-memory-oom) |
| Long file must be sent again after a cancel or restart | Jobs are not stored on disk, and one upload is capped at `--body-limit-bytes` (default 50 MiB) | Chunk under the cap, or raise it with `--jobs-max-bytes`. See [Long recordings](long-recordings.md) |

## Graceful drain (SIGTERM)

When the server receives `SIGTERM` (or the `run_with_shutdown` oneshot fires):

1. A process-wide `CancellationToken` is cancelled.
2. Every live WebSocket session observes shutdown even while waiting for a blocking decode. An in-flight run receives the shared abort flag; the client receives its last available `Partial`, a `cancelled` error, `Final`, and `Close(1001 Going Away)`. Idle sessions flush and close as before.
3. SSE producers also link shutdown and receiver disconnect to their decode abort flag. Cancellation ends the stream; any partial already received remains usable. A full output queue is interrupted immediately by shutdown or disconnect; terminal events are best effort and never delay shutdown.
4. After `axum::serve` returns, the main task waits up to `shutdown_drain_secs` seconds for the `TaskTracker` to report all tracked WS / SSE futures complete.
5. If the drain window expires with tracked tasks still running, a WARN is emitted (`Drain window expired with tracked tasks still running`) and the process exits anyway.

File-stream output backpressure (`/v1/transcribe/stream` and OpenAI
`stream=true`) has a separate **30-second send limit**. The native stream keeps
at most 16 queued events and the OpenAI stream at most 32; events remain in FIFO
order and are not coalesced. If a full queue cannot accept the next event within
30 seconds, only that transcription is cancelled and its inference reservation
is released. Pending final/error events may be omitted; OpenAI `[DONE]` is never
queued after a failed preceding completion event. Already queued events can
still drain if the reader resumes.

File streams also apply `--inference-timeout-secs` to the initial container
probe and to processing without a completed chunk (`0` disables this timeout).
The watchdog samples every 100 ms; runtime scheduling can add delay. A pending
output send suspends the inference timer and retains the separate 30-second
backpressure limit. Progress restarts the processing deadline.

A probe timeout returns HTTP 504. After SSE headers have been sent, timeout
emits one `inference_timeout` error and closes the response, independently of
the blocking worker. Native SSE first emits the latest available partial;
OpenAI includes an optional `partial` snapshot in its error object. Buffered
ordinary events are discarded on timeout and no terminal success or `[DONE]`
is emitted. A completed request that already claimed terminal delivery instead
finishes under the output-send bound.

Timeout requests cooperative cancellation: it cannot interrupt an active codec
or native inference call. The tracked worker retains its pool reservation and
upload admission guard until it actually exits, including during probing. A
client disconnect likewise requests cancellation without returning an in-use
session to the pool.

### Rollback: disable graceful drain

If a release breaks WS clients, the runtime supports a tiered rollback:

1. **Shrink the drain window to 1 s** (effectively disabling the wait):
   ```sh
   gigastt serve --shutdown-drain-secs 0
   # or: GIGASTT_SHUTDOWN_DRAIN_SECS=0 gigastt serve
   ```
   Note: `0` is internally clamped to `1` second. The cancel + Final path still fires, but the process won't wait longer than 1 s before exiting.

2. **Disable the session cap independently** (see the section below).

3. **Roll back the binary** to the previous release tag (do not invent an
   in-tree revert of ancient WS-lifecycle work). Re-cut only if you must.

## Max session duration

`idle_timeout` is reset on every frame, so a silence-streaming client could hold a `SessionTriplet` forever. `max_session_secs` is a *wall-clock* deadline that fires regardless of frame activity.

On cap expiry the server sends:
1. `ServerMessage::Error { message: "Maximum session duration exceeded", code: "max_session_duration_exceeded" }`
2. A best-effort `Final` frame (empty if no text accumulated).
3. `Close(1008 Policy Violation)`.

The session deadline is also observed while decoding and finalizing. It closes
the client connection and requests cooperative cancellation without waiting for
the current encoder call to return.

### Rollback: disable the session cap

```sh
gigastt serve --max-session-secs 0
# or: GIGASTT_MAX_SESSION_SECS=0 gigastt serve
```

`0` parks the deadline ≈30 years in the future, so `sleep_until` never fires. The session then runs as long as the idle timeout allows (default 300 s of silence).

### Config pitfalls

- If you set `--max-session-secs` *below* `--idle-timeout-secs`, the cap will always fire before the idle timer can apply. The server emits a `warn` at startup flagging this as a likely misconfiguration but does not refuse to start.
- Caps smaller than your typical transcription window will produce noisy `max_session_duration_exceeded` errors for legitimate clients.

## Pool exhaustion & backpressure

Each concurrent inference holds one `SessionTriplet` from a pool sized by
`--pool-size` (default 2). When all triplets are busy, callers wait up to
`--pool-checkout-timeout-secs` (default 30) for one to free up, then get
backpressure:

- **REST** `/v1/transcribe` and `/v1/transcribe/stream` → `503` with
  `Retry-After: <secs>` and `{"code":"timeout","retry_after_ms":…}`.
- **WebSocket** → `ServerMessage::Error { code: "timeout", retry_after_ms }`.

**Checkout timeout is the queue-vs-fail-fast knob.** A longer value keeps
callers waiting in line (absorb short saturation bursts); a shorter value
returns **503 / `timeout` + `retry_after_ms` sooner** so clients can back off or
retry another replica. See [Pool checkout timeout](#pool-checkout-timeout-queue-vs-fail-fast).

A wedged inference run is bounded by `--inference-timeout-secs` (default 600):
the client gets `inference_timeout` (REST `504`, WS error + close).

For REST and jobs, this is a **no-progress watchdog**. The deadline
resets every time a decode window completes, so a file that keeps making
progress never trips it no matter how long it is — do not raise this value
"for long files". Audio length is governed by `--max-audio-secs` (default
`0` = unlimited) instead. Split-channel decoding accumulates work across
channels, so starting the next channel does not reset the watchdog. Progress is
sampled every 100 ms. With prompt runtime scheduling, a stalled run is detected
within the timeout plus at most 100 ms after its last progress update.

WebSocket applies the timeout to each chunk decode and the Stop/finalize decode.
On timeout, disconnect, job DELETE, or shutdown, the same per-run abort flag
is observed before encoding and between RNN-T tokens or CTC frames (including
beam search). The blocking worker returns its triplet when it observes abort.

**Mid-encoder abort is unsupported.** No runtime termination hook is used;
a native encoder call already in progress must return before the flag can be
observed. A wedged native call can therefore retain its slot after the client
has received the timeout. There is no fixed cancellation-latency guarantee.

### Cancellation boundaries and resource ownership

A cancellation flag prevents later work when the worker next checks it. It is
not a thread kill. HTTP timeout closes the response independently of worker
termination; disconnect and shutdown request the same cooperative stop.

| Stage | Cancellation boundary | Work that must finish before the next check |
|---|---|---|
| Source probing and raw telephony conversion | Before preparation, after probe/decode, and before invoking recognition | Active container probe, raw codec call, or WAV encoding |
| Split-channel scan and buffered decode | Before each container packet and after codec calls; between resampler blocks and before final drain | Active codec/resampler call; Opus full-buffer correlation runs synchronously between checks |
| Flat mono decode for offline diarization | Between container packets or WAVE blocks, preserving the normal mixing/resampling order | Active packet decode or WAVE block receive; dropping the WAVE source joins its decoder worker |
| Windowed file decode and VAD | Between requested windows; streaming VAD polls every 16 two-second blocks, buffered VAD every 64 frames (about two seconds of audio) | Active source fill or VAD call; audio duration between checks is not elapsed wall time |
| Recognition | Before encoding and between decoder tokens/frames; after completed recognition | Active feature extraction or native encoder/decoder call |
| Lazy speaker model loading | Before loading and immediately after it returns | Concurrent load lock wait and native model construction |
| Offline speaker diarization | Before/after each VAD and embedding call; after the pipeline returns and before speaker assignment | Active embedding call and the pipeline's synchronous clustering/assembly stage |
| Speaker assignment and text postprocessing | Before/after assignment; before ITN, between ITN and punctuation, and after punctuation | Active assignment, ITN pass, or punctuation restoration call |
| Response output | Disconnect/shutdown and bounded output-send checks | A send may wait up to the separate 30-second backpressure bound |

Packet/block regression tests cancel after a known checkpoint and verify that
no later checkpoint or inference callback is reached. Diarization adapter tests
cancel inside one embedding call and verify that the next embedding never
starts. These are cooperative step bounds, not maximum wall-clock latency:
container/native calls, speaker loading, clustering, and decoder-thread joins
have no interrupt hook. A permanently stuck call can retain resources until
process termination.

The detached blocking worker owns its inference reservation throughout source
preparation, recognition, diarization, and finalization. It also keeps the
upload admission guard when encoded bytes are replaced by PCM or a raw-codec
WAV. Stream probe workers retain these owners even if the async handler is
dropped. Reservations and admitted upload bytes become available only when the
owning worker exits and drops them, which may be later than the client timeout.
Do not interpret a completed error response as proof that pool capacity has
already recovered.

A file-stream timeout uses the latest available provisional snapshot. Successfully
finalized segments whose snapshots were already cleared, and native results
produced after timeout, are not reconstructed into that error response.
Previously delivered text remains readable by the client.

The latest provisional text survives interruption. REST errors may include
`partial`; WebSocket sends its last available `partial` before an error when
the socket remains writable; jobs retain `partial` in their status response,
including after a late blocking worker finishes. Cancelled jobs stay cancelled,
and timeout failures are not retried. See [cancellation semantics](api.md).

For embedded Rust callers, attach a `TranscriptSnapshot` with
`TranscribeRequest::with_partial` or `StreamingState.partial`, alongside the
shared `AtomicBool` abort flag. After `GigasttError::Cancelled`, the snapshot
and streaming assembler remain readable. `StreamingState::is_failed()` is
true and further feed/finalize calls cannot restart decoding; create a fresh
state for a new stream.

**Knobs**
- `--pool-size N` — total triplets (more concurrency, more RAM, and a small
  single-job RTF cost from thread split — see [Pool size tradeoffs](#pool-size-tradeoffs-ram-vs-concurrency-vs-rtf)).
- `--batch-pool-size N` — **split** N of those triplets for long REST file jobs
  so they can't starve interactive WebSocket / SSE (default 0 = shared pool).
  Not additive — total loaded sessions stay at `--pool-size` (see
  [batch_pool_size splits the pool](#batch_pool_size-splits-the-pool-not-additive)).
- `--pool-checkout-timeout-secs` — how long callers wait before backpressure
  (long = queue, short = fail-fast 503).
- `--inference-timeout-secs` — per-run ceiling; `0` disables.

**Metrics** (with `--metrics`)
- `gigastt_pool_available` / `gigastt_pool_waiters` — free triplets vs queued
  callers. Sustained `available == 0` with rising `waiters` = saturation.
- `gigastt_pool_timeouts_total` — checkout timeouts (backpressure events).
- `gigastt_inference_timeouts_total` — runs aborted by the inference timeout
  (a non-zero rate points at wedged runs or an over-tight timeout).
- When `--batch-pool-size` is set, the batch pool exports its own gauges
  `gigastt_batch_pool_available` and `gigastt_batch_pool_waiters`. Sample these
  to spot batch-pool saturation separately from the interactive pool.

**Triage**
1. If streaming is being starved by batch uploads, set `--batch-pool-size 1+`
   (remember it **splits** the existing pool). Monitor
   `gigastt_batch_pool_available` / `gigastt_batch_pool_waiters` to confirm the
   split is sized correctly.
2. If `gigastt_inference_timeouts_total` is climbing, capture a stuck run's
   input and check for an adversarial / huge file; raise the timeout only if the
   inputs are legitimately long.
3. If saturation is steady, scale `--pool-size` (watch RSS and single-job RTF)
   or add replicas.

## ORT optimized-graph cache warnings

The CPU encoder writes an ORT optimized-graph cache (`*_optimized.ort`,
~224 MiB) to `--optimized-cache-dir` (default `<model-dir>/optimized_cache`;
`/var/cache/gigastt` under the shipped systemd unit).

Filenames contain the source SHA-256 and a digest of the ORT build/API version,
CPU session configuration, optimization policy, target platform and detected
x86/ARM64 vector capabilities. Other architectures use their platform identity;
do not share their optimized cache across different CPU implementations. Replacing
weights invalidates the graph even when the basename, size and timestamp stay
the same. Old basename-only caches are ignored and rebuilt once; `cache-gc`
removes them and entries for obsolete weights, retaining all configuration
variants of installed encoders. Use a separate cache directory per model
installation when running GC.

Every encoder is hashed before cache lookup, and that digest is shared across
pool slots. Locally quantized RNN-T INT8 files retain their actual content identity;
their canonical filename does not require the published bundle digest. See
[model startup measurements](model-hashing.md). For self-contained sources,
a cold write rechecks the source before atomic publication. Keep model files
immutable while loading; install replacements between engine loads. A possible
ONNX external-data `location` marker disables graph caching because hashing the
main protobuf cannot identify separately stored weights. This conservative scan
can also disable caching for self-contained models containing that byte string.

**Symptoms** — WARN lines in the journal (once per engine load):
- `optimized graph cache directory cannot be created; loading source model without the cache (slower cold start, higher per-session RAM)`
- `optimized graph cache directory is not writable; loading source model without the cache (slower cold start, higher per-session RAM)`
- `optimized graph cache write failed; retrying without the cache (slower cold start, higher per-session RAM)` — the directory is writable but the cache *file* write failed (stale root-owned `*_optimized.ort`, ENOSPC mid-write).

**Effect** — the service starts and transcribes correctly either way; only
cold start is slower (the graph is re-optimized on every boot) and
per-session RAM is higher (no shared memory-mapped weights).

**Fix** — point `GIGASTT_OPTIMIZED_CACHE_DIR` at a writable directory
(under systemd, set it in `/etc/gigastt/gigastt.env` — the `EnvironmentFile=`
overrides the unit's built-in `Environment=` default) or fix
ownership/permissions on the existing one (e.g.
`chown -R gigastt:gigastt /var/cache/gigastt`, removing a stale root-owned
`*_optimized.ort` if present).

## Model download failures

On first run `gigastt download` / `gigastt serve` fetches the **lean INT8**
bundle into `~/.gigastt/models/`: default RNN-T heads from the pinned **GitHub
Release**, CTC heads (`ml_ctc` / `ml_ctc_large`) from **HuggingFace**. Each
file streams to a `.partial` path, is SHA-256-verified, then atomically
renamed. Concurrent processes coordinate via an advisory `flock`; downloads use
connect/read timeouts and a bounded redirect policy. Runtime never downloads or
loads FP32.

**Symptoms & recovery**
- *SHA-256 mismatch* — a corrupt or tampered mirror. The `.partial` is deleted
  and nothing is promoted; just re-run. Persistent mismatches mean a bad mirror
  or a stale pinned checksum.
- *Hang / timeout mid-download* — network / GitHub Releases (or HuggingFace for
  CTC) issue; re-run (the `.partial` is re-fetched, not resumed).
- *"Model not found" / INT8 encoder missing after a crash* — a crash before
  rename leaves only a `.partial`; presence checks ignore it, so re-running
  re-downloads the INT8 set.
- *Air-gapped / repeatable deploys* — bake the model into the image
  (`GIGASTT_BAKE_MODEL=1`, see `docs/deployment.md`) or pre-populate
  `~/.gigastt/models/` from a trusted copy.
- *Lean INT8-only tree* — four files for default `rnnt` (~220 MB class):
  `v3_rnnt_encoder_int8.onnx`, `v3_rnnt_decoder.onnx`, `v3_rnnt_joint.onnx`,
  `v3_vocab.txt`. Use `gigastt download` or copy those files; serve **requires**
  this INT8 set (FP32-only is rejected). See
  [Lean INT8-only install](deployment.md#lean-int8-only-install).

To force a clean re-download, remove `~/.gigastt/models/` and re-run.

## Out-of-memory (OOM)

The current RNN-T INT8 encoder is **305 MiB** and its optimized mapping is
shared across pool slots. Measure process/container peak memory under realistic
request concurrency, audio length and sidecar settings before setting limits.
The new bundle increased peak RSS by about 22–23% in the measured Linux
long-form comparison. Historical M1 resident/RSS figures used the old
ConvInteger bundle and are not a current-model sizing guarantee.
See [model measurements and trade-offs](rnnt-quantization.md).

**Reduce footprint**
- Runtime is **INT8 only** (`gigastt download` / first `serve`).
- Lower `--pool-size` (e.g. `1`–`2` on a 4 GB box). See also
  [Pool size tradeoffs](#pool-size-tradeoffs-ram-vs-concurrency-vs-rtf).
- `--pool-min-size 1` lets the server **boot on a degraded pool** if some
  triplets fail to load under memory pressure, instead of failing outright.
- On edge hosts, leave punctuation off (`--punctuation off`) if you do not need
  restored casing — the RuPunct model adds a small ready-RSS tax when present
  (see [Optional model ready tax](#optional-model-ready-tax)).
- Upload admission bounds concurrent encoded inputs before body collection;
  the limit is the boot-time batch/shared pool capacity times the body cap,
  plus the separate jobs-store byte budget. See
  [aggregate upload admission](long-recordings.md#aggregate-upload-admission).
  Buffering overhead, prepared/decoded audio and encoder scratch still require
  additional memory.

**Triage**
1. Check `terminationGracePeriodSeconds` isn't masking an OOM-kill as a slow
   shutdown.
2. Confirm the INT8 encoder is in use (`/v1/models` reports `"encoder":"int8"`).
3. Cap concurrency: `--pool-size` × (per-triplet RSS + peak scratch) must fit
   the box with headroom. Keep free RAM for admin reload if you use it
   ([Admin reload headroom](#admin-reload-headroom)).

## Resource & performance knobs

Operator notes for pool sizing, SKUs, VAD, and reload. Full flag list:
[`docs/cli.md`](cli.md).

### Pool size tradeoffs (RAM vs concurrency vs RTF)

- Each extra slot adds runtime arenas, scratch and decoder state; the encoder
  mapping is shared. Measure the marginal cost with the current model on your
  host. Historical M1 estimates of ~20 MB resident per slot used the old bundle.
- Pool > 1 also **splits encoder intra-op threads** across concurrent triplets.
  A **single** job on a busy multi-slot pool is therefore slower than the same
  job on pool=1 — typically about **+10–20% RTF** on a quiet serial workload
  (lab ≈ **+18%** at pool=2 vs pool=1). Raise pool for concurrent clients, not
  for single-stream latency.
- Edge / low-RAM: prefer **`--pool-size 1`**. Raise only when concurrent
  sessions need it and the host has free RAM after peak scratch.
- **Containers:** pool clamp uses **min(host RAM, cgroup `memory.max`)** on
  Linux (Docker/k8s limits). A 1 GiB container on a large host no longer
  over-admits pool slots based on host RAM alone.
- Shorthand: **`gigastt serve --profile edge`** sets **pool-size 1** and
  **`--vad`** when those flags are left at defaults (explicit `--pool-size` /
  `--vad` / `--vad=false` still win). Optional: add `--punctuation off` for
  the smallest ready RSS.

### Pool checkout timeout (queue vs fail-fast)

`--pool-checkout-timeout-secs` (default **30**) is how long a handler waits
for a free triplet when the pool is full:

| Setting | Behaviour |
|---|---|
| **Longer** (e.g. 60–120) | Callers **queue** longer; fewer 503s under short bursts; higher tail latency and more in-flight waiters |
| **Shorter** (e.g. 5–15) | **Fail-fast**: REST `503` + `Retry-After` / WS `timeout` + `retry_after_ms` sooner so clients can back off or hit another replica |

Tune with `gigastt_pool_waiters` and `gigastt_pool_timeouts_total`. Details:
[Pool exhaustion & backpressure](#pool-exhaustion--backpressure).

### batch_pool_size splits the pool (not additive)

`--batch-pool-size N` **carves N triplets out of** `--pool-size` for long REST
file jobs. It does **not** allocate extra idle triplets.

Example: `--pool-size 4 --batch-pool-size 1` → **3** interactive (WS/SSE) +
**1** batch, total still **4** loaded sessions. `0` (default) = shared pool.
Clamped so at least one interactive triplet remains.

### Admin reload headroom

`POST /v1/admin/reload` builds a **second** engine, then **swaps before warmup**
so the warm peak is not forced to stack on the previous copy once in-flight
work finishes. The 305 MiB encoder mapping is shared; ORT arenas and decoder
state are not. Peak after mmap is **unmeasured** (do not quote the copy-era
+536 MiB figure). Edge boxes with almost no free RAM can still OOM mid-build;
keep headroom or restart the process instead.

**Soft reload** (`POST /v1/admin/reload?soft=true`): after swap, wait up to ~5 s
for the previous engine's last in-flight holders to release, then warm — lowers
the warm+old double stack on quiet edge hosts. Response includes `"soft":true`
and `"soft_drained":true/false`. See [Admin reload](api.md#admin-reload).

### VAD for pause-rich long files

Enable **`--vad`** (or `GIGASTT_VAD=1`) for **meetings, podcasts, and other
pause-rich** long audio: Silero skips silence before decode and can finalize
streaming segments on trailing silence. On silence-rich material, wall time can
improve by up to about **×2.6** RTF vs running the full encoder over every
quiet stretch. Continuous speech gains little; VAD still downloads the Silero
model on first use (~few MB).

```sh
# Long meeting / podcast file
gigastt transcribe meeting.wav --vad
# Server-wide for REST + WS
gigastt serve --vad --pool-size 1
```

### Long-form decode paths (product)

| Path | When | Behaviour |
|------|------|-----------|
| **Speech regions (`--vad`)** | VAD loaded and not overridden off | Silero scores the stream causally, kept audio is decoded in the same overlapping windows as the plain path, word times remapped to the original timeline. Peak audio memory is O(one window) — no duration ceiling. |
| **Empty VAD regions** | VAD returns zero spans (e.g. tone), or the model fails mid-stream | **Fallback** to full / fixed-window decode (does not return empty text); the clip is re-read from the source. |
| **Fixed-window chunking** | File ≳ 30 s and no usable VAD path | Overlapping ~24 s windows (30 s on ANE), stitch words at overlap midpoints — bounds encoder activation memory. |

There is no separate client-side stitch API: operators use **`--vad`** for
pause-rich long-form quality + peak RAM, and rely on automatic chunking when
VAD is off. Per-request `?vad=false` forces whole-buffer / chunked decode on a
VAD-enabled server.

### Head SKU: ml_ctc is speed, not lean-RAM

`--model-variant ml_ctc` is a **throughput / RTF** choice (~**1.5×** faster RTF
than default `rnnt` in lab — e.g. RTF **~0.023** vs **~0.034**), **not** a
low-memory SKU. Ready RSS for `ml_ctc` is **about the same class as `rnnt`**
in older measurements made before the RNN-T encoder grew to 305 MiB.
The former speed and ready-memory comparison is not established for the new
RNN-T bundle. For less RAM use
**`--pool-size 1`**, not a head switch. Use `ml_ctc` / `ml_ctc_large` when you
need **ru/en/kk/ky/uz** or higher encode speed; `rnnt` remains the Russian-only
default. Its primary accuracy comparison awaits complete benchmark evidence.

### Optional model ready tax

When the **punctuation** model is present and the pass is enabled (`auto`/`on`
for `rnnt`), ready RSS grows by about **+4…28 MiB** depending on host and
load path. Edge profiles that only need bare text can set
`--punctuation off` (and skip downloading `~/.gigastt/models/punct/`) to avoid
that tax. Accuracy of the acoustic model is unchanged; only casing/punctuation
restoration is skipped.

## Metrics

`gigastt_http_requests_total{path="/v1/ws",status="503"}` with code `shutting_down` in the body is the signal that upgrades are being rejected because shutdown was already in flight. Usually correlated with `terminationGracePeriodSeconds` being shorter than `shutdown_drain_secs`.

(A counter for cancelled-WS by reason is not exported today; track cancellations via server logs instead.)

## On-call triage checklist

1. Pull a WS trace from the affected client. Confirm presence (or absence) of `Final` and the `Close` code.
2. Check server logs for `Shutdown signalled`, `Session cap reached`, or `Drain window expired`.
3. Confirm orchestrator `terminationGracePeriodSeconds` ≥ `shutdown_drain_secs + 5` (see `docs/deployment.md`).
4. If clients are seeing unexpected 503 `shutting_down`, the proxy LB may still be routing traffic after the pod started draining — add a `preStop` sleep to the k8s manifest so the LB deregisters the pod before the app sees `SIGTERM`.
5. If the cap is firing for legitimate long sessions, raise it — there's no correctness downside to `max_session_secs = 14400` (4 h), only a weaker guarantee against wedged sessions.

### Streaming retained audio

The streaming window setting is a soft slide trigger when stable-prefix
commits are enabled. Uncommittable moving-edge hypotheses can retain audio
beyond it. See [streaming retention](stream-retention.md) for the reproduced
policy counterexample, server-limit limitations and the bounded correction
contract.
