# Appendix A — Error codes and close codes

Jump table from **symptom → code → what to do**. Full field-level contracts stay
in [docs/api.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/api.md)
and [docs/troubleshooting.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/troubleshooting.md);
this page is the cookbook index.

## Before you dig into codes

| Check | Expect | Chapter |
|---|---|---|
| `GET /health` | `200`, `model` is `"loading"` or a head name | [01](01-getting-started.md) |
| `GET /ready` | `200` before first audio / job | [04](04-streaming-ws.md), [02](02-cli-batch.md) |
| `version` in `/health` | matches the binary/image you deployed | [06](06-deployment-ops.md) |

Gate **liveness** on `/health`, **readiness** on `/ready` — never on "TCP port is open".

## REST / SSE / jobs

| HTTP | Code | Typical cause | Fix |
|---|---|---|---|
| 400 | `empty_body` | Empty POST body | Send audio bytes |
| 400 | `invalid_format` | Bad `?format=` | Use `json` / `txt` / `srt` / `vtt` / `md` — [02](02-cli-batch.md) |
| 400 | `unsupported_codec` | Bad `?codec=` | `pcmu` / `pcma` / `g722` — [03](03-telephony-voip.md) |
| 400 | `invalid_sample_rate` | Missing/out-of-range rate with raw codec | Pass `sample_rate=8000` or `16000` — [03](03-telephony-voip.md) |
| 400 | `conflicting_modes` | `channels=split` **and** `diarization=true` | Pick one — [03](03-telephony-voip.md) |
| 403 | `loopback_only` | Non-loopback `POST /v1/admin/reload` | Call from `127.0.0.1` — [06](06-deployment-ops.md#hot-reload-the-model-without-restart) |
| 404 | (no body) / `jobs_disabled` | Jobs API off | `--enable-jobs` — [02](02-cli-batch.md) |
| 404 | `job_not_found` | Unknown or TTL-evicted job | Persist results client-side — [02](02-cli-batch.md) |
| 409 | `job_not_finished` | Result polled too early | Wait for `done` or poll status |
| 409 | `job_not_cancellable` | Cancel on terminal job | Ignore / treat as done |
| 409 | `reload_in_progress` | Parallel admin reloads | Wait and retry once — [06](06-deployment-ops.md) |
| 409 | `punctuation_not_available` | Forced `punctuation=true` without model | Install punct model or use `auto` — [07](07-models-and-backends.md) |
| 413 | `payload_too_large` | Body > `--body-limit-bytes` | Raise limit or chunk via jobs — [02](02-cli-batch.md), [06](06-deployment-ops.md) |
| 422 | `invalid_audio` | Corrupt / unsupported container | Check format table — [03](03-telephony-voip.md) |
| 422 | `transcription_error` | Decode ok, inference failed | Check logs; try INT8 model present — [07](07-models-and-backends.md) |
| 429 | `rate_limited` | Per-IP bucket empty | Wait `Retry-After`; or raise limits / `--trust-proxy` — [06](06-deployment-ops.md) |
| 429 | `queue_full` | Jobs store full | Drain results / raise `--jobs-max` — [02](02-cli-batch.md) |
| 503 | `timeout` | Pool saturated | Back off `Retry-After`; raise `--pool-size` — [07](07-models-and-backends.md) |
| 503 | `pool_closed` | Shutdown in progress | Reconnect after deploy — [06](06-deployment-ops.md) |
| 503 | `initializing` | Model still loading (WS upgrade or ready) | Poll `/ready` — [01](01-getting-started.md) |
| 503 | `reload_failed` / `reload_unsupported` | Hot-reload build failed / no builder | Fix model files; keep old engine — [06](06-deployment-ops.md) |
| 504 | `inference_timeout` | One run > `--inference-timeout-secs` | Raise timeout for long files — [02](02-cli-batch.md) |

## WebSocket error codes

| Code | Session | Meaning | Fix |
|---|---|---|---|
| `timeout` | never opened | Pool checkout timed out | Wait `retry_after_ms`, reconnect — [04](04-streaming-ws.md) |
| `pool_closed` | ends | Server draining | Reconnect after upgrade — [06](06-deployment-ops.md) |
| `idle_timeout` | ends (1001) | No frames for `--idle-timeout-secs` | Keep streaming PCM (silence is fine) — [04](04-streaming-ws.md) |
| `max_session_duration_exceeded` | ends (1008) | Hit `--max-session-secs` | Reconnect; `final` was flushed — [04](04-streaming-ws.md) |
| `policy_violation` | ends (1008) | Empty-frame spam | Stop sending empty binaries |
| `inference_timeout` | ends | Chunk inference too long | Shorter audio / raise timeout |
| `inference_error` | continues | Bad chunk | Fix client audio; session kept |
| `inference_panic` | continues, state reset | Panic isolated | Treat as new utterance; keep prior finals |
| `configure_too_late` | continues | `configure` after first audio | Send configure first — [04](04-streaming-ws.md) |
| `invalid_sample_rate` | continues | Rate not in `supported_rates` | Use a rate from `ready` |
| `unsupported_protocol_version` | ends | Client protocol mismatch | Speak `1.0` — [04](04-streaming-ws.md) |

## WebSocket close codes

| Code | When | Client action |
|---|---|---|
| 1001 Going Away | SIGTERM drain, idle, ping timeout | Reconnect; expect a flushed `final` on shutdown |
| 1008 Policy Violation | max session / empty spam | Adjust caps or client behaviour |
| 1009 Message Too Big | Frame > `--ws-frame-max-bytes` | Smaller PCM frames |
| 1006 Abnormal Closure | No close frame (kill, crash, middlebox) | Check process health; not a protocol code the server sends |

## Links

- [docs/api.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/api.md) — canonical tables
- [docs/troubleshooting.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/troubleshooting.md) — long symptom list
- [docs/runbook.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/runbook.md) — operator escapes
- [Streaming](04-streaming-ws.md) · [Deployment](06-deployment-ops.md) · [CLI/jobs](02-cli-batch.md)
