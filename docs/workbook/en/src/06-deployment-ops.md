# Deployment & ops

## Scenario

You are the admin who runs gigastt on a server, not a laptop. The path of
this chapter: **install → restrict → observe → upgrade** — one supervised
service (systemd or Docker), metrics flowing into Prometheus/Grafana, alerts
for the failure modes that matter, and an upgrade routine that does not cut
live transcription sessions.

Every recipe ends with a **Verify** step. Flags are checked against
`gigastt serve --help`; the full flag reference lives in
[docs/cli.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/cli.md)
and is not repeated here.

## Prerequisites

- gigastt installed (binary, package, or image) — see
  [Getting started](01-getting-started.md).
- A Linux host with **4+ GB RAM**: the default `--pool-size 2` with the INT8
  encoder sits at ~790 MiB RSS; leave headroom for the OS and request peaks.
- For the systemd path: systemd 241 or newer (any modern distro, including
  Astra Linux, RED OS, ALT) and root access.
- For the Docker path: Docker 20.10+; the NVIDIA Container Toolkit only for
  the CUDA variant.
- The model either downloadable once (~850 MB FP32, auto-quantized to INT8 on
  first start) or pre-installed from the offline bundle / model deb.

## Recipe

### Docker

Each tagged release publishes multi-arch images to GHCR — prefer pulling over
building:

```sh
docker pull ghcr.io/ekhodzitsky/gigastt:2.13.0        # CPU, linux/amd64 + linux/arm64
docker pull ghcr.io/ekhodzitsky/gigastt:2.13.0-cuda   # CUDA, linux/amd64
```

Pin a concrete tag for reproducible deploys; `:latest` / `:cuda` float.

Run with a named volume so the ~850 MB model (and the auto-generated INT8
encoder) survives container replacement:

```sh
docker run -d --name gigastt \
  -p 127.0.0.1:9876:9876 \
  -v gigastt-models:/home/gigastt/.gigastt/models \
  ghcr.io/ekhodzitsky/gigastt:2.13.0
```

Notes:

- The image's default command is `serve --port 9876 --host 0.0.0.0
  --bind-all` (container networking needs it); `-p 127.0.0.1:9876:9876`
  keeps the host-side exposure on loopback. Put your TLS proxy in front
  exactly as with a bare install.
- The container runs as the unprivileged `gigastt` user; the model directory
  inside is `/home/gigastt/.gigastt/models` — that is the volume mount point.
- The image carries a `HEALTHCHECK` on `/health`, so `docker ps` shows
  `healthy` as soon as the port serves. During the first-run model download
  and INT8 quantization (~2 min) `/health` answers `200` with
  `model:"loading"` while `/ready` answers
  `503 {"status":"not_ready","reason":"initializing"}` — gate traffic on
  `/ready`, not `/health`.
- **Baked image** (zero cold start, +~850 MB): build locally with the model
  inside — `docker build --build-arg GIGASTT_BAKE_MODEL=1 -t gigastt:baked .`
- **CUDA**: `docker run --gpus all -p 127.0.0.1:9876:9876
  ghcr.io/ekhodzitsky/gigastt:2.13.0-cuda` (requires the NVIDIA Container
  Toolkit; the binary falls back to CPU when no GPU is present).

**Verify:**

```sh
curl -s http://127.0.0.1:9876/ready
# {"status":"ready","pool_available":2,"pool_total":2}
curl -s http://127.0.0.1:9876/health
# {"status":"ok","model":"gigaam-v3-rnnt","variant":"rnnt","version":"2.13.0","punctuation":true,"itn":true}
```

### Air-gapped / offline installation

For hosts with no internet access, each release ships a self-contained
tarball per Linux target — binary + pre-quantized INT8 `rnnt` model +
punctuation model + systemd unit + installer — and two Debian packages
(`gigastt_<ver>_<arch>.deb` + `gigastt-model-int8_<ver>_all.deb`). The full
bundle inventory lives in
[README-OFFLINE.md](https://github.com/ekhodzitsky/gigastt/blob/main/packaging/offline/README-OFFLINE.md)
and is not repeated here.

On a connected machine, download and **verify before** carrying the files
over (the why and the threat model:
[docs/verifying-releases.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/verifying-releases.md)):

```sh
gh release download v2.13.0 -R ekhodzitsky/gigastt \
    -p 'gigastt-2.13.0-offline-x86_64-unknown-linux-gnu.tar.gz' \
    -p 'gigastt-2.13.0-offline-x86_64-unknown-linux-gnu.tar.gz.sha256' \
    -p 'gigastt-2.13.0-offline-x86_64-unknown-linux-gnu.tar.gz.minisig'
sha256sum -c gigastt-2.13.0-offline-x86_64-unknown-linux-gnu.tar.gz.sha256
minisign -Vm gigastt-2.13.0-offline-x86_64-unknown-linux-gnu.tar.gz -p gigastt.pub
gh attestation verify gigastt-2.13.0-offline-x86_64-unknown-linux-gnu.tar.gz \
    --repo ekhodzitsky/gigastt
```

On the target host:

```sh
tar xf gigastt-2.13.0-offline-x86_64-unknown-linux-gnu.tar.gz
cd gigastt-2.13.0-offline
sudo ./install.sh    # verifies SHA256SUMS.txt, then installs binary + models + unit
sudo systemctl enable --now gigastt
```

Debian-family alternative:

```sh
sudo dpkg -i gigastt_2.13.0_amd64.deb gigastt-model-int8_2.13.0_all.deb
sudo systemctl enable --now gigastt
```

The model is already INT8 — no download, no quantization step, no network.
The installed unit sets `GIGASTT_OFFLINE=1` via `/etc/gigastt/gigastt.env`,
so any code path that would fetch a model (enabling `--vad`, diarization, an
alternative recognition head) **fails fast with an error naming the file to
provide** instead of hanging on a connect timeout. To add optional models
later, run `gigastt download` on a connected machine and copy the files into
`/usr/share/gigastt/models/`.

Typical failures:

- `install.sh` aborts with `sha256sum: WARNING: 1 computed checksum did NOT
  match` — the tarball was corrupted on its way to the air-gapped host.
  Re-verify the outer `.sha256`, re-copy, re-run; nothing is installed
  partially.
- An offline-mode error naming a missing file (e.g. the VAD model after
  enabling `--vad`) — that model is not in the bundle; fetch it on a
  connected machine and copy it over.

**Verify:**

```sh
systemctl is-active gigastt
# active
curl -s http://127.0.0.1:9876/health
# {"status":"ok",...} — served immediately, the model is pre-installed
```

### systemd service

The hardened unit ships in
[packaging/systemd/](https://github.com/ekhodzitsky/gigastt/tree/main/packaging/systemd)
and is installed by both the deb and the offline bundle. Key properties (the
unit itself is short and commented — read it for the full list):

- Runs as the unprivileged `gigastt` user; models under
  `/usr/share/gigastt/models` are read-only to it.
- Loopback bind only (`127.0.0.1:9876`); expose the API through a reverse
  proxy.
- `Restart=on-failure`, `RestartSec=5` — a crash is restarted, a clean
  `systemctl stop` is not.
- Hardening set compatible with systemd 241 (`ProtectSystem=strict`,
  `NoNewPrivileges`, `PrivateTmp`, …), so the unit works unmodified on Astra
  Linux, RED OS and ALT.
- Overrides live in `/etc/gigastt/gigastt.env` (`GIGASTT_*` variables,
  `RUST_LOG`), loaded via `EnvironmentFile`.

Logs go to the journal:

```sh
journalctl -u gigastt -f          # follow
journalctl -u gigastt -n 100      # recent
```

Change log verbosity by editing `/etc/gigastt/gigastt.env`
(`RUST_LOG=gigastt=debug`), then `sudo systemctl restart gigastt`.

Change flags with a drop-in — never edit the shipped unit (package upgrades
replace it). `ExecStart` must be cleared before being re-set:

```ini
# sudo systemctl edit gigastt
[Service]
ExecStart=
ExecStart=/usr/bin/gigastt serve --model-dir /usr/share/gigastt/models --punct-model-dir /usr/share/gigastt/models/punct --metrics
```

`systemctl restart gigastt` sends `SIGTERM`; the server drains live
WebSocket/SSE sessions — each client receives a `Final` frame +
`Close(1001 Going Away)` — for up to `--shutdown-drain-secs` (default 10 s),
well inside systemd's default 90 s stop timeout. How to use this for version
bumps: [Upgrades and rollback](#upgrades-and-rollback) below.

**Verify:**

```sh
systemctl status gigastt --no-pager
curl -s http://127.0.0.1:9876/health
```

### Observability

Metrics are opt-in and served on a **separate listener** — never on the API
port, so they sit outside the CORS allowlist and the per-IP rate limiter:

```sh
gigastt serve --metrics                                  # http://127.0.0.1:9090/metrics
gigastt serve --metrics --metrics-listen 127.0.0.1:9100  # custom port
```

Keep the listener on loopback unless your Prometheus runs on another host —
and even then bind it to a trusted interface, never a public one.

Minimal Prometheus wiring (`prometheus.yml`):

```yaml
scrape_configs:
  - job_name: gigastt
    static_configs:
      - targets: ["127.0.0.1:9090"]

rule_files:
  - /etc/prometheus/rules/gigastt-alerts.yml   # copy of docs/observability/alerts.yml
```

The metrics that matter (all prefixed `gigastt_`):

| Metric | Meaning |
|---|---|
| `gigastt_http_requests_total` | Requests by path/method/status — 5xx rate, 503s |
| `gigastt_http_request_duration_seconds` | HTTP latency histogram (p50/p95/p99) |
| `gigastt_pool_available` / `gigastt_pool_waiters` | Free inference triplets vs queued callers — the saturation signal |
| `gigastt_pool_timeouts_total` | Checkout timeouts → clients received 503 + `Retry-After` |
| `gigastt_inference_timeouts_total` | Runs aborted by `--inference-timeout-secs` |
| `gigastt_inference_duration_seconds` | Inference latency histogram |
| `gigastt_ws_active_connections` | Live WebSocket sessions |
| `gigastt_rate_limit_rejections_total` | 429s from the per-IP limiter |
| `gigastt_batch_pool_available` / `gigastt_batch_pool_waiters` | Same pool gauges, for the `--batch-pool-size` split |

Ready-made assets — import, don't reinvent:

- [docs/observability/alerts.yml](https://github.com/ekhodzitsky/gigastt/blob/main/docs/observability/alerts.yml)
  — Prometheus rules: 5xx above 5%, `gigastt_pool_available == 0` for 1m,
  p95 above 10 s, sustained pool timeouts, health probe down (blackbox
  exporter).
- [docs/observability/dashboard.json](https://github.com/ekhodzitsky/gigastt/blob/main/docs/observability/dashboard.json)
  — Grafana dashboard (Dashboards → Import): request rate, latency, 5xx,
  pool availability, active WebSockets, inference duration, rate-limit
  rejections.

What to alert on in practice: **pool saturation** (`gigastt_pool_available
== 0` sustained — clients are getting 503s), **5xx ratio**, and **RAM** at
the node level (gigastt exports no self-RSS metric; use node_exporter or
cAdvisor).

Logs: `tracing` env-filter via `RUST_LOG` (default `gigastt=info`;
`gigastt=debug` for triage). Logs carry request metadata — durations, word
counts — never transcript text
([docs/privacy.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/privacy.md)).

**Verify:**

```sh
curl -s http://127.0.0.1:9876/ready > /dev/null   # samples the pool gauges once
curl -s http://127.0.0.1:9090/metrics | grep '^gigastt_pool_available'
# gigastt_pool_available 2
```

### Secure by default

The defaults are already the hardened posture — this recipe is mostly about
not weakening them:

- **Loopback bind.** `serve` refuses non-loopback addresses unless
  `--bind-all` / `GIGASTT_ALLOW_BIND_ANY=1` is set. Remote access = a
  TLS-terminating reverse proxy on the same host (Caddy/nginx configs:
  [docs/deployment.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/deployment.md)).
- **Origin allowlist.** Loopback origins are always allowed; every other
  `Origin` must be listed via `--allow-origin` (repeatable, exact match).
  `--cors-allow-any` is development-only. Disallowed origins get `403`.
- **Request limits.** `--body-limit-bytes` (default 50 MiB),
  `--ws-frame-max-bytes` (512 KiB), `--idle-timeout-secs` (300),
  `--max-session-secs` (3600), `--inference-timeout-secs` (600),
  `--pool-checkout-timeout-secs` (30) — 503 + `Retry-After` on pool
  saturation.
- **Rate limiting** (opt-in): `--rate-limit-per-minute N` with
  `--rate-limit-burst` → `429` + `Retry-After`. Behind a proxy it works
  per-client only if the proxy *overwrites* `X-Forwarded-For` and
  `--trust-proxy` is set — copy the proxy snippets in
  [docs/deployment.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/deployment.md#rate-limiter--x-forwarded-for)
  verbatim.
- **Model integrity.** Downloads are SHA-256-verified and atomically renamed
  (`.partial` → final); a corrupt file is never promoted.
- **Release verification.** Every release asset has a `.sha256` sidecar +
  `SHA256SUMS.txt`, a minisign signature, a CycloneDX SBOM, and SLSA build
  provenance. Verify before installing anything —
  [docs/verifying-releases.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/verifying-releases.md).
  Minimum routine:

```sh
minisign -Vm gigastt-2.13.0-x86_64-unknown-linux-gnu.tar.gz -p gigastt.pub
gh attestation verify gigastt-2.13.0-x86_64-unknown-linux-gnu.tar.gz \
    --repo ekhodzitsky/gigastt
```

- **Privacy.** No telemetry, no outbound calls after the one-time model
  download, transcripts never logged
  ([docs/privacy.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/privacy.md)).

**Verify:**

```sh
ss -ltnp | grep 9876
# tcp LISTEN 0 ... 127.0.0.1:9876 ...   (loopback only)
curl -s -o /dev/null -w '%{http_code}\n' \
    -H 'Origin: https://attacker.example' http://127.0.0.1:9876/v1/models
# 403
```

### Upgrades and rollback

Pin what you deploy (image tag, deb version) so an upgrade is a deliberate,
reversible step. The model directory is state: it persists across upgrades,
the engine auto-detects the installed recognition head, and **no silent
re-download happens** when you bump the binary.

Docker:

```sh
docker pull ghcr.io/ekhodzitsky/gigastt:2.13.1
docker stop --time 15 gigastt && docker rm gigastt
docker run -d --name gigastt \
  -p 127.0.0.1:9876:9876 \
  -v gigastt-models:/home/gigastt/.gigastt/models \
  ghcr.io/ekhodzitsky/gigastt:2.13.1
```

`docker stop` sends `SIGTERM`; `--time 15` gives the drain window
(`--shutdown-drain-secs`, default 10 s) room to finish before `SIGKILL` —
Docker's default of 10 s races the drain. Clients receive `Final` +
`Close(1001)` and reconnect; short REST uploads in flight may need a retry.

systemd / deb:

```sh
sudo dpkg -i gigastt_2.13.1_amd64.deb
sudo systemctl restart gigastt
journalctl -u gigastt -f    # expect a clean drain, no "Drain window expired"
```

On Kubernetes the same rule applies from the orchestrator side:
`terminationGracePeriodSeconds` ≥ `shutdown_drain_secs + 5` (full manifest:
[docs/deployment.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/deployment.md#graceful-shutdown--session-caps)).

**Rollback.** Re-deploy the previous tag or package — the on-disk model set
is unchanged, so the old binary starts against the same files:

```sh
docker run -d --name gigastt \
  -p 127.0.0.1:9876:9876 \
  -v gigastt-models:/home/gigastt/.gigastt/models \
  ghcr.io/ekhodzitsky/gigastt:2.13.0
# or: sudo dpkg -i gigastt_2.13.0_amd64.deb && sudo systemctl restart gigastt
```

If a drain-related regression breaks your WebSocket clients after an upgrade,
the escape hatch is `--shutdown-drain-secs 0` (clamped to 1 s) — see
[docs/runbook.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/runbook.md)
for the full symptom table.

**Verify** (after every upgrade or rollback):

```sh
curl -s http://127.0.0.1:9876/health
# "version" is the release you deployed; "model"/"variant" are unchanged
curl -s http://127.0.0.1:9876/ready
```

## Verifying the result

End-to-end smoke after any recipe above:

```sh
systemctl is-active gigastt || docker ps --filter name=gigastt --format '{{.Status}}'
curl -s http://127.0.0.1:9876/health     # status ok, expected version
curl -s http://127.0.0.1:9876/ready      # ready, pool_available >= 1
curl -s http://127.0.0.1:9090/metrics | grep '^gigastt_pool_available'
```

Then transcribe one short file through the API you actually expose (REST
recipes: [Server integration](05-server-integration.md); CLI check:
[File transcription](02-file-transcription.md)).

## Common pitfalls

- **OOM — container or service killed.** RSS scales with `--pool-size`: the
  INT8 encoder is ~400 MiB per triplet, ~790 MiB at the default pool of 2;
  the FP32 encoder is ~4× larger (never pass `--skip-quantize` in
  production). On a 4 GB box keep `--pool-size` at 1–2; `--pool-min-size 1`
  lets the server boot on a degraded pool instead of crashing when memory is
  tight. If Kubernetes reports `OOMKilled`, lower the pool or raise the pod
  limit — details in
  [docs/runbook.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/runbook.md).
- **503 `timeout` under load.** Every triplet is busy and a caller waited out
  `--pool-checkout-timeout-secs` (30 s): REST gets `503` + `Retry-After`,
  WebSocket gets an error with `retry_after_ms`. This is backpressure, not a
  bug — raise `--pool-size`, split batch work off with `--batch-pool-size`,
  and watch `gigastt_pool_waiters` / `gigastt_pool_timeouts_total`.
- **`/metrics` unreachable from the Prometheus host.** By design: the
  listener defaults to `127.0.0.1:9090`. Point the scraper at the gigastt
  host itself, or deliberately re-bind with `--metrics-listen` on a trusted
  interface — never a public one. A scraper still aimed at `:9876/metrics`
  gets 404: metrics moved off the API port.
- **Readiness probes flapping on first start.** The first run downloads
  ~850 MB and quantizes (~2 min). `/health` returns `200` with
  `model:"loading"` the whole time, but `/ready` returns `503 initializing`.
  If your load balancer routes on `/health`, early clients get 503s — probe
  `/ready`, or pre-install / bake the model so the window disappears.
- **Rate limiter punishing everyone behind a proxy.** Without
  `--trust-proxy` — and a proxy that overwrites rather than appends
  `X-Forwarded-For` — all clients share one bucket keyed on the proxy's
  address. Symptoms and the exact proxy configuration:
  [docs/deployment.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/deployment.md#rate-limiter--x-forwarded-for).

## Links

- [Getting started](01-getting-started.md) — install and first transcription
- [Server integration](05-server-integration.md) — REST / SSE / WebSocket recipes
- [docs/deployment.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/deployment.md) — reverse proxy, TLS, Kubernetes manifests
- [docs/runbook.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/runbook.md) — symptom → cause → escape hatch
- [docs/cli.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/cli.md) — full `serve` flag reference
- [docs/observability/alerts.yml](https://github.com/ekhodzitsky/gigastt/blob/main/docs/observability/alerts.yml) — Prometheus alerting rules
- [docs/observability/dashboard.json](https://github.com/ekhodzitsky/gigastt/blob/main/docs/observability/dashboard.json) — Grafana dashboard
- [docs/verifying-releases.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/verifying-releases.md) — minisign, SBOM, SLSA provenance
- [docs/privacy.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/privacy.md) — what data moves where
- [packaging/systemd/](https://github.com/ekhodzitsky/gigastt/tree/main/packaging/systemd) — unit + env file
- [packaging/offline/README-OFFLINE.md](https://github.com/ekhodzitsky/gigastt/blob/main/packaging/offline/README-OFFLINE.md) — offline bundle contents
