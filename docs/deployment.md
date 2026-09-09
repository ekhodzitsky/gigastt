# Deployment

gigastt is a **local-first server**: it listens on `127.0.0.1:9876` by default and refuses to bind to non-loopback addresses unless you pass `--bind-all` (or set `GIGASTT_ALLOW_BIND_ANY=1`). This is intentional — it prevents accidental public exposure.

For remote access, **terminate TLS and add authentication at a reverse proxy**. The server stays on localhost; the proxy handles the internet boundary.

## Security model

- **Server**: binds localhost only, no TLS, no auth
- **Proxy**: handles HTTPS, authentication, rate limiting, origin validation
- **Network**: proxy talks to server via localhost; proxy faces the internet

This separation keeps gigastt simple and lets you choose your proxy, TLS, and auth strategy.

## Caddy (recommended)

Caddy auto-provisions Let's Encrypt certificates and requires zero manual TLS config.

**Caddyfile:**

```
stt.example.com {
    reverse_proxy 127.0.0.1:9876 {
        transport http {
            versions h1 h2c
        }
        # Forward the real peer address so gigastt's per-IP rate-limiter
        # sees each client individually. `{remote_host}` comes from Caddy's
        # view of the TCP connection — clients cannot spoof it, unlike any
        # `X-Forwarded-For` header they may supply.
        header_up X-Real-IP {remote_host}
        header_up X-Forwarded-For {remote_host}
    }
    basic_auth /* {
        admin {env.CADDY_BASIC_AUTH_HASH}
    }
}
```

**Setup:**

```sh
# Generate bcrypt hash for basic_auth
caddy hash-password
# Enter password, copy the hash

# Export hash as environment variable
export CADDY_BASIC_AUTH_HASH='$2a$14$...'

# Run Caddy
caddy run

# Run gigastt with the browser-facing origin allowlisted — the proxy
# forwards the Origin header unchanged, and gigastt denies unknown
# cross-origin Origins with 403 origin_denied:
gigastt serve --allow-origin https://stt.example.com
```

**Why this works:**
- Caddy auto-provisions Let's Encrypt HTTPS
- Redirects HTTP → HTTPS automatically
- `reverse_proxy` upgrades WebSocket connections without extra config
- `h2c` (HTTP/2 Cleartext) to gigastt; browser talks h1/h2 to Caddy
- `basic_auth` protects both REST and WebSocket
- Browsers send `Origin: https://stt.example.com` and the proxy passes it through, so that origin must be in gigastt's `--allow-origin` list (see [Origin header and CORS](#origin-header-and-cors)) — or strip the `Origin` header at the proxy

## nginx

**nginx.conf:**

Add this at the top of the `http {}` block:

```nginx
map $http_upgrade $connection_upgrade {
    default upgrade;
    '' close;
}
```

In your server block for `stt.example.com`:

```nginx
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name stt.example.com;

    ssl_certificate /etc/letsencrypt/live/stt.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/stt.example.com/privkey.pem;

    auth_basic "STT API";
    auth_basic_user_file /etc/nginx/.htpasswd;

    location / {
        proxy_pass http://127.0.0.1:9876;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        # Overwrite, do NOT append. Using $proxy_add_x_forwarded_for would
        # concatenate the client-supplied X-Forwarded-For header, letting a
        # malicious client spoof their source IP and bypass the per-IP
        # rate-limiter (`--rate-limit-per-minute`). We want gigastt to see
        # the real peer address only.
        proxy_set_header X-Forwarded-For $remote_addr;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 600s;
    }

    # Optional: redirect HTTP to HTTPS
    error_page 497 https://$server_name$request_uri;
}

server {
    listen 80;
    listen [::]:80;
    server_name stt.example.com;
    return 301 https://$server_name$request_uri;
}
```

**Certificates (Let's Encrypt + certbot):**

```sh
sudo certbot certonly --webroot -w /var/www/html \
    -d stt.example.com
# Renews automatically with certbot timer
```

**Basic auth (.htpasswd):**

```sh
htpasswd -c /etc/nginx/.htpasswd admin
# Enter password
sudo chmod 644 /etc/nginx/.htpasswd
sudo nginx -t && sudo systemctl reload nginx

# Run gigastt with the browser-facing origin allowlisted — nginx forwards
# the Origin header unchanged, and gigastt denies unknown cross-origin
# Origins with 403 origin_denied:
gigastt serve --allow-origin https://stt.example.com
# Alternative: strip the header at the proxy instead of allowlisting it —
# add `proxy_set_header Origin "";` to the location block above.
```

**Why these settings:**
- `proxy_http_version 1.1` + `Upgrade`/`Connection` headers handle WebSocket upgrade
- `proxy_read_timeout 600s` — transcribing 10 minutes of audio takes time
- `X-Forwarded-For $remote_addr` (overwrite, not append) — see warning below for the rate-limiter implications
- `$connection_upgrade` map prevents connection pooling on HTTP/1.0

## Rate-limiter & X-Forwarded-For

When `--rate-limit-per-minute` is enabled, gigastt reads the peer IP from `X-Forwarded-For` (first hop, trimmed), then `X-Real-IP`, then the TCP `ConnectInfo` — see `crates/gigastt/src/server/rate_limit.rs::extract_client_ip` — so each real client gets its own token bucket instead of hashing every request behind the single proxy IP. Forwarded headers are honoured only when `--trust-proxy` is set **and** the direct peer is loopback, RFC1918, IPv6 unique-local (`fc00::/7`), or IPv6 link-local.

**The proxy is the trust boundary.** A client can put any value they want in an `X-Forwarded-For` header they send you; if the proxy blindly passes that header through (or _appends_ the peer address to the client's forgery), the rate-limiter bucket is keyed on attacker-controlled data and easily bypassed.

Both recipes above **overwrite** the header with the proxy's view of the TCP peer (`$remote_addr` in nginx, `{remote_host}` in Caddy) — never `$proxy_add_x_forwarded_for` or the default Caddy behaviour, which concatenate. Copy the snippets verbatim unless you know you need per-hop tracing.

If you deploy without a proxy (not recommended for public exposure), leave `--rate-limit-per-minute 0` (default). The real concurrency limits are then `--pool-size` (concurrent inference sessions) plus the body / WS-frame size caps (`--body-limit-bytes`, `--ws-frame-max-bytes`); they prevent resource exhaustion but will not keep a single attacker from reconnecting as fast as the kernel allows.

## Origin header and CORS

When a browser at `https://stt.example.com` makes a request through the proxy, it sets `Origin: https://stt.example.com`. Neither Caddy nor nginx strips or rewrites the `Origin` header, so gigastt's origin middleware sees it and returns `403 {"code":"origin_denied"}` unless that origin is allowlisted.

**Action needed when serving a browser app through the proxy:** pass the browser-facing origin explicitly (as both recipes above do):
```sh
gigastt serve --allow-origin https://stt.example.com
```
or strip the header at the proxy (e.g. `proxy_set_header Origin "";` in nginx) so requests arrive Origin-less and are treated like curl / native SDK calls.

**When no action is needed:**
Loopback Origins (`http://127.0.0.1:*`, `http://[::1]:*`, `http://localhost:*`) are always allowed, and a missing `Origin` header (curl, native SDK, header stripped at the proxy) is allowed. Opaque `Origin: null` (sandboxed iframe / `data:` document) is **denied**.

**If you want to talk directly to gigastt** (same machine, `http://localhost:9876`):
```sh
gigastt serve --allow-origin http://localhost:9876
```

**Multiple origins:**
```sh
gigastt serve \
    --allow-origin https://stt.example.com \
    --allow-origin https://app.example.com
```

**Warning:** `--cors-allow-any` disables origin validation (wildcard CORS). Only use for development.

## Docker

### Prebuilt images (GHCR)

Each tagged release publishes multi-arch images to GitHub Container Registry, so
you can `docker pull` instead of building from source (`cargo install`):

```sh
docker pull ghcr.io/ekhodzitsky/gigastt:2.18.0        # CPU, linux/amd64 + linux/arm64
docker pull ghcr.io/ekhodzitsky/gigastt:2.18.0-cuda   # CUDA, linux/amd64
docker run -p 127.0.0.1:9876:9876 ghcr.io/ekhodzitsky/gigastt:2.18.0
```

Pin a concrete version (`:2.18.0`) for reproducible deploys; `:latest` / `:cuda`
track the newest release. Want zero cold-start? Build a model-baked image
locally with `docker build --build-arg GIGASTT_BAKE_MODEL=1 -t gigastt:baked .`
(adds ~225 MB INT8).

### Build from source

The Dockerfile defaults to `--host 0.0.0.0 --bind-all` (allows the Docker bridge). Keep the server port on loopback when binding to the host:

```sh
docker run -p 127.0.0.1:9876:9876 gigastt
```

This publishes port 9876 inside the container to `127.0.0.1:9876` on the host. The proxy (on the host or in another container) connects via the loopback interface.

**Multi-container setup (docker-compose):**

```yaml
services:
  gigastt:
    build: .
    ports:
      - "127.0.0.1:9876:9876"
    # No --bind-all needed; container listens on 0.0.0.0:9876
    # Host bridges it to 127.0.0.1:9876

  caddy:
    image: caddy:latest
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy_data:/data
      - caddy_config:/config
    depends_on:
      - gigastt
    environment:
      - CADDY_BASIC_AUTH_HASH=${CADDY_BASIC_AUTH_HASH}

volumes:
  caddy_data:
  caddy_config:
```

## Health checks

Both proxies can target `http://127.0.0.1:9876/health` for health checks. The `/health` endpoint is exempted from origin validation, so no CORS headers needed.

```sh
curl http://127.0.0.1:9876/health
# {"status":"ok","model":"gigaam-v3-rnnt","variant":"rnnt","version":"2.18.0","punctuation":true,"itn":true}
```

**Non-blocking first run.** The port binds immediately, before the ~225 MB INT8 model
download finishes (if the model dir is empty). During that window `/health` returns `200` with
`model:"loading"` and `/ready` returns `503 {"reason":"initializing"}` — so a
Docker `HEALTHCHECK` / load-balancer probe on `/health` does not flap, and an
orchestrator can gate traffic on `/ready`. The `model`/`variant` fields report
the head actually loaded, so a client can confirm an upgrade switched heads.

## Metrics scraping

Enable Prometheus metrics with `--metrics`. They are served on a **separate listener** (default `127.0.0.1:9090`, override with `--metrics-listen`), not on the main API port — so `/metrics` is intentionally not behind the CORS allowlist or the per-IP rate limiter.

Keep the metrics listener on loopback and behind your own network controls; expose it deliberately (e.g. to a Prometheus scraper on a trusted network), never to the public internet.

```sh
gigastt serve --metrics                              # scrape http://127.0.0.1:9090/metrics
gigastt serve --metrics --metrics-listen 127.0.0.1:9100  # custom port
```

## Graceful shutdown & session caps

gigastt drains live WebSocket / SSE sessions on `SIGTERM` so clients receive a `Final` frame + `Close(1001 Going Away)` instead of a TCP reset. Two flags control the behaviour:

- `--max-session-secs N` / `GIGASTT_MAX_SESSION_SECS` (default `3600`). Wall-clock cap per WebSocket session. When exceeded the server emits `Error { code: "max_session_duration_exceeded" }` + `Close(1008 Policy Violation)`. `0` disables the cap (not recommended — a silence-streaming client will hold an inference slot forever).
- `--shutdown-drain-secs N` / `GIGASTT_SHUTDOWN_DRAIN_SECS` (default `10`, clamped to `>= 1`). Grace window after `SIGTERM` during which in-flight sessions may finish. Should comfortably fit inside your orchestrator's termination grace period so the process is not `SIGKILL`ed mid-drain.

### Kubernetes

Set `terminationGracePeriodSeconds` to **at least `shutdown_drain_secs + 5`** so the kubelet doesn't `SIGKILL` before the drain completes. Example (defaults):

```yaml
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      # drain (10 s) + safety margin (5 s) + LB deregistration hook (~15 s)
      terminationGracePeriodSeconds: 30
      containers:
        - name: gigastt
          image: ghcr.io/ekhodzitsky/gigastt:latest
          env:
            - name: GIGASTT_SHUTDOWN_DRAIN_SECS
              value: "10"
            - name: GIGASTT_MAX_SESSION_SECS
              value: "3600"
          lifecycle:
            preStop:
              exec:
                # Give the LB a beat to stop routing new traffic before SIGTERM
                command: ["/bin/sh", "-c", "sleep 10"]
```

### docker-compose

```yaml
services:
  gigastt:
    build: .
    # drain (10 s) + safety margin
    stop_grace_period: 15s
    environment:
      GIGASTT_SHUTDOWN_DRAIN_SECS: "10"
      GIGASTT_MAX_SESSION_SECS: "3600"
```

If you observe clients hanging past the cap or not receiving `Final` on deploy, see `docs/runbook.md` for the rollback escape hatches.

## Lean INT8-only install

Production inference needs only the **pre-quantized INT8 set** for one head —
about **~225 MB** on disk for default `rnnt`. The engine **requires**
`*_encoder_int8.onnx` (or the CTC INT8 basename); FP32-only trees are not
loadable. There is no FP32 download path for runtime.

**Minimum files for `rnnt` (default):**

| File | Role |
|------|------|
| `v3_rnnt_encoder_int8.onnx` | INT8 Conformer encoder (~215 MB) |
| `v3_rnnt_decoder.onnx` | LSTM decoder |
| `v3_rnnt_joint.onnx` | RNN-T joiner |
| `v3_vocab.txt` | Char vocabulary |

**Minimum files for `e2e_rnnt`:** `v3_e2e_rnnt_encoder_int8.onnx`,
`v3_e2e_rnnt_decoder.onnx`, `v3_e2e_rnnt_joint.onnx`, `v3_e2e_rnnt_vocab.txt`.

**CTC heads** (`ml_ctc` / `ml_ctc_large`) are already INT8-only:
`multilingual_ctc.int8.onnx` or `multilingual_large_ctc.int8.onnx` plus
`multilingual_vocab.txt`.

```sh
# Lean INT8 bundle from the pinned GitHub Release (only runtime path)
gigastt download
# or copy the four rnnt INT8 files into --model-dir / a volume, then:
GIGASTT_OFFLINE=1 gigastt serve --model-dir /path/to/models --pool-size 1
```

Optional side models (not required for core ASR):

| Path | When |
|------|------|
| `punct/` (RUPunct) | `--punctuation auto/on` on `rnnt` |
| `vad/silero_vad.onnx` | `--vad` |
| `speaker_model.onnx` (name per build) | diarization feature |

Runtime never loads FP32. `gigastt quantize` is packaging-only (needs a local
FP32 ONNX as source). `gigastt cache-gc` drops stale ORT optimized graphs.

Under the shipped systemd unit the ORT optimized-graph cache
(`*_optimized.ort`, written by the CPU encoder on first load) lives in
`/var/cache/gigastt` (systemd `CacheDirectory=gigastt` + `--optimized-cache-dir`),
not in the model directory — `/usr/share/gigastt/models` stays read-only
under `ProtectSystem=strict`. Pass `--optimized-cache-dir` (env
`GIGASTT_OPTIMIZED_CACHE_DIR`) to relocate it. Read-only model dirs are
supported either way: if the cache directory cannot be created or is not
writable, the server logs a warning and starts without the cache (slower
cold start, higher per-session RAM) instead of failing. When relocating the
cache, pass the same path to `gigastt cache-gc` so it prunes the right
directory.

## Air-gapped / offline installation

For hosts with no internet access, every release publishes a self-contained
tarball per Linux target — `gigastt-<ver>-offline-<target>.tar.gz` (binary +
INT8 `rnnt` model + punctuation model + systemd unit + `install.sh`) — plus two
Debian packages: `gigastt_<ver>_<arch>.deb` (binary + unit) and
`gigastt-model-int8_<ver>_all.deb` (the same model set under
`/usr/share/gigastt/models/`). All are signed like the other release assets
(per-asset `.sha256`, `SHA256SUMS.txt`, minisign).

```sh
# Tarball flow (any distro, incl. rpm-based like RED OS — rpm is deferred):
tar xf gigastt-<ver>-offline-x86_64-unknown-linux-gnu.tar.gz
cd gigastt-<ver>-offline && sudo ./install.sh   # binary, model, unit, gigastt user
sudo systemctl start gigastt
curl 127.0.0.1:9876/health

# Debian flow:
sudo dpkg -i gigastt_<ver>_amd64.deb gigastt-model-int8_<ver>_all.deb
sudo systemctl start gigastt
```

**Offline mode.** `GIGASTT_OFFLINE=1` (or the `--offline` flag) makes every
code path that would download a model — `gigastt download`, the punctuation /
VAD auto-fetch inside `serve` — fail fast with an error naming the **missing
file path** (for example `…/v3_rnnt_encoder_int8.onnx` for a lean install),
instead of a network timeout. Place the lean `rnnt` set (see
[Lean INT8-only install](#lean-int8-only-install)) under the model directory,
or install the offline tarball / `gigastt-model-int8` deb above. The shipped
systemd unit sets offline mode via `/etc/gigastt/gigastt.env`, so the service
never attempts a connection. To add optional models later (punctuation, VAD,
speaker diarization, other heads), fetch them on a connected machine and copy
them into the model directory; see the bundled `README-OFFLINE.md` for paths.

Note on builds: the *runtime* needs no network, but building from source does
(`ort` fetches a prebuilt onnxruntime) — use the prebuilt artifacts above in
air-gapped contours, or see `docs/embedding-packaging.md` for vendored-ort
builds.

## Upgrading from 2.0.x / 2.1.x

**The default recognition head changed.** Fresh installs from 2.3.0+ default to
the lower-WER `rnnt` head (bare lowercase from the acoustic model, then casing +
punctuation restored by a **separate** RUPunct model, and digits by an ITN pass —
both `auto`-on for `rnnt`). Older releases shipped only the `e2e_rnnt` head,
which bakes punctuation/casing/ITN into the acoustic model.

What this means when you bump the image / `cargo install` version:

- **Existing model volume** (e.g. a mounted `~/.gigastt`/`./data/gigastt` that
  already holds `e2e_rnnt` files): the engine auto-detects the on-disk head and
  keeps using `e2e_rnnt` — **no silent re-download, no transcript-style change.**
- **Fresh install / empty volume:** you get the `rnnt` head. Output is still
  cased and punctuated, but via the auto-downloaded RUPunct model (~29 MB) as a
  post-processing pass that **fails open** — if that model can't be fetched you
  get bare lowercase plus a warning, not an error.

To pin behavior explicitly regardless of what's on disk:

```sh
# Keep the old end-to-end head:
GIGASTT_MODEL_VARIANT=e2e_rnnt
# …or take the rnnt head but force the restoration passes on:
GIGASTT_PUNCTUATION=on
GIGASTT_ITN=on
```

Confirm which head is live without opening a WebSocket — `GET /health` (and
`/v1/models`) now report `model` / `variant` / `punctuation` / `itn`.

**Metrics moved off the main port.** Since 2.3.0 `/metrics` is served on a
separate listener (default `127.0.0.1:9090`), not the API port. If you pass
`--metrics`, publish/scrape `:9090` (or set `--metrics-listen`); a scraper still
pointed at `:9876/metrics` will get a 404.

## Sizing & performance (operators)

Quick defaults; full knobs and numbers live in
[runbook — Resource & performance knobs](runbook.md#resource--performance-knobs).

| Goal | Start with |
|---|---|
| Low RAM / edge | `--profile edge` (pool=1 + VAD) or `--pool-size 1`; optional `--punctuation off` |
| Lean disk (~220 MB model) | `gigastt download` (lean default) or copy INT8+dec+joint+vocab only |
| Concurrent streams | Raise `--pool-size` only with free RAM; expect ~+10–20% single-job RTF |
| Long meetings / podcasts | `--vad` (silence-rich RTF up to ~×2.6) |
| Multilingual or max throughput | `ml_ctc` / `ml_ctc_large` for languages/speed — **not** for lower ready RSS |
| Isolate long file jobs from WS | `--batch-pool-size N` **splits** `--pool-size` (not additive) |
| Saturation policy | Long `--pool-checkout-timeout-secs` = queue; short = fail-fast 503 + `retry_after_ms` |
| Hot reload on a small box | Second engine during build (mapped encoder shared; peak unmeasured after mmap) — keep headroom or restart |

## Hardening checklist

- **Bind address:** Keep `--host 127.0.0.1` unless you're running in a container (then use the port binding strategy above).
- **Rate limiting:** Use `--rate-limit-per-minute N` (v0.8.0+) on the server, or rate-limit at the proxy.
- **TLS termination:** Only at the proxy, never expose the server's raw port to the internet.
- **Origin allowlist:** Explicit `--allow-origin` values; never use `--cors-allow-any` in production.
- **Authentication:** At the proxy (Caddy/nginx basic auth, OAuth, JWT, mTLS, etc.).
- **Audit:** Run `cargo audit` and `cargo deny check` in your CI pipeline.

## See also

- [CLI Reference](cli.md) — `--bind-all`, `--allow-origin`, `--cors-allow-any` flags
- [Runbook](runbook.md) — pool exhaustion, OOM, resource knobs
- [Security](../SECURITY.md) — server-side security features
