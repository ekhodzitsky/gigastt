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
```

**Why this works:**
- Caddy auto-provisions Let's Encrypt HTTPS
- Redirects HTTP → HTTPS automatically
- `reverse_proxy` upgrades WebSocket connections without extra config
- `h2c` (HTTP/2 Cleartext) to gigastt; browser talks h1/h2 to Caddy
- `basic_auth` protects both REST and WebSocket
- Loopback Origin (`http://127.0.0.1:9876`) is always allowed at the server; browser request goes to `https://stt.example.com` so no CORS issues

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
```

**Why these settings:**
- `proxy_http_version 1.1` + `Upgrade`/`Connection` headers handle WebSocket upgrade
- `proxy_read_timeout 600s` — transcribing 10 minutes of audio takes time
- `X-Forwarded-For $remote_addr` (overwrite, not append) — see warning below for the rate-limiter implications
- `$connection_upgrade` map prevents connection pooling on HTTP/1.0

## Rate-limiter & X-Forwarded-For (V1-11)

When `--rate-limit-per-minute` is enabled, gigastt reads the peer IP from `X-Forwarded-For` (first hop, trimmed), then `X-Real-IP`, then the TCP `ConnectInfo` — see `crates/gigastt/src/server/rate_limit.rs::extract_client_ip` — so each real client gets its own token bucket instead of hashing every request behind the single proxy IP.

**The proxy is the trust boundary.** A client can put any value they want in an `X-Forwarded-For` header they send you; if the proxy blindly passes that header through (or _appends_ the peer address to the client's forgery), the rate-limiter bucket is keyed on attacker-controlled data and easily bypassed.

Both recipes above **overwrite** the header with the proxy's view of the TCP peer (`$remote_addr` in nginx, `{remote_host}` in Caddy) — never `$proxy_add_x_forwarded_for` or the default Caddy behaviour, which concatenate. Copy the snippets verbatim unless you know you need per-hop tracing.

If you deploy without a proxy (not recommended for public exposure), leave `--rate-limit-per-minute 0` (default). The server-level semaphore (`MAX_CONCURRENT_CONNECTIONS = 4`) is your only limit; it prevents exhaustion but will not keep a single attacker from reconnecting as fast as the kernel allows.

## Origin header and CORS

When a browser at `https://stt.example.com` makes a request through the proxy, it sets `Origin: https://stt.example.com`.

**Default (no action needed):**
Loopback Origins (`http://127.0.0.1:*`, `http://[::1]:*`, `http://localhost:*`) are always allowed. Since your browser talks to the proxy (not directly to the server), you don't need to add the origin to gigastt.

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

The Dockerfile defaults to `--host 0.0.0.0 --bind-all` (allows the Docker bridge). Keep the server port on loopback when binding to the host:

```sh
docker run -p 127.0.0.1:9876:9876 gigastt
```

This publishes port 9876 inside the container to `127.0.0.1:9876` on the host. The proxy (on the host or in another container) connects via the loopback interface.

**Multi-container setup (docker-compose):**

```yaml
version: '3.8'
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
# {"status":"ok"}
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

## Hardening checklist

- **Bind address:** Keep `--host 127.0.0.1` unless you're running in a container (then use the port binding strategy above).
- **Rate limiting:** Use `--rate-limit-per-minute N` (v0.8.0+) on the server, or rate-limit at the proxy.
- **TLS termination:** Only at the proxy, never expose the server's raw port to the internet.
- **Origin allowlist:** Explicit `--allow-origin` values; never use `--cors-allow-any` in production.
- **Authentication:** At the proxy (Caddy/nginx basic auth, OAuth, JWT, mTLS, etc.).
- **Audit:** Run `cargo audit` and `cargo deny check` in your CI pipeline.

## See also

- [CLI Reference](../README.md#cli-reference) — `--bind-all`, `--allow-origin`, `--cors-allow-any` flags
- [Security](../README.md#security) — server-side security features
