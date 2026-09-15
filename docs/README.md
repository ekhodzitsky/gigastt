# gigastt documentation

Start with the [README](../README.md) for installation and examples.
Use the references below for a specific API, command or deployment question.

## Start here

| Guide | Audience | Contents |
|---|---|---|
| **[API](api.md)** | Integrators | WebSocket protocol, REST + SSE, jobs, admin reload, error codes, client examples |
| **[CLI](cli.md)** | Operators | Every subcommand and flag (drift-checked against `main.rs`) |
| **[Quickstarts](quickstarts.md)** | Embedders | In-process Python / Node / Swift / Kotlin |
| **[Who uses gigastt](who-uses.md)** | Integrators | Public applications, optional adapters and user reports, with source links |

## Reference

| Guide | Contents |
|---|---|
| **[Architecture](architecture.md)** | Pipeline, crates, surfaces, model heads, hardware EPs, INT8, air-gapped builds |
| **[OpenAPI](openapi.yaml)** | Machine-readable REST schema (`/health`, `/ready`, transcribe, jobs, admin) |
| **[AsyncAPI](asyncapi.yaml)** | Machine-readable WebSocket schema (`/v1/ws`) |
| **[Benchmarks](benchmarks.md)** | WER / RTF / footprint methodology and tables |
| **[Held-out datasets roadmap](../specs/held-out-datasets-roadmap.md)** | Public RU sets beyond Golos/OpenSTT (CV, FLEURS, RuLS, SOVA, Podlodka, ToneWebinars) |
| **[Embedding & packaging](embedding-packaging.md)** | Static vs `ort-load-dynamic`, wheel/AAR notes |
| **[Android / C ABI](android.md)** | Native library build, model bundling and Kotlin/JNI integration |

## Operations

| Guide | Contents |
|---|---|
| **[Deployment](deployment.md)** | Reverse proxy (Caddy/nginx), TLS, Docker |
| **[Runbook](runbook.md)** | Drain, pool saturation, timeouts, OOM, model download failures |
| **[Troubleshooting](troubleshooting.md)** | Symptom → cause → fix table |
| **[Observability](observability/)** | Prometheus alerts + dashboard |
| **[Privacy](privacy.md)** | What leaves the device (runtime vs build) |
| **[Verifying releases](verifying-releases.md)** | SHA256 / minisign / SLSA |
| **[Self-hosted runner](self-hosted-runner.md)** | Optional CI hardware |

## Backends (opt-in features)

| Guide | Feature |
|---|---|
| **[ANE backend](ane-backend.md)** | `--features ane` (macOS ARM64 Neural Engine) |
| **[Candle backend](candle-backend.md)** | `--features candle` (experimental parity path) |

## Specs & history

- [`specs/prod-readiness-v1.0.md`](../specs/prod-readiness-v1.0.md) — production readiness tracker
- [`specs/todo.md`](../specs/todo.md) — historical critique follow-ups
- [`specs/held-out-datasets-roadmap.md`](../specs/held-out-datasets-roadmap.md) — extra public benchmark sets (one-by-one)
- [`specs/resource-ttx-roadmap.md`](../specs/resource-ttx-roadmap.md) — completed resource program (lean INT8, pool defaults, cache GC, …)
- [`CHANGELOG.md`](../CHANGELOG.md) — release notes
- [Contributing](../.github/CONTRIBUTING.md) — development, pull requests and releases
- [Security policy](../.github/SECURITY.md) — vulnerability reporting + supported versions
- [`NOTICE`](../NOTICE) — third-party notices (opus, WeSpeaker, benchmark data)

Archive / design notes under `docs/archive/` and `docs/superpowers/` are historical
and may lag the current release. Dated `project-audit-*` snapshots (local,
not shipped) are **not** current product truth — do not quote them.

Experimental UniFFI AAR: [`packaging/android/README.md`](../packaging/android/README.md).
