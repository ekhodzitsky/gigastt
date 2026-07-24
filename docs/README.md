# gigastt documentation

Index of the guides under `docs/`. The product hub remains the root
[README](../README.md) / [README_RU](../README_RU.md). Scenario-driven
onboarding lives in the bilingual workbook on GitHub Pages.

## Start here

| Guide | Audience | Contents |
|---|---|---|
| **[Workbook](https://ekhodzitsky.github.io/gigastt/)** | Everyone | Scenario recipes EN+RU: install → CLI/batch → telephony → WebSocket → desktop/embed → deploy → models |
| **[API](api.md)** | Integrators | WebSocket protocol, REST + SSE, jobs, admin reload, error codes, client examples |
| **[CLI](cli.md)** | Operators | Every subcommand and flag (drift-checked against `main.rs`) |
| **[Quickstarts](quickstarts.md)** | Embedders | In-process Python / Node / Swift / Kotlin |

## Reference

| Guide | Contents |
|---|---|
| **[Architecture](architecture.md)** | Pipeline, crates, surfaces, model heads, hardware EPs, INT8, air-gapped builds |
| **[OpenAPI](openapi.yaml)** | Machine-readable REST schema (`/health`, `/ready`, transcribe, jobs, admin) |
| **[AsyncAPI](asyncapi.yaml)** | Machine-readable WebSocket schema (`/v1/ws`) |
| **[Benchmarks](benchmarks.md)** | WER / RTF / footprint methodology and tables |
| **[Embedding & packaging](embedding-packaging.md)** | Static vs `ort-load-dynamic`, wheel/AAR notes |

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
- [`CHANGELOG.md`](../CHANGELOG.md) — release notes
- [`SECURITY.md`](../SECURITY.md) — vulnerability reporting + supported versions
- [`NOTICE`](../NOTICE) — third-party notices (opus, WeSpeaker, benchmark data)

Archive / design notes under `docs/archive/` and `docs/superpowers/` are historical
and may lag the current release.
