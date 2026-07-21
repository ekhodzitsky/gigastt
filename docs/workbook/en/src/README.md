# GigaSTT Workbook

Scenario-driven recipes for [gigastt](https://github.com/ekhodzitsky/gigastt),
the local Russian speech-to-text server powered by GigaAM v3. Each chapter
follows the same shape: **scenario → prerequisites → recipe → verifying the
result → common pitfalls → links**.

This book is a **cookbook, not a reference**. The canonical references stay in
[`docs/`](../../../) — the workbook links to them instead of duplicating them.

## Chapters

1. [Getting started](01-getting-started.md) — install, download the model,
   first transcription.
2. [CLI and batch processing](02-cli-batch.md) — CLI, batch, and watch-mode
   recipes for audio files.
3. [Telephony & VoIP](03-telephony-voip.md) — G.711/G.722/Opus and PBX
   recordings.
4. [Streaming over WebSocket](04-streaming-ws.md) — real-time transcription
   over WebSocket.
5. [Desktop & embedded](05-desktop-embedded.md) — Swift/SPM, sidecar,
   Electron, UniFFI.
6. [Deployment & ops](06-deployment-ops.md) — production deployment,
   monitoring, and operations.
7. [Models and backends](07-models-and-backends.md) — model variants,
   quantization, execution providers, alternative backends (in progress).

The [Russian version](../../ru/src/README.md) mirrors this book chapter by
chapter.

## Documentation map

Full inventory of the documentation in this repository: what each file is and
where it lives.

### References (canonical — never duplicated in the workbook)

| File | Contents | Fate |
|---|---|---|
| [docs/api.md](../../../api.md) | HTTP / WebSocket / SSE API reference | stays |
| [docs/asyncapi.yaml](../../../asyncapi.yaml) | AsyncAPI schema for the WS protocol | stays |
| [docs/openapi.yaml](../../../openapi.yaml) | OpenAPI schema for the REST API | stays |
| [docs/cli.md](../../../cli.md) | CLI reference (`serve`, `download`, `transcribe`, …) | stays |
| [docs/architecture.md](../../../architecture.md) | Architecture overview | stays |
| [docs/benchmarks.md](../../../benchmarks.md) | WER / RTF measurements | stays |
| [docs/privacy.md](../../../privacy.md) | Privacy and data-flow statement | stays |
| [docs/troubleshooting.md](../../../troubleshooting.md) | Symptom → cause → fix table | stays |
| [docs/observability/](../../../observability/) | Prometheus alerts and Grafana dashboard assets | stays |

### Guides (current)

| File | Contents | Fate |
|---|---|---|
| [docs/deployment.md](../../../deployment.md) | Reverse proxy, TLS, systemd, Docker | stays |
| [docs/quickstarts.md](../../../quickstarts.md) | In-process embedding quickstarts (FFI bindings) | stays |
| [docs/runbook.md](../../../runbook.md) | Operator runbook for production | stays |
| [docs/self-hosted-runner.md](../../../self-hosted-runner.md) | Self-hosted CI runners for benchmarks | stays |
| [docs/embedding-packaging.md](../../../embedding-packaging.md) | onnxruntime linking and packaging | stays |
| [docs/verifying-releases.md](../../../verifying-releases.md) | Verifying release artifacts | stays |
| [docs/ane-backend.md](../../../ane-backend.md) | ANE (Core ML) backend note — live `--features ane` code | stays |
| [docs/candle-backend.md](../../../candle-backend.md) | Candle/Metal backend note — live `--features candle` code | stays |
| [sdks/go/README.md](../../../../sdks/go/README.md) | Go WebSocket client SDK | stays |
| [sdks/js/README.md](../../../../sdks/js/README.md) | TypeScript WebSocket client SDK | stays |

### Historical (archived)

Completed design/plan documents kept for archaeology in
[`docs/archive/`](../../../archive/):

| File | Contents | Fate |
|---|---|---|
| [docs/archive/candle-metal-backend-plan.md](../../../archive/candle-metal-backend-plan.md) | Candle/Metal backend implementation plan (completed) | archived |
| [docs/archive/candle-metal-backend-design.md](../../../archive/candle-metal-backend-design.md) | Candle/Metal backend design (superseded by the shipped backend) | archived |

## Rules for contributors

- The workbook holds **recipes**; `docs/api.md`, `docs/cli.md`, and the
  AsyncAPI/OpenAPI schemas remain the canonical references. Link to them —
  do not copy their content.
- Every command and example in a chapter must be verified before merge.
- Inside the book (chapter ↔ chapter, chapter ↔ intro) use **relative `.md`
  links** — they work both on GitHub and in the rendered book. Links from the
  book to repository files (`docs/`, `crates/`, …) must be **absolute GitHub
  URLs** — relative ones 404 on the published site. No mdBook-specific
  templating.
- New chapters follow the [`_template.md`](_template.md) structure.
- **English is canonical.** The Russian book (`docs/workbook/ru/`) mirrors this
  one with identical file names; both versions are updated in the same PR.
