# Documentation

Start with the [root README](../README.md) to install gigastt and transcribe a
file. These guides describe the checked-out source. Released binaries, model
bundles and language bindings can have different versions; use the matching
release tag when integrating a packaged version.

## Use and integrate

| Guide | What it answers |
|---|---|
| [Models](models.md) | Supported heads, files, downloads, formatting and offline setup |
| [Languages](languages.md) | Measured multilingual reading/conversation quality and limitations |
| [CLI](cli.md) | Commands, flags, environment variables and defaults |
| [API](api.md) | HTTP, WebSocket, SSE, jobs, errors and output contracts |
| [OpenAPI](openapi.yaml) / [AsyncAPI](asyncapi.yaml) | Machine-readable HTTP / WebSocket contracts |
| [In-process quickstarts](quickstarts.md) | Rust/Python/Node/Swift/Kotlin integration paths |
| [Embedding and linking](embedding-packaging.md) | Native ONNX Runtime packaging and dynamic linking |
| [Android C ABI](android.md) | Native library, JNI and model provisioning |
| [Go client](../sdks/go/README.md) / [TypeScript client](../sdks/js/README.md) | Clients for a running server |
| [Swift package](../packaging/swift/README.md) / [Android AAR](../packaging/android/README.md) | Platform packaging, availability and limits |
| [Managed server lifecycle](managed-lifecycle.md) | Running gigastt as an application-owned child process |
| [OpenClaw recipe](integrations/openclaw.md) | A version-specific external integration |
| [Public integrations](who-uses.md) | Dated source-backed usage examples |

## Operate

| Guide | What it answers |
|---|---|
| [Deployment](deployment.md) | Docker, TLS/authentication proxies, offline bundles and systemd |
| [Runbook](runbook.md) | Readiness, drain, backpressure, cancellation, memory and caches |
| [Troubleshooting](troubleshooting.md) | Diagnose installation, capture and recognition failures |
| [Long recordings](long-recordings.md) | Upload limits, in-memory jobs, progress and interruption |
| [Speaker attribution](speaker-attribution.md) | Measured diarization behavior on public meetings |
| [Privacy](privacy.md) | Processing host, network traffic, storage and logs |
| [Observability assets](observability/) | Prometheus alerts and dashboard |
| [Release verification](verifying-releases.md) | Checksums, optional signatures and build provenance |
| [Edge / Raspberry Pi](edge-raspberry-pi-roadmap.md) | Target-hardware measurement requirements; Pi remains unmeasured |

## Develop and release

- [Contributing](../.github/CONTRIBUTING.md): prerequisites, review and manual release procedure.
- [Testing](testing.md): explicit unit, integration, load/soak and benchmark commands.
- [Architecture](architecture.md): components, data flow and backend choices.
- [Performance acceptance](performance-regressions.md): dependency/model regressions and manual CI.
- [Homebrew releases](homebrew-release.md): manually propose and review Formula updates.
- [Self-hosted benchmarks](self-hosted-runner.md): dedicated hardware and workflow adaptation.
- [ANE backend](ane-backend.md) / [Candle backend](candle-backend.md): experimental assets and hardware limits.
- [Security policy](../.github/SECURITY.md), [Changelog](../CHANGELOG.md), [third-party notices](../NOTICE).

## Measurements and implementation evidence

These reports retain the model, code, hardware and corpus context of their
experiments. A historical result is not a current release guarantee. Available raw
benchmark results and fixtures are preserved; the benchmark report identifies
historical claims whose referenced artifacts are missing.

| Area | Reports |
|---|---|
| Recognition and models | [Benchmarks](benchmarks.md), [RNN-T quantization](rnnt-quantization.md), [public dataset protocol](held-out-datasets-roadmap.md), [benchmark harness](../benchmark/README.md) |
| File transcription | [Long-form stitching](longform-stitching.md), [audio source contract](audio-source-contract.md), [transcription contract matrix](transcription-contracts.md), [window scheduling](file-window-scheduling.md), [raw telephony PCM](raw-telephony-pcm.md) |
| Live decoding | [Window computation](live-window-compute.md), [retained audio](stream-retention.md), [transcript snapshots](transcript-snapshot-memory.md), [historical latency](archive/streaming-latency-2026-06-13.md) |
| Resource use | [Short recordings](short-recording-latency.md), [sidecar contention](sidecar-contention.md), [model hashing](model-hashing.md), [runtime output buffers](runtime-output-workspace.md), [CTC beam profile](ctc-beam-profile.md) |

For unmerged model/punctuation experiments, see the clearly labelled
[research links](models.md#quality-evidence-and-research). They do not extend the
released CLI catalog.
