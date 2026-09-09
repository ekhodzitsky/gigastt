# gigastt — Agent Guide

> Local speech-to-text server powered by GigaAM v3 (rnnt head by default). On-device Russian
> speech recognition via ONNX Runtime. No cloud APIs, no API keys, full privacy.
>
> Repository: https://github.com/ekhodzitsky/gigastt  
> crates.io: https://crates.io/crates/gigastt  
> License: MIT

## Project Overview

**gigastt** is a single-binary Rust server that turns any machine into an
on-device Russian speech-to-text endpoint. File/REST WER is the headline
number; live WebSocket is incremental partials over a buffered offline RNN-T
(not batch-equal WER — see [docs/benchmarks.md](docs/benchmarks.md#streaming-measurement-protocol)).
It loads the GigaAM v3 RNN-T model
(Conformer encoder + LSTM decoder + joiner, 240M params) via ONNX Runtime and
exposes:

- **WebSocket** (`/v1/ws`) — streaming transcription with partial/final results
- **REST** (`/v1/transcribe`) — file upload, full JSON response
- **SSE** (`/v1/transcribe/stream`) — file upload, streaming Server-Sent Events
- **OpenAI-compatible** (`/v1/audio/transcriptions`) — multipart `file` + `model` → `{"text":"..."}`
- **CLI** — `serve`, `download`, `transcribe`, `transcribe-batch`, `watch`, `cache-gc`, `quantize` commands

The product path is **INT8 only** (~225 MB prequantized bundle from Releases):
`serve` / `download` / engine load never use FP32. `gigastt quantize` remains a
packaging tool that needs a local FP32 ONNX as source (not a runtime path).

### Key metrics

| Property | Value |
|---|---|
| WER (Russian) | **3.55%** clean read (rnnt head, `golos_crowd_1k`); leads far-field/phone/YouTube — see [docs/benchmarks.md](docs/benchmarks.md) |
| RTF (INT8, M1 CPU) | ~0.10 |
| Memory | ~46 MB resident / ~277 MB `ps` RSS at `--pool-size 1`; ~66 MB / ~510 MB at the default `--pool-size 2` (INT8, M1 Pro, steady state — the 215 MB model is memory-mapped and shared, so RSS overstates; resident footprint is the honest figure) |
| Concurrent sessions | 2 (configurable via `--pool-size`) |

## Technology Stack

- **Language**: Rust 2024 edition, stable toolchain (1.94+)
- **ONNX Runtime**: `ort` pinned to exactly `2.0.0-rc.13`
- **Async runtime**: tokio (full features)
- **HTTP + WebSocket server**: axum 0.8 (`ws`, `multipart`)
- **CLI**: clap 4 (derive, env)
- **Serialization**: serde + serde_json
- **Logging**: tracing + tracing-subscriber (env-filter)
- **Error handling**: anyhow (internal), `GigasttError` (public API)
- **Audio decoding**: ryf (WAVE family), symphonia (AAC, MP3, OGG, FLAC), opus-rs (Opus)
- **Audio resampling**: rubato 5
- **FFT**: rustfft 6
- **Protobuf**: prost 0.14 + prost-build 0.14 (build-time)
- **Rate limiting**: in-tree token-bucket (parking_lot Mutex + HashMap)
- **Metrics**: in-tree Prometheus text encoder (optional `--metrics` flag)

### Execution providers (compile-time features)

| Platform | Feature | Provider |
|---|---|---|
| macOS ARM64 | `--features coreml` | CoreML + Neural Engine |
| Linux x86_64 + NVIDIA | `--features cuda` | CUDA 12+ |
| Android / ARM64 | `--features nnapi` | NNAPI (via `ort/nnapi`) |
| macOS ARM64 | `--features ane` | Apple Neural Engine via a compiled `.mlpackage`, file mode only (see `docs/ane-backend.md`) |
| macOS ARM64 | `--features candle` | Candle/Metal, experimental — output byte-identical to `ort` (see `docs/candle-backend.md`) |
| Any | _(default)_ | CPU |

Features `coreml` and `cuda` are **mutually exclusive**. `nnapi` is not mutually exclusive with either.
`ane` and `candle` select a non-`ort` backend under `runtime/` rather than an `ort` execution provider.

Lean-build axes, all **on** by default: `diarization`, `net`, `async-pool`, `file-decode`. Turn them
off (`--no-default-features`) for embedded targets that side-load models and feed raw PCM — note
this means `--features diarization` is redundant, and opting *out* is the meaningful direction.

## Build Requirements

- Rust 1.94+ (stable)
- `protoc` (Protocol Buffers compiler) on `PATH` — required by `build.rs` which
  regenerates ONNX protobuf types via `prost-build`
  - macOS: `brew install protobuf`
  - Debian/Ubuntu: `apt install protobuf-compiler`
- **Build-time network fetch:** `ort`'s default `download-binaries` feature downloads a
  prebuilt onnxruntime native library over the network at build time (outside `Cargo.lock`,
  verified by an embedded checksum). The "no cloud" guarantee is runtime-only. For
  air-gapped builds, use `ort` with `default-features = false` + `load-dynamic` and pin the
  native library via `ORT_*` env vars / `.cargo/config.toml`.

## Build Commands

```sh
# Debug build (CPU only, any platform)
cargo build

# Release build (LTO, stripped, single codegen unit)
cargo build --release

# macOS ARM64 with CoreML / Neural Engine
cargo build --release --features coreml

# Linux x86_64 with NVIDIA CUDA 12+
cargo build --release --features cuda

# Android with NNAPI
cargo build --release --features nnapi

# Lean embedded build: drop diarization / HTTP download / tokio / file decode
cargo build --release --no-default-features
```

## Test Commands

The project uses a three-tier test architecture:

### Unit tests (no model required, run in CI on every PR)

```sh
cargo test --workspace --lib --bins  # unit tests across the workspace (see note below)
cargo clippy                         # Lint (zero warnings expected)
cargo fmt --check                    # Format check
```

Unit tests live in `#[cfg(test)] mod tests` at the bottom of each source file.
They use synthetic data. Test naming convention: `test_<what>_<expected_behavior>`.

### E2E tests (require model ~225 MB INT8, run in CI on main push only)

```sh
# Download model first
cargo run -- download

# Run all e2e tests serially (single-threaded to avoid OOM)
cargo test -p gigastt --test e2e_rest --test e2e_ws --test e2e_errors --test e2e_shutdown --test e2e_rate_limit --test e2e_jobs --test e2e_cli --test e2e_admin_reload --test e2e_http_cov -- --ignored --test-threads=1
```

| Test file | Coverage |
|---|---|
| `tests/e2e_rest.rs` | REST API: health, transcribe, SSE streaming, error paths |
| `tests/e2e_ws.rs` | WebSocket: ready, audio, stop, configure, errors, concurrent |
| `tests/e2e_errors.rs` | Error paths: oversized body/frame, pool saturation, idle timeout |
| `tests/e2e_shutdown.rs` | Graceful shutdown: WS final + close, SSE termination, max-session cap |
| `tests/e2e_rate_limit.rs` | Per-IP rate limiter 429 behavior |
| `tests/e2e_jobs.rs` | Async `/v1/jobs` queue (requires `--enable-jobs`) |
| `tests/e2e_cli.rs` | CLI `transcribe` / batch / watch smoke |
| `tests/e2e_admin_reload.rs` | Loopback `POST /v1/admin/reload` |
| `tests/e2e_http_cov.rs` | Extra HTTP/export coverage |

Shared helpers are in `tests/common/mod.rs` (server startup with shutdown handle,
WAV generation, WebSocket connect, readiness polling).

Long-form stitch quality (`tests/longform_quality.rs`) is local-only (needs the
RuLS corpus). See `CLAUDE.md` for the command. Node in-process binding:
`engines.node` in `crates/gigastt-node/package.json` is `>=18`; the WS client
SDK is Node ≥ 20.

### Load & soak tests (require model, run locally + nightly CI)

```sh
cargo test -p gigastt --test load_test -- --ignored           # 3 load tests
cargo test -p gigastt --test soak_test -- --ignored           # Continuous WS cycling
```

Soak duration is configurable via `GIGASTT_SOAK_DURATION_SECS` (default 300s).

### Benchmark suite

```sh
cargo test -p gigastt --test benchmark -- --ignored            # WER on Golos fixtures
```

Custom harness (`harness = false` in `Cargo.toml`).

## Code Organization

```
crates/
  gigastt-quantize/       # Native Rust INT8 dynamic quantizer (optional; lean INT8 default skips it)
  gigastt-core/src/       # Core library (inference engine, no server deps)
    lib.rs                # Public module exports
    error.rs              # Typed error types (GigasttError)
    export/               # Transcript export: TXT / SRT / VTT / Markdown
    punctuation/          # Punctuation restoration (windowed)
    itn.rs · lexicon.rs   # Inverse text normalization, lexicon
    vad/                  # Voice activity detection
    inference/
      mod.rs              # Module wiring + shared constants (N_MELS, N_FFT, HOP_LENGTH, PRED_HIDDEN)
      engine/             # Engine: load, warmup, transcribe, streaming decode loop
                          #   (config / load / stream / transcribe / infer)
      pool.rs             # SessionPool (checkout, backpressure)
      state.rs            # StreamingState / DecoderState
      features.rs         # Mel spectrogram (64 bins, FFT=320, hop=160, HTK)
      tokenizer.rs        # Vocabulary: char (rnnt, 34) / BPE (e2e_rnnt, 1025) / multilingual char (ml_ctc, 71)
      decode/             # RNN-T greedy decode loop
      ctc.rs              # Greedy CTC decode (ml_ctc heads — no decoder / joiner)
      bias.rs             # Hotword biasing
      diarization.rs      # polyvoice glue: Embedder adapter, offline + streaming pipelines
      types.rs            # TranscribeRequest / TranscribeResult and friends
      audio/              # Decode (WAVE via ryf, else symphonia), resample, windowing, telephony
    runtime/              # Backend seam — THIS is where execution providers are chosen
      factory.rs          # RuntimeFactory / Runtime traits only
      ort/factory.rs      # cfg-gated EP/backend selection (coreml / cuda / nnapi / ane / candle / CPU)
      session.rs · tensor.rs · error.rs
      ort/ · coreml/ · candle/ · mock/
    protocol/mod.rs       # WebSocket JSON message types (Ready, Partial, Final, Error)
    model/                # Model download (streaming + SHA256 + atomic rename)
                          #   (progress / variant / download / cache / manifest)
  gigastt-quantize/proto/
    onnx.proto            # Vendored ONNX protobuf schema (quantizer crate)
  gigastt-ffi/src/        # C-ABI FFI layer (cdylib for Android/mobile)
    lib.rs                # Exported C functions: engine_new, transcribe_file, stream_*, etc.
  gigastt-node/           # napi-rs Node binding
  gigastt-uniffi/         # UniFFI bindings (Swift / Kotlin / Python)
  gigastt/src/            # Server binary + CLI
    lib.rs                # Re-exports gigastt-core::* for backward compat
    main.rs               # CLI (clap): serve, download, transcribe, quantize
    server/
      mod.rs              # axum router, origin middleware, graceful shutdown
      http/                # REST handlers: health (incl. GET /v1/models), transcribe, stream (SSE),
                           #   openai_api, export, jobs_api, admin
      rate_limit.rs       # In-tree per-IP token-bucket rate limiter
      metrics.rs          # In-tree Prometheus text encoder
  gigastt/tests/
    common/mod.rs         # Shared e2e helpers
    benchmark.rs          # WER evaluation (custom harness)
    e2e_*.rs              # E2E test suites
    load_test.rs          # Load tests
    soak_test.rs          # Soak test
sdks/
  go/                   # Go module: typed WS client (protocol v1.0), reconnect honoring retry_after_ms
  js/                   # npm @gigastt/client: TypeScript WS client, Node >= 20 + browsers, vitest
```

## Key Constants

Defined in `crates/gigastt-core/src/inference/mod.rs` (`N_MELS` / `N_FFT` /
`HOP_LENGTH` / `PRED_HIDDEN`) and `engine/mod.rs` (`DEFAULT_POOL_SIZE`):

| Constant | Value | Meaning |
|---|---|---|
| `N_MELS` | 64 | Mel frequency bins |
| `N_FFT` | 320 | FFT window size (20ms @ 16kHz) |
| `HOP_LENGTH` | 160 | Hop length (10ms @ 16kHz) |
| `PRED_HIDDEN` | 320 | Decoder LSTM hidden dim |
| `DEFAULT_POOL_SIZE` | 2 | Concurrent inference sessions (RAM-capped at load) |

## Model Files

Default lean install under `~/.gigastt/models/` (prequantized INT8 from Releases):

| File | Size | Purpose |
|---|---|---|
| `v3_rnnt_encoder_int8.onnx` | ~215 MB | Conformer encoder (INT8; default) |
| `v3_rnnt_decoder.onnx` | ~3.3 MB | LSTM decoder |
| `v3_rnnt_joint.onnx` | ~1.4 MB | RNN-T joiner |
| `v3_vocab.txt` | small | char vocabulary (34 tokens) |

The `e2e_rnnt` head (`--model-variant e2e_rnnt`) uses the parallel `v3_e2e_rnnt_*` filenames with a 1025-token BPE vocab. The multilingual heads `ml_ctc` / `ml_ctc_large` (`--model-variant ml_ctc` / `ml_ctc_large`) are encoder-only: they download the pre-quantized `multilingual_ctc.int8.onnx` (~225 MB) / `multilingual_large_ctc.int8.onnx` (~592 MB) plus `multilingual_vocab.txt` (71-class multilingual char vocab, ru/en/kk/ky/uz) from `istupakov/gigaam-multilingual-ctc-onnx` / `istupakov/gigaam-multilingual-large-ctc-onnx` — no decoder/joiner.

## Development Conventions

### Code style

- Rust 2024 edition
- `anyhow` for internal error handling, `GigasttError` for public API
- `tracing` for logging (never `println!` in library code)
- **No `unwrap()` in production paths** — use `?`, `.context()`, or `unwrap_or_else`
- Shared constants live in `inference/mod.rs`, referenced by sub-modules
- `ort` errors are converted to typed `RuntimeError` at the `runtime/ort` seam (no `anyhow` wrapping)
- Execution provider / backend selection lives in `crates/gigastt-core/src/runtime/ort/factory.rs`
  (`#[cfg(feature = "…")]` blocks, default falls through to the CPU EP) — **not** in `inference/`.
  `runtime/factory.rs` holds only the `RuntimeFactory` / `Runtime` traits.
- **No internal task-tracker IDs outside the tracker itself.** Never write tracker indices (`TTX-NN`, `T-NNN`, `V1-NN`, `SUS-NN`, `TODO-NN`, ticket keys, etc.) into:
  - source comments or code strings
  - `CHANGELOG.md`, `docs/`, CI/workflows, README, user-facing text
  - **git branch names**, **commit subjects/bodies**, **PR titles/descriptions**, tags
  They mean nothing without the tracker and are not conventional git/product language.
  - **Do** describe *what* and *why* in plain English (e.g. branch `ttx/lazy-speaker`, commit `feat(core): lazy-load speaker encoder until diarization is requested`).
  - **Do** keep the link from work → tracked item only in tracker docs: anything under `specs/` (notably `specs/todo.md`, `specs/plan.md`, `specs/prod-readiness-v1.0.md`, `specs/resource-ttx-roadmap.md`, and lab notes under `specs/research/`) or `roadmap/`. Everything outside those two directories must stay index-free.

### TDD workflow

1. Write failing test first
2. Implement minimal code to pass
3. Refactor, verify tests still pass
4. `cargo test --workspace --lib --bins && cargo clippy` before every commit (or `make check`)
5. Enable the pre-commit hook so step 4 is enforced automatically:
   ```sh
   git config core.hooksPath .githooks
   ```

### API versioning & backward compatibility

- WebSocket protocol version: `PROTOCOL_VERSION = "1.0"` (in `protocol/mod.rs`)
- Canonical WS path: `/v1/ws` (v0.7.0+). The `/ws` alias was removed in v0.8.0.
- New fields are **additive only** — never remove or rename existing fields
- Fields like `supported_rates`, `diarization`, `retry_after_ms` use
  `skip_serializing_if` to keep older clients happy
- Breaking changes require a new message type, not modification of existing
- Deprecation: add `deprecated: true` field, support old format for 2 minor versions

### Audio format support

- **File transcription**: WAV family (PCM/IEEE, G.711, G.722, GSM 06.10,
  MS/IMA ADPCM, RF64/RIFX/BW64/Wave64) via `ryf`; M4A/AAC, MP3, OGG/Vorbis, FLAC via
  symphonia; OGG/Opus and `.opus` (Telegram voice) plus WebM/Opus and Matroska
  (a browser's `MediaRecorder` emits nothing else) — symphonia demuxes the
  container, packets are decoded by the pure-Rust BSD-3 `opus-rs` crate,
  mono/stereo only. Packet framing is sliced in-tree per RFC 6716 §3.2 rather
  than by `opus-rs`, whose own parser mis-reads CBR code 3 and long explicit
  frame lengths.
- **Telephony codecs**: G.711 A-law/μ-law, G.722 ADPCM, and GSM 06.10 in WAV
  (via `ryf`; G.722 tags 0x0064/0x0065/0x028F, GSM tag 0x0031 / wav49);
  headerless raw `.ulaw`/`.alaw`/`.g722`
  streams via `?codec=pcmu|pcma|g722&sample_rate=N` on `/v1/transcribe`
  or `transcribe --codec … --sample-rate …` on the CLI (also `ryf`)
- **WebSocket streaming**: raw PCM16 binary frames at configurable sample rate
  (8/16/24/44.1/48 kHz, default 48kHz); resampled to 16kHz server-side via rubato
- Auto mono mix for multi-channel files

## CI / CD

### Workflows

| Workflow | Trigger | What it does |
|---|---|---|
| `.github/workflows/ci.yml` | PR + main push | fmt, clippy, unit tests, feature compile checks (coreml, cuda, diarization, candle, ane), `cargo audit`, `cargo deny` |
| `.github/workflows/soak.yml` | Nightly 03:17 UTC + manual | soak_test + load_test with cached model |
| `.github/workflows/release.yml` | Tag push `v*` + manual | Multi-arch build, tarball + SHA256, CycloneDX SBOM, SLSA provenance, minisign signatures |
| `.github/workflows/homebrew.yml` | Release published | Update Homebrew tap Formula |

### E2E test strategy

- E2E tests run **only on main push**, not on PRs, to keep PR feedback fast
- Model is cached via `actions/cache` with key derived from `crates/gigastt-core/src/model/`
- E2E tests run with `--test-threads=1` because each loads the full ONNX model
  into memory; concurrent runs OOM on CI runners

### Branch protection (repository settings)

The following rules must be enabled in **GitHub Settings → Branches** for `main`:

- **Require a pull request before merging** — direct push to `main` is blocked.
- **Require status checks to pass** — at minimum `fmt`, `clippy`, and `unit-tests` jobs from `ci.yml` must be green.
- **Include administrators** — rules apply to everyone, no exceptions.

This guarantees that a regression like a broken `cargo test` cannot reach `main` even if the local pre-commit hook is bypassed.

## Security Considerations

- **Loopback bind by default.** Server refuses non-loopback addresses unless
  `--bind-all` or `GIGASTT_ALLOW_BIND_ANY=1` is set. Prevents accidental public
  exposure.
- **Origin allowlist.** Cross-origin requests denied by default. Loopback origins
  always allowed. Extra origins via `--allow-origin` (repeatable). Wildcard CORS
  is opt-in via `--cors-allow-any`.
- **Runtime limits** (all configurable via CLI flags and env vars):
  - `--idle-timeout-secs` (default 300) — WebSocket idle timeout
  - `--ws-frame-max-bytes` (default 512 KiB) — max WS frame size
  - `--body-limit-bytes` (default 50 MiB) — max REST body size
  - `--pool-size` (default 2) — concurrent inference sessions
  - `--pool-min-size` (default 1) — minimum triplets required to boot (degraded-pool floor)
  - `--batch-pool-size` (default 0) — triplets reserved for batch REST jobs (0 = shared pool)
  - `--inference-timeout-secs` (default 600) — per-request inference timeout; 0 disables
  - `--max-session-secs` (default 3600) — wall-clock session cap
  - `--shutdown-drain-secs` (default 10) — graceful shutdown drain window
  - `--enable-jobs` (default false) — enable the asynchronous `/v1/jobs` API for long-file
    and batch transcription; when disabled the routes are not registered and return 404
  - `--jobs-ttl-secs` (default 3600) — TTL for finished/failed/cancelled jobs in the store
  - `--jobs-max` (default 100) — max jobs kept in memory; `POST /v1/jobs` returns 429 when full
  - `--jobs-retry` (default 3) — max retries for a job on panic; an
    `inference_timeout` is deterministic and is never retried
- **Per-IP rate limiting** (opt-in, off by default): `--rate-limit-per-minute N`
  enables token-bucket limiter on `/v1/*`; `/health` is exempt. Returns HTTP 429
  + `Retry-After` when exhausted.
- **Pool saturation backpressure.** REST returns 503 + `Retry-After: 30`;
  WebSocket error includes `retry_after_ms: 30000`.
- **SHA-256 verification + atomic rename** on model files. Download stages to
  `.partial`, verifies hash, then atomically renames. Corrupt downloads are
  removed, not promoted.
- **Internal errors sanitized** — no path or model leakage to clients.
- **Prometheus `/metrics`** (opt-in via `--metrics`): 12 metric families —
  HTTP (`gigastt_http_requests_total`, `gigastt_http_request_duration_seconds`),
  pool gauges/counters (`gigastt_pool_available`, `gigastt_pool_waiters`,
  `gigastt_pool_timeouts_total`, batch-pool twins, checkout duration),
  `gigastt_ws_active_connections`, inference duration and
  `gigastt_inference_timeouts_total`, rate-limit rejections.
  Operational detail: [docs/runbook.md](docs/runbook.md).
  Served on a separate loopback listener (default `127.0.0.1:9090`, override
  via `--metrics-listen` / `GIGASTT_METRICS_LISTEN`) — not the main API port,
  and therefore off the CORS allowlist and the per-IP rate limiter.

## Docker

```sh
# CPU (any platform)
docker build -t gigastt .
docker run -p 9876:9876 gigastt

# CUDA (Linux, requires NVIDIA Container Toolkit)
docker build -f Dockerfile.cuda -t gigastt-cuda .
docker run --gpus all -p 9876:9876 gigastt-cuda

# Baked image (model included at build time, ~1.1 GB)
docker build --build-arg GIGASTT_BAKE_MODEL=1 -t gigastt:baked .
```

Docker images run with `--bind-all --host 0.0.0.0` because container networking
requires listening on all interfaces. The non-Docker default is `127.0.0.1`.

## Environment Variables

Most CLI flags map to `GIGASTT_*` env vars (clap `env =`). Canonical flag
reference: [`docs/cli.md`](docs/cli.md) (enforced by `scripts/check-docs-drift.py`).

| Env var | CLI flag / notes | Default |
|---|---|---|
| `GIGASTT_ALLOW_BIND_ANY` | Opt-in for non-loopback bind (same intent as `--bind-all`; not a clap `env=`) | — |
| `GIGASTT_OFFLINE` | Equivalent to global `--offline` | — |
| `GIGASTT_IDLE_TIMEOUT_SECS` | `--idle-timeout-secs` | 300 |
| `GIGASTT_WS_FRAME_MAX_BYTES` | `--ws-frame-max-bytes` | 524288 |
| `GIGASTT_BODY_LIMIT_BYTES` | `--body-limit-bytes` | 52428800 |
| `GIGASTT_RATE_LIMIT_PER_MINUTE` | `--rate-limit-per-minute` | 0 |
| `GIGASTT_RATE_LIMIT_BURST` | `--rate-limit-burst` | 10 |
| `GIGASTT_TRUST_PROXY` | `--trust-proxy` | false |
| `GIGASTT_POOL_MIN_SIZE` | `--pool-min-size` | 1 |
| `GIGASTT_BATCH_POOL_SIZE` | `--batch-pool-size` | 0 |
| `GIGASTT_POOL_CHECKOUT_TIMEOUT_SECS` | `--pool-checkout-timeout-secs` | (see serve help) |
| `GIGASTT_INFERENCE_TIMEOUT_SECS` | `--inference-timeout-secs` | 600 |
| `GIGASTT_MAX_SESSION_SECS` | `--max-session-secs` | 3600 |
| `GIGASTT_SHUTDOWN_DRAIN_SECS` | `--shutdown-drain-secs` | 10 |
| `GIGASTT_ENABLE_JOBS` | `--enable-jobs` | false |
| `GIGASTT_JOBS_TTL_SECS` | `--jobs-ttl-secs` | 3600 |
| `GIGASTT_JOBS_MAX` | `--jobs-max` | 100 |
| `GIGASTT_JOBS_MAX_BYTES` | `--jobs-max-bytes` | 536870912 (512 MiB) |
| `GIGASTT_JOBS_RETRY` | `--jobs-retry` | 3 |
| `GIGASTT_MAX_AUDIO_SECS` | `--max-audio-secs` | 0 (unlimited) |
| `GIGASTT_ENDPOINT_MODE` | `--endpoint-mode` | auto |
| `GIGASTT_STREAM_MAX_WINDOW_SECS` | `--stream-max-window-secs` | 2.5 |
| `GIGASTT_STREAM_STABLE_PREFIX` | `--stream-stable-prefix` | true |
| `GIGASTT_PROFILE` | `--profile` (`default` / `edge`) | default |
| `GIGASTT_DOWNLOAD_PROGRESS` | `download --progress` | human |
| `GIGASTT_METRICS` | `--metrics` | false |
| `GIGASTT_METRICS_LISTEN` | `--metrics-listen` | 127.0.0.1:9090 |
| `GIGASTT_MODEL_VARIANT` | `--model-variant` | rnnt (fresh installs) |
| `GIGASTT_OPTIMIZED_CACHE_DIR` | `--optimized-cache-dir` (also on `cache-gc`) | `<model-dir>/optimized_cache` |
| `GIGASTT_PUNCTUATION` | `--punctuation` | auto |
| `GIGASTT_PUNCT_MODEL_DIR` | `--punct-model-dir` | `~/.gigastt/models/punct/` |
| `GIGASTT_ITN` | `--itn` | auto |
| `GIGASTT_HOTWORDS_FILE` | `--hotwords-file` | — |
| `GIGASTT_HOTWORDS_DEFAULT` | `--hotwords-default` | false |
| `GIGASTT_HOTWORDS_BOOST` | `--hotwords-boost` | 5.0 |
| `GIGASTT_VAD` | `--vad` | false |
| `GIGASTT_VAD_THRESHOLD` | `--vad-threshold` | 0.5 |
| `GIGASTT_VAD_MIN_SILENCE_MS` | `--vad-min-silence-ms` | (see serve help) |
| `GIGASTT_VAD_MODEL_DIR` | `--vad-model-dir` | `~/.gigastt/models/vad/` |
| `GIGASTT_ENCODER_INTRA_THREADS` | `--encoder-intra-threads` | auto |
| `GIGASTT_FILE_WINDOW_CONCURRENCY` | `--file-window-concurrency` | 1 |
| `GIGASTT_FORMAT` | `transcribe --format` | `txt` (`txt,json` for batch / watch) |
| `GIGASTT_OUTPUT` | `transcribe --output` | — |
| `GIGASTT_MAX_CHARS_PER_LINE` | `--max-chars-per-line` | — |
| `GIGASTT_MAX_WORDS_PER_LINE` | `--max-words-per-line` | — |
| `GIGASTT_WORD_TIMESTAMPS` | `--word-timestamps` | false |
| `GIGASTT_BATCH_MOVE_TO` | `transcribe-batch / watch --move-to` | — |
| `GIGASTT_BATCH_DELETE_SOURCE` | `transcribe-batch / watch --delete-source` | false |
| `GIGASTT_BATCH_RETRIES` | `transcribe-batch / watch --retries` | 0 / 2 |
| `GIGASTT_WATCH_POLL_INTERVAL_MS` | `watch --poll-interval-ms` | 1000 |
| `GIGASTT_WATCH_SETTLE_POLLS` | `watch --settle-polls` | 2 |
| `GIGASTT_STEREO_SPEAKERS` | `--stereo-speakers` | false |
| `GIGASTT_CODEC` | `transcribe --codec` | — |
| `GIGASTT_SAMPLE_RATE` | `transcribe --sample-rate` | — |
| `GIGASTT_BAKE_MODEL` | Docker build-arg only (bake model into image) | — |
| `GIGASTT_SOAK_DURATION_SECS` | soak test duration (tests only) | 300 |
| `RUST_LOG` | tracing filter | `gigastt=info` |

`--pool-size` is CLI-only (no `GIGASTT_POOL_SIZE`); default 2 for multi-connection hosts. RAM is no longer the reason to drop to `--pool-size 1` — an extra slot costs only ~20 MB resident (~66 MB pool-2 vs ~46 MB pool-1; `ps` RSS ~510 vs ~277 MB because it counts the shared memory-mapped model). Pool > 1 can still cost ~10–20% single-job RTF (thread split), which is the real edge trade-off. Leave `--encoder-intra-threads` unset (auto); avoid `1` on multi-core (~3× slower; explicit `1` still allowed for debug).

## Useful Commands for Agents

```sh
# Quick iteration cycle
cargo test --workspace --lib --bins && cargo clippy

# Run with model (after `cargo run -- download`)
cargo run --release -- serve
cargo run --release -- transcribe recording.wav

# Check all feature combinations compile
cargo check --features coreml
cargo check --features cuda
cargo check --features ane
cargo check --features candle
cargo check --no-default-features

# Security audit
cargo audit
cargo deny check

# Run a specific e2e test
cargo test -p gigastt --test e2e_ws -- --ignored test_ws_ready_message

# Run with tracing at debug level
RUST_LOG=gigastt=debug cargo run -- serve
```

## Agent Skills

- **rust-skills** ([leonardomso/rust-skills](https://github.com/leonardomso/rust-skills), v1.5.1) — 265 idiomatic Rust rules (ownership, errors, async, unsafe, API design, anti-patterns, …).
  - Canonical install: [`.agents/skills/rust-skills/`](.agents/skills/rust-skills/) (`SKILL.md` + `rules/`).
  - Wired for Claude Code and Grok via symlinks under `.claude/skills/` and `.grok/skills/`.
  - Lockfile: [`skills-lock.json`](skills-lock.json). Restore/update:
    `npx skills experimental_install` or `npx skills add leonardomso/rust-skills -y`.
  - Use when writing, reviewing, or refactoring Rust (`/rust-skills`). Load only the
    relevant `rules/<prefix>-*.md` files (progressive disclosure) — do not dump all 265 into context.

## Notes for AI Agents

- **Always run `cargo test --workspace --lib --bins && cargo clippy` before finishing any change.**
  Never a bare `cargo test` / `cargo test --workspace`: the WER benchmark is a `harness = false`
  target, so `--ignored` does not skip it and the run takes ~2.5 hours.
- When modifying the WebSocket protocol, update `PROTOCOL_VERSION` in
  `protocol/mod.rs` and add tests in `tests/e2e_ws.rs`.
- When adding new CLI flags, add the corresponding env var and document it in
  both `main.rs` and this file.
- The `quantize` Cargo feature enables `crates/gigastt-quantize` (on by default
  for the server binary). Lean embedders may disable it and side-load INT8 only.
- Model download logic is in `crates/gigastt-core/src/model/`. If you change HF repo or file
  names, update the per-head checksum tables (`RNNT_CHECKSUMS`, `E2E_RNNT_CHECKSUMS`,
  `ML_CTC_CHECKSUMS`, `ML_CTC_LARGE_CHECKSUMS` in `model/variant.rs`) and the cache key in
  `.github/workflows/ci.yml`.
- The project uses English for all code comments, documentation, and commit
  messages.
