# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**gigastt** — local speech-to-text server powered by GigaAM v3 (default `rnnt` head, optional `e2e_rnnt`), with optional GigaAM Multilingual `ml_ctc` (220M) / `ml_ctc_large` (600M) heads (ru/en/kk/ky/uz). On-device speech recognition via ONNX Runtime. No cloud APIs, no API keys, full privacy.

- **Repository**: https://github.com/ekhodzitsky/gigastt
- **crates.io**: https://crates.io/crates/gigastt
- **License**: MIT

## Build & Test

```sh
cargo build                          # CPU-only debug build (default, any platform)
cargo build --features coreml        # macOS ARM64 (CoreML / Neural Engine)
cargo build --features cuda          # Linux x86_64 (CUDA 12+)
cargo build --features ane           # macOS ARM64 native ANE (file mode)
cargo build --features candle        # experimental Candle/Metal
cargo build --release                # Release build (LTO, stripped)
cargo test --workspace --lib --bins  # Run all unit tests, CPU (no model required)
cargo test --workspace --lib --bins --features coreml  # Same tests with CoreML EP enabled (macOS)
cargo test --test e2e_rest --test e2e_ws --test e2e_errors --test e2e_shutdown --test e2e_rate_limit --test e2e_jobs --test e2e_cli --test e2e_admin_reload --test e2e_http_cov -- --ignored --test-threads=1  # E2E tests (requires model)
cargo test --test load_test -- --ignored           # Load tests (requires model, local only)
cargo test --test soak_test -- --ignored           # Soak test (requires model, local only)
cargo clippy             # Lint (no expected warnings)
```

Note: `cargo build` requires `protoc` in `PATH` for the in-tree ONNX quantization pipeline (see `build.rs`). Install via `brew install protobuf` (macOS) or `apt install protobuf-compiler` (Debian/Ubuntu).

Note (build-time network fetch): `ort`'s default `download-binaries` feature makes `ort-sys` download a prebuilt onnxruntime native library over the network at build time, outside `Cargo.lock` (the download is verified by an embedded checksum). The "no cloud / full privacy" guarantee covers **runtime** inference, not the build process. For air-gapped/offline builds, use `ort` with `default-features = false` + the `load-dynamic` feature (or a vendored onnxruntime) and pin the native library via `ORT_*` env vars / `.cargo/config.toml`. See `docs/embedding-packaging.md` (static-default vs `ort-load-dynamic`) and `docs/quickstarts.md` (in-process Python/Node/Swift/Kotlin examples).

Model download (required for E2E testing and file transcription, ~225 MB INT8):
```sh
cargo run -- download                    # Lean INT8 bundle (~225 MB) from Releases (only path)
cargo run -- quantize                    # Packaging: rebuild INT8 from a local FP32 ONNX
```

Runtime is **INT8 only** — no FP32 download flags, no FP32 engine load.

## Docker

Multi-stage production build:
```sh
# CPU / macOS (default Dockerfile)
docker build -t gigastt .
docker run -p 9876:9876 gigastt
# Model auto-downloads on first run, binds to 0.0.0.0:9876

# CUDA (Linux, requires NVIDIA Container Toolkit)
docker build -f Dockerfile.cuda -t gigastt-cuda .
docker run --gpus all -p 9876:9876 gigastt-cuda

# Bake the model into the image (zero cold-start, ~1.1 GB image):
docker build --build-arg GIGASTT_BAKE_MODEL=1 -t gigastt:baked .
```

The Dockerfile passes `--bind-all` so the server listens on `0.0.0.0` inside the container. Local deployments use `127.0.0.1` by default; `--bind-all` (or `GIGASTT_ALLOW_BIND_ANY=1`) is required to listen on non-loopback addresses.

## Architecture

```
crates/
  gigastt-core/src/       # Core library (inference engine, no server deps)
    lib.rs                # Public module exports
    model/                # Model download (GitHub Releases INT8; HF for CTC / sidecars)
    inference/
      mod.rs              # Module wiring + shared constants
      engine/             # Engine: load, warmup, transcribe, streaming (split impl)
      pool.rs             # SessionPool (checkout, backpressure)
      state.rs            # StreamingState / DecoderState
      features.rs         # Mel spectrogram (64 bins, FFT=320, hop=160, HTK)
      tokenizer.rs        # Vocabulary per head: char (rnnt) / BPE (e2e_rnnt) / multilingual char (ml_ctc)
      decode/             # RNN-T greedy decode loop
      ctc.rs              # Greedy CTC decode (ml_ctc heads)
      bias.rs             # Hotword biasing
      diarization.rs      # polyvoice glue (Embedder adapter, offline + streaming)
      types.rs            # TranscribeRequest / TranscribeResult
      audio/              # Decode, resample, channel mixing, windowing, VAD windows, telephony
    runtime/              # Backend seam: EP/backend selection lives here, NOT in inference/
      factory.rs          # RuntimeFactory / Runtime traits
      ort/factory.rs      # cfg-gated coreml / cuda / nnapi / ane / candle / CPU
      ort/ · coreml/ · candle/ · mock/
    error.rs              # Typed error types (GigasttError)
    protocol/mod.rs       # JSON message types (Ready, Partial, Final, Error + retry_after_ms)
  gigastt-quantize/       # Native Rust INT8 dynamic quantizer (optional feature)
  gigastt-ffi/src/        # C-ABI FFI layer (cdylib for Android/mobile)
    lib.rs                # Exported C functions: engine_new, transcribe_file, stream_*, etc.
  gigastt-node/           # napi-rs Node binding
  gigastt-uniffi/         # UniFFI bindings (Swift / Kotlin / Python)
  gigastt/src/            # Server binary + CLI
    lib.rs                # Re-exports gigastt-core::* for backward compat
    main.rs               # CLI (clap): serve, download, transcribe, quantize
    server/
      mod.rs              # axum router: HTTP + WebSocket on single port, origin middleware, graceful drain
      http/               # REST handlers: health (incl. GET /v1/models), transcribe, stream (SSE), openai_api, export, jobs_api, admin
      rate_limit.rs       # In-tree per-IP token-bucket rate limiter (parking_lot Mutex + HashMap)
      metrics.rs          # In-tree Prometheus text encoder (counters + histograms)
```

### Performance optimizations (v0.9)
- **CoreML execution provider** (`--features coreml`, macOS ARM64): MLProgram format, static-shape subgraphs only
  - Dynamic-shape CoreML partitions fail at prediction time (issue #42), so the EP is configured with `RequireStaticInputShapes`: heavy conv/matmul blocks run on the Neural Engine, dynamic-shape ops stay on the CPU EP (~3x faster encoder on a 4 s clip, ~5.6x on a 2-minute file vs pure-CPU build; M1 Pro, INT8, release)
  - Startup warmup probe (~1 s of silence through the full pipeline) verifies CoreML actually executes; on failure (session load OR first Run) the engine logs `falling back to CPU execution provider` and rebuilds all sessions on the CPU EP — never crashes
  - `Engine::warmup()` warms every pooled triplet; the server calls it before `axum::serve`
  - Automatically loads quantized encoder if available (~4x smaller)
  - Caches compiled models in `~/.gigastt/models/coreml_cache/`
- **CUDA execution provider** (`--features cuda`, Linux x86_64 CUDA 12+): GPU inference via ONNX Runtime CUDA EP
  - Features are compile-time and mutually exclusive; default build uses CPU EP on all platforms
- **INT8 only**: `download` / `serve` / engine load use lean prequantized INT8 (~225 MB)
  - Engine rejects FP32-only installs (no fallback)
  - `gigastt quantize` is packaging-only (needs local FP32 source)
- **Zero-copy REST upload path** (v0.9.0): `bytes::Bytes` flows end-to-end from axum into decode — WAVE via ryf (`Cursor<Bytes>`), other containers via a crate-private `BytesMediaSource` into symphonia — eliminating the 4× upload copy that used to OOM small containers on concurrent 10-minute uploads.

### Key constants (defined in `crates/gigastt-core/src/inference/mod.rs`)
- `N_MELS = 64`, `N_FFT = 320`, `HOP_LENGTH = 160`, `PRED_HIDDEN = 320`
- Encoder dim: 768 (shared across heads). Vocab depends on the head: `rnnt` 34 tokens (char, default) / `e2e_rnnt` 1025 tokens (BPE) / `ml_ctc` + `ml_ctc_large` 71 tokens (multilingual char, blank id 70)

### Data flow
```
Audio (PCM16) → Mel Spectrogram → Conformer Encoder (ONNX)
  → RNN-T Decoder+Joiner loop → tokens → Text          (rnnt / e2e_rnnt)
  → greedy CTC decode → tokens → Text                  (ml_ctc / ml_ctc_large, no decoder/joiner)
```

### Streaming
- `StreamingState` persists LSTM h/c and audio buffer across WebSocket chunks
- `DecoderState` holds decoder hidden state (h, c, prev_token)
- Server accepts configurable sample rates (8kHz, 16kHz, 24kHz, 44.1kHz, 48kHz) via `Configure` message
- Default 48kHz for backward compatibility; resamples to 16kHz via rubato (polyphase FIR)
- Odd-length PCM16 frames are carried across to the next frame (v0.9.0) to avoid single-byte phase drift.

### Graceful shutdown (v0.9.0)
- A single `CancellationToken` + `TaskTracker` cascades through every WebSocket / SSE handler.
- On SIGTERM each live session flushes, emits an empty-if-needed `Final`, and closes with `Close(1001 Going Away)`.
- After `axum::serve` returns, `run_with_config` waits up to `--shutdown-drain-secs` (default 10) for the tracker to drain.
- A wall-clock `--max-session-secs` cap (default 3600) closes silence-stream DoS attempts with `Close(1008) + code=max_session_duration_exceeded`.

## Development guidelines

### TDD workflow
1. Write failing test first
2. Implement minimal code to pass
3. Refactor, verify tests still pass
4. `cargo test --workspace --lib --bins && cargo clippy` before every commit

### API versioning & backward compatibility
- WebSocket protocol version: `PROTOCOL_VERSION = "1.0"` (in `crates/gigastt-core/src/protocol/mod.rs`)
- `ServerMessage::Ready` includes `version` field sent on connection
- WebSocket path: `/v1/ws`
- WebSocket protocol messages are versioned via `type` field
- New fields are additive only (never remove or rename existing fields). `supported_rates`, `diarization`, and `retry_after_ms` are serialized with `skip_serializing_if` to keep older clients happy.
- Breaking changes require new message type, not modification of existing
- Deprecation: add `deprecated: true` field, support old format for 2 minor versions

### Testing

Three-tier test architecture:

**Unit tests** (no model required, run in CI on every PR):
- Live in `#[cfg(test)] mod tests` at bottom of each file
- Use synthetic data, test names: `test_<what>_<expected_behavior>`
- 600+ unit tests across the workspace (`cargo test --workspace --lib --bins`)
- Always pass `--lib --bins`. A bare `cargo test` (or `cargo test --workspace`) pulls in the
  ~2.5-hour WER benchmark: it is a `harness = false` target, so `--ignored` does not skip it.

**E2E tests** (require model ~225 MB INT8, run in CI on main push only):
- `tests/e2e_rest.rs` — REST API tests (health, transcribe, SSE streaming, error paths)
- `tests/e2e_ws.rs` — WebSocket protocol tests (ready, audio, stop, configure, errors, concurrent)
- `tests/e2e_errors.rs` — error path tests (oversized body/frame, pool saturation, idle timeout)
- `tests/e2e_shutdown.rs` — graceful shutdown tests (WS final + close, SSE termination, max-session cap, shutdown under pool saturation)
- `tests/e2e_rate_limit.rs` — per-IP rate limiter 429 behavior
- `tests/e2e_jobs.rs` — async `/v1/jobs` queue
- `tests/e2e_cli.rs` — CLI transcribe / batch / watch smoke
- `tests/e2e_admin_reload.rs` — loopback `POST /v1/admin/reload`
- `tests/e2e_http_cov.rs` — extra HTTP/export coverage
- `tests/common/mod.rs` — shared helpers (start_server with shutdown handle, WAV generation, WS connect)
- `cargo test --test e2e_rest --test e2e_ws --test e2e_errors --test e2e_shutdown --test e2e_rate_limit --test e2e_jobs --test e2e_cli --test e2e_admin_reload --test e2e_http_cov -- --ignored --test-threads=1`

**Load/soak tests** (require model, run locally + nightly CI via `.github/workflows/soak.yml`):
- `tests/load_test.rs` — 3 load tests (concurrent WS, concurrent REST, burst connections)
- `tests/soak_test.rs` — 1 soak test (continuous WS cycling, configurable via `GIGASTT_SOAK_DURATION_SECS`)
- `cargo test --test load_test -- --ignored` / `cargo test --test soak_test -- --ignored`

**Long-form quality tests** (require model + the RuLS corpus, local-only — the corpus is a ~9 GB
OpenSLR download that does not fit the CI cache budget, so these never run in CI):
- `tests/longform_quality.rs` — stitch cost of the chunked long-form path against a length-matched
  segment baseline, plus the encoder-length degradation curve. Both skip loudly when the corpus is
  absent; they never substitute another corpus.
- `python3 scripts/prepare_rulslib.py` to fetch the corpus, then
  `cargo test --release -p gigastt --test longform_quality -- --ignored --test-threads=1`
- Default ceiling +2.0 pp on the stitch cost; override with `GIGASTT_LONGFORM_MAX_STITCH_PP`.

**Benchmark suite:**
- `tests/benchmark.rs` — WER evaluation on Golos fixtures (custom harness, `harness = false`)

### CI structure
- **PR CI** (`.github/workflows/ci.yml`, fast): fmt, clippy, unit tests, feature compile checks (CoreML, CUDA, Diarization, Candle/Metal, ANE), `cargo audit`, `cargo deny`
- **Main push CI**: all PR checks + e2e tests with cached model (~225 MB INT8, OS-independent cache key) + CoreML runtime smoke on macos-14 (transcribes `golos_00.wav`, fails on inference error or silent CPU fallback)
- **Nightly soak** (`.github/workflows/soak.yml`): `cargo test --test soak_test` at 03:17 UTC, reuses the main CI model cache
- **Release** (`.github/workflows/release.yml`, tag-triggered): multi-arch tarballs, per-asset `.sha256` + `SHA256SUMS.txt`, CycloneDX SBOM, SLSA provenance, minisign signatures
- Load tests are local-only, not in CI

### Code style
- Rust 2024 edition
- `anyhow` for error handling, `tracing` for logging
- No `unwrap()` in production paths (use `?`, `context()`, or `unwrap_or_else`)
- Shared constants in `crates/gigastt-core/src/inference/mod.rs`, referenced by sub-modules
- `ort` errors are converted to typed `RuntimeError` at the `runtime/ort` seam (no `anyhow` wrapping)
- Execution provider / backend selection lives in `crates/gigastt-core/src/runtime/ort/factory.rs` (`#[cfg(feature = "…")]` blocks for coreml / cuda / nnapi / ane / candle); default falls through to CPU EP. `runtime/factory.rs` is the trait surface only. It is **not** in `inference/`
- **No internal task-tracker IDs outside the tracker itself.** Never write tracker indices (`TTX-NN`, `T-NNN`, `V1-NN`, `SUS-NN`, `TODO-NN`, ticket keys, etc.) into source comments/code, `CHANGELOG.md`, `docs/`, CI/workflows, README, user-facing text, **git branch names**, **commit subjects/bodies**, or **PR titles/descriptions**. They are noise without the tracker. Use conventional language only (e.g. branch `ttx/lazy-speaker`, commit `feat(core): lazy-load speaker encoder…`). Link work to a tracked item only inside tracker docs: anything under `specs/` (notably `specs/todo.md`, `specs/plan.md`, `specs/prod-readiness-v1.0.md`, `specs/resource-ttx-roadmap.md`, and lab notes under `specs/research/`) or `roadmap/` — both are the tracker. Everything outside those two directories must stay index-free.

### Audio format support
- File transcription: WAV family via `ryf` (PCM/IEEE, G.711, G.722, GSM 06.10, ADPCM, RF64); M4A/AAC, MP3, OGG/Vorbis, FLAC (via symphonia); OGG/Opus and WebM/Opus (symphonia demux + the pure-Rust `opus-rs` decoder)
- WebSocket: raw PCM16 binary frames at configurable sample rate (8kHz/16kHz/24kHz/44.1kHz/48kHz, default 48kHz); resampled to 16kHz server-side via rubato
- Auto mono mix for multi-channel files

### Security
- **Loopback bind by default.** `127.0.0.1` only; `--bind-all` / `GIGASTT_ALLOW_BIND_ANY=1` required for non-loopback.
- **Origin allowlist.** Cross-origin callers denied by default; loopback origins always allowed. `--allow-origin` (repeatable) for explicit additions; `--cors-allow-any` for wildcard.
- **Runtime limits** via CLI / env: `--idle-timeout-secs` (default 300), `--ws-frame-max-bytes` (512 KiB), `--body-limit-bytes` (50 MiB), `--max-session-secs` (3600), `--shutdown-drain-secs` (10). `--pool-size` (default 2) is **CLI-only** (no `GIGASTT_POOL_SIZE`). RAM after mmap: ~46 MB resident / ~277 MB `ps` RSS at pool 1, ~66 / ~510 at pool 2.
- **Per-IP rate limiting** (v0.8.0, opt-in): `--rate-limit-per-minute N` + `--rate-limit-burst` on `/v1/*` (`/health` exempt); HTTP 429 + `Retry-After` when exhausted.
- **Pool saturation backpressure.** REST returns 503 + `Retry-After: 30`; WebSocket error includes `retry_after_ms: 30000`.
- **SHA-256 verification + atomic rename** on both encoder/decoder/joiner model files and the optional speaker diarization model.
- **Internal errors sanitized** — no path or model leakage to clients.
- **Prometheus `/metrics`** (v0.8.0, opt-in via `--metrics`): `gigastt_http_requests_total`, `gigastt_http_request_duration_seconds`. Served on a separate loopback listener (default `127.0.0.1:9090`, override via `--metrics-listen` / `GIGASTT_METRICS_LISTEN`), off the CORS allowlist and per-IP rate limiter; the primary port no longer serves `/metrics`.

## Model

Runtime is **INT8 only** — `download` / `serve` / engine load never use FP32.
Four selectable heads via `--model-variant` — an explicit value is honored even when
the model dir holds more than one head's files (fixed in 2.11.1); with no flag the
head is auto-detected from the files on disk (`rnnt` precedence):
- **`rnnt`** (default since v2.3): lean INT8 set from GitHub Releases —
  `v3_rnnt_encoder_int8.onnx` + `v3_rnnt_{decoder,joint}.onnx` + `v3_vocab.txt`
  (34-token char vocab). Much lower WER than e2e (clean read 3.55% on
  `golos_crowd_1k` via the cross-engine harness vs e2e 8.60%; leads
  far-field/phone/YouTube — see `docs/benchmarks.md`); bare lowercase output, so
  pair with `--punctuation` / `--itn` for readable text.
- **`e2e_rnnt`**: `v3_e2e_rnnt_encoder_int8.onnx` + `v3_e2e_rnnt_{decoder,joint}.onnx`
  + `v3_e2e_rnnt_vocab.txt` (1025-token BPE). Punctuation/casing/ITN baked in.
- **`ml_ctc`**: GigaAM Multilingual charwise-CTC head (220M) from
  `istupakov/gigaam-multilingual-ctc-onnx`. Encoder-only:
  `multilingual_ctc.int8.onnx` + `multilingual_vocab.txt` (71-class multilingual
  char vocab; blank id 70). Downloads the upstream pre-quantized INT8 encoder
  directly (~225MB). Best-in-class WER on ru/kk/ky/uz (moderate on en); bare
  lowercase output. Shares the 64-mel frontend; file transcription (greedy CTC
  decode, no prediction network / joiner).
- **`ml_ctc_large`**: the 600M GigaAM Multilingual head from
  `istupakov/gigaam-multilingual-large-ctc-onnx`
  (`multilingual_large_ctc.int8.onnx`, ~592MB; shares `multilingual_vocab.txt`).
  Same charwise-CTC architecture as `ml_ctc`, higher accuracy across all five
  languages (clean-read WER ru 4.44 / en 4.63 / kk 6.52 / ky 7.39 / uz 9.21% —
  see `docs/benchmarks.md`).
- Encoder (rnnt/e2e_rnnt): ~215 MB INT8; decoder/joiner a few MB each. Total lean
  install ~225 MB.
- Sample rate: 16kHz, Features: 64 mel bins
- ONNX tensors: encoder out `[1, 768, T]` (channels-first), decoder state `[1, 1, 320]`

### Quantization

Product path never quantizes on device — INT8 is pre-shipped. `gigastt quantize`
is packaging-only and needs a **local FP32** encoder ONNX as source
(`crates/gigastt-quantize`):
```sh
cargo run -- quantize --model-dir ~/.gigastt/models
# Requires v3_*_encoder.onnx on disk; writes v3_*_encoder_int8.onnx
```

Engine **requires** the INT8 encoder; FP32-only installs are rejected.

## Agent Skills

- **rust-skills** — 265 idiomatic Rust rules for write/review/refactor. Path: `.agents/skills/rust-skills/` (also `.claude/skills/rust-skills`). Invoke with `/rust-skills`; open only relevant `rules/` files. See [`AGENTS.md`](AGENTS.md) § Agent Skills.

## Known limitations
- CPU EP runs on any platform; CoreML EP requires macOS ARM64; CUDA EP requires Linux x86_64 with CUDA 12+
- `protoc` must be on `PATH` at build time (in-tree ONNX quantization pipeline regenerates types via `prost-build`)
- The model can be hot-reloaded after `serve` boot without a restart via the loopback-only `POST /v1/admin/reload` endpoint (rebuilds the engine from the boot recipe, warms it, then atomically swaps; keeps the old engine on failure). Replacing the model *files* on disk still requires that endpoint (or a restart) for the change to take effect.
- Agent-facing setup and conventions: prefer [`AGENTS.md`](AGENTS.md) as the longer form; keep this file aligned when changing build/test commands.

## Which file owns what

This file is loaded into every agent's context automatically; `AGENTS.md` is not.
So both stay self-contained on the essentials, and the two overlap on purpose:

- **This file** — the operational minimum an agent needs without opening anything
  else: build/test commands, the code map, key constants, and the hard rules
  (safe test invocation, no `unwrap()` in production paths, loopback-only bind,
  no tracker indices outside `specs/` and `roadmap/`).
- **[`AGENTS.md`](AGENTS.md)** — the canonical long form: full crate layout,
  dependency and feature matrix, test tiers, CI structure, release process.

When a fact appears in both, `AGENTS.md` wins and this file must be updated to
match. Facts that live in exactly one place: the shipped state of the code is
`CHANGELOG.md`. The standing local task queue is Backlog.md — see
[agents/backlog.md](agents/backlog.md).

Coding principles for this repo live in [agents/coding-principles.md](agents/coding-principles.md) — read them before writing code.
