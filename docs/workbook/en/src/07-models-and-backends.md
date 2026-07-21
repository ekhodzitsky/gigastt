# Models and backends

## Scenario

You have gigastt running with the default `rnnt` head on the CPU backend, and
now you need to make an informed change: a different recognition head
(punctuation baked in, or languages beyond Russian), a leaner model download, a
faster execution provider for your hardware, or a bigger session pool. This
chapter answers four questions with checkable recipes: **which head**, **INT8
or FP32**, **which backend**, and **how much RAM** the pool needs.

WER and RTF numbers are **not duplicated here** — the canonical, CI-annotated
tables live in
[docs/benchmarks.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/benchmarks.md);
this chapter links into them. Flags are checked against `gigastt <command>
--help`; the full flag reference lives in
[docs/cli.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/cli.md).

## Prerequisites

- gigastt installed (binary, package, or image) — see
  [Getting started](01-getting-started.md).
- Disk: ~1.1 GB free for the default FP32-download path (FP32 set + the
  generated INT8 encoder), or ~250 MB for the lean pre-quantized path.
- For building a non-default backend from source: Rust 1.88+ and `protoc` on
  `PATH` (build requirements:
  [docs/architecture.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/architecture.md)).

## Recipe

### Choosing a recognition head

The recognition head is selected with `--model-variant` (env
`GIGASTT_MODEL_VARIANT`) on `serve` / `download` / `transcribe`. All heads
share the mel frontend and the 16 kHz mono input contract; they differ in
ONNX files, vocabulary, and decoding.

| Head | Size on disk | Languages | Output text | Accuracy | Pick when |
|---|---|---|---|---|---|
| `rnnt` (default) | 844 MB FP32 encoder → ~215 MB INT8 (auto-quantized) + decoder/joiner/vocab (a few MB) | Russian | Bare lowercase; pair with `--punctuation` / `--itn` (on by default in `auto`) | Lowest Russian WER of the four — [table](https://github.com/ekhodzitsky/gigastt/blob/main/docs/benchmarks.md#accuracy-by-domain--wer--95-ci) | Russian-only workloads; the default for a reason |
| `e2e_rnnt` | Same size class as `rnnt` (~850 MB FP32 → INT8 generated locally) | Russian | Punctuation / casing / ITN **baked in**, one pass | Higher WER than `rnnt`, but the best punctuation/casing F1 — [comparison](https://github.com/ekhodzitsky/gigastt/blob/main/docs/benchmarks.md#punctuation-quality--e2e_rnnt-vs-rnnt--rupunct-restore) | You want readable Russian in a single pass with no restore step |
| `ml_ctc` | ~225 MB pre-quantized INT8, encoder-only (no decoder/joiner) | ru/en/kk/ky/uz | Bare lowercase | [Multilingual tables](https://github.com/ekhodzitsky/gigastt/blob/main/docs/benchmarks.md#english--wer--librispeech-test-clean) | Multilingual audio on a small footprint |
| `ml_ctc_large` | ~592 MB pre-quantized INT8, encoder-only | ru/en/kk/ky/uz | Bare lowercase | Best multilingual accuracy; approaches `rnnt` on Russian clean read — [table](https://github.com/ekhodzitsky/gigastt/blob/main/docs/benchmarks.md#accuracy-by-domain--wer--95-ci) | Mixed-language audio, or English/Kazakh/Kyrgyz/Uzbek at all |

Two hard constraints the table implies:

- `rnnt` / `e2e_rnnt` have a **Cyrillic-only** vocabulary — they cannot
  transcribe English at all. For English (or kk/ky/uz) the Multilingual heads
  are the only option.
- The Multilingual heads are encoder-only CTC: they always ship and run as
  pre-quantized INT8 — there is no FP32 download and no on-device quantization
  step for them.

Download and run a different head:

```sh
gigastt download --model-variant e2e_rnnt
gigastt serve --model-variant e2e_rnnt
```

Auto-detection rules (from `crates/gigastt-core/src/model/mod.rs`):

- No `--model-variant` → the engine detects the installed head from the files
  in the model directory and uses a complete install **as-is** (no network
  request at all).
- An explicit `--model-variant` that differs from the installed head → the
  requested set is downloaded **alongside**; variants are never mixed within
  one inference, and a warning is logged. When several heads' files coexist
  and no variant is requested, auto-detect prefers `rnnt`.
- An empty model directory + no flag → the default `rnnt` is downloaded.

**Verify:**

```sh
curl -s http://127.0.0.1:9876/health
# {"status":"ok","model":"gigaam-v3-e2e-rnnt","variant":"e2e_rnnt",...}
curl -s http://127.0.0.1:9876/v1/models
# .id / .name reflect the loaded head; .encoder reports int8 vs fp32
```

### INT8 or FP32

Short answer: **INT8, always, unless you are debugging the model itself.** The
INT8 encoder runs as true integer compute (`DynamicQuantizeLinear` +
`MatMulInteger`/`ConvInteger` kernels), shrinks the encoder 844 MB → 215 MB
(~3.9×), and measures ~0% WER degradation — numbers and methodology:
[docs/benchmarks.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/benchmarks.md#headline-single-engine-metrics).

What happens on each path (RNN-T heads):

- **Default** — `gigastt download` (or the first `serve` / `transcribe`)
  fetches the FP32 set from HuggingFace (`istupakov/gigaam-v3-onnx`), then
  runs the native-Rust quantization pass once (~2 min) and writes
  `v3_rnnt_encoder_int8.onnx` next to it. The engine **prefers the INT8
  encoder** whenever the file is present.
- **Lean** — `gigastt download --prequantized` fetches the pre-quantized INT8
  bundle (INT8 encoder + decoder + joiner + vocab) from the pinned GitHub
  Release instead: no ~844 MB FP32 download, no ~2-minute quantization. Also
  the fallback when HuggingFace is unreachable but GitHub is not.
- **Multilingual** — `ml_ctc` / `ml_ctc_large` download istupakov's
  pre-quantized INT8 encoder directly from HuggingFace; `--prequantized` is a
  no-op refinement for them (there is no separate bundle).
- **Manual** — `gigastt quantize [--force]` re-runs the quantization on the
  head detected in `--model-dir` (e.g. after replacing the FP32 encoder with
  your own fine-tune).
- **Opt out** — `--skip-quantize` (env `GIGASTT_SKIP_QUANTIZE=1`) skips the
  quantization step; the engine then loads the FP32 encoder at ~4× the RAM per
  pool slot and slower CPU kernels. Keep this for debugging only.

**Verify:**

```sh
ls ~/.gigastt/models/
# v3_rnnt_encoder_int8.onnx present alongside decoder/joint/vocab
gigastt transcribe sample.wav 2>&1 | grep 'transcribe complete'
# ... encoder=int8/cpu ... rtf=0.1xx
```

The `encoder=int8/<backend>` field in the completion log line is the ground
truth for which encoder file was loaded.

### Execution provider for your hardware

The backend is a **compile-time Cargo feature**, not a runtime flag — the
provider is baked into the binary you install or build:

| Your hardware | Feature | Provider | Notes |
|---|---|---|---|
| Any (default) | — | CPU (ONNX Runtime) | The reference build; RTF well under 1.0 with the INT8 encoder |
| macOS ARM64 (M1–M4) | `--features coreml` | CoreML + Neural Engine | Prebuilt macOS release binaries **already ship this feature**; ~3× encoder on short clips, ~5.6× on long files vs CPU |
| Linux x86_64 + NVIDIA | `--features cuda` | CUDA 12+ | No prebuilt tarball — use the `-cuda` Docker image or `Dockerfile.cuda`; falls back to CPU when no GPU is present at runtime |
| Android / ARM64 | `--features nnapi` | NNAPI (NPU/DSP) | Not mutually exclusive with the others |
| macOS ARM64, experimental | `--features ane` | Native Apple Neural Engine (Core ML `.mlpackage`) | `rnnt` head only, file-mode acceleration; see below |
| Apple Silicon, experimental | `--features candle` | Pure-Rust Candle on Metal | `rnnt` head only, FP32; see below |

Build the one that matches:

```sh
cargo build --release                      # CPU, any platform
cargo build --release --features coreml    # macOS ARM64
cargo build --release --features cuda      # Linux x86_64 + NVIDIA (CUDA 12+)
cargo build --release --features nnapi     # Android / ARM64
```

Exclusivity is enforced at compile time: `coreml` and `cuda` are mutually
exclusive; `ane` conflicts with `coreml`/`cuda`/`nnapi`/`candle`; `candle`
conflicts with `coreml`/`cuda`. A `compile_error!` fires on a bad combination.

Runtime behavior worth knowing:

- **CPU fallback is deliberate, never a crash.** The CoreML build runs a ~1 s
  warmup probe at startup; on failure it logs `falling back to CPU execution
  provider` and rebuilds the sessions on CPU. The CUDA binary likewise runs on
  CPU when no GPU is visible (e.g. a container started without `--gpus all`).
- **CUDA packaging.** There is no prebuilt CUDA tarball — the release matrix
  ships CPU binaries for Linux (x86_64, aarch64), Windows, and a
  CoreML-enabled macOS ARM64 binary. For GPU use the published
  `ghcr.io/ekhodzitsky/gigastt:<ver>-cuda` image (recipes:
  [Deployment & ops](06-deployment-ops.md)) or build with
  [Dockerfile.cuda](https://github.com/ekhodzitsky/gigastt/blob/main/Dockerfile.cuda).
- **Native ANE backend** (`--features ane`, macOS ARM64): runs the `rnnt`
  encoder on the Neural Engine via per-bucket fixed-shape Core ML packages,
  ~10× warm end-to-end over the CPU build (decode-bound). Fetch the packages
  with `gigastt download --ane` (into `~/.gigastt/models/ane/`); an `e2e_rnnt`
  model transparently falls back to the `ort` encoder, and streaming windows
  always take the CPU path — ANE is a file-mode accelerator. Full design and
  honest numbers:
  [docs/ane-backend.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/ane-backend.md).
- **Candle backend** (`--features candle`, experimental): pure-Rust inference
  on the Metal GPU, byte-for-byte parity with `ort`, `rnnt` head only, FP32
  weights converted once with `scripts/convert_gigaam_candle.py`. Details:
  [docs/candle-backend.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/candle-backend.md).

**Verify:**

```sh
gigastt transcribe sample.wav 2>&1 | grep 'transcribe complete'
# encoder=int8/coreml  → the CoreML build is active (int8/cuda, int8/cpu, ...)
# encoder=int8/ane     → the native ANE backend handled the encoder
```

On a CoreML build, also watch startup: the absence of `falling back to CPU
execution provider` means the warmup probe passed.

### Model files in an offline perimeter

Everything the engine needs lives in one directory — `~/.gigastt/models/` by
default, overridable with `--model-dir` on `serve` / `download` / `transcribe`
/ `quantize`. That makes the model set a plain file artifact you can stage and
move around:

```sh
# On a connected machine:
gigastt download --prequantized --model-dir /srv/gigastt-models

# Copy to the target host any way you like (rsync, USB, artifact store):
rsync -a /srv/gigastt-models/ offline-host:/srv/gigastt-models/

# On the offline host — refuse every network fetch, fail fast instead of hanging:
gigastt --offline serve --model-dir /srv/gigastt-models
```

Properties that make this safe:

- **Integrity is verified by the binary, not by you.** Every download is
  SHA-256-checked against checksums pinned in the code, staged as `.partial`,
  and atomically renamed into place; a corrupt file is removed, never
  promoted. A checksum failure exits with code `65` (network `69`, disk `74`,
  Ctrl-C `130`) — scriptable.
- **A complete model set means zero network.** With a full head's files
  present (encoder — INT8 or FP32 — plus decoder/joiner/vocab for the RNN-T
  heads, or INT8 encoder + vocab for the Multilingual ones), startup performs
  no network request, even without `--offline`. `--offline` /
  `GIGASTT_OFFLINE=1` turns any *missing* optional model (punctuation, VAD,
  diarization) into an immediate error naming the file to provide.
- **The head is auto-detected from the files** — a copied directory needs no
  `--model-variant` flag on the target host.
- When copying, include the `*_int8.onnx` encoder (or copy the FP32 encoder
  too and run `gigastt quantize --model-dir …` on the target), the vocab, and
  — for `rnnt`/`e2e_rnnt` — the decoder and joiner.

For a fully packaged offline install (tarball with binary + INT8 model +
punctuation model + systemd unit, or the two-deb variant), use the release
offline bundle — recipe and verification steps in
[Deployment & ops](06-deployment-ops.md); inventory in
[README-OFFLINE.md](https://github.com/ekhodzitsky/gigastt/blob/main/packaging/offline/README-OFFLINE.md).

**Verify:**

```sh
gigastt --offline transcribe sample.wav --model-dir /srv/gigastt-models
# transcribes with no network; a missing file errors immediately, naming the file
```

### Sizing the session pool (RAM)

Every pool slot deserializes **its own encoder copy**, so RSS scales linearly
with `--pool-size` (default 2). The engine budgets each slot at roughly
`2 × encoder-file-size` resident (measured ~1.9× on the INT8 `rnnt` encoder,
CPU provider, release build):

| Head (as loaded) | Encoder file | ≈ RAM per pool slot | Default pool 2 |
|---|---|---|---|
| `rnnt` / `e2e_rnnt` INT8 | ~215 MB | ~0.4 GB | ~790 MB total RSS |
| `rnnt` / `e2e_rnnt` FP32 (`--skip-quantize`) | 844 MB | ~1.6 GB | ~3.3 GB — never in production |
| `ml_ctc` INT8 | ~225 MB | ~0.45 GB | ~0.9 GB |
| `ml_ctc_large` INT8 | ~592 MB | ~1.2 GB | ~2.4 GB |

Two safety nets are built in:

- **RAM auto-cap.** At load, the requested pool is clamped so the pooled
  encoders stay under half of total RAM — a `Capping pool size N -> M` warning
  is logged when it fires. The cap never raises your request, and never goes
  below 1.
- **Degraded boot.** `--pool-min-size 1` (the default) lets the server start
  on a partially loaded pool instead of crashing when memory runs out mid-load.

Rule of thumb: `RAM ≥ pool_size × per-slot + ~1 GB for the OS and request
peaks`. On a 4 GB box that means `--pool-size 1–2` with the INT8 encoder —
the same conclusion as the OOM pitfall in
[Deployment & ops](06-deployment-ops.md).

**Verify:**

```sh
# no "Capping pool size" warning at startup, and:
curl -s http://127.0.0.1:9876/ready
# {"status":"ready","pool_available":2,"pool_total":2}
```

## Verifying the result

End-to-end smoke after any change in this chapter:

```sh
ls ~/.gigastt/models/                  # the head's full file set, incl. *_int8.onnx + vocab
gigastt transcribe sample.wav 2>&1 | grep 'transcribe complete'
# encoder=<int8|fp32>/<cpu|coreml|cuda|ane|candle>, rtf well under 1.0
curl -s http://127.0.0.1:9876/health   # "model"/"variant" match the head you picked
curl -s http://127.0.0.1:9876/ready    # ready, pool_available >= 1
```

Then confirm the accuracy expectation against
[docs/benchmarks.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/benchmarks.md)
rather than re-measuring casually — the harness, manifests, and normalization
matter more than the stopwatch.

## Common pitfalls

- **English audio comes out as garbage with the default head.** Not a bug:
  `rnnt` / `e2e_rnnt` have a Cyrillic-only vocabulary and cannot emit Latin
  text at all. Switch to `--model-variant ml_ctc` (or `ml_ctc_large`).
- **`--model-variant` seems to be ignored after the first download.** The
  engine auto-detects the head from the model directory; an explicit variant
  that differs from the installed one triggers a *second* download alongside
  (with a `variants are never mixed` warning), and a later flag-less start
  prefers `rnnt` again. Delete the unused head's files to keep the directory
  unambiguous and reclaim disk.
- **First `serve` looks hung.** That is the one-time ~850 MB FP32 download +
  ~2 min quantization; `/health` answers `200` with `model:"loading"` while
  `/ready` stays `503 initializing`. Skip the window with `gigastt download
  --prequantized`, and gate clients on `/ready`, never on `/health`.
- **OOM after switching to `ml_ctc_large`.** Each slot now costs ~1.2 GB.
  Lower `--pool-size`, keep `--pool-min-size 1` so a tight host boots
  degraded, and watch for the `Capping pool size` warning at startup.
- **The CoreML build is no faster than CPU.** Look for `falling back to CPU
  execution provider` in the startup log — the warmup probe failed and the
  engine is (deliberately) running on CPU. The completion log's
  `encoder=int8/cpu` field confirms it.
- **The CUDA container runs at CPU speed.** The GPU is not visible: the
  container needs the NVIDIA Container Toolkit and `--gpus all`. The binary
  falls back to CPU silently — check `encoder=int8/cuda` in the completion log.
- **`error: ane and coreml are mutually exclusive` (or similar) at build
  time.** Backend features conflict by design; build exactly one of `coreml` /
  `cuda` / `ane` / `candle` (`nnapi` is the exception).
- **`SHA-256 mismatch` during `gigastt download`.** The staged download was
  corrupt or tampered with; it is removed, not promoted, and the CLI exits `65`.
  Re-run the command — do not hand-rename a `.partial` file into place.

## Links

- [Getting started](01-getting-started.md) — install and first transcription
- [CLI and batch processing](02-cli-batch.md) — throughput/memory recipe for the offline CLI
- [Deployment & ops](06-deployment-ops.md) — offline bundle, systemd, OOM runbook entries
- [docs/benchmarks.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/benchmarks.md) — canonical WER / RTF / footprint tables
- [docs/architecture.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/architecture.md) — pipeline, providers, quantization internals
- [docs/cli.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/cli.md) — full flag reference
- [docs/ane-backend.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/ane-backend.md) — native Apple Neural Engine backend
- [docs/candle-backend.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/candle-backend.md) — Candle/Metal backend
- [docs/verifying-releases.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/verifying-releases.md) — release artifact verification
- [packaging/offline/README-OFFLINE.md](https://github.com/ekhodzitsky/gigastt/blob/main/packaging/offline/README-OFFLINE.md) — offline bundle inventory
