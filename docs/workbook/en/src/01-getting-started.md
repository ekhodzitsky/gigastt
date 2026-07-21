# Getting started

## Scenario

You have never run gigastt and want a working local transcription in about five
minutes: install the binary, download the GigaAM v3 model, transcribe a first
audio file — on macOS, Linux, or in Docker. This chapter is the whole path;
you should not need any other document to get here.

## Prerequisites

- **Disk:** ~1.5 GB free (model + tools). The lean `--prequantized` path needs
  only ~250 MB during setup.
- **RAM:** ~800 MB free at the default `--pool-size 2` (~400 MB per session).
- **Network** (unless you follow the air-gapped recipe): reach either
  `huggingface.co` (full model) or `github.com` (pre-quantized bundle).
- **An audio file to transcribe** — WAV, M4A, MP3, OGG, or FLAC. Any short
  recording of Russian speech works.
- Only for `cargo install` (build from source): Rust 1.88+ and `protoc` on
  `PATH` (`brew install protobuf` / `apt install protobuf-compiler`).

Pick **one** recipe below — macOS, Linux, Docker, or air-gapped — then read
[Choosing the recognition head](#choosing-the-recognition-head) and
[The expensive first run](#the-expensive-first-run) once.

## Recipe: macOS (Homebrew)

Homebrew is the fastest path on Apple Silicon (the tap ships a CoreML-enabled
binary). On an Intel Mac use `cargo install gigastt` instead — see the Linux
recipe for the protoc prerequisite.

```sh
brew tap ekhodzitsky/gigastt https://github.com/ekhodzitsky/gigastt
brew install gigastt

# Fetch the model (~850 MB FP32 from HuggingFace, then a one-time ~2 min
# INT8 quantization pass — see "The expensive first run" below):
gigastt download

# Transcribe your first file:
gigastt transcribe recording.wav
```

**Verify:** the last command prints the recognized text on stdout, e.g.

```text
$ gigastt transcribe recording.wav
Привет, как дела?
```

and `ls ~/.gigastt/models/` shows `v3_rnnt_encoder_int8.onnx`,
`v3_rnnt_decoder.onnx`, `v3_rnnt_joint.onnx`, and `v3_vocab.txt`.

## Recipe: Linux (prebuilt binary or cargo)

**Option A — prebuilt binary** (no Rust toolchain, no protoc). Each release
publishes tarballs for `x86_64-unknown-linux-gnu` and
`aarch64-unknown-linux-gnu`:

```sh
# Resolve the latest release tag (or set TAG=v2.13.0 by hand):
TAG=$(curl -fsSL https://api.github.com/repos/ekhodzitsky/gigastt/releases/latest \
      | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p')
VER=${TAG#v}

curl -fLO "https://github.com/ekhodzitsky/gigastt/releases/download/${TAG}/gigastt-${VER}-x86_64-unknown-linux-gnu.tar.gz"
curl -fLO "https://github.com/ekhodzitsky/gigastt/releases/download/${TAG}/gigastt-${VER}-x86_64-unknown-linux-gnu.tar.gz.sha256"
sha256sum -c "gigastt-${VER}-x86_64-unknown-linux-gnu.tar.gz.sha256"

tar xf "gigastt-${VER}-x86_64-unknown-linux-gnu.tar.gz"
sudo install -m 0755 gigastt /usr/local/bin/gigastt
```

(On ARM64 replace `x86_64-unknown-linux-gnu` with
`aarch64-unknown-linux-gnu`. Homebrew on Linux x86_64 — `brew install
gigastt` after the tap from the macOS recipe — works too.)

**Option B — cargo** (any platform, needs Rust 1.88+ and `protoc`):

```sh
sudo apt install protobuf-compiler   # Debian/Ubuntu; skip if protoc exists
cargo install gigastt
```

Then fetch the model — the lean way, a ~225 MB pre-quantized INT8 bundle from
the pinned GitHub Release (no ~850 MB FP32 download, no ~2-minute on-device
quantization; also handy when HuggingFace is unreachable but GitHub is not):

```sh
gigastt download --prequantized
gigastt transcribe recording.wav
```

**Verify:** `gigastt transcribe recording.wav` prints the recognized text on
stdout, and `ls ~/.gigastt/models/` shows the `v3_rnnt_*` model files.

## Recipe: Docker

Prebuilt multi-arch images (amd64 + arm64) are published to GHCR for every
release; `-cuda` tags carry the CUDA variant:

```sh
docker pull ghcr.io/ekhodzitsky/gigastt:latest   # pin :<version> in production

docker run -d --name gigastt \
  -p 127.0.0.1:9876:9876 \
  -v gigastt-models:/home/gigastt/.gigastt/models \
  ghcr.io/ekhodzitsky/gigastt:latest
```

The named volume keeps the model across container restarts; without it the
container re-downloads ~850 MB on every recreation. On first start the
container downloads the model and quantizes it — the port binds immediately,
but inference is only up when `/ready` turns green:

```sh
# Wait until the model is loaded (503 while initializing):
until curl -sf http://127.0.0.1:9876/ready > /dev/null; do sleep 5; done

curl http://127.0.0.1:9876/health
```

Then transcribe a file from the host (the file path is on the host — `curl`
reads it, not the container):

```sh
curl -F file=@recording.wav http://127.0.0.1:9876/v1/transcribe
```

**Verify:** `/health` returns

```json
{"status":"ok","model":"gigaam-v3-rnnt","variant":"rnnt","version":"2.13.0","punctuation":true,"itn":true}
```

(the `version` field reflects the image you pulled), and the POST returns a
JSON transcript:

```json
{"text":"Привет, как дела?","words":[{"word":"привет","start":0.0,"end":0.4,"confidence":0.99}],"duration":1.2}
```

## Recipe: air-gapped (offline bundle)

For hosts with no internet access, every release publishes a self-contained
offline bundle per Linux target — binary + pre-quantized INT8 `rnnt` model +
punctuation model + systemd unit + installer — plus two Debian packages with
the same content. Download them on a **connected** machine, carry them over,
install.

Tarball flow (any distro):

```sh
# On a connected machine (see the Linux recipe for resolving TAG/VER):
curl -fLO "https://github.com/ekhodzitsky/gigastt/releases/download/${TAG}/gigastt-${VER}-offline-x86_64-unknown-linux-gnu.tar.gz"
curl -fLO "https://github.com/ekhodzitsky/gigastt/releases/download/${TAG}/gigastt-${VER}-offline-x86_64-unknown-linux-gnu.tar.gz.sha256"
sha256sum -c "gigastt-${VER}-offline-x86_64-unknown-linux-gnu.tar.gz.sha256"

# On the target machine:
tar xf "gigastt-${VER}-offline-x86_64-unknown-linux-gnu.tar.gz"
cd "gigastt-${VER}-offline-x86_64-unknown-linux-gnu"
sudo ./install.sh                      # verifies SHA256SUMS, installs binary + model + unit
sudo systemctl enable --now gigastt
```

Debian flow: install `gigastt_<ver>_amd64.deb` (binary + unit) together with
`gigastt-model-int8_<ver>_all.deb` (the same model set), then
`sudo systemctl enable --now gigastt`.

The bundle deliberately omits optional pieces — speaker diarization and the
`e2e_rnnt` / `ml_ctc` heads. The installed unit runs with `GIGASTT_OFFLINE=1`,
so a missing optional model is a fast, instructive error naming the exact path
to fill (fetch it on a connected machine with `gigastt download` and copy the
file over), never a network timeout. The full contents list and signature
verification are in
[packaging/offline/README-OFFLINE.md](https://github.com/ekhodzitsky/gigastt/blob/main/packaging/offline/README-OFFLINE.md).

**Verify:** `curl http://127.0.0.1:9876/health` returns `{"status":"ok",...}`
with `"model":"gigaam-v3-rnnt"`, and
`gigastt transcribe sample.wav --model-dir /usr/share/gigastt/models` prints
text (the flag is only needed when running uninstalled — the systemd unit
already points at the installed model).

## Choosing the recognition head

gigastt ships four recognition heads; `--model-variant` picks one at
`download` / `serve` / `transcribe` time. When you omit the flag, an existing
model directory is used as-is (auto-detect), and a fresh install defaults to
`rnnt`.

| Head | Languages | Output style | Pick it when |
|---|---|---|---|
| `rnnt` (default) | Russian | Bare lowercase from the acoustic model; casing + punctuation restored by an auto-downloaded RuPunct pass, digits by ITN | Default: lowest WER on Russian speech |
| `e2e_rnnt` | Russian | Punctuation / casing / ITN baked into the acoustic model | You want one self-contained model with no post-processing passes |
| `ml_ctc` | ru/en/kk/ky/uz | Bare lowercase, no restoration passes | Mixed Russian/English (or kk/ky/uz) speech; lighter 220M encoder |
| `ml_ctc_large` | ru/en/kk/ky/uz | Bare lowercase, no restoration passes | Multilingual speech where accuracy matters more than footprint (600M encoder) |

The `ml_ctc*` heads download pre-quantized INT8 directly, so there is no
quantization step for them. Switching heads after install:

```sh
gigastt download --model-variant e2e_rnnt   # fetch another head
gigastt serve --model-variant e2e_rnnt      # and serve it explicitly
```

WER/RTF numbers per head are in
[docs/benchmarks.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/benchmarks.md);
the deeper model/backend tour is in [Models and backends](07-models-and-backends.md).

## The expensive first run

The very first `gigastt download` (or first `gigastt serve`, which
auto-downloads a missing model) does two one-time things:

1. **Downloads ~850 MB** of FP32 ONNX files from HuggingFace
   (SHA-256-verified, staged to `.partial`, atomically renamed).
2. **Quantizes the encoder to INT8** (~2 minutes, one-time), producing the
   ~225 MB encoder the engine actually loads. Later runs reuse it.

Three levers change what you pay:

- `gigastt download --prequantized` — the recommended shortcut: fetch the
  ~225 MB pre-quantized INT8 bundle from the pinned GitHub Release. No FP32
  download, no local quantization, no `protoc`. Note it pulls from
  `github.com`, not `huggingface.co` — useful when one of the two is blocked.
- `gigastt download --skip-quantize` (or `GIGASTT_SKIP_QUANTIZE=1` on
  `serve`) — keep the FP32 encoder and skip quantization. The engine then
  loads FP32: slower inference and ~4× the model RAM. Only for debugging.
- Nothing — just let the first `serve` do it. The port binds immediately;
  `/health` answers `200` with `"model":"loading"` and `/ready` returns
  `503 {"reason":"initializing"}` until the model is usable, so clients should
  gate on `/ready`, never on the process being alive.

## Verifying the result

End-to-end checklist that works after any of the recipes above:

```sh
# 1. Model files are in place:
ls ~/.gigastt/models/
#   v3_rnnt_encoder_int8.onnx  v3_rnnt_decoder.onnx  v3_rnnt_joint.onnx  v3_vocab.txt  ...

# 2. Offline transcription works (no server needed):
gigastt transcribe recording.wav
#   → prints the recognized text on stdout

# 3. The server comes up and reports the loaded head:
gigastt serve &                      # Ctrl-C to stop; default http://127.0.0.1:9876
curl http://127.0.0.1:9876/ready     # 200 once the model is loaded
curl http://127.0.0.1:9876/health
#   {"status":"ok","model":"gigaam-v3-rnnt","variant":"rnnt","version":"...","punctuation":true,"itn":true}

# 4. REST transcription works:
curl -F file=@recording.wav http://127.0.0.1:9876/v1/transcribe
#   → {"text":"...","words":[...],"duration":N}
```

## Common pitfalls

- **`protoc` not found** during `cargo install` or a source build — install
  the Protocol Buffers compiler (`brew install protobuf` /
  `apt install protobuf-compiler`), or skip the toolchain entirely with the
  prebuilt binary / Homebrew.
- **First `serve` sits there for minutes** — that is the one-time model
  download + INT8 quantization, not a hang: `/health` returns
  `{"model":"loading"}` meanwhile. Pre-seed with `gigastt download
  --prequantized` and gate clients on `/ready`.
- **`Address already in use` on port 9876** — find the holder with
  `lsof -nP -tiTCP:9876 -sTCP:LISTEN`; confirm it is gigastt
  (`ps -p <pid> -o command=`), then `kill <pid>` (SIGTERM drains cleanly), or
  start on another port with `--port`.
- **Model download fails or hangs** (proxy, firewall, HuggingFace
  unreachable) — retry `gigastt download`; the resume-safe staging file makes
  it idempotent, and exit codes distinguish causes (65 = checksum, 69 =
  network, 74 = disk). If `huggingface.co` is blocked but `github.com` is
  not, use `gigastt download --prequantized`; in a fully closed contour use
  the air-gapped bundle. Check `~/.gigastt/models/` permissions on disk
  errors.
- **OOM or heavy swap on startup** — each pool session loads its own encoder
  copy (~400 MB resident with INT8); the default `--pool-size 2` peaks around
  790 MB. On small machines run with `--pool-size 1`.

The full symptom → cause → fix table lives in
[docs/troubleshooting.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/troubleshooting.md).

## Links

- [docs/cli.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/cli.md) —
  canonical CLI reference (every flag and env var)
- [docs/api.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/api.md) —
  REST / SSE / WebSocket API reference
- [docs/benchmarks.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/benchmarks.md) —
  per-head WER / RTF numbers
- [docs/troubleshooting.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/troubleshooting.md) —
  symptom → cause → fix
- [docs/deployment.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/deployment.md) —
  Docker details, reverse proxy, systemd, offline installs
- [docs/verifying-releases.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/verifying-releases.md) —
  checksums, minisign, SLSA provenance for release artifacts
- [packaging/offline/README-OFFLINE.md](https://github.com/ekhodzitsky/gigastt/blob/main/packaging/offline/README-OFFLINE.md) —
  offline bundle contents and installer options
- [CLI and batch processing](02-cli-batch.md) — the next chapter: batch,
  watch mode, export formats
- [Models and backends](07-models-and-backends.md) — heads, quantization,
  execution providers in depth
