# App-owned server lifecycle

Use this when your application starts gigastt, instead of embedding the engine.
The app owns the process: it picks the binary and the model directory, gives
the CPU encoder a writable cache, binds loopback, waits until the pool is
ready, and on shutdown `wait`s so no child is left running.

This is not a new SDK. Talk to the server with the existing clients in
[Quickstarts](quickstarts.md) (Go module, `@gigastt/client`, or the one-file
programs under [`examples/`](../examples/)). In-process Python, Node, Swift,
and Kotlin bindings in that same guide do not spawn a process at all. Packaging
of the library itself is [embedding-packaging.md](embedding-packaging.md).

The runnable form of this recipe is
[`scripts/managed_server_lifecycle.sh`](../scripts/managed_server_lifecycle.sh).
It does not download weights and it does not delete anything in the model
directory.

## What the process is given

| Piece | Recipe |
|---|---|
| Binary | A release asset from [verifying releases](verifying-releases.md), or a binary you built. `gigastt --version` must accept `serve --optimized-cache-dir`. |
| Model | Lean INT8 only. Default head `rnnt`: `v3_rnnt_encoder_int8.onnx`, `v3_rnnt_decoder.onnx`, `v3_rnnt_joint.onnx`, `v3_vocab.txt`. See [Lean INT8-only install](deployment.md#lean-int8-only-install). |
| Cache | A writable directory that is **not** inside the model dir. `--optimized-cache-dir` / `GIGASTT_OPTIMIZED_CACHE_DIR`. |
| Bind | `--host 127.0.0.1`. Do not pass `--bind-all`. Do not set `GIGASTT_ALLOW_BIND_ANY`. |
| Pool | `--pool-size 1` for a single owner. Raise it only when you want concurrent sessions. |

`gigastt download` fetches the INT8 bundle and leaves every other file in the
directory alone. An install that only has `e2e_rnnt` files stays on that head
when `--model-variant` is omitted. The reference script passes
`--model-variant rnnt` and requires the four `rnnt` files above. To exercise
`e2e_rnnt` instead, point the same flags at that head's INT8 set; do not delete
the other head to switch.

The ORT optimized-graph cache is about 216 MB for the `rnnt` encoder
(`v3_rnnt_encoder_int8_optimized.ort`). If the cache directory cannot be
created or is not writable, the server logs a warning and still starts, without
the cache (slower cold start, higher per-session RAM). That warning is not a
successful setup. Keep the model directory read-only if you want, and put the
cache somewhere the app can write. The systemd unit already does this with
`/var/cache/gigastt`; see [Deployment](deployment.md#lean-int8-only-install)
and [Runbook — ORT cache warnings](runbook.md#ort-optimized-graph-cache-warnings).
`gigastt cache-gc` must be given the same `--optimized-cache-dir`.

The reference script listens on `127.0.0.1:9881` so it does not take the
default port 9876. An application uses any free loopback port it owns. Flags
other than the port match the script:

```sh
gigastt serve \
  --host 127.0.0.1 --port 9881 --pool-size 1 \
  --model-dir "$MODEL_DIR" --model-variant rnnt \
  --optimized-cache-dir "$CACHE_DIR" \
  --punctuation off --itn off
```

Punctuation and ITN are off in the lifecycle check so a missing sidecar cannot
start a download. Leave them at `auto` in production if you want restoration;
keep the `punct/` files, do not delete them while cleaning up FP32 leftovers.

## Readiness and startup failure

The port binds before the model is loaded.

| Probe | While loading | Pool ready |
|---|---|---|
| `GET /health` | `200` `{"status":"ok","model":"loading",...}` | `200` `model` / `variant` of the loaded head |
| `GET /ready` | `503` `{"status":"not_ready","reason":"initializing"}` | `200` `{"status":"ready","pool_available":N,...}` |

`/health` staying up is liveness, not readiness. Gate traffic on `/ready`.

A missing INT8 file must fail with the path, not a hang. With
`GIGASTT_OFFLINE=1` (or `--offline`) and an empty model directory, `serve`
exits non-zero and the log names the file it refused to download, for the
default head:

```text
Error: offline mode (GIGASTT_OFFLINE=1): refusing to download v3_rnnt_encoder_int8.onnx; place the file at <model-dir>/v3_rnnt_encoder_int8.onnx manually (see docs/deployment.md, "Air-gapped / offline installation")
```

Place that file (and the rest of the INT8 set) yourself, or run
`gigastt download` on a machine that is allowed to fetch. Do not point this
failure check at a directory that already holds models. An empty
`.download.lock` may be left next to nothing; delete that lock if you care,
and do not delete neighboring model files to get rid of it.

## What the reference script runs

[`scripts/managed_server_lifecycle.sh`](../scripts/managed_server_lifecycle.sh)
is the application. It starts one child at a time and reaps it with `wait`.

```sh
# optional: GIGASTT_BIN, GIGASTT_MODEL_DIR, GIGASTT_AUDIO
scripts/managed_server_lifecycle.sh
```

Linux only. It reads `/proc/net/tcp` to prove the listener is `127.0.0.1` and
`/proc/<pid>/environ` to prove the offline child inherited `GIGASTT_OFFLINE`.
Needs `curl` and `python3`. It aborts if port 9881 is taken, if the `rnnt`
INT8 files are missing, or if the server log shows a download starting.

Phases, in order:

1. **Retired flags.** `download --fp32`, `--prequantized`, and `--skip-quantize`
   must exit non-zero with `unexpected argument`. Nothing is written.
2. **Startup failure.** Empty temp model dir, `GIGASTT_OFFLINE=1`. Non-zero
   exit, log contains `<empty>/v3_rnnt_encoder_int8.onnx`, no weight file is
   written, PID is reaped. The server may create an empty `.download.lock`
   before it bails; that lock is not a model. The process often exits before
   a probe lands. Readiness (503 then 200) is checked on the boots that stay
   up.
3. **Successful request.** Real model dir, writable temp cache, loopback.
   `/ready` is 200 with `variant=rnnt`, then
   `POST /v1/transcribe` of a short WAV returns 200 and a non-empty `text`.
   The cache dir contains `v3_rnnt_encoder_int8_optimized.ort` and the log
   has no "cache directory is not writable" warning.
4. **Offline restart.** `SIGTERM`, `wait`, port free, start again with
   `GIGASTT_OFFLINE=1`, `/ready` 200, same transcribe request succeeds. No
   download.
5. **Early exit.** New child, `SIGKILL` as soon as it is alive, `wait`.
   The PID is gone afterwards (it may be `Z` before `wait`). Port 9881 is
   free. No owned child remains.
6. **Clean shutdown.** Start, wait for `/ready`, `SIGTERM`, `wait`. Port
   free. `ps` shows no gigastt child of the script.

A successful `POST` is the raw body, not multipart:

```sh
curl -X POST "http://127.0.0.1:9881/v1/transcribe" \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @recording.wav
```

Streaming clients use `/v1/ws` after `/ready`. See the Go and TypeScript
clients linked below; they already honor `retry_after_ms`.

## Retiring FP32 marker and download workarounds

Runtime is INT8-only. There is no FP32 download flag and the engine will not
load an FP32 encoder. `gigastt quantize` is packaging-only: it needs a real
local FP32 ONNX as input and is not part of `serve` or `download`.

Older applications papered over the previous download behavior. Remove those
workarounds from the launcher. Do **not** delete the model directory, the
cache directory, or `~/.gigastt` to do it. Other heads, `punct/`, `vad/`,
speaker models, and any files the user put next to the weights are not FP32
debris.

| Historical workaround | What to do instead |
|---|---|
| `gigastt download --fp32` (Hugging Face FP32 set, then on-device quantize) | Stop passing it. Current binaries exit 2: `unexpected argument '--fp32'`. `gigastt download` fetches the lean INT8 bundle only and skips files already present. |
| `gigastt download --prequantized` | Stop passing it. INT8 is the only download path, so the flag was removed (same clap error). |
| `--skip-quantize` / `GIGASTT_SKIP_QUANTIZE` | Stop passing the flag (rejected; the hint `--skip-diarization` is unrelated). The env var is ignored. Unset it in the app so it does not look live. Unsetting it is not a reason to delete models. |
| A stub `v3_rnnt_encoder.onnx`, `v3_e2e_rnnt_encoder.onnx`, `multilingual_ctc.onnx`, or `multilingual_large_ctc.onnx` created so an older `serve` would treat the directory as already downloaded | Those basenames are still presence markers for auto-detect (`rnnt` wins when several exist), but they are not loadable weights. If the file is a stub (empty or a few bytes) and the matching `*_encoder_int8.onnx` plus the rest of that head's set is present, delete **the stub only**. A real FP32 encoder is hundreds of MB; keep it when you still run `gigastt quantize`. |
| `rm -rf` of the model dir to force a clean INT8 fetch | Do not. Copy or `gigastt download` the missing INT8 names in place. Existing files with the right basename are left as they are. |

A stub `v3_rnnt_encoder.onnx` next to an `e2e_rnnt`-only INT8 set makes
auto-detect choose `rnnt`, then fail with `INT8 encoder not found at
<model-dir>/v3_rnnt_encoder_int8.onnx`. Remove the stub, or pass
`--model-variant` for the head you actually have. Do not wipe the tree.

Stale optimized graphs (`*_optimized.ort`, legacy `*_optimized.onnx`) from an
FP32 run or a head you no longer load are disk waste, not user data.
`gigastt cache-gc --dry-run --optimized-cache-dir <the serve cache>` lists
them; without `--dry-run` it deletes only those graphs (and stale CoreML
cache dirs). It does not delete ONNX weights, vocabs, or unrelated files.
Incomplete `*.partial` downloads can be removed individually; do not remove a
finished ONNX or vocab as part of this cleanup.

Air-gapped installs copy the INT8 set in and run with `GIGASTT_OFFLINE=1`.
See [Air-gapped / offline installation](deployment.md#air-gapped--offline-installation).

## Clients and bindings

| Need | Use |
|---|---|
| Supervise `gigastt serve` | This page and [`scripts/managed_server_lifecycle.sh`](../scripts/managed_server_lifecycle.sh) |
| Typed WebSocket client | [sdks/go](../sdks/go/README.md), [sdks/js](../sdks/js/README.md) (`@gigastt/client`) |
| One-file client, no package | [`examples/`](../examples/) |
| In-process, no child | [Quickstarts](quickstarts.md) (Python, Node, Swift, Kotlin) |
| CLI surface | [cli.md](cli.md) |

## Platforms and the recorded run

The reference script is Linux-only. Tagged release assets also exist for
Linux aarch64, macOS aarch64, and Windows x86_64; those were not executed.
The `serve` flags above are the same on every target. macOS and Windows
supervisors should keep the same shape (own the PID, `wait` after the stop
signal, bind loopback) with their own process API.

**Recorded on this tree.** `gigastt --version` prints `gigastt 2.21.0` because
that is still the package version. The GitHub asset
`gigastt-2.21.0-x86_64-unknown-linux-gnu.tar.gz`
(sha256 `b660fbdbde40054c59a8ac4740cd447daa6123a4e334c09f1a76deecb0e89aa7`)
was checked and is **not** the binary under test: that asset rejects
`--optimized-cache-dir`. The binary that ran this script was
`cargo build --release -p gigastt` from commit
`824bf8ae094c0f28705aa646204b34db216d0494` (Rust sources; this documentation
commit does not change them).

Host: Linux 7.0.0-31-generic x86_64, AMD Ryzen AI 9 HX 370 w/ Radeon 890M.
Model dir: `~/.gigastt/models` (existing INT8 `rnnt` set, not re-downloaded).
Audio: `crates/gigastt/tests/fixtures/golos_00.wav`.
Command:

```sh
GIGASTT_BIN=target/release/gigastt scripts/managed_server_lifecycle.sh
```

Output:

```text
RESULT platform Linux 7.0.0-31-generic x86_64
RESULT cpu AMD Ryzen AI 9 HX 370 w/ Radeon 890M
RESULT binary /home/ekhodzitsky/.grok/worktrees/projects-gigastt/subagent-01a0d3fb-15ca-7fa1-837e-41a331374087/target/release/gigastt
RESULT version gigastt 2.21.0
RESULT binary_sha256 2519091760d9eb2c4a8ccb2291fc10193fccd9887becdca284f4f8dc1c9c80e6
RESULT model_dir /home/ekhodzitsky/.gigastt/models
RESULT audio /home/ekhodzitsky/.grok/worktrees/projects-gigastt/subagent-01a0d3fb-15ca-7fa1-837e-41a331374087/crates/gigastt/tests/fixtures/golos_00.wav
RESULT port 9881 pool_size 1 host 127.0.0.1
RESULT retired_flag flag=--fp32 exit=2
RESULT retired_flag flag=--prequantized exit=2
RESULT retired_flag flag=--skip-quantize exit=2
RESULT startup_failure exit=1 health_200=0 ready_503=0
RESULT startup_failure_line Error: offline mode (GIGASTT_OFFLINE=1): refusing to download v3_rnnt_encoder_int8.onnx; place the file at /tmp/gigastt-lifecycle.Fu5aWs/empty-model/v3_rnnt_encoder_int8.onnx manually (see docs/deployment.md, "Air-gapped / offline installation")
RESULT ready health=200 ready=200 initializing_seen=1 variant=rnnt
RESULT cache dir=/tmp/gigastt-lifecycle.Fu5aWs/cache files=v3_rnnt_encoder_int8_optimized.ort
RESULT success http=200 text_chars=43
RESULT ready health=200 ready=200 initializing_seen=1 variant=rnnt
RESULT offline_restart http=200 text_chars=43
RESULT early_exit wait_status=137 stat_before_wait=none stat_after_wait=gone
RESULT ready health=200 ready=200 initializing_seen=1 variant=rnnt
RESULT clean_shutdown wait_status=0
RESULT children none
RESULT model_dir_unchanged
RESULT ok
```

What that run showed:

- **Startup failure.** Exit 1. The log named `v3_rnnt_encoder_int8.onnx` under the empty temp dir. Curl did not observe `/health` or `/ready` (the process exited first). No weight file was written. The model directory was unchanged at the end of the script.
- **Successful request.** `/ready` was 503 (`initializing_seen=1`) and then 200 with `variant=rnnt`. `POST /v1/transcribe` returned 200 and a 43-character `text`. The writable cache received `v3_rnnt_encoder_int8_optimized.ort`. The script's `/proc/net/tcp` check (loopback only) passed, or the script would have stopped.
- **Offline restart.** After the first child was reaped, a new child with `GIGASTT_OFFLINE=1` reached `/ready` the same way and transcribed again (HTTP 200, 43-character `text`). No download started.
- **Early exit.** `SIGKILL` as soon as the next child was alive. `wait` returned 137 (128+9). The shell also printed a job-control "Killed" notice for that pid (Russian locale: `Убито`). The pid was gone after `wait` (`stat_after_wait=gone`); a zombie was not observed in the sample window (`stat_before_wait=none`).
- **Clean shutdown.** `/ready` 200, then `SIGTERM`, `wait` status 0. Final check: no gigastt child of the script, port free, and every file under the model directory had the same path, size, and mtime as before the script started.
