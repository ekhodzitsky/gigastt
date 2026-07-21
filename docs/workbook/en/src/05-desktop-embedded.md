# Desktop & embedded: Swift/SPM, sidecar, Electron, UniFFI

## Scenario

You are building a desktop or mobile app with on-device Russian speech-to-text:
a macOS dictation app in Swift, an Electron meeting recorder, or a Kotlin /
Python tool. The model must run locally (no cloud), and you have two ways to
ship the engine — embed it **in-process** via a native binding, or run
`gigastt serve` as a **sidecar** subprocess and talk to it over the loopback
HTTP/WS API. This chapter helps you choose, then walks each path to the first
transcript.

## Choosing a path: embedded vs sidecar

| | Embedded (in-process) | Sidecar (`gigastt serve` + client) |
|---|---|---|
| How | Engine linked into your app: SwiftPM `GigaSTT`, npm `gigastt`, PyPI `gigastt`, UniFFI bindings | Engine runs as a child process; your app talks WS/REST over a loopback port |
| Interface | Native calls; typed errors (`throws` / exceptions / rejected promises) | Wire protocol (`/v1/ws`, REST `/v1/transcribe`, SSE) |
| Deployment | One package install; no process supervision | Binary discovery on the user's machine, spawn, supervision, port selection |
| Memory | Model + pool live inside your app (~350–400 MB RSS per pool session) | Model lives in the separate server process |
| Concurrency | One engine/pool instance per app process | One server shared by several apps/clients |
| Failure isolation | An engine crash takes the app down (and vice versa) | Server crashes are isolated; the app survives and can restart it |
| Upgrades | Redeploy the app with the new engine | Upgrade the server binary independently of any client |
| Versioning | App and engine are one artifact | App must gate on the discovered server's version (`/health` → `version`) |

Rules of thumb:

- **Embedded** when your app owns its whole audio pipeline (a dictation tool,
  a recorder) and is the only consumer. On iOS it is the *only* option — apps
  cannot spawn subprocesses.
- **Sidecar** when one engine is shared by several clients, when the app must
  survive an engine crash, or when you want to upgrade the engine without
  redeploying the app. Both confirmed external desktop integrations use this
  pattern.

## Prerequisites

- **A model directory** — every path below assumes one. Fetch the
  pre-quantized INT8 bundle once (~215 MB, no FP32 download, no on-device
  quantization):

  ```sh
  gigastt download --prequantized          # -> ~/.gigastt/models
  ```

- **Embedded**: the toolchain for your binding — Xcode 15+ (Swift), Node.js
  (npm package), Python 3 (wheel), or an Android SDK/NDK (Kotlin).
- **Sidecar**: the `gigastt` binary — bundled inside your app's resources, or
  installed (`brew tap ekhodzitsky/gigastt https://github.com/ekhodzitsky/gigastt && brew install gigastt`,
  release tarball, or `cargo install gigastt`) — plus `curl` for probing.

## Recipe — Swift/SPM (iOS + macOS)

The `GigaSTT` Swift package wraps the C ABI in a safe Swift interface. The
native code ships as a prebuilt `GigasttFFI.xcframework` (iOS device `arm64`,
simulator `arm64`/`x86_64`, macOS `arm64`) with ONNX Runtime statically
linked — there is no separate runtime to bundle. Requires iOS 15 / macOS 13
(Apple Silicon only — there is no Intel macOS slice) and Xcode 15+.

1. **Add the package.** Xcode → File → Add Package Dependencies… → enter the
   mirror repository URL `https://github.com/ekhodzitsky/gigastt-swift` and add
   the `GigaSTT` product to your target. The mirror is the canonical remote
   source (SwiftPM needs `Package.swift` at the repository root, so the
   monorepo subdirectory `packaging/swift` cannot be consumed via URL — use it
   only as a local path dependency for development).
2. **Bundle the model.** Copy `~/.gigastt/models` into your app target as a
   **folder reference** (the blue folder — it preserves the directory layout;
   a yellow "group" flattens the files and the engine will not find them).
   Alternatively download the model on first launch and cache it.
3. **Load the engine and transcribe:**

   ```swift
   import GigaSTT

   guard let modelDir = Bundle.main.url(
       forResource: "models", withExtension: nil
   )?.path else {
       fatalError("bundle the model directory as a folder reference")
   }

   // poolSize: 1 keeps RAM around ~350 MB, recommended on device.
   let engine = try Engine(modelDir: modelDir, poolSize: 1)

   // Path is relative to the current working directory; absolute paths and
   // ".." are rejected by the engine.
   let text = try engine.transcribeFile(path: "audio.wav")
   print(text)
   ```

4. **Stream** little-endian mono PCM16 chunks at your capture rate (resampled
   to 16 kHz internally):

   ```swift
   let stream = try Stream(engine: engine)

   // pcm16: Data of little-endian Int16 mono samples at 48 kHz.
   for segment in try stream.processChunk(pcm16, sampleRate: 48000) {
       print(segment.text, segment.isFinal)
   }

   // Drain the tail at end-of-stream.
   for segment in try stream.flush() {
       print(segment.text)
   }
   ```

   `processChunk` and `flush` return `[TranscriptSegment]`; each segment
   carries `text`, `words` (per-word `word`/`start`/`end`/`confidence` and an
   optional `speaker`), `isFinal`, and a `timestamp`.

5. **Handle errors.** The wrapper throws `GigasttError`:
   `engineLoadFailed(modelDir:)` (missing/unreadable model directory),
   `streamCreationFailed` (pool checkout failed), `inferenceFailed`, and
   `decodingFailed(underlying:)`. The C ABI signals failure with `NULL`
   returns, so a case tells you *where* it failed rather than carrying an
   engine-side message.

Verify: run the app, transcribe a known WAV, and confirm the expected text is
printed. If the engine throws `engineLoadFailed` at boot, the model directory
is not where `Bundle.main.url(forResource: "models", withExtension: nil)`
points — see Common pitfalls.

## Recipe — sidecar server (macOS / Electron)

Run `gigastt serve` as a managed child process. The full lifecycle: discover
the binary, pre-stage the model, spawn, gate on readiness, transcribe, stop
gracefully.

1. **Discover the binary**, in order: an env override (e.g. `MYAPP_GIGASTT_BIN`)
   → a copy bundled in your app's resources → `/opt/homebrew/bin/gigastt` →
   `/usr/local/bin/gigastt` → `PATH`. Record which one you picked for logging.
2. **Pre-stage the model** at install time or first launch, with
   machine-readable progress for your UI:

   ```sh
   gigastt download --prequantized --progress json
   ```

   stdout carries one NDJSON event per line
   (`{"phase":"download","file":...,"bytes_done":N,"bytes_total":M}`, then
   `verify`, then `done`) and exit codes distinguish network/disk/checksum
   failures — see [docs/cli.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/cli.md).
   `--prequantized` skips the ~2-minute on-device INT8 pass, so the first
   `serve` starts in seconds, not minutes.
3. **Pick a port.** Today: a fixed high port (e.g. `49876`). Ephemeral
   auto-selection (`--port 0` with a machine-readable `LISTENING` line),
   `--die-with-parent`, and `--log-file` are planned server additions — they
   require a gigastt version that ships them, so do not rely on their syntax
   yet; use a fixed port and the lifecycle below.
4. **Spawn and capture logs.** Never send the sidecar's output to `/dev/null` —
   pipe stdout/stderr to a log file (if you connect pipes instead, keep
   draining them: a full pipe buffer can deadlock the child). Example in an
   Electron main process / Node:

   ```js
   import { spawn } from 'node:child_process';
   import fs from 'node:fs';

   const PORT = 49876;
   const BASE = `http://127.0.0.1:${PORT}`;

   const log = fs.createWriteStream('gigastt-sidecar.log', { flags: 'a' });
   const server = spawn(gigasttBin, [
     'serve',
     '--port', String(PORT),
     '--pool-size', '1',
     '--model-dir', modelDir,
   ], { stdio: ['ignore', log, log] });

   async function waitForReady(timeoutMs = 120_000) {
     const deadline = Date.now() + timeoutMs;
     for (;;) {
       try {
         const res = await fetch(`${BASE}/ready`);
         // 200 {"status":"ready","pool_available":N,"pool_total":M}
         if (res.ok) return;
         // 503 {"status":"not_ready","reason":"initializing"} — keep waiting.
       } catch {
         // Connection refused: the listener is not up (yet) — or the process
         // is gone. Distinguish by the child exit code, not by killing it.
         if (server.exitCode !== null) {
           throw new Error(`gigastt exited with code ${server.exitCode}`);
         }
       }
       if (Date.now() > deadline) throw new Error('readiness timeout');
       await new Promise((resolve) => setTimeout(resolve, 500));
     }
   }
   ```

5. **Gate on `/ready`, not on TCP connect.** The server binds the port
   immediately and answers probes from a bootstrap responder while the model
   loads, so "port listening" does not mean "ready". Poll `GET /ready` until
   it returns 200; a 503 body carries
   `{"status":"not_ready","reason":"initializing"|"pool_exhausted"|"shutting_down"}`.
   Connection-refused means the process is dead or not yet spawned — not that
   it is stuck.
6. **Version handshake over HTTP.** `GET /health` returns 200 in *both*
   phases — `{"status":"ok","model":"loading","version":"2.13.0"}` during
   bootstrap, then `{"status":"ok","model":"gigaam-v3-rnnt","variant":"rnnt","version":"2.13.0","punctuation":true,"itn":true}`.
   Gate your minimum engine version on the `version` field instead of running
   `gigastt --version` in a subprocess.
7. **Transcribe.** For live partial results open a WebSocket session on
   `/v1/ws` (patterns in [Streaming over WebSocket](04-streaming-ws.md)); for whole files POST to
   `/v1/transcribe` (recipes in [CLI and batch processing](02-cli-batch.md)).
   On pool saturation the server answers 503 + `Retry-After` (REST) or an
   error with `retry_after_ms` (WS) — honor it instead of inventing a backoff.
   Before closing a WS session, send `{"type":"stop"}` so the server flushes
   trailing words into a final segment.
8. **Stop gracefully.** Send SIGTERM: the server drains in-flight sessions
   for up to `--shutdown-drain-secs` (default 10), flushing a `final` and
   closing WS clients with code 1001. Escalate to SIGKILL only after the drain
   window. Track the child pid and terminate it when your app exits — an
   orphaned sidecar holds ~1 GB RSS and the port.

Verify:

```sh
gigastt serve --port 49876 --pool-size 1 &
curl -s http://127.0.0.1:49876/ready    # -> {"status":"ready","pool_available":1,"pool_total":1}
curl -s http://127.0.0.1:49876/health   # -> {"status":"ok","model":"gigaam-v3-rnnt",...,"version":"..."}
kill %1                                  # SIGTERM — process exits within the drain window
```

## Recipe — Node/Electron in-process

The `gigastt` npm package (napi-rs) embeds the engine inside your Node/Electron
process — no sidecar, no port, no version gate. Inference runs on a libuv
worker thread, so calls return Promises and never block the event loop;
onnxruntime is statically linked, so the `.node` addon is self-contained.

1. **Install:** `npm install gigastt`. The postinstall downloads exactly one
   prebuilt binary (`gigastt.<platform>.node`, ~47 MB) for the install platform
   from the GitHub release. Prebuilt platforms: `darwin-arm64`,
   `linux-x64-gnu`, `linux-arm64-gnu`, `win32-x64-msvc` (no Intel macOS).
2. **Side-load the model** — it is not bundled: `gigastt download --prequantized`
   → `~/.gigastt/models`, or set `GIGASTT_MODEL_DIR`.
3. **Use it:**

   ```js
   const { Engine, Stream } = require('gigastt');

   const engine = new Engine('/path/to/gigastt/models'); // new Engine(modelDir, poolSize?)
   const t = await engine.transcribeFile('recording.wav');
   console.log(t.text, t.durationS);
   for (const w of t.words) console.log(w.text, w.startS, w.endS, w.confidence);

   // streaming — await each chunk before sending the next to keep order
   const s = new Stream(engine);
   for (const seg of await s.processChunk(pcm16, 16000)) console.log(seg.text);
   console.log((await s.flush()).map((seg) => seg.text));

   // errors are thrown JS Errors whose message starts with a stable code
   try { new Engine('/no/such/dir'); } catch (e) { /* e.message starts "ModelNotFound:" */ }
   ```

4. **Electron layout:** construct the `Engine` in the **main process** (never
   in a renderer) and keep one `Stream` per audio channel (e.g. mic + system).
   Each `Stream` holds one pool session for its lifetime, so `poolSize` must
   cover the number of live streams — a third `Stream` on a pool of 2 throws
   `PoolExhausted`. The full dual-channel pattern with IPC handlers is in
   [examples/electron_main.mjs](https://github.com/ekhodzitsky/gigastt/blob/main/examples/electron_main.mjs).
5. **Size the thread pool.** Inference runs on libuv's worker pool (default 4
   threads, shared with `fs`/`crypto`). For `N` channels transcribing
   simultaneously, start with `UV_THREADPOOL_SIZE >= N`
   (e.g. `UV_THREADPOOL_SIZE=4 electron .`) and a matching `poolSize`.
6. **Packaging (asar):** keep the model directory *and* the native `.node`
   addon **out of the asar archive** — native code memory-maps the weights and
   `dlopen`s the addon, neither of which works from asar's virtual filesystem.
   With electron-builder, use `asarUnpack` for `**/*.node` and ship the model
   via `extraResources`.

Verify: in a checkout, `GIGASTT_MODEL_DIR=~/.gigastt/models npm run smoke`
(from `crates/gigastt-node`) transcribes a test fixture; in your app, await
`engine.transcribeFile` on a known WAV and check the text.

## Recipe — Kotlin and Python (UniFFI)

`crates/gigastt-uniffi` generates idiomatic Swift, Kotlin, and Python bindings
from one Rust source with UniFFI. They wrap the synchronous lean core (models
side-loaded, no tokio runtime), surface typed `GigasttFfiError` errors
(`ModelNotFound`, `InvalidAudio`, `PoolExhausted`, `Inference`,
`InvalidArgument`), and manage objects by reference counting.

Shared surface (casing follows each language's idiom):

- `Engine(model_dir)` / `Engine.new_with_pool_size(model_dir, pool_size)` ·
  `transcribe_file(path) -> Transcript { text, words, duration_s }`
- `Stream(engine)` · `process_chunk(pcm16, sample_rate) -> [TranscriptSegment]` ·
  `flush() -> [TranscriptSegment]`
- `TranscriptSegment { text, words, is_final }` ·
  `Word { text, start_s, end_s, confidence, speaker }`

**Python — published.** `pip install gigastt` installs a self-contained
prebuilt wheel (`py3-none-<platform>`; onnxruntime statically linked):

```python
import gigastt_uniffi as g

engine = g.Engine("/path/to/gigastt/models")        # side-loaded model dir
t = engine.transcribe_file("recording.wav")
print(t.text)
for w in t.words:
    print(w.text, w.start_s, w.end_s, w.confidence)

# streaming
s = g.Stream(engine)
for seg in s.process_chunk(pcm16_bytes, 16000):
    print(seg.text)
print([seg.text for seg in s.flush()])

# typed errors
try:
    g.Engine("/no/such/dir")
except g.GigasttFfiError.ModelNotFound as e:
    ...
```

**Kotlin — experimental AAR.** The Rust cross-build is proven (CI
cross-compiles `libgigastt_uniffi.so` per ABI via cargo-ndk), but the
Gradle/Maven AAR assembly and publish have not been validated end-to-end yet.
To build locally:

```sh
# native libs per ABI
cargo ndk -t arm64-v8a -t armeabi-v7a -t x86_64 \
  -o packaging/android/gigastt/src/main/jniLibs build --release -p gigastt-uniffi
# Kotlin bindings (from a host build of the cdylib; metadata is arch-independent)
cargo build --release -p gigastt-uniffi
cargo run --release -p gigastt-uniffi --bin uniffi-bindgen -- generate \
  --library target/release/libgigastt_uniffi.* --language kotlin \
  --out-dir packaging/android/gigastt/src/main/kotlin
# assemble
cd packaging/android && gradle :gigastt:assembleRelease
```

**Swift (UniFFI):** the bindings generate (`--language swift`), but the shipped
SwiftPM package uses the handwritten C-ABI wrapper from the Swift recipe above
— prefer it for app development.

To regenerate any binding from a checkout:

```sh
cargo build -p gigastt-uniffi
LIB=target/debug/libgigastt_uniffi.dylib   # .so on Linux

cargo run -p gigastt-uniffi --bin uniffi-bindgen -- generate --library "$LIB" --language python --out-dir bindings/python
cargo run -p gigastt-uniffi --bin uniffi-bindgen -- generate --library "$LIB" --language swift  --out-dir bindings/swift
cargo run -p gigastt-uniffi --bin uniffi-bindgen -- generate --library "$LIB" --language kotlin --out-dir bindings/kotlin
```

Generated bindings are build artifacts (`bindings/` is git-ignored). Status and
packaging details:
[crates/gigastt-uniffi/README.md](https://github.com/ekhodzitsky/gigastt/blob/main/crates/gigastt-uniffi/README.md),
[packaging/android/README.md](https://github.com/ekhodzitsky/gigastt/blob/main/packaging/android/README.md).

Verify: the Python snippet above prints the transcript of a known file; for
Kotlin, assemble the AAR and transcribe from an instrumented test or a debug
screen.

## Verifying the result

End-to-end checklist, whichever path you took:

- **First transcript**: your app prints the expected text for a known WAV
  (any Russian speech file, e.g. a Golos sample).
- **Model on disk**: `ls ~/.gigastt/models` shows `v3_rnnt_*` files (the
  `--prequantized` bundle includes `v3_rnnt_encoder_int8.onnx`).
- **Memory**: RSS stays in the expected band — roughly 350–400 MB per pool
  session; a `poolSize`/`pool-size` of 1 is the right default on-device.
- **Sidecar health**: `curl -s http://127.0.0.1:<port>/ready` returns
  `{"status":"ready",...}` and `/health` reports the `version` you gate on;
  SIGTERM shuts the process down within the drain window with a clean log
  tail.
- **Streaming order**: partial segments (`isFinal == false`) may be revised;
  persist only finals, and call `flush()` (or send `{"type":"stop"}` on WS)
  before tearing down a session so the tail is not lost.

## Common pitfalls

- **Model not bundled as a folder reference (Swift).** Added as a yellow
  "group", Xcode flattens the directory and the engine fails at boot with
  `engineLoadFailed`. Fix: add `models` as a folder reference (blue folder)
  and assert `Bundle.main.url(forResource: "models", withExtension: nil)` is
  non-nil in a debug run.
- **Undefined `std::__1::*` symbols at link time (Swift).** ONNX Runtime is
  C++; the current package already declares `.linkedLibrary("c++")` — this
  error means you are consuming an old or hand-rolled xcframework. Use the
  released package from the mirror (or current `packaging/swift`).
- **Blocking call on the UI thread.** `Engine`/`Stream` calls are synchronous
  and the handles are not thread-safe. Run all engine work on a dedicated
  background queue/actor (Swift) and serialize access; never transcribe on the
  main thread. In Node the calls already run on the libuv pool — just `await`
  them.
- **Readiness timeouts on the first run (sidecar).** A plain
  `gigastt download` leaves a ~2-minute on-device INT8 quantization pass to the
  first `serve`, and the punctuation model fetches lazily on first start — a
  client with a 10–30 s boot timeout gives up too early. Fix: pre-stage with
  `gigastt download --prequantized`, keep the readiness timeout generous, and
  branch on the `/ready` `reason` instead of killing the process.
- **Killing the server on 503.** During model load the port is bound by the
  bootstrap responder and every API answers 503 `initializing` — that is
  progress, not a hang. Only connection-refused plus a dead child process
  means failure.
- **Hardcoded port collision.** A fixed port already in use (often by an
  orphaned gigastt from a previous run) surfaces as a silent timeout. Check
  `/health` → `version` to learn whether the listener is *your* server, and
  terminate your child on app exit (a `--die-with-parent` flag is planned).
- **Sidecar logs to `/dev/null`.** You will need them the first time the model
  directory is corrupt. Pipe stdout/stderr to a file — and if you use pipes,
  keep draining them to avoid a pipe-buffer deadlock.
- **`require('gigastt')` fails after `npm install --ignore-scripts`.** The
  postinstall that downloads the native binary was skipped; run `node
  install.js` inside the package.
- **Model or addon inside asar (Electron).** Native code memory-maps the
  weights and `dlopen`s the `.node` — both need real files. `asarUnpack` the
  addon and ship the model via `extraResources`.
- **Absolute path to `transcribeFile` (Swift).** The C ABI accepts only
  relative paths inside the working directory — point the working directory at
  the file's location first (or copy the file there).

## Links

In this book:

- [Getting started](01-getting-started.md) — install and model download
- [Streaming over WebSocket](04-streaming-ws.md) — WebSocket protocol patterns for the sidecar
- [CLI and batch processing](02-cli-batch.md) — REST / SSE / jobs recipes

Reference material (do not duplicate — read here):

- [packaging/swift/README.md](https://github.com/ekhodzitsky/gigastt/blob/main/packaging/swift/README.md) — Swift package API reference
- [crates/gigastt-node/README.md](https://github.com/ekhodzitsky/gigastt/blob/main/crates/gigastt-node/README.md) — npm package API + in-process vs sidecar trade-offs
- [crates/gigastt-uniffi/README.md](https://github.com/ekhodzitsky/gigastt/blob/main/crates/gigastt-uniffi/README.md) — UniFFI bindings and generation
- [packaging/android/README.md](https://github.com/ekhodzitsky/gigastt/blob/main/packaging/android/README.md) — Kotlin/AAR build status
- [examples/electron_main.mjs](https://github.com/ekhodzitsky/gigastt/blob/main/examples/electron_main.mjs) — full Electron main-process pattern
- [docs/quickstarts.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/quickstarts.md) — per-binding quickstarts and availability table
- [docs/embedding-packaging.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/embedding-packaging.md) — onnxruntime static vs dynamic linking
- [docs/api.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/api.md) — `/health`, `/ready`, REST/WS/SSE reference
- [docs/cli.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/cli.md) — `serve` and `download` flags (lifecycle knobs used above)

A dedicated embedding & distribution guide (`docs/embedding.md`, covering
topics like macOS notarization) is in progress.
