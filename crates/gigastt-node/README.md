# gigastt-node

Node.js binding for [gigastt](https://github.com/ekhodzitsky/gigastt) — on-device Russian speech-to-text (**GigaAM v3**) — built with [napi-rs](https://napi.rs).

Wraps the synchronous `gigastt-core` engine: models are **side-loaded** (no HTTP download) and inference runs on a libuv worker thread via napi's `AsyncTask`, so calls return **Promises** and never block the event loop. Errors are thrown JS `Error`s and objects are garbage-collected (no manual free). onnxruntime is statically linked, so the `.node` addon is self-contained.

## In-process (this package) vs sidecar server

This package embeds the engine **inside your Node/Electron process** — desktop JS
with no sidecar. The alternative is shipping `gigastt serve` next to your app and
talking to it over WebSocket/REST. Pick consciously:

| | In-process (`npm install gigastt`) | Sidecar (`gigastt serve` + WS/REST client) |
|---|---|---|
| Deployment | Single `npm install` — one prebuilt binary fetched by postinstall | Binary discovery on the user's machine, spawn, supervision, port selection |
| Interface | Plain JS calls; errors are thrown `Error`s | Wire protocol over a loopback port (`/v1/ws`, REST, SSE) |
| Versioning | App and engine are one artifact, shipped together | App must gate on the discovered server's version |
| Memory | Model + pool live in your app's process (budget ~400 MB RSS per pool session) | Model lives in the separate server process |
| Concurrency | One engine/pool instance per app process | One server shared by several apps/clients |
| Failure isolation | An engine crash takes the app down (and vice versa) | Server crashes are isolated; the app survives and can restart it |
| Upgrades | Redeploy the app | Upgrade the server independently of any client |

Rule of thumb: a desktop app that owns its whole audio pipeline (an Electron
recorder, a dictation tool) is simplest in-process; a setup shared by several
clients, or one that must survive engine crashes, belongs behind the sidecar.
For Electron, construct the `Engine` in the **main process** and keep one
`Stream` per audio channel — see [`examples/electron_main.mjs`](../../examples/electron_main.mjs).

## API

| Type | Members |
|---|---|
| `Engine` | `new Engine(modelDir, poolSize?)` · `transcribeFile(path): Promise<Transcript>` |
| `Stream` | `new Stream(engine)` · `processChunk(pcm16, sampleRate): Promise<TranscriptSegment[]>` · `flush(): Promise<TranscriptSegment[]>` |
| objects | `Transcript { text, words, durationS }` · `TranscriptSegment { text, words, isFinal }` · `Word { text, startS, endS, confidence, speaker? }` |

`processChunk` takes little-endian mono PCM16 (`Uint8Array`/`Buffer`) and resamples to 16 kHz internally. Await each `processChunk` before sending the next chunk to preserve ordering.

Errors are thrown JS `Error`s whose message is prefixed with a stable code — `ModelNotFound`, `InvalidAudio`, `PoolExhausted`, `Inference` — matching the C-ABI / UniFFI contract across bindings.

## Quickstart

```js
const { Engine } = require('gigastt');

const engine = new Engine('/path/to/gigastt/models'); // side-loaded model dir
const t = await engine.transcribeFile('recording.wav');
console.log(t.text);
for (const w of t.words) console.log(w.text, w.startS, w.endS, w.confidence);

// streaming
const { Stream } = require('gigastt');
const s = new Stream(engine);
for (const seg of await s.processChunk(pcm16, 16000)) console.log(seg.text);
console.log((await s.flush()).map((seg) => seg.text));
```

## Building locally

```sh
cd crates/gigastt-node
npm install                 # installs @napi-rs/cli
npm run build               # napi build --release -> index.js + index.d.ts + *.node
GIGASTT_MODEL_DIR=~/.gigastt/models npm run smoke   # transcribes the golos_00.wav fixture
```

`npm run build` produces the native `gigastt.<platform>.node` addon (git-ignored); `loader.js`, `install.js`, and `gigastt.d.ts` are the committed, published package files.

## Performance note

Inference runs on libuv's worker threadpool (default 4 threads, shared with Node's `fs`/`crypto`). For `N` concurrent transcriptions set `UV_THREADPOOL_SIZE >= N` and construct the `Engine` with a matching `poolSize`.

## Packaging

**Single npm package.** `gigastt` is one JS-only package (`loader.js` + `install.js` + `gigastt.d.ts`) — no per-platform sub-packages. On `npm install gigastt`, the postinstall (`install.js`) downloads exactly **one** native binary — the `gigastt.<platform>.node` matching the install platform — from the `node-v<version>` GitHub release, so only ~47 MB is fetched (not all platforms). `loader.js` then loads it. onnxruntime is statically linked into each `.node`, so there is no native dylib to locate (no `LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH`).

Supported prebuilt platforms: `darwin-arm64`, `linux-x64-gnu`, `linux-arm64-gnu`, `win32-x64-msvc` (Intel macOS omitted — ort ships no onnxruntime for it). The binaries are built per native runner and attached to the `node-v<version>` release by the **Node Prebuilds** workflow (`.github/workflows/node-prebuilds.yml`, `workflow_dispatch`); the single `gigastt` package is then published.

Caveat: `npm install --ignore-scripts` skips the postinstall, so the binary is not fetched — `require('gigastt')` then errors with instructions to run `node install.js`. The ~215 MB INT8 model is **not** bundled either; side-load it at runtime (e.g. `gigastt download --prequantized`).

## License

MIT.
