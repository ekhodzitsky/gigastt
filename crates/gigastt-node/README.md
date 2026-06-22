# gigastt-node

Node.js binding for [gigastt](https://github.com/ekhodzitsky/gigastt) — on-device Russian speech-to-text (**GigaAM v3**) — built with [napi-rs](https://napi.rs).

Wraps the synchronous `gigastt-core` engine: models are **side-loaded** (no HTTP download) and inference runs on a libuv worker thread via napi's `AsyncTask`, so calls return **Promises** and never block the event loop. Errors are thrown JS `Error`s and objects are garbage-collected (no manual free). onnxruntime is statically linked, so the `.node` addon is self-contained.

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
