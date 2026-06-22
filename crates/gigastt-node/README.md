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

`npm run build` auto-generates the JS loader (`index.js`), the TypeScript types (`index.d.ts`), and the native `.node` addon; all are git-ignored build artifacts.

## Performance note

Inference runs on libuv's worker threadpool (default 4 threads, shared with Node's `fs`/`crypto`). For `N` concurrent transcriptions set `UV_THREADPOOL_SIZE >= N` and construct the `Engine` with a matching `poolSize`.

## Packaging

Per-platform prebuilt npm packages follow napi-rs's optional-dependency model: a JS-only root package (`gigastt`) plus one binary sub-package per platform (`gigastt-darwin-arm64`, `gigastt-linux-x64-gnu`, …) listed under `optionalDependencies`, so `npm install gigastt` pulls exactly the matching binary — no compiler, no node-gyp, no postinstall download. onnxruntime is statically linked into each `.node`, so there is no native dylib to locate (and no `LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH` setup).

The prebuilts are produced and published by the **Node Prebuilds** workflow (`.github/workflows/node-prebuilds.yml`, manual `workflow_dispatch`): it builds the addon on a native runner per target — `darwin-arm64`, `linux-x64-gnu`, `linux-arm64-gnu`, `win32-x64-msvc` — and, when `publish` is set and `NPM_TOKEN` is configured, publishes all packages via `napi prepublish`. (Intel macOS is omitted: ort ships no prebuilt onnxruntime for it.)

The ~215 MB INT8 model is **not** bundled in the npm package; side-load it at runtime (e.g. `gigastt download --prequantized`).

## License

MIT.
