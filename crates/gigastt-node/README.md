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
const { Engine } = require('gigastt-node');

const engine = new Engine('/path/to/gigastt/models'); // side-loaded model dir
const t = await engine.transcribeFile('recording.wav');
console.log(t.text);
for (const w of t.words) console.log(w.text, w.startS, w.endS, w.confidence);

// streaming
const { Stream } = require('gigastt-node');
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

Per-platform prebuilt npm packages (optional-dependency model, `npm install` with no compiler) are produced by the release pipeline (prebuilt-artifacts task), not by this crate's local build.

## License

MIT.
