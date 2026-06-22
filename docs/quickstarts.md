# In-process quickstarts

Embed gigastt directly in your app — on-device, no server, no cloud. Each
binding wraps the same engine: load a model directory once, then transcribe a
file or stream PCM16 from a microphone. Errors are native (exceptions / `throws`
/ rejected promises); the model is **side-loaded** (not bundled in the package).

## Get a model (once)

The bindings do not download models. Fetch the pre-quantized INT8 bundle and
point the binding at the directory:

```sh
gigastt download --prequantized            # -> ~/.gigastt/models (no protoc, no on-device quantize)
# or: gigastt download --prequantized --model-dir ./models
```

On a device, ship that directory with your app (or download it on first run) and
pass its path to the engine constructor.

## Python — `pip install gigastt`

```python
import gigastt_uniffi as g

engine = g.Engine("/path/to/models")
t = engine.transcribe_file("recording.wav")
print(t.text)
for w in t.words:
    print(w.text, w.start_s, w.end_s, w.confidence)

# streaming (little-endian mono PCM16)
s = g.Stream(engine)
for seg in s.process_chunk(pcm16_bytes, 16000):
    print(seg.text)
print([seg.text for seg in s.flush()])

# typed errors
try:
    g.Engine("/no/such/dir")
except g.GigasttError.ModelNotFound:
    ...
```

## Node — `npm install gigastt`

```js
const { Engine, Stream } = require('gigastt');

const engine = new Engine('/path/to/models');     // new Engine(modelDir, poolSize?)
const t = await engine.transcribeFile('recording.wav');
console.log(t.text, t.durationS);
for (const w of t.words) console.log(w.text, w.startS, w.endS, w.confidence);

// streaming — await each chunk before sending the next
const s = new Stream(engine);
for (const seg of await s.processChunk(pcm16, 16000)) console.log(seg.text);
console.log((await s.flush()).map((seg) => seg.text));

// errors are thrown JS Errors (message prefixed with a stable code)
try { new Engine('/no/such/dir'); } catch (e) { /* e.message starts "ModelNotFound:" */ }
```

## Swift — SwiftPM

```swift
import GigaSTT

let engine = try Engine(modelDir: "/path/to/models")
let t = try engine.transcribeFile(path: "recording.wav")
print(t.text)
for w in t.words { print(w.text, w.startS, w.endS, w.confidence) }

// streaming
let s = try Stream(engine: engine)
for seg in try s.processChunk(pcm16: data, sampleRate: 16000) { print(seg.text) }

// errors: do/catch on GigasttError
```

## Kotlin — Android (AAR)

```kotlin
val engine = Engine("/path/to/models")
val t = engine.transcribeFile("recording.wav")
println(t.text)
t.words.forEach { println("${it.text} ${it.startS} ${it.endS} ${it.confidence}") }

// streaming
val s = Stream(engine)
s.processChunk(pcm16, 16000u).forEach { println(it.text) }

// errors throw GigasttException subtypes
```

## Surface (all bindings)

- `Engine(modelDir[, poolSize])` · `transcribeFile(path) -> Transcript`
- `Stream(engine)` · `processChunk(pcm16, sampleRate) -> [TranscriptSegment]` · `flush() -> [TranscriptSegment]`
- `Transcript { text, words, durationS }` · `TranscriptSegment { text, words, isFinal }` · `Word { text, startS, endS, confidence, speaker? }`
- errors: `ModelNotFound`, `InvalidAudio`, `PoolExhausted`, `Inference`, `InvalidArgument`

(Method/field casing follows each language's idiom — `transcribe_file`/`start_s`
in Python, `transcribeFile`/`startS` in Node/Swift/Kotlin.)

## Availability

| Binding | Package | Status |
|---|---|---|
| Python | `gigastt` (PyPI) | published — `pip install gigastt` |
| Node | `gigastt` (npm) | published — `npm install gigastt` |
| Swift | SwiftPM (xcframework) | packaging in progress |
| Kotlin | Maven (AAR) | packaging in progress |

Packages are self-contained (onnxruntime is statically linked — see
[embedding-packaging.md](embedding-packaging.md)); only the model directory is
side-loaded.
