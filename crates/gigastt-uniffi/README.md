# gigastt-uniffi

Idiomatic **Swift**, **Kotlin**, and **Python** bindings for [gigastt](https://github.com/ekhodzitsky/gigastt) — on-device Russian speech-to-text — generated from one Rust source with [UniFFI](https://mozilla.github.io/uniffi-rs/).

Wraps the synchronous `gigastt-core` engine: models are **side-loaded** (no HTTP download dependency) and inference uses the blocking pool path (**no tokio runtime**), so the bindings are lean. Errors are **typed** (`GigasttError` → Swift `throws` / Kotlin exceptions / Python exceptions) and objects are reference-counted (no manual free).

## API

| Type | Methods |
|---|---|
| `Engine` | `new(model_dir)` · `new_with_pool_size(model_dir, pool_size)` · `transcribe_file(path) -> Transcript` |
| `Stream` | `new(engine)` · `process_chunk(pcm16, sample_rate) -> [TranscriptSegment]` · `flush() -> [TranscriptSegment]` |
| records | `Transcript { text, words, duration_s }` · `TranscriptSegment { text, words, is_final }` · `Word { text, start_s, end_s, confidence, speaker }` |
| errors | `GigasttError`: `ModelNotFound` · `InvalidAudio` · `PoolExhausted` · `Inference` · `InvalidArgument` |

`process_chunk` takes little-endian mono PCM16 and resamples to 16 kHz internally.

## Generating the bindings

Build the cdylib, then run the version-pinned generator against it:

```sh
cargo build -p gigastt-uniffi
LIB=target/debug/libgigastt_uniffi.dylib   # .so on Linux

cargo run -p gigastt-uniffi --bin uniffi-bindgen -- generate --library "$LIB" --language python --out-dir bindings/python
cargo run -p gigastt-uniffi --bin uniffi-bindgen -- generate --library "$LIB" --language swift  --out-dir bindings/swift
cargo run -p gigastt-uniffi --bin uniffi-bindgen -- generate --library "$LIB" --language kotlin --out-dir bindings/kotlin
```

Generated bindings are build artifacts (`bindings/` is git-ignored). Packaging them into a SwiftPM `.xcframework`, an Android `.aar`, and a PyPI wheel is the next step (prebuilt-artifacts task).

## Python (quickstart, verified)

Install the prebuilt wheel — `pip install gigastt` — no compiler, no `protoc`, no
onnxruntime download (it is statically linked; the wheel is `py3-none-<platform>`,
one per platform across all Python 3.x). The ~215 MB model is side-loaded at
runtime. Wheels are built + published by `.github/workflows/python-wheels.yml`.

```python
# pip install gigastt
import gigastt_uniffi as g

engine = g.Engine("/path/to/gigastt/models")        # side-loaded model dir
t = engine.transcribe_file("recording.wav")
print(t.text)                                       # -> "шестьдесят тысяч тенге сколько будет стоить"
for w in t.words:
    print(w.text, w.start_s, w.end_s, w.confidence)

# streaming
s = g.Stream(engine)
for seg in s.process_chunk(pcm16_bytes, 16000):
    print(seg.text)
print([seg.text for seg in s.flush()])
```

Errors surface as exceptions:

```python
try:
    g.Engine("/no/such/dir")
except g.GigasttError.ModelNotFound as e:
    ...
```

## Features

`coreml` / `cuda` / `nnapi` forward to `gigastt-core` for hardware acceleration. The core dependency is lean (`file-decode` only — no `net`, no `async-pool`).

## License

MIT.
