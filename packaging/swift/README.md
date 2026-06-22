# GigaSTT for Swift

On-device Russian speech-to-text for iOS, powered by GigaAM v3 and ONNX Runtime.
No cloud APIs, no network calls at inference time, full privacy.

This package wraps the gigastt C ABI in a safe Swift interface. The native code
ships as a prebuilt `GigasttFFI.xcframework` (device `arm64` + simulator
`arm64`), with ONNX Runtime statically linked into each slice — there is no
separate runtime to bundle.

## Requirements

- iOS 15 or later
- Xcode 15 or later (Swift 5.9 tools)

## Installation

### Swift Package Manager (Xcode)

File -> Add Package Dependencies, then enter the repository URL:

```
https://github.com/ekhodzitsky/gigastt
```

Point the package to the `packaging/swift` directory and add the `GigaSTT`
product to your target.

### Package.swift

```swift
dependencies: [
    .package(url: "https://github.com/ekhodzitsky/gigastt", from: "0.9.0")
],
targets: [
    .target(name: "MyApp", dependencies: [
        .product(name: "GigaSTT", package: "gigastt")
    ])
]
```

## Shipping the model

The xcframework contains only the inference code. The GigaAM v3 model
(~215 MB INT8, or ~850 MB FP32) is not embedded — download it once on a
developer machine and bundle the directory with your app (or fetch it on
first launch and cache it).

Download the prequantized model with the gigastt CLI:

```sh
gigastt download --prequantized
```

This writes the model directory to `~/.gigastt/models/`. Copy that directory
into your app bundle as a folder reference (so the file layout is preserved),
then pass its on-device path to `Engine(modelDir:)`.

```swift
import GigaSTT

guard let modelDir = Bundle.main.url(
    forResource: "models", withExtension: nil
)?.path else {
    fatalError("bundle the model directory as a folder reference")
}

// poolSize: 1 keeps RAM around ~350 MB, recommended on device.
let engine = try Engine(modelDir: modelDir, poolSize: 1)
```

## File transcription

```swift
// Path is relative to the current working directory; absolute paths and ".."
// are rejected by the engine.
let text = try engine.transcribeFile(path: "audio.wav")
print(text)
```

## Real-time streaming

Feed little-endian mono PCM16 chunks at your capture sample rate. Audio is
resampled to 16 kHz internally.

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

`processChunk` and `flush` return `[TranscriptSegment]`, where each segment
carries `text`, `words` (per-word `start`/`end`/`confidence` and an optional
`speaker`), `isFinal`, and a `timestamp`.

## Memory management

`Engine` and `Stream` own their native handles and free them in `deinit`.
Strings returned by the C ABI are copied and freed internally — there is
nothing to release manually.

## License

MIT. See the repository root for details.
