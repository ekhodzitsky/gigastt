# gigastt Android (AAR)

Android library for [gigastt](https://github.com/ekhodzitsky/gigastt) —
on-device Russian speech-to-text (GigaAM v3) — via the UniFFI Kotlin bindings.

> **Status: experimental.** The Rust cross-build is proven (CI cross-compiles the
> native library via cargo-ndk), but the Gradle/Maven AAR assembly and publish
> have not yet been validated end-to-end on a real Android toolchain. Verify with
> a local Android SDK/NDK before relying on a published artifact.

## What the AAR contains

- `jniLibs/<abi>/libgigastt_uniffi.so` for `arm64-v8a`, `armeabi-v7a`, `x86_64`.
  onnxruntime is statically linked into each `.so`, so the AAR is self-contained.
- The UniFFI-generated Kotlin bindings (idiomatic `Engine` / `Stream` + typed
  exceptions).
- A JNA dependency (`net.java.dev.jna:jna@aar`) — UniFFI Kotlin calls the native
  library through JNA.

The ~215 MB INT8 model is **not** bundled; side-load it at runtime (ship the
model directory with the app or download it) and pass its path to `Engine`.

## Build

The native libs + Kotlin are generated before assembling (not committed):

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

CI: `.github/workflows/android-aar.yml` (`workflow_dispatch`) runs the above and,
with `publish: true` + Maven credentials, publishes the AAR.

## Usage

```kotlin
val engine = Engine("/path/to/models")     // side-loaded model dir
val t = engine.transcribeFile("recording.wav")
println(t.text)
```

## License

MIT.
