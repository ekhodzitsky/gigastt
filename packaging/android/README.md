# gigastt Android (AAR)

Android library for [gigastt](https://github.com/ekhodzitsky/gigastt) —
on-device Russian speech-to-text (GigaAM v3) — via the UniFFI Kotlin bindings.

> **Status: experimental.** A [nonpublishing CI run](https://github.com/ekhodzitsky/gigastt/actions/runs/36887391146)
> verified native builds for all three ABIs, Kotlin generation, and Gradle AAR
> assembly. Device execution, Maven publication, and release attachment have
> not been tested. Validate device behavior before relying on a published artifact.

## What the AAR contains

- `jniLibs/<abi>/libgigastt_uniffi.so` + `libonnxruntime.so` for `arm64-v8a`,
  `armeabi-v7a`, `x86_64`. onnxruntime is dynamically linked: each ABI folder
  carries its own `libonnxruntime.so` (from the official Microsoft
  onnxruntime-android AAR), which Android's dynamic linker resolves from the
  same `jniLibs/<abi>/` directory at load time.
- The UniFFI-generated Kotlin bindings (idiomatic `Engine` / `Stream` + typed
  exceptions).
- A JNA dependency (`net.java.dev.jna:jna@aar`) — UniFFI Kotlin calls the native
  library through JNA.

The ~215 MB INT8 model is **not** bundled; side-load it at runtime (ship the
model directory with the app or download it) and pass its path to `Engine`.

## Build

The native libs + Kotlin are generated before assembling (not committed):

```sh
# Per-ABI native libs. A single `cargo ndk -t arm64-v8a -t armeabi-v7a -t x86_64`
# call does NOT work: ort's pyke prebuilts do not cover armeabi-v7a.
# Instead (mirrors android-aar.yml):
# 1. Fetch the official Microsoft onnxruntime-android AAR and unpack one
#    libonnxruntime.so per ABI:
ORT_VER=$(python3 -c 'import json; print(json.load(open("packaging/android/onnxruntime.json"))["version"])')
curl -fsSL -o ort-android.aar \
  "https://repo.maven.apache.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/${ORT_VER}/onnxruntime-android-${ORT_VER}.aar"
cargo metadata --locked --format-version 1 > android-cargo-metadata.json
python3 scripts/check-android-runtime.py --metadata android-cargo-metadata.json --aar ort-android.aar
for abi in arm64-v8a armeabi-v7a x86_64; do
  mkdir -p "ort-lib/$abi"
  unzip -p ort-android.aar "jni/$abi/libonnxruntime.so" > "ort-lib/$abi/libonnxruntime.so"
done
# 2. Build each ABI separately with ORT_LIB_LOCATION pointing at its own
#    onnxruntime .so (dynamic link), then copy the .so next to our cdylib:
export ORT_PREFER_DYNAMIC_LINK=1
for abi in arm64-v8a armeabi-v7a x86_64; do
  ORT_LIB_LOCATION="$PWD/ort-lib/$abi" \
  cargo ndk -t "$abi" \
    -o packaging/android/gigastt/src/main/jniLibs build --release -p gigastt-uniffi
  cp "ort-lib/$abi/libonnxruntime.so" \
     "packaging/android/gigastt/src/main/jniLibs/$abi/"
done
# Kotlin bindings (from a host build of the cdylib; metadata is arch-independent)
cargo build --release -p gigastt-uniffi
cargo run --release -p gigastt-uniffi --bin uniffi-bindgen -- generate \
  --library target/release/libgigastt_uniffi.* --language kotlin \
  --out-dir packaging/android/gigastt/src/main/kotlin
# assemble
cd packaging/android && gradle :gigastt:assembleRelease
```

CI: `.github/workflows/android-aar.yml` (`workflow_dispatch`) runs the same
per-ABI flow above (fetch the onnxruntime-android AAR, build each ABI with
`ORT_LIB_LOCATION`, copy `libonnxruntime.so` into each `jniLibs/<abi>/`) and,
with `publish: true` + Maven credentials, runs the Maven publication step.

An empty `tag` performs a build of the dispatch commit and uploads a workflow
artifact only. Publication requires an explicit existing version tag such as
`v2.22.0`. The workflow resolves that tag to a commit before compilation and
rejects a mismatch with the Rust workspace or member versions. Native libraries
and generated Kotlin bindings are built from that exact commit; both Gradle
assembly and publication receive its version through `-PVERSION_NAME`, overriding
stale properties in older source tags. The AAR filename uses the same version.

A tagged run also requires the GitHub release to exist before building. Attachment
uses an upload-only operation and never creates a release. Selecting `main` as the
workflow dispatch ref therefore cannot substitute its native source for the chosen
tag. These provenance checks do not validate the Android toolchain or configure a
Maven repository; the experimental status above still applies.

## Usage

```kotlin
val engine = Engine("/path/to/models")     // side-loaded model dir
val t = engine.transcribeFile("recording.wav")
println(t.text)
```

## License

MIT.

## Native runtime compatibility

The official Microsoft AAR is pinned to **1.31.0** with its SHA-256 in
[`onnxruntime.json`](onnxruntime.json). Its C header provides API 31, matching
`ort` 2.0.0-rc.15 default features. Older native runtimes cannot satisfy
that API request. The archive contains ELF libraries for all three packaged ABIs:
`arm64-v8a`, `armeabi-v7a`, and `x86_64`.

Before cross-compilation, the workflow compares the resolved `ort-sys` API features
from Cargo metadata with the pin, verifies the archive checksum and C API header,
and checks each ABI library. PR CI checks dependency-versus-pin drift without
downloading the AAR. Tagged builds use the validator and pin from the invoking
workflow revision, even when the selected source tag predates those files.

Archive/header checks and x86_64 symbol inspection establish the supplied API.
The nonpublishing CI run above also passed the runtime archive/API checks, native builds for
all three ABIs, Kotlin generation, and Gradle assembly on source commit
`673d1e2`. It did not execute the library on an Android device, publish to Maven,
or attach an AAR to a release; those remain unverified.
Official installation guidance: [ONNX Runtime for Android](https://onnxruntime.ai/docs/install/#install-on-android).
