# Embedding & packaging: onnxruntime linking

gigastt embeds ONNX Runtime through the [`ort`](https://ort.pyke.io) crate. How
the native onnxruntime library is linked decides whether a binding artifact is
self-contained or needs a companion shared library — the single most important
packaging decision when shipping gigastt inside another app.

## Default: static linking (recommended)

The default build (`ort`'s `download-binaries` feature) **statically links**
onnxruntime into supported desktop CPU outputs. These artifacts need no separate
`libonnxruntime.{so,dylib,dll}` to ship or locate at runtime:

- the `gigastt` server/CLI binary,
- the `gigastt-ffi` `cdylib` (`.so`/`.dylib`/`.dll`) and `staticlib` (`.a`),
- the `gigastt-node` Node addon (`.node`),
- the `gigastt-uniffi` desktop Python wheel and supported ARM64 Apple slices.

This is not a universal platform guarantee: the Android AAR packages native
ONNX Runtime libraries separately, and the Swift x86_64 simulator slice defers
linking to the consumer. Follow the [Android](../packaging/android/README.md)
and [Swift](../packaging/swift/README.md) packaging instructions.

This is verified in CI and locally: the binaries carry no dynamic onnxruntime
dependency. For supported desktop CPU targets this is the default — there is no `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH` to set, no RPATH to
manage, and no dylib to bundle alongside the addon.

The trade-off is build-time: `download-binaries` fetches a prebuilt onnxruntime
over the network at compile time (verified by an embedded checksum). The "no
cloud / full privacy" guarantee covers **runtime** inference, not the build.

## Opt-in: `ort-load-dynamic`

`gigastt-core` exposes an `ort-load-dynamic` Cargo feature (off by default) that
switches `ort` to its `load-dynamic` strategy: the build links nothing, and
onnxruntime is `dlopen`-ed at runtime. Enable it when you need one of:

- **Air-gapped / offline builds** — no build-time download; point at a vendored
  or system onnxruntime instead.
- **Native-library size control** — share one onnxruntime across several
  artifacts, or pick a slimmer/custom onnxruntime build per target.
- **Execution-provider variants** — accelerated providers (e.g. CUDA) that are
  distributed as dynamic libraries.

With `load-dynamic` you become responsible for shipping and locating
`libonnxruntime` at runtime. `ort` resolves it from the `ORT_DYLIB_PATH`
environment variable (an absolute path is the robust choice — a relative one is
resolved against the executable, not the loading library), or programmatically
via `ort::init_from(<path>)` before the first inference.

```sh
# Lean core compiled against a system/vendored onnxruntime, no build-time fetch:
cargo build -p gigastt-core --no-default-features --features "file-decode,ort-load-dynamic"
ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so ./your-app
```

## Which to choose

| | Default (`download-binaries`, static) | `ort-load-dynamic` |
|---|---|---|
| Artifact | Self-contained — no companion lib | Needs `libonnxruntime` at runtime |
| Runtime setup | None | `ORT_DYLIB_PATH` / `init_from` |
| Native runtime build-time fetch | Fetches ONNX Runtime (checksummed) | None; provision it yourself |
| Best for | Supported desktop CPU bindings, simplest distribution | Air-gapped builds, size control, dynamic EP variants |

The native runtime is only one build input: a fully offline build must also
provide Cargo dependencies, toolchains and generators locally. GPU libraries
and platform system libraries can still be required even with statically linked
ONNX Runtime. Inspect the exact packaged target rather than assuming all slices
have identical linking behavior.

## Optional GigaAM v4 file engine

GigaAM v4 is a separate **side-loaded FP32 CPU file-only** engine. Its optional
`v4-mkl` Cargo feature is available only on Linux x86-64 GNU, with numerical
qualification limited to the measured runtime and host. Build
with that feature in the desired core or binding crate. The default v3 engine,
model downloads and existing binding constructors/stream APIs are unchanged.
Enabling v4 does not itself fetch or distribute its model or MKL libraries;
provision the exact validated v4 model pack and trusted local MKL runtime as
separate assets. See [v4 pack validation](v4-pack-validation.md) for the format
and runtime requirements. No implicit fallback is performed.

The Rust API exposes `gigastt_core::inference::v4::V4Engine::load(pack_dir,
mkl_runtime_dir)`, `transcribe` for validated native PCM, and with `file-decode`
`transcribe_file` / `transcribe_bytes` for native mono container decode. Select
`TranscriptStyle::Raw` or `TranscriptStyle::E2e` explicitly. The `_with_limit`
file methods can lower but never raise the 30-second whole-context cap. No
streaming, long-form stitching, hotwords, VAD, diarization or raw codec options
are implemented by this engine. Rust methods accept a cooperative cancellation
flag; foreign file-only bindings do not expose cancellation. The literal decoded
text, word timing/groups, exact token IDs/frames/probabilities and duration are
reported; confidence is an uncalibrated score.

### C ABI

Build with `cargo build -p gigastt-ffi --features v4-mkl`; the generated
`gigastt.h` declares new symbols without altering the existing v3 ABI. The
capability probe and separate v4 symbols also exist in default builds, and
return an explicit `unsupported_capability` error when unavailable.

```c
#include "gigastt.h"
#include <stdio.h>

void recognize_v4(void) {
    if (!gigastt_v4_supported()) return;
    struct GigasttV4EngineResult engine = gigastt_v4_engine_new(
        "/path/to/v4-pack", "/path/to/mkl-runtime", "e2e");
    if (engine.engine) {
        char *json = gigastt_v4_transcribe_file(engine.engine, "recording.wav");
        if (json) puts(json);
        gigastt_string_free(json);
        gigastt_v4_engine_free(engine.engine);
    } else {
        if (engine.error) puts(engine.error);
        gigastt_string_free(engine.error);
    }
}
```

The constructor returns either a separate opaque v4 engine ID or an owned
UTF-8 JSON error string. File transcription returns owned JSON containing
`result` (`style`, `text`, `words`, `tokens`, `duration_s`, `confidence`) or
`error` (`code`, `message`). Free every returned string with
`gigastt_string_free`, and every successful v4 handle with
`gigastt_v4_engine_free`. Like the legacy C ABI, audio paths must be relative
and stay inside the working directory. V4 handles cannot be used with v3
file/stream APIs, or vice versa. One engine serializes inference; freeing a
handle does not cancel work that already acquired it.

### Other bindings

- [Node.js](../crates/gigastt-node/README.md#optional-gigaam-v4-file-engine-linux-x86-64-gnu):
  `v4Supported()`, `new V4Engine(packDir, mklRuntimeDir, style)`,
  asynchronous `transcribeFile`.
- [UniFFI](../crates/gigastt-uniffi/README.md#optional-gigaam-v4-file-engine-linux-x86-64-gnu):
  `v4_supported`, separate `V4Engine` constructor and synchronous
  `transcribe_file`, with typed `V4BindingError`.

These are opt-in source-build capabilities; no v4 binary or model publication
is implied for existing npm, Python, mobile or Apple artifacts.
