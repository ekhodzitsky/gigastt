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

## Rust hosts that initialize `ort` themselves

The pinned `ort` 2.0.0-rc.15 uses explicit environment handles and permits only
one environment per process. If the host creates that environment, share it
with gigastt **before the first engine or runtime session**:

```rust,ignore
let environment = ort::init().with_name("my-app").build()?;
if !gigastt_core::runtime_api::set_ort_environment(environment.clone()) {
    return Err("gigastt environment was already initialized".into());
}
// The host may also pass &environment to its own ort::session::Session::builder.
```

Use the same exact `ort` crate version as gigastt. The sharing function retains
the handle for the process lifetime; `false` means initialization was already
attempted and the existing configuration was preserved. This replaces the old
implicit adoption of a host's global environment. Hosts using `init_from`
should likewise call `.build()?` and pass the resulting handle. Standalone
servers and bindings need no additional initialization.

Dynamic native libraries must provide API 31 (ONNX Runtime 1.31 or newer).
Android packages pin the tested native runtime in
[`onnxruntime.json`](../packaging/android/onnxruntime.json).

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
