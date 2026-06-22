# Embedding & packaging: onnxruntime linking

gigastt embeds ONNX Runtime through the [`ort`](https://ort.pyke.io) crate. How
the native onnxruntime library is linked decides whether a binding artifact is
self-contained or needs a companion shared library — the single most important
packaging decision when shipping gigastt inside another app.

## Default: static linking (recommended)

The default build (`ort`'s `download-binaries` feature) **statically links**
onnxruntime into the output. Every gigastt artifact is therefore
**self-contained** — the onnxruntime code is baked in, with no separate
`libonnxruntime.{so,dylib,dll}` to ship or locate at runtime:

- the `gigastt` server/CLI binary,
- the `gigastt-ffi` `cdylib` (`.so`/`.dylib`/`.dll`) and `staticlib` (`.a`),
- the `gigastt-node` Node addon (`.node`),
- the `gigastt-uniffi` Python wheel and the iOS static library.

This is verified in CI and locally: the binaries carry no dynamic onnxruntime
dependency. For the CPU execution provider this is the right default for every
binding — there is no `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH` to set, no RPATH to
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
| Build-time network | Fetches onnxruntime (checksummed) | None (vendor it yourself) |
| Best for | All bindings, CPU EP, simplest distribution | Air-gapped builds, size control, dynamic EP variants |

For the prebuilt bindings (npm, PyPI wheel, AAR, xcframework) gigastt ships, the
static default is used — each package is self-contained. `load-dynamic` is the
escape hatch for the cases above, not the default integration path.
