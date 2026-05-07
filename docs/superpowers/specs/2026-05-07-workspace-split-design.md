# Workspace Split — Design Spec

**Date:** 2026-05-07
**Sub-project:** 2 of 3 (README ✅ → Workspace split → Python bindings)

---

## Goal

Split the single-crate gigastt project into a Cargo workspace with 3 crates to enable independent compilation, FFI bindings with auto-generated C headers, and a clean separation of concerns.

## Workspace Layout

```
gigastt/
  Cargo.toml                  # [workspace] root
  crates/
    gigastt-core/             # lib — inference engine, model, quantize, protocol, error
      Cargo.toml
      build.rs                # prost-build for onnx.proto
      proto/onnx.proto
      src/
        lib.rs
        error.rs
        onnx_proto.rs
        quantize.rs
        inference/
          mod.rs
          audio.rs
          decode.rs
          features.rs
          tokenizer.rs
        model/
          mod.rs
        protocol/
          mod.rs

    gigastt-ffi/              # cdylib — C FFI + cbindgen
      Cargo.toml
      cbindgen.toml
      build.rs                # cbindgen → include/gigastt.h
      src/
        lib.rs                # current ffi.rs adapted

    gigastt/                  # bin — axum server + CLI
      Cargo.toml
      src/
        main.rs
        server/
          mod.rs
          config.rs
          http.rs
          ws.rs
          metrics.rs
          middleware.rs
          rate_limit.rs
```

## Crate Details

### gigastt-core (lib)

- **Exports:** inference::Engine, inference::StreamingState, model::download, quantize, protocol types, error types
- **Dependencies:** ort, tokio (rt + sync), rustfft, rubato, symphonia, bytes, reqwest, sha2, hex, serde, serde_json, prost, anyhow, thiserror, tracing, futures-util
- **Optional deps:** polyvoice (feature = "diarization")
- **Features:** diarization, coreml (ort/coreml), cuda (ort/cuda)
- **Build deps:** prost-build
- **crate-type:** ["rlib"]

### gigastt-ffi (cdylib)

- **Depends on:** gigastt-core
- **Dependencies:** serde_json (for JSON serialization of results)
- **Build deps:** cbindgen (generates include/gigastt.h)
- **Features:** nnapi (ort/nnapi, for Android), coreml, cuda (forwarded to gigastt-core)
- **crate-type:** ["cdylib"]

### gigastt (bin)

- **Depends on:** gigastt-core
- **Dependencies:** axum (ws, multipart), tokio (full), tokio-stream, tokio-util, futures-util, clap, tracing-subscriber, arc-swap, dashmap, uuid, toml, serde, serde_json, anyhow, tracing
- **Features:** diarization, coreml, cuda (forwarded to gigastt-core)
- **`cargo install gigastt`** continues to work — binary crate keeps the name

## Import Changes

All `use crate::inference`, `use crate::error`, `use crate::protocol`, `use crate::model`, `use crate::quantize` in server/ and ffi.rs become `use gigastt_core::...`.

Server-internal imports (`use crate::server::...`) stay as `use crate::server::...` within the gigastt crate.

## Feature Forwarding

```toml
# gigastt/Cargo.toml
[features]
default = ["diarization"]
diarization = ["gigastt-core/diarization"]
coreml = ["gigastt-core/coreml"]
cuda = ["gigastt-core/cuda"]
```

## Backward Compatibility

- `cargo install gigastt` — works (binary crate named `gigastt`)
- `gigastt-core` published to crates.io as a library for embedding
- Existing `use gigastt::inference::Engine` in downstream code → `use gigastt_core::inference::Engine`
- This is a semver-major change (2.0.0) for the library API

## Files That Stay at Root

- README.md, README_RU.md, CLAUDE.md, CHANGELOG.md, LICENSE, CONTRIBUTING.md
- .github/, docs/, examples/, ffi/, tests/
- Dockerfile, Dockerfile.cuda, deny.toml

## Test Strategy

- Unit tests stay inside each crate's source files
- Integration/e2e tests stay in root `tests/` and depend on `gigastt` (server) crate
- `cargo test --workspace` runs all tests

## Success Criteria

- `cargo build --workspace` compiles
- `cargo test --workspace` passes all 153 unit tests
- `cargo build -p gigastt-ffi` produces libgigastt.so/dylib + gigastt.h
- `cargo clippy --workspace` clean
- `cargo install --path crates/gigastt` works
