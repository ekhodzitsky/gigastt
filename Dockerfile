# Multi-stage build for gigastt
# Build: docker build -t gigastt .
# Run:   docker run -p 9876:9876 gigastt

# --- Builder stage ---
# Pinned to the workspace MSRV (`rust-version` in Cargo.toml) so the image
# build doubles as an MSRV check; bump both together. `--locked` keeps the
# image on the audited Cargo.lock graph — a fresh resolve could pull deps
# with a newer MSRV (polyvoice 0.18 raised the floor to 1.94).
# trixie (not bookworm): ort's prebuilt onnxruntime statics are compiled with
# gcc >= 13 and reference `__cxa_call_terminate` (CXXABI_1.3.15), which
# bookworm's libstdc++ 12 lacks — the final link fails there.
FROM rust:1.94-trixie AS builder

# `prost-build` (via build.rs) requires `protoc` at compile time; without it
# the build aborts with "prost-build failed to compile gigastt-quantize proto/onnx.proto".
RUN apt-get update && \
    apt-get install -y --no-install-recommends protobuf-compiler && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Dependency-compilation cache: copy manifests + build.rs + proto/ first and
# compile a dummy binary so `cargo build` downloads + builds every transitive
# crate. Subsequent edits to src/ only invalidate the final compilation
# layer, cutting incremental rebuild time from minutes to seconds.
COPY Cargo.toml Cargo.lock ./
COPY crates/gigastt-core/Cargo.toml crates/gigastt-core/
COPY crates/gigastt-quantize/Cargo.toml crates/gigastt-quantize/
COPY crates/gigastt-quantize/build.rs crates/gigastt-quantize/
COPY crates/gigastt-quantize/proto/ crates/gigastt-quantize/proto/
COPY crates/gigastt-ffi/Cargo.toml crates/gigastt-ffi/
COPY crates/gigastt/Cargo.toml crates/gigastt/
COPY crates/gigastt-uniffi/Cargo.toml crates/gigastt-uniffi/
COPY crates/gigastt-node/Cargo.toml crates/gigastt-node/
# Every [[bench]]/[[test]]/[[bin]] target declared in the manifests must exist
# on disk for cargo to parse the workspace, hence the extra stubs below.
# (gigastt-uniffi and gigastt-node are workspace members but not dependencies of
# `gigastt`, so they are never compiled here — their stubs only satisfy parsing.)
RUN mkdir -p crates/gigastt-core/src crates/gigastt-quantize/src crates/gigastt-ffi/src crates/gigastt/src/server crates/gigastt-uniffi/src/bin crates/gigastt-node/src && \
    echo 'pub mod error; pub mod inference; pub mod model; pub mod protocol;' > crates/gigastt-core/src/lib.rs && \
    echo '#[cfg(feature = "quantize")] pub mod quantize { pub use gigastt_quantize::*; }' >> crates/gigastt-core/src/lib.rs && \
    echo '#[cfg(feature = "quantize")] pub mod onnx_proto { pub use gigastt_quantize::onnx_proto::*; }' >> crates/gigastt-core/src/lib.rs && \
    mkdir -p crates/gigastt-core/src/inference crates/gigastt-core/src/model crates/gigastt-core/src/protocol && \
    touch crates/gigastt-core/src/error.rs && \
    touch crates/gigastt-core/src/inference/mod.rs crates/gigastt-core/src/model/mod.rs crates/gigastt-core/src/protocol/mod.rs && \
    echo 'pub mod onnx_proto {} pub fn quantize_model(_: &std::path::Path, _: &std::path::Path) -> anyhow::Result<()> { Ok(()) }' > crates/gigastt-quantize/src/lib.rs && \
    echo 'pub use gigastt_core::*; pub mod server;' > crates/gigastt/src/lib.rs && \
    echo 'fn main() {}' > crates/gigastt/src/main.rs && \
    touch crates/gigastt/src/server/mod.rs && \
    echo '' > crates/gigastt-ffi/src/lib.rs && \
    echo 'fn main() {}' > crates/gigastt-ffi/build.rs && \
    echo '' > crates/gigastt-uniffi/src/lib.rs && \
    echo 'fn main() {}' > crates/gigastt-uniffi/src/bin/uniffi-bindgen.rs && \
    echo '' > crates/gigastt-node/src/lib.rs && \
    echo 'fn main() {}' > crates/gigastt-node/build.rs && \
    mkdir -p crates/gigastt-core/benches crates/gigastt-core/examples crates/gigastt/tests && \
    echo 'fn main() {}' > crates/gigastt-core/benches/mel.rs && \
    echo 'fn main() {}' > crates/gigastt-core/benches/resample.rs && \
    echo 'fn main() {}' > crates/gigastt-core/benches/tokenizer.rs && \
    echo 'fn main() {}' > crates/gigastt-core/benches/decode.rs && \
    echo 'fn main() {}' > crates/gigastt-core/examples/quantize_file.rs && \
    echo 'fn main() {}' > crates/gigastt/tests/benchmark.rs && \
    echo '# stub' > crates/gigastt-quantize/README.md && \
    cargo build --release -p gigastt --locked && \
    rm -rf crates/*/src target/release/deps/gigastt* target/release/gigastt*

# Now bring in the actual source and build the real binary.
COPY crates/ crates/

# COPY preserves host mtimes (older than the dummy artifacts above), so cargo
# would consider the dummy-built workspace rlibs fresh and link the real
# binary against them. Touch the sources to force a rebuild of the workspace
# crates; the dependency cache stays valid.
RUN find crates -name '*.rs' -exec touch {} + && \
    cargo build --release -p gigastt --locked && \
    strip target/release/gigastt

# --- Model bake stage (runs only when GIGASTT_BAKE_MODEL=1) ---
# trixie-slim to match the builder's glibc/libstdc++ (the stage runs the binary).
FROM debian:trixie-slim AS model-fetcher

ARG GIGASTT_BAKE_MODEL=0

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/gigastt /usr/local/bin/gigastt

RUN mkdir -p /models && \
    if [ "$GIGASTT_BAKE_MODEL" = "1" ]; then \
        gigastt download --model-dir /models; \
    fi

# --- Runtime stage ---
FROM debian:trixie-slim

ARG GIGASTT_BAKE_MODEL=0

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/gigastt /usr/local/bin/gigastt

RUN groupadd -r gigastt && useradd -r -g gigastt gigastt && \
    mkdir -p /home/gigastt/.gigastt/models && chown -R gigastt:gigastt /home/gigastt

# Copy baked model files (only present when GIGASTT_BAKE_MODEL=1)
COPY --from=model-fetcher --chown=gigastt:gigastt /models/. /home/gigastt/.gigastt/models/

USER gigastt

ENV RUST_LOG=gigastt=info

EXPOSE 9876

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:9876/health || exit 1

# Download model if not present, then start server.
# `--bind-all` acknowledges that container networking requires listening on
# 0.0.0.0; outside Docker the default `127.0.0.1` bind stays in effect.
ENTRYPOINT ["gigastt"]
CMD ["serve", "--port", "9876", "--host", "0.0.0.0", "--bind-all"]
