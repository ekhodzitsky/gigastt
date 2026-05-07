# Multi-stage build for gigastt
# Build: docker build -t gigastt .
# Run:   docker run -p 9876:9876 gigastt

# --- Builder stage ---
FROM rust:1.85-bookworm AS builder

# `prost-build` (via build.rs) requires `protoc` at compile time; without it
# the build aborts with "prost-build failed to compile proto/onnx.proto".
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
COPY crates/gigastt-core/build.rs crates/gigastt-core/
COPY crates/gigastt-core/proto/ crates/gigastt-core/proto/
COPY crates/gigastt-ffi/Cargo.toml crates/gigastt-ffi/
COPY crates/gigastt/Cargo.toml crates/gigastt/
RUN mkdir -p crates/gigastt-core/src crates/gigastt-ffi/src crates/gigastt/src/server && \
    echo 'pub mod error; pub mod inference; pub mod model; pub mod onnx_proto; pub mod protocol; pub mod quantize;' > crates/gigastt-core/src/lib.rs && \
    mkdir -p crates/gigastt-core/src/inference crates/gigastt-core/src/model crates/gigastt-core/src/protocol && \
    touch crates/gigastt-core/src/error.rs crates/gigastt-core/src/onnx_proto.rs crates/gigastt-core/src/quantize.rs && \
    touch crates/gigastt-core/src/inference/mod.rs crates/gigastt-core/src/model/mod.rs crates/gigastt-core/src/protocol/mod.rs && \
    echo 'pub use gigastt_core::*; pub mod server;' > crates/gigastt/src/lib.rs && \
    echo 'fn main() {}' > crates/gigastt/src/main.rs && \
    touch crates/gigastt/src/server/mod.rs && \
    echo '' > crates/gigastt-ffi/src/lib.rs && \
    echo 'fn main() {}' > crates/gigastt-ffi/build.rs && \
    cargo build --release -p gigastt && \
    rm -rf crates/*/src target/release/deps/gigastt* target/release/gigastt*

# Now bring in the actual source and build the real binary.
COPY crates/ crates/

RUN cargo build --release -p gigastt && \
    strip target/release/gigastt

# --- Model bake stage (runs only when GIGASTT_BAKE_MODEL=1) ---
FROM debian:bookworm-slim AS model-fetcher

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
FROM debian:bookworm-slim

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
