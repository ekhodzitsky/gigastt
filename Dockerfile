# Multi-stage build for gigastt
# Build: docker build -t gigastt .
# Run:   docker run -p 9876:9876 gigastt

# --- Builder stage ---
FROM rust:1.85-bookworm AS builder

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY tests/ tests/

# Build release binary
RUN cargo build --release && \
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
