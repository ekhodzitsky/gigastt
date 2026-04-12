# Multi-stage build for gigastt
# Build: docker build -t gigastt .
# Run:   docker run -p 9876:9876 gigastt

# --- Builder stage ---
FROM rust:1.83-bookworm AS builder

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY tests/ tests/

# Build release binary
RUN cargo build --release && \
    strip target/release/gigastt

# --- Runtime stage ---
FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/gigastt /usr/local/bin/gigastt

# Model will be downloaded on first run to /root/.gigastt/models
ENV RUST_LOG=gigastt=info

EXPOSE 9876

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:9876/health || exit 1

# Download model if not present, then start server
# Bind 0.0.0.0 inside container (not 127.0.0.1)
ENTRYPOINT ["gigastt"]
CMD ["serve", "--port", "9876", "--host", "0.0.0.0"]
