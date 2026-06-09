# Reproducible cross-ASR benchmark environment.
# Builds gigastt + whisper.cpp from source and installs Python ASR libraries.
#
# Usage:
#   docker build -f benchmark/Dockerfile -t gigastt-benchmark .
#   docker run -v ~/.gigastt/models:/root/.gigastt/models \
#              -v ~/.gigastt/benchmarks:/root/.gigastt/benchmarks \
#              -v $(pwd)/benchmark/results:/results \
#              gigastt-benchmark \
#              --max-samples 100 --runners all --output /results/results.json

FROM ubuntu:24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV RUST_LOG=error

# --- System dependencies ---
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    protobuf-compiler \
    python3.12 \
    python3.12-venv \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# --- Rust toolchain ---
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# --- Python virtual environment ---
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
COPY benchmark/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace

# =============================================================================
# Stage: whisper.cpp build
# =============================================================================
FROM base AS whisper-builder

RUN git clone --depth 1 --branch v1.7.5 https://github.com/ggml-org/whisper.cpp.git /whisper.cpp \
    && cmake -B /whisper.cpp/build -S /whisper.cpp -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /whisper.cpp/build --config Release -j$(nproc)

# =============================================================================
# Stage: gigastt build
# =============================================================================
FROM base AS gigastt-builder

COPY Cargo.toml Cargo.lock /workspace/
COPY crates/ /workspace/crates/
COPY proto/ /workspace/proto/
RUN cargo build --release -p gigastt

# =============================================================================
# Final runtime image
# =============================================================================
FROM base AS runtime

# Copy built artifacts
COPY --from=whisper-builder /whisper.cpp/build/bin/whisper-cli /usr/local/bin/whisper-cli
COPY --from=gigastt-builder /workspace/target/release/gigastt /usr/local/bin/gigastt

# Copy benchmark sources
COPY benchmark/ /workspace/benchmark/

WORKDIR /workspace/benchmark

# Ensure cache dirs exist
RUN mkdir -p /root/.cache/whisper.cpp /root/.cache/vosk

# Default: run benchmark on 100 samples across all available engines
ENTRYPOINT ["python3", "benchmark.py"]
CMD ["--max-samples", "100", "--output", "/workspace/benchmark/results.json"]
