# Self-Hosted GitHub Actions Runner for Benchmarks

GitHub-hosted runners cover correctness CI — the `macos-14` image even runs on
Apple Silicon, which is why the release workflow builds the CoreML binary there
(`.github/workflows/release.yml`). What they do not give you is stable,
representative hardware for performance work:

- **macOS** — hosted runners are shared, variably loaded VMs; RTF (real-time
  factor) timings fluctuate run to run, so CoreML / Neural Engine speedups
  cannot be measured meaningfully there.
- **Linux** — a CPU-only runner can compile a CUDA build but cannot validate
  GPU inference or its performance without a compatible NVIDIA GPU.

For accurate RTF (real-time factor) measurements you need self-hosted runners on your actual target hardware.

## Supported configurations

| Runner tag | OS | Hardware | Use case |
|------------|-----|----------|----------|
| `benchmark-macos-arm64` | macOS 14+ | Apple Silicon (M1–M4) | CoreML + Neural Engine |
| `benchmark-linux-cuda` | Ubuntu 22.04+ | NVIDIA GPU + CUDA 12+ | CUDA inference |
| `benchmark-linux-cpu` | Ubuntu 22.04+ | x86_64 CPU | Baseline CPU comparison |

## Quick start (macOS ARM64)

### 1. Install the runner agent

The version below is an example — take the current one from
[actions/runner releases](https://github.com/actions/runner/releases/latest)
(GitHub also auto-updates a registered runner).

```bash
mkdir ~/actions-runner && cd ~/actions-runner
RUNNER_VERSION=2.320.0   # replace with the latest release
curl -o "actions-runner-osx-arm64-${RUNNER_VERSION}.tar.gz" \
  -L "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-osx-arm64-${RUNNER_VERSION}.tar.gz"
tar xzf "actions-runner-osx-arm64-${RUNNER_VERSION}.tar.gz"
```

### 2. Configure

Get a token from **GitHub → Settings → Actions → Runners → New self-hosted runner**.

```bash
./config.sh --url https://github.com/ekhodzitsky/gigastt \
  --token YOUR_TOKEN \
  --name benchmark-m1-studio \
  --labels benchmark-macos-arm64 \
  --work _work
```

### 3. Install dependencies

```bash
# Rust
brew install rustup protobuf
rustup-init -y

# Python
brew install python@3.12

# gigastt model cache (one-time)
git clone https://github.com/ekhodzitsky/gigastt /tmp/gigastt-setup
cd /tmp/gigastt-setup
cargo build --release -p gigastt --features coreml
./target/release/gigastt download
```

### 4. Run

```bash
./run.sh
```

For auto-start on boot create a `launchd` plist or use `screen` / `tmux`.

## Quick start (Linux + CUDA)

### 1. Install the runner agent

```bash
mkdir ~/actions-runner && cd ~/actions-runner
RUNNER_VERSION=2.320.0   # replace with the latest release
curl -o "actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
  -L "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
tar xzf "actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
```

### 2. Configure

```bash
./config.sh --url https://github.com/ekhodzitsky/gigastt \
  --token YOUR_TOKEN \
  --name benchmark-rtx4090 \
  --labels benchmark-linux-cuda \
  --work _work
```

### 3. Install dependencies

```bash
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

# System
sudo apt update
sudo apt install -y build-essential cmake pkg-config libssl-dev protobuf-compiler ffmpeg python3.12 python3.12-venv

# CUDA 12 (already present on most GPU servers)
nvidia-smi

# gigastt model cache (one-time)
git clone https://github.com/ekhodzitsky/gigastt /tmp/gigastt-setup
cd /tmp/gigastt-setup
cargo build --release --features cuda -p gigastt
./target/release/gigastt download
```

### 4. Run as a service

```bash
sudo ./svc.sh install
sudo ./svc.sh start
```

## Workflow adaptation

Edit `.github/workflows/benchmark.yml` and replace:

```yaml
runs-on: ubuntu-latest
```

with a matrix:

```yaml
strategy:
  matrix:
    include:
      - runner: benchmark-macos-arm64
        name: macOS-ARM64-CoreML
      - runner: benchmark-linux-cuda
        name: Linux-CUDA
      - runner: benchmark-linux-cpu
        name: Linux-CPU
runs-on: ${{ matrix.runner }}
```

Then cache keys and artifact names should include `${{ matrix.name }}` to avoid collisions.

## Security hardening

Self-hosted runners execute arbitrary code from PRs. Mitigations:

1. **Require approval** for first-time contributors:  
   Settings → Actions → General → "Require approval for all outside collaborators"

2. **Dispatch benchmarks manually on a reviewed revision.** The current
   workflow uses only `workflow_dispatch`; keep it that way when adapting runner
   labels. Approval of a workflow run is not a review of arbitrary source code.

3. **Network isolation** — place the runner in a restricted VLAN with no access to internal infrastructure.

4. **Ephemeral runners** (recommended for production): use a VM or container that is destroyed after each job.  
   GitHub supports ephemeral mode: `./config.sh --ephemeral`.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `protoc not found` | `brew install protobuf` / `apt install protobuf-compiler` |
| `CUDA not available` inside Docker | Add `--gpus all` to `docker run` or `runtime: nvidia` in compose |
| Model download hangs in CI | Pre-download to `~/.gigastt/models` on the host and mount as volume |
| Runner shows offline | Check `./run.sh` logs; verify token is not expired |
