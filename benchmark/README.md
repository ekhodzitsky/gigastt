# gigastt Cross-ASR Benchmark

Reproducible benchmark comparing **gigastt** against popular open-source ASR engines on Russian speech.

## Supported Engines

| Engine | Backend | Language | Installation |
|--------|---------|----------|--------------|
| gigastt | ONNX Runtime / Rust | Russian | Built from source or `cargo install` |
| whisper.cpp | GGML / C++ | Multilingual | Auto-downloaded on first run |
| faster-whisper | CTranslate2 / Python | Multilingual | `pip install faster-whisper` |
| Vosk | Kaldi / C++ | Russian | `pip install vosk` (model auto-downloaded) |

## Metrics

- **WER** (Word Error Rate) — lower is better. Computed with the same text-normalization pipeline across all engines (lowercase, ё→е, digit-to-words, anglicisms, punctuation stripped).
- **RTF** (Real-Time Factor) — `processing_time / audio_duration`. Lower is better; < 1.0 means faster than real-time.

## Quick Start

```bash
cd benchmark
pip install -r requirements.txt

# Run on 100 samples (default)
python benchmark.py

# Run on full Golos crowd dataset (slow!)
python benchmark.py --max-samples 0 --output results_full.json

# Run only specific engines
python benchmark.py --runners gigastt,whisper_cpp

# Use environment variable for limit
GIGASTT_BENCHMARK_MAX_SAMPLES=50 python benchmark.py
```

## Docker (fully isolated)

If you prefer not to install Python dependencies locally, use the provided Dockerfile:

```bash
# Build image
docker build -f benchmark/Dockerfile -t gigastt-benchmark .

# Run benchmark with mounted model caches
docker run -v ~/.gigastt/models:/root/.gigastt/models:ro \
           -v ~/.gigastt/benchmarks:/root/.gigastt/benchmarks:ro \
           -v $(pwd)/benchmark/results:/workspace/benchmark/results \
           gigastt-benchmark \
           --max-samples 100 --runners all
```

Or use Docker Compose:

```bash
cd benchmark
GIGASTT_BENCHMARK_MAX_SAMPLES=100 docker-compose up
```

> **Note:** On macOS, Docker Desktop must be running. On Linux with NVIDIA GPUs, add `runtime: nvidia` to `docker-compose.yml` and use `--gpus all` with `docker run`.

## Dataset

The benchmark uses the **Golos crowd** test set (9 994 samples of Russian speech).

```bash
# Download and extract (one-time)
python ../scripts/extract_golos.py
```

If the external dataset is missing, the benchmark falls back to the bundled fixtures (15 samples) from `crates/gigastt/tests/fixtures/`.

## Output Format

`results.json` contains per-engine summaries and per-sample details:

```json
{
  "manifest_samples": 100,
  "runners": [
    {
      "name": "gigastt",
      "samples": 100,
      "wer": 11.40,
      "rtf": 0.045,
      "total_errors": 57,
      "total_ref_words": 500,
      "details": [...]
    }
  ]
}
```

## CI / Automation

A GitHub Action runs the benchmark nightly on a self-hosted runner and commits `results.json` to the `benchmark-results` branch. See `.github/workflows/benchmark.yml`.

### Badges

Add to your README:

```markdown
![WER](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fekhodzitsky%2Fgigastt%2Fbenchmark-results%2Fresults.json&query=%24.runners%5B0%5D.wer&suffix=%25&label=WER&color=blue)
![RTF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fekhodzitsky%2Fgigastt%2Fbenchmark-results%2Fresults.json&query=%24.runners%5B0%5D.rtf&suffix=x&label=RTF&color=green)
```
