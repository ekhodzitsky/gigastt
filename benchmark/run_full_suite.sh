#!/bin/bash
# Run the full cross-ASR benchmark across all committed 1000-sample datasets.
# This is intentionally sequential: engines must not compete for CPU.
set -euo pipefail

DATASETS=(golos_crowd_1k golos_farfield openstt_calls openstt_youtube)
RUNNERS=(gigastt whisper_cpp faster_whisper vosk)
RESULTS_DIR="${BENCHMARK_RESULTS_DIR:-$(pwd)/results_local}"
LOG_DIR="${BENCHMARK_LOG_DIR:-$RESULTS_DIR/logs}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

for dataset in "${DATASETS[@]}"; do
  for runner in "${RUNNERS[@]}"; do
    out="$RESULTS_DIR/${dataset}_${runner}.json"
    log="$LOG_DIR/${dataset}_${runner}.log"
    echo "[$(date -Iseconds)] Starting $dataset / $runner"
    "$(dirname "$0")/.venv/bin/python" benchmark.py \
      --dataset "$dataset" \
      --runners "$runner" \
      --max-samples 0 \
      --output "$out" \
      > "$log" 2>&1
    echo "[$(date -Iseconds)] Finished $dataset / $runner -> $out"
  done
done

echo "[$(date -Iseconds)] All runs complete. Results in $RESULTS_DIR"
