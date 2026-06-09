#!/bin/bash
set -e

BENCH_DIR="/Users/ekhodzitsky/Documents/personal/gigastt/benchmark"
PYTHON="$BENCH_DIR/.venv/bin/python"
BENCHMARK="$BENCH_DIR/benchmark.py"
LOG="/tmp/bench_sequential.log"

cd "$BENCH_DIR"

echo "[$(date)] Starting faster-whisper..." >> "$LOG"
$PYTHON -u "$BENCHMARK" --max-samples 0 --runners faster-whisper --output /tmp/results_full_faster.json >> /tmp/bench_full_faster.log 2>&1
echo "[$(date)] faster-whisper done." >> "$LOG"

echo "[$(date)] Starting whisper.cpp..." >> "$LOG"
$PYTHON -u "$BENCHMARK" --max-samples 0 --runners whisper_cpp --output /tmp/results_full_whisper.json >> /tmp/bench_full_whisper.log 2>&1
echo "[$(date)] whisper.cpp done." >> "$LOG"
