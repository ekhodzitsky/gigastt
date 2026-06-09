#!/bin/bash
# Sequential benchmark monitor for long-running ASR benchmarks
# Usage: ./monitor.sh &

LOG="/tmp/bench_sequential.log"
FASTER_LOG="/tmp/bench_full_faster.log"
WHISPER_LOG="/tmp/bench_full_whisper.log"
INTERVAL=600  # 10 minutes

echo "[$(date -Iseconds)] Monitor started" >> "$LOG"

while true; do
    # Check which engine is currently running
    FASTER_PID=$(pgrep -f "runners faster-whisper" || true)
    WHISPER_PID=$(pgrep -f "runners whisper_cpp" || true)

    if [ -n "$FASTER_PID" ]; then
        PROGRESS=$(tail -1 "$FASTER_LOG" 2>/dev/null | grep -o '\[[0-9]*/9994\]' || echo "[?/9994]")
        echo "[$(date -Iseconds)] faster-whisper running $PROGRESS" >> "$LOG"
    elif [ -n "$WHISPER_PID" ]; then
        PROGRESS=$(tail -1 "$WHISPER_LOG" 2>/dev/null | grep -o '\[[0-9]*/9994\]' || echo "[?/9994]")
        echo "[$(date -Iseconds)] whisper.cpp running $PROGRESS" >> "$LOG"
    else
        # Neither running — check if results exist
        if [ -f "/tmp/results_full_faster.json" ] && [ -f "/tmp/results_full_whisper.json" ]; then
            echo "[$(date -Iseconds)] All engines complete!" >> "$LOG"
            # Merge all results
            cd "$(dirname "$0")"
            python3 << 'PYEOF'
import json, sys
from datetime import datetime, timezone

files = {
    'vosk': '/tmp/results_full_vosk.json',
    'gigastt': '/tmp/results_full_gigastt.json',
    'faster-whisper': '/tmp/results_full_faster.json',
    'whisper_cpp': '/tmp/results_full_whisper.json',
}

runners = []
for name, path in files.items():
    try:
        with open(path) as f:
            data = json.load(f)
        runners.extend(data.get('runners', []))
    except Exception as e:
        print(f"Skipping {name}: {e}", file=sys.stderr)

merged = {
    "manifest_samples": 9994,
    "runners": runners,
    "merged_at": datetime.now(timezone.utc).isoformat(),
}

out = '/Users/ekhodzitsky/Documents/personal/gigastt/benchmark/results.json'
with open(out, 'w') as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print(f"Merged {len(runners)} runners into {out}")
for r in runners:
    print(f"  {r['name']}: WER={r['wer']}% RTF={r['rtf']}x")
PYEOF
            break
        fi
    fi

    sleep $INTERVAL
done
