#!/bin/bash
# Monitor all 4 benchmark processes and finalize results when all complete.

RESULTS=(
  "/tmp/results_full_gigastt.json"
  "/tmp/results_full_whisper.json"
  "/tmp/results_full_faster.json"
  "/tmp/results_full_vosk.json"
)
LOG="/tmp/bench_monitor.log"
REPORT="/tmp/bench_final_report.txt"

echo "[$(date)] Monitor started" >> "$LOG"

# Wait for all result files to exist
while true; do
  all_done=true
  for r in "${RESULTS[@]}"; do
    if [ ! -f "$r" ]; then
      all_done=false
      break
    fi
  done
  if $all_done; then
    echo "[$(date)] All results ready" >> "$LOG"
    break
  fi
  
  # Log current progress
  echo "[$(date)] Progress:" >> "$LOG"
  for f in gigastt whisper faster vosk; do
    tail -1 "/tmp/bench_full_${f}.log" 2>/dev/null >> "$LOG"
  done
  echo "" >> "$LOG"
  
  sleep 300
done

# Build final results
cd /Users/ekhodzitsky/Documents/personal/gigastt
python3 << 'PY'
import json, os
results = {"manifest_samples": 0, "runners": []}
files = [
    ("/tmp/results_full_gigastt.json", "gigastt"),
    ("/tmp/results_full_whisper.json", "whisper.cpp"),
    ("/tmp/results_full_faster.json", "faster-whisper"),
    ("/tmp/results_full_vosk.json", "vosk"),
]
for path, name in files:
    try:
        d = json.load(open(path))
        for r in d.get("runners", []):
            results["runners"].append(r)
        results["manifest_samples"] = max(results["manifest_samples"], d.get("manifest_samples", 0))
    except Exception as e:
        print(f"Skip {name}: {e}")

with open("results.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

report = []
report.append("=" * 90)
report.append(f"{'Engine':<20} {'Samples':>8} {'WER %':>8} {'RTF':>8} {'Errors':>10} {'Words':>10}")
report.append("-" * 90)
for r in results["runners"]:
    report.append(
        f"{r['name']:<20} {r['samples']:>8} {r['wer']:>8.2f} {r['rtf']:>8.3f} {r['total_errors']:>10} {r['total_ref_words']:>10}"
    )
report.append("=" * 90)

with open("/tmp/bench_final_report.txt", "w") as f:
    f.write("\n".join(report) + "\n")

print("\n".join(report))
PY

# Commit to benchmark-results branch
git checkout benchmark-results-local
git add results.json
git commit -m "benchmark: full 9994-sample cross-ASR results ($(date +%Y-%m-%d))"
git checkout main

echo "[$(date)] Finalization complete" >> "$LOG"
echo "[$(date)] Report saved to $REPORT" >> "$LOG"
