#!/usr/bin/env python3
"""Rescore our results and archived baselines with the pinned upstream scorer.

This does not call commercial APIs or rerun Whisper. Baseline hypotheses are
those published by the dataset author. Run after benchmark_multilingual_public.py.
"""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import random

from prepare_multilingual_public import KAZAKH_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != KAZAKH_REVISION:
        raise ValueError("Unexpected upstream scorer revision")
    destination = args.results.resolve() / "upstream_scoring"
    destination.mkdir(parents=True, exist_ok=True)
    predictions = []
    for path in sorted(args.results.glob("kazakh_ml_ctc*.json")):
        result = json.loads(path.read_text())
        if "summary" not in result:
            raise ValueError(f"Incomplete run: {path}")
        for condition in sorted({r["condition"] for r in result["details"]}):
            name = result["health"]["variant"] + "_" + condition
            output = destination / f"predictions_{name}.jsonl"
            rows = [{"audio_id": r["id"], "raw_output": r["hypothesis"], "service": name}
                    for r in result["details"] if r["condition"] == condition]
            output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            predictions.append(str(output))
    baselines = sorted((source / "results").glob("predictions_*.jsonl"))
    if not predictions or not baselines:
        raise ValueError("Missing measured or archived predictions")
    subprocess.run([sys.executable, str(source / "evaluate.py"), "--metadata", str(source / "metadata.csv"),
                    "--predictions", *predictions, *(str(p) for p in baselines),
                    "--normalize-numbers", "--outdir", str(destination)], check=True)
    def rates(name):
        with (destination / f"scores_{name}.csv").open() as stream:
            return {r["audio_id"]: float(r["wer_best"]) for r in csv.DictReader(stream)
                    if r["excluded"] == "False"}
    ours = rates("ml_ctc_original")
    baseline = rates("hf_shyngys879_kazakh-whisper-large-v3-turbo")
    if ours.keys() != baseline.keys():
        raise ValueError("Paired comparison requires identical clip IDs")
    ids = sorted(ours)
    rng = random.Random(42)
    differences = sorted(100 * sum(ours[k] - baseline[k] for k in rng.choices(ids, k=len(ids))) / len(ids)
                         for _ in range(10000))
    bootstrap = {"metric": "paired clip-bootstrap difference in macro WER_best, percentage points; negative favors gigastt",
                 "comparison": "ml_ctc_original minus archived Kazakh Whisper fine-tune", "n": len(ids),
                 "seed": 42, "resamples": 10000,
                 "mean_difference_pp": 100 * sum(ours[k] - baseline[k] for k in ids) / len(ids),
                 "percentile_95_ci_pp": [differences[250], differences[9750]],
                 "caveat": "Clip-level resampling ignores correlation between clips from the same sources and speakers; not a population-level guarantee."}
    (destination / "paired_bootstrap.json").write_text(json.dumps(bootstrap, indent=2) + "\n")


if __name__ == "__main__":
    main()
