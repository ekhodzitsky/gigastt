#!/usr/bin/env python3
"""Local, paired production-path AAC measurements; never dispatches CI.

Build the identical aac_probe example at both revisions first. See docs/aac-audit.md.
A >5% median paired latency increase fails acceptance, including opt-in builds.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess

CASES = [("speech0.m4a", "file"), ("speech0.m4a", "bytes"),
         ("lc.adts", "file"), ("long-stereo.m4a", "file"),
         ("long-stereo.m4a", "split"), ("long-stereo.m4a", "scan")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root, destination):
    destination.mkdir(parents=True, exist_ok=True)
    source = root / "crates/gigastt/tests/fixtures/golos_00.wav"
    commands = []

    def ff(*args):
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *map(str, args)]
        subprocess.run(command, check=True)
        commands.append(command)

    ff("-i", source, "-ar", 48000, "-ac", 1, "-c:a", "aac", "-b:a", "64k", destination / "speech0.m4a")
    ff("-i", destination / "speech0.m4a", "-c:a", "copy", "-f", "adts", destination / "lc.adts")
    ff("-stream_loop", 19, "-i", source, "-ar", 48000, "-ac", 2, "-c:a", "aac", "-b:a", "128k", destination / "long-stereo.m4a")
    (destination / "generation.json").write_text(json.dumps({
        "source_sha256": sha(source), "commands": commands,
        "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True),
    }, indent=2) + "\n")


def validate(sample):
    if "error" in sample:
        raise ValueError(f"decoder failed: {sample['error']}")
    times = sample["seconds"]
    if len(times) != 7 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                              or not math.isfinite(x) or x <= 0 for x in times):
        raise ValueError("require seven finite positive timings")
    rss = sample["peak_rss_kib"]
    if isinstance(rss, bool) or not isinstance(rss, int) or rss <= 0:
        raise ValueError("Linux peak RSS is required")
    for plane in sample["planes"]:
        energy = plane["energy"]
        if (not plane["finite"] or plane["samples"] <= 0 or isinstance(energy, bool)
                or not isinstance(energy, (int, float)) or not math.isfinite(energy) or energy <= 0):
            raise ValueError("invalid or silent decoded speech PCM")


def compare(results):
    if set(results) != {f"{file}:{mode}" for file, mode in CASES}:
        raise ValueError("missing or unexpected benchmark cases")
    summary = {}
    for case, pairs in results.items():
        if len(pairs["base"]) != 7 or len(pairs["candidate"]) != 7:
            raise ValueError("require seven paired processes")
        for sample in pairs["base"] + pairs["candidate"]:
            validate(sample)
            mode = case.split(":")[1]
            expected = 0 if mode == "scan" else 2 if mode == "split" else 1
            if len(sample["planes"]) != expected:
                raise ValueError("missing decoded channels")
            if mode == "scan" and sample["detail"] != {"channels": 2, "dual_mono": True}:
                raise ValueError("channel decision changed")
        ratio = statistics.median(statistics.median(c["seconds"]) / statistics.median(b["seconds"])
                                  for b, c in zip(pairs["base"], pairs["candidate"]))
        summary[case] = {"median_paired_time_ratio": ratio, "latency_pass": ratio <= 1.05,
                         "median_peak_rss_kib": {label: statistics.median(s["peak_rss_kib"] for s in pairs[label])
                                                 for label in ["base", "candidate"]}}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-probe", required=True, type=Path)
    parser.add_argument("--candidate-probe", required=True, type=Path)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prepare", action="store_true", help="generate the three timed fixtures with FFmpeg")
    args = parser.parse_args()
    if args.prepare:
        prepare(Path(__file__).resolve().parents[1], args.fixtures)
    binaries = {"base": args.base_probe.resolve(), "candidate": args.candidate_probe.resolve()}
    report = {"platform": platform.platform(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "binary_sha256": {label: sha(path) for label, path in binaries.items()},
              "fixture_sha256": {name: sha(args.fixtures / name) for name, _ in CASES},
              "protocol": "seven alternating pairs; two warmups and seven timed decodes per process",
              "results": {}}
    for file, mode in CASES:
        pairs = {"base": [], "candidate": []}
        for round_index in range(7):
            order = ["base", "candidate"] if round_index % 2 == 0 else ["candidate", "base"]
            for label in order:
                sample = json.loads(subprocess.check_output([str(binaries[label]), mode,
                                    str((args.fixtures / file).resolve()), "7"], text=True))
                validate(sample)
                pairs[label].append(sample)
        report["results"][f"{file}:{mode}"] = pairs
    report["summary"] = compare(report["results"])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for case, result in report["summary"].items():
        print(f"{case}: {result['median_paired_time_ratio']:.3f}x ({'PASS' if result['latency_pass'] else 'FAIL'})")
    return 0 if all(r["latency_pass"] for r in report["summary"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
