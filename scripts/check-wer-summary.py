#!/usr/bin/env python3
"""Assert that the hosted rounded WER threshold received full measured evidence."""
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent


def check(data, threshold, expected_samples):
    if not math.isfinite(threshold) or threshold < 0 or expected_samples < 1:
        raise ValueError("invalid hosted threshold or expected sample count")
    if (
        data.get("pass") is not True
        or data.get("status") != "measured"
        or data.get("skipped") is not False
        or data.get("set") != "bundled"
        or type(data.get("samples")) is not int
        or data["samples"] != expected_samples
    ):
        raise ValueError("expected complete, successful bundled measurements")
    for key in (
        "wer", "rtf", "rss_after_load_mb", "rss_after_decode_mb",
        "cold_start_ms", "total_words", "audio_s",
    ):
        value = data.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid measured {key}")
        if key in ("total_words", "audio_s") and value == 0:
            raise ValueError(f"zero measured {key}")
    if data["wer"] > threshold:
        raise ValueError(
            f"WER {data['wer']}% exceeds .github/wer-threshold.txt {threshold}%"
        )


def self_test():
    full = dict(
        pass_=True, status="measured", skipped=False, set="bundled", samples=15,
        wer=13.0, rtf=0.5, rss_after_load_mb=600, rss_after_decode_mb=600,
        cold_start_ms=1000, total_words=75, audio_s=45,
    )
    full["pass"] = full.pop("pass_")
    check(full, 13.0, 15)
    for change in (
        {"pass": False}, {"status": "skipped"}, {"skipped": True},
        {"set": "external"}, {"samples": 0}, {"samples": 14}, {"wer": 13.1},
        {"wer": None}, {"rtf": float("nan")}, {"total_words": 0}, {"audio_s": 0},
    ):
        try:
            check(full | change, 13.0, 15)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid result {change}")
    print("WER summary self-test: ok")


def main():
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        return
    threshold = float((ROOT / ".github/wer-threshold.txt").read_text().strip())
    baseline = json.loads(
        (ROOT / "crates/gigastt/tests/benchmark_baseline.json").read_text()
    )
    data = json.loads(sys.argv[1])
    check(data, threshold, baseline["sets"]["bundled"]["samples"])
    print(
        f"PASS: WER={data['wer']}% RTF={data['rtf']} "
        f"RSS_load={data['rss_after_load_mb']}MB "
        f"RSS_decode={data['rss_after_decode_mb']}MB "
        f"cold_start={data['cold_start_ms']}ms samples={data['samples']}"
    )


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        sys.exit(str(error))
