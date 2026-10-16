#!/usr/bin/env python3
"""Fail-closed checks for the AAC acceptance harness."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("aac_compare", Path(__file__).with_name("compare-aac-decoders.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def measurements(ratio=1.0):
    results = {}
    for file, mode in module.CASES:
        count = 0 if mode == "scan" else 2 if mode == "split" else 1
        sample = {"seconds": [1.0] * 7, "peak_rss_kib": 1234,
                  "planes": [{"finite": True, "samples": 64000, "energy": 10.0}] * count,
                  "detail": {"channels": 2, "dual_mono": True} if mode == "scan" else None}
        candidate = copy.deepcopy(sample)
        candidate["seconds"] = [ratio] * 7
        results[f"{file}:{mode}"] = {"base": [sample] * 7, "candidate": [candidate] * 7}
    return results


class AcceptanceTests(unittest.TestCase):
    def test_latency_regression_rejected(self):
        self.assertTrue(all(r["latency_pass"] for r in module.compare(measurements()).values()))
        self.assertFalse(any(r["latency_pass"] for r in module.compare(measurements(1.20)).values()))

    def test_missing_case_or_round_rejected(self):
        data = measurements()
        data.pop(next(iter(data)))
        with self.assertRaises(ValueError):
            module.compare(data)
        data = measurements()
        next(iter(data.values()))["base"].pop()
        with self.assertRaises(ValueError):
            module.compare(data)

    def test_bad_measurement_rejected(self):
        for field, value in [("seconds", [float("nan")] * 7), ("peak_rss_kib", None),
                             ("planes", []), ("planes", [{"finite": True, "samples": 64000, "energy": 0}]),
                             ("error", "unsupported codec")]:
            data = measurements()
            next(iter(data.values()))["candidate"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                module.compare(data)


if __name__ == "__main__":
    unittest.main()
