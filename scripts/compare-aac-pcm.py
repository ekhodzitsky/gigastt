#!/usr/bin/env python3
"""Compare exact production-adapter PCM from two aac_probe release binaries.

Manifest: [{"id": "speech", "path": "/local/input.m4a", "mode": "file"}].
Only hashes and geometry are retained; temporary PCM is removed on exit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["base-probe", "candidate-probe", "manifest", "output"]:
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    binaries = {"base": args.base_probe.resolve(), "candidate": args.candidate_probe.resolve()}
    cases = json.loads(args.manifest.read_text())
    if not cases or len({c["id"] for c in cases}) != len(cases):
        parser.error("require nonempty manifest with unique IDs")
    report = {"binary_sha256": {n: sha(p) for n, p in binaries.items()},
              "runner_sha256": sha(Path(__file__)), "cases": [], "pass": False}
    with tempfile.TemporaryDirectory(prefix="aac-pcm-") as directory:
        for index, case in enumerate(cases):
            row = {"id": case["id"], "input_sha256": sha(Path(case["path"])),
                   "mode": case.get("mode", "file"), "outputs": {}, "pass": False}
            report["cases"].append(row)
            try:
                if row["mode"] not in ("file", "bytes", "split"):
                    raise ValueError("mode must produce PCM")
                for name, binary in binaries.items():
                    prefix = Path(directory) / f"{index}-{name}"
                    result = json.loads(subprocess.check_output(
                        [str(binary), row["mode"], str(Path(case["path"]).resolve()),
                         "1", str(prefix)], text=True, timeout=90))
                    if "error" in result:
                        raise ValueError(result["error"])
                    planes = result["planes"]
                    if not planes or any(not p["finite"] or p["samples"] <= 0 for p in planes):
                        raise ValueError("empty or nonfinite PCM")
                    output = []
                    for channel, plane in enumerate(planes):
                        pcm = Path(f"{prefix}.{channel}.f32")
                        if pcm.stat().st_size != plane["samples"] * 4:
                            raise ValueError("PCM dump length does not match probe geometry")
                        output.append({"samples": plane["samples"], "sha256": sha(pcm)})
                        pcm.unlink()
                    row["outputs"][name] = output
                row["pass"] = row["outputs"]["base"] == row["outputs"]["candidate"]
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                row["error"] = str(error)
            print(row["id"], row["pass"], flush=True)
    report["pass"] = all(c["pass"] for c in report["cases"])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return int(not report["pass"])


if __name__ == "__main__":
    raise SystemExit(main())
