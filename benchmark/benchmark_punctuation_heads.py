#!/usr/bin/env python3
"""Legacy position-based comparison of two readable-Russian paths.

For alignment-aware scoring and the native rupunct comparison, use
``scripts/compare-punctuation-heads.py`` and
``docs/punctuation-heads-evaluation.md``. These legacy scores must not be used
to rank heads with different recognition errors.

Historical paths:

  1. ``e2e_rnnt``            — punctuation/casing/ITN baked into the head (one pass)
  2. ``rnnt`` + RuPunct      — low-WER bare head, punctuation restored as a second pass
                               (``--punctuation on --itn on``)

Reports position-based punctuation / capitalization F1 using the same primitives as
``benchmark_punctuation.py``. This backs the "Punctuation quality" table in
``docs/benchmarks.md``; it does not isolate punctuation accuracy from ASR errors.

Data — a punctuated reference manifest, produced by::

    python scripts/prepare_fleurs.py --config ru_ru --field raw_transcription

which writes ``benchmark/manifests/fleurs_ru_punct.json`` + the WAVs under
``~/.gigastt/benchmarks/fleurs_ru_punct/``. FLEURS Russian ``raw_transcription`` writes
numbers as digits. Number-word versus digit differences can shift every downstream
punctuation position in this legacy metric, independently of punctuation accuracy.

Caveat: the F1 metric is position-based, so it conflates recognition errors with
punctuation placement. Misrecognized words can shift downstream positions for either
head. This does not establish a mathematical lower bound on either head's lead.

This legacy harness uses separate model directories from the environment. Current
Engine explicit-variant loading also supports a directory containing both heads::

    GIGASTT_RNNT_DIR   dir holding the ``rnnt`` head     (default ~/.gigastt/models)
    GIGASTT_E2E_DIR    dir holding the ``e2e_rnnt`` head (default ~/.gigastt/models-e2e)

Usage::

    python benchmark/benchmark_punctuation_heads.py [manifest.json] [max_samples]
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "benchmark"))
from benchmark_punctuation import (  # noqa: E402
    aggregate,
    extract_capitalization,
    extract_punctuation,
    f1_score,
)

BIN = str(REPO / "target/release/gigastt")
RNNT_DIR = os.environ.get("GIGASTT_RNNT_DIR", str(Path("~/.gigastt/models").expanduser()))
E2E_DIR = os.environ.get("GIGASTT_E2E_DIR", str(Path("~/.gigastt/models-e2e").expanduser()))

# (label, model_dir, extra serve args, port). Both emit digits (--itn on) to match the
# digit-bearing FLEURS raw_transcription references.
CONFIGS = [
    ("e2e_rnnt", E2E_DIR, [], 9902),
    ("rnnt+RuPunct", RNNT_DIR, ["--punctuation", "on", "--itn", "on"], 9901),
]


def start(port: int, model_dir: str, extra: list[str]) -> subprocess.Popen:
    proc = subprocess.Popen(
        [BIN, "serve", "--port", str(port), "--model-dir", model_dir, *extra],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "RUST_LOG": "error"},
    )
    for _ in range(120):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/ready", timeout=1) as r:
                if r.status == 200:
                    return proc
        except Exception:
            pass
        time.sleep(0.5)
    proc.kill()
    raise RuntimeError(f"server on port {port} failed to become ready")


def transcribe(port: int, wav: Path) -> str:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/transcribe",
        data=wav.read_bytes(),
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode()).get("text", "").strip()


def main() -> int:
    manifest_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        REPO / "benchmark/manifests/fleurs_ru_punct.json"
    )
    man = json.loads(manifest_path.read_text())
    root = Path(man["audio_root"]).expanduser()
    samples = man["samples"]
    if len(sys.argv) > 2:
        samples = samples[: int(sys.argv[2])]
    print(f"{len(samples)} punctuated references from {man['dataset']}")

    results = {}
    for name, model_dir, extra, port in CONFIGS:
        print(f"\n=== {name} (port {port}, {model_dir}) ===")
        proc = start(port, model_dir, extra)
        try:
            pf, cf = [], []
            for i, s in enumerate(samples):
                hyp = transcribe(port, root / s["filename"])
                ref = s["reference"]
                pf.append(f1_score(extract_punctuation(ref), extract_punctuation(hyp)))
                cf.append(f1_score(extract_capitalization(ref), extract_capitalization(hyp)))
                if (i + 1) % 100 == 0:
                    print(f"  {i + 1}/{len(samples)}")
            results[name] = {"punct": aggregate(pf), "cap": aggregate(cf)}
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()

    print("\n================ RESULTS ================")
    for name, r in results.items():
        print(
            f"{name:14s} punct F1={r['punct']['f1']:.3f} "
            f"(P={r['punct']['precision']:.3f} R={r['punct']['recall']:.3f})  "
            f"cap F1={r['cap']['f1']:.3f} "
            f"(P={r['cap']['precision']:.3f} R={r['cap']['recall']:.3f})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
