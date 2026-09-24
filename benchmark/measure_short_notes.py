#!/usr/bin/env python3
"""Measure short-note REST latency on a two-CPU desktop substitute.

The source clips are the labelled Golos fixtures already in the repo
(short Russian voice commands, not the reporter's Telegram notes).
Telegram-style Ogg/Opus and browser-style WebM/Opus are encoded locally.
The server is pinned to two logical CPUs with taskset. That is not the
two-vCPU Xeon VM from the report.

Writes benchmark/short_notes/measurements.json and corpus.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))

from common import compute_wer, compute_wer_naive  # noqa: E402

FIXTURE_DIR = ROOT / "crates/gigastt/tests/fixtures"
MANIFEST = FIXTURE_DIR / "manifest.json"
OUT_DIR = ROOT / "benchmark/short_notes"
ENCODED = OUT_DIR / "encoded"
BIN = ROOT / "target/release/gigastt"
PORT = 9891
CPUS = "0,1"
N_COLD = 6
N_WARM_ROUNDS = 8
READY_TIMEOUT_S = 180


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wav_duration_s(path: Path) -> float:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(probe.stdout.strip())


def encode_corpus(items: list[dict]) -> None:
    ENCODED.mkdir(parents=True, exist_ok=True)
    for item in items:
        wav = FIXTURE_DIR / item["filename"]
        stem = Path(item["filename"]).stem
        ogg = ENCODED / f"{stem}.telegram.ogg"
        webm = ENCODED / f"{stem}.browser.webm"
        if not ogg.exists():
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(wav),
                    "-ar",
                    "48000",
                    "-ac",
                    "1",
                    "-c:a",
                    "libopus",
                    "-application",
                    "voip",
                    "-b:a",
                    "32k",
                    str(ogg),
                ],
                check=True,
            )
        if not webm.exists():
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(wav),
                    "-ar",
                    "48000",
                    "-ac",
                    "1",
                    "-c:a",
                    "libopus",
                    "-frame_duration",
                    "60",
                    "-f",
                    "webm",
                    "-live",
                    "1",
                    str(webm),
                ],
                check=True,
            )
        seconds = wav_duration_s(wav)
        item["files"] = {
            "wav": file_record(wav, seconds),
            "ogg_opus_voip": file_record(ogg, seconds),
            "webm_opus_live": file_record(webm, seconds),
        }


def file_record(path: Path, seconds: float) -> dict:
    # Duration is the source WAV length. A live WebM has no container duration.
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "duration_s": round(seconds, 6),
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise RuntimeError("empty sample")
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * fraction
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    weight = index - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def summarize(values: list[float]) -> dict:
    return {
        "n": len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def wait_ready(proc: subprocess.Popen[bytes], started: float) -> float:
    url = f"http://127.0.0.1:{PORT}/ready"
    deadline = time.perf_counter() + READY_TIMEOUT_S
    while time.perf_counter() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited {proc.returncode} before ready")
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:
                if response.status == 200:
                    return time.perf_counter() - started
        except (urllib.error.URLError, TimeoutError, socket.timeout):
            time.sleep(0.05)
    raise RuntimeError("server did not become ready")


def transcribe(path: Path) -> tuple[float, str]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/transcribe",
        data=path.read_bytes(),
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        status = response.status
        body = json.loads(response.read().decode())
    elapsed = time.perf_counter() - started
    if status != 200:
        raise RuntimeError(f"transcribe status {status}")
    return elapsed, str(body.get("text", ""))


def start_server(extra: list[str], log_path: Path) -> tuple[subprocess.Popen[bytes], float]:
    log_handle = log_path.open("ab")
    command = [
        "taskset",
        "-c",
        CPUS,
        str(BIN),
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--pool-size",
        "1",
        "--model-variant",
        "rnnt",
        *extra,
    ]
    started = time.perf_counter()
    proc = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        ready_s = wait_ready(proc, started)
    except Exception:
        stop_server(proc)
        raise
    return proc, ready_s


def stop_server(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def corpus_wer(rows: list[tuple[str, str]]) -> dict:
    errors = naive_errors = ref_words = naive_words = 0
    for reference, hypothesis in rows:
        _, err, count = compute_wer(reference, hypothesis)
        _, naive_err, naive_count = compute_wer_naive(reference, hypothesis)
        errors += err
        naive_errors += naive_err
        ref_words += count
        naive_words += naive_count
    return {
        "n_clips": len(rows),
        "normalized_wer_pct": (100.0 * errors / ref_words) if ref_words else None,
        "normalized_errors": errors,
        "normalized_ref_words": ref_words,
        "naive_wer_pct": (100.0 * naive_errors / naive_words) if naive_words else None,
        "naive_errors": naive_errors,
        "naive_ref_words": naive_words,
    }


def measure_config(name: str, extra: list[str], items: list[dict]) -> dict:
    log_path = OUT_DIR / f"{name}.log"
    log_path.write_bytes(b"")
    cold_ready: list[float] = []
    first_request: list[float] = []
    first_file = FIXTURE_DIR / items[0]["filename"]
    for _ in range(N_COLD):
        proc, ready_s = start_server(extra, log_path)
        try:
            elapsed, _text = transcribe(first_file)
        finally:
            stop_server(proc)
        cold_ready.append(ready_s)
        first_request.append(elapsed)

    proc, ready_s = start_server(extra, log_path)
    try:
        transcribe(first_file)  # discard: not part of the warm sample
        warm: dict[str, list[float]] = {"wav": [], "ogg_opus_voip": [], "webm_opus_live": []}
        requests: list[dict] = []
        hypotheses: list[tuple[str, str]] = []
        examples: dict[str, str] = {}
        for round_index in range(N_WARM_ROUNDS):
            for item in items:
                for encoding, record in item["files"].items():
                    path = ROOT / record["path"]
                    elapsed, text = transcribe(path)
                    warm[encoding].append(elapsed)
                    requests.append(
                        {
                            "round": round_index,
                            "file": item["filename"],
                            "encoding": encoding,
                            "seconds": elapsed,
                            "duration_s": record["duration_s"],
                            "rtf": elapsed / record["duration_s"],
                        }
                    )
                    if round_index == 0 and encoding == "wav":
                        hypotheses.append((item["reference"], text))
                        if item["filename"] in {"golos_00.wav", "golos_06.wav"}:
                            examples[item["filename"]] = text
    finally:
        stop_server(proc)

    return {
        "name": name,
        "serve_args": extra,
        "cold_ready_s": summarize(cold_ready),
        "cold_ready_samples_s": cold_ready,
        "first_request_s": summarize(first_request),
        "first_request_samples_s": first_request,
        "warm_request_s": {key: summarize(values) for key, values in warm.items()},
        "warm_rtf": {
            key: summarize([row["rtf"] for row in requests if row["encoding"] == key])
            for key in warm
        },
        "warm_requests": requests,
        "quality_wav": corpus_wer(hypotheses),
        "examples_wav": examples,
        "resident_ready_s": ready_s,
    }


def host_facts() -> dict:
    version = subprocess.run([str(BIN), "--version"], check=True, capture_output=True, text=True)
    cpu = Path("/proc/cpuinfo").read_text(errors="replace")
    model = next(
        (line.split(":", 1)[1].strip() for line in cpu.splitlines() if line.startswith("model name")),
        "unknown",
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "binary": str(BIN),
        "version": version.stdout.strip(),
        "commit": commit,
        "cpu_model": model,
        "logical_cpus_on_host": os.cpu_count(),
        "taskset_cpus": CPUS,
        "constraint": (
            "Desktop substitute. The process is pinned with taskset to two "
            "logical CPUs on this machine. It is not the reporter's two-vCPU Xeon VM."
        ),
        "port": PORT,
        "pool_size": 1,
        "model_variant": "rnnt",
        "n_cold": N_COLD,
        "n_warm_rounds": N_WARM_ROUNDS,
        "loadavg_start": Path("/proc/loadavg").read_text().strip(),
    }


def main() -> None:
    if not BIN.is_file():
        raise SystemExit(f"missing release binary: {BIN}")
    items = json.loads(MANIFEST.read_text())
    encode_corpus(items)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "corpus.json").write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n")
    results = {
        "host": host_facts(),
        "configs": [
            measure_config("punct_itn_auto", ["--punctuation", "auto", "--itn", "auto"], items),
            measure_config("punct_itn_off", ["--punctuation", "off", "--itn", "off"], items),
        ],
    }
    results["host"]["loadavg_end"] = Path("/proc/loadavg").read_text().strip()
    (OUT_DIR / "measurements.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    auto = results["configs"][0]
    off = results["configs"][1]
    print("cold ready auto", auto["cold_ready_s"])
    print("first auto", auto["first_request_s"])
    print("warm wav auto", auto["warm_request_s"]["wav"])
    print("warm ogg auto", auto["warm_request_s"]["ogg_opus_voip"])
    print("warm webm auto", auto["warm_request_s"]["webm_opus_live"])
    print("quality auto", auto["quality_wav"])
    print("cold ready off", off["cold_ready_s"])
    print("warm wav off", off["warm_request_s"]["wav"])
    print("quality off", off["quality_wav"])
    print("examples", auto["examples_wav"], off["examples_wav"])


if __name__ == "__main__":
    main()
