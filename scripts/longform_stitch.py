#!/usr/bin/env python3
"""Prepare and evaluate the small, frozen long-form stitching corpus.

Run `run` inside a separate systemd unit with MemoryMax/MemorySwapMax and
RuntimeMaxSec (see docs/longform-stitching.md). Each CLI process is serial,
has a timeout, and writes its transcript and GNU time measurements to disk.
Only `prepare` accesses the network. Python standard library + GNU time.
"""

import argparse
from array import array
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import urllib.request
import wave


MANIFEST = Path(__file__).resolve().parents[1] / "benchmark/manifests/longform_stitch.json"


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def normalize(text):
    return re.findall(r"[^\W_]+", text.lower().replace("ё", "е"), re.UNICODE)


def edits(reference, hypothesis):
    """Levenshtein counts (substitution, deletion, insertion), stable tie-break."""
    row = [(0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, a in enumerate(reference, 1):
        nxt = [(0, i, 0)]
        for j, b in enumerate(hypothesis, 1):
            if a == b:
                nxt.append(row[j - 1])
                continue
            s, d, ins = row[j - 1]
            candidates = [(s + 1, d, ins)]
            s, d, ins = row[j]
            candidates.append((s, d + 1, ins))
            s, d, ins = nxt[j - 1]
            candidates.append((s, d, ins + 1))
            nxt.append(min(candidates, key=sum))
        row = nxt
    return row[-1]


def checked_audio(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"audio checksum mismatch: {path}")


def fetch(url, path):
    temporary = path.with_suffix(".partial")
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as out:
        while chunk := response.read(65536):
            out.write(chunk)
    temporary.replace(path)


def layout(sample):
    """(rate, channels) of a sample's WAV; 16 kHz mono unless the manifest says otherwise."""
    return sample.get("rate", 16000), sample.get("channels", 1)


def pcm(path, rate=16000, channels=1):
    with wave.open(str(path)) as source:
        assert (source.getnchannels(), source.getsampwidth(), source.getframerate()) == (channels, 2, rate)
        return source.readframes(source.getnframes())


def write_wav(path, data, rate=16000, channels=1):
    with wave.open(str(path), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(data)


# Hugging Face datasets-server row layouts for samples assembled from parts.
DATASETS = {
    "istupakov/russian_librispeech": {"split": "test", "text": "text"},
    "bond005/podlodka_speech": {"split": "train", "text": "transcription"},
}


def prepare(args, manifest):
    args.audio.mkdir(parents=True, exist_ok=True)
    parts_dir = args.audio / "ruls-parts"
    parts_dir.mkdir(exist_ok=True)
    for sample in manifest["samples"]:
        path = args.audio / sample["file"]
        if not path.exists():
            if "parts" not in sample:
                fetch(sample["source"], path)
            else:
                dataset = sample.get("dataset", "istupakov/russian_librispeech")
                fields = DATASETS[dataset]
                rate, channels = layout(sample)
                chunks = []
                for part in sample["parts"]:
                    part_path = parts_dir / Path(part.get("file", f"{sample['id']}-{part['row']:04d}.wav")).name
                    if not part_path.exists():
                        url = ("https://datasets-server.huggingface.co/rows?"
                               f"dataset={dataset}&config=default&split={fields['split']}"
                               f"&offset={part['row']}&length=1")
                        with urllib.request.urlopen(url, timeout=60) as response:
                            row = json.load(response)["rows"][0]["row"]
                        if "file" in part:
                            assert row["audio_filepath"] == part["file"]
                        if "episode" in sample:
                            assert row["episode"] == sample["episode"]
                        assert row[fields["text"]] == part["reference"]
                        fetch(row["audio"][0]["src"], part_path)
                    checked_audio(part_path, part["sha256"])
                    chunks.append(pcm(part_path, rate, channels))
                write_wav(path, b"".join(chunks), rate, channels)
        checked_audio(path, sample["sha256"])
        print(sample["id"], "verified", flush=True)


def cases(manifest):
    for sample in manifest["samples"]:
        modes = sample.get("modes", ["serial"])
        for head in ("rnnt", "e2e_rnnt", "ml_ctc"):
            for shift in (0, 3, 22):
                for mode in modes:
                    if mode != "short":
                        yield sample, head, shift, mode
            if "short" in modes:
                yield sample, head, 0, "short"


def require_limits():
    # Fail closed: workload limits must isolate these processes from the UI.
    group = Path("/proc/self/cgroup").read_text().strip().split("::")[-1]
    cgroup = Path("/sys/fs/cgroup") / group.lstrip("/")
    ceiling = (cgroup / "memory.max").read_text().strip()
    if not group.endswith(".service") or "app-orca" in group or ceiling == "max":
        raise RuntimeError("run in a separate systemd service with MemoryMax; see the guide")
    return group, cgroup, ceiling


def run(args, manifest):
    group, cgroup, ceiling = require_limits()
    args.output.mkdir(parents=True, exist_ok=False)
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GIGASTT_")}
    environment["RUST_LOG"] = "warn"
    metadata = {
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "platform": platform.platform(), "cgroup": group, "memory_max": ceiling,
        "memory_swap_max": (cgroup / "memory.swap.max").read_text().strip(),
        "cpu": Path("/proc/cpuinfo").read_text().split("model name", 1)[-1].splitlines()[0],
        "encoder_threads": 4, "case_timeout_seconds": 120,
        "models": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in args.models.glob("*") if p.suffix in (".onnx", ".txt")},
    }
    write_json(args.output / "metadata.json", metadata)
    results = []
    for sample, head, shift, mode in cases(manifest):
        if args.modes and mode not in args.modes:
            continue
        key = f"{sample['id']}-{head}-{shift}-{mode}"
        source = args.audio / sample["file"]
        checked_audio(source, sample["sha256"])
        rate, channels = layout(sample)
        frame_bytes = 2 * channels
        data = pcm(source, rate, channels)
        if mode == "short":
            data = data[:20 * rate * frame_bytes]
        shifted = args.output / "input.wav"
        write_wav(shifted, b"\0" * (shift * rate * frame_bytes) + data, rate, channels)
        destination = args.output / f"{key}.json"
        timing = args.output / f"{key}.time"
        command = ["/usr/bin/time", "-f", "%e %M", "-o", str(timing),
                   str(args.binary), "transcribe", str(shifted), "--model-dir", str(args.models),
                   "--model-variant", head, "--offline", "--punctuation", "off", "--itn", "off",
                   "--encoder-intra-threads", "4", "--file-window-concurrency",
                   "2" if mode == "parallel" else "1", "--format", "json", "--output", str(destination)]
        if mode == "vad":
            command.append("--vad")
        with (args.output / f"{key}.log").open("w") as log:
            subprocess.run(command, env=environment, stdout=log, stderr=log, check=True, timeout=120)
        transcript = read_json(destination)
        words = transcript["words"]
        assert all(a["start"] <= b["start"] for a, b in zip(words, words[1:])), key
        wall, rss = timing.read_text().split()
        row = {"id": key, "head": head, "sample": sample["id"], "shift": shift, "mode": mode,
               "wall_seconds": float(wall), "peak_rss_kib": int(rss),
               "audio_seconds": len(data) / (rate * frame_bytes) + shift, "text": transcript["text"]}
        if mode != "short":
            reference = normalize(sample["reference"])
            row["reference_words"] = len(reference)
            row["sdi"] = edits(reference, normalize(row["text"]))
        results.append(row)
        write_json(args.output / "results.json", results)
        print(key, row.get("sdi"), wall, rss, flush=True)
    shifted.unlink()


def pause(args, manifest):
    """Ablate midpoint vs quietest 100 ms region using identical window decodes."""
    require_limits()
    args.output.mkdir(parents=True, exist_ok=False)
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GIGASTT_")}
    environment["RUST_LOG"] = "warn"
    results = []
    for sample, head, shift, mode in cases(manifest):
        if mode != "serial" or layout(sample) != (16000, 1):
            continue
        key = f"{sample['id']}-{head}-{shift}"
        source = args.audio / sample["file"]
        checked_audio(source, sample["sha256"])
        data = b"\0" * (shift * 32000) + pcm(source)
        midpoint_words, quiet_words, windows = [], [], []
        for start in range(0, len(data) // 2, 22 * 16000):
            chunk = data[start * 2:(start + 24 * 16000) * 2]
            window_file = args.output / "window.wav"
            write_wav(window_file, chunk)
            output = args.output / "window.json"
            command = [str(args.binary), "transcribe", str(window_file), "--model-dir", str(args.models),
                       "--model-variant", head, "--offline", "--punctuation", "off", "--itn", "off",
                       "--encoder-intra-threads", "4", "--format", "json", "--output", str(output)]
            with (args.output / "window.log").open("w") as log:
                subprocess.run(command, env=environment, stdout=log, stderr=log, check=True, timeout=120)
            words = read_json(output)["words"]
            for word in words:
                word["start"] += start / 16000
                word["end"] += start / 16000
            seam = start / 16000 + 1.0
            audio = array("h", chunk[:2 * 32000])
            if sys.byteorder != "little":
                audio.byteswap()
            # Stay 250 ms away from either encoder edge, seek the quietest
            # 100 ms region, require RMS <= 0.01 (-40 dBFS). Otherwise use
            # the midpoint; an unvoiced consonant is not assumed to be silence.
            candidates = [(sum(x*x for x in audio[i:i+1600]) / 1600,
                           abs((i+800)/16000 - 1), (i+800)/16000)
                          for i in range(4000, min(len(audio), 32000) - 5600 + 1, 320)]
            quiet = min(candidates) if candidates else None
            quiet_seam = start / 16000 + quiet[2] if quiet and quiet[0] <= (32768 * 0.01)**2 else seam
            def merge(previous, boundary):
                if not previous:
                    return words.copy()
                return [w for w in previous if w["start"] <= boundary] + [w for w in words if w["start"] > boundary]
            midpoint_words = merge(midpoint_words, seam)
            quiet_words = merge(quiet_words, quiet_seam)
            windows.append({"start": start / 16000, "midpoint": seam, "quiet_seam": quiet_seam, "words": words})
            if start * 2 + len(chunk) >= len(data):
                break
        actual = read_json(args.baseline / f"{key}-serial.json")["words"]
        assert [w["word"] for w in actual] == [w["word"] for w in midpoint_words], key
        assert all(abs(a["start"] - b["start"]) < 1e-6 for a, b in zip(actual, midpoint_words)), key
        reference = normalize(sample["reference"])
        row = {"id": key, "head": head, "reference_words": len(reference),
               "midpoint_sdi": edits(reference, normalize(" ".join(w["word"] for w in midpoint_words))),
               "quiet_sdi": edits(reference, normalize(" ".join(w["word"] for w in quiet_words)))}
        write_json(args.output / f"{key}-windows.json", windows)
        results.append(row)
        write_json(args.output / "results.json", results)
        print(key, row["midpoint_sdi"], "->", row["quiet_sdi"], flush=True)


def compare(args):
    baseline = {r["id"]: r for r in read_json(args.baseline / "results.json")}
    candidate = {r["id"]: r for r in read_json(args.candidate / "results.json")}
    if args.subset:
        baseline = {k: v for k, v in baseline.items() if k in candidate}
    assert baseline.keys() == candidate.keys()
    assert read_json(args.baseline / "metadata.json")["manifest_sha256"] == read_json(args.candidate / "metadata.json")["manifest_sha256"]
    summary = {}
    for key, b in baseline.items():
        c = candidate[key]
        if b["mode"] == "short":
            assert read_json(args.baseline / f"{key}.json") == read_json(args.candidate / f"{key}.json"), key
            continue
        group = f"{b['head']}/{b['mode']}"
        s = summary.setdefault(group, {"reference_words": 0, "baseline_sdi": [0, 0, 0], "candidate_sdi": [0, 0, 0], "changed": []})
        s["reference_words"] += b["reference_words"]
        for i in range(3):
            s["baseline_sdi"][i] += b["sdi"][i]
            s["candidate_sdi"][i] += c["sdi"][i]
        if b["text"] != c["text"]:
            s["changed"].append({"id": key, "before": b["text"], "after": c["text"], "before_sdi": b["sdi"], "after_sdi": c["sdi"],
                                 "before_words": read_json(args.baseline / f"{key}.json")["words"],
                                 "after_words": read_json(args.candidate / f"{key}.json")["words"]})
    for s in summary.values():
        for variant in ("baseline", "candidate"):
            s[f"{variant}_wer"] = 100 * sum(s[f"{variant}_sdi"]) / s["reference_words"]
    write_json(args.candidate / "comparison.json", summary)
    resources, disagreement = {}, []
    for variant, rows in (("baseline", baseline), ("candidate", candidate)):
        resources[variant] = {"peak_rss_kib": max(r["peak_rss_kib"] for r in rows.values()),
                              "median_rtf": statistics.median(r["wall_seconds"] / r["audio_seconds"] for r in rows.values() if r["mode"] == "serial")}
        for row in rows.values():
            if row["mode"] == "short" or row["shift"] == 0:
                continue
            unshifted = rows[f"{row['sample']}-{row['head']}-0-{row['mode']}"]
            disagreement.append({"variant": variant, "id": row["id"],
                                 "sdi_vs_unshifted": edits(normalize(unshifted["text"]), normalize(row["text"]))})
    write_json(args.candidate / "resources.json", resources)
    write_json(args.candidate / "shift-disagreement.json", disagreement)
    for group, s in summary.items():
        print(group, s["baseline_sdi"], "->", s["candidate_sdi"], len(s["changed"]), "changed")
    if any(sum(s["candidate_sdi"]) > sum(s["baseline_sdi"]) for s in summary.values()):
        raise SystemExit("WER regression; inspect comparison.json")


def align(reference, hypothesis):
    """Levenshtein alignment as (op, i, j) with op in eq/sub/del/ins; same costs as `edits`."""
    n, m = len(reference), len(hypothesis)
    dist = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dist[i][0] = i
    for j in range(1, m + 1):
        dist[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dist[i][j] = min(dist[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]),
                             dist[i - 1][j] + 1, dist[i][j - 1] + 1)
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dist[i][j] == dist[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            ops.append(("eq" if reference[i - 1] == hypothesis[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and dist[i][j] == dist[i - 1][j] + 1:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    return ops[::-1]


def timed_tokens(words, shift):
    """Normalized tokens with original-audio times (a hypothesis word may split into several)."""
    return [{"tok": tok, "word": w["word"], "start": round(w["start"] - shift, 2),
             "end": round(w["end"] - shift, 2), "confidence": round(w["confidence"], 3)}
            for w in words for tok in normalize(w["word"])]


def residuals(args, manifest):
    """Adjudicate every shifted-vs-unshifted disagreement against the reference.

    Hypothesis disagreement is reported next to, not instead of, the reference
    verdict of each side. With `--windows` (a `pause` output directory) each
    item also lists the per-window copies decoded before stitching.
    """
    sample = next(s for s in manifest["samples"] if s["id"] == args.sample)
    reference = normalize(sample["reference"])
    report = {"sample": args.sample, "reference_words": len(reference), "heads": {}}
    for head in ("rnnt", "e2e_rnnt", "ml_ctc"):
        hyps = {}
        for shift in (0, 3, 22):
            words = read_json(args.run / f"{args.sample}-{head}-{shift}-serial.json")["words"]
            tokens = timed_tokens(words, shift)
            ops = align(reference, [t["tok"] for t in tokens])
            verdict = {j: (op, reference[i] if i is not None else None) for op, i, j in ops if j is not None}
            hyps[shift] = {"tokens": tokens, "verdict": verdict,
                           "sdi": [sum(1 for o in ops if o[0] == k) for k in ("sub", "del", "ins")]}
        entry = {"sdi_vs_reference": {s: h["sdi"] for s, h in hyps.items()}, "disagreements": {}}
        for shift in (3, 22):
            items = []
            for op, i, j in align([t["tok"] for t in hyps[0]["tokens"]], [t["tok"] for t in hyps[shift]["tokens"]]):
                if op == "eq":
                    continue
                item = {"op": op}
                for label, s, k in (("shift0", 0, i), ("shifted", shift, j)):
                    if k is not None:
                        item[label] = {**hyps[s]["tokens"][k], "vs_reference": hyps[s]["verdict"].get(k)}
                span = item.get("shift0") or item["shifted"]
                if args.windows:
                    copies = {}
                    for s in (0, shift):
                        for w in read_json(args.windows / f"{args.sample}-{head}-{s}-windows.json"):
                            start = w["start"] - s
                            if not start <= span["start"] <= start + 24:
                                continue
                            hit = [(x["word"], round(x["start"] - s, 2), round(x["end"] - s, 2), round(x["confidence"], 3))
                                   for x in w["words"] if x["start"] - s < span["end"] + 0.05 and x["end"] - s > span["start"] - 0.05]
                            copies.setdefault(str(s), []).append({"window": [round(start, 2), round(start + 24, 2)],
                                                                  "seam": round(w["midpoint"] - s, 2), "copy": hit})
                    item["window_copies"] = copies
                items.append(item)
            entry["disagreements"][str(shift)] = {"count": len(items), "items": items}
        report["heads"][head] = entry
        print(head, "S/D/I by shift", entry["sdi_vs_reference"])
        for shift, block in entry["disagreements"].items():
            for item in block["items"]:
                a, b = item.get("shift0", {}), item.get("shifted", {})
                print(f"  shift {shift}: {a.get('word', '-')} {a.get('vs_reference')} | {b.get('word', '-')} {b.get('vs_reference')}"
                      f" @ {a.get('start', b.get('start'))}s")
    write_json(args.output, report)


def performance(args, manifest):
    """Paired resource check on the official recording, alternating binary order."""
    require_limits()
    args.output.mkdir(parents=True, exist_ok=False)
    sample = manifest["samples"][0]
    source = args.audio / sample["file"]
    checked_audio(source, sample["sha256"])
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GIGASTT_")}
    environment["RUST_LOG"] = "warn"
    results = []
    for head in ("rnnt", "e2e_rnnt", "ml_ctc"):
        for repeat in range(3):
            order = [("baseline", args.baseline), ("candidate", args.candidate)]
            if repeat % 2:
                order.reverse()
            for variant, binary in order:
                key = f"{head}-{repeat}-{variant}"
                timing = args.output / f"{key}.time"
                command = ["/usr/bin/time", "-f", "%e %M", "-o", str(timing),
                           str(binary), "transcribe", str(source), "--model-dir", str(args.models),
                           "--model-variant", head, "--offline", "--punctuation", "off", "--itn", "off",
                           "--encoder-intra-threads", "4", "--format", "json",
                           "--output", str(args.output / f"{key}.json")]
                with (args.output / f"{key}.log").open("w") as log:
                    subprocess.run(command, env=environment, stdout=log, stderr=log, check=True, timeout=120)
                wall, rss = timing.read_text().split()
                results.append({"head": head, "repeat": repeat, "variant": variant,
                                "wall_seconds": float(wall), "peak_rss_kib": int(rss),
                                "audio_seconds": sample["seconds"]})
                write_json(args.output / "results.json", results)
                print(key, wall, rss, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--audio", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--models", type=Path, default=Path.home() / ".gigastt/models")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--modes", nargs="*", help="restrict to these modes (e.g. serial) for geometry sweeps")
    p = sub.add_parser("compare")
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--subset", action="store_true", help="candidate may cover a subset of the baseline cases")
    p = sub.add_parser("performance")
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--models", type=Path, default=Path.home() / ".gigastt/models")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("residuals")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--windows", type=Path)
    p.add_argument("--sample", default="long_example")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("pause")
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--models", type=Path, default=Path.home() / ".gigastt/models")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "compare":
        compare(args)
    else:
        {"prepare": prepare, "run": run, "pause": pause, "performance": performance,
         "residuals": residuals}[args.command](args, read_json(MANIFEST))


if __name__ == "__main__":
    main()
