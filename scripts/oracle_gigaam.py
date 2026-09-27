#!/usr/bin/env python3
"""Compare gigastt INT8 rnnt with the author gigaam FP32 package.

The short clip always runs when both the author package and the gigastt
binary are available. FLEURS-ru runs only when
``~/.gigastt/benchmarks/fleurs_ru`` and ``benchmark/manifests/fleurs_ru.json``
exist. Either missing piece skips that half and exits 0.

Author load is ``gigaam.load_model("v3_rnnt", fp16_encoder=False, device="cpu")``,
greedy, no external language model. Utterances the author package rejects as
longer than its single-pass limit are counted as empty hypotheses (100%
deletion) and are also reported on the subset both sides decode.

Word error is raw ``jiwer.wer`` (whitespace tokens, no number normalization).
That is not the harness normalizer in ``benchmark/common.py``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SHORT = REPO / "crates/gigastt/tests/fixtures/golos_00.wav"
ORACLE = "шестьдесят тысяч тенге сколько будет стоить"
MANIFEST = REPO / "benchmark/manifests/fleurs_ru.json"
AUDIO = Path.home() / ".gigastt/benchmarks/fleurs_ru"
BIN = REPO / "target/release/gigastt"


def gigastt_text(wav: Path) -> str:
    proc = subprocess.run(
        [
            str(BIN),
            "--offline",
            "transcribe",
            str(wav),
            "--model-variant",
            "rnnt",
            "--punctuation",
            "off",
            "--itn",
            "off",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    line = next(line for line in reversed(proc.stdout.splitlines()) if line.startswith("{"))
    return json.loads(line)["text"]


def wer(refs: list[str], hyps: list[str]) -> float:
    import jiwer

    if not refs:
        return 0.0
    return float(jiwer.wer(refs, hyps))


def author_text(model, wav: Path) -> str | None:
    try:
        result = model.transcribe(str(wav))
    except Exception as exc:  # author rejects clips past its single-pass limit
        print(f"  author reject {wav.name}: {exc}", file=sys.stderr)
        return None
    text = getattr(result, "text", result)
    return str(text)


def gigastt_batch(rows: list[dict]) -> dict[str, str]:
    """One process, one model load. Missing files are omitted."""
    out = REPO / "benchmark/oracle/fleurs_ru_int8"
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(BIN),
            "--offline",
            "transcribe-batch",
            str(AUDIO),
            str(out),
            "--model-variant",
            "rnnt",
            "--punctuation",
            "off",
            "--itn",
            "off",
            "--format",
            "json",
            "--pool-size",
            "1",
        ],
        check=True,
    )
    texts: dict[str, str] = {}
    for row in rows:
        path = out / (Path(row["filename"]).stem + ".json")
        if path.is_file():
            texts[row["filename"]] = json.loads(path.read_text())["text"]
    return texts


def max_abs(tensor, path: Path) -> float:
    import numpy as np

    got = tensor.detach().float().cpu().contiguous().numpy().ravel().astype(np.float32)
    ref = np.fromfile(path, dtype=np.float32)
    if got.shape != ref.shape:
        raise RuntimeError(f"shape {got.shape} != dump {ref.shape} at {path}")
    return float(np.max(np.abs(got - ref)))


def check_author_dump(model) -> int:
    """Author preprocessor and encoder must still match the committed dump."""
    mel_path = REPO / "benchmark/oracle/golos_00_author_mel.f32"
    enc_path = REPO / "benchmark/oracle/golos_00_author_encoder.f32"
    if not mel_path.is_file() or not enc_path.is_file():
        print("skip tensor dump: committed author activations are missing")
        return 0
    wav, length = model.prepare_wav(str(SHORT))
    feats, feat_len = model.preprocessor(wav, length)
    encoded, _encoded_len = model.encoder(feats, feat_len)
    mel_err = max_abs(feats, mel_path)
    enc_err = max_abs(encoded, enc_path)
    print(f"author mel max_abs {mel_err:.3e} encoder max_abs {enc_err:.3e}")
    # A fresh CPU forward of this revision matched the dump exactly.
    if mel_err > 1e-4 or enc_err > 1e-4:
        print("author activations diverged from the committed dump", file=sys.stderr)
        return 1
    return 0


def fleurs(model) -> None:
    if not MANIFEST.is_file() or not AUDIO.is_dir():
        print(
            f"skip FLEURS: need {MANIFEST} and {AUDIO} "
            "(python scripts/prepare_fleurs.py --config ru_ru)"
        )
        return
    manifest = json.loads(MANIFEST.read_text())
    rows = manifest["samples"]
    print(f"gigastt batch on {len(rows)} FLEURS files", flush=True)
    gig_by_name = gigastt_batch(rows)
    refs: list[str] = []
    gig: list[str] = []
    auth: list[str] = []
    both_ref: list[str] = []
    both_gig: list[str] = []
    both_auth: list[str] = []
    rejected = 0
    rejected_files: list[str] = []
    for i, row in enumerate(rows, 1):
        wav = AUDIO / row["filename"]
        ref = row["reference"]
        if not wav.is_file() or row["filename"] not in gig_by_name:
            print(f"skip missing {wav}", file=sys.stderr)
            continue
        hyp_g = gig_by_name[row["filename"]]
        hyp_a = author_text(model, wav)
        refs.append(ref)
        gig.append(hyp_g)
        if hyp_a is None:
            rejected += 1
            rejected_files.append(row["filename"])
            auth.append("")
        else:
            auth.append(hyp_a)
            both_ref.append(ref)
            both_gig.append(hyp_g)
            both_auth.append(hyp_a)
        if i % 25 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    summary = {
        "n": len(refs),
        "wer": "jiwer.wer, whitespace tokens, no number normalization",
        "author_rejected_long": rejected,
        "author_rejected_files": rejected_files,
        "gigastt_int8_wer": wer(refs, gig),
        "author_fp32_wer_reject_as_deletion": wer(refs, auth),
        "both_decoded_n": len(both_ref),
        "gigastt_int8_wer_both_decoded": wer(both_ref, both_gig),
        "author_fp32_wer_both_decoded": wer(both_ref, both_auth),
    }
    out = REPO / "benchmark/oracle/fleurs_ru_rnnt.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> int:
    if not BIN.is_file():
        print(f"skip: release binary missing at {BIN}", file=sys.stderr)
        return 0
    try:
        import gigaam
    except ImportError:
        print("skip: author package gigaam is not installed", file=sys.stderr)
        return 0
    text = gigastt_text(SHORT)
    print(f"gigastt {text!r}")
    if text != ORACLE:
        print("INT8 transcript diverged from the recorded author transcript", file=sys.stderr)
        return 1
    model = gigaam.load_model("v3_rnnt", fp16_encoder=False, device="cpu")
    author = author_text(model, SHORT)
    print(f"author  {author!r}")
    if author != ORACLE:
        print("author transcript diverged from the recorded oracle", file=sys.stderr)
        return 1
    if check_author_dump(model) != 0:
        return 1
    fleurs(model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
