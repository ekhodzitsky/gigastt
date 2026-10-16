#!/usr/bin/env python3
"""Laboratory source/FP32/INT8 comparison on exactly the same log-mel arrays.

Run prepare/source first, then onnx with --model and --label. Audio, references,
features and hypotheses stay in --work; only aggregate summaries are exported.
The source stage uses the official checkpoint frontend and FFmpeg PCM16 loader.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import time

from wer_unicode import normalize, word_edit_distance


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare(args):
    rows = []
    for manifest_path, language, limit in [
        (args.kazakh_manifest, "kazakh", 31),
        (args.uzbek_manifest, "uzbek", 10),
        (args.telephone_manifest, "telephone_oracle_segments", 1000),
    ]:
        manifest = json.loads(manifest_path.read_text())
        for sample in manifest["samples"][:limit]:
            for condition in (["original"] if language.startswith("telephone") else ["original", "simulated_telephone"]):
                key = f'{language}_{sample["id"]}_{condition}'
                source = Path(sample["path"])
                path = source
                if condition == "simulated_telephone":
                    path = args.work / f"{key}.wav"
                    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-ac", "1", "-af", "highpass=f=300,lowpass=f=3400", "-ar", "8000", "-c:a", "pcm_mulaw", str(path)], check=True)
                rows.append(dict(id=key, language=language, condition=condition, path=str(path), audio_sha256=digest(path), reference=sample.get("reference_normalized", sample["reference"])))
    if args.telephone_full_manifest:
        sample = json.loads(args.telephone_full_manifest.read_text())["samples"][0]
        segments = json.loads(args.telephone_manifest.read_text())["samples"]
        for duration, key in [(30, "telephone_first30"), (24.815, "telephone_first24_815")]:
            path = args.work / f"{key}.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", sample["path"], "-t", str(duration), str(path)], check=True)
            reference = " ".join(s["reference"] for s in segments if s["start"] < duration)
            rows.insert(2, dict(id=key, language=key + "_diagnostic", condition="original", path=str(path), audio_sha256=digest(path), reference=reference))
    if args.extra_telephone_manifest:
        for sample in json.loads(args.extra_telephone_manifest.read_text())["samples"]:
            path = Path(sample["path"])
            rows.append(dict(id=sample["id"], language="material_telephone_oracle_segments", condition="original", path=str(path), audio_sha256=digest(path), reference=sample["reference"]))
    dump(args.work / "manifest.json", rows)


def prepare_frontends(args):
    kazakh = json.loads(args.kazakh_manifest.read_text())["samples"][0]
    segments = json.loads(args.telephone_manifest.read_text())["samples"]
    phone_reference = " ".join(s["reference"] for s in segments if s["start"] < 30)
    rows = []
    for name in ["original", "internal_mulaw8", "external16", "telephone"]:
        for variant in ["rust_features", "source_frontend_float", "source_frontend_pcm16"]:
            if name == "telephone" and variant == "rust_features":
                continue
            pcm = args.dump_dir / (name + ".pcm.f32")
            row = dict(id=name + "_" + variant, language="control_" + name,
                       condition=variant, path=kazakh["path"], pcm_f32=str(pcm),
                       audio_sha256=digest(pcm), reference=phone_reference if name == "telephone" else kazakh.get("reference_normalized", kazakh["reference"]))
            if variant == "rust_features":
                row["features_f32"] = str(args.dump_dir / (name + ".mel.f32"))
            if variant == "source_frontend_pcm16":
                row["round_pcm16"] = True
            if name == "telephone":
                row["max_samples"] = 480000
            rows.append(row)
    dump(args.work / "manifest.json", rows)


def score(row, text, elapsed):
    ref = normalize(row["reference"])
    return {**row, "hypothesis": text, "ref_words": len(ref), "errors": word_edit_distance(ref, normalize(text)), "elapsed_seconds": elapsed}


def finish(args, rows, metadata):
    groups = {}
    for row in rows:
        key = row["language"] + "/" + row["condition"]
        g = groups.setdefault(key, dict(n=0, ref_words=0, errors=0, empty=0, elapsed_seconds=0))
        for name in ["ref_words", "errors", "elapsed_seconds"]:
            g[name] += row[name]
        g["n"] += 1
        g["empty"] += not bool(row["hypothesis"])
    for g in groups.values():
        g["wer_percent"] = 100 * g["errors"] / g["ref_words"] if g["ref_words"] else None
    dump(args.work / f"{args.label}_private.json", dict(metadata=metadata, groups=groups, rows=rows))
    args.output.mkdir(parents=True, exist_ok=True)
    public_rows = [{k: v for k, v in row.items() if k not in {"reference", "hypothesis", "path"}} for row in rows]
    dump(args.output / f"{args.label}.json", dict(metadata=metadata, groups=groups, rows=public_rows))
    print(json.dumps(groups), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "prepare-frontends", "source", "onnx"])
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path)
    p.add_argument("--label", default="source")
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--limit", type=int)
    p.add_argument("--resume", action="store_true", help="Reuse completed rows after verifying model and audio hashes")
    p.add_argument("--kazakh-manifest", type=Path)
    p.add_argument("--uzbek-manifest", type=Path)
    p.add_argument("--telephone-manifest", type=Path)
    p.add_argument("--telephone-full-manifest", type=Path)
    p.add_argument("--extra-telephone-manifest", type=Path)
    p.add_argument("--dump-dir", type=Path)
    args = p.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        prepare(args)
        return
    if args.stage == "prepare-frontends":
        prepare_frontends(args)
        return
    import numpy as np
    manifest = json.loads((args.work / "manifest.json").read_text())
    if args.limit:
        manifest = manifest[:args.limit]
    rows = []
    previous_path = args.work / f"{args.label}_private.json"
    if args.resume and previous_path.exists():
        previous = json.loads(previous_path.read_text())
        if previous["metadata"]["model_sha256"] != digest(args.model):
            raise ValueError("Cannot resume with different model bytes")
        current = {r["id"]: r for r in manifest}
        rows = []
        for row in previous["rows"]:
            feature_path = args.work / (row["id"] + ".npy")
            item = current.get(row["id"])
            if (item is not None and row["audio_sha256"] == item["audio_sha256"]
                    and row["reference"] == item["reference"] and feature_path.exists()
                    and digest(feature_path) == row["feature_sha256"]):
                audio_path = item.get("pcm_f32", item["path"])
                if digest(audio_path) != item["audio_sha256"]:
                    raise ValueError(f"Audio checksum mismatch: {row['id']}")
                rows.append(row)
        done = {r["id"] for r in rows}
        manifest = [r for r in manifest if r["id"] not in done]
    if args.stage == "source":
        import torch
        import gigaam
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        model = gigaam.load_model("multilingual_ctc", device="cpu", fp16_encoder=False, download_root=str(args.model.parent))
        import torchaudio
        metadata = dict(stage="official PyTorch source, FP32 CPU", torch=torch.__version__, torchaudio=torchaudio.__version__, source_revision=subprocess.check_output(["git", "-C", "/tmp/gigastt-gigaam", "rev-parse", "HEAD"], text=True).strip(), model_sha256=digest(args.model), threads=args.threads)
        if args.resume and previous_path.exists() and previous["metadata"] != metadata:
            raise ValueError("Cannot resume with different source/runtime metadata")
        vocab = list(model.decoding.tokenizer.vocab)
        dump(args.work / "vocab.json", vocab)
        with torch.inference_mode():
            for row in manifest:
                if "pcm_f32" in row:
                    pcm = np.fromfile(row["pcm_f32"], dtype="<f4")
                    pcm = pcm[:row.get("max_samples", len(pcm))]
                    if row.get("round_pcm16"):
                        pcm = np.clip(np.round(pcm * 32768), -32768, 32767) / 32768
                    wav = torch.from_numpy(pcm)[None]
                    length = torch.tensor([len(pcm)])
                else:
                    wav, length = model.prepare_wav(row["path"])
                if "features_f32" in row:
                    features = torch.from_numpy(np.fromfile(row["features_f32"], dtype="<f4").reshape(1, 64, -1))
                    flen = torch.tensor([features.shape[2]])
                else:
                    features, flen = model.preprocessor(wav, length)
                feature_path = args.work / (row["id"] + ".npy")
                np.save(feature_path, features.numpy())
                start = time.monotonic()
                encoded, elen = model.encoder(features, flen)
                logits = model.head(encoded)
                text = model.decoding.decode(model.head, encoded, elen)[0][0]
                elapsed = time.monotonic() - start
                np.save(args.work / (row["id"] + ".source_logits.npy"), logits.numpy())
                out = score(row, text, elapsed)
                out.update(feature_sha256=digest(feature_path), frames=int(flen[0]))
                rows.append(out)
                print(row["id"], out["errors"], "/", out["ref_words"], flush=True)
                finish(args, rows, metadata)
    else:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(args.model), options, providers=["CPUExecutionProvider"])
        vocab = json.loads((args.work / "vocab.json").read_text())
        metadata = dict(stage="ONNX exact source features", onnxruntime=ort.__version__, model_sha256=digest(args.model), threads=args.threads)
        if args.resume and previous_path.exists() and previous["metadata"] != metadata:
            raise ValueError("Cannot resume with different ONNX/runtime metadata")
        for row in manifest:
            feature_path = args.work / (row["id"] + ".npy")
            features = np.load(feature_path)
            start = time.monotonic()
            logits = session.run(None, {"features": features, "feature_lengths": np.array([features.shape[2]], dtype="int64")})[0]
            elapsed = time.monotonic() - start
            ids = np.argmax(logits, axis=-1).ravel()
            text = "".join(vocab[i] for i, _ in itertools.groupby(ids) if i != len(vocab)).replace("▁", " ").strip()
            source_logits = np.load(args.work / (row["id"] + ".source_logits.npy"))
            delta = np.abs(logits - source_logits)
            out = score(row, text, elapsed)
            out.update(feature_sha256=digest(feature_path), frames=int(features.shape[2]), source_logits_mae=float(delta.mean()), source_logits_max_error=float(delta.max()), source_frame_argmax_agreement=float((logits.argmax(-1) == source_logits.argmax(-1)).mean()))
            rows.append(out)
            print(row["id"], out["errors"], "/", out["ref_words"], flush=True)
            finish(args, rows, metadata)


if __name__ == "__main__":
    main()
