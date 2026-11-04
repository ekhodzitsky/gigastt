#!/usr/bin/env python3
"""CPU laboratory pilot: freeze GigaAM encoder, cache embeddings, tune CTC head.

Training/validation use separate manifests. Checkpoint selection uses only the
validation micro WER, including the unmodified checkpoint as epoch zero. Test
manifests are evaluated only after selection. Private transcripts stay in work.
This does not export or replace a production model.
"""
import argparse
import copy
import hashlib
import inspect
import itertools
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
import gigaam

from wer_unicode import normalize, word_edit_distance


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_rows(path):
    data = json.loads(path.read_text())
    rows = data["samples"] if isinstance(data, dict) else data
    return [{**row, "reference": row.get("reference_normalized", row["reference"])} for row in rows]


def cache_manifest(model, path, cache, feature_dir):
    rows = load_rows(path)
    for i, row in enumerate(rows):
        # File identity plus model and frontend are recorded in cache metadata.
        key = hashlib.sha256((row["path"] + "\0" + row.get("condition", "original")).encode()).hexdigest()
        target = cache / (key + ".npz")
        metadata_path = cache / (key + ".json")
        row["embedding_path"] = str(target)
        actual_audio_hash = digest(row["path"])
        expected_audio_hash = row.get("audio_sha256", row.get("sha256"))
        if expected_audio_hash and actual_audio_hash != expected_audio_hash:
            raise ValueError(f"Audio checksum mismatch: {row['id']}")
        features_path = None if feature_dir is None else feature_dir / (row["id"] + ".npy")
        if features_path is not None and not features_path.exists():
            features_path = None
        identity = {**model._pilot_cache_identity, "audio_sha256": actual_audio_hash,
                    "exact_features_sha256": None if features_path is None else digest(features_path)}
        if target.exists() and metadata_path.exists():
            previous = json.loads(metadata_path.read_text())
            if previous.get("identity") == identity and previous.get("embedding_sha256") == digest(target):
                continue
        with torch.inference_mode():
            if features_path is not None:
                features = torch.from_numpy(np.load(features_path))
                length = torch.tensor([features.shape[2]])
            else:
                wav, wavlen = model.prepare_wav(row["path"])
                features, length = model.preprocessor(wav, wavlen)
            encoded, elen = model.encoder(features, length)
            partial = target.with_suffix(".npz.partial")
            with partial.open("wb") as stream:
                np.savez(stream, encoded=encoded[0, :, :int(elen[0])].numpy())
            partial.replace(target)
            dump(metadata_path, dict(identity=identity, embedding_sha256=digest(target)))
        print(f"Cached {i + 1}/{len(rows)} {row['id']} {row.get('condition', 'original')}", flush=True)
    return rows


def embedding(row):
    with np.load(row["embedding_path"]) as value:
        return torch.from_numpy(value["encoded"])


def evaluate(head, rows, vocab):
    details = []
    groups = {}
    with torch.inference_mode():
        for row in rows:
            encoded = embedding(row)
            ids = head(encoded[None]).argmax(-1).ravel().tolist()
            text = "".join(vocab[i] for i, _ in itertools.groupby(ids) if i != len(vocab)).strip()
            ref = normalize(row["reference"])
            errors = word_edit_distance(ref, normalize(text))
            key = row.get("language", row.get("dataset", "unknown")) + "/" + row.get("condition", "original")
            group = groups.setdefault(key, dict(n=0, errors=0, ref_words=0, empty=0))
            group["n"] += 1
            group["errors"] += errors
            group["ref_words"] += len(ref)
            group["empty"] += not bool(text)
            details.append(dict(id=row["id"], condition=row.get("condition", "original"), group=key, errors=errors, ref_words=len(ref), hypothesis=text, reference=row["reference"], audio_sha256=row.get("audio_sha256", row.get("sha256"))))
    for group in groups.values():
        group["wer_percent"] = 100 * group["errors"] / group["ref_words"] if group["ref_words"] else None
    total_words = sum(d["ref_words"] for d in details)
    return dict(wer_percent=100 * sum(d["errors"] for d in details) / total_words, groups=groups, details=details)


def public(score):
    return {**score, "details": [{k: v for k, v in row.items() if k not in {"reference", "hypothesis"}} for row in score["details"]]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["cache", "train"])
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--feature-dir", type=Path)
    p.add_argument("--train-manifest", type=Path)
    p.add_argument("--validation-manifest", type=Path)
    p.add_argument("--test-manifest", type=Path)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--threads", type=int, default=3)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    args.work.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.work / "embeddings"
    cache.mkdir(exist_ok=True)
    model = gigaam.load_model("multilingual_ctc", device="cpu", fp16_encoder=False, download_root=str(args.model_dir))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    model_hash = digest(args.model_dir / "multilingual_ctc.ckpt")
    import torchaudio
    frontend_hash = hashlib.sha256(inspect.getsource(type(model.preprocessor)).encode())
    frontend_hash.update(inspect.getsource(gigaam.load_audio).encode())
    for name, buffer in model.preprocessor.named_buffers():
        frontend_hash.update(name.encode())
        frontend_hash.update(buffer.numpy().tobytes())
    model._pilot_cache_identity = dict(source_model_sha256=model_hash, torch=torch.__version__,
                                      torchaudio=torchaudio.__version__, frontend_sha256=frontend_hash.hexdigest(),
                                      encoder_source_sha256=hashlib.sha256(inspect.getsource(type(model.encoder)).encode()).hexdigest(),
                                      ffmpeg_version=subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0])
    cache_meta = {**model._pilot_cache_identity, "schema_version": 2,
                  "feature_source": "official checkpoint frontend and FFmpeg PCM16, or exact cached source features"}
    meta_path = cache / "metadata.json"
    if meta_path.exists() and json.loads(meta_path.read_text()).get("source_model_sha256") != model_hash:
        raise ValueError("Embedding cache model mismatch")
    dump(meta_path, cache_meta)
    if args.stage == "cache":
        rows = cache_manifest(model, args.manifest, cache, args.feature_dir)
        dump(args.work / (args.manifest.stem + "_cached.json"), rows)
        return
    train = cache_manifest(model, args.train_manifest, cache, None)
    val = cache_manifest(model, args.validation_manifest, cache, None)
    vocab = list(model.decoding.tokenizer.vocab)
    tokens = {ch: i for i, ch in enumerate(vocab)}
    eligible, excluded = [], []
    for row in train:
        text = " ".join(normalize(row["reference"]))
        unknown = sorted(set(text) - set(tokens))
        if unknown:
            excluded.append(dict(id=row["id"], condition=row["condition"], reason="characters outside model vocabulary", characters=unknown))
            continue
        target = torch.tensor([tokens[ch] for ch in text], dtype=torch.long)
        frames = embedding(row).shape[1]
        minimum = len(target) + sum(a == b for a, b in zip(text, text[1:]))
        if minimum > frames:
            excluded.append(dict(id=row["id"], condition=row["condition"], reason="CTC target cannot fit encoded frames"))
            continue
        eligible.append((row, target))
    if not eligible:
        raise ValueError("No representable CTC training examples")
    head = model.head
    baseline = copy.deepcopy(head.state_dict())
    for param in head.parameters():
        param.requires_grad_(True)
    optim = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=0.01)
    criterion = torch.nn.CTCLoss(blank=len(vocab), reduction="mean", zero_infinity=False)
    base_val = evaluate(head, val, vocab)
    best_wer, best_epoch, best_state = base_val["wer_percent"], 0, copy.deepcopy(baseline)
    history = [dict(epoch=0, validation=public(base_val))]
    for epoch in range(1, args.epochs + 1):
        order = list(eligible)
        random.Random(42 + epoch).shuffle(order)
        losses = []
        start = time.monotonic()
        head.train()
        for offset in range(0, len(order), args.batch_size):
            batch = order[offset:offset + args.batch_size]
            tensors = [embedding(row) for row, _ in batch]
            lengths = torch.tensor([t.shape[1] for t in tensors])
            padded = torch.zeros(len(tensors), tensors[0].shape[0], int(lengths.max()))
            for i, tensor in enumerate(tensors):
                padded[i, :, :tensor.shape[1]] = tensor
            targets = torch.cat([target for _, target in batch])
            target_lengths = torch.tensor([len(target) for _, target in batch])
            optim.zero_grad()
            log_probs = head(padded).transpose(0, 1)
            loss = criterion(log_probs, targets, lengths, target_lengths)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite CTC loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optim.step()
            losses.append(float(loss.detach()))
        head.eval()
        score = evaluate(head, val, vocab)
        history.append(dict(epoch=epoch, mean_batch_ctc_loss=sum(losses)/len(losses), elapsed_seconds=time.monotonic()-start, validation=public(score)))
        if score["wer_percent"] < best_wer:
            best_wer, best_epoch, best_state = score["wer_percent"], epoch, copy.deepcopy(head.state_dict())
        dump(args.output / "training_history.json", history)
        print(f"Epoch {epoch}: validation WER {score['wer_percent']:.4f}; best epoch {best_epoch}", flush=True)
    # Held-out evaluation starts only after checkpoint selection is final.
    test = cache_manifest(model, args.test_manifest, cache, args.feature_dir)
    head.load_state_dict(baseline)
    before = evaluate(head, test, vocab)
    head.load_state_dict(best_state)
    after = evaluate(head, test, vocab)
    checkpoint_path = args.work / "selected_ctc_head.pt"
    torch.save(best_state, checkpoint_path)
    metadata = dict(source_model_sha256=model_hash, selected_head_sha256=digest(checkpoint_path), selected_epoch=best_epoch, selection="Lowest validation combined clean/telephone micro WER; unmodified epoch zero included; ties retain earlier checkpoint", train_count=len(train), eligible_training_count=len(eligible), excluded_training=excluded, validation_count=len(val), test_count=len(test), train_manifest_sha256=digest(args.train_manifest), validation_manifest_sha256=digest(args.validation_manifest), test_manifest_sha256=digest(args.test_manifest), epochs=args.epochs, learning_rate=args.learning_rate, batch_size=args.batch_size, seed=42, torch=torch.__version__, trainable_parameters=sum(p.numel() for p in head.parameters()), frozen_encoder=True, checkpoint_scope="CTC head only; laboratory artifact; not exported or deployed")
    dump(args.work / "evaluation_private.json", dict(metadata=metadata, before=before, after=after))
    dump(args.output / "evaluation.json", dict(metadata=metadata, before=public(before), after=public(after)))
    dump(args.output / "training_history.json", history)
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
