#!/usr/bin/env python3
"""Fixed exploratory LR sweep after the negative head-only pilot.

Reuses only the freshly prepared embeddings whose source manifests and model
match the completed pilot; locks their byte hashes before fitting. No encoder
updates. All checkpoint selection uses validation; test evaluation is last.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random

import torch
import gigaam

from finetune_multilingual_pilot import digest, dump, load_rows, embedding, evaluate, public
from wer_unicode import normalize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--validation-manifest", type=Path, required=True)
    p.add_argument("--test-manifest", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    random.seed(42)
    initial_path = args.work / "evaluation_private.json"
    initial = json.loads(initial_path.read_text())
    prior = initial["metadata"]
    model_sha = digest(args.model_dir / "multilingual_ctc.ckpt")
    if model_sha != prior["source_model_sha256"] or torch.__version__ != prior["torch"]:
        raise ValueError("Model/runtime differs from completed fresh-cache pilot")
    cache_meta = json.loads((args.work / "embeddings/metadata.json").read_text())
    if cache_meta["source_model_sha256"] != model_sha or cache_meta["torch"] != torch.__version__:
        raise ValueError("Embedding cache provenance differs from initial pilot")
    datasets = {}
    locked = []
    for split in ["train", "validation", "test"]:
        path = getattr(args, split + "_manifest")
        if digest(path) != prior[split + "_manifest_sha256"]:
            raise ValueError(f"{split} manifest changed since initial pilot")
        rows = load_rows(path)
        for row in rows:
            expected = row.get("audio_sha256", row.get("sha256"))
            if expected != digest(row["path"]):
                raise ValueError(f"Audio checksum mismatch: {row['id']}")
            key = hashlib.sha256((row["path"] + "\0" + row.get("condition", "original")).encode()).hexdigest()
            target = args.work / "embeddings" / (key + ".npz")
            row["embedding_path"] = str(target)
            locked.append(dict(split=split, id=row["id"], condition=row.get("condition", "original"), audio_sha256=expected, embedding_path=str(target), embedding_sha256=digest(target)))
        datasets[split] = rows
    lock_path = args.work / "sweep_embedding_lock.json"
    dump(lock_path, dict(initial_pilot_sha256=digest(initial_path), source_model_sha256=model_sha, entries=locked))
    model = gigaam.load_model("multilingual_ctc", device="cpu", fp16_encoder=False, download_root=str(args.model_dir))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head = model.head
    for parameter in head.parameters():
        parameter.requires_grad_(True)
    baseline = copy.deepcopy(head.state_dict())
    vocab = list(model.decoding.tokenizer.vocab)
    tokens = {ch: i for i, ch in enumerate(vocab)}
    eligible = []
    for row in datasets["train"]:
        text = " ".join(normalize(row["reference"]))
        if set(text) - set(tokens):
            continue
        target = torch.tensor([tokens[ch] for ch in text], dtype=torch.long)
        if len(target) + sum(a == b for a, b in zip(text, text[1:])) > embedding(row).shape[1]:
            continue
        eligible.append((row, target))
    if len(eligible) != prior["eligible_training_count"]:
        raise ValueError("Training eligibility changed")
    baseline_val = evaluate(head, datasets["validation"], vocab)
    best_score = baseline_val["wer_percent"]
    best_state, best_lr, best_epoch = copy.deepcopy(baseline), None, 0
    history = [dict(learning_rate=None, epoch=0, validation=public(baseline_val), weight_delta_l2=0)]
    criterion = torch.nn.CTCLoss(blank=len(vocab), reduction="mean", zero_infinity=False)
    for lr in [1e-4, 1e-3]:
        head.load_state_dict(baseline)
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
        for epoch in range(1, 4):
            order = list(eligible)
            random.Random(42 + epoch).shuffle(order)
            losses = []
            head.train()
            for offset in range(0, len(order), 8):
                batch = order[offset:offset + 8]
                tensors = [embedding(row) for row, _ in batch]
                lengths = torch.tensor([t.shape[1] for t in tensors])
                padded = torch.zeros(len(tensors), tensors[0].shape[0], int(lengths.max()))
                for i, tensor in enumerate(tensors):
                    padded[i, :, :tensor.shape[1]] = tensor
                targets = torch.cat([target for _, target in batch])
                target_lengths = torch.tensor([len(target) for _, target in batch])
                optimizer.zero_grad()
                loss = criterion(head(padded).transpose(0, 1), targets, lengths, target_lengths)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite CTC loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            head.eval()
            score = evaluate(head, datasets["validation"], vocab)
            delta = sum(float(((value - baseline[name]) ** 2).sum()) for name, value in head.state_dict().items()) ** 0.5
            history.append(dict(learning_rate=lr, epoch=epoch, mean_batch_ctc_loss=sum(losses)/len(losses), weight_delta_l2=delta, validation=public(score)))
            if score["wer_percent"] < best_score:
                best_score, best_lr, best_epoch, best_state = score["wer_percent"], lr, epoch, copy.deepcopy(head.state_dict())
            dump(args.output / "sweep_history.json", history)
            print(f"LR {lr} epoch {epoch}: val WER {score['wer_percent']:.4f}, weight delta {delta:.6f}", flush=True)
    # Global selection is final before the first adapted-head test evaluation.
    for row in locked:
        if digest(row["embedding_path"]) != row["embedding_sha256"]:
            raise ValueError("Prepared embeddings changed during sweep")
    head.load_state_dict(baseline)
    before = evaluate(head, datasets["test"], vocab)
    old = {(r["id"], r["condition"]): r for r in initial["before"]["details"]}
    if any((r["errors"], r["ref_words"]) != (old[r["id"],r["condition"]]["errors"], old[r["id"],r["condition"]]["ref_words"]) for r in before["details"]):
        raise ValueError("Baseline test results differ from initial pilot")
    head.load_state_dict(best_state)
    after = evaluate(head, datasets["test"], vocab)
    checkpoint = args.work / "sweep_selected_ctc_head.pt"
    torch.save(best_state, checkpoint)
    metadata = dict(source_model_sha256=model_sha, selected_head_sha256=digest(checkpoint), selected_learning_rate=best_lr, selected_epoch=best_epoch, selected_validation_wer=best_score, learning_rates=[1e-4,1e-3], epochs_per_rate=3, seed=42, batch_size=8, eligible_training_count=len(eligible), validation_count=len(datasets['validation']), test_count=len(datasets['test']), embedding_lock_sha256=digest(lock_path), initial_pilot_sha256=digest(initial_path), selection="Exploratory fixed two-rate follow-up after negative initial pilot; combined validation micro WER only; epoch0 eligible; no test-based selection; no further sweep", checkpoint_scope="CTC head only; FP32 laboratory artifact; not exported or deployed", test_aggregate_warning="Overlapping diagnostic crops and conditions; interpret separate groups, not pooled WER")
    dump(args.work / "sweep_evaluation_private.json", dict(metadata=metadata,before=before,after=after))
    dump(args.output / "sweep_evaluation.json", dict(metadata=metadata,before=public(before),after=public(after)))
    print(json.dumps(metadata),flush=True)


if __name__ == "__main__":
    main()
