# Frozen-encoder multilingual adaptation pilot — 2026-10-01

An exploratory CTC-head adaptation reduced simulated Kazakh telephone WER
from **41.21% to 13.80%** in the source FP32 model, but regressed clean Kazakh
and Uzbek recognition and did not solve real-call accuracy. **Do not deploy
this candidate.** These are source-model experiments, not INT8 product scores.

## Selected candidate: held-out measurements

Selection used FLEURS validation only: learning rate 0.001, epoch 2. The table
was evaluated after that selection; it was not used to choose the checkpoint.
Both columns use the same frozen encoder, exact checkpoint frontend, FFmpeg
PCM16 input preparation and greedy CTC decoding.

| Condition | Inputs | Reference words | Original head | Selected head |
|---|---:|---:|---:|---:|
| Kazakh/Russian natural speech, original | 31 | 529 | 6.81% | 9.64% |
| Same speech, simulated telephone | 31 | 529 | 41.21% | 13.80% |
| Uzbek FLEURS original control | 10 | 190 | 19.47% | 22.63% |
| Same Uzbek control, simulated telephone | 10 | 190 | 21.05% | 25.79% |
| Babel real call, oracle utterance cuts | 24 | 77 | 77.92% | 75.32% |
| MATERIAL real call, oracle utterance cuts | 24 | 93 | 45.16% | 46.24% |
| Babel 0–24.815 s diagnostic | 1 | 28 | 100% | 75.00% |
| Babel 0–30 s diagnostic | 1 | 28 | 100% | 71.43% |

The real-call evidence is two public example calls; the 48 utterance cuts
are not independent calls. Oracle cuts use reference timestamps. The longer
crops overlap the same Babel call; do not pool them into an overall accuracy
claim. The 30 s crop bypasses the original short-transcribe API's 25 s guard,
whereas 24.815 s is within it. This is not the upstream Pyannote long-form
pipeline. Data provenance and the original source/ONNX comparison are in the
[precision audit](../multilingual_precision_20261001/README.md).

This result shows that even a small head-only update can alter narrowband
behavior substantially. It does **not** show that read-speech augmentation is
sufficient for actual calls, or that the selected candidate is an overall
improvement. A dedicated Russian-only regression suite, INT8 export and
quantization validation were not run for this trained head. The observed clean
and Uzbek regressions already fail a production non-regression requirement.

## Training data and fixed protocol

- Official FLEURS `kk_kz` train/validation splits, revision
  `168de341b3db6859a9bac1c50a2ef5e3b47647e0`, CC BY 4.0, Conneau et al. (2022).
  `scripts/prepare_multilingual_adaptation.py` verifies source-file checksums
  and rejects cross-split sentence-ID, normalized-text or audio-hash overlap.
- Initially selected 128 train recordings and 32 validation recordings, each
  paired with a 300–3400 Hz, 8 kHz mu-law simulation. This is read speech with
  simulated degradation, not real-call training data.
- **30 train recordings were excluded** because references contain digits
  absent from the model's character vocabulary. No digits were silently
  deleted from training targets. The actual training set is **98 recordings,
  196 clean/telephone entries**, about 20.1 minutes of original audio and
  40.3 minutes including its simulations. All 64 validation entries remain
  scored, including six with digit-bearing references.
- All encoder parameters were frozen. Only the 54,599-parameter CTC head was
  optimized using cached float32 encoder embeddings. AdamW, weight decay
  0.01, batch size 8, gradient clipping at 1.0, seed 42, three epochs per run.
  Training targets use the existing Unicode-aware normalization. No LDC,
  YouTube benchmark or Uzbek evaluation recording was used for training.
- Checkpoint selection minimizes combined clean/telephone validation micro
  WER. Epoch zero is eligible; ties retain the earlier checkpoint. Validation
  is only 32 source recordings, so this is a small exploratory pilot.

## All validation results

The initial prescribed learning rate 0.00002 changed training loss but left
validation WER unchanged. Epoch zero was retained, and its held-out results
are identical to the precision audit. After this negative result, a **fixed
exploratory follow-up** tested only 0.0001 and 0.001, three epochs each, using
the same fresh embeddings and validation set. No further sweep followed.

| Learning rate | Epoch | Mean batch CTC loss | Validation WER | Head weight delta L2 |
|---:|---:|---:|---:|---:|
| Baseline | 0 | — | 10.1790% | 0 |
| 0.00002 | 1 | 0.118505 | 10.1790% | Not recorded |
| 0.00002 | 2 | 0.115477 | 10.1790% | Not recorded |
| 0.00002 | 3 | 0.126469 | 10.1790% | Not recorded |
| 0.0001 | 1 | 0.114376 | 10.2908% | 0.199285 |
| 0.0001 | 2 | 0.102752 | 10.4027% | 0.363848 |
| 0.0001 | 3 | 0.107330 | 10.6264% | 0.513492 |
| 0.001 | 1 | 0.098514 | 10.4027% | 1.527667 |
| **0.001** | **2** | **0.059856** | **9.0604%** | **2.452992** |
| 0.001 | 3 | 0.044789 | 9.0604% | 3.091597 |

Loss and nonzero parameter deltas confirm actual optimization, not an inert
training loop. Mean batch loss is not an evaluation loss and can fluctuate
with shuffled batches. The selected validation reduction is ten word errors,
from 91/894 to 81/894. It is not an independent estimate of generalization:
the validation set was reused for model selection, and this follow-up was
motivated by the first negative experiment. Test datasets were already used
for baseline research; no claim of a newly blinded benchmark is made.

## Artifacts and reproduction

- `evaluation.json` / `training_history.json`: initial three-epoch pilot,
  including the full list of 60 excluded clean/telephone entries.
- `sweep_evaluation.json` / `sweep_history.json`: final selected candidate and
  all fixed-sweep validation results, with model/manifest/embedding-lock hashes.
- `validation.json`: initial baseline score parity, epoch-zero invariance,
  Python compilation and cache-identity smoke checks.
- Local selected head:
  `/home/ekhodzitsky/.cache/gigastt-adaptation/sweep_selected_ctc_head.pt`.
  This is a head state dictionary, not a complete deployable GigaAM model.
  It was not exported, quantized, installed in gigastt or published.

Use the source environment from the precision audit, with three CPU threads:

```sh
/tmp/gigastt-precision-venv/bin/python scripts/finetune_multilingual_pilot.py train \
  --model-dir ~/.cache/gigastt-precision/models \
  --work ~/.cache/gigastt-adaptation \
  --output benchmark/results/multilingual_adaptation_20261001 \
  --train-manifest ~/.cache/gigastt-training-data/pilot/train_manifest.json \
  --validation-manifest ~/.cache/gigastt-training-data/pilot/validation_manifest.json \
  --test-manifest /tmp/gigastt-precision/manifest.json \
  --feature-dir /tmp/gigastt-precision

/tmp/gigastt-precision-venv/bin/python scripts/finetune_multilingual_sweep.py \
  --model-dir ~/.cache/gigastt-precision/models \
  --work ~/.cache/gigastt-adaptation \
  --output benchmark/results/multilingual_adaptation_20261001 \
  --train-manifest ~/.cache/gigastt-training-data/pilot/train_manifest.json \
  --validation-manifest ~/.cache/gigastt-training-data/pilot/validation_manifest.json \
  --test-manifest /tmp/gigastt-precision/manifest.json
```

The first run's embeddings were freshly generated in this session. The sweep
verifies its completed-pilot model/runtime and source manifest/audio hashes,
locks every embedding's SHA-256 before fitting, rechecks those hashes afterward
and verifies original-head test score parity. General future cache reuse now
also checks per-entry model/frontend/runtime identity and embedding checksums;
legacy entries without this metadata are recomputed. Cache smoke tests verify
that corruption triggers recomputation and an audio checksum mismatch fails.
Audio, references and generated transcripts remain local; repository results
contain scores/hashes only. Runtime contention makes elapsed times unsuitable
for comparing throughput.
