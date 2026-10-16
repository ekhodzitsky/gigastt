# Kazakh/Russian Whisper alternative: bounded CPU pilot

Measured on 2026-10-01. The alternative preserves quality under the simulated telephone degradation much better on this small paired subset, but still makes too many errors on actual calls. It is a laboratory candidate, not a production replacement.

## Matched public clips

Normalized **single-reference micro WER**, lower is better:

| Model | Original | Simulated 8 kHz G.711 telephone |
|---|---:|---:|
| Kazakh/Russian Whisper LoRA, BF16, beam 5 | 10.64% | 10.64% |
| gigastt `ml_ctc` INT8, PCM16-resampling candidate | 9.57% | 55.32% |
| gigastt `ml_ctc_large` INT8, PCM16-resampling candidate | 9.57% | 36.17% |

These are **five clips**, 35.17 seconds and 94 reference words, evaluated in two conditions. They are the first five entries of the existing 31-clip manifest, selected before this subset run; the first clip had already served as an implementation smoke test. This is a nonrandom, small pilot. Do not compare these numbers to the complete 31-clip headline or the upstream macro WER_best scorer.

The input WAV SHA-256 and normalized reference were checked for every paired request. The gigastt results are reused from the previously measured `narrowband_20261001/candidate_kazakh_*.json` files, filtered to these exact five clips. The simulated condition uses the same 300–3400 Hz filters, 8 kHz sampling and G.711 mu-law encoding. Whisper then decodes through FFmpeg to mono 16 kHz PCM16, as recorded in the runner.

Full paired metrics and source paths: [matched_first5_comparison.json](matched_first5_comparison.json). Public hypotheses: [kazakh_first5_bf16.json](kazakh_first5_bf16.json).

## Actual telephone speech

The main alternative configuration is the same pinned model, **PyTorch BF16, CPU, three threads, beam 5, automatic language selection**, without VAD. Each provided input is split into independent, nonoverlapping chunks of at most 30 seconds. These are full-channel results, not oracle segmentation.

| Public demonstration | Recordings | Channels | Reference words | WER | CER | S / D / I |
|---|---:|---:|---:|---:|---:|---:|
| [LDC2018S13](https://catalog.ldc.upenn.edu/LDC2018S13) | 1 | 1 | 77 | 75.32% | 43.85% | 47 / 9 / 2 |
| [LDC2025S03 MATERIAL](https://catalog.ldc.upenn.edu/LDC2025S03) | 1 | 2 | 93 | 52.69% | 33.80% | 20 / 24 / 5 |

The MATERIAL channels are two sides of one conversation, not two independent calls. Both were cut at 56.140 seconds to match the extent of the published transcript; the source WAV is longer. See [source preparation](../telephone_corpora_20261001/material_sample_provenance.json). Corpus overlap beyond the distinct public samples cannot be ruled out.

None of the three full-channel outputs was empty or reached the generation token cap. Seven inserted words were counted across both calls. Insertions alone do not identify semantic hallucinations; dedicated silence controls were not run in this bounded suite.

Only aggregate phone results are published: [telephone_full_bf16_summary.json](telephone_full_bf16_summary.json). Audio, references and hypotheses stay under `/tmp`. Two demonstration calls cannot establish deployment-level accuracy.

## Model and implementation controls

The candidate is [abilmansplus/whisper-turbo-kaz-rus-v1](https://huggingface.co/abilmansplus/whisper-turbo-kaz-rus-v1/blob/904d5d22339952909001d1abcbc1ecf29d26d706/README.md), a LoRA adapter on `abilmansplus/whisper-turbo-ksc2`. Both cards declare MIT; the author describes training on KSC2 and Golos and recommends additional domain training for telephone speech. Exact base/adapter revisions and weight hashes are in [provenance.json](provenance.json). Only safetensors/configuration/tokenizer files were loaded, without remote model code or cloud audio upload.

The LoRA was merged in FP32 before the main BF16 conversion. On one repeated implementation-check clip:

- PyTorch FP32, BF16 and CTranslate2 FP32/INT8 produced the same text.
- PyTorch dynamic INT8 produced a different text, increasing that clip's normalized WER from 5.88% to 11.76%; it was not used for the main suite.
- BF16 repeated inference took 32.81 and 33.76 seconds. Encoder time was 23.49 and 24.28 seconds, so cold initialization or disabling the decoder cache did not explain most of the cost. Decoder KV caching was enabled.

These are single-clip controls, **not precision-equivalence evidence**. CTranslate2 used one thread while the main BF16 runs used three; the machine had other active CPU workloads. Timing must not be used as a speed comparison. The main BF16 runs reached about 4.87 GiB process peak RSS, including model loading and adapter merging, not an isolated steady-state footprint. See [precision_smoke.json](precision_smoke.json), [repeat_smoke.json](repeat_smoke.json), and [ct2_conversion.json](ct2_conversion.json).

## Reproduction and scope

Lab-only packages are pinned in [requirements.txt](requirements.txt). Install the CPU build `torch==2.9.1+cpu` from the PyTorch CPU index before installing the remaining requirements. Model snapshots must match the revisions and SHA-256 values in `provenance.json`; the runner checks the weight hashes.

From the repository root, with the two pinned snapshots under `base/` and `adapter/`:

```sh
ALT_PY="$HOME/.cache/gigastt-alternative/venv312/bin/python"
ALT_MODELS="$HOME/.cache/gigastt-alternative/models"
"$ALT_PY" scripts/benchmark_alternative_asr.py \
  --models "$ALT_MODELS" \
  --manifest benchmark/results/multilingual_public_20261001/kazakh_manifest.json \
  --limit 5 --conditions original simulated_telephone \
  --dtype bfloat16 --threads 3 --beams 5 \
  --output /tmp/kazakh_first5_bf16.json
```

Phone manifests use the same `samples` schema and must retain their restriction/provenance fields. Use a `/tmp` output for detailed phone runs, or `--summary-only` for aggregate publication. The runner rejects LDC-tagged detailed outputs inside the repository. `scripts/convert_alternative_asr.py` exports the pinned merged FP32 model for the optional CTranslate2 control; no CTranslate2 production integration was made.

Completed: five original/degraded pairs, three full telephone channels from two calls, and separately labelled single-clip precision/runtime controls. **Not run:** the remaining 26 public clips, the 48 oracle telephone cuts, dedicated silence controls, Uzbek alternative benchmarks, or production integration. The full alternative suite was narrowed because of measured CPU cost; results are not represented as a completed 31-clip benchmark.

Validation: both new scripts passed `py_compile`; the restrictive-output error path was exercised and created no output file; all ten paired audio hashes/references matched; both main runs have `completed=true`. Workspace Rust checks are handled in the parent investigation.
