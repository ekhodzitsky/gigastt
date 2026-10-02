# Five-language multilingual benchmark — 1,000 recordings per language

Evaluation of the original `ml_ctc` and `ml_ctc_large` INT8 models through
**gigastt**, on Russian, English, Kazakh, Kyrgyz and Uzbek. No model training,
adaptation, checkpoint selection or language-specific output correction is
part of this experiment. All **10,000 measured requests completed**: 5,000
recordings through each model, with **zero request failures and zero empty
hypotheses**. The selected audio totals 60,992.4 seconds (16.94 hours).

## Results

Exactly 1,000 recordings per language and model; lower WER/CER is better.

| Language | Small WER | Large WER | Small CER | Large CER |
|---|---:|---:|---:|---:|
| Russian | 9.97% | 8.78% | 4.71% | 4.50% |
| English | 16.21% | 13.62% | 6.98% | 5.93% |
| Kazakh | 12.32% | 11.47% | 4.59% | 4.47% |
| Kyrgyz | 13.68% | 12.34% | 5.04% | 4.69% |
| Uzbek | 16.75% | 13.89% | 4.89% | 4.27% |

Large has lower aggregate WER than small on each of these five fixed subsets.
This read-speech evidence does not support a blanket claim of extremely poor
Kazakh or Uzbek recognition. It does not resolve the separate telephone-call
failures or establish quality across all domains. These are INT8 gigastt
pipeline results, not a reproduction of the publisher's FP32 benchmark.

## Supplementary digit-free subset

The primary table above includes all 1,000 recordings. These subsets remove
entire references containing digits, without correcting model hypotheses.
They expose a text-formatting confound but do not replace the headline scores.

| Language | Recordings | Small WER | Large WER |
|---|---:|---:|---:|
| Russian | 815 | 6.44% | 5.13% |
| English | 834 | 13.03% | 10.49% |
| Kazakh | 818 | 7.27% | 6.49% |
| Kyrgyz | 816 | 8.71% | 7.35% |
| Uzbek | 819 | 11.92% | 9.27% |

Machine-readable aggregate: [summary.json](summary.json). Each full result
contains per-recording outputs, S/D/I counts and test/validation summaries:

- [ml_ctc_ru.json](ml_ctc_ru.json)
- [ml_ctc_large_ru.json](ml_ctc_large_ru.json)
- [ml_ctc_en.json](ml_ctc_en.json)
- [ml_ctc_large_en.json](ml_ctc_large_en.json)
- [ml_ctc_kk.json](ml_ctc_kk.json)
- [ml_ctc_large_kk.json](ml_ctc_large_kk.json)
- [ml_ctc_ky.json](ml_ctc_ky.json)
- [ml_ctc_large_ky.json](ml_ctc_large_ky.json)
- [ml_ctc_uz.json](ml_ctc_uz.json)
- [ml_ctc_large_uz.json](ml_ctc_large_uz.json)

## Fixed protocol

- Google FLEURS, CC BY 4.0, Conneau et al. (2022), pinned parquet revision
  `168de341b3db6859a9bac1c50a2ef5e3b47647e0`.
- Exactly 1,000 distinct audio recordings per language, all eligible test
  recordings followed by a seeded validation top-up. No train data or repeated
  audio padding. Both models receive exactly the same recordings/references.
- Original 16 kHz audio, no simulated telephone condition, no VAD, punctuation
  or ITN. CPU servers use two pool slots and three encoder threads per slot;
  two measured requests may run concurrently per model.
- Primary metric: normalized single-reference **micro WER** over all 1,000
  recordings. CER includes normalized spaces. Full substitution/deletion/
  insertion counts, empty hypotheses, failures, duration and split-level
  results are retained. A failed request is counted as an empty hypothesis,
  not silently removed.
- Digit-free and digit-free/at-most-30-second subsets are supplementary and
  must not replace the 1,000-recording headline. The normalizer preserves all
  language letters but removes punctuation/apostrophes and does not perform
  number conversion. See [methodology](methodology.md) for exact caveats.
- These are read-speech results, **not telephone-call quality measurements**.
  The test+validation mixture is not the official test-only benchmark. Model
  pretraining overlap is unknown; recording counts are not speaker counts.

Per-language manifest and provenance files contain selected splits, source
row IDs, sentence IDs, references, audio hashes, source file hashes and URLs.
Audio stays in the local dataset cache. The installed original model hashes,
actual execution providers, binary hash and server commands are recorded in
[environment.json](environment.json) and every result's metadata.

## Reproduce

Install the existing benchmark Python environment with `pyarrow`, `numpy`
and `jiwer`; FFmpeg/FFprobe must be available. Prepare pinned source files and
manifests using `scripts/prepare_multilingual_1000.py --help`.

Start an original-model server, for example:

```sh
/path/to/gigastt --offline serve --model-variant ml_ctc \
  --execution-provider cpu --pool-size 2 --encoder-intra-threads 3 \
  --punctuation off --itn off --port 19881
```

For each language `en`, `ru`, `kk`, `ky`, `uz`, run:

```sh
benchmark/.venv/bin/python scripts/benchmark_multilingual_1000.py \
  --manifest benchmark/results/multilingual_1000_20261001/en_manifest.json \
  --variant ml_ctc --url http://127.0.0.1:19881 --workers 2 --fsync \
  --binary /path/to/gigastt \
  --model "$HOME/.gigastt/models/multilingual_ctc.int8.onnx" \
  --model "$HOME/.gigastt/models/multilingual_vocab.txt" \
  --output "$HOME/.cache/gigastt-multilingual-1000/runs/ml_ctc_en.json"
```

Repeat for `ml_ctc_large` with its original
`multilingual_large_ctc.int8.onnx` encoder and a matching server. Output JSONL
ledgers and metadata allow interruption recovery; changed input, model, binary,
scorer or runtime identity rejects resume. Completed JSON files contain the
entire measured results and summaries; only these final files are copied into
this directory. Timing reflects concurrent CPU workloads and is not a fair
model throughput comparison.

The runner's failure, recovery and identity protections have an actual HTTP
smoke test: `benchmark/.venv/bin/python scripts/test_benchmark_multilingual_1000.py`.
Evidence: [runner_smoke.json](runner_smoke.json).

## Validation and interruption recovery

The dataset audit reconstructed fixed selection from pinned parquet files and
verified all 5,000 audio files. Source training data was never selected. Model
checksums match the original unadapted encoders and shared vocabulary.

A runtime interruption removed temporary control files and stopped processes;
the persistent ledger retained 4,838 small and 2,280 large results. The runner
resumed only after matching manifest, audio, model, binary, scorer and runtime
identities. Completed records were not discarded. After small completed, a
second equally configured large server processed Russian while the first
finished Kazakh and then Kyrgyz. This does not provide a controlled speed test.

Verification during this task: workspace unit tests 1,117 passed / 39 ignored;
`cargo clippy`, `cargo fmt --check`, Python syntax and `git diff --check` passed.
The real-HTTP interruption/resume and error-accounting smoke test passed again
after continuation. No commits, model replacement or training were performed.

The final [independent audit](validation_summary.json) passed: all 10,000 word
and character distances were recomputed, with 500 additional Python character
DP checks. All summary counters and durations agree. The artifact includes
seed-42, 1,000-resample, utterance-bootstrap 95% WER intervals; repeated
sentences/speakers mean these are selected-set descriptive intervals, not
speaker-independent or language-wide confidence bounds.

```sh
benchmark/.venv/bin/python scripts/audit_multilingual_1000.py \
  --directory benchmark/results/multilingual_1000_20261001 \
  --cache "$HOME/.cache/gigastt-multilingual-1000" \
  --vocab "$HOME/.gigastt/models/multilingual_vocab.txt" \
  --results benchmark/results/multilingual_1000_20261001/ml_ctc*.json \
  --output benchmark/results/multilingual_1000_20261001/validation_summary.json
```
