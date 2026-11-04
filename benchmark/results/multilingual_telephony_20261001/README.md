# Paired multilingual telephone-channel benchmark

**Completed: 20,000 telephone recognitions, independently audited against 10,000 original-audio recognitions.** Exactly 1,000 fixed recordings per language, two original INT8 models, and two codecs. There were zero failed requests. Small returned eight empty hypotheses (Kazakh: three per codec; Kyrgyz: one per codec); large returned none. Empty hypotheses remain scored. No training or model replacement was performed.

The experiment reuses the exact 5,000 recordings and references from the [original-audio benchmark](../multilingual_1000_20261001/README.md). It measures simulated telephone degradation of FLEURS read speech, not accuracy on real calls.

## Paired accuracy

All scores are corpus-level percentages on the complete 1,000 recordings. Lower is better. Deltas are percentage points computed before rounding.

| Language | Model | Original WER | A-law WER | Delta | Mu-law WER | Delta |
|---|---|---:|---:|---:|---:|---:|
| Russian | small | 9.97 | 10.72 | +0.76 | 10.52 | +0.55 |
| Russian | large | 8.78 | 9.07 | +0.29 | 9.05 | +0.27 |
| English | small | 16.21 | 22.57 | +6.35 | 19.44 | +3.22 |
| English | large | 13.62 | 17.31 | +3.69 | 15.57 | +1.95 |
| Kazakh | small | 12.32 | 18.24 | +5.92 | 16.38 | +4.06 |
| Kazakh | large | 11.47 | 12.15 | +0.68 | 11.95 | +0.48 |
| Kyrgyz | small | 13.68 | 16.03 | +2.34 | 14.91 | +1.22 |
| Kyrgyz | large | 12.34 | 13.71 | +1.37 | 13.08 | +0.74 |
| Uzbek | small | 16.75 | 18.80 | +2.05 | 18.86 | +2.11 |
| Uzbek | large | 13.89 | 14.56 | +0.66 | 14.65 | +0.75 |

Large has lower telephone WER and smaller channel penalties than small in every matched language/codec condition here. The largest losses are English small A-law (+6.35 pp) and Kazakh small A-law (+5.92 pp). Kazakh large changes by only +0.68/+0.48 pp and Uzbek large by +0.66/+0.75 pp. These results do not support a blanket claim that the models fail to recognize Kazakh or Uzbek. They do not establish real-call quality or superiority over another product.

| Language | Model | Original CER | A-law CER | Mu-law CER |
|---|---|---:|---:|---:|
| Russian | small | 4.71 | 4.89 | 4.83 |
| Russian | large | 4.50 | 4.56 | 4.55 |
| English | small | 6.98 | 10.20 | 8.54 |
| English | large | 5.93 | 8.06 | 7.03 |
| Kazakh | small | 4.59 | 8.36 | 7.11 |
| Kazakh | large | 4.47 | 4.59 | 4.51 |
| Kyrgyz | small | 5.04 | 5.66 | 5.34 |
| Kyrgyz | large | 4.69 | 5.08 | 4.86 |
| Uzbek | small | 4.89 | 5.73 | 5.79 |
| Uzbek | large | 4.27 | 4.55 | 4.58 |

## Error changes and numeric-reference diagnostics

S/D/I means substitutions/deletions/insertions, using the same jiwer alignment in both conditions. These are changes in error counts, not percentages. Independent dynamic programming verifies total word-edit distances; alternate optimal alignments can distribute S/D/I differently.

| Language | Model | A-law delta S / D / I | Mu-law delta S / D / I |
|---|---|---:|---:|
| Russian | small | +117 / +21 / +6 | +85 / +13 / +7 |
| Russian | large | +43 / +6 / +6 | +41 / +2 / +8 |
| English | small | +1225 / +159 / -9 | +661 / +52 / -16 |
| English | large | +697 / +152 / -50 | +345 / +137 / -59 |
| Kazakh | small | +280 / +832 / -97 | +177 / +572 / -53 |
| Kazakh | large | +111 / +13 / -7 | +81 / +11 / -10 |
| Kyrgyz | small | +360 / +63 / -17 | +187 / +35 / -10 |
| Kyrgyz | large | +198 / +26 / +13 | +115 / +5 / +9 |
| Uzbek | small | +270 / +93 / +2 | +273 / +102 / +0 |
| Uzbek | large | +103 / +15 / +0 | +111 / +28 / -5 |

Substitutions dominate most increases. Kazakh small is the exception: A-law adds 832 deletions and mu-law adds 572. Both have three empty successful responses. Large has no empty hypotheses and only 13/11 additional Kazakh deletions. VAD was disabled; deletion counts alone do not prove an audio segmentation bug.

The two Kyrgyz small empty answers are the same source recording. Its original WER was already 100% from a repeated phrase; the telephone WER is also 100%, now from deletion of the whole phrase. Equal total WER can conceal a different failure mode.

WER on references containing no digits (a diagnostic subset, not the headline benchmark):

| Language | Model | N | Original | A-law | Mu-law |
|---|---|---:|---:|---:|---:|
| Russian | small | 815 | 6.44 | 7.17 | 6.93 |
| Russian | large | 815 | 5.13 | 5.37 | 5.40 |
| English | small | 834 | 13.03 | 19.52 | 16.21 |
| English | large | 834 | 10.49 | 14.32 | 12.55 |
| Kazakh | small | 818 | 7.27 | 13.21 | 11.24 |
| Kazakh | large | 818 | 6.49 | 7.13 | 6.90 |
| Kyrgyz | small | 816 | 8.71 | 11.00 | 9.86 |
| Kyrgyz | large | 816 | 7.35 | 8.70 | 8.05 |
| Uzbek | small | 819 | 11.92 | 14.01 | 13.92 |
| Uzbek | large | 819 | 9.27 | 9.90 | 10.04 |

The channel penalty persists without digits and cannot be explained solely by numeric formatting. Digit-bearing references already have higher baseline WER. The [audit](validation_summary.json) includes both subsets, full S/D/I/CER totals, and paired improved/worsened/unchanged counts. These subsets measure exposure to numeric formatting; they do not label which individual errors are numeric. Normalization removes punctuation/apostrophe variants, including potentially meaningful Uzbek apostrophes; no ITN is applied. CER includes spaces. FLEURS lacks validated code-switch boundaries, so no language-switch-specific conclusion is drawn.

## Protocol

- Fixed FFmpeg high-pass 300 Hz, low-pass 3,400 Hz, mono, 8,000 Hz, and separate
  `pcm_alaw` / `pcm_mulaw` WAV encoding. No denoising or gain normalization.
- The same binary, original model weights, shared vocabulary, CPU provider,
  pool size 2 and encoder thread count 3 per slot as the baseline. Punctuation,
  ITN and VAD are off. Two measured requests per server may run concurrently.
- Prepared file SHA, original file SHA, source ID, reference and split permit
  exact pairing. Each manifest has exactly 1,000 recordings; no sample is
  replaced based on recognition output.
- Primary scores use all 1,000 recordings. WER/CER and S/D/I are paired against
  original audio. Digit-free and digit-bearing subsets diagnose formatting
  effects; they do not replace the complete benchmark.
- Failed HTTP requests remain in the ledger and score as empty transcriptions;
  silence/empty hypotheses remain counted. Resume requires matching identities.
- This measures **controlled degradation of read speech**, not natural call
  accuracy, packet loss, echo or actual customer-service conversations.
  FLEURS does not supply validated code-switch boundaries, so this experiment
  cannot attribute errors specifically to language switching.

The scorer is unchanged. The resumable runner was minimally extended to retain
and validate each manifest's condition instead of hardcoding `original`.
Historical baseline runner hashes therefore differ; Unicode normalization,
`jiwer`, model bytes and inference settings remain comparable. A regression
smoke failed before this condition fix and passed afterward, including tampered
condition rejection and the existing interrupted-HTTP resume tests.

## Data and reproduction

See [preparation.md](preparation.md),
[telephony_dataset_provenance.json](telephony_dataset_provenance.json), and
[environment.json](environment.json). Audio and durable progress ledgers stay
under `~/.cache/gigastt-multilingual-telephony`. All 10,000 prepared WAVs were
independently decoded and checked with libsndfile before manifest publication.

Prepare with `scripts/prepare_multilingual_telephony.py`. Start each server
using the recorded commands. For a single 1,000-recording cell:

```sh
benchmark/.venv/bin/python scripts/benchmark_multilingual_1000.py \
  --manifest benchmark/results/multilingual_telephony_20261001/ru_alaw_manifest.json \
  --variant ml_ctc --url http://127.0.0.1:19891 --workers 2 --fsync \
  --binary "$HOME/.cache/gigastt-multilingual-1000/gigastt-candidate" \
  --model "$HOME/.gigastt/models/multilingual_ctc.int8.onnx" \
  --model "$HOME/.gigastt/models/multilingual_vocab.txt" \
  --output "$HOME/.cache/gigastt-multilingual-telephony/runs/ml_ctc_ru_alaw.json"
```

Repeat for every language, codec and model. Completed result files retain all
hypotheses and per-recording counts. Concurrent CPU and unrelated workloads
make these timings unsuitable for a controlled throughput comparison.

Final independent validation:

```sh
benchmark/.venv/bin/python scripts/audit_multilingual_telephony.py \
  --baseline benchmark/results/multilingual_1000_20261001 \
  --directory benchmark/results/multilingual_telephony_20261001 \
  --output benchmark/results/multilingual_telephony_20261001/validation_summary.json
```

## Official-source controls and remaining access limit

See the [official control report](../official_telephony_20261001/README.md) and [frontend/precision analysis](../official_telephony_20261001/frontend_results.md). The fixed short-API control attempts the first ten baseline recordings per language in all three conditions: 147 successful recognitions, with one 27-second Russian recording rejected in all three conditions by the official 25-second limit. No replacement or truncation is used. These are diagnostic subsets, not another 1,000-recording language benchmark.

The separate 135-execution frontend control finds identical source-FP32 and FP32-ONNX frame argmax on all 45 feature arrays. It also finds input-preparation/precision sensitivity, but Python INT8 is not exactly equivalent to the native server on every checked input. The product benchmark above uses the actual frozen server. Changing the resampler changes both sample values and sequence length in these controls; it is not an isolated test of the filter.

The unmodified official long-form pipeline was attempted on two public call recordings (three channels). Initially required segmentation weights were missing; after local HF authentication was configured, access to `pyannote/segmentation-3.0` returned HTTP 403. The [access record](../official_telephony_20261001/segmentation_access.json) preserves the current blocker and prior attempts. Thus full official long-form call recognition is **not completed**. Automatic 20-second short-API call controls are reported separately and cannot substitute for it. No private call transcripts or credentials are published.

## Verification

The independent [validation summary](validation_summary.json) confirms all 20 result files contain exactly 1,000 paired records; it recomputes WER, CER and error counts for all 30,000 original-plus-telephone recognitions and verifies source/prepared audio SHA-256, manifests, original model/vocabulary bytes, binary identity, scorer and inference settings. There are zero failed requests and eight scored empty hypotheses.

Repository checks passed: 1,117 unit tests, 39 ignored, zero failed; `cargo clippy` and `cargo fmt --check` passed. Runner resume/condition regression checks and audit smoke checks also passed. No production code or model was changed during this telephone benchmark stage; the frozen binary includes the previously measured narrowband PCM16 resampling mitigation. Detailed settings and check metadata are in [environment.json](environment.json).
