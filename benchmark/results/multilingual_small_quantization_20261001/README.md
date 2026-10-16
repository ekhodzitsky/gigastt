# Multilingual small quantization without training

Status: candidate conversion, native screening, full product benchmark, independent audits, and resource comparison complete.

All candidates use existing, pinned pretrained weights. No training, fine-tuning,
calibration dataset, gradients, adapters, or audio preprocessing changes are used.
The installed production model and earlier benchmark artifacts are unchanged.

## Reproducible conversion

The existing UINT8 per-tensor model was reproduced byte-for-byte from the cached
FP32 ONNX. Each of three candidate conversions was repeated with identical output
SHA-256. See `graph_audit.json`, `build_validation.json`, and individual candidate
reports for source identity, tensor checks, versions, graph structure, and sizes.

The conversions preserve the original FP32 convolution weights and quantize the
constant MatMul weights. The first preserves the exact original MatMul quantization;
the others use signed per-channel full or reduced range. Candidate files are about
319–320 MB, versus 225 MB for the original (decimal units).

## Frozen native screen

`protocol.json` was written before candidate inference. The same 37 recordings from
the prior diagnosis supply 111 original/A-law/mu-law inputs: 22 selected failure
recordings and 15 controls. These are a deliberately biased diagnostic subset,
not a representative language benchmark. No feature or resampling changes were made.

Every candidate runs the same frozen native product RuntimeFactory, three intra-op
threads, and one inter-op thread. Model, probe, input feature, protocol, and tool
identities are retained. Python ONNX Runtime is used for conversion only, never
as a substitute for native product quality measurements.

| Candidate | Failure telephone word errors | Failure telephone deletions | Empty telephone answers | Control errors: original / A / mu | Failure original errors |
|---|---:|---:|---:|---|---:|
| Original INT8 baseline | 836 | 750 | 8 | 22 / 29 / 26 | 80 |
| Original U8 MatMul, FP32 Conv | 337 | 205 | 2 | 22 / 28 / 25 | 80 |
| Per-channel S8 full range, FP32 Conv | 376 | 235 | 2 | 22 / 29 / 26 | 80 |
| Per-channel S8 reduced range, FP32 Conv | 325 | 200 | 2 | 22 / 29 / 25 | 80 |

All candidates have no new empty or catastrophic (at least 80% reference-word
deletions) ID/condition pairs in this screen. The reduced-range candidate is first
under the prespecified quality tie-break. Independent `screen_validation.json`
verified all 333 native outputs and all gates. For the selected candidate, 85.51%
of original initializer elements are quantized; INT8 weights occupy 59.07% of all
stored initializer/constant tensor bytes, using a conservative denominator.
Passing this screen alone does not justify replacing the model.

The two remaining empty outputs are A-law and mu-law versions of Kyrgyz
`fleurs_ky_kg_test_0578`. The prior diagnosis also found both empty in the original
FP32 source model; changing quantization alone has not recovered this recording.

## Full product validation

All 15,000 native HTTP recognitions completed with zero failed requests. The
independent audit recalculated both candidate and baseline scores (30,000 records),
verified model/audio/manifest/binary/scorer identities, and independently joined
all 111 screened HTTP texts to the native screening results. See
`full_validation.json` and `execution_completion.json`.

WER (%), original small INT8 → selected candidate, 1,000 recordings per cell:

| Language | Original audio | Simulated A-law | Simulated mu-law |
|---|---:|---:|---:|
| Russian | 9.97 → 9.98 | 10.72 → 10.59 | 10.52 → 10.42 |
| English | 16.21 → 16.13 | 22.57 → 21.97 | 19.44 → 19.40 |
| Kazakh | 12.32 → 12.33 | 18.24 → 13.84 | 16.38 → 13.48 |
| Kyrgyz | 13.68 → 13.75 | 16.03 → 15.55 | 14.91 → 14.67 |
| Uzbek | 16.75 → 16.71 | 18.80 → 18.45 | 18.86 → 18.40 |

Every telephone cell has lower point-estimate WER. Across 10,000 telephone inputs:
word errors fell from 31,140 to 29,440; deletions from 3,142 to 1,786; empty outputs
from eight to two; catastrophic deletion outputs (at least 80% of reference words)
from 35 to five. No new empty or catastrophic ID/condition pairs occurred.
Deletions do not improve in every cell: English mu-law increased from 301 to 318.

Across 5,000 original recordings, word errors changed from 12,875 to 12,864, with
small per-language regressions in Russian, Kazakh, and Kyrgyz. This is a telephone
robustness improvement, not a uniform improvement on clean speech.

Kazakh telephone WER improves outside the screening population too: −2.858 points
for A-law and −1.591 for mu-law. For all 1,000 Kazakh recordings, paired utterance
bootstrap 95% intervals for the WER change are [−5.400, −3.381] and
[−3.805, −2.043] percentage points, respectively. Some small changes on other
languages have intervals spanning zero. These intervals are descriptive, not
multiplicity-adjusted or independent-speaker estimates.

### Digit-free reference subset

WER (%) excluding recordings whose supplied reference contains digits; each cell
retains its original eligibility rule. No text-normalization change was introduced.

| Language | Original audio | Simulated A-law | Simulated mu-law |
|---|---:|---:|---:|
| Russian | 6.44 → 6.48 | 7.17 → 7.05 | 6.93 → 6.84 |
| English | 13.03 → 12.98 | 19.52 → 18.93 | 16.21 → 16.25 |
| Kazakh | 7.27 → 7.26 | 13.21 → 8.78 | 11.24 → 8.55 |
| Kyrgyz | 8.71 → 8.80 | 11.00 → 10.45 | 9.86 → 9.56 |
| Uzbek | 11.92 → 11.85 | 14.01 → 13.80 | 13.92 → 13.63 |

## Resource comparison

The same frozen binary ran baseline/candidate/candidate/baseline (ABBA), with
15 fixed inputs per round (one recording per language under each condition), two
excluded warmups per launch, pool size two, and three encoder threads. All 60
measured requests completed. See `resources.json` for per-request and per-round
measurements, input hashes, environment, startup, and model identities.

| Metric | Original small INT8 | Selected candidate |
|---|---:|---:|
| Model file, decimal MB | 224.8 | 320.4 |
| Aggregate single-job request RTF | 0.04792 | 0.03476 |
| Aggregate server CPU RTF | 0.09851 | 0.08034 |
| Peak sampled process PSS, MiB | 792.5 | 1078.4 |
| Peak sampled process RSS, MiB | 796.5 | 1082.6 |

Request wall time was 27.5% lower and CPU time 18.5% lower; peak sampled PSS was
36.1% higher. These are short, host-specific Linux CPU measurements under unrelated
host load, not an isolated throughput result or a cross-platform speed guarantee.
RSS/PSS were sampled every 200 ms and may miss transient peaks. Fresh optimized
caches were used per launch, but OS page cache was uncontrolled; startup measurements
are recorded separately. One excluded readiness attempt performed zero inference;
all four successful rounds followed the same corrected readiness rule. All benchmark
servers were stopped after completion.

## Interpretation and availability

This candidate is a working, separately packaged small-model option, not a new
trained model. The existing large model still has lower WER in all 15 corresponding
baseline cells. For these read-speech and simulated-channel inputs, large remains the more
accurate choice. This candidate improves small-model robustness to the simulated
channel at an increased model size. Subsequent
[real-call controls](../multilingual_real_calls_20261001/README.md) did not reproduce
the full-call improvement and do not establish call-center readiness for either model. Speed and memory trade-offs are reported separately above.

The candidate is available locally through the isolated pack:

```sh
~/.cache/gigastt-multilingual-1000/gigastt-candidate --offline serve \
  --model-dir ~/.cache/gigastt-small-quantization/selected_product_pack \
  --model-variant ml_ctc --execution-provider cpu \
  --pool-size 2 --encoder-intra-threads 3 --punctuation off --itn off
```

`manifest.toml` selects `lab_multilingual_ctc.int8.onnx`; its SHA-256 is
`18fa5fab6ece0123d7813f7abc8da129530f6d1a7da17c85c4b3948f27b8e3e0`.
The pack also contains a symlink to the unchanged original pinned encoder only
because the existing CLI presence check expects its standard filename. The
manifest-selected candidate is the actual loaded encoder, verified through logs
and exact HTTP/native transcript parity. No download/checksum enforcement code
was changed. This is an experimental local pack, not a published model release.
The production installation remains unchanged.

## Reproduction

Conversion: `scripts/build_small_quantization.py` (isolated conversion environment,
pinned existing FP32 weights; refuses to overwrite candidates).
Screening: `scripts/screen_small_quantization.py` and independent
`scripts/audit_small_quantization.py`.
Full product run: `benchmark/.venv/bin/python scripts/benchmark_small_quantization.py --mode parity-and-full`.
Independent full audit: `scripts/audit_small_quantization_full.py`.
Resource comparison: `scripts/measure_small_quantization_resources.py`.
Individual JSON reports record exact source/model/tool identities and versions.

## Limits

The corpus and all scoring rules are the same as the earlier
[1,000-per-language benchmark](../multilingual_1000_20261001/README.md) and
[simulated telephone benchmark](../multilingual_telephony_20261001/README.md).
These are FLEURS read recordings with simulated telephone codecs, not real calls.
The 37 screened versus 4,963 outside-screening recordings are reported separately;
the latter are not a blind holdout because the full baseline had already been
inspected. Apostrophe-removing normalization can forgive Uzbek spelling errors.

Native validation is on this Linux CPU and frozen runtime build; it does not
establish identical behavior on other execution providers or target platforms.
No default model replacement or release was performed.

Repository checks: 1,117 unit tests passed, 39 ignored, zero failures; clippy and
format checks passed. No product code was changed for these candidate conversions.

## Standalone packaging follow-up

A reproducible archive and verified installer are now available; see
[standalone model packs](../../../docs/model-packs.md). The updated CLI accepts
the standalone manifest without the original encoder companion. The earlier
benchmark pack and binary above remain frozen for reproducibility.
