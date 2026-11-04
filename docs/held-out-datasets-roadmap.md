# Public dataset evaluation protocol

Evaluate recognition across different recording domains. The filename is retained
for existing links; completed dataset work and scores are recorded in the
[benchmark report](benchmarks.md), rather than duplicated as a task queue here.

Public availability does not establish that a dataset was excluded from model
training. Unless training provenance proves otherwise, describe it as an
additional public evaluation set, not a guaranteed uncontaminated holdout.

## Available datasets

The [benchmark guide](../benchmark/README.md) documents preparation and runners;
[manifests](../benchmark/manifests) fix the selected samples. Existing additional
Russian evaluations cover:

| Dataset | Domain | Preparation |
|---|---|---|
| Common Voice RU | Crowd read speech | `prepare_common_voice_ru.py` |
| FLEURS `ru_ru` | Read speech | `prepare_fleurs.py --config ru_ru` |
| RuLS | Audiobooks | `prepare_rulslib.py` |
| SOVA RuDevices | Device commands | Committed manifest |
| Podlodka Speech | Podcasts | `prepare_podlodka.py` |
| ToneWebinars | Webinars and lectures | `prepare_tone_webinars.py` |

Preparation scripts live in `scripts/`. The historical Podlodka evaluation
contains only 67 samples; report that limitation with its scores. Golos and
OpenSTT provide other recording domains, while separate multilingual and
punctuation sets answer different questions. See [languages](languages.md).

## Protocol

1. Pin the upstream dataset revision, split, selection rule and seed. Commit a
   manifest or reproducible preparation script plus sample list. Record source
   licensing in [DATA_LICENSE](../benchmark/DATA_LICENSE) and [NOTICE](../NOTICE).
2. Choose currently relevant model versions and pin the binary/source revision,
   weights and tokenizer hashes. Use identical audio samples for each candidate.
3. Fix normalization before comparing: casing, punctuation, numbers, fillers and
   `ё` handling can change WER without changing acoustic recognition. Report raw
   and normalized scores where these policies materially affect the result.
4. Measure recognition WER/CER and deletion rates, empty outputs and failures.
   Measure punctuation/casing separately. Include short commands, natural speech
   and long recordings appropriate to the intended deployment.
5. Measure speed and memory on the same host with explicit provider, threads,
   pool size, warm-up, cache state and optional processing. Separate load time
   from warm inference and ASR time from the full user-visible pipeline.
6. Publish per-sample hypotheses and timings, aggregate scores, sample count,
   audio duration and uncertainty. Use paired comparisons on identical samples;
   overlapping confidence intervals alone do not establish equivalence.

For an existing manifest, a gigastt run uses the cross-engine harness:

```sh
cd benchmark
python benchmark.py --dataset fleurs_ru --max-samples 0 \
  --runners gigastt --output results_full/fleurs_ru_gigastt.json
```

Check the benchmark guide for environment setup and runner configuration before
running. A runner name alone does not identify the selected model or processing
settings. Use the same explicit protocol for every compared runner.

## Publication requirements

`results_full/` is ignored by default. Before publishing a new headline, make its
raw evidence retrievable in a reviewed repository artifact or linked immutable
release artifact, and record its checksum. A local-only file is insufficient for
an independently verifiable claim. Preserve older dated results as historical
measurements; do not silently relabel them as results of a newer binary or model.

Update the [benchmark report](benchmarks.md) with provenance, limitations and a
link to raw evidence. Do not promote a model on punctuation alone or a speed
improvement without checking recognition quality. File recognition and live
WebSocket behavior require separate measurements.
