# Public multilingual ASR pilot — 2026-10-01

The tested gigastt 2.21.0 binary recognizes natural Kazakh/Russian mixed speech and
Uzbek speech. It outperforms the archived comparison systems on the small original
Kazakh mixed-speech set under that dataset's scoring protocol. This does **not**
establish superiority over the Habr telephone ensemble: narrowband inputs degrade
sharply, and both heads fail on the single real Kazakh telephone demonstration.

## Scope and provenance

- **Kazakh/Russian natural speech:** all 31 clips, 221.93 seconds, from
  [Timur Seidalin's benchmark](https://github.com/Tim2190/Kaz-ASR-codeswitch-benchmark),
  revision `6575c5cd9e9e481edcb4d9928fda2b2f60d3e5ad`. Stand-up and interview speech;
  11 clips carry the Russian-insertion flag. Audio CC BY according to the publisher;
  code/annotations MIT. Original channel/video attribution is retained in
  `kazakh_manifest.json`. These are not telephone calls.
- **Uzbek read control:** 100 of 862 FLEURS test rows, selected before inference with
  `random.Random(42).sample(range(862), 100)`. 1,227.24 seconds. Dataset
  [google/fleurs](https://huggingface.co/datasets/google/fleurs), `uz_uz`, CC BY 4.0,
  Conneau et al. (2022). Exact downloaded Parquet SHA-256 and row indices are saved.
- **Uzbek conversational speech:** 100 of 745 rows, seed 42, from
  [BoburAmirov/asr_evaluate_set](https://huggingface.co/datasets/BoburAmirov/asr_evaluate_set),
  revision `c6e1b4f66d45e9d7644003d1564837786295850a`. Public Telegram voice messages
  with manual references according to the publisher; Apache-2.0 as declared on the
  dataset card. 758.78 seconds. The sole published split is named `train`; it is not
  a separately held-out test split. Voice messages are not phone calls, and this
  dataset does not establish coverage of Uzbek/Russian switching within phrases.
- **Actual Kazakh telephone speech:** one 110.04-second public demonstration from
  [LDC2018S13 / IARPA Babel Kazakh](https://catalog.ldc.upenn.edu/LDC2018S13).
  Original sample: 8 kHz A-law SPHERE, decoded losslessly to 8 kHz PCM WAV with
  ffmpeg before submission. Timestamp and nonlexical event tags are removed from
  its reference. The scored reference has 77 words. Raw sample audio and transcript
  stay in the local cache, because the full corpus has a separate LDC license.
  Only summary metrics and source checksums are included here. One call cannot
  estimate a population WER.

No audio was sent to cloud recognition APIs. Models and public data were downloaded;
all new recognition ran locally. The article's private recordings, specialized
Uzbek model and complete selection/filtering pipeline were unavailable.

## Kazakh mixed speech: like-for-like scoring on original audio

The pinned upstream scorer evaluates both verbatim and normalized references,
normalizes digit runs to Kazakh words, chooses the lower error per clip, and reports
macro `WER_best` / `CER_best`. This is an optimistic dual-reference metric; it must
not be compared directly to a single-reference, micro-averaged WER from another
corpus. We reran the scorer on all systems' hypotheses with identical settings.
Competitor recognition was **not** rerun; those hypotheses were published upstream.

| System | Macro WER_best % | Macro CER_best % | Evidence |
| --- | ---: | ---: | --- |
| ml_ctc_original | 6.62 | 1.86 | Measured locally |
| ml_ctc_large_original | 7.04 | 2.14 | Measured locally |
| hf_shyngys879_kazakh-whisper-large-v3-turbo | 11.87 | 3.77 | Archived upstream hypotheses, rescored locally |
| yandex_stt | 17.13 | 6.51 | Archived upstream hypotheses, rescored locally |
| gemini_gemini-2.5-flash | 29.46 | 12.54 | Archived upstream hypotheses, rescored locally |
| whisper_local_large-v3 | 42.55 | 12.98 | Archived upstream hypotheses, rescored locally |
| google_stt | 64.86 | 38.15 | Archived upstream hypotheses, rescored locally |

On the 11 Russian-insertion clips, macro WER_best is **7.17%** for ml_ctc,
**8.33%** for ml_ctc_large, **18.17%** for the archived Kazakh Whisper fine-tune,
and **17.69%** for archived Yandex. The unit is the full flagged clip; these are
not error rates computed on Russian words alone.

For transparency, the single-reference micro WER of original Kazakh audio is
**16.54% / 16.54%** against verbatim spelling, and **6.81% / 6.62%** against
normalized spelling (small / large). Reference morphology changes the number.
The small-vs-fine-tuned-Whisper paired clip-bootstrap difference is -5.25 percentage
points, percentile 95% interval [-9.16, -1.27]. Clips share sources/speakers, so
this interval is not a population-level or speaker-independent guarantee.

## Uzbek recognition

Unicode-preserving lowercase/punctuation/apostrophe normalization from
`scripts/wer_unicode.py`, no number conversion, micro pooled edit counts.
All selected rows are retained, including empty hypotheses and numeric references.
CER includes spaces. Full hypotheses, reference text, edit counts and durations
are saved in the result JSON files.

| Dataset | Clips | ml_ctc WER / CER % | ml_ctc_large WER / CER % |
| --- | ---: | ---: | ---: |
| FLEURS read speech | 100 | 16.76 / 5.11 | 14.34 / 4.60 |
| Telegram conversational voice messages | 100 | 25.79 / 9.16 | 22.99 / 8.24 |

The FLEURS and Telegram rows are different domains; do not pool them into one
headline accuracy. Ten selected Telegram references contain digits, while the
CTC outputs commonly spell numbers out. This is a known formatting contribution
to the reported full-set WER; no digit-bearing rows were excluded.

## Narrowband sensitivity

The simulated condition uses ffmpeg mono mixing, `highpass=f=300,lowpass=f=3400`,
8 kHz resampling and G.711 mu-law WAV encoding. Filters use ffmpeg defaults; this
is a controlled degradation, not a real telephone-network or MP3 simulation.
The `external_16k` control decodes/resamples that **same degraded audio** to 16 kHz
PCM using ffmpeg before gigastt receives it. It does not restore lost bandwidth.

The following table uses single-reference **micro** WER, normalized-written
references for Kazakh and the original FLEURS references for Uzbek. It uses the
repository's Unicode scorer rather than upstream's dual-reference macro metric.

| Dataset | Input | ml_ctc WER % | ml_ctc_large WER % |
| --- | --- | ---: | ---: |
| Kazakh/Russian natural clips | original | 6.81 | 6.62 |
| Kazakh/Russian natural clips | simulated_telephone | 93.95 | 85.26 |
| Kazakh/Russian natural clips | simulated_telephone_external_16k | 46.69 | 36.67 |
| Uzbek FLEURS | original | 16.76 | 14.34 |
| Uzbek FLEURS | simulated_telephone | 58.81 | 23.47 |
| Uzbek FLEURS | simulated_telephone_external_16k | 18.58 | 14.98 |

Direct 8 kHz input is much worse than the external 16 kHz control. This motivates
an investigation of the 8→16 kHz ingestion/resampling and acoustic-feature path;
it does not by itself prove a particular implementation bug. Even external
resampling leaves substantial Kazakh degradation. No competitor was measured on
the degraded audio, so these rows cannot rank competing systems.

## Real telephone demonstration

Both variants mostly omit the words in this sample. The external resampling
control does not rescue recognition. VAD was off for all runs, matching the fixed
configuration; tuning VAD or segmentation was not part of this pilot.

| Input | ml_ctc WER % | ml_ctc_large WER % |
| --- | ---: | ---: |
| original | 97.40 | 96.10 |
| original_external_16k | 96.10 | 98.70 |

This is a concrete failing example for the tested binary, not an estimate over
Kazakh calls in general. Resolving it and validating on more genuine calls is
necessary before claiming telephone readiness or superiority to the Habr system.

## Runtime and configuration

CPU: AMD Ryzen AI 9 HX 370, Linux x86_64, one inference slot, six encoder threads,
INT8, punctuation off, ITN off, no VAD. Fresh servers per model; one unmeasured
warm-up before each dataset. HTTP request timing includes file submission and
response parsing but excludes model startup, audio preparation and external
ffmpeg conversion. These are pilot latency measurements on a workstation,
not isolated throughput/load-test results. No competitor runtime was measured.

| Original audio | Audio seconds | ml_ctc wall seconds | large wall seconds | RTF small / large |
| --- | ---: | ---: | ---: | ---: |
| Kazakh/Russian natural | 221.93 | 6.04 | 13.20 | 0.027 / 0.059 |
| Uzbek FLEURS | 1227.24 | 33.15 | 79.96 | 0.027 / 0.065 |
| Uzbek Telegram | 758.78 | 25.47 | 50.75 | 0.034 / 0.067 |

Linux process VmHWM was about 479 MiB for small and 2,246 MiB for large in these
runs; this includes mapped model pages and is not macOS resident/dirty footprint.
`environment.json` contains exact process readings, model hashes and binary hash.
The release binary predates this work and reports 2.21.0; its exact source revision
is **unknown**. The current checkout revision is recorded separately and must not
be mistaken for a proven build provenance of that binary.

## Reproduction

Run from the repository root. The scripts use the existing benchmark environment
plus jiwer; dependency versions are recorded in `environment.json`. Downloaded
audio stays outside the repository. The cached source repositories/files are
revision- or SHA-256-pinned; no remote Python dataset loader is executed.

```sh
uv pip install --python benchmark/.venv/bin/python jiwer==4.0.0
benchmark/.venv/bin/python scripts/prepare_multilingual_public.py \
  --cache /tmp/gigastt-multilingual-audit \
  --output benchmark/results/multilingual_public_20261001
benchmark/.venv/bin/python scripts/prepare_multilingual_conversation.py \
  --cache /tmp/gigastt-multilingual-audit \
  --output benchmark/results/multilingual_public_20261001

gigastt download --model-variant ml_ctc
gigastt download --model-variant ml_ctc_large
```

Start one server in a separate terminal, replacing MODEL with `ml_ctc` or
`ml_ctc_large`; use the recorded binary/model hashes when reproducing this exact run:

```sh
target/release/gigastt --offline serve --model-variant MODEL \
  --pool-size 1 --encoder-intra-threads 6 --punctuation off --itn off --port 19876
```

Run each dataset serially against that server (replace MODEL in output and variant):

```sh
benchmark/.venv/bin/python scripts/benchmark_multilingual_public.py \
  --manifest benchmark/results/multilingual_public_20261001/kazakh_manifest.json \
  --variant MODEL --output benchmark/results/multilingual_public_20261001/kazakh_MODEL.json
benchmark/.venv/bin/python scripts/benchmark_multilingual_public.py \
  --manifest benchmark/results/multilingual_public_20261001/uzbek_manifest.json \
  --variant MODEL --output benchmark/results/multilingual_public_20261001/uzbek_MODEL.json
benchmark/.venv/bin/python scripts/benchmark_multilingual_public.py \
  --manifest benchmark/results/multilingual_public_20261001/telegram_manifest.json \
  --variant MODEL --output benchmark/results/multilingual_public_20261001/telegram_MODEL.json \
  --conditions original
benchmark/.venv/bin/python scripts/benchmark_multilingual_public.py \
  --manifest /tmp/gigastt-multilingual-audit/telephone_manifest.json \
  --variant MODEL --output /tmp/gigastt-multilingual-audit/telephone_MODEL.json \
  --conditions original original_external_16k
```

Stop the server before starting the other variant. Large was measured on port
19877 and the supplementary small runs on 19878 via `--url`; port choice does not
change recognition. Rescore the Kazakh comparisons after both runs:

```sh
benchmark/.venv/bin/python scripts/score_public_kazakh.py \
  --source /tmp/gigastt-multilingual-audit/kaz-codeswitch \
  --results benchmark/results/multilingual_public_20261001
```

## Evidence boundaries and next experiment

These results establish actual multilingual recognition and a promising small-set
Kazakh mixed-speech comparison. They do not establish that gigastt replaces the
article's full speech-analytics product or wins on its private telephone corpus.
The urgent next experiment is to isolate the 8 kHz failure with matched PCM inputs,
then retest real calls with manual references and the same Whisper ensemble.
Long bilingual conversations, Uzbek/Russian within-phrase switching, business
number/name accuracy, diarization and streaming accuracy remain unmeasured here.
Neither training-set overlap nor the public references' annotation quality has
been independently audited. Preserve these caveats when citing the numbers.

## Validation

Verified all 990 scored recognition requests across 232 unique recordings, with
no missing sample/condition pairs. Recomputed the stored edit counts and summaries.
Kazakh alphabet retention, Uzbek apostrophe folding, empty-output deletions,
insertions and micro aggregation passed focused scoring checks.

`cargo test --workspace --lib --bins && cargo clippy` passed (1115 tests passed, 39 ignored).
`cargo fmt --check`, Python compilation and `git diff --check` passed.
Missing OpenSSL development headers were supplied from a locally extracted
package under `/tmp`, without changing system packages. See `validation.json`.
Production Rust code was not modified. These checks do not replace model-dependent
integration tests or establish the build provenance of the measured release binary.

Scenario motivating this pilot: [Habr article 1088214](https://habr.com/ru/articles/1088214/).
