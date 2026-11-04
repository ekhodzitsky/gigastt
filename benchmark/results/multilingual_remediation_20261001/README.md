# Multilingual telephone remediation experiments — 2026-10-01

The telephone failure is present in the original small multilingual checkpoint,
not just the Rust implementation or INT8 export. The completed matched precision
audit found identical source/FP32 ONNX frame decisions on 132 inputs. INT8 adds
6.43 percentage points of simulated Kazakh WER, but the source already has
41.21% WER there. These results do not establish acceptable real-call quality.

## Evidence and scope

- [Source / FP32 ONNX / INT8 comparison](../multilingual_precision_20261001/README.md):
  identical saved features isolate export and quantization. Both the original
  frontend and the Rust frontend fail on a float-resampled narrowband control;
  PCM16 input precision and resampling behavior materially affect recognition.
- [Telephone corpus provenance](../telephone_corpora_20261001/README.md): a second
  reference-bearing Kazakh public recording was obtained. The total is **two
  recordings, three channels**, not a representative call-center benchmark.
  Their source corpora may overlap. No accessible additional Uzbek real-call
  corpus with usable human references was found in this search.
- [Earlier public pilot](../multilingual_public_20261001/README.md) and
  [PCM16 mitigation](../narrowband_20261001/README.md) retain the full original
  Kazakh/Uzbek and Russian controls. Simulated telephone audio, read speech,
  Telegram messages and real calls are separate conditions.

## Second real recording: full channels

MATERIAL public demo: two annotated channel excerpts of 56.14 s, 93 reference
words total. Micro WER sums word errors across both channels. The underlying
recording is one call. Same candidate binary and model hashes as the earlier
PCM16 mitigation report; punctuation and ITN disabled, CPU, pool size 1.

| Processing | Small INT8 | Large INT8 |
|---|---:|---:|
| Native 8 kHz, no VAD | 100.00% | 98.92% |
| Native 8 kHz, VAD | 94.62% | 95.70% |
| FFmpeg 16 kHz PCM16, no VAD | 98.92% | 97.85% |
| FFmpeg 16 kHz PCM16, VAD | 90.32% | 86.02% |

The baseline/candidate factorial results, including the previous resampler,
are in [material_comparison.json](material_comparison.json). Eight server
configurations produced 32 measured channel transcriptions. This confirms
that the PCM16 mitigation is insufficient on actual calls.

## Automatic segmentation

Predeclared nonoverlapping 10 s / 20 s windows start at sample zero, without
human timestamps, overlap or VAD. All chunk hypotheses are concatenated and
scored against the complete channel reference. Boundaries can split words.

| Recording | Small 10 s | Small 20 s | Large 10 s | Large 20 s |
|---|---:|---:|---:|---:|
| Babel demo | 77.92% | 94.81% | 81.82% | 98.70% |
| MATERIAL demo, both channels | 78.49% | 98.92% | 95.70% | 97.85% |

Artifacts: [small](automatic_segmentation_small.json),
[large](automatic_segmentation_large.json).
Reproduce with `scripts/benchmark_telephone_segmentation.py`; manifest
hashes and model/binary hashes are recorded in the artifacts. These settings
are diagnostics on already inspected recordings, not an independently selected
production configuration.

## Resampling filter control

Three filters were declared before this run. On the same 31 simulated Kazakh
clips (529 words), with small INT8 and PCM16 rounding in every condition:

| Sinc length / cutoff | WER |
|---|---:|
| Current 256 / 0.95 | 51.61% |
| 32 / 0.97 | 36.86% |
| 64 / 0.97 | 44.99% |

The current filter exactly reproduces the earlier candidate aggregate. The
shorter filter helps this simulated set, but selecting it from these scores
would reuse evaluation data for tuning. It is not a verified general-purpose
replacement and runtime filter defaults remain unchanged.

The 32 / 0.97 filter was then checked on the same three full telephone channels,
without VAD or fixed-duration pre-segmentation. Float output was rounded to
PCM16 with ties away from zero, clipped, and supplied as 16 kHz WAV to the
candidate server. It does **not** solve the real-call failure:

| Recording | Small INT8 | Large INT8 |
|---|---:|---:|
| Babel demo | 96.10% | 100.00% |
| MATERIAL demo, both channels | 98.92% | 97.85% |

[Small full-channel results](automatic_segmentation_short32_small.json),
[large full-channel results](automatic_segmentation_short32_large.json), and
[filter/input provenance](automatic_segmentation_short32_provenance.json).
These runs use native file windows after external resampling, not the 10 s /
20 s segmentation above. The filter was chosen after inspecting the simulated
set, and both recordings had already been inspected; these are development
controls, not an independent validation of filter selection.

[Aggregates](resampler_controls.json). Reproduction:

```sh
cargo build -p gigastt-core --example resample_narrowband
benchmark/.venv/bin/python scripts/prepare_resampler_controls.py \
  --manifest /tmp/gigastt-precision/manifest.json \
  --output /tmp/gigastt-resampler-lab
# Start the candidate ml_ctc server with punctuation/ITN off, then:
benchmark/.venv/bin/python scripts/benchmark_multilingual_public.py \
  --manifest /tmp/gigastt-resampler-lab/manifest.json \
  --url http://127.0.0.1:19879 --variant ml_ctc --conditions original \
  --output /tmp/gigastt-resampler-lab/results.json
```

Laboratory examples accept local raw audio and never replace model files or
runtime settings. Full LDC audio, references and hypotheses stay outside the
repository; only provenance, hashes and aggregate scores are published here.
Concurrent CPU work means recorded elapsed times are not deployment throughput
measurements.

## Alternative recognizer

The local `abilmansplus/whisper-turbo-kaz-rus-v1` adapter and its pinned base
model were evaluated with BF16, automatic language selection and beam size 5.
On the exact full-channel inputs and references, WER was **75.32% for Babel**
and **52.69% for MATERIAL**. This is a useful alternative baseline, but remains
far from reliable telephone transcription. GigaAM's VAD result on Babel
(62.34%, large model) also prevents claiming universal Whisper superiority.

The comparison concerns complete pipelines, with each recognizer's preprocessing
and chunking. It is not an isolated architecture or precision comparison. See
[alternative report](../alternative_asr_20261001/README.md) for pinned weights,
settings, restricted-data-safe summaries and the fixed natural-speech subset.
No alternate backend was integrated into gigastt.

On a fixed five-clip natural-speech subset (35.17 s, 94 normalized words),
original / simulated-telephone WER was Whisper **10.64% / 10.64%**, small
GigaAM INT8 **9.57% / 55.32%**, and large GigaAM INT8 **9.57% / 36.17%**.
All ten paired inputs have identical audio hashes and reference strings.
This nonrandom subset supports further alternative-model evaluation, not a
31-clip benchmark claim or a real-call quality claim. The first clip was also
used for implementation smoke tests. See the
[matched comparison](../alternative_asr_20261001/matched_first5_comparison.json).

## Controlled adaptation pilot

A frozen-encoder, CTC-head-only FP32 pilot trained on separate official Kazakh
FLEURS training data: 98 eligible utterances, each clean and telephone-augmented
(196 inputs, about 40.3 minutes including augmentation). The first 128 duration-eligible
source rows were selected before training; 30 utterances containing digits
outside the model vocabulary were excluded as whole clean/telephone pairs.
Validation used 32 separate utterances in both conditions. The evaluation
calls and natural-speech clips were excluded from fitting and model selection.

The initial learning rate 2e-5 over three epochs did not improve validation.
An explicitly exploratory, fixed follow-up with 1e-4 and 1e-3, three epochs each,
selected 1e-3 / epoch 2 using validation only: combined validation WER fell
from 10.18% to 9.06%. All settings and epoch results are disclosed.

| Evaluation condition | Original source FP32 | Selected head FP32 |
|---|---:|---:|
| Kazakh natural speech, 31 clips | 6.81% | 9.64% |
| Same clips, simulated telephone | 41.21% | 13.80% |
| Uzbek original, 10-clip control | 19.47% | 22.63% |
| Uzbek simulated telephone, same 10 clips | 21.05% | 25.79% |
| Babel, human-timestamp oracle cuts | 77.92% | 75.32% |
| MATERIAL, human-timestamp oracle cuts | 45.16% | 46.24% |

This is a substantial synthetic narrowband gain **with clean and multilingual
regressions**, and no convincing gain on actual calls. It fails the production
promotion gate. The adapted head was not exported to INT8 or deployed; no
Russian-only nonregression suite was run for it. These are source-checkpoint
results, not gigastt runtime results. Oracle cuts must not be compared directly
with full-channel alternative-model scores.

[Adaptation report and reproduction](../multilingual_adaptation_20261001/README.md).
The evidence supports testing representative telephone training data and mixed
multilingual retention objectives next. It does not prove that full encoder
adaptation will succeed. The practical constraint is access to a larger,
human-transcribed call corpus and an independent final evaluation partition;
two repeatedly inspected public examples cannot establish deployment quality.

## Verification

- `cargo test --workspace --lib --bins`: 1,117 passed, 39 ignored, no failures.
- `cargo clippy`: passed without warnings.
- `cargo clippy -p gigastt-core --example resample_narrowband --example inspect_audio`: passed.
- `cargo fmt --check` and `git diff --check`: passed.
- Research Python scripts parsed successfully; both root preparation scripts
  compiled. Re-running `prepare_resampler_controls.py` reproduced all 93
  generated audio SHA-256 hashes exactly.

No additional production filter, segmentation, frontend or model change was
promoted from these experiments. The earlier narrowband PCM16 mitigation is
still the only runtime change in this working tree; its measured benefits and
limitations are documented separately.
