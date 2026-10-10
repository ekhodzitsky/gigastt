# Multilingual source, FP32 ONNX and INT8 audit — 2026-10-01

The small multilingual model has substantial narrowband errors **before export
or quantization**. FP32 ONNX reproduces the original PyTorch checkpoint on all
132 evaluated inputs: identical frame argmax and transcriptions after trimming
outer whitespace, using exactly the same saved feature arrays. INT8 adds errors on some conditions, but is not
the primary cause of the telephone failure. No production model was replaced.

## Matched results

Micro WER with `scripts/wer_unicode.py` normalization; lower is better. All
three systems use the original checkpoint's frontend and FFmpeg PCM16 loader.
The ONNX runs consume the actual feature bytes saved by the source run.

| Condition | Inputs | Reference words | Source FP32 | FP32 ONNX | INT8 ONNX |
|---|---:|---:|---:|---:|---:|
| Kazakh/Russian natural speech, original | 31 | 529 | 6.81% | 6.81% | 7.37% |
| Same speech, simulated telephone | 31 | 529 | 41.21% | 41.21% | 47.64% |
| Uzbek FLEURS, original control subset | 10 | 190 | 19.47% | 19.47% | 18.42% |
| Same Uzbek subset, simulated telephone | 10 | 190 | 21.05% | 21.05% | 22.63% |
| Babel real call, oracle utterance cuts | 24 | 77 | 77.92% | 77.92% | 83.12% |
| MATERIAL real call, oracle utterance cuts | 24 | 93 | 45.16% | 45.16% | 45.16% |
| Babel first 0–24.815 s, diagnostic | 1 | 28 | 100% | 100% | 100% |
| Babel first 0–30 s, diagnostic | 1 | 28 | 100% | 100% | 100% |

The 24.815 s crop is within the official short-transcription API's 25 s limit.
The 30 s crop calls the encoder directly and bypasses that API guard. Neither
is a run of the official Pyannote-based long-form pipeline, which was not
evaluated. Both crops overlap the Babel oracle clips and are not independent
data. Oracle boundaries are privileged reference information, not automatic
segmentation or production VAD results. The actual telephone evidence consists
of **two public example calls**, not 48 independent calls.

The Uzbek control is the first ten entries of the pilot's seeded 100-item
manifest, not the full 100-item benchmark. The Kazakh and Uzbek source
manifests and provenance are in the [pilot report](../multilingual_public_20261001/README.md).
The extra call is documented in
[`material_sample_provenance.json`](../telephone_corpora_20261001/material_sample_provenance.json).
MATERIAL contains Babel material, so this audit does not claim corpus-level
independence or absence of pretraining overlap.

## Export and quantization isolation

[`comparison.json`](comparison.json) records model hashes and numerical checks.
The maximum source/FP32 logit difference over all evaluated frames/classes is
0.0009618; the mean of per-input mean absolute differences is 0.00000453.
Every output frame has the same top class. This is strong direct evidence
against an export defect for the tested inputs. It is not a proof covering
every possible input or the separate large model.
Five raw text strings differ only in outer whitespace because the ONNX harness
trims it; all normalized scores and decoded non-whitespace content agree.

INT8 increases simulated Kazakh WER by 6.43 percentage points on identical
features. It increases Babel oracle WER by 5.19 points; the MATERIAL oracle
aggregate is unchanged. Better quantization may recover part of the loss, but
reverting to FP32 would still leave 41.21% simulated Kazakh WER and poor real
call accuracy. FP32 was used only as a laboratory control; the product remains
INT8-only.

Models:

- Original `multilingual_ctc.ckpt` from the
  [official GigaAM source](https://github.com/salute-developers/GigaAM/tree/7447938d791c4f3e643386ee22c33777004293a5),
  downloaded from its configured Sber CDN and verified against the source's
  expected MD5 `5379d887c53ccd9cb95981e2a1832720`.
- FP32 ONNX from
  [`istupakov/gigaam-multilingual-ctc-onnx`](https://huggingface.co/istupakov/gigaam-multilingual-ctc-onnx/tree/458860e1983aef670dd9795fb6af603c82767d5d),
  pinned revision `458860e1983aef670dd9795fb6af603c82767d5d`.
- The existing production INT8 file, hash recorded alongside both controls.

## Frontend controls

Eleven additional inputs per precision isolate the frontend and resampled
waveform. These use the saved baseline Rust PCM/mel dumps from the preceding
[narrowband audit](../narrowband_20261001/README.md), not a rebuilt approximation.
Results are in `frontend_source.json`, `frontend_fp32.json`,
`frontend_int8.json` and `frontend_numeric.json`.

For the first Kazakh clip, the old float-preserving Rust 8→16 kHz waveform
produces **empty output in the original FP32 model**, with either Rust features
or the exact checkpoint frontend. Replacing the frontend alone does not fix
it. Rounding that waveform to the PCM16 grid recovers some words in FP32
(82.35% WER on this single clip); FFmpeg-resampled PCM16 gives 29.41% WER.
The different resampling kernels therefore remain a material input difference.
This is a diagnostic, not a new aggregate accuracy claim. The rounding control
uses NumPy's round-to-nearest-even, not a byte-identical recreation of Rust's
halfway rounding.

On the original full-band waveform both frontends produce zero word errors
for this clip. On the external FFmpeg waveform, source FP32 has 29.41% WER
with either frontend; INT8 has 41.18% with Rust features and 29.41% with source
features. Small frontend differences can change individual outcomes, but
neither a frontend-only nor a quantization-only explanation fits all controls.

The original multilingual checkpoint frontend also differs from the generic
`onnx-asr` v3 reference bank used in the earlier diagnostic. Its Hann buffer is
smooth periodic float32 (maximum difference from the analytic periodic Hann
is 1.87e-7), **not BF16-rounded**. Relative to `onnx-asr`'s packaged
`gigaam_v3_window` and `gigaam_v3` bank, maximum coefficient differences are
0.001946 and 0.001951. Thus the previous BF16 frontend ablation should be read
as an `onnx-asr` preset control, not the exact original multilingual frontend.
No production frontend change was made based on this small control.

## Reproduction and artifacts

`scripts/compare_multilingual_precision.py` prepares matched inputs, saves
source features/logits and evaluates ONNX against them. Dependencies are
development-only. Source environment: Python 3.13, torch 2.14.1+cpu,
torchaudio 2.11.0+cpu, GigaAM from the pinned source revision, NumPy 2.5.3,
hydra-core 1.3.7, omegaconf 2.3.1, soundfile, sentencepiece and tqdm. ONNX
evaluation used the existing Python 3.12 benchmark environment with
onnxruntime 1.30.0. Both use three inference threads. Timings overlapped other
work and must not be interpreted as a throughput comparison.

After preparing the pilot and both LDC demonstration manifests:

```sh
python scripts/compare_multilingual_precision.py prepare \
  --work /tmp/gigastt-precision --output benchmark/results/multilingual_precision_20261001 \
  --kazakh-manifest benchmark/results/multilingual_public_20261001/kazakh_manifest.json \
  --uzbek-manifest benchmark/results/multilingual_public_20261001/uzbek_manifest.json \
  --telephone-manifest /tmp/gigastt-narrowband/phone_segments/manifest.json \
  --telephone-full-manifest /tmp/gigastt-multilingual-audit/telephone_manifest.json \
  --extra-telephone-manifest /tmp/gigastt-telephone-expansion/telephone_segments_manifest.json

/tmp/gigastt-precision-venv/bin/python scripts/compare_multilingual_precision.py source \
  --work /tmp/gigastt-precision --output benchmark/results/multilingual_precision_20261001 \
  --model ~/.cache/gigastt-precision/models/multilingual_ctc.ckpt --label source

benchmark/.venv/bin/python scripts/compare_multilingual_precision.py onnx \
  --work /tmp/gigastt-precision --output benchmark/results/multilingual_precision_20261001 \
  --model ~/.cache/gigastt-precision/models/multilingual_ctc.fp32.onnx --label fp32

benchmark/.venv/bin/python scripts/compare_multilingual_precision.py onnx \
  --work /tmp/gigastt-precision --output benchmark/results/multilingual_precision_20261001 \
  --model ~/.gigastt/models/multilingual_ctc.int8.onnx --label int8
```

For the eleven frontend controls, use `prepare-frontends` with
`--work /tmp/gigastt-precision/frontends --dump-dir /tmp/gigastt-narrowband`,
the same Kazakh and oracle telephone manifest arguments, then run the source
and ONNX stages in that work directory with labels `frontend_source`,
`frontend_fp32` and `frontend_int8`. The baseline dump directory must contain
the four `original`, `internal_mulaw8`, `external16` and `telephone` PCM/mel
prefixes described in the preceding audit.

Audio, references, hypotheses and feature arrays stay under `/tmp`; repository
JSON contains per-input scores and hashes only. Do not redistribute the LDC
sample audio or transcript. The source/FP32/INT8 comparison contains 396
scored model executions, plus 33 frontend controls. Intermediate three-input
INT8 probes are not counted again. The script passed `py_compile`; workspace
Rust checks are coordinated by the parent task.
