# Official GigaAM multilingual telephone controls — 2026-10-01

The original small GigaAM checkpoint was evaluated through its unmodified
`transcribe` API on fixed matched examples. Narrowband degradation also affects
the source model, particularly Kazakh in this small subset. This is a diagnostic
comparison, not a replacement for the 1,000-per-language product benchmark.
No model was trained, fine-tuned, or deployed.

## Fixed short-API subset

Selection was fixed before inference: the first ten entries of each language's
[baseline manifest](../multilingual_1000_20261001/README.md), with the identical
original, G.711 A-law, and G.711 mu-law inputs prepared for the full telephone
benchmark. The official API rejects the 27-second Russian recording
`fleurs_ru_ru_test_0007`; it was excluded in all three conditions without
replacement, truncation, or bypassing the 25-second guard. Thus there are 49
unique utterances and **147 successful short-API executions**, plus three
explicit length rejections. Check `source.json` for the exact rejected ID.

Micro WER, Unicode normalization, no ITN; lower is better:

| Language | Utterances per condition | Original | A-law 8 kHz | Mu-law 8 kHz |
|---|---:|---:|---:|---:|
| Russian | 9 | 10.39% | 10.39% | 11.04% |
| English | 10 | 12.92% | 16.75% | 16.27% |
| Kazakh | 10 | 13.48% | 24.72% | 24.16% |
| Kyrgyz | 10 | 10.40% | 10.98% | 10.98% |
| Uzbek | 10 | 19.05% | 22.22% | 22.22% |

The source uses the official FFmpeg PCM16 loader, original torchaudio frontend,
and original greedy CTC decoder. It is FP32 CPU, two PyTorch inference threads.
`source.json` records the source revision, checkpoint SHA-256, package versions,
audio hashes, per-input errors, and reference word counts. Timing overlapped
other experiments and is not a throughput benchmark.

`matched_comparison.json` now contains all **30 completed comparisons**: five
languages, three conditions, and both gigastt variants. `missing_groups` is
empty. It compares exactly the same successful source IDs, checking audio
hashes and reference word totals. All fifteen helper/server checks are also
complete: ten hypotheses match exactly except outer whitespace; five do not.
These are small diagnostic subsets, not the full 1,000-per-language scores.
The large product model is a
different model, so its comparison with the small source checkpoint cannot
isolate export or quantization effects. Even small source versus small product
changes both numerical precision and the audio/frontend pipeline. The earlier
[matched-feature precision audit](../multilingual_precision_20261001/README.md)
is the separate experiment that isolates export and quantization.

An additional [fixed PCM/frontend/precision control](frontend_results.md)
completed 135 executions on 15 overlapping inputs. It confirms source/FP32
frame decisions on all 45 feature arrays, while recording an unresolved exact
text mismatch between the Python INT8 helper and the native product runtime.

## Actual long-form attempt and real-call controls

The pinned official source requires `pyannote/segmentation-3.0` for its
[long-form segmentation](https://github.com/salute-developers/GigaAM/blob/7447938d791c4f3e643386ee22c33777004293a5/gigaam/vad_utils.py).
The model's [access conditions](https://huggingface.co/pyannote/segmentation-3.0)
require an authenticated account with accepted terms. Initially there was no
cached model or configured `HF_TOKEN`; an anonymous weights request returned
HTTP 401. After local login was configured, an authenticated request to the
required model file returned **HTTP 403 from Hugging Face**, classified as
missing gated-model authorization/accepted conditions. Thus login alone has
not yet made the segmentation weights available. Credentials and response
headers were never written to artifacts.

Initial API attempts failed on the missing optional Python dependency. After
installing the official source's specified pyannote.audio 4.0 series, the
unmodified `transcribe_longform` was retried on all three channels. Every call
reached the official resolver and raised:

> Model pyannote/segmentation-3.0 was not found locally, and no HF_TOKEN was provided to download it.

TorchCodec was pinned to 0.8.1 for PyTorch 2.9, and its missing `libavdevice62`
runtime library was extracted into the isolated cache. The final retry had no
captured warnings and reached the same model-access failure on every channel.
See `longform_retry.json` and `segmentation_access.json`. **No official long-form transcription
completed.** The segmentation model was not substituted and the access gate
was not bypassed.

A separate permitted control split each call channel at automatic,
nonoverlapping 20-second boundaries, ran the official short API on each piece,
concatenated hypotheses, and scored the full reference. This is not the official
long-form pipeline, VAD, or privileged reference-timestamp segmentation.

| Real recording | Reference words | Errors | WER |
|---|---:|---:|---:|
| Babel LDC2018S13 public example, full 110.04 s | 77 | 72 | 93.51% |
| MATERIAL LDC2025S03, two 56.14 s channels combined | 93 | 91 | 97.85% |

These are **two public recordings**, not three independent calls or a
representative telephone corpus. Fixed cuts may split words. MATERIAL includes
Babel material, so corpus-level independence is not asserted. Restricted audio,
references, and hypotheses are retained privately under
`~/.cache/gigastt-official-telephony/`; repository artifacts contain scores and
hashes only. Restored Babel SPH and transcript hashes match the earlier public
pilot; MATERIAL preparation verifies its published provenance hashes.

## Reproduction

Use the original checkpoint under `~/.cache/gigastt-precision/models/` and the
official source pinned to `7447938d791c4f3e643386ee22c33777004293a5`. The isolated
environment is `~/.cache/gigastt-official-telephony/venv`; runtime versions are
recorded in the JSON artifacts. Prepare the exact baseline and degraded
manifests first; restore the LDC demonstrations privately using the existing
preparation scripts.

```sh
LD_LIBRARY_PATH="$HOME/.cache/gigastt-official-telephony/deps/usr/lib/x86_64-linux-gnu" \
~/.cache/gigastt-official-telephony/venv/bin/python scripts/benchmark_official_telephony.py \
  --baseline benchmark/results/multilingual_1000_20261001 \
  --degraded benchmark/results/multilingual_telephony_20261001 \
  --phone-manifest ~/.cache/gigastt-official-telephony/babel/telephone_manifest.json \
  --phone-manifest ~/.cache/gigastt-official-telephony/material/telephone_expansion_manifest.json \
  --work ~/.cache/gigastt-official-telephony/run \
  --output benchmark/results/official_telephony_20261001 \
  --checkpoint ~/.cache/gigastt-precision/models/multilingual_ctc.ckpt \
  --source ~/.cache/gigastt-official-telephony/GigaAM --threads 2

benchmark/.venv/bin/python scripts/compare_official_telephony.py \
  --source benchmark/results/official_telephony_20261001/source.json \
  --baseline benchmark/results/multilingual_1000_20261001 \
  --degraded benchmark/results/multilingual_telephony_20261001 \
  --frontend-private ~/.cache/gigastt-official-telephony/frontends/frontend_private.json \
  --output benchmark/results/official_telephony_20261001/matched_comparison.json
```

Both new scripts pass `py_compile`. Repository Rust validation is coordinated
by the parent benchmark task; these scripts do not alter production inference.
