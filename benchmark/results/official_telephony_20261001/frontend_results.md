# Matched PCM, frontend, and precision controls

The [protocol](frontend_protocol.md) fixed all inputs and chains before the run:
one recording per language, original plus both G.711 encodings, three feature
chains, and three precisions. All **135 executions completed**. This is 15
overlapping diagnostic inputs, not another independent corpus.

On all **45 feature arrays**, original source FP32 and FP32 ONNX produce
identical frame argmax and normalized hypotheses. The largest absolute logit
difference is 0.0006809235. Thus this control finds no export defect on the
tested features, including the Rust frontend arrays.

Micro WER across the five selected recordings (96 reference words per row):

| Condition | PCM / mel chain | Source FP32 = FP32 ONNX | INT8 ONNX, Python ORT 1.30 |
|---|---|---:|---:|
| Original | Official / official | 7.29% | 7.29% |
| Original | Rust / official | 7.29% | 7.29% |
| Original | Rust / Rust | 7.29% | 7.29% |
| A-law | Official / official | 26.04% | 28.12% |
| A-law | Rust / official | 8.33% | 26.04% |
| A-law | Rust / Rust | 10.42% | 26.04% |
| Mu-law | Official / official | 26.04% | 26.04% |
| Mu-law | Rust / official | 7.29% | 26.04% |
| Mu-law | Rust / Rust | 7.29% | 26.04% |

The large changes are concentrated in the single fixed Kazakh example. For its
A-law input, original source inference has 20/20 word errors with official
PCM/official mel, 3/20 with Rust PCM/official mel, and 5/20 with Rust PCM/Rust
mel. On exactly those feature arrays INT8 has 20/20, 20/20, and 19/20 errors.
For mu-law, the corresponding FP32 counts are 20/20, 2/20, and 2/20; INT8 has
20/20 for each. This demonstrates an interaction between input preparation and
numerical precision on this example, not a general estimate of Kazakh WER.
There are no empty strings in this diagnostic, even where WER reaches 100%.

## Numerical differences and interpretation limits

- The original FLEURS WAVs contain float samples. The official loader rounds
  to PCM16; Rust preserves float input. Their mean absolute sample differences
  are approximately 7.5–7.6e-6 even without resampling.
- All ten Rust telephone waveforms have **four fewer 16 kHz samples** than the
  official FFmpeg output. At these selected lengths, that also removes one mel
  frame. Changing the PCM chain therefore changes both the resampling kernel
  and sequence length; this experiment does not isolate those two effects.
- On the same Rust PCM, official and Rust mel banks differ numerically. The
  JSON reports full array shapes plus aligned mean/max differences; unmatched
  tails are not silently treated as equal.
- Debug `inspect_audio` was rebuilt against the current source, including the
  PCM16 narrowband mitigation. Its binary and source-file hashes are recorded.
- **The Python INT8 control is not asserted to be byte-equivalent to the
  production server.** The completed exact hypothesis join matches **10/15**
  cases. All three English conditions and both degraded Kazakh conditions
  differ. Four of these five mismatches have identical word-error counts;
  Kazakh A-law has 19/20 errors in the helper versus 20/20 in the product.
  A repeated frozen candidate CLI
  reproduced the server text. Python ORT 1.30 and a separate ORT 1.28 control
  produced different mixed-script tokens, and loading the server's cached ORT
  graph did not resolve that difference. The production native binary reports
  ORT 1.28.0. Consequently the INT8 column describes the named laboratory
  runtime; the full 20,000-request benchmark measures the actual product.

The subsequent runtime-mismatch check installed a separate Python ORT 1.28
wheel and rebuilt the release inspection helper. **Debug and release PCM and
mel dumps are byte-identical on the mismatched English input**, so that build
profile difference does not explain the mismatch. `frontend_runtime_probe.json`
records this bounded follow-up, native repeat equivalence, runtime versions,
binary/feature hashes, and hypothesis comparison outcomes without the texts.
These checks are outside the predeclared 135-execution factorial; no checkpoint
was downloaded or replaced for them. The remaining native/Python difference
is unresolved; no production change is justified from this helper mismatch.

`frontend_factorial.json` contains all 135 scores, 45 feature hashes, per-input
numeric comparisons, checkpoint hashes, and execution metadata. Full arrays
and hypotheses remain in `~/.cache/gigastt-official-telephony/frontends/`.
`matched_comparison.json` includes all fifteen completed helper/server checks,
with no missing product results, alongside all thirty matched short-subset
comparisons. These checks compare the full text except outer whitespace, not
only aggregate WER. The runtime mismatch remains unresolved; the production
benchmark is the authoritative measurement of the product.

Reproduce with `scripts/compare_official_frontends.py --help`. Use the same
baseline/degraded directories as the short-API experiment, the cached original
checkpoint and `multilingual_ctc.fp32.onnx`, production
`~/.gigastt/models/multilingual_ctc.int8.onnx`, and rebuilt
`target/debug/examples/inspect_audio`. To refresh the product join, add
`--frontend-private ~/.cache/gigastt-official-telephony/frontends/frontend_private.json`
to the `compare_official_telephony.py` command in the main report.

No production inference or model was changed. No parameters were trained.
