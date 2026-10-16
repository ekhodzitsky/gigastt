# Small multilingual deletion-failure diagnosis

**Outcome: INT8 substantially amplifies the selected telephone failures; input-length repair does not resolve them, and the tested resampler finalization correction introduces new regressions. No production correction or model replacement was applied. No parameters were trained.**

This is a causal diagnostic follow-up to the [complete five-language benchmark](../multilingual_telephony_20261001/README.md). Its deliberately failure-enriched scores must not be presented as general Kazakh, Kyrgyz, or multilingual WER. The earlier 1,000-recording-per-language benchmark remains unchanged.

## Fixed cases and controls

- All 22 unique recordings with an empty small-model response or at least 80% reference-word deletions in either telephone condition: 21 Kazakh and one Kyrgyz recording.
- The first three baseline-ordered nonfailure recordings from each of the five languages: 15 controls.
- Original, A-law and mu-law versions of every recording: 111 fixed inputs. Selection was frozen before new interventions.
- The reference totals per condition are 453 words for the 22 failure recordings and 264 words for the 15 controls.
- 444 native HTTP recognitions test original-file/decoded-PCM parity and three telephone transformations. Another 666 native encoder executions compare INT8 and FP32 on the same 333 feature arrays. These executions reuse the same recordings; they are not additional independent samples.

[cases.json](cases.json) records the exact selection and original scores. [provenance.json](provenance.json) pins the original product binary, INT8 model, vocabulary, scorer and audio/feature code.

## Precision on identical product features

The diagnostic Rust encoder uses the same runtime seam as the product, the same CPU configuration and three intra-op threads. All 111 baseline INT8 texts match the original benchmark. Across all 333 transformed inputs, its production-style word assembly matches the HTTP server exactly. Raw CTC text has two additional internal double spaces; retaining this distinction avoids confusing text formatting with numerical inference.

Word errors / reference words and empty responses on the fixed 22 failure recordings:

| Input | INT8 errors | INT8 empty | FP32 errors | FP32 empty |
|---|---:|---:|---:|---:|
| original | 80 / 453 | 0 | 80 / 453 | 0 |
| alaw | 437 / 453 | 4 | 181 / 453 | 1 |
| mulaw | 399 / 453 | 4 | 189 / 453 | 1 |

On A-law the deletion count falls from 401 with INT8 to 105 with FP32; on mu-law it falls from 349 to 122. Thus quantized execution substantially worsens these particular failures. Because cases were selected using INT8 failures, these differences are not an unbiased estimate of the quantization penalty over a language or corpus. FP32 was used only as a diagnostic control; the product remains INT8-only.

The remaining two baseline FP32 empty responses are the same Kyrgyz recording, `fleurs_ky_kg_test_0578`, in both codecs. All 666 output tensors contain finite values; empty cases represent actual all-blank frame argmax, not NaN handling or a transcript-formatting bug.

An original-source check covers all five unique recordings that had empty telephone responses, in all three original/telephone conditions (15 inputs). PyTorch source and native FP32 ONNX agree on every frame argmax for all 15; maximum absolute logit difference is 0.000761. The Kyrgyz recording is already all-blank in the original source on both telephone variants. This rules out an ONNX export defect on these tested features, while not claiming equivalence for every possible input or completion of the separate official long-form pipeline. See [source_empty_case_parity.json](source_empty_case_parity.json).

## Length and resampler controls

The [synthetic audit](resampling_audit.md) proves a separate file-finalization defect. With the pinned sinc resampler, 8 kHz input produces `2N - 4` output samples, but its measured impulse delay is about 255 output samples. The API reports a delay of 256 (16 ms). The current file path neither removes startup delay nor drains the delayed tail; a final-sample impulse disappears. Nine deterministic probes reproduce the behavior in whole-buffer and cached chunked processing.

Three transformations were fixed before recognition and applied to all 74 telephone inputs:
1. `length_only_pad4`: append four zero samples, changing duration only.
2. `delay_only_shift256`: discard 256 output samples and append 256 zeros, preserving the original output length but not restoring the missing real tail.
3. `flush_trim256`: append 512 source-rate zeros, resample with the existing implementation, discard the documented 256-sample delay and keep exactly `2N` output samples. This drains the real filter state. The extended output prefix is checked to be bit-identical to the original output.

Every transformed 16 kHz float WAV is verified to preserve PCM exactly through the native decoder. The documented delay value was not tuned against WER. Four-sample length repair can add a feature frame at a hop boundary; a 256-sample shift also changes the complete waveform alignment relative to the STFT grid.

INT8 errors on the failure recordings (453 reference words per row):

| Transformation | A-law errors | A-law empty | Mu-law errors | Mu-law empty |
|---|---:|---:|---:|---:|
| decoded_float | 437 | 4 | 399 | 4 |
| length_only_pad4 | 445 | 4 | 405 | 4 |
| delay_only_shift256 | 336 | 5 | 377 | 5 |
| flush_trim256 | 336 | 6 | 385 | 6 |

Length-only repair fixes none of the failure recordings by word-error count: four A-law and two mu-law cases worsen. Proper drain/trim improves eight A-law cases but worsens two; for mu-law it improves four and worsens four. It recovers two previous empty cases while creating four new empty cases in each codec. On the 15 controls, drain/trim changes A-law errors from 29 to 32, with three regressions, and mu-law errors from 26 to 25.

FP32 does not make drain/trim safe either: failure-case A-law errors increase from 181 to 270 and mu-law errors from 189 to 244. Therefore the demonstrated DSP defect and the observed recognition collapse cannot be treated as the same bug. A correct duration/alignment change is not automatically a safe ASR-quality change. The candidate was rejected before production modification or a full 1,000-recording rollout. No sample-specific rescue, random dither, alternative trim search, or trained adaptation was introduced.

## Runtime reproducibility finding

The earlier Python/native discrepancy was isolated by using the exact same feature bytes, shapes, valid lengths and thread counts:
- Native CPU factory and production factory (including the optimized graph cache) produce bit-identical outputs on 15 controls.
- The bundled native runtime reports ORT 1.28.0 build commit `da9b5e3`; Python ORT 1.28.0 reports `45de2a8b06`.
- Running the same Rust probe with the Python package's exact shared library produces bit-identical Rust/Python logits on all 15 controls.
- Different builds agree on FP32 frame argmax on these controls, while INT8 logits/text can differ substantially.

This localizes the discrepancy to the concrete runtime build rather than the language adapter, feature layout or optimized cache. It does not identify a particular upstream kernel bug or prove that switching runtime builds improves quality. No runtime upgrade or downgrade was applied. Pinning only the displayed version number is insufficient for these numerical comparisons; the report records build info and binary hashes. See [native_runtime_parity.json](native_runtime_parity.json).

## Decision and verification

No production code or model change is justified by the tested interventions. The safe outcome of this investigation is a reproducible diagnosis: quantization amplifies selected failures, source-model narrowband sensitivity also exists, and file finalization has a separately demonstrated defect whose straightforward correction causes recognition regressions. The next optimization, if pursued, needs its own validated candidate and the fixed full benchmark; these diagnostic scores are not a replacement for that gate.

- [native_controls_validation.json](native_controls_validation.json): independently checked all 444 native results, all 222 baseline parity checks, word/CER scores, model/binary identities, PCM transformations and feature hashes.
- [native_precision.json](native_precision.json): all 666 INT8/FP32 results, S/D/I, actual blank-frame counts, valid lengths and feature identity.
- [native_precision_http_join.json](native_precision_http_join.json): all 333 native/product comparisons with explicit raw-CTC versus product word formatting.
- [native_precision_audit.json](native_precision_audit.json): independent precision/HTTP/scoring audit.

Required repository checks passed: 1,117 unit tests, 39 ignored, zero failures; `cargo clippy` and `cargo fmt --check` passed. The pre-existing narrowband PCM16 mitigation and previous benchmark artifacts were retained unchanged. The temporary HTTP diagnostic server was stopped. No HF credentials or gated models were required for this stage.

## Reproduction

Use `scripts/prepare_small_diagnosis.py` to recreate the fixed case selection from the preserved benchmark artifacts. Start the frozen binary with the command in `provenance.json`, then run `scripts/diagnose_small_inputs.py --help` for the 444 HTTP/audio controls. Use `scripts/analyze_small_precision.py --help` and `crates/gigastt-core/examples/native_ctc_probe.rs` for matched native precision inference. `scripts/run_resampling_length_probe.py` reproduces the synthetic resampling evidence without changing the repository Cargo configuration. Raw PCM, features, logits and diagnostic checkpoints remain in `~/.cache/gigastt-small-diagnosis/`; published JSON retains identities and scores.
