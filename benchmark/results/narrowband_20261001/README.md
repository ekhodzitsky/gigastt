# Narrowband multilingual diagnosis — 2026-10-01

This audit repeats the [public multilingual pilot](../multilingual_public_20261001/README.md)
with two binaries built from the same checkout, before and after a scoped audio
compatibility change. This is an accuracy audit, not a fresh throughput benchmark.
Some baseline requests overlapped compilation or short diagnostic runs.

## Results

The precision change substantially reduces simulated narrowband degradation. It does
not establish acceptable real-call accuracy. All 277 original-input hypotheses in
the matched Kazakh, Uzbek and Russian controls are unchanged byte-for-byte.

| Simulated telephone input | Model | n | Before WER | After WER |
|---|---|---:|---:|---:|
| kazakh | ml_ctc | 31 | 93.95% | 51.61% |
| kazakh | ml_ctc_large | 31 | 85.26% | 45.37% |
| uzbek | ml_ctc | 100 | 58.81% | 17.83% |
| uzbek | ml_ctc_large | 100 | 23.47% | 15.09% |
| russian | rnnt | 15 | 1.33% | 1.33% |

Original Kazakh WER remains 6.81% / 6.62% (small / large); Uzbek 16.76% /
14.34%. Russian fixture WER remains 0% on originals and 1.33% on the simulated
telephone condition (15 recordings). These small controls are not a full Russian
accuracy benchmark.

### Real Kazakh call: all conditions, one sample

| Model | Input | VAD | Before WER | After WER |
|---|---|---|---:|---:|
| ml_ctc | Native 8 kHz | off | 97.40% | 96.10% |
| ml_ctc | FFmpeg PCM16 16 kHz | off | 96.10% | 96.10% |
| ml_ctc_large | Native 8 kHz | off | 96.10% | 100.00% |
| ml_ctc_large | FFmpeg PCM16 16 kHz | off | 98.70% | 98.70% |
| ml_ctc | Native 8 kHz | on | 100.00% | 100.00% |
| ml_ctc | FFmpeg PCM16 16 kHz | on | 100.00% | 100.00% |
| ml_ctc_large | Native 8 kHz | on | 100.00% | 62.34% |
| ml_ctc_large | FFmpeg PCM16 16 kHz | on | 94.81% | 94.81% |

The large model without VAD regresses on this sample (96.10% -> 100% WER);
with VAD it improves from 100% to 62.34%, which is still poor. No general
telephone-quality or superiority claim is supported by this single recording.

Reference-timestamp segmentation into 24 clips (the same 77 reference words)
gives candidate WER 79.22% / 74.03% (small / large), with 1 / 4 empty
hypotheses respectively. This is an oracle diagnostic, not automatic VAD quality.
The short voiced interval can be recovered by PCM16 rounding, but even the
reference-style window with PCM16 produces only one word on a 30-second control.
Simply replacing the window or shortening utterances does not resolve the call.

The audit contains 1,172 measured HTTP requests, excluding warm-ups and NumPy
ablations. `comparison.json` includes matched-input checksum and original-text
invariance checks. Both binaries used the same model files and settings.

## Change and causal evidence

The float sinc resampler itself is not mathematically incorrect. Its very low
out-of-band energy exposes a sensitivity of the tested multilingual INT8 models.
The reference [GigaAM file loader](https://github.com/salute-developers/GigaAM/blob/7447938d791c4f3e643386ee22c33777004293a5/gigaam/preprocess.py)
resamples with FFmpeg and emits **signed PCM16**, then divides by 32768.
Our 8 kHz path retained sub-LSB float precision after interpolation.

The change rounds and saturates **only 8000 -> 16000 Hz** output to the PCM16
grid. It applies to whole-buffer and cached/streaming resampling, leaves the
sinc filter intact and adds no random dither. Other rate conversions and native
16 kHz input are unchanged. This matches reference input precision, not FFmpeg's
exact interpolation kernel or output length. It is a compatibility mitigation,
not a claim that every narrowband recording will become accurately recognizable.

Controls:

- PCM and G.711 mu-law versions of the same Kazakh clip both show the original
  failure. The narrowband degradation is not specific to a companding decoder.
- Independently decoding the real LDC A-law SPHERE file with FFmpeg and libsndfile
  gives identical PCM samples. Speech has substantial amplitude; it is not
  accidentally interpreted as silence by the file decoder.
- A Rust PCM/mel dump isolates preprocessing from HTTP and CTC token stitching.
  The same small INT8 ONNX model also collapses with those inputs in Python ORT.
- Changing only the log floor from 1e-10 to 1e-9 does not recover the lost text.
  The [ONNX reference frontend](https://github.com/istupakov/onnx-asr/blob/675f0e68c24d846ee1775743e92d9b4ed452380e/preprocessors/gigaam.py)
  uses a periodic bf16-rounded Hann window and bf16-rounded mel coefficients;
  those differ from our frontend, but substituting the window alone is also
  insufficient. A subsequent [checkpoint audit](../multilingual_precision_20261001/README.md)
  established that the actual multilingual checkpoint uses float32 periodic
  buffers, not this ONNX frontend preset. Do not interpret this earlier preset
  control as exact checkpoint parity. The production frontend was not changed.
- On the real call's 15.795–24.815 s voiced interval, float input produces no
  words in the diagnostic frontend. PCM16 rounding produces speech with either
  window. This is a causal precision control, not a full-call accuracy score.
- Coarser 12/14/15-bit diagnostic controls can improve individual outputs, but
  were not selected as product behavior: PCM16 has a reference-loader basis.

`precision_ablation.json` and `telephone_precision_ablation.json` contain these
controls. The waveform ablation uses NumPy FFT, author bf16 mel filters and a
1e-9 floor; its symmetric-window row is **not** an exact Rust frontend replica.
The floor-only rows operate directly on the original Rust feature dump. The
production before/after scores below come from Rust HTTP inference.

## Data and interpretation

- All 31 Kazakh/Russian natural-speech clips and the same seeded 100 Uzbek
  FLEURS test recordings from the pilot. Original files and their 300–3400 Hz,
  8 kHz mu-law simulations are scored separately.
- One real 110.04 s Kazakh telephone demonstration, 77 reference words, from
  [LDC2018S13](https://catalog.ldc.upenn.edu/LDC2018S13). Both native 8 kHz and
  externally converted 16 kHz PCM16 inputs are checked, with and without VAD.
  This single sample cannot establish general telephone quality. Reference,
  audio and generated transcript are kept outside the repository; summaries
  contain provenance hashes and aggregate scores only.
- The repository's Russian Golos fixtures are a small default RNN-T regression
  control, with original and simulated telephone versions.
- Scores are micro WER using `scripts/wer_unicode.py` normalization and jiwer.
  Kazakh uses the publisher's normalized reference layer, not best-of-two macro
  scoring. These values must not be compared directly with the Habr table.
  Empty hypotheses remain in the denominator. ITN/punctuation are disabled.
- Native 8 kHz and externally resampled 16 kHz inputs are different conditions.
  Likewise, VAD and no-VAD measurements must not be pooled.

## Reproduction

Build the checkout before the patch and preserve its binary; then build the
patched version. Binary hashes, revision, model hashes and validation counts
are in `environment.json`. The pilot manifests contain source revisions,
selection rules, local paths and audio checksums. Recreate their data using
`scripts/prepare_multilingual_public.py` and the instructions in the pilot.

Run each binary/model serially:

```sh
/path/to/gigastt --offline serve --model-variant ml_ctc \
  --pool-size 1 --encoder-intra-threads 6 --punctuation off --itn off --port 19876
benchmark/.venv/bin/python scripts/benchmark_multilingual_public.py \
  --manifest benchmark/results/multilingual_public_20261001/kazakh_manifest.json \
  --variant ml_ctc --conditions original simulated_telephone --output /tmp/results.json
```

Repeat with `uzbek_manifest.json` and `ml_ctc_large`. For the real call use its
locally generated manifest with `--conditions original original_external_16k`;
restart the server with `--vad` for the VAD control. Use `russian_manifest.json`
and variant `rnnt` for the Russian fixture control. Report all conditions.

For a preprocessing ablation, build the `inspect_audio` example against the
**baseline** resampler. Decode the narrowband source to a PCM/mel dump:

```sh
cargo run -p gigastt-core --example inspect_audio -- /tmp/input8.wav /tmp/input8
uv pip install --python benchmark/.venv/bin/python numpy==2.5.3 onnxruntime==1.30.0 onnx-asr==0.12.0
benchmark/.venv/bin/python scripts/diagnose_narrowband.py \
  --input-prefix /tmp/input8 --model-dir "$HOME/.gigastt/models" --output /tmp/ablation.json
```

The diagnostic dependencies are development-only. See the pinned upstream
revision above for the reference window/filterbank. Use `--start 15.795
--end 24.815 --omit-text` on the LDC dump to repeat the restricted-text control.

## Validation

The new precision/chunk-invariance regression failed before the code change and
passed afterward. Silence remains exactly zero; overload saturates without
wrapping. Workspace unit tests, clippy and formatting checks passed (counts in
`environment.json`). Model-dependent HTTP results are saved separately below;
this audit does not claim WebSocket recognition accuracy equal to file mode.

To reproduce the oracle segmentation control after preparing the LDC sample:

```sh
python3 scripts/prepare_telephone_segments.py --cache /tmp/gigastt-multilingual-audit \
  --output /tmp/phone_segments
benchmark/.venv/bin/python scripts/benchmark_multilingual_public.py \
  --manifest /tmp/phone_segments/manifest.json --variant ml_ctc \
  --conditions original --output /tmp/phone_segment_results.json
```

Repeat with the large-model server. Keep the resulting text-bearing files local.
Use `--end 30 --omit-text` in `diagnose_narrowband.py` for the long-window control.
