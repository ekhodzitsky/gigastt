# Models and formatting

This page describes the model catalog in `main`. Model assets are pinned in
[`model/variant.rs`](../crates/gigastt-core/src/model/variant.rs); an older
installed binary can pin different weights even when the filenames match.
The published v2.22.0 binary uses the June RNN-T bundle. Current main selects the
2026-10-09 bundle; [migration and evidence](rnnt-quantization.md) explain the change.

## Supported catalog

| CLI variant | Languages | Built-in formatting | ASR files, approximate disk size |
|---|---|---|---|
| `rnnt` | Russian | None | Encoder + decoder + joiner + vocabulary, 310 MiB |
| `e2e_rnnt` | Russian | Punctuation, casing, text normalization | Encoder + decoder + joiner + vocabulary, 312 MiB |
| `ml_ctc` | ru/en/kk/ky/uz | None | Encoder + vocabulary, 214 MiB |
| `ml_ctc_large` | ru/en/kk/ky/uz | None | Encoder + vocabulary, 564 MiB |

The RNN-T files come from a pinned GitHub model release. Multilingual encoders
come from the pinned Hugging Face repositories listed in the source catalog.
Standard ONNX inference requires the prequantized encoder. Its convolutions and
RNN-T decoder/joiner may still use floating point; “INT8 bundle” does not mean
every operator is integer. `quantize` rebuilds an encoder from a separately
provisioned FP32 source and is not needed for normal installation.

The experimental [Candle](candle-backend.md) and [ANE](ane-backend.md) backends
use separate converted assets and have their own precision and compatibility
limits. They are not an alternative download mode for the standard ONNX bundle.

## Download and select

```sh
gigastt download --model-variant rnnt
gigastt transcribe recording.wav --model-variant rnnt
gigastt serve --model-variant rnnt
```

Use an explicit variant when a directory holds more than one model. With no
selection, the CLI detects installed heads (preferring `rnnt`) and chooses
`rnnt` for an empty installation. Selection is per engine/server, not per
request; the OpenAI-compatible `model` field does not hot-swap loaded weights.

Default storage is `~/.gigastt/models/`; `--model-dir` selects another directory.
For `rnnt`, retain all four files:

- `v3_rnnt_encoder_int8.onnx`
- `v3_rnnt_decoder.onnx`
- `v3_rnnt_joint.onnx`
- `v3_vocab.txt`

The E2E head uses `v3_e2e_rnnt_encoder_int8.onnx`,
`v3_e2e_rnnt_decoder.onnx`, `v3_e2e_rnnt_joint.onnx` and
`v3_e2e_rnnt_vocab.txt`. Multilingual heads need `multilingual_ctc.int8.onnx`
or `multilingual_large_ctc.int8.onnx`, plus `multilingual_vocab.txt`.

`download` also fetches the speaker model in default builds. Add
`--skip-diarization` for ASR-only provisioning. Punctuation and VAD are downloaded
by `serve`/`transcribe` when enabled, rather than by a bare `download` command.
Bindings load local files; they do not perform this CLI setup themselves.

## Punctuation, numbers and speaker labels

For `rnnt`, CLI/server `--punctuation auto` and `--itn auto` enable the separate
Russian punctuation model and number normalization. For `e2e_rnnt`, auto leaves
its built-in formatting alone. Library users attach optional processors explicitly.
The multilingual examples in [Languages](languages.md) explicitly disable the
Russian postprocessors; they are not multilingual punctuation models.

| Optional asset | Location | Provisioning/use |
|---|---|---|
| RUPunct ONNX | `punct/` | `--punctuation auto/on` when enabled for the head |
| Silero VAD | `vad/silero_vad.onnx` | `--vad` |
| WeSpeaker | `wespeaker_resnet34.onnx` | `download` unless skipped; diarization requested separately |

Punctuation/VAD download or load failures can fall back to recognition without
that processor. Check readiness/health and logs for the capabilities actually
loaded. A per-request override cannot supply a missing model. Postprocessing
updates joined `text`; word timings and raw word text have a separate contract
in the [API](api.md).

## Offline use and upgrades

Provision the intended ASR and side models on a connected machine, then copy
that directory to the target. For bare recognition:

```sh
gigastt --offline transcribe recording.wav --model-dir /path/to/models \
  --punctuation off --itn off
```

Offline mode prevents downloads; it does not create missing assets. The runtime
also writes optimized graph caches, which need their own disk budget. For `serve`, use
`--optimized-cache-dir` for a writable location separate from read-only weights.
That flag is not exposed by `transcribe`.
See [deployment](deployment.md#air-gapped--offline-installation).

Verify model identity before benchmarking or upgrading. A recognizable filename
and a successful load do not establish that the directory contains current
release weights. Stage upgrades in a separate directory, verify checksums and
compare quality/speed before replacing the deployed model.

## Quality evidence and research

[Benchmarks](benchmarks.md), [Languages](languages.md) and the
[RNN-T quantization comparison](rnnt-quantization.md) retain their exact model,
corpus and hardware context. Do not infer current speed or RAM from model size,
or use a historical measurement as a current-model guarantee.

The [six-head comparison in PR #453](https://github.com/ekhodzitsky/gigastt/pull/453)
also evaluates current Russian v3 CTC/E2E CTC and RUPunct alternatives. Those
research adapters and additional heads are not part of main's named model catalog.
Its ordinary RNN-T and E2E results use the active October weights, but the PR's
code/configuration must accompany any quoted result. Neither a punctuation score
nor a corpus-specific WER establishes a universal default-model improvement.
