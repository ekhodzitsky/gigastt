# Explicit FP32 E2E RNNT inference on CPU

The optional FP32 pack uses the original E2E RNNT encoder weights and runs on
CPU without training. Default downloads remain INT8. Select FP32 explicitly
with a manifest in a separate model directory.

FP32 can recover phrases that INT8 omits, but can also omit words that INT8
recognizes, including negation. It is not a general fix for missing speech.
Compare both profiles on your recordings before choosing one.

## Install

Use Python 3.11 or newer and a build containing explicit FP32 manifest support.
Build the CPU binary with `cargo build --release -p gigastt` without accelerator
features, then substitute `target/release/gigastt` for `gigastt` below.

```sh
python3 scripts/install_e2e_fp32.py \
  --destination "$HOME/.gigastt/models-e2e-fp32"
```

The installer downloads four original model assets from
[`istupakov/gigaam-v3-onnx`](https://huggingface.co/istupakov/gigaam-v3-onnx/tree/322c3b29492673eb7d0b434bfa9dfb8653e34d02),
pinned to revision `322c3b29492673eb7d0b434bfa9dfb8653e34d02`. It verifies sizes
and SHA-256 hashes before publishing the staged directory. The destination must
not exist. Failed verification leaves no partial destination.

To install without downloading, supply those exact files locally:

```sh
python3 scripts/install_e2e_fp32.py \
  --source-dir /path/to/original-onnx-files \
  --destination "$HOME/.gigastt/models-e2e-fp32"
```

Local installation copies and verifies the files. It does not modify the
sources. The four model files occupy 892,410,694 bytes (about 892 MB), versus
about 327 MB for the default INT8 E2E bundle. These are disk sizes, not peak RAM.
Allow additional space for optimized graphs. Start with one server pool slot
and measure memory and latency on your CPU.

## Run

```sh
gigastt --offline transcribe recording.wav \
  --model-dir "$HOME/.gigastt/models-e2e-fp32" \
  --model-variant e2e_rnnt \
  --file-window-concurrency 1 --punctuation off --itn off

gigastt --offline serve \
  --model-dir "$HOME/.gigastt/models-e2e-fp32" \
  --model-variant e2e_rnnt --execution-provider cpu \
  --pool-size 1 --punctuation off --itn off
```

The single-file command uses the build's default execution provider and has no
`--execution-provider` flag. Use the CPU build above. The server supports an
explicit `--execution-provider cpu` selection.

E2E RNNT already supplies punctuation and text normalization. These commands
disable additional postprocessing and its optional models. For comparisons,
also omit hotwords and keep the decoded audio and window boundaries identical.
`GET /v1/models` reports the selected encoder precision and execution provider.

The first run may create an optimized cache in the model directory. Keep a
separate cache for each pack, including when overriding the server's
`--optimized-cache-dir`.

## Manifest contract

The installer writes `manifest.toml`:

```toml
architecture = "e2e_rnnt"
precision = "fp32"

[files]
encoder = "v3_e2e_rnnt_encoder.onnx"
decoder = "v3_e2e_rnnt_decoder.onnx"
joint = "v3_e2e_rnnt_joint.onnx"
vocab = "v3_e2e_rnnt_vocab.txt"
```

Explicit FP32 selects `files.encoder`, even if `encoder_int8` is also declared.
Omitting precision retains the INT8 policy: `encoder_int8` must be present and
its file must exist. A missing INT8 file never causes an FP32 fallback. Unknown
precision values, an incompatible architecture and missing required files
produce errors. Explicit FP32 currently supports `e2e_rnnt` only; validation
covers the CPU execution provider.

Do not rename FP32 weights to a published `_int8.onnx` basename. Published model
names retain their pinned checksum checks. `gigastt download` provisions the
default INT8 bundles; the separate installer creates the opt-in FP32 pack.
When pointed at a complete explicit FP32 pack, `download` validates its files
without replacing them with INT8 weights.

Rust callers constructing `ModelManifest` directly must supply its new
`precision` field (`ModelPrecision::Int8` retains the previous policy).
Existing manifests without that field continue to parse with INT8 precision.

## Recognition behavior

FP32 file transcription uses the original checkpoint's feature extraction,
one encoder pass and greedy RNN-T decoding per window. Default INT8 file
transcription keeps its existing feature extraction and greedy decoder.
Streaming retains its existing feature extraction and decoder state handling
in both modes. File checks do not establish streaming quality.

These are two recognition profiles with different precision and feature
extraction. A difference between their transcripts cannot be attributed solely
to quantization. On the existing diagnostic inputs, FP32 restores
“молодёжной политики” in one recording but loses “не было” in another where
INT8 retains it. The missing negation also occurs in the original PyTorch
checkpoint under the matched CPU conditions examined during investigation.
Neither profile preserves every phrase.

Follow the [performance acceptance policy](performance-regressions.md) before
deploying a changed backend or dependency. CPU functional checks do not validate
accelerator behavior or establish latency and memory equivalence.
