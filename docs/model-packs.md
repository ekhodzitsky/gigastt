# Standalone multilingual small model pack

`scripts/package_small_multilingual.py` builds and installs the validated
selective-precision small multilingual model as a separate model directory.
It uses the Python standard library (Python 3.11 or newer).

The pack keeps MatMul weights in signed, per-channel reduced-range INT8 and
the 51 convolutions in FP32. The encoder is 320,394,571 bytes. This is a
conversion of the existing pretrained weights, with no training or calibration
dataset. Quality measurements and limitations are in the
[quantization report](../benchmark/results/multilingual_small_quantization_20261001/README.md).

## Package contents

The installed directory contains exactly these three files:

```text
multilingual_small_selective.int8.onnx
multilingual_vocab.txt
manifest.toml
```

The manifest selects architecture `ml_ctc` and names the custom encoder in both
`files.encoder` and `files.encoder_int8`. There is no original-model companion,
decoder, joiner, or prebuilt optimized cache. The archive additionally contains
`SHA256SUMS`, which is verified during installation and then omitted from the
runtime directory. The server may create its optimized cache on first startup.

The validated encoder SHA-256 is
`18fa5fab6ece0123d7813f7abc8da129530f6d1a7da17c85c4b3948f27b8e3e0`.
The vocabulary SHA-256 is
`4d130287892e1099fedfb3f93c4b4cf8a263151158801680b28977d1be4133f4`.

## Build from local weights

The builder requires those exact source hashes. It never downloads or modifies
weights. Supply the local paths to the validated encoder and original vocabulary:

```sh
python3 scripts/package_small_multilingual.py build \
  --encoder /path/to/validated/multilingual_ctc.int8.onnx \
  --vocab /path/to/multilingual_vocab.txt \
  --output /path/to/gigastt-small-multilingual-selective.tar.gz
```

The builder verifies the finished archive before publishing it and writes a
`.tar.gz.sha256` sidecar. Existing outputs are rejected. File ordering, tar
owners, modes, timestamps, and gzip timestamp/name are fixed. Repeated builds
with the same Python/zlib toolchain produce identical bytes; gzip implementations
may differ across toolchains, so always use the digest of the actual archive.

## Install locally or from HTTPS

Installation requires the expected archive SHA-256 from a trusted source.
For the locally validated archive, it is
`2de9e00e954b2bd4a0135f3a1036a44c224c42ca974136af358335459fd38bad`:

```sh
python3 scripts/package_small_multilingual.py install \
  --archive /path/to/gigastt-small-multilingual-selective.tar.gz \
  --sha256 2de9e00e954b2bd4a0135f3a1036a44c224c42ca974136af358335459fd38bad \
  --destination "$HOME/.gigastt/models-small-selective"
```

For an archive hosted by your organization, replace `--archive` with
`--url "$PACK_URL"` and supply its independently obtained expected digest.
Only HTTPS URLs and HTTPS redirects are accepted; embedded URL credentials
are rejected. No public hosting URL is provided by this tooling.

The installer checks the archive digest, every payload digest, the fixed manifest,
member types, names, and size limits. It rejects duplicate entries, path traversal,
symlinks, hard links, and unexpected files. It stages all verified files in a
sibling directory, then publishes the directory with one rename. The destination
must be absent or an empty real directory; existing nonempty installations are
never overwritten. Directory publication uses POSIX rename semantics.

## Start the server

Use a build that supports standalone `manifest.toml` packs during the CLI's
model-presence check. Older binaries may try to fetch the default encoder even
when this custom manifest is present; update the binary instead of adding a
second model to the pack. A present manifest is authoritative: missing declared
files, a missing INT8 encoder, malformed TOML, or an explicit model variant that
conflicts with its architecture cause a clear error instead of a download fallback.

```sh
gigastt serve --offline \
  --model-dir "$HOME/.gigastt/models-small-selective" \
  --model-variant ml_ctc --execution-provider cpu \
  --punctuation off --itn off
```

This uses the separate pack directory and leaves the default model installation
unchanged. Punctuation and ITN are disabled to match the multilingual benchmark.

Run the packaging checks with:

```sh
python3 -m unittest discover -s scripts -p test_package_small_multilingual.py
```

## Real telephone evidence

The simulated-channel gains do not establish real-call readiness. The
[real-call comparison](../benchmark/results/multilingual_real_calls_20261001/README.md)
found unchanged full-call WER for the selective small model on two Kazakh
recordings; enabling the existing VAD helped some cases but quality remained
poor. No verified Uzbek real-call evaluation set was acquired. Keep these limits
separate from the successful packaging and startup checks.

## Verified local build

This checkout passed 1,119 workspace unit tests (39 ignored), clippy and format
checks. A fresh release binary loaded the installed archive without the original
encoder companion and reproduced all 111 frozen diagnostic transcripts exactly.
See [standalone startup evidence](../benchmark/results/multilingual_small_quantization_20261001/standalone_smoke.json)
and [archive validation](../benchmark/results/multilingual_small_quantization_20261001/package_validation.json).

The ready-to-run local installation can be started with:

```sh
~/.cache/gigastt-small-package/gigastt-standalone-b875baf0e31371bd --offline serve \
  --model-dir ~/.cache/gigastt-small-package/installed-small \
  --model-variant ml_ctc --execution-provider cpu \
  --punctuation off --itn off
```

The binary SHA-256 is
`b875baf0e31371bd3e4aa8272616c8749d6681081ff8a674e495e1382197786a`.
No public release or default-model switch was performed.
