# GigaAM v4 RNNT FP32 CPU bundle

The versioned `gigaam-v4-rnnt-fp32-cpu-v1` bundle contains two separate
verified model directories, `file/` and `native/`, each with its own
manifest. Both include the raw and e2e heads. Use `file/` alone for file
transcription; the native WebSocket listener additionally takes `native/`.
The measured v1 archive is 1.22 GB compressed and 1.93 GB unpacked
(decimal GB), excluding the application and the separate MKL runtime. It never replaces the default v3 INT8 models or
changes what `gigastt download` does.

This is the initial opt-in Linux x86-64 GNU/glibc profile, measured with one
CPU inference thread. It requires a `gigastt` binary built with the
`v4-mkl` feature and the separately prepared, exact
[oneMKL runtime](v4-pack-validation.md#optional-experimental-cpu-acoustic-frontend).
Nothing in this bundle downloads, bundles, or installs Intel binaries.
INT8/weight-only, VAD, other execution providers and default migration
are not part of this profile. Build the feature explicitly if your binary
does not include it: `cargo build --locked --release -p gigastt --features v4-mkl`.

## Build from qualified local exports

Use Python 3.11+ on Linux/glibc with a filesystem supporting atomic
no-replace rename. Give the packager the directories holding the exact
[full-context](v4-pack-validation.md) and
[cached streaming](v4-streaming-pack.md) role files. It does not export,
quantize, run inference, or modify those source directories.

```sh
python3 scripts/package_v4_rnnt.py --offline build \
  --file-dir /path/to/qualified-file \
  --stream-dir /path/to/qualified-native \
  --output-dir /path/to/new-output
```

The output contains `gigastt-gigaam-v4-rnnt-fp32-cpu-v1.tar.gz` and
`SHA256SUMS.txt`. Fixed USTAR member order, headers and zero padding and a
gzip header without a timestamp or input name produce repeatable bytes with
the same Python/zlib toolchain. Each required export is hashed while writing.
The archive contains the versioned
[catalogue](../models/gigaam-v4-rnnt-fp32-cpu-v1.json) as
`PROVENANCE.json`, per-file `SHA256SUMS`, and the upstream MIT notice.
The streaming state/config metadata is pinned and remains unmodified; no
external tensor files, fused advance graph, model checkpoint, audio,
or machine-specific data are included.

## Verify and install

There is no built-in v4 release URL. Use a local archive or an explicit HTTPS
URL supplied by a trusted distributor. Get the archive SHA-256 from trusted
release metadata; a matching downloaded checksum alone does not authenticate
the distributor. If the distributor provides a minisign signature, use its
independently trusted public key, not a key obtained only from the same
untrusted mirror. See [release verification](verifying-releases.md).

```sh
# From an authenticated channel (64 hexadecimal characters):
# ARCHIVE_SHA256=...
python3 scripts/package_v4_rnnt.py --offline install \
  --archive /path/to/gigastt-gigaam-v4-rnnt-fp32-cpu-v1.tar.gz \
  --sha256 "$ARCHIVE_SHA256" --destination "$HOME/.local/share/gigastt/v4-rnnt-fp32-cpu-v1"

# Online alternative: explicitly opt in to the distributor's HTTPS URL.
python3 scripts/package_v4_rnnt.py download \
  --url "$ARCHIVE_URL" --sha256 "$ARCHIVE_SHA256" \
  --signature /path/to/archive.minisig --public-key /path/to/trusted-minisign.pub \
  --destination "$HOME/.local/share/gigastt/v4-rnnt-fp32-cpu-v1"
```

Supply both signature options together or omit both. The helper invokes
`minisign` only when requested. Its built-in catalogue checks every
required member regardless of the caller's outer archive hash or signature.
It rejects extra members, links, incorrect lengths, corrupt bytes and
unsupported compression/metadata. The destination must not already exist,
including as an empty directory. Publication is a single atomic no-replace
rename after the whole bundle verifies. Use a versioned destination in a
directory you own; do not point it at the legacy model cache. No source
archives or previous installations are removed. `--offline` (or
`GIGASTT_OFFLINE=1`) rejects `download`; local `build` and `install`
never access the network.

The existing signature format and release procedure apply to any eventual
publication. A locally generated test signature or a Git tag is not a
production-signed published release.

## Run explicitly

```sh
PACK="$HOME/.local/share/gigastt/v4-rnnt-fp32-cpu-v1"
# MKL_RUNTIME is your separately constructed, verified runtime directory.
gigastt --offline transcribe recording.wav --model-dir "$PACK/file" \
  --model-family gigaam_v4 --model-head rnnt --model-precision fp32 \
  --transcript-style e2e --v4-mkl-runtime-dir "$MKL_RUNTIME" --format json

gigastt --offline serve --host 127.0.0.1 --model-dir "$PACK/file" \
  --model-family gigaam_v4 --model-head rnnt --model-precision fp32 \
  --transcript-style e2e --v4-mkl-runtime-dir "$MKL_RUNTIME" \
  --v4-streaming-model-dir "$PACK/native" --v4-streaming-future-context-ms 320
```

The file route has a 30-second limit; it does not silently split longer audio.
Use the explicit [native WebSocket protocol](api.md#websocket--streaming) for 16 kHz mono PCM16,
contexts 160/320/640 ms, and explicit Stop. The option does not change legacy
WebSocket negotiation or add automatic endpointing.

## What was measured

On one Linux/glibc test host, the exact CPU profile passed 20 file requests
and 12 real-time-paced native WebSocket sessions in raw and e2e styles.
File servers took about 4.32 s to become ready and peaked at 2.28 GiB RSS;
the dual-pack native server took 10.25 s and peaked at 3.37 GiB RSS.
REST request wall time was 0.100–0.101 times audio duration. The first
lexical WebSocket partial appeared after 1.18–4.74 s, and Stop-to-Final
was 70–198 ms on those fixed cases. Cold means a fresh process with local
files, not a cold disk. This small controlled cohort does not establish
general accuracy, language-switching quality, or machine-independent speed.

**Encoder context is not a word-visibility bound.** The native publisher
waits for a complete word boundary or Stop; sparse speech can hold a
trailing word even after raw token events appear. In two separate diagnostic
replays, individual protected words appeared 10–28 s later than in older v3
wire captures under a different transport. No unsupported target
`да`/`не` was observed in those four replays, but one human-labelled
source had a lexical substitution in both styles, and the old adverse
interval had no comparable native lexical event. The known phrase-loss examples were not recovered, and the
proposed weight-only conversion did not qualify. This bundle is
not a correction for the original v3 issues, not an INT8 release, and not a
default change.
