# gigastt offline bundle

A self-contained, **air-gapped** installation of gigastt — the local Russian
speech-to-text server (GigaAM v3 RNN-T, INT8). Everything needed to install
and run is inside this tarball: no HuggingFace, no GitHub, no network at all.

## Contents

```
bin/gigastt                          the server / CLI binary (statically linked onnxruntime)
models/v3_rnnt_encoder_int8.onnx     pre-quantized INT8 Conformer encoder (~305 MiB in the current model bundle)
models/v3_rnnt_decoder.onnx          LSTM prediction network
models/v3_rnnt_joint.onnx            RNN-T joiner
models/v3_vocab.txt                  34-token character vocabulary
models/punct/                        RUPunct punctuation/casing restorer (INT8 + tokenizer + config)
systemd/gigastt.service              hardened systemd unit (systemd 241-compatible)
systemd/gigastt.env                  environment overrides example (all defaults commented)
install.sh                           offline installer (verifies checksums, installs everything)
SHA256SUMS.txt                       SHA-256 of every binary/model payload file (checked by install.sh)
LICENSE / README.md / CHANGELOG.md
```

## Quick start

```sh
mkdir -p gigastt-offline
tar xf gigastt-<version>-offline-<target>.tar.gz -C gigastt-offline
cd gigastt-offline
sudo ./install.sh
sudo systemctl enable --now gigastt
curl --fail http://127.0.0.1:9876/ready
# Wait for HTTP 200 before transcribing; /health can answer during startup.
```

Transcribe a file to verify the full pipeline:

```sh
gigastt --offline transcribe sample.wav --model-dir /usr/share/gigastt/models \
  --punct-model-dir /usr/share/gigastt/models/punct
```

`install.sh` options: `--prefix`, `--model-root`, `--systemd`, `--no-systemd`, `--user`
(run `./install.sh --help`). Without systemd it prints the manual
`gigastt serve` command instead of installing a unit.

## Verify before you install

Verify the tarball before unpacking on the target machine. Checksums and build
provenance are published; minisign signatures are optional when signing is not
configured. See the [release verification guide](https://github.com/ekhodzitsky/gigastt/blob/main/docs/verifying-releases.md):

```sh
# 1. SHA-256 sidecar published next to the tarball on the release page:
sha256sum -c gigastt-<version>-offline-<target>.tar.gz.sha256

# 2. If this release provides a minisign signature:
minisign -Vm gigastt-<version>-offline-<target>.tar.gz -p gigastt.pub

# 3. SLSA build provenance:
gh attestation verify gigastt-<version>-offline-<target>.tar.gz --repo ekhodzitsky/gigastt
```

Inside the bundle, `install.sh` re-verifies every binary/model payload file
(`bin/` + `models/`) against `SHA256SUMS.txt` before copying anything
(corruption check).

## Installed layout

| What | Path |
|---|---|
| Binary | `/usr/local/bin/gigastt` |
| Models | `/usr/share/gigastt/models/` (+ `models/punct/`) |
| systemd unit | `/etc/systemd/system/gigastt.service` |
| Env overrides | `/etc/gigastt/gigastt.env` |
| ORT optimized-graph cache | `/var/cache/gigastt/` (systemd `CacheDirectory=`) |
| Service account | `gigastt` (system user, nologin) |

The unit binds `127.0.0.1:9876` (loopback only), restarts on failure, and
runs with a systemd 241-compatible hardening set (ProtectSystem=strict,
PrivateTmp, NoNewPrivileges, …) — works on Astra Linux, RED OS and ALT.
Model directories are read-only at runtime; logs go to the journal
(`journalctl -u gigastt -f`).

## What is NOT in the bundle

These optional features need model files that are not part of the bundle.
The shipped systemd unit runs with `GIGASTT_OFFLINE=1`, so a missing file is
a fast, instructive error naming the path to fill — never a network attempt
or a connect timeout:

- `--vad` (Silero VAD, off by default),
- speaker diarization (opt-in per request: `?diarization=true` on REST,
  `Configure{diarization}` over WebSocket; needs the speaker model fetched by
  `gigastt download` on a connected machine and copied over),
- the `e2e_rnnt` / `ml_ctc` / `ml_ctc_large` recognition heads
  (fetch on a connected machine with `gigastt download`,
  then copy `~/.gigastt/models/` over).

Prefer the `.deb` packages on Debian-family systems: `gigastt_<ver>_amd64.deb`
(binary + unit) and `gigastt-model-int8_<ver>_all.deb` (this same model set)
are published next to this tarball. An rpm package is intentionally deferred
— RED OS and other rpm-based distributions should use this tarball for now.
