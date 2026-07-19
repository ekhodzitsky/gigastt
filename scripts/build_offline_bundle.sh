#!/usr/bin/env bash
# build_offline_bundle.sh — assemble gigastt-<version>-offline-<suffix>.tar.gz:
# the air-gapped all-in-one installer (binary + pre-quantized INT8 model +
# punctuation model + systemd unit + install.sh + README-OFFLINE.md).
#
# Run by .github/workflows/release.yml for the Linux targets; also usable
# locally. Model files must be fetched beforehand (and are SHA-256-verified)
# by scripts/fetch_offline_models.sh.
#
# Usage:
#   scripts/build_offline_bundle.sh \
#       --version 2.12.0 \
#       --suffix x86_64-unknown-linux-gnu \
#       --binary target/x86_64-unknown-linux-gnu/release/gigastt \
#       --models /path/to/offline-models \
#       --output dist
#
# Emits <output>/gigastt-<version>-offline-<suffix>.tar.gz plus a .sha256
# sidecar, mirroring the other release assets.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VERSION=""
SUFFIX=""
BINARY=""
MODELS=""
OUTPUT="."

usage() {
    sed -n '2,22p' "$0"
    exit "${1:-2}"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --version) VERSION="$2"; shift 2 ;;
        --suffix) SUFFIX="$2"; shift 2 ;;
        --binary) BINARY="$2"; shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "error: unknown option '$1'" >&2; usage ;;
    esac
done

for var in VERSION SUFFIX BINARY MODELS; do
    if [ -z "${!var}" ]; then
        echo "error: --$(echo "$var" | tr '[:upper:]' '[:lower:]') is required" >&2
        usage
    fi
done

[ -f "$BINARY" ] || { echo "error: binary not found: $BINARY" >&2; exit 1; }

MODEL_FILES=(v3_rnnt_encoder_int8.onnx v3_rnnt_decoder.onnx v3_rnnt_joint.onnx v3_vocab.txt)
PUNCT_FILES=(rupunct_small_int8.onnx tokenizer.json config.json)
for f in "${MODEL_FILES[@]}"; do
    [ -f "${MODELS}/${f}" ] || { echo "error: missing model file ${MODELS}/${f} (run scripts/fetch_offline_models.sh first)" >&2; exit 1; }
done
for f in "${PUNCT_FILES[@]}"; do
    [ -f "${MODELS}/punct/${f}" ] || { echo "error: missing model file ${MODELS}/punct/${f}" >&2; exit 1; }
done

ASSET="gigastt-${VERSION}-offline-${SUFFIX}.tar.gz"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "${STAGE}/bin" "${STAGE}/models/punct" "${STAGE}/systemd"

install -m 0755 "$BINARY" "${STAGE}/bin/gigastt"
for f in "${MODEL_FILES[@]}"; do
    install -m 0644 "${MODELS}/${f}" "${STAGE}/models/"
done
for f in "${PUNCT_FILES[@]}"; do
    install -m 0644 "${MODELS}/punct/${f}" "${STAGE}/models/punct/"
done

install -m 0644 "${REPO_ROOT}/packaging/systemd/gigastt.service" "${REPO_ROOT}/packaging/systemd/gigastt.env" "${STAGE}/systemd/"
install -m 0755 "${REPO_ROOT}/packaging/offline/install.sh" "${STAGE}/install.sh"
install -m 0644 "${REPO_ROOT}/packaging/offline/README-OFFLINE.md" "${STAGE}/README-OFFLINE.md"
for f in LICENSE README.md CHANGELOG.md; do
    [ -f "${REPO_ROOT}/${f}" ] && install -m 0644 "${REPO_ROOT}/${f}" "${STAGE}/${f}"
done

# In-bundle integrity manifest: install.sh verifies this before copying.
# Covers the executable payload only (bin/ + models/).
(cd "$STAGE" && find bin models -type f | LC_ALL=C sort | xargs sha256sum > SHA256SUMS.txt)

mkdir -p "$OUTPUT"
tar -C "$STAGE" -czf "${OUTPUT}/${ASSET}" .
if command -v sha256sum >/dev/null 2>&1; then
    (cd "$OUTPUT" && sha256sum "$ASSET" > "${ASSET}.sha256")
else
    (cd "$OUTPUT" && shasum -a 256 "$ASSET" > "${ASSET}.sha256")
fi

ls -la "${OUTPUT}/${ASSET}" "${OUTPUT}/${ASSET}.sha256"
echo "offline bundle ready: ${OUTPUT}/${ASSET}"
