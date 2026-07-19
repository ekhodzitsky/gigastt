#!/usr/bin/env bash
# fetch_offline_models.sh — download the model files shipped inside the
# offline all-in-one tarball and the gigastt-model-int8 data deb.
#
# Fetches the pre-quantized INT8 `rnnt` bundle from the pinned gigastt model
# release (the same source `gigastt download --prequantized` uses) plus the
# RUPunct punctuation model from HuggingFace, and verifies every file against
# the SHA-256 pinned in crates/gigastt-core/src/model/mod.rs — the constants
# below MUST mirror RNNT_CHECKSUMS / ModelVariant::encoder_int8_checksum /
# PUNCT_FILES there; bump both places together when re-quantizing.
#
# Usage: scripts/fetch_offline_models.sh <dest-dir>
#
# Output layout consumed by build_offline_bundle.sh and build_model_deb.sh:
#   <dest>/v3_rnnt_encoder_int8.onnx
#   <dest>/v3_rnnt_decoder.onnx
#   <dest>/v3_rnnt_joint.onnx
#   <dest>/v3_vocab.txt
#   <dest>/punct/rupunct_small_int8.onnx
#   <dest>/punct/tokenizer.json
#   <dest>/punct/config.json
#
# Idempotent: a file already present with a matching checksum is not
# re-downloaded.

set -euo pipefail

PREQUANT_BASE="https://github.com/ekhodzitsky/gigastt/releases/download/models-v3-2026-06-22"
PUNCT_BASE="https://huggingface.co/ekhodzitsky/rupunct-small-onnx/resolve/main"

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <dest-dir>" >&2
    exit 2
fi
DEST="$1"
mkdir -p "$DEST/punct"

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

download() {
    local url="$1" out="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -fSL --retry 3 --retry-delay 5 -o "$out" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$out" "$url"
    else
        echo "error: neither curl nor wget found on PATH" >&2
        exit 1
    fi
}

# fetch <url> <dest-file> <pinned-sha256>
fetch() {
    local url="$1" dest="$2" want="$3"
    if [ -f "$dest" ] && [ "$(sha256_of "$dest")" = "$want" ]; then
        echo "ok (cached): ${dest}"
        return 0
    fi
    echo "fetching: ${url}"
    download "$url" "${dest}.partial"
    local got
    got="$(sha256_of "${dest}.partial")"
    if [ "$got" != "$want" ]; then
        rm -f "${dest}.partial"
        echo "error: SHA-256 mismatch for ${dest}" >&2
        echo "  want: ${want}" >&2
        echo "  got:  ${got}" >&2
        exit 1
    fi
    mv "${dest}.partial" "$dest"
    echo "ok: ${dest}"
}

fetch "${PREQUANT_BASE}/v3_rnnt_encoder_int8.onnx" "${DEST}/v3_rnnt_encoder_int8.onnx" \
    "c52665e9d96c4ca3a153c063d2ee9af6c567fe2975ca50fd038b75bbf2f60e7f"
fetch "${PREQUANT_BASE}/v3_rnnt_decoder.onnx" "${DEST}/v3_rnnt_decoder.onnx" \
    "443c3b7bd42b453611618135d6b1e7d9467e5dd97c8a68501da4aa355750c0da"
fetch "${PREQUANT_BASE}/v3_rnnt_joint.onnx" "${DEST}/v3_rnnt_joint.onnx" \
    "fd1d02f45c2ad3d6b67cc149811ad794ab4b020ed49a0a9e2790a8619d1cddd8"
fetch "${PREQUANT_BASE}/v3_vocab.txt" "${DEST}/v3_vocab.txt" \
    "a9143c30844d3c0bee3e9e927e4084774eb1b9eeaafc473b2c4521e4911a7c07"

fetch "${PUNCT_BASE}/rupunct_small_int8.onnx" "${DEST}/punct/rupunct_small_int8.onnx" \
    "b105da023474d98aa13ba18953ae67b04b17bd0595034bc06030c17536893933"
fetch "${PUNCT_BASE}/tokenizer.json" "${DEST}/punct/tokenizer.json" \
    "7ca617388c2092a3a84272025c52bbf3c6db0aee225c0351186295c0b5d3ddc6"
fetch "${PUNCT_BASE}/config.json" "${DEST}/punct/config.json" \
    "6924a8cf41ec2bd3a3aa73a387ae0ccd0aed253ec7cac4d2f53c7d27440891eb"

echo "All offline model files ready in ${DEST}"
