#!/usr/bin/env bash
# build_model_deb.sh — build gigastt-model-int8_<version>_all.deb, the
# architecture-independent data package holding the pre-quantized INT8 model
# and the punctuation model under /usr/share/gigastt/models/.
#
# Companion to the `gigastt` deb built by cargo-deb (see
# [package.metadata.deb] in crates/gigastt/Cargo.toml). A data-only package
# has no cargo involvement, so plain dpkg-deb is used.
#
# Model files must be fetched beforehand (SHA-256-verified) by
# scripts/fetch_offline_models.sh.
#
# Usage:
#   scripts/build_model_deb.sh \
#       --version 2.12.0 \
#       --models /path/to/offline-models \
#       --output gigastt-model-int8_2.12.0_all.deb

set -euo pipefail

PACKAGE="gigastt-model-int8"
VERSION=""
MODELS=""
OUTPUT=""

usage() {
    sed -n '2,18p' "$0"
    exit "${1:-2}"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --version) VERSION="$2"; shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "error: unknown option '$1'" >&2; usage ;;
    esac
done

[ -n "$VERSION" ] || { echo "error: --version is required" >&2; usage; }
[ -n "$MODELS" ] || { echo "error: --models is required" >&2; usage; }
[ -n "$OUTPUT" ] || OUTPUT="${PACKAGE}_${VERSION}_all.deb"
command -v dpkg-deb >/dev/null 2>&1 || { echo "error: dpkg-deb not found (install dpkg)" >&2; exit 1; }

MODEL_FILES=(v3_rnnt_encoder_int8.onnx v3_rnnt_decoder.onnx v3_rnnt_joint.onnx v3_vocab.txt)
PUNCT_FILES=(rupunct_small_int8.onnx tokenizer.json config.json)
for f in "${MODEL_FILES[@]}"; do
    [ -f "${MODELS}/${f}" ] || { echo "error: missing ${MODELS}/${f} (run scripts/fetch_offline_models.sh first)" >&2; exit 1; }
done
for f in "${PUNCT_FILES[@]}"; do
    [ -f "${MODELS}/punct/${f}" ] || { echo "error: missing ${MODELS}/punct/${f}" >&2; exit 1; }
done

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
DATA_DIR="${STAGE}/usr/share/gigastt/models"
mkdir -p "${STAGE}/DEBIAN" "${DATA_DIR}/punct"

for f in "${MODEL_FILES[@]}"; do
    install -m 0644 "${MODELS}/${f}" "$DATA_DIR/"
done
for f in "${PUNCT_FILES[@]}"; do
    install -m 0644 "${MODELS}/punct/${f}" "${DATA_DIR}/punct/"
done

INSTALLED_SIZE="$(du -sk "$STAGE" | awk '{print $1}')"

cat > "${STAGE}/DEBIAN/control" <<EOF
Package: ${PACKAGE}
Version: ${VERSION}
Section: sound
Priority: optional
Architecture: all
Installed-Size: ${INSTALLED_SIZE}
Maintainer: Evgeny Khodzitsky <e@khodzitsky.ru>
Homepage: https://github.com/ekhodzitsky/gigastt
Description: GigaAM v3 INT8 speech-recognition model data for gigastt
 Pre-quantized INT8 model files for the gigastt speech-to-text server:
 the rnnt head (encoder + decoder + joiner + vocab) plus the RUPunct
 punctuation/casing model, installed under /usr/share/gigastt/models/.
 .
 With this package installed, gigastt starts with no network access —
 suitable for air-gapped deployments. Ships the same SHA-256-verified
 files as the offline tarball and \`gigastt download --prequantized\`.
EOF

# md5sums enable `dpkg --verify` on the installed payload.
(cd "$STAGE" && find usr -type f | LC_ALL=C sort | xargs md5sum > DEBIAN/md5sums) 2>/dev/null \
    || (cd "$STAGE" && find usr -type f | LC_ALL=C sort | xargs md5 -r > DEBIAN/md5sums)

dpkg-deb --build --root-owner-group "$STAGE" "$OUTPUT"
ls -la "$OUTPUT"
echo "model data deb ready: ${OUTPUT}"
