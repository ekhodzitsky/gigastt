#!/usr/bin/env bash
# install.sh — offline installer shipped inside gigastt-<ver>-offline-<target>.tar.gz.
#
# Installs the gigastt binary, the pre-quantized INT8 model, the punctuation
# model, and (on systemd hosts) the gigastt.service unit — with no network
# access whatsoever. Everything this script needs is inside the tarball.
#
# Usage (from the unpacked tarball directory):
#   sudo ./install.sh [options]
#
# Options:
#   --prefix DIR      Binary install prefix        (default: /usr/local)
#   --model-root DIR  Model install root           (default: /usr/share/gigastt)
#   --systemd         Force-install the systemd unit + gigastt user
#   --no-systemd      Skip the systemd unit entirely (manual `gigastt serve`)
#   --user NAME       Service account to create/use (default: gigastt)
#   -h, --help        Show this help
#
# Defaults auto-detect systemd: the unit is installed when systemctl exists.

set -euo pipefail

PREFIX="/usr/local"
MODEL_ROOT="/usr/share/gigastt"
SYSTEMD="auto"
SVC_USER="gigastt"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --prefix) PREFIX="$2"; shift 2 ;;
        --model-root) MODEL_ROOT="$2"; shift 2 ;;
        --systemd) SYSTEMD="yes"; shift ;;
        --no-systemd) SYSTEMD="no"; shift ;;
        --user) SVC_USER="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *)
            echo "error: unknown option '$1' (see --help)" >&2
            exit 2
            ;;
    esac
done

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$(id -u)" -ne 0 ]; then
    echo "error: run as root (sudo ./install.sh) — installing to ${PREFIX} and" >&2
    echo "       ${MODEL_ROOT} requires elevated privileges" >&2
    exit 1
fi

if [ "$SYSTEMD" = "auto" ]; then
    if command -v systemctl >/dev/null 2>&1 && [ -d /etc/systemd/system ]; then
        SYSTEMD="yes"
    else
        SYSTEMD="no"
    fi
fi

# 1. Integrity: verify every payload file against the sums generated at
#    bundle-build time (covers bin/ and models/). This is a corruption check;
#    for supply-chain verification check the release-page .minisig signature
#    first (see README-OFFLINE.md).
echo ">> verifying bundle integrity (SHA256SUMS.txt)"
if command -v sha256sum >/dev/null 2>&1; then
    (cd "$BUNDLE_DIR" && sha256sum -c SHA256SUMS.txt)
else
    (cd "$BUNDLE_DIR" && shasum -a 256 -c SHA256SUMS.txt)
fi

# 2. Binary.
echo ">> installing binary to ${PREFIX}/bin/gigastt"
install -d "${PREFIX}/bin"
install -m 0755 "${BUNDLE_DIR}/bin/gigastt" "${PREFIX}/bin/gigastt"

# 3. Models (world-readable; the service runs unprivileged and only reads).
echo ">> installing models to ${MODEL_ROOT}/models"
install -d "${MODEL_ROOT}/models/punct"
install -m 0644 \
    "${BUNDLE_DIR}/models/v3_rnnt_encoder_int8.onnx" \
    "${BUNDLE_DIR}/models/v3_rnnt_decoder.onnx" \
    "${BUNDLE_DIR}/models/v3_rnnt_joint.onnx" \
    "${BUNDLE_DIR}/models/v3_vocab.txt" \
    "${MODEL_ROOT}/models/"
install -m 0644 \
    "${BUNDLE_DIR}/models/punct/rupunct_small_int8.onnx" \
    "${BUNDLE_DIR}/models/punct/tokenizer.json" \
    "${BUNDLE_DIR}/models/punct/config.json" \
    "${MODEL_ROOT}/models/punct/"

# 4. systemd unit + service account.
if [ "$SYSTEMD" = "yes" ]; then
    echo ">> installing systemd unit"
    if ! id "$SVC_USER" >/dev/null 2>&1; then
        NOLOGIN_SHELL="/usr/sbin/nologin"
        [ -x "$NOLOGIN_SHELL" ] || NOLOGIN_SHELL="/sbin/nologin"
        [ -x "$NOLOGIN_SHELL" ] || NOLOGIN_SHELL="/bin/false"
        useradd --system --no-create-home --shell "$NOLOGIN_SHELL" "$SVC_USER"
        echo "   created system user '${SVC_USER}'"
    fi

    # Point the unit at the chosen prefix/model root when they differ from
    # the deb defaults (/usr/bin, /usr/share/gigastt).
    sed \
        -e "s|/usr/bin/gigastt|${PREFIX}/bin/gigastt|" \
        -e "s|/usr/share/gigastt|${MODEL_ROOT}|g" \
        -e "s|^User=gigastt$|User=${SVC_USER}|" \
        -e "s|^Group=gigastt$|Group=${SVC_USER}|" \
        "${BUNDLE_DIR}/systemd/gigastt.service" > /etc/systemd/system/gigastt.service
    chmod 0644 /etc/systemd/system/gigastt.service

    install -d /etc/gigastt
    if [ ! -f /etc/gigastt/gigastt.env ]; then
        install -m 0644 "${BUNDLE_DIR}/systemd/gigastt.env" /etc/gigastt/gigastt.env
        echo "   installed /etc/gigastt/gigastt.env (offline mode on; other defaults commented out)"
    else
        echo "   kept existing /etc/gigastt/gigastt.env"
    fi

    # Tolerate systemd being installed but not running (chroot, containers,
    # debootstrap) — the unit file is in place either way.
    if ! systemctl daemon-reload 2>/dev/null; then
        echo "   (systemd not running — skipped daemon-reload)"
    fi
fi

cat <<EOF

gigastt installed successfully (fully offline — no downloads needed).

EOF
if [ "$SYSTEMD" = "yes" ]; then
    cat <<EOF
Start the service:
  systemctl enable --now gigastt
  systemctl status gigastt
  curl http://127.0.0.1:9876/health

Logs:  journalctl -u gigastt -f
EOF
else
    cat <<EOF
Start manually:
  ${PREFIX}/bin/gigastt serve --model-dir ${MODEL_ROOT}/models --punct-model-dir ${MODEL_ROOT}/models/punct
  curl http://127.0.0.1:9876/health
EOF
fi
