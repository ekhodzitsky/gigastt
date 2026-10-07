#!/usr/bin/env bash
# Runner images have usable package indexes; avoid refreshing unrelated feeds
# unless installation actually needs it. Bound network waits and package locks.
set -euo pipefail

# The runner's Azure HTTP mirror stalled indefinitely while fetching indexes.
# Keep the runner's repository/suite selection, using Ubuntu's HTTPS mirror.
if [ -f /etc/apt/apt-mirrors.txt ]; then
  sudo sed -i 's|http://azure.archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' \
    /etc/apt/apt-mirrors.txt
fi

apt_bounded() {
  timeout --kill-after=10s 120s sudo env DEBIAN_FRONTEND=noninteractive \
    apt-get -o DPkg::Lock::Timeout=30 -o Acquire::Retries=2 \
    -o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20 "$@"
}

if ! apt_bounded install -y --no-install-recommends protobuf-compiler pkg-config libssl-dev; then
  apt_bounded update
  apt_bounded install -y --no-install-recommends protobuf-compiler pkg-config libssl-dev
fi
protoc --version
pkg-config --exists openssl
