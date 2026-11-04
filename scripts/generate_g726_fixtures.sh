#!/usr/bin/env bash
# Independent G.726 WAVE and decoded PCM references for the audio unit tests.
# Run from the repository root. Originally generated with FFmpeg 7.0.2-static.
set -euo pipefail
fixture_dir=crates/gigastt-core/tests/fixtures/telephony
for rate in 16 24 32 40; do
  ffmpeg -y -v error -f lavfi \
    -i 'sine=frequency=440:sample_rate=8000:duration=3.12' \
    -c:a g726 -b:a "${rate}k" "$fixture_dir/g726_${rate}k.wav"
  ffmpeg -y -v error -i "$fixture_dir/g726_${rate}k.wav" \
    -c:a pcm_s16le "$fixture_dir/g726_${rate}k_reference.wav"
done
