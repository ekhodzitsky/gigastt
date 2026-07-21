#!/usr/bin/env bash
# Regenerate the Opus test fixtures.
#
# Two fixture sets are produced:
#
#   crates/gigastt-core/tests/fixtures/opus/
#     A synthetic two-tone signal encoded to OGG/Opus and ffmpeg's own decode
#     of it (`opus_tone_ffmpeg.pcm`), used to verify the pure-Rust opus-rs
#     decoder against an independent reference implementation (libopus via
#     ffmpeg). The unit test compares our decode against it at the best lag
#     (we do not trim the OpusHead pre-skip; ffmpeg does).
#
#   crates/gigastt/tests/fixtures/opus/
#     Real speech (the golos_00 e2e fixture, 4 s of Russian read speech)
#     transcoded into the Opus containers the e2e tests POST to
#     /v1/transcribe and feed to the CLI: OGG/Opus from a 16 kHz source,
#     a Telegram-voice-style OGG/Opus (48 kHz mono, voip tuning), and a
#     stereo .opus file (browser MediaRecorder style).
#
# Requires ffmpeg with libopus on PATH. Run from the repository root:
#   scripts/generate_opus_fixtures.sh
set -euo pipefail

CORE_DIR="crates/gigastt-core/tests/fixtures/opus"
E2E_DIR="crates/gigastt/tests/fixtures/opus"
SPEECH="crates/gigastt/tests/fixtures/golos_00.wav"
TONE="crates/gigastt-core/tests/fixtures/telephony/tone_src.wav"
mkdir -p "$CORE_DIR" "$E2E_DIR"

# ── Core: synthetic two-tone (3 s, 16 kHz mono source) ──────────────────────

# OGG/Opus encode (the OpusHead input-rate field says 16 kHz; per RFC 7845
# the decode rate is always 48 kHz).
ffmpeg -y -v error -i "$TONE" -ac 1 -c:a libopus -b:a 24k "$CORE_DIR/opus_tone.ogg"

# Independent reference: ffmpeg's own decode of that OGG/Opus, resampled to
# 16 kHz mono PCM16 (the rate our public decode path returns).
ffmpeg -y -v error -i "$CORE_DIR/opus_tone.ogg" -f s16le -acodec pcm_s16le \
  -ar 16000 -ac 1 "$CORE_DIR/opus_tone_ffmpeg.pcm"

# ── E2E: real speech transcodes ─────────────────────────────────────────────

# OGG/Opus from a 16 kHz mono source.
ffmpeg -y -v error -i "$SPEECH" -ar 16000 -ac 1 -c:a libopus -b:a 24k \
  "$E2E_DIR/speech_16k.ogg"

# Telegram voice style: OGG/Opus 48 kHz mono, voip tuning.
ffmpeg -y -v error -i "$SPEECH" -ar 48000 -ac 1 -c:a libopus -application voip \
  "$E2E_DIR/speech_telegram.ogg"

# Browser MediaRecorder style: .opus file, 48 kHz stereo.
ffmpeg -y -v error -i "$SPEECH" -ar 48000 -ac 2 -c:a libopus \
  "$E2E_DIR/speech.opus"

echo "Wrote fixtures:"
ls -l "$CORE_DIR" "$E2E_DIR"
