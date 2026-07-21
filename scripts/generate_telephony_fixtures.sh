#!/usr/bin/env bash
# Regenerate the telephony-codec test fixtures.
#
# Two fixture sets are produced:
#
#   crates/gigastt-core/tests/fixtures/telephony/
#     A synthetic two-tone signal and its G.722 roundtrip, used to verify the
#     G.722 decoder against an independent reference implementation (ffmpeg's
#     libavcodec G.722). `g722_tone_ffmpeg.pcm` is ffmpeg's own decode of
#     `g722_tone.wav`; the unit test compares our decode against it.
#
#   crates/gigastt/tests/fixtures/telephony/
#     Real speech (the golos_00 e2e fixture, 4 s of Russian read speech)
#     transcoded into the containers and raw streams the e2e tests POST to
#     /v1/transcribe: G.711 A-law / μ-law WAV (symphonia-decoded), G.722 WAV
#     (our fallback decoder), and headerless .alaw / .ulaw / .g722 streams
#     (the `?codec=` raw path).
#
# Requires ffmpeg on PATH. Run from the repository root:
#   scripts/generate_telephony_fixtures.sh
set -euo pipefail

CORE_DIR="crates/gigastt-core/tests/fixtures/telephony"
E2E_DIR="crates/gigastt/tests/fixtures/telephony"
SPEECH="crates/gigastt/tests/fixtures/golos_00.wav"
mkdir -p "$CORE_DIR" "$E2E_DIR"

# ── Core: synthetic two-tone source (3 s @ 16 kHz mono) ─────────────────────

ffmpeg -y -v error \
  -f lavfi -i "sine=frequency=440:duration=3" \
  -f lavfi -i "sine=frequency=1200:duration=3" \
  -filter_complex "[0:a][1:a]amix=inputs=2:normalize=0,volume=0.5" \
  -ar 16000 -ac 1 -c:a pcm_s16le "$CORE_DIR/tone_src.wav"

# G.722 WAV (format tag 0x0064) as written by ffmpeg.
ffmpeg -y -v error -i "$CORE_DIR/tone_src.wav" -c:a g722 "$CORE_DIR/g722_tone.wav"

# Independent reference: ffmpeg's own decode of that G.722 WAV back to PCM16.
ffmpeg -y -v error -i "$CORE_DIR/g722_tone.wav" -f s16le -acodec pcm_s16le \
  "$CORE_DIR/g722_tone_ffmpeg.pcm"

# ── E2E: real speech transcodes ─────────────────────────────────────────────

# G.711 in WAV (8 kHz): decoded by symphonia today — pinned by e2e tests.
ffmpeg -y -v error -i "$SPEECH" -ar 8000 -ac 1 -c:a pcm_alaw "$E2E_DIR/speech_alaw.wav"
ffmpeg -y -v error -i "$SPEECH" -ar 8000 -ac 1 -c:a pcm_mulaw "$E2E_DIR/speech_mulaw.wav"

# G.722 in WAV (native 16 kHz, format tag 0x0064): decoded by our fallback.
ffmpeg -y -v error -i "$SPEECH" -ar 16000 -ac 1 -c:a g722 "$E2E_DIR/speech_g722.wav"

# Headerless raw streams (RTP-dump style) for the ?codec= path.
ffmpeg -y -v error -i "$SPEECH" -ar 8000 -ac 1 -c:a pcm_alaw -f alaw "$E2E_DIR/speech.alaw"
ffmpeg -y -v error -i "$SPEECH" -ar 8000 -ac 1 -c:a pcm_mulaw -f mulaw "$E2E_DIR/speech.ulaw"
ffmpeg -y -v error -i "$SPEECH" -ar 16000 -ac 1 -c:a g722 -f g722 "$E2E_DIR/speech.g722"

echo "Wrote fixtures:"
ls -l "$CORE_DIR" "$E2E_DIR"
