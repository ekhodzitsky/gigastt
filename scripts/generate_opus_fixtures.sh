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

# Multi-frame packets, the shape Chromium's MediaRecorder emits: 48 kHz stereo
# CELT at the browser's 60 ms default, i.e. three 20 ms frames per packet
# (code 3, CBR). Every packet of this file is code 3, where opus_tone.ogg is
# code 0 throughout. Plus ffmpeg's own decode of it as the reference.
ffmpeg -y -v error -i "$TONE" -ar 48000 -ac 2 -c:a libopus -frame_duration 60 \
  "$CORE_DIR/opus_tone_60ms.ogg"
ffmpeg -y -v error -i "$CORE_DIR/opus_tone_60ms.ogg" -f s16le -acodec pcm_s16le \
  -ar 16000 -ac 1 "$CORE_DIR/opus_tone_60ms_ffmpeg.pcm"

# WebM/Opus in the shape a browser's MediaRecorder writes: a *live* stream,
# where the Segment and every Cluster carry an unknown size because the length
# is not known while recording. ffmpeg's `-live 1` gives an unknown-size Segment
# but still writes known-size Clusters, so the Clusters are rewritten afterwards
# to the one-byte unknown-size vint (0xFF). No sample is touched — only the size
# field — and ffmpeg decodes the result to byte-identical PCM.
ffmpeg -y -v error -i "$TONE" -ar 48000 -ac 1 -c:a libopus -frame_duration 60 \
  -f webm -live 1 "$CORE_DIR/.tone_live_known_clusters.webm"

python3 - "$CORE_DIR/.tone_live_known_clusters.webm" \
         "$CORE_DIR/opus_tone_webm_live.webm" <<'PY'
import sys

CLUSTER = bytes.fromhex("1f43b675")
SEGMENT = bytes.fromhex("18538067")


def read_id(d, i):
    b, n = d[i], 1
    while n <= 4 and not (b & (0x80 >> (n - 1))):
        n += 1
    return d[i : i + n], i + n


def read_size(d, i):
    s, m = d[i], 1
    while m <= 8 and not (s & (0x80 >> (m - 1))):
        m += 1
    raw = d[i : i + m]
    val = raw[0] & (0xFF >> m)
    for k in raw[1:]:
        val = (val << 8) | k
    return val, val == (1 << (7 * m)) - 1, i + m


d = open(sys.argv[1], "rb").read()
out, i = bytearray(), 0

# EBML header, verbatim.
_, j = read_id(d, i)
size, _, j = read_size(d, j)
out += d[i : j + size]
i = j + size

# Segment header, verbatim (already unknown-size thanks to -live 1).
eid, j = read_id(d, i)
size, unknown, j = read_size(d, j)
assert eid == SEGMENT and unknown, "expected an unknown-size Segment from -live 1"
out += d[i:j]
i = j

# Children of the Segment: every Cluster gets the unknown-size vint.
clusters = 0
while i < len(d):
    start = i
    eid, j = read_id(d, i)
    size, unknown, j = read_size(d, j)
    end = len(d) if unknown else j + size
    if eid == CLUSTER and not unknown:
        out += eid + b"\xff" + d[j:end]
        clusters += 1
    else:
        out += d[start:end]
    i = end

open(sys.argv[2], "wb").write(bytes(out))
print(f"rewrote {clusters} clusters to unknown size")
PY

rm -f "$CORE_DIR/.tone_live_known_clusters.webm"

ffmpeg -y -v error -i "$CORE_DIR/opus_tone_webm_live.webm" -f s16le -acodec pcm_s16le \
  -ar 16000 -ac 1 "$CORE_DIR/opus_tone_webm_live_ffmpeg.pcm"

# Same Opus packets, two container rate tags. libopus always emits 48 kHz PCM
# (RFC 7845). OpusHead "input sample rate" and Matroska SamplingFrequency are
# the capture rate a browser records, not the decode rate. The 16 kHz copies
# are byte patches of the 48 kHz files (page CRC recomputed for Ogg) so a
# decode-length difference is the tag, not a different bitstream.
ffmpeg -y -v error -i "$TONE" -ar 48000 -ac 1 -c:a libopus -b:a 32k \
  "$CORE_DIR/opus_rate48.ogg"
ffmpeg -y -v error -i "$TONE" -ar 48000 -ac 1 -c:a libopus -b:a 32k -f webm \
  "$CORE_DIR/opus_rate48.webm"

python3 - "$CORE_DIR/opus_rate48.ogg" "$CORE_DIR/opus_head16.ogg" \
         "$CORE_DIR/opus_rate48.webm" "$CORE_DIR/opus_sf16.webm" <<'PY'
import struct
import sys


def ogg_crc(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte << 24
        for _ in range(8):
            if crc & 0x80000000:
                crc = ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF
            else:
                crc = (crc << 1) & 0xFFFFFFFF
    return crc


def patch_ogg_input_rate(data: bytes, rate: int) -> bytes:
    d = bytearray(data)
    i = d.find(b"OpusHead")
    if i < 0:
        raise SystemExit("OpusHead not found")
    d[i + 12 : i + 16] = rate.to_bytes(4, "little")
    page = d.rfind(b"OggS", 0, i)
    if page < 0:
        raise SystemExit("Ogg page not found")
    nseg = d[page + 26]
    body = sum(d[page + 27 : page + 27 + nseg])
    end = page + 27 + nseg + body
    d[page + 22 : page + 26] = b"\x00\x00\x00\x00"
    crc = ogg_crc(bytes(d[page:end]))
    d[page + 22 : page + 26] = crc.to_bytes(4, "little")
    return bytes(d)


def patch_webm_sampling_frequency(data: bytes, rate: float) -> bytes:
    d = bytearray(data)
    needle = b"\xb5\x88" + struct.pack(">d", 48000.0)
    n = d.count(needle)
    if n != 1:
        raise SystemExit(f"expected one 48 kHz SamplingFrequency, found {n}")
    i = d.find(needle)
    d[i + 2 : i + 10] = struct.pack(">d", rate)
    return bytes(d)


ogg48, ogg16, webm48, webm16 = sys.argv[1:]
open(ogg16, "wb").write(patch_ogg_input_rate(open(ogg48, "rb").read(), 16000))
open(webm16, "wb").write(
    patch_webm_sampling_frequency(open(webm48, "rb").read(), 16000.0)
)
print("patched OpusHead input rate and WebM SamplingFrequency to 16 kHz")
PY

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
