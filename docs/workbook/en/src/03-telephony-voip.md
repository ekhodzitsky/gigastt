# Telephony & VoIP: G.711, G.722, Opus, and PBX recordings

## Scenario

You run a call center or integrate a PBX (Asterisk, FreeSWITCH, Cisco, Teams),
and call recordings are your main source of audio. The folder in front of you
contains a mix of `wav49` files, G.711/G.722 WAVs, headerless `.ulaw` dumps
pulled from RTP captures, and the odd Telegram voice note — and you want
transcripts without manually converting every file.

gigastt decodes most telephony formats natively: G.711 A-law/μ-law in WAV,
G.722 in WAV (both registered format tags), OGG/Opus, and headerless raw
streams via an explicit codec hint. This chapter gets you from "a folder of
weird files" to working transcripts, including per-channel speaker split for
stereo recordings.

## Prerequisites

- gigastt installed and the model downloaded — see
  [Getting started](01-getting-started.md).
- A running server for the REST recipes (`gigastt serve`, default
  `http://127.0.0.1:9876`). The CLI recipes work offline, no server needed.
- `ffprobe`/`ffmpeg` — only to identify formats and for the two conversion
  fallbacks (wav49, G.729). Not needed for the supported formats.

## Recipe

### Step 0 — identify what you actually have

PBX exports rarely say what they are. Two commands tell you everything:

```sh
file recording.wav
ffprobe -v error -show_entries stream=codec_name,codec_tag_string,sample_rate,channels \
  -of default=noprint_wrappers=1 recording.wav
```

Match the output against this table:

| ffprobe / `file` says | What it is | What to do |
|---|---|---|
| `codec_name=pcm_alaw` or `pcm_mulaw`, 8000 Hz | G.711 in WAV | upload as-is |
| `codec_name=adpcm_g722`, tag `[0x0064]` or `[0x028f]` | G.722 in WAV | upload as-is |
| `codec_name=gsm_ms` | wav49 (GSM 06.10 in WAV) | convert first — see Asterisk below |
| `codec_name=opus` in an Ogg container | Opus (Telegram, MediaRecorder) | upload as-is |
| ffprobe fails with `Invalid data found when processing input`, `file` says `data` | headerless raw stream | declare the codec — see RTP dump below |

The two G.722 tags are the same codec from different writers: `0x0064` comes
from SBC/Asterisk-style exports, `0x028F` from ffmpeg-based tooling. gigastt
accepts both.

### Asterisk Monitor recordings (wav49, G.722 WAV, raw streams)

What `Monitor()`/`MixMonitor` writes depends on the configured format:

- **`wav`** — plain PCM16 WAV. Upload as-is.
- **`wav49`** — GSM 06.10 in a WAV container (`codec_name=gsm_ms`). gigastt
  does not decode GSM, so convert once to PCM16 WAV:

  ```sh
  ffmpeg -y -i call.wav -ar 16000 -ac 1 -c:a pcm_s16le call_16k.wav
  ```

  Verify: `ffprobe call_16k.wav` reports `codec_name=pcm_s16le`,
  `sample_rate=16000`. Then transcribe `call_16k.wav` like any WAV.
- **`ulaw` / `alaw` / `g722`** — raw headerless streams (no container to
  sniff). Declare the codec explicitly:

  ```sh
  # REST — codec aliases: pcmu=ulaw, pcma=alaw
  curl -X POST "http://127.0.0.1:9876/v1/transcribe?codec=pcmu&sample_rate=8000" \
    -H "Content-Type: application/octet-stream" --data-binary @call.ulaw

  # CLI
  gigastt transcribe call.ulaw --codec pcmu --sample-rate 8000
  ```

  Verify: HTTP 200 and a transcript that matches the call. If you pick the
  wrong companding law (A-law vs μ-law), the request still returns 200 but
  the text is garbage — swap the codec name and retry.

### Cisco and Teams exports (G.722 WAV)

Cisco phone systems and Teams-adjacent tooling typically hand you G.722 in a
WAV container — `codec_name=adpcm_g722`, tag `0x0064` or `0x028F`. The
container declares the codec, so no flags are needed:

```sh
curl -X POST http://127.0.0.1:9876/v1/transcribe \
  -H "Content-Type: application/octet-stream" --data-binary @call_g722.wav

gigastt transcribe call_g722.wav
```

Verify: HTTP 200 with JSON text, or the transcript on stdout for the CLI.

The typical failure here is `422` — the server could not decode the upload.
It means the file is not what the extension claims (a G.729 payload, a
vendor-proprietary wrapper) or the file is truncated. Go back to Step 0 and
check what ffprobe actually says.

### Telegram and WhatsApp voice notes (Opus)

Telegram voice messages (Bot API `voice.oga`), WhatsApp voice notes (`.ogg`),
and browser MediaRecorder captures (`.opus`) are all Opus in an Ogg container.
Upload them as-is — the server sniffs the container from the bytes, so the
file extension does not matter:

```sh
curl -X POST http://127.0.0.1:9876/v1/transcribe \
  -H "Content-Type: application/octet-stream" --data-binary @voice.ogg

gigastt transcribe voice.opus
```

Verify: HTTP 200 with the transcript. Opus decodes at its native 48 kHz and
is resampled internally; mono and stereo are supported (stereo is mixed to
mono unless you use channel split below), multistream (>2 channel) OGG/Opus
is rejected.

### A stereo call → two speakers (channels=split)

Many PBXs record one party per channel: left = one speaker, right = the
other. Channel split transcribes each channel as its own speaker instead of
mixing to mono: channel 0 (left) becomes `speaker_0`, channel 1 (right)
becomes `speaker_1`.

```sh
# REST
curl -X POST "http://127.0.0.1:9876/v1/transcribe?channels=split" \
  -H "Content-Type: application/octet-stream" --data-binary @call.wav

# CLI — speaker-aware SRT with [SPEAKER_0] / [SPEAKER_1] cue prefixes
gigastt transcribe call.wav --stereo-speakers -f srt -o call.srt
```

In the JSON every word carries a `speaker` field, ordered by start time:

```json
{
  "text": "…",
  "words": [
    {"word": "покажи", "start": 0.08, "end": 0.48, "confidence": 0.95, "speaker": 1},
    {"word": "шестьдесят", "start": 0.52, "end": 1.08, "confidence": 0.96, "speaker": 0}
  ],
  "duration": 3.43
}
```

For reports, group words by `speaker` to get per-party text and talk time
(agent vs customer). Which party sits on which channel is your PBX's
convention — calibrate once on a call with known speakers.

Verify: both labels appear —

```sh
curl -s -X POST "http://127.0.0.1:9876/v1/transcribe?channels=split" \
  -H "Content-Type: application/octet-stream" --data-binary @call.wav \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(sorted({w.get('speaker') for w in d['words']}))"
# [0, 1]
```

Fallbacks to know about: if the file is mono, has more than two channels, or
is dual-mono (both channels nearly identical — some PBXs mix the call into
both), gigastt falls back to a plain mono transcript with no `speaker`
fields and logs a `falling back to mono transcription` warning. Channel split
is mutually exclusive with diarization: `channels=split&diarization=true`
returns `400 conflicting_modes`.

### An RTP dump without a container

An RTP capture stripped to payload bytes has no header to sniff, so the codec
must be declared. Export payload-only from your tooling (in Wireshark:
Telephony → RTP → Stream Analysis → save payload; SBCs have similar payload
exports) — the 12-byte RTP headers must not be in the file.

```sh
# G.711 A-law capture at 8 kHz
curl -X POST "http://127.0.0.1:9876/v1/transcribe?codec=pcma&sample_rate=8000" \
  -H "Content-Type: application/octet-stream" --data-binary @dump.alaw

# G.722 capture — see the SDP clock-rate note below
curl -X POST "http://127.0.0.1:9876/v1/transcribe?codec=g722&sample_rate=8000" \
  -H "Content-Type: application/octet-stream" --data-binary @dump.g722

# CLI equivalent
gigastt transcribe dump.g722 --codec g722 --sample-rate 8000
```

The G.722 SDP quirk: for historical reasons SDP/RTP announces G.722 with an
8000 Hz clock rate, while the stream actually decodes to 16 kHz. gigastt
accepts both `8000` and `16000` for `g722` and always decodes to 16 kHz, so
either value works. For raw G.711 (`pcmu`/`pcma`) any rate in 8000–48000 Hz
is accepted and resampled.

Verify: HTTP 200 and sensible text. Parameter errors fail fast, before any
inference: `codec` without `sample_rate` → `400 invalid_sample_rate`
("sample_rate is required when codec is set"); an unknown codec →
`400 unsupported_codec` ("Unsupported codec. Supported: pcmu (ulaw),
pcma (alaw), g722").

Lossy captures: the stream is decoded as-is. Packet-loss gaps and reordered
audio are not repaired — de-jitter or re-capture if the transcript has holes.

### A folder of recordings (batch)

`gigastt transcribe-batch` scans files with `wav`, `mp3`, `m4a`, `ogg`, and
`flac` extensions — which covers G.711/G.722 WAV and OGG/Opus (`.ogg`) out of
the box:

```sh
gigastt transcribe-batch recordings/ out/ --format txt,json
```

Two kinds of telephony files are **not** scanned:

- **`.opus` files** — rename them to `.ogg`. The content is an Ogg container
  and is sniffed from bytes, so the rename is safe:

  ```sh
  for f in recordings/*.opus; do mv "$f" "${f%.opus}.ogg"; done
  ```

- **raw `.ulaw` / `.alaw` / `.g722` streams** — wrap them into WAV first
  (batch has no `--codec` flag), then run batch on the wrapped files:

  ```sh
  mkdir -p wav
  for f in recordings/*.ulaw; do
    ffmpeg -y -v error -f mulaw -ar 8000 -ac 1 -i "$f" \
      -ar 16000 -ac 1 -c:a pcm_s16le "wav/$(basename "${f%.ulaw}").wav"
  done
  gigastt transcribe-batch wav/ out/ --format txt,json
  ```

  For A-law use `-f alaw`, for G.722 `-f g722` (G.722 needs no input `-ar`).

Verify: `out/` contains one `.txt`/`.json` per source file and the command
exits 0. For an inbox that keeps receiving new recordings, `gigastt watch`
does the same continuously — see [CLI and batch processing](02-cli-batch.md)
for batch/watch details and long recordings.

## Format cheat sheet

| Input | Sniffed from container | Needs `?codec=` / `--codec` | Notes and limits |
|---|---|---|---|
| WAV PCM (8–32 bit, IEEE float) | yes | no | stereo auto-mixed to mono |
| WAV with G.711 A-law / μ-law | yes | no | typically 8 kHz, resampled to 16 kHz |
| WAV with G.722 ADPCM (tags `0x0064`, `0x028F`) | yes | no | decodes to native 16 kHz |
| OGG/Opus, `.opus` | yes | no | mono/stereo only; >2ch rejected |
| raw `.ulaw` / `.alaw` | no (headerless) | yes — `pcmu` / `pcma` | `sample_rate` 8000–48000 |
| raw `.g722` | no (headerless) | yes — `g722` | `sample_rate` 8000 (SDP convention) or 16000; decodes to 16 kHz |
| wav49 (GSM 06.10 in WAV) | yes | n/a | not decoded — convert to PCM16 WAV first |
| G.729 (any wrapper) | yes | n/a | not supported — convert to PCM16 WAV first |

Applies everywhere: uploads are capped at 30 minutes of decoded audio, and
`?codec=` / `?sample_rate=` work on `/v1/transcribe`, `/v1/transcribe/stream`,
and `/v1/jobs` alike. A Deepgram-compatible `/v1/listen` endpoint that accepts
the same telephony inputs is in progress.

## Verifying the result

The repository ships telephony fixtures used by its own end-to-end tests —
4 seconds of Russian speech («шестьдесят тысяч тенге сколько будет стоить»)
transcoded into every supported telephony format. You need the downloaded
model and a running server:

```sh
# raw path
curl -s -X POST "http://127.0.0.1:9876/v1/transcribe?codec=pcmu&sample_rate=8000" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @crates/gigastt/tests/fixtures/telephony/speech.ulaw

# container path — no parameters
curl -s -X POST http://127.0.0.1:9876/v1/transcribe \
  -H "Content-Type: application/octet-stream" \
  --data-binary @crates/gigastt/tests/fixtures/telephony/speech_g722.wav
```

Both return HTTP 200 with a transcript mentioning «тенге» and «стоить» —
with the server's default punctuation and ITN enabled the text comes out as
«60000 тенге, сколько будет стоить?». If your own file fails while these
fixtures pass, the problem is the file, not the server — go back to Step 0.

## Common pitfalls

- **`422` — "Check audio format"** (`invalid_audio` / `transcription_error`).
  The bytes were probed as a container and decoding failed. Usual causes: a
  headerless raw stream posted without `codec=`, a wav49 (GSM) or G.729 file,
  or a truncated/corrupt upload. Identify the file with Step 0.
- **G.729 is not supported.** A raw upload with `?codec=g729` returns
  `400 unsupported_codec`; G.729-in-WAV fails with `422`. Convert with
  ffmpeg and upload the result:

  ```sh
  ffmpeg -y -i call_g729.wav -ar 16000 -ac 1 -c:a pcm_s16le call_16k.wav
  ```

- **`?codec=` on a container file.** The parameter overrides container
  sniffing entirely: a WAV posted with `?codec=pcmu` decodes the WAV header
  as μ-law noise and returns 200 with garbage text. Use `codec=` only for
  headerless streams.
- **RTP dump with headers or jitter.** `?codec=` expects payload bytes only.
  Leftover 12-byte RTP headers decode as periodic clicks, and loss/reorder
  gaps become holes in the transcript — nothing is repaired server-side.
  Export payload-only and de-jitter before uploading.
- **"8-bit WAV" is not broken.** G.711 in WAV legitimately reports
  `bits_per_sample=8` (companded samples) — upload it as-is. The real trap
  is the reverse: when converting files yourself, always write 16-bit PCM
  (`pcm_s16le`); 8-bit linear PCM (`pcm_u8`) destroys recognition accuracy.
- **Mono/stereo confusion.** `channels=split` on a mono file, a >2-channel
  file, or dual-mono stereo (some PBXs record the mixed call into both
  channels) silently falls back to a plain mono transcript — no `speaker`
  fields, just a `falling back to mono transcription` warning in the log.
  If your PBX records mixed mono, no flag can split the speakers after the
  fact; record stereo or use `diarization=true` instead (mutually exclusive
  with `channels=split`).
- **"Audio file too long"**. A single upload is capped at 30 minutes of
  decoded audio ("Maximum supported: 1800s"). Split longer recordings — for
  example per call leg — before uploading.
- **A-law/μ-law swapped.** Both laws decode "successfully", so the wrong
  choice returns 200 with garbage instead of an error. If a raw stream
  transcribes to noise, retry with the other codec name.

## Links

- [CLI and batch processing](02-cli-batch.md) — batch and watch mode,
  export formats, long recordings.
- [Getting started](01-getting-started.md) — installation and model download.
- [Introduction](README.md) — documentation map.
- [docs/api.md — Audio formats and telephony codecs](https://github.com/ekhodzitsky/gigastt/blob/main/docs/api.md#audio-formats-and-telephony-codecs) —
  the canonical format table and all query parameters.
- [docs/cli.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/cli.md) —
  `transcribe`, `transcribe-batch`, and `watch` flag reference.
- [crates/gigastt-core/src/inference/audio.rs](https://github.com/ekhodzitsky/gigastt/blob/main/crates/gigastt-core/src/inference/audio.rs) —
  decoder internals (G.722 sniffing, the Opus path, dual-mono detection).
- [Telephony and Opus test fixtures](https://github.com/ekhodzitsky/gigastt/tree/main/crates/gigastt/tests/fixtures)
  with their generators
  [generate_telephony_fixtures.sh](https://github.com/ekhodzitsky/gigastt/blob/main/scripts/generate_telephony_fixtures.sh)
  and
  [generate_opus_fixtures.sh](https://github.com/ekhodzitsky/gigastt/blob/main/scripts/generate_opus_fixtures.sh).
