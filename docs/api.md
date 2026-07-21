# API

> **Recipes:** the [GigaSTT Workbook](https://ekhodzitsky.github.io/gigastt/) holds scenario-driven guides (EN + RU); this document stays the canonical API reference.

gigastt exposes WebSocket (streaming), REST, and SSE on a single port (default `9876`).
Machine-readable specs: [`docs/asyncapi.yaml`](asyncapi.yaml) (WebSocket) and
[`docs/openapi.yaml`](openapi.yaml) (REST).

## WebSocket — real-time streaming

Connect to `ws://127.0.0.1:9876/v1/ws`, send PCM16 audio frames, receive transcription
in real time. This section is the human-readable protocol reference; the
machine-readable schema (same fields, same error codes) lives in
[`docs/asyncapi.yaml`](asyncapi.yaml). Field-level source of truth:
`crates/gigastt-core/src/protocol/mod.rs`.

```
Client                            Server
  |-------- connect --------------> |
  | <------- ready ----------------- |  {type:"ready", model, version, supported_rates, ...}
  |------- configure (optional) --> |  {type:"configure", sample_rate:16000}
  |-------- binary PCM16 ---------> |
  | <------- partial --------------- |  {type:"partial", text, words[], ...}
  |-------- binary PCM16 ---------> |
  | <------- final ----------------- |  {type:"final", text, words[], ...}
  |--------- stop ----------------> |  {type:"stop"}
  | <------- final ----------------- |  trailing words flushed, then the client closes
```

**Versioning.** Protocol messages are discriminated by the `type` field; the
current version is `1.0`, reported in `ready.version`. New fields are additive
only — never removed or renamed — so clients must ignore fields they do not
know. A client may announce its version via `configure.protocol_version`; an
unsupported value is rejected with `unsupported_protocol_version`.

### Server → Client messages

#### `ready`

Sent immediately after the WebSocket handshake, before any audio is accepted.

```json
{
  "type": "ready",
  "model": "gigaam-v3-rnnt",
  "sample_rate": 48000,
  "version": "1.0",
  "supported_rates": [8000, 16000, 24000, 44100, 48000],
  "diarization": false,
  "max_session_secs": 3600,
  "idle_timeout_secs": 300
}
```

| Field | Type | Meaning |
|---|---|---|
| `model` | string | Loaded head: `gigaam-v3-rnnt`, `gigaam-v3-e2e-rnnt`, `gigaam-multilingual-ctc`, or `gigaam-multilingual-large-ctc` |
| `sample_rate` | integer | Default input rate in Hz (48000 for backward compatibility) — used when no `configure` overrides it |
| `version` | string | WebSocket protocol version (semver-lite `major.minor`) |
| `supported_rates` | integer[] | Accepted input sample rates in Hz; omitted if empty. Use it instead of hardcoding a rate list |
| `diarization` | boolean | `true` when the speaker-encoder model is loaded and diarization can be enabled per session; omitted when `false` |
| `min_protocol_version` | string | Oldest protocol version the server accepts; omitted when equal to `version` (only one version supported) |
| `max_session_secs` | integer | Wall-clock session cap (`--max-session-secs`); `0` means disabled. Always sent — plan long recordings around it instead of discovering it via close 1008 |
| `idle_timeout_secs` | integer | Idle timeout (`--idle-timeout-secs`): close 1001 after this many seconds without frames. Always sent |

#### `partial` / `final`

Both carry the same `TranscriptSegment` payload; `final` means the utterance is
complete (endpointing detected, or the stream was flushed by `stop`/shutdown).

```json
{
  "type": "final",
  "text": "Привет, как дела?",
  "timestamp": 1712700001.456,
  "is_final": true,
  "confidence": 0.95,
  "words": [
    {"word": "привет", "start": 0.0, "end": 0.4, "confidence": 0.97},
    {"word": "как", "start": 0.5, "end": 0.7, "confidence": 0.93},
    {"word": "дела", "start": 0.8, "end": 1.1, "confidence": 0.95}
  ]
}
```

| Field | Type | Meaning |
|---|---|---|
| `text` | string | Joined transcript text (see post-processing below) |
| `timestamp` | number | Unix time (seconds) when the segment was produced |
| `is_final` | boolean | Mirrors the `type` discriminator (`true` in `final`) |
| `confidence` | number | Segment-level confidence: the duration-weighted mean of `words[].confidence` (plain mean when all word durations are zero). Omitted when the segment has no words. It is an average of per-word softmax scores — not a calibrated probability |
| `words[]` | object[] | Per-word detail; may be empty on flushed/empty finals |

Each `words[]` entry:

| Field | Type | Meaning |
|---|---|---|
| `word` | string | Recognized word (BPE tokens joined, raw decoder casing) |
| `start` / `end` | number | Word boundaries in seconds from the start of the stream |
| `confidence` | number | Mean softmax confidence over the word's BPE tokens, 0.0–1.0. Real decoder output — use it instead of a hardcoded constant |
| `speaker` | integer | Zero-based speaker label; present only when diarization is active |

**Transcript post-processing.** `partial` messages are always the raw decoder
hypothesis (lowercase, no punctuation) and may change with more audio. `final`
messages are enriched at the finalization boundary when the server has the
resources loaded: inverse text normalization (number-words → digits), then
punctuation/casing restoration (`--punctuation` / `--itn`; default `auto` = on
for the bare `rnnt` head, off for `e2e_rnnt` which is already punctuated). The
`words[]` payload always keeps the raw decoder output — only the joined `text`
is rewritten. The same applies to SSE `final` events on `/v1/transcribe/stream`
(server defaults; there are no per-request parameters there yet).

#### `error`

```json
{"type": "error", "message": "Server busy, try again later", "code": "timeout", "retry_after_ms": 30000}
```

| Field | Type | Meaning |
|---|---|---|
| `message` | string | Generic user-facing description (internal details are never leaked) |
| `code` | string | Machine-readable code — branch on this, not on `message` (table below) |
| `retry_after_ms` | integer | Suggested backoff in milliseconds; present only for transient backpressure (`timeout` on pool saturation). Honor it instead of guessing a retry delay |

### Client → Server messages

#### `configure`

Optional; must be sent **before the first audio frame** (otherwise the server
replies with `configure_too_late` and keeps the previous settings).

```json
{"type": "configure", "sample_rate": 16000, "punctuation": false, "itn": false}
```

| Field | Type | Meaning |
|---|---|---|
| `sample_rate` | integer | Input rate in Hz; must be one of `ready.supported_rates`, else `invalid_sample_rate` (the session continues at the previous rate) |
| `diarization` | boolean | Enable speaker diarization for this session (requires `ready.diarization: true`) |
| `protocol_version` | string | Version the client wants to speak; unsupported values are rejected with `unsupported_protocol_version` and the server ends the session |
| `punctuation` | boolean | Per-session punctuation/casing override for `final` segments only. `true` on a server without a punctuation model is a graceful no-op — finals stay raw, no error |
| `itn` | boolean | Per-session inverse text normalization override (`final` segments only) |

All fields are optional. Omitting a field keeps the server default; repeated
`configure` messages compose (an absent field leaves the previous value).

#### `stop`

```json
{"type": "stop"}
```

Asks the server to **finalize the session**: it decodes any audio still buffered
since the last partial (trailing words are not lost), emits one last `final`
message — possibly with empty `text` if nothing was pending — and only then is
the session over. The correct end-of-stream pattern is: send `stop`, wait for
the `final`, then close the socket. Do **not** close immediately after the last
audio frame or insert a fixed drain delay — the `final` after `stop` is the
explicit, lossless end marker.

#### Binary audio frames

Raw PCM16 signed little-endian, mono, at the negotiated rate (default 48 kHz;
resampled to 16 kHz internally). Frame size up to `--ws-frame-max-bytes`
(default 512 KiB ≈ 16 s at 16 kHz). Odd-length frames are fine — the trailing
byte is carried into the next frame. Empty frames are tolerated up to a
per-session cap (1000) and then closed with `policy_violation` + close code
1008.

### Keepalive

The server sends a WebSocket **ping every 30 s** and closes the connection after
**2 consecutive pings** with no inbound frame in between (≈ 90 s detection of a
half-open peer). Any inbound frame — pong, binary audio, or text — counts as
liveness and resets the counter. Standard WS clients answer pings automatically;
you do not need to send your own pings.

### Error codes

Codes emitted on the WebSocket (`error` message, sometimes followed by a close
frame). The same enum is declared in [`docs/asyncapi.yaml`](asyncapi.yaml).

| Code | Session after | Meaning |
|---|---|---|
| `timeout` | ends (never opened) | Pool saturation: no inference slot freed within the checkout window (default 30 s). `retry_after_ms` is set — wait and reconnect |
| `pool_closed` | ends | Server is shutting down, pool closed to new sessions |
| `idle_timeout` | ends (close 1001) | No frames for `--idle-timeout-secs` (default 300 s) |
| `max_session_duration_exceeded` | ends (close 1008) | Wall-clock session cap `--max-session-secs` (default 3600 s) hit; a `final` is flushed before the close |
| `policy_violation` | ends (close 1008) | Empty-frame spam (over 1000 empty binary frames) |
| `inference_timeout` | ends | One inference run exceeded `--inference-timeout-secs` (default 600 s) |
| `inference_error` | continues | Inference failed on the last chunk (bad audio format, etc.); the session state is intact |
| `inference_panic` | continues, state reset | Inference panicked; the decoder state was reset, so earlier audio context is lost — already-received `final`s remain valid |
| `configure_too_late` | continues | `configure` arrived after the first audio frame; previous settings kept |
| `invalid_sample_rate` | continues | Rate not in `supported_rates`; previous rate kept |
| `unsupported_protocol_version` | ends | Client requested a protocol version the server does not speak |
| `payload_too_large` | — | REST/SSE surface (body over `--body-limit-bytes`). On WS, an oversized frame is rejected by the transport with close code 1009 instead |

### Close codes

| Code | When |
|---|---|
| 1001 Going Away | Graceful shutdown drain (SIGTERM), `idle_timeout`, or keepalive ping timeout. A `final` is flushed first on shutdown |
| 1008 Policy Violation | `max_session_duration_exceeded`, `policy_violation` |
| 1009 Message Too Big | A frame exceeded `--ws-frame-max-bytes` |
| 1006 Abnormal Closure | Never sent by the server — the client observes it when an established socket drops with no close frame (process killed, crash, middlebox). Do not confuse it with an *upgrade refusal*: while the model is still loading, `/v1/ws` answers HTTP 503 `{"code":"initializing"}` before any socket exists — poll `/ready` and retry later instead of killing the process. If the port listens but even `/health` fails, suspect an orphaned process holding the port — see [troubleshooting](troubleshooting.md) |

### Session lifecycle: `/health` vs `/ready`

Use the HTTP probes to drive client-side state machines — do not spawn
`gigastt --version` or guess from TCP connectability:

- `GET /health` — **liveness**. Returns 200 as soon as the listener is up,
  including during first-run model download/quantization, when it is served by a
  minimal bootstrap responder:
  `{"status":"ok","model":"loading","version":"2.13.0"}`. Once the engine is up:
  `{"status":"ok","model":"gigaam-v3-rnnt","variant":"rnnt","version":"2.13.0","punctuation":true,"itn":true}`.
  The `version` field is present in **both** phases — use it for version gates
  instead of executing the binary.
- `GET /ready` — **readiness**. 200 `{"status":"ready","pool_available":N,"pool_total":M}`
  when at least one inference slot is free; 503 with
  `{"status":"not_ready","reason":"initializing"}` while the model loads,
  `"pool_exhausted"` when all slots are busy, or `"shutting_down"` during drain.
  Gate first audio on `/ready`, gate process liveness on `/health`.

### Session and frame limits

Defaults; every limit is a CLI flag with a matching `GIGASTT_*` env var
(see `gigastt serve --help`):

| Limit | Default | Behavior at the limit |
|---|---|---|
| `--pool-size` | 2 | Concurrent inference sessions; the next connect waits up to 30 s, then `timeout` + `retry_after_ms` (WS) or 503 + `Retry-After` (REST) |
| `--idle-timeout-secs` | 300 | No frames for 5 min → `idle_timeout` + close 1001. Streaming silence (quiet PCM) keeps the session alive — silence is still audio |
| `--max-session-secs` | 3600 | Wall-clock cap → `max_session_duration_exceeded` + flushed `final` + close 1008. `0` disables |
| `--ws-frame-max-bytes` | 512 KiB | Larger frame → close 1009 |
| `--inference-timeout-secs` | 600 | One hung inference run → `inference_timeout`, session ends |

**Long sessions (interviews over an hour):** the 1-hour cap is the default, not
a hard limit — raise `--max-session-secs` (or set `0`) and keep frames flowing
so the idle timeout never trips. Reconnecting on the cap is also safe: the
server flushes a `final` before closing, so nothing recognized is lost. See
[troubleshooting](troubleshooting.md) for the failure scenarios these limits
produce. Embedding the binary as a managed sidecar (spawn, readiness probing,
version gating) is covered by the embedding guide (`docs/embedding.md`, in
progress); the onnxruntime linking trade-offs are in
[embedding-packaging](embedding-packaging.md).

## REST

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check. Reports the loaded head + effective punctuation/ITN policy (`{"status":"ok","model":"gigaam-v3-rnnt","variant":"rnnt","punctuation":true,"itn":true,...}`). During first-run model download it stays up with `model:"loading"`. |
| `/ready` | GET | Readiness probe (200 when the engine pool is ready; 503 `initializing` while the model loads, `pool_exhausted` when saturated) |
| `/v1/models` | GET | Model info (encoder type, pool size, capabilities) |
| `/v1/transcribe` | POST | File transcription, full JSON response or export format |
| `/v1/transcribe/stream` | POST | File transcription with SSE streaming |
| `/v1/jobs` | POST | Submit an asynchronous transcription job (requires `--enable-jobs`) |
| `/v1/jobs/{id}` | GET | Poll job status and progress |
| `/v1/jobs/{id}` | DELETE | Cancel a queued or processing job |
| `/v1/jobs/{id}/result` | GET | Fetch the finished transcription |
| `/v1/jobs/{id}/events` | GET | SSE stream of progress / done / failed / cancelled |
| `/v1/ws` | GET | WebSocket upgrade for real-time streaming |
| `/metrics` | GET | Prometheus metrics (enabled with `--metrics`). Served on the separate `--metrics-listen` port (default `127.0.0.1:9090`), not the main API port. |

```sh
# Full JSON
curl -X POST http://127.0.0.1:9876/v1/transcribe \
  -H "Content-Type: application/octet-stream" --data-binary @recording.wav
# {"text":"Привет, как дела?","words":[{"word":"Привет,","start":0.5,"end":0.9,"confidence":0.97}, ...],"confidence":0.94,"duration":3.5}

# SSE streaming
curl -X POST http://127.0.0.1:9876/v1/transcribe/stream \
  -H "Content-Type: application/octet-stream" --data-binary @recording.wav
# data: {"type":"partial","text":"привет как"}
# data: {"type":"final","text":"Привет, как дела?","confidence":0.94}
```

The full-JSON response, SSE `final` events, and job results also carry an
optional top-level `confidence` — the duration-weighted mean of
`words[].confidence` (an average of per-word softmax scores, not a calibrated
probability; omitted when there are no words).

## Asynchronous jobs

For long files or batch ingestion, `POST /v1/jobs` enqueues a transcription job
and returns immediately with a `job_id`:

```sh
curl -X POST http://127.0.0.1:9876/v1/jobs \
  -H "Content-Type: application/octet-stream" --data-binary @recording.wav
# {"job_id":"...","status":"queued","created_at":1712700000.5}
```

The endpoint is disabled by default. Enable it with `--enable-jobs` or
`GIGASTT_ENABLE_JOBS=1`. When disabled, all `/v1/jobs` paths return `404`.

Poll for status:

```sh
curl http://127.0.0.1:9876/v1/jobs/{job_id}
# {"job_id":"...","status":"processing","processed_seconds":12.5,"percent":42}
```

Fetch the result once `status` is `done`:

```sh
curl http://127.0.0.1:9876/v1/jobs/{job_id}/result
# {"text":"Привет, как дела?", ...}
```

Cancel a queued or processing job:

```sh
curl -X DELETE http://127.0.0.1:9876/v1/jobs/{job_id}
```

Subscribe to SSE events:

```sh
curl http://127.0.0.1:9876/v1/jobs/{job_id}/events
# data: {"type":"progress","percent":42,"processed_seconds":12.5}
# data: {"type":"done"}
```

Job submission accepts the same query parameters as `POST /v1/transcribe`
(`format`, `word_timestamps`, `segments`, `channels`, `diarization`,
`punctuation`, `itn`, `vad`). The result is rendered in the requested format
when `/v1/jobs/{id}/result` is called.

Queue behavior is controlled by:

- `--batch-pool-size` (default 0, clamped to at least 1 when jobs are enabled) —
  triplet slots reserved for batch/job work so long files do not starve
  real-time WebSocket or synchronous REST sessions.
- `--jobs-ttl-secs` (default 3600) — how long finished/failed/cancelled jobs are
  kept before eviction.
- `--jobs-max` (default 100) — maximum number of jobs kept in memory; when full,
  `POST /v1/jobs` returns `429 queue_full` with `Retry-After`.
- `--jobs-retry` (default 3) — maximum retry attempts for jobs that hit an
  inference timeout or worker panic.

The default `rnnt` head emits bare lowercase; punctuation, casing, and Russian ITN are
applied per server configuration (`--punctuation` / `--itn`). The `e2e_rnnt` head bakes
them in.

### Query parameters

`/v1/transcribe` accepts the following query parameters:

- `channels` (optional, string) — use `split` to transcribe the left and right channels
  as separate speakers (`speaker_0`, `speaker_1`). Defaults to mono mix.
- `diarization` (optional, boolean) — set to `true` to request polyvoice speaker
  diarization. The mutual-exclusion check with `channels=split` treats `diarization=true`
  as an explicit request; returns `400` with code `conflicting_modes` if both are set.
  Actual diarization output requires the speaker-encoder model to be loaded on the
  server (downloaded automatically when the `diarization` feature is enabled).

When either channel split or diarization produces speaker labels, each word object
includes a `speaker` integer:

```json
{
  "text": "привет да как дела",
  "words": [
    {"word": "привет", "start": 0.0, "end": 0.4, "confidence": 0.95, "speaker": 0},
    {"word": "да", "start": 0.5, "end": 0.8, "confidence": 0.91, "speaker": 1},
    {"word": "как", "start": 1.0, "end": 1.3, "confidence": 0.93, "speaker": 0},
    {"word": "дела", "start": 1.4, "end": 1.8, "confidence": 0.94, "speaker": 1}
  ],
  "duration": 2.0
}
```

- `segments` (optional, boolean) — set to `true` to add a `segments` array to the
  default JSON response. Segments are grouped from word timestamps by pause
  (≈0.9 s), sentence-ending punctuation (`.`, `!`, `?`), speaker change, or a
  maximum segment length of ≈30 s. Each segment carries `start`, `end`, `text`,
  `words`, and an optional `speaker` when channel split or diarization is active.
  With `format=md`, `segments=true` switches Markdown output to `### [mm:ss]`
  section headers instead of a single flat transcript blob.
- `codec` (optional, string) — declare a raw headerless telephony stream instead
  of a container: `pcmu` (alias `ulaw`), `pcma` (alias `alaw`), or `g722`. See
  "Audio formats and telephony codecs" below.
- `sample_rate` (optional, integer, Hz) — mandatory when `codec` is set
  (typical telephony: `8000`). G.722 always decodes to its native 16 kHz; both
  `8000` (the SDP clock-rate convention) and `16000` are accepted for it.

```sh
# JSON with word timestamps and grouped segments
curl -X POST "http://127.0.0.1:9876/v1/transcribe?segments=true&word_timestamps=true" \
  -H "Content-Type: application/octet-stream" --data-binary @recording.wav
# {
#   "text": "Привет. Как дела?",
#   "words": [...],
#   "segments": [
#     {"start": 0.0, "end": 0.9, "text": "Привет.", "words": [...]},
#     {"start": 1.6, "end": 2.8, "text": "Как дела?", "words": [...]}
#   ],
#   "duration": 3.5
# }

# Markdown with segment headers
curl -X POST "http://127.0.0.1:9876/v1/transcribe?format=md&segments=true" \
  -H "Content-Type: application/octet-stream" --data-binary @recording.wav
# ---
# duration: 3.5
# language: ru
# speakers: 0
# ---
#
# ### [00:00]
#
# Привет.
#
# ### [00:01]
#
# Как дела?
```

### Audio formats and telephony codecs

The container is sniffed from the upload bytes — `Content-Type` is ignored.
This applies to every file-transcription endpoint (`/v1/transcribe`,
`/v1/transcribe/stream`, `/v1/jobs`):

| Input | Decoder |
|---|---|
| WAV (PCM 8–32 bit, IEEE float) | symphonia |
| WAV with G.711 A-law / μ-law (8 kHz typical) | symphonia |
| WAV with G.722 ADPCM (tags `0x0064`, `0x028F`) | built-in fallback (`audio-codec`) |
| MP3, M4A/AAC, OGG/Vorbis, FLAC | symphonia |
| OGG/Opus, `.opus` (Telegram voice, MediaRecorder) | built-in fallback (`opus-rs`, pure Rust) |
| Raw headerless `.ulaw` / `.alaw` / `.g722` | `audio-codec` — requires `?codec=` |

**Opus notes**

- **OGG/Opus and `.opus`** (Telegram voice notes, browser MediaRecorder
  captures) decode at their native 48 kHz and are resampled to the model's
  16 kHz; stereo is mixed to mono. Multistream (>2 channel) OGG/Opus is not
  supported.

**Telephony notes**

- **G.711 A-law / μ-law in WAV** decode natively; a 8 kHz file is resampled to
  the model's 16 kHz.
- **G.722 in WAV** (what Asterisk, Cisco, and Teams-player exports write —
  format tag `0x0064`, ffmpeg writes `0x028F`) decodes to its native 16 kHz.
- **Headerless streams** (RTP dumps, Asterisk Monitor raw) carry no container
  to sniff, so the codec must be declared explicitly on `/v1/transcribe`:

```sh
curl -X POST "http://127.0.0.1:9876/v1/transcribe?codec=pcmu&sample_rate=8000" \
  -H "Content-Type: application/octet-stream" --data-binary @call.ulaw

curl -X POST "http://127.0.0.1:9876/v1/transcribe?codec=g722&sample_rate=8000" \
  -H "Content-Type: application/octet-stream" --data-binary @call.g722
```

A raw stream posted **without** `codec` is rejected with `422` (code
`invalid_audio` or `transcription_error`, depending on which stage rejects
it), exactly as before. The CLI mirrors this with
`gigastt transcribe call.ulaw --codec pcmu --sample-rate 8000`.

### Export formats

`/v1/transcribe` supports alternative output formats via the `format` query
parameter:

| Format | Query value | Content-Type | Notes |
|---|---|---|---|
| JSON (default) | `json` | `application/json` | Same response as before |
| Plain text | `txt` | `text/plain` | Transcript text only |
| SubRip | `srt` | `application/x-subrip` | Speaker-aware cues |
| WebVTT | `vtt` | `text/vtt` | Speaker-aware cues |
| Markdown | `md` | `text/markdown` | YAML frontmatter + transcript |

```sh
# SubRip subtitles
curl -X POST "http://127.0.0.1:9876/v1/transcribe?format=srt" \
  -H "Content-Type: application/octet-stream" --data-binary @recording.wav

# Markdown with word timings
curl -X POST "http://127.0.0.1:9876/v1/transcribe?format=md&word_timestamps=true" \
  -H "Content-Type: application/octet-stream" --data-binary @recording.wav

# Force download as attachment
curl -X POST "http://127.0.0.1:9876/v1/transcribe?format=vtt&download=recording.vtt" \
  -H "Content-Type: application/octet-stream" --data-binary @recording.wav
```

Optional formatter controls:

- `max_chars_per_line` — subtitle line length limit (default 80, 0 = unlimited)
- `max_words_per_line` — subtitle word count limit (default 14, 0 = unlimited)
- `word_timestamps` — include per-word table in Markdown (default false)

### Long recordings — send the whole file

gigastt chunks long audio **internally** (overlapping windows, de-duplicated and
stitched with monotonic timestamps), so you do **not** need to pre-segment with
ffmpeg. POST the whole recording and let the server return real, media-relative
timestamps — every `word` in the JSON response carries `start`/`end` in seconds,
and the `srt`/`vtt`/`md` exports key off those, so there's no need to fabricate
per-chunk offsets client-side.

```sh
# One long recording → Markdown transcript with real per-word timings
curl -X POST "http://127.0.0.1:9876/v1/transcribe?format=md&word_timestamps=true" \
  --data-binary @meeting.m4a -o meeting.md
```

Content-Type is ignored — the container format (WAV/MP3/M4A/OGG/FLAC) is sniffed
from the bytes, so `--data-binary @file` is enough; multipart form uploads are
not accepted. The practical ceiling is the body limit (`--body-limit-bytes`,
default 50 MiB ≈ 26 min of 16 kHz mono WAV) and the per-request inference cap
(`--inference-timeout-secs`, default 600 s); raise **both** together for longer
single files. A batch worker should gate on `GET /ready` (not just `/health`) so
it backs off on `503` pool saturation instead of failing mid-job.

### Error responses

| HTTP | Code | When |
|---|---|---|
| 400 | `empty_body` | Request body is empty |
| 400 | `invalid_format` | Unsupported `format` query value |
| 400 | `unsupported_codec` | Unknown `codec` query value (supported: `pcmu`, `pcma`, `g722`) |
| 400 | `invalid_sample_rate` | `sample_rate` missing with `codec`, or outside the accepted range |
| 400 | `conflicting_modes` | Both `channels=split` and `diarization=true` were requested |
| 404 | — (no JSON body) | `/v1/jobs/*` routes are not registered when `--enable-jobs` is off |
| 404 | `job_not_found` | Unknown or expired job id |
| 409 | `job_not_finished` | `GET /v1/jobs/{id}/result` called before the job is done |
| 409 | `job_not_cancellable` | `DELETE /v1/jobs/{id}` called on a terminal job |
| 413 | `payload_too_large` | Body exceeds `--body-limit-bytes` (default 50 MiB) |
| 422 | `invalid_audio` | Audio could not be decoded (unsupported/corrupt format) |
| 422 | `transcription_error` | Audio decoded but inference failed |
| 429 | `queue_full` | In-memory job store is full; `Retry-After` header included |
| 429 | `rate_limited` | Per-IP token bucket exhausted; `Retry-After` header included |
| 503 | `timeout` | All inference sessions busy; `Retry-After` + `retry_after_ms` |
| 503 | `pool_closed` | Server is shutting down, pool closed to new checkouts |

```
HTTP/1.1 503 Service Unavailable
Retry-After: 30

{"error":"Server busy, try again later","code":"timeout","retry_after_ms":30000}
```

## Client libraries

Ready-to-use WebSocket clients live in [`examples/`](../examples/):

```sh
pip install websockets && python examples/python_client.py recording.wav   # Python
bun examples/bun_client.ts recording.wav                                    # Bun / TypeScript
go run examples/go_client.go recording.wav                                  # Go (gorilla/websocket)
kotlinc examples/KotlinClient.kt -include-runtime -d client.jar && java -jar client.jar recording.wav  # Kotlin
```
