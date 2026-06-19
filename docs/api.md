# API

gigastt exposes WebSocket (streaming), REST, and SSE on a single port (default `9876`).
Machine-readable specs: [`docs/asyncapi.yaml`](asyncapi.yaml) (WebSocket) and
[`docs/openapi.yaml`](openapi.yaml) (REST).

## WebSocket — real-time streaming

Connect to `ws://127.0.0.1:9876/v1/ws`, send PCM16 audio frames, receive transcription
in real time.

```
Client                            Server
  |-------- connect --------------> |
  | <------- ready ----------------- |  {type:"ready", version:"1.0"}
  |------- configure (optional) --> |  {type:"configure", sample_rate:16000}
  |-------- binary PCM16 ---------> |
  | <------- partial --------------- |  {type:"partial", text:"привет"}
  | <------- final ----------------- |  {type:"final", text:"Привет, как дела?"}
```

**Supported sample rates:** 8, 16, 24, 44.1, 48 kHz (default 48 kHz, resampled to
16 kHz internally). Protocol messages are versioned via the `type` field; new fields
are additive only.

## REST

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check (`{"status":"ok"}`) |
| `/ready` | GET | Readiness probe (200 when the engine pool is ready) |
| `/v1/models` | GET | Model info (encoder type, pool size, capabilities) |
| `/v1/transcribe` | POST | File transcription, full JSON response or export format |
| `/v1/transcribe/stream` | POST | File transcription with SSE streaming |
| `/v1/ws` | GET | WebSocket upgrade for real-time streaming |
| `/metrics` | GET | Prometheus metrics (enabled with `--metrics`). Served on the separate `--metrics-listen` port (default `127.0.0.1:9090`), not the main API port. |

```sh
# Full JSON
curl -X POST http://127.0.0.1:9876/v1/transcribe \
  -H "Content-Type: application/octet-stream" --data-binary @recording.wav
# {"text":"Привет, как дела?","words":[{"word":"Привет,","start":0.5,"end":0.9,"confidence":0.97}, ...],"duration":3.5}

# SSE streaming
curl -X POST http://127.0.0.1:9876/v1/transcribe/stream \
  -H "Content-Type: application/octet-stream" --data-binary @recording.wav
# data: {"type":"partial","text":"привет как"}
# data: {"type":"final","text":"Привет, как дела?"}
```

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

### Error responses

| HTTP | Code | When |
|---|---|---|
| 400 | `empty_body` | Request body is empty |
| 400 | `invalid_format` | Unsupported `format` query value |
| 413 | `payload_too_large` | Body exceeds `--body-limit-bytes` (default 50 MiB) |
| 422 | `invalid_audio` | Audio could not be decoded (unsupported/corrupt format) |
| 422 | `transcription_error` | Audio decoded but inference failed |
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
