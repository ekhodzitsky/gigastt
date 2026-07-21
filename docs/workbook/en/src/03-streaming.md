# Streaming over WebSocket

Live, partial-results transcription: a microphone feed, a call leg, or a
browser capture goes in as raw PCM16, and text comes out within about a
second of speech. This chapter is the recipe book for that integration —
the field-by-field protocol reference stays in
[docs/api.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/api.md#websocket--real-time-streaming)
and the machine-readable schema in
[docs/asyncapi.yaml](https://github.com/ekhodzitsky/gigastt/blob/main/docs/asyncapi.yaml);
we link to them instead of repeating them.

## Scenario

You are building a real-time integration — live captions for a meeting
tool, a voice bot on a phone line, or a "dictaphone with instant text"
feature. Audio arrives continuously; users expect to watch the transcript
grow while they speak, see each utterance finalized cleanly, and never lose
trailing words when the stream ends. Some sessions run for hours, some
setups capture two audio sources at once (mic + system audio), and the
client must survive pool saturation and network drops without babysitting.

## Prerequisites

- A running server with the model, per [Getting started](01-getting-started.md):
  `gigastt serve` (first run downloads ~850 MB and quantizes — wait for
  readiness, don't kill it):

  ```sh
  until curl -sf http://127.0.0.1:9876/ready; do sleep 2; done
  # {"status":"ready","pool_available":2,"pool_total":2}
  ```

- A WebSocket client stack for your language: `pip install websockets` for
  the Python recipes, Node.js ≥ 22 (global `WebSocket`, no dependencies) for
  the JavaScript one, Go 1.23+ for the SDK one.
- A PCM16 mono source. For the copy-paste checks below, the repository
  ships a 4-second Russian speech fixture at exactly the right format
  (16 kHz mono Int16): `crates/gigastt/tests/fixtures/golos_00.wav`. For a
  real microphone, `ffmpeg` turns any capture device into raw PCM16 on
  stdout.

## Recipe

The session shape every recipe builds on — connect, read `ready`,
optionally `configure`, stream binary PCM16, `stop` to finalize:

```
Client                            Server
  |-------- connect --------------> |
  | <------- ready ----------------- |  limits + supported rates (read these!)
  |------- configure (optional) --> |  before the first audio frame
  |-------- binary PCM16 ---------> |
  | <------- partial / final ------ |
  |--------- stop ----------------> |
  | <------- final ----------------- |  trailing words flushed, then close
```

### Recipe 1: connect, negotiate, and stream the microphone

1. **Connect and read `ready` first.** The server's first message is always
   `ready`. It carries everything you must not hardcode: the protocol
   `version`, the accepted `supported_rates`, and the session caps
   (`max_session_secs`, `idle_timeout_secs`) that Recipe 4 plans around:

   ```json
   {
     "type": "ready",
     "model": "gigaam-v3-rnnt",
     "sample_rate": 48000,
     "version": "1.0",
     "supported_rates": [8000, 16000, 24000, 44100, 48000],
     "max_session_secs": 3600,
     "idle_timeout_secs": 300
   }
   ```

   `sample_rate` (48000 by default) is what the server expects if you never
   send `configure`. One caveat to know before debugging "silent connect":
   on a saturated pool the server answers with an `error` message *instead*
   of `ready` — see Recipe 5.

2. **Send `configure` before the first audio frame.** Pick the rate your
   capture pipeline actually emits — it must be one of
   `ready.supported_rates`. 16 kHz is the sweet spot when you control the
   source (the model runs at 16 kHz internally, so no resampling work); a
   browser capture at 48 kHz can also be sent as-is. An unsupported rate is
   not fatal: you get an `invalid_sample_rate` error and the session keeps
   the previous rate. A `configure` that arrives after the first audio
   frame is rejected with `configure_too_late` and the previous settings
   stay — so always send it right after `ready`:

   ```json
   {"type": "configure", "sample_rate": 16000}
   ```

3. **Send binary frames of PCM16** (signed 16-bit little-endian, mono) at
   the negotiated rate. Sensible chunking: **100–500 ms per frame**
   (3,200–16,000 bytes at 16 kHz). Partials are produced on a decode stride
   of roughly 0.8 s of new audio, so sub-second chunks keep the preview
   feeling live without per-frame overhead. The hard cap is
   `--ws-frame-max-bytes` (default 512 KiB) — a larger frame closes the
   socket with 1009. Odd byte counts are fine (the trailing byte is carried
   into the next frame), and occasional empty frames are tolerated. Stereo
   sources must be mixed down to mono first (`-ac 1` in ffmpeg).

4. **Put it together — microphone to live text.** This pipes the default
   mic through ffmpeg into a minimal Python client:

   ```sh
   # macOS (-f avfoundation); Linux: -f alsa -i default; Windows: -f dshow -i audio="..."
   ffmpeg -hide_banner -loglevel error -f avfoundation -i ":default" \
     -ac 1 -ar 16000 -f s16le - | python3 mic_stream.py
   ```

   `mic_stream.py` — the complete reference loop this chapter uses:

   ```python
   #!/usr/bin/env python3
   """Stream raw PCM16 from stdin to gigastt and print live transcripts.

   Usage: ffmpeg ... -f s16le - | python3 mic_stream.py [label] [server]
   """
   import asyncio
   import json
   import sys

   import websockets

   LABEL = sys.argv[1] if len(sys.argv) > 1 else "mic"
   SERVER = sys.argv[2] if len(sys.argv) > 2 else "ws://127.0.0.1:9876/v1/ws"
   RATE = 16000
   CHUNK = RATE * 2 // 5  # 400 ms of PCM16


   async def main() -> None:
       async with websockets.connect(SERVER) as ws:
           ready = json.loads(await ws.recv())
           assert ready["type"] == "ready", ready
           assert RATE in ready.get("supported_rates", [ready["sample_rate"]])
           await ws.send(json.dumps({"type": "configure", "sample_rate": RATE}))
           print(f"{LABEL}: connected to {ready['model']}", file=sys.stderr)

           async def receive() -> None:
               async for raw in ws:
                   msg = json.loads(raw)
                   if msg["type"] == "partial":
                       print(f"\r{LABEL} ... {msg['text']}   ", end="", flush=True)
                   elif msg["type"] == "final":
                       conf = msg.get("confidence")
                       suffix = f" ({conf:.2f})" if conf is not None else ""
                       print(f"\r{LABEL} >>> {msg['text']}{suffix}   ")
                   elif msg["type"] == "error":
                       print(f"\n{LABEL} ERR {msg['code']}: {msg['message']}")

           receiver = asyncio.create_task(receive())
           try:
               while data := await asyncio.to_thread(sys.stdin.buffer.read, CHUNK):
                   await ws.send(data)
               # Capture ended: finalize — never close before the trailing final.
               await ws.send(json.dumps({"type": "stop"}))
               await receiver
           except websockets.ConnectionClosed:
               pass  # server closed after the trailing final (or an error we printed)


   asyncio.run(main())
   ```

   Blocking stdin reads run in a thread (`asyncio.to_thread`) so the
   receiver task keeps printing while you speak.

**Check:** while you speak, `...` lines refresh about once a second; a
pause of roughly half a second produces a `>>>` final. Closing ffmpeg
(Ctrl+C) triggers one last `final` and a clean close — nothing is cut off.

### Recipe 2: show partials live, commit finals

The two transcript message types carry the same payload but play different
roles in the UI — treat them differently.

- **`partial` is a preview.** Always the raw decoder hypothesis: lowercase,
  no punctuation, and it *may change* as more audio arrives. A new partial
  lands roughly every 0.8 s of speech (the decode stride):

  ```json
  {
    "type": "partial",
    "text": "привет как",
    "timestamp": 1712700000.123,
    "is_final": false,
    "confidence": 0.93,
    "words": [
      {"word": "привет", "start": 0.0, "end": 0.4, "confidence": 0.97},
      {"word": "как", "start": 0.5, "end": 0.7, "confidence": 0.89}
    ]
  }
  ```

- **`final` is the committed line.** Enriched at the finalization boundary:
  inverse text normalization (number-words → digits), then punctuation and
  casing restoration. The `rnnt` head needs the punctuation model attached
  (server `--punctuation auto` default; check `GET /health` →
  `"punctuation":true`); the `e2e_rnnt` head punctuates by itself — see
  [Models and backends](04-models-and-backends.md). `words[]` always keep
  the raw decoder output in both message types; only the joined `text` is
  rewritten:

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

The UI rule of thumb: render the latest `partial` in a "preview" style
(grey/italic, replaceable), and when a `final` for the utterance lands,
replace the preview with `final.text` and append it to the committed
transcript. Never persist partial text.

A `final` fires when an utterance ends: the built-in endpointing triggers
on roughly 0.6 s of trailing silence, and with the optional Silero VAD
(server `--vad`, model auto-downloads) the endpoint follows
`--vad-min-silence-ms` (default 500 ms). Continuous speech without pauses
is also finalized — the streaming window caps out at ~2.5 s and slides, so
the transcript keeps committing even through a monologue. A `final` also
arrives on `stop` (Recipe 3) and before a server-initiated close (Recipe
4).

Per-session post-processing overrides compose with the same `configure`
message (finals only; partials always stay raw). Asking for
`punctuation: true` on a server without the model is a graceful no-op, not
an error:

```json
{"type": "configure", "sample_rate": 16000, "punctuation": false, "itn": false}
```

**Check:** say "привет как дела" with short pauses. The preview shows
`привет как дела` in lowercase; the committed line arrives as
`Привет, как дела?` (when the punctuation model is attached — verify with
`curl -s http://127.0.0.1:9876/health` showing `"punctuation":true`).

### Recipe 3: end the session cleanly — `stop` is the drain

The only correct end-of-stream pattern:

1. Send `{"type": "stop"}`.
2. **Wait for the `final`** — the server decodes whatever is still buffered
   since the last partial (trailing words are not lost) and emits one last
   `final`, possibly with empty `text` if nothing was pending.
3. Then close the socket (or let the server close it — it ends the session
   right after that final).

Do **not** close immediately after the last audio frame (the sub-stride
remainder would be dropped) and do **not** insert a fixed `sleep` drain —
the `final` after `stop` is the explicit, lossless end marker. The
`mic_stream.py` loop in Recipe 1 already implements this.

Keepalive needs no code on your side: the server pings every 30 s and
closes the connection after two consecutive pings with no inbound frame in
between (≈ 90 s detection of a half-open peer). Any inbound frame — a pong,
binary audio, or text — resets the counter, and standard WebSocket clients
answer pings automatically. Only raw-socket implementations must reply to
pings themselves.

**Check:** say one word and send `stop` immediately, before any partial
arrives. The word still shows up in the trailing `final`. Then watch the
close handshake: it happens *after* that final, not before.

### Recipe 4: keep long sessions alive (idle timeout, session cap)

Two server-side clocks govern every session; both are announced in `ready`
so you can plan instead of being surprised by a close frame.

- **Idle timeout** (`idle_timeout_secs`, default 300): any frame — audio,
  pong, or text — resets it. Pauses in speech only count as idle if the
  client stops sending; **streaming quiet PCM keeps the session alive
  (silence is still audio)**. When it trips: an `idle_timeout` error and
  close code 1001. If your app mutes capture during pauses, send silence
  frames instead of nothing.
- **Session cap** (`max_session_secs`, default 3600, `0` = disabled):
  wall-clock, starts at connect. When it trips: the server sends a
  `max_session_duration_exceeded` error, **flushes a `final` first**, then
  closes with 1008 — nothing already recognized is lost, so reconnecting is
  safe.

For recordings longer than an hour you have two options, and combining them
is fine:

1. **Operator side:** raise or disable the cap — `gigastt serve
   --max-session-secs 0` (every limit is a CLI flag; see
   [docs/cli.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/cli.md)).
2. **Client side:** rotate on a schedule. Read the cap from `ready`, and at
   ~90 % of it finalize with `stop` and open a fresh session — a planned
   reconnect, not an abrupt cut:

   ```python
   # rotate_sessions.py — keep transcription running across the server cap.
   import asyncio
   import json

   import websockets

   SERVER = "ws://127.0.0.1:9876/v1/ws"


   def handle_message(msg: dict) -> None:
       if msg["type"] == "final":
           print(">>>", msg["text"])
       elif msg["type"] == "error":
           print("ERR", msg["code"], msg["message"])


   async def run_session(ws, stream_audio, rotate_at: float | None) -> None:
       async def rotate() -> None:
           await asyncio.sleep(rotate_at)
           await ws.send(json.dumps({"type": "stop"}))  # flush now, reconnect fresh

       tasks = [asyncio.create_task(stream_audio(ws))]
       if rotate_at:
           tasks.append(asyncio.create_task(rotate()))
       try:
           async for raw in ws:  # ends when the server closes after the final
               handle_message(json.loads(raw))
       finally:
           for task in tasks:
               task.cancel()


   async def run_shift(stream_audio) -> None:
       """Rotate before the cap; back off on transient drops."""
       backoff = 0.25
       while True:
           try:
               async with websockets.connect(SERVER) as ws:
                   ready = json.loads(await ws.recv())
                   cap = ready["max_session_secs"]  # always sent; 0 = no cap
                   await run_session(ws, stream_audio, rotate_at=cap * 0.9 or None)
                   backoff = 0.25  # clean stop → final → close: rotate at once
           except (OSError, websockets.ConnectionClosed):
               # 1006 drop, a proxy-injected 1011, the 1008 cap close, network...
               await asyncio.sleep(backoff)
               backoff = min(backoff * 2, 30)
   ```

   `stream_audio(ws)` is your capture loop from Recipe 1. Note what this
   pattern does *not* do: it never retries fatal, fix-your-client errors
   (`unsupported_protocol_version`, a rate outside `supported_rates`) —
   those fail fast at handshake and must be fixed, not retried.

**Check:** start the server with a tiny cap to see the whole lifecycle in a
minute: `gigastt serve --max-session-secs 30 --idle-timeout-secs 10`. With
audio flowing you observe the `max_session_duration_exceeded` error, a
flushed `final`, close 1008, and the rotator rejoining without losing
text. Stop sending frames entirely and the `idle_timeout` error with close
1001 fires after 10 s.

### Recipe 5: confidence thresholds and backpressure

**Confidence.** Every transcript segment carries an optional `confidence` —
the duration-weighted mean of its `words[].confidence` (each word's is the
mean softmax score over its BPE tokens). It is an average of softmax
scores, **not a calibrated probability**, and it is omitted when the
segment has no words. As starting thresholds to tune on your own data:
highlight words below ~0.7 for human review, flag segments below ~0.8:

```js
const unsure = (msg.words ?? []).filter((w) => w.confidence < 0.7);
if (unsure.length) console.log("review:", unsure.map((w) => w.word).join(" "));
```

**Backpressure.** Inference slots come from a pool (`--pool-size`, default
2), and a WebSocket session holds its slot for its whole lifetime. When all
slots are busy, a new connection waits up to 30 s for one — and if none
frees up, the server answers *instead of `ready`* with a `timeout` error
carrying `retry_after_ms`, then closes. Honor the hint exactly instead of
guessing a delay:

```json
{"type": "error", "message": "Server busy, try again later", "code": "timeout", "retry_after_ms": 30000}
```

```python
first = json.loads(await ws.recv())          # may be an error, not ready
if first["type"] == "error" and first["code"] == "timeout":
    await asyncio.sleep(first["retry_after_ms"] / 1000)  # then reconnect
```

For everything else that closes the socket unexpectedly — a 1006 abnormal
drop, a proxy-injected code such as 1011, a network blip — reconnect with
exponential backoff (start ~250 ms, double up to a few seconds, cap the
attempts), as in `run_shift` from Recipe 4. Never retry the fatal
handshake errors (`unsupported_protocol_version`). Both official SDKs
implement exactly this policy — configure-first handshake, `retry_after_ms`
honored on pool saturation, exponential backoff otherwise — so prefer them
over hand-rolling: [sdks/go](https://github.com/ekhodzitsky/gigastt/tree/main/sdks/go),
[sdks/js](https://github.com/ekhodzitsky/gigastt/tree/main/sdks/js).

**Check:** start with `--pool-size 1` and connect two clients at once. The
second sits ~30 s, then receives `{"code":"timeout","retry_after_ms":30000}`;
after sleeping it off, it connects once the first session ends. For
confidence: stream a clean speech file and confirm finals carry
`confidence` close to 1.0, then try quiet/noisy audio and watch words drop
below 0.7.

### Recipe 6: two channels — mic + system audio

The meeting-assistant pattern (your mic + the far end's system audio, each
labeled) maps onto **two independent WebSocket sessions** — the server does
not tag sources, so label them client-side. The constraint to design around
is the pool: each session holds one inference slot for its entire lifetime,
so the default `--pool-size 2` fits exactly two channels and nothing else.
Give the server headroom if other clients also connect — each extra slot
costs roughly 0.4 GB RAM with the INT8 encoder (the server caps the pool to
available RAM at load):

```sh
gigastt serve --pool-size 4
```

Then run one capture pipeline per source, each with its own label (the
`mic_stream.py` from Recipe 1 takes the label as its first argument):

```sh
# terminal 1 — your microphone (macOS example; Linux: -f alsa -i default)
ffmpeg -hide_banner -loglevel error -f avfoundation -i ":default" \
  -ac 1 -ar 16000 -f s16le - | python3 mic_stream.py mic

# terminal 2 — system audio (Linux Pulse/PipeWire monitor source;
# macOS: a virtual device such as BlackHole selected via avfoundation)
ffmpeg -hide_banner -loglevel error -f pulse -i default.monitor \
  -ac 1 -ar 16000 -f s16le - | python3 mic_stream.py system
```

Two lighter-weight alternatives when speaker-labeled channels are not
required: mix both sources into one stream client-side (one pool slot, one
session), or use per-word speaker labels from diarization — a server built
with `--features diarization` advertises `"diarization": true` in `ready`,
and `{"type":"configure","diarization":true}` then adds a `speaker` field
to each word (see the reference in
[docs/api.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/api.md)).

**Check:** both terminals print with their distinct `mic` / `system`
labels; while both run, `curl -s http://127.0.0.1:9876/ready` shows
`"pool_available":0,"pool_total":2` (with the default pool), and a third
client gets the `timeout` + `retry_after_ms` treatment from Recipe 5.

### Recipe 7: client skeletons (Python, Node.js, Go)

Minimal but complete loops to start from — each does the full
ready → configure → stream → stop → wait-for-final → close cycle with
error handling. More clients (Bun, Kotlin, Rust) live in
[examples/](https://github.com/ekhodzitsky/gigastt/tree/main/examples).

**Python** — streams a 16 kHz mono PCM16 WAV, honors `retry_after_ms` on
pool saturation:

```python
#!/usr/bin/env python3
"""Stream a 16 kHz mono PCM16 WAV to gigastt, honoring server backpressure.

Usage: python3 stream_wav.py <audio.wav> [ws://host:port]
"""
import asyncio
import json
import sys
import wave

import websockets


class PoolBusy(Exception):
    """Pool saturated: the server sent retry_after_ms instead of ready."""


async def session(path: str, server: str) -> None:
    async with websockets.connect(server) as ws:
        first = json.loads(await ws.recv())
        if first["type"] == "error":  # pool busy: error + close instead of ready
            raise PoolBusy(first.get("retry_after_ms", 30000))
        assert first["type"] == "ready", first
        await ws.send(json.dumps({"type": "configure", "sample_rate": 16000}))

        with wave.open(path, "rb") as wav:
            assert wav.getnchannels() == 1 and wav.getsampwidth() == 2, "mono PCM16 WAV"
            pcm = wav.readframes(wav.getnframes())

        async def receive() -> None:
            async for raw in ws:
                msg = json.loads(raw)
                if msg["type"] == "partial":
                    print(f"\r... {msg['text']}   ", end="", flush=True)
                elif msg["type"] == "final":
                    print(f"\r>>> {msg['text']}   ")
                elif msg["type"] == "error":
                    print(f"\nERR {msg['code']}: {msg['message']}")

        receiver = asyncio.create_task(receive())
        try:
            for off in range(0, len(pcm), 16000):  # 0.5 s of PCM16 at 16 kHz
                await ws.send(pcm[off : off + 16000])
                await asyncio.sleep(0.1)  # pace it like a live feed
            await ws.send(json.dumps({"type": "stop"}))
            await receiver  # trailing final, then the server closes
        except websockets.ConnectionClosed:
            pass


async def main() -> None:
    server = sys.argv[2] if len(sys.argv) > 2 else "ws://127.0.0.1:9876/v1/ws"
    while True:
        try:
            await session(sys.argv[1], server)
            return
        except PoolBusy as busy:
            wait = busy.args[0] / 1000
            print(f"pool busy, retrying in {wait:.0f}s")
            await asyncio.sleep(wait)  # honor retry_after_ms exactly


asyncio.run(main())
```

**Node.js** — no dependencies (global `WebSocket`, Node ≥ 22), same WAV
file contract:

```js
// stream_wav.mjs — stream a 16 kHz mono PCM16 WAV to gigastt (Node.js ≥ 22).
import { readFile } from "node:fs/promises";

const [wavPath, server = "ws://127.0.0.1:9876/v1/ws"] = process.argv.slice(2);
if (!wavPath) {
  console.error("usage: node stream_wav.mjs <audio.wav> [ws://host:port]");
  process.exit(1);
}

const pcm = (await readFile(wavPath)).subarray(44); // skip the WAV header
const ws = new WebSocket(server);
ws.binaryType = "arraybuffer";

let stopped = false;
const done = new Promise((resolve, reject) => {
  ws.onmessage = async (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "ready") {
      ws.send(JSON.stringify({ type: "configure", sample_rate: 16000 }));
      for (let off = 0; off < pcm.byteLength; off += 16000) { // 0.5 s at 16 kHz
        ws.send(pcm.subarray(off, off + 16000));
        await new Promise((r) => setTimeout(r, 100)); // pace like a live feed
      }
      stopped = true;
      ws.send(JSON.stringify({ type: "stop" })); // finalize — do NOT close yet
    } else if (msg.type === "partial") {
      process.stdout.write(`\r... ${msg.text}   `);
    } else if (msg.type === "final") {
      console.log(`\r>>> ${msg.text}   `);
      if (stopped) { ws.close(); resolve(); } // trailing final: safe to close
    } else if (msg.type === "error") {
      const hint = msg.retry_after_ms ? ` (retry in ${msg.retry_after_ms} ms)` : "";
      reject(new Error(`${msg.code}: ${msg.message}${hint}`));
    }
  };
  ws.onerror = () => reject(new Error("websocket transport error"));
});
await done;
```

For anything beyond a skeleton, the typed SDK adds the reconnect policy
from Recipe 5 out of the box:
`npm install @gigastt/client` — see
[sdks/js](https://github.com/ekhodzitsky/gigastt/tree/main/sdks/js).

**Go** — on the official SDK (`go get github.com/ekhodzitsky/gigastt/sdks/go@latest`),
which pins the protocol version, sends `configure` first, and retries
backpressure with the server's `retry_after_ms`:

```go
// stream_wav.go — stream a 16 kHz mono PCM16 WAV to gigastt via the Go SDK.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	gigastt "github.com/ekhodzitsky/gigastt/sdks/go"
)

func main() {
	if len(os.Args) < 2 {
		log.Fatal("usage: go run stream_wav.go <audio.wav>")
	}
	done := make(chan struct{})

	client, err := gigastt.Dial(context.Background(), gigastt.DefaultURL,
		gigastt.WithSampleRate(16000), // must be in the server's supported_rates
		gigastt.WithReconnect(250*time.Millisecond, 5*time.Second, 10),
		gigastt.WithHandlers(gigastt.Handlers{
			OnPartial: func(t gigastt.Transcript) { fmt.Printf("\r... %s   ", t.Text) },
			OnFinal:   func(t gigastt.Transcript) { fmt.Printf("\r>>> %s   \n", t.Text) },
			OnError:   func(e *gigastt.ServerError) { log.Printf("server error: %v", e) },
			OnClose: func(err error) {
				if err != nil {
					log.Printf("connection closed: %v", err)
				}
				close(done)
			},
		}),
	)
	if err != nil {
		log.Fatal(err) // e.g. *gigastt.ServerError unsupported_protocol_version
	}
	defer client.Close()

	wav, err := os.ReadFile(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}
	for off := 44; off < len(wav); off += 16000 { // skip WAV header; 0.5 s chunks
		if err := client.SendPCM(wav[off:min(off+16000, len(wav))]); err != nil {
			log.Fatal(err) // ErrReconnecting: drop or retry the frame yourself
		}
		time.Sleep(100 * time.Millisecond) // pace like a live feed
	}
	if err := client.Stop(); err != nil { // finalize; server closes after the final
		log.Fatal(err)
	}
	<-done
}
```

**Check:** point any of the three at the repository fixture —
`python3 stream_wav.py crates/gigastt/tests/fixtures/golos_00.wav` — and
you get a few `...` partials followed by `>>>` finals of Russian speech,
then a clean exit after the trailing final.

## Verifying the result

End-to-end sanity in two terminals:

```sh
# terminal 1
gigastt serve

# terminal 2 — wait for readiness, then stream the bundled speech fixture
until curl -sf http://127.0.0.1:9876/ready; do sleep 2; done
python3 stream_wav.py crates/gigastt/tests/fixtures/golos_00.wav
```

You have a healthy streaming integration when all of these hold:

- The client's first message is `ready` with `version: "1.0"`, and your
  configured rate is one of its `supported_rates`.
- `...` partials appear about once a second while audio flows and are
  lowercase/raw; `>>>` finals commit enriched text after each pause and
  carry `words[]` with timings and (usually) `confidence`.
- Sending `stop` yields exactly one more `final` (possibly empty) and the
  socket closes only after it — stopping mid-word still recognizes that
  word.
- `ready.max_session_secs` / `ready.idle_timeout_secs` match the values you
  passed to `gigastt serve`, and your rotation/backoff logic (Recipe 4/5)
  survives a forced cap: `gigastt serve --max-session-secs 30`.

## Common pitfalls

Symptom → cause, with a pointer to the fix — the full operator table lives
in [docs/troubleshooting.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/troubleshooting.md)
and the error/close code tables in
[docs/api.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/api.md#error-codes);
do not debug from memory.

| Symptom | Cause | Where to look |
|---|---|---|
| Connect hangs ~30 s, then `{"code":"timeout","retry_after_ms":30000}` | Pool saturation — the checkout happens *before* `ready`, so the error replaces it | Recipe 5; [api.md error codes](https://github.com/ekhodzitsky/gigastt/blob/main/docs/api.md#error-codes) |
| Socket closes 1008 exactly at the 1-hour mark | `--max-session-secs` cap; a `final` is flushed first, so just reconnect | Recipe 4; [troubleshooting](https://github.com/ekhodzitsky/gigastt/blob/main/docs/troubleshooting.md) |
| Socket closes 1001 after ~5 min of silence | Idle timeout — no frames at all; stream quiet PCM to stay alive | Recipe 4; [troubleshooting](https://github.com/ekhodzitsky/gigastt/blob/main/docs/troubleshooting.md) |
| Socket closes 1009 | A frame exceeded `--ws-frame-max-bytes` (default 512 KiB) — chunk smaller | Recipe 1; [api.md limits](https://github.com/ekhodzitsky/gigastt/blob/main/docs/api.md#session-and-frame-limits) |
| Upgrade refused with HTTP 503 `{"code":"initializing"}` | Model still downloading/quantizing — poll `/ready`, don't restart | [troubleshooting](https://github.com/ekhodzitsky/gigastt/blob/main/docs/troubleshooting.md) |
| Browser app from another origin can't connect | Origin allowlist — loopback origins only by default; add `--allow-origin` | [docs/cli.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/cli.md) |
| Finals arrive bare lowercase, no punctuation | Punctuation model not attached, or policy off; `e2e_rnnt` punctuates itself | Recipe 2; [troubleshooting](https://github.com/ekhodzitsky/gigastt/blob/main/docs/troubleshooting.md) |
| `configure` has no effect | Sent after the first audio frame (`configure_too_late`) — send it right after `ready` | Recipe 1 |
| No transcript at all | Three independent failure domains: server readiness, audio capture, language/head | [troubleshooting triage](https://github.com/ekhodzitsky/gigastt/blob/main/docs/troubleshooting.md#no-transcript-audio-capture-vs-stt-startup-vs-language-config) |

A Deepgram-compatible WebSocket mode (a drop-in endpoint for Deepgram
clients) is in progress; this chapter covers the native protocol only.

## Links

- Reference (canonical, not duplicated here):
  [docs/api.md — WebSocket protocol](https://github.com/ekhodzitsky/gigastt/blob/main/docs/api.md#websocket--real-time-streaming),
  [docs/asyncapi.yaml](https://github.com/ekhodzitsky/gigastt/blob/main/docs/asyncapi.yaml),
  [docs/troubleshooting.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/troubleshooting.md),
  [docs/cli.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/cli.md)
- Client code:
  [examples/](https://github.com/ekhodzitsky/gigastt/tree/main/examples)
  (Python, Bun/TypeScript, Go, Kotlin, Rust),
  [sdks/go](https://github.com/ekhodzitsky/gigastt/tree/main/sdks/go),
  [sdks/js](https://github.com/ekhodzitsky/gigastt/tree/main/sdks/js)
- In this book: [Getting started](01-getting-started.md) for install and
  first run, [Models and backends](04-models-and-backends.md) for heads and
  punctuation/ITN behavior, [Server integration](05-server-integration.md)
  for REST/SSE/jobs alternatives to live streaming,
  [Deployment & ops](06-deployment-ops.md) for running `serve` in
  production.
