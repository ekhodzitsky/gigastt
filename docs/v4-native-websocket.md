# Native GigaAM v4 WebSocket profile

On GNU/Linux x86-64, build with `--features v4-mkl` and explicitly provision
both the [full-context file pack](v4-pack-validation.md) and the separate
[qualified cached-streaming pack](v4-streaming-pack.md). The operator-supplied
oneMKL directory is shared. This enables native v4 on the existing `/v1/ws`
route of that v4 server only; the ordinary v3 listener and wire version 1.3
are unchanged. The full-context file profile without a streaming pack still
rejects `/v1/ws`.

```sh
gigastt serve --model-family gigaam_v4 --model-head rnnt --model-precision fp32 \
  --transcript-style raw --model-dir /path/to/v4-file-pack \
  --v4-mkl-runtime-dir /path/to/runtime \
  --v4-streaming-model-dir /path/to/v4-streaming-pack \
  --v4-streaming-future-context-ms 320
```

`--v4-streaming-model-dir` (`GIGASTT_V4_STREAMING_MODEL_DIR`) is opt-in, not a
download instruction. The optional `--v4-streaming-future-context-ms`
(`GIGASTT_V4_STREAMING_FUTURE_CONTEXT_MS`) defaults to 320. Native contexts are
160, 320, and 640 ms. The file server's style is the default for a native
connection, not a constraint on that connection's explicitly selected style.
`/health` and `/v1/models` expose the native capability only when the second
pack was successfully loaded. HTTP readiness remains the file admission view.
No model, runtime, provider, or protocol fallback occurs.

## Upgrade and control

Offer the `gigastt-native-v4` WebSocket subprotocol. The server must negotiate
it and send a Ready message with `version: "1.4"`, `sample_rate: 16000`,
`supported_rates: [16000]`, and `native_v4`. A legacy server rejects the
reserved subprotocol before WebSocket upgrade. A native server does not accept
an upgrade without this explicit offer; no legacy 48 kHz audio is silently
interpreted as 16 kHz.

The upgrade URL can select the fixed connection style and context, for example
`/v1/ws?transcript_style=e2e&future_context_ms=640`. Optional selector echoes
`model_family=gigaam_v4`, `model_head=rnnt`, and `model_precision=fp32` are
checked. Unknown or duplicate selectors fail before starting the session.
`native_v4` reports `model_family`, `model_head`, `model_precision`,
`transcript_style`, `future_context_ms`, `supported_future_context_ms`,
`max_packet_bytes`, and `endpoint_mode: "manual"`.

Before the first audio message, a `configure` message can confirm 16 kHz, the
selected style/context and the protocol version. Configuration cannot change
the Ready profile; open a new connection to change it or reset the state.
After audio, `configure` is rejected. Native v4 does not implement legacy VAD,
hotwords, speaker diarization, language hints, blank-run endpointing, context
windows, or extra punctuation/ITN. Unsupported control fields fail rather
than activate a different recognizer.

## Audio, deltas, and lifecycle

Send nonempty, whole little-endian PCM16 samples, mono at exactly 16 kHz. The
server converts each signed sample to float by multiplying by `1/32768`, with
no resampling, automatic packet splitting, or waveform padding. Packet sizes
need not be aligned to the mel hop. Each packet is at most 32,000 bytes (one
second), or the configured lower `--ws-frame-max-bytes` bound. The stream owns
its model state and accepts at most an hour of audio; the independent
`--max-session-secs` wall-clock and `--idle-timeout-secs` limits also apply.

`partial` and `final` keep the familiar cumulative `text` and `words` snapshot.
`text == committed`, `tentative` is empty, and the optional `native_delta`
contains *only* the new packet's `text`, `words`, and `tokens` plus absolute
`packet_one_based`, total `samples`, and total `encoded_frames`. A native token
is a nonblank ID, absolute encoder frame and selected-label confidence. It may
precede the completed text of its word; do not render those tokens as a second
transcript. A packet with no text, words or tokens does not emit a partial.
Consume each delta once, or replace a snapshot. Do not append the full final
snapshot to earlier snapshots.

`{"type":"stop"}` flushes once, sends one `final` (even if no speech was
recognized), then closes normally. Timestamps are relative to the utterance's
40 ms encoder grid, with final word ends clamped to accepted audio duration.
Closing or cancelling a socket does not flush or synthesize a successful
final. A failed/timed-out processing call cannot publish a partial from that
call. The two session reservations are retained by blocking work until it
actually exits; cancellation is cooperative, not native preemption. Busy
sessions report a retry hint; unread writes are bounded. Reconnect starts a
fresh stream, and clients do not replay buffered audio.

## Context and accuracy

160/320/640 ms selects 4/8/16 encoder frames of configured future context.
Shorter context can make the first acoustic block available sooner; the
first complete input block needs approximately 170/330/650 ms of continuously
arriving audio, including the feature window. This is a geometry bound, not a
measured end-to-end latency: arrival schedule, decode time and completion of
word boundaries matter. Different context choices and packet schedules can
change recognition. This native cached model is not equivalent to offline
full-context, legacy v3 streaming, or any particular word-error rate.

JavaScript and Go clients can opt in using their explicit native v4 options.
Both verify the negotiated subprotocol and exact Ready capability before
returning a client, including on reconnect.
