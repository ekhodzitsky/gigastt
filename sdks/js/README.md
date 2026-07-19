# @gigastt/client

Typed TypeScript WebSocket client for the
[gigastt](https://github.com/ekhodzitsky/gigastt) speech-to-text server
(GigaAM v3, Russian ASR). Protocol version **1.0** (`PROTOCOL_VERSION` in
`crates/gigastt-core/src/protocol/mod.rs`). Works in **Node.js ≥ 20** and in
**browsers**.

- Typed `ReadyMessage` / `Transcript` (partial & final) / `ServerError` events
  with all protocol fields, including `words[]`, `error.code` and
  `error.retry_after_ms`
- Callback-based dispatch (`onPartial` / `onFinal` / `onError` / `onClose`)
- Optional automatic reconnect with exponential backoff that honors the
  server's `retry_after_ms` hint exactly on pool-saturation backpressure
- `AbortController` governs the whole client lifetime
- Transport: the global `WebSocket` (browsers, Node ≥ 22), the `ws` package as
  an automatic Node fallback, or any transport injected via `webSocketFactory`

## Install

```sh
npm install @gigastt/client
```

The `ws` package is a regular dependency but is only loaded (via dynamic
import) when no global `WebSocket` exists; browser bundlers can safely mark it
external — it is never imported on the browser path.

## Quickstart

```ts
import { GigasttClient, DEFAULT_URL } from "@gigastt/client";
import { readFile } from "node:fs/promises";

const controller = new AbortController();

const client = await GigasttClient.connect(DEFAULT_URL, {
  sampleRate: 16000, // must be one of the server's supported_rates
  signal: controller.signal,
  reconnect: { minDelayMs: 250, maxDelayMs: 5000, maxAttempts: 10 },
  handlers: {
    onPartial: (t) => process.stdout.write(`\r  ... ${t.text}`),
    onFinal: (t) => {
      console.log(`\n>>> ${t.text}`);
      for (const w of t.words ?? []) {
        console.log(`    ${w.start.toFixed(2)}s-${w.end.toFixed(2)}s ${w.word} (${w.confidence})`);
      }
    },
    onError: (err) => console.error(`server error ${err.code}: ${err.message}`),
    onClose: ({ intentional, error }) => {
      if (!intentional) console.error("connection lost:", error);
    },
  },
});

console.log(`connected: ${client.ready.model} @ ${client.ready.sample_rate} Hz`);

// Stream a 16 kHz mono PCM16 WAV file (skipping the 44-byte header) in chunks.
const wav = await readFile("recording.wav");
for (let offset = 44; offset < wav.byteLength; offset += 32768) {
  client.sendPCM(wav.subarray(offset, offset + 32768));
}

client.stop(); // ask the server to flush and emit the final transcript
```

## Behavior notes

- **Configure first.** `connect` always sends a `configure` frame pinning
  `protocol_version` (plus `sample_rate` / `diarization` when set) and resolves
  only after the server's `ready`. A version mismatch rejects with a
  `ServerError` whose `code` is `unsupported_protocol_version`.
- **Stop finalizes.** `stop()` asks the server to flush and emit a final
  transcript; the server then ends the session. This close is treated as a
  normal end — `onClose` fires with `intentional: true` and no reconnect is
  attempted.
- **Reconnect.** Transient failures (network drops, abnormal closes) retry with
  exponential backoff from `minDelayMs` up to `maxDelayMs`, bounded by
  `maxAttempts` (0 = unlimited). Server errors are retried **only** when they
  carry `retry_after_ms` (transient backpressure such as pool saturation), and
  then the client waits exactly that hint. Fatal errors
  (`unsupported_protocol_version`, `invalid_sample_rate`, …) are never retried.
  While a reconnect is in flight, `sendPCM` throws — the SDK does not buffer
  audio across reconnects; drop or retry frames explicitly.
- **Abort.** Aborting the signal rejects a pending `connect`, interrupts
  reconnect backoff with an `AbortError`, and closes an established session.

## Compatibility policy

The SDK pins protocol `1.0` (`PROTOCOL_VERSION`). Additive protocol changes are
tolerated by design: unknown JSON fields and unknown message types are ignored
on decode. A server that cannot speak protocol 1.x rejects the session during
the handshake.

## Testing

Unit tests run against an in-process mock WebSocket server (`ws` package +
vitest):

```sh
npm install
npm test          # vitest run
npm run typecheck # tsc --noEmit (strict)
npm run build     # emit dist/ (ESM + .d.ts)
```

### Integration test against a live server

Requires a running server with the model (~850 MB download):

```sh
# once: download the model (~850 MB), then start the server from the repository root
cargo run -- download
cargo run --release -- serve

# in another terminal, from this directory
GIGASTT_TEST_WS_URL=ws://127.0.0.1:9876/v1/ws npx vitest run test/live.test.ts
```

The live test (`test/live.test.ts`) is skipped unless `GIGASTT_TEST_WS_URL` is
set; it streams a generated sine tone and asserts a complete
ready → audio → stop → final session.

## Publishing

The package manifest (`package.json`) is release-ready:
`npm publish --access public` from this directory (runs `prepublishOnly` →
`npm run build`). Versioning follows the monorepo release process; publishing
is done by a human during the release, **not** from CI in this change.

## License

MIT — see [LICENSE](../../LICENSE).
