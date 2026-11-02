# gigastt-go

Typed Go WebSocket client for the [gigastt](https://github.com/ekhodzitsky/gigastt)
speech-to-text server (GigaAM v3, Russian ASR). Protocol version **1.0**
(`PROTOCOL_VERSION` in `crates/gigastt-core/src/protocol/mod.rs`).

- Typed `Ready` / `Transcript` (partial & final) / `ServerError` events with all
  protocol fields, including `words[]`, `error.code` and `error.retry_after_ms`
- Callback-based dispatch (`OnPartial` / `OnFinal` / `OnError` / `OnClose`)
- Optional automatic reconnect with exponential backoff that honors the
  server's `retry_after_ms` hint exactly on pool-saturation backpressure
- Single serialized writer; safe for concurrent `SendPCM` / `Stop` / `Close`
- `context.Context` governs the whole client lifetime

## Install

```sh
go get github.com/ekhodzitsky/gigastt/sdks/go@latest
```

Requires Go 1.23+.

## Quickstart

Prepare raw little-endian mono PCM16 before streaming:

```sh
ffmpeg -i recording.wav -f s16le -acodec pcm_s16le -ac 1 -ar 16000 recording.pcm
```

WAV headers have variable length; do not strip a fixed 44-byte prefix.
The example prints every final segment and waits for session closure.

```go
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
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	closed := make(chan error, 1)

	client, err := gigastt.Dial(ctx, gigastt.DefaultURL,
		gigastt.WithSampleRate(16000), // must be one of the server's supported_rates
		gigastt.WithHandlers(gigastt.Handlers{
			OnPartial: func(t gigastt.Transcript) { fmt.Printf("\r  ... %s", t.Text) },
			OnFinal: func(t gigastt.Transcript) {
				fmt.Printf("\n>>> %s (%.0f%% avg confidence)\n", t.Text, avgConfidence(t)*100)
			},
			OnClose: func(err error) { closed <- err },
			OnError:   func(e *gigastt.ServerError) { log.Printf("server error: %v", e) },
		}),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	pcm, err := os.ReadFile("recording.pcm")
	if err != nil {
		log.Fatal(err)
	}
	for offset := 0; offset < len(pcm); offset += 32768 {
		end := min(offset+32768, len(pcm))
		if err := client.SendPCM(pcm[offset:end]); err != nil {
			log.Fatal(err)
		}
	}

	if err := client.Stop(); err != nil { // ask the server to finalize
		log.Fatal(err)
	}
	select {
	case err := <-closed:
		if err != nil {
			log.Fatal(err)
		}
	case <-ctx.Done():
		log.Fatal(ctx.Err())
	}
}

func avgConfidence(t gigastt.Transcript) float64 {
	var sum float64
	for _, w := range t.Words {
		sum += w.Confidence
	}
	if len(t.Words) == 0 {
		return 0
	}
	return sum / float64(len(t.Words))
}
```

## Behavior notes

- **Configure first.** `Dial` always sends a `configure` frame pinning
  `protocol_version` (plus `sample_rate` / `diarization` when set) and waits
  for the server's `ready` before returning. A version mismatch fails loudly
  with an `unsupported_protocol_version` `*ServerError` (use `errors.As`).
- **Stop finalizes.** `Stop()` asks the server to flush and emit a final
  transcript; the server then ends the session. This close is treated as a
  normal end — `OnClose(nil)` fires and no reconnect is attempted.
- **Reconnect.** Transient failures (network drops, abnormal closes) retry with
  exponential backoff from `minBackoff` up to `maxBackoff`, bounded by
  `maxAttempts` (0 = unlimited). Server errors are retried **only** when they
  carry `retry_after_ms` (transient backpressure such as pool saturation), and
  then the client waits exactly that hint. Fatal errors
  (`unsupported_protocol_version`, `invalid_sample_rate`, …) are never retried.
  While a reconnect is in flight, `SendPCM` fails fast with `ErrReconnecting` —
  the SDK does not buffer audio across reconnects.
- **Callbacks** run on the client's read goroutine: keep them fast and do not
  call `Stop`/`Close` synchronously from inside one (use a goroutine).

## Compatibility policy

The SDK pins protocol `1.0` (`gigastt.ProtocolVersion`). Additive protocol
changes are tolerated by design: unknown JSON fields and unknown message types
are ignored on decode. A server that cannot speak protocol 1.x rejects the
session during the handshake.

## Testing

Unit tests run against an in-process mock WebSocket server
(`httptest` + `gorilla/websocket`):

```sh
go test ./...
```

### Integration test against a live server

Requires a running server with the model (~310 MiB for the current main ASR bundle):

```sh
# once: download the model (~310 MiB for the current main ASR bundle), then start the server from the repository root
cargo run -- download
cargo run --release -- serve

# in another terminal, from this directory
GIGASTT_TEST_WS_URL=ws://127.0.0.1:9876/v1/ws go test -run Live -v
```

The live test (`live_test.go`) is skipped unless `GIGASTT_TEST_WS_URL` is set;
it streams a generated sine tone and asserts a complete
ready → audio → final session.

## Publishing

The module path is `github.com/ekhodzitsky/gigastt/sdks/go`. Releases are cut
from the monorepo with subdirectory tags of the form `sdks/go/vX.Y.Z` (Go
modules convention). Publishing is part of the human release process and is
**not** performed from CI.

## License

MIT — see [LICENSE](../../LICENSE).

The optional `WithEndpointSilenceMS(1000)` setting requires server protocol 1.3. It controls
decoder blank-run endpoint duration (40–2000 ms, default 600), only in auto mode
without VAD. It is not acoustic silence detection. When set, the SDK requests
protocol 1.3 so older servers reject the handshake instead of ignoring the
setting. When omitted, the SDK retains its protocol 1.0 compatibility.
