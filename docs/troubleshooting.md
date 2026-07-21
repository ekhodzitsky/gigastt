# Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `protoc` not found during build | Missing Protocol Buffers compiler | `brew install protobuf` (macOS) or `apt install protobuf-compiler` (Debian/Ubuntu) |
| Model download hangs or fails | Network / HuggingFace availability | Retry `gigastt download`; check `~/.gigastt/models/` permissions |
| `Cannot quantize: FP32 encoder not found` | Partial download | Delete `~/.gigastt/models/` and re-run `gigastt download` |
| OOM on startup | Pool size too large for available RAM | Lower `--pool-size` (default 2); each session loads the full encoder |
| CoreML not used on macOS | Built without `--features coreml` | Re-build: `cargo build --release --features coreml` |
| `falling back to CPU execution provider` in logs | CoreML failed to compile/execute on this macOS/model combo | Transcription still works on CPU; clear `~/.gigastt/models/coreml_cache/` and retry, or file an issue with the warning text |
| CUDA not available on Linux | Built without `--features cuda` or missing CUDA 12+ | Re-build: `cargo build --release --features cuda`; verify `nvidia-smi` |
| WebSocket closes with 1008 | Session exceeded `--max-session-secs` (default 3600 — a stream dying *exactly at the 1-hour mark* is this cap, not a crash). The server sends `max_session_duration_exceeded`, flushes a `final`, then closes | Raise `--max-session-secs` (0 disables), or reconnect on the close — the flushed `final` means nothing recognized is lost |
| WebSocket closes with 1001 after ~5 min of silence | No frames for `--idle-timeout-secs` (default 300) — pauses in speech count as idle if the client stops sending | Keep streaming PCM silence while the mic is open (silence is still audio), or raise `--idle-timeout-secs` |
| `Address already in use` on startup, port 9876 | Another process holds the port — often an orphaned gigastt from a previous run | Find it: `lsof -nP -tiTCP:9876 -sTCP:LISTEN`; confirm it is ours before killing: `ps -p <pid> -o command=` should show `gigastt serve`. Then `kill <pid>` (SIGTERM drains sessions cleanly), or start on a different `--port` |
| gigastt keeps running after the parent app was force-killed | SIGKILL gives the parent no chance to forward SIGTERM, so a managed sidecar is orphaned and keeps the port and ~800 MB of RAM | Same `lsof`/`ps` recipe as above to find and stop it. Long-term: supervise with launchd/systemd, or have the parent poll its own PID liveness and SIGTERM the child on exit paths that do run |
| WS `final` text arrives bare lowercase, no punctuation | The punctuation model is not attached (check `GET /health` → `punctuation:false`), or policy is off for this server/session. `e2e_rnnt` punctuates by itself; the bare `rnnt` head needs the punct model | Enable server-wide with `--punctuation on` (model auto-downloads to `~/.gigastt/models/punct/`), per WS session with `{"type":"configure","punctuation":true}`, or per REST request with `?punctuation=true`. `words[]` always stay raw by design — only the joined `text` is rewritten |
| First `serve` takes minutes before `/ready` turns green | One-time setup: ~850 MB model download + INT8 quantization of the encoder. `/health` answers 200 `{"model":"loading"}` meanwhile — the server is not hung | Pre-seed during install: `gigastt download` (or `gigastt download --prequantized` to fetch the ~215 MB pre-quantized INT8 bundle and skip local quantization entirely). Gate clients on `/ready`, never on the process being alive |
| Every inference fails on a Homebrew install (CoreML) | Builds earlier than 2.0.14 shipped a broken CoreML runtime; the brew tarball is compiled `--features coreml` | Check the version without exec-ing the binary: `GET /health` → `version`. Upgrade: `brew update && brew upgrade gigastt`. Integrations should gate on **≥ 2.0.14** |
| WS upgrade fails with HTTP 503 `{"code":"initializing"}` but the port is listening | The model is still loading (bootstrap responder) — this is the healthy first-run state, not a stuck server | Poll `GET /ready` until 200, then connect. Killing and restarting only restarts the download/quantization |
| 429 Too Many Requests | Rate limiter enabled and bucket exhausted | Wait for `Retry-After`, or disable with `--rate-limit-per-minute 0` |
| Empty transcription for noisy audio | Input too quiet or wrong format | Ensure 16-bit PCM; normalize level; check supported formats |
| Diarization returns no speaker labels; logs show `Got: 2 Expected: 3` | Pre-2.11.2 bug — the WeSpeaker encoder was fed rank-2 waveform instead of rank-3 fbank features | Upgrade to **2.11.2+** (fixed); ensure `wespeaker_resnet34.onnx` is present in `~/.gigastt/models/` |
| `--model-variant` ignored when the model dir holds more than one head | Pre-2.11.1 bug — the engine re-detected the head from disk (`rnnt` precedence) and dropped the requested variant | Upgrade to **2.11.1+** (the resolved variant is now honored); or keep each head in its own `--model-dir` |

## No transcript: audio capture vs STT startup vs language config

"No text comes out" has three independent failure domains. Triage them in this
order — each step isolates one:

1. **STT startup** — is the server actually ready?
   `curl -s localhost:9876/ready` must return 200 with `"status":"ready"`.
   A 503 `initializing` means the model is still downloading/quantizing (wait);
   `pool_exhausted` means all inference slots are busy (retry with backoff —
   WS clients get `retry_after_ms`). A WS client that connected must have
   received the `ready` message before any audio counts.
2. **Audio capture** — is audio actually reaching the server?
   Log frame sizes client-side: a stream of zero-length or silent frames
   produces empty partials. Confirm the negotiated rate
   (`ready.supported_rates`, your `configure.sample_rate`) matches what the
   capture pipeline emits, PCM16 little-endian mono. On macOS, check the host
   app's microphone permission (System Settings → Privacy & Security →
   Microphone). To cut the server out of the loop entirely, POST a known-good
   WAV to `/v1/transcribe` — if REST returns text, startup and the model are
   fine and the bug is in capture or the streaming client.
3. **Language / model config** — is the loaded head the one you think?
   gigastt has no per-request language switch: language is a property of the
   loaded model head. Check `ready.model` (WS) or `/health` → `model`: `rnnt`
   and `e2e_rnnt` are Russian-only; use `--model-variant ml_ctc` /
   `ml_ctc_large` for multilingual (ru/en/kk/ky/uz) speech. Client-side
   "recognition language" settings have no effect on the server.

## Sharing logs safely

Error payloads the server sends to clients are already sanitized (generic
messages, no internal paths or model details). **Log files are not**: tracing
output can contain local filesystem paths (including your username via
`$HOME`), hostnames, and peer IP addresses. Recognized transcript text is never
logged, and error `code`s, version numbers, timings, and pool/limits
configuration are always safe to paste publicly. Before attaching a log to an
issue, skim it once for paths, hostnames, and IPs and redact those.

## See also

- [API reference](api.md) — the full WebSocket protocol (messages, error
  codes, close codes, session limits) these fixes refer to.
- [`asyncapi.yaml`](asyncapi.yaml) — the machine-readable WebSocket schema.
