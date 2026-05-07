# CLI Reference

Complete command-line interface for `gigastt`.

All flags have corresponding environment variables (see individual options below).

```
gigastt [OPTIONS] <COMMAND>

Options:
  --log-level <LEVEL>    Log level [default: info]

Commands:
  serve        Start STT server
  download     Download model (~850 MB) and auto-generate INT8 encoder
  transcribe   Transcribe audio file (offline)
  quantize     Quantize encoder to INT8 (always available since v0.9.0)

gigastt serve [OPTIONS]
  --port <PORT>             Listen port [default: 9876]
  --host <HOST>             Bind address [default: 127.0.0.1]
  --model-dir <DIR>         Model directory [default: ~/.gigastt/models]
  --pool-size <N>           Concurrent inference sessions [default: 4]
  --bind-all                Required to listen on a non-loopback address.
                            Also: GIGASTT_ALLOW_BIND_ANY=1.
  --allow-origin <URL>      Additional Origin allowed (repeatable).
                            Loopback origins are always allowed.
  --cors-allow-any          Accept any cross-origin caller (wildcard CORS).
  --idle-timeout-secs <S>   WebSocket idle timeout [default: 300].
                            Env: GIGASTT_IDLE_TIMEOUT_SECS.
  --ws-frame-max-bytes <B>  Max WS frame size [default: 524288 = 512 KiB].
                            Env: GIGASTT_WS_FRAME_MAX_BYTES.
  --body-limit-bytes <B>    Max REST body size [default: 52428800 = 50 MiB].
                            Env: GIGASTT_BODY_LIMIT_BYTES.
  --rate-limit-per-minute <N>  Per-IP rate limit (requests/min). 0 = off (default).
                            Applies to /v1/* only; /health is exempt.
                            Env: GIGASTT_RATE_LIMIT_PER_MINUTE.
  --rate-limit-burst <N>    Token-bucket burst size [default: 10].
                            Env: GIGASTT_RATE_LIMIT_BURST.
  --metrics                 Expose Prometheus metrics at GET /metrics.
                            Off by default. Env: GIGASTT_METRICS.

  --max-session-secs <S>        Wall-clock session cap [default: 3600]. 0 = disabled.
                                Env: GIGASTT_MAX_SESSION_SECS.
  --shutdown-drain-secs <S>     Max wait for in-flight sessions on SIGTERM [default: 10].
                                Env: GIGASTT_SHUTDOWN_DRAIN_SECS.
  --skip-quantize               Skip auto-quantization step on first run.
                                Env: GIGASTT_SKIP_QUANTIZE.

gigastt download [OPTIONS]
  --model-dir <DIR>      Model directory [default: ~/.gigastt/models]
  --skip-diarization     Skip downloading the speaker diarization model
  --skip-quantize        Skip auto-quantization after download (FP32 only)

gigastt transcribe [OPTIONS] <FILE>
  --model-dir <DIR>      Model directory [default: ~/.gigastt/models]
  Supports: WAV, M4A, MP3, OGG, FLAC (mono or auto-mixed)

gigastt quantize [OPTIONS]          # always available since v0.9.0
  --model-dir <DIR>      Model directory [default: ~/.gigastt/models]
  --force                Re-quantize even if INT8 model exists
```
