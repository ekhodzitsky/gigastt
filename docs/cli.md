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
  --model-variant <V>       Recognition head: rnnt | e2e_rnnt. Omit to use the model
                            already installed; fresh installs default to rnnt (lower WER,
                            no punctuation). e2e_rnnt keeps punctuation/casing/ITN.
                            Env: GIGASTT_MODEL_VARIANT.
  --punctuation <MODE>      Restore punctuation/casing on output: auto | on | off
                            [default: auto = on for rnnt, off for e2e_rnnt].
                            Optional ONNX pass; absent model → text unchanged.
                            Env: GIGASTT_PUNCTUATION.
  --punct-model-dir <DIR>   Punctuation model directory [default: ~/.gigastt/models/punct].
                            Env: GIGASTT_PUNCT_MODEL_DIR.
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
  --metrics-listen <ADDR>   Bind address for the separate metrics listener
                            [default: 127.0.0.1:9090]. Only used with --metrics.
                            Env: GIGASTT_METRICS_LISTEN.

  --pool-min-size <N>           Minimum session triplets that must load for the server to
                                boot; degraded-pool boot floor, clamped to 1..=pool_size
                                [default: 1]. Env: GIGASTT_POOL_MIN_SIZE.
  --batch-pool-size <N>         Triplets reserved for batch REST file transcription, split
                                off from --pool-size so a long file job can't starve
                                WebSocket/SSE streaming. 0 disables the split [default: 0].
                                Env: GIGASTT_BATCH_POOL_SIZE.
  --inference-timeout-secs <N>  Per-request inference timeout; a run exceeding it returns
                                inference_timeout (REST 504 / WS close). 0 disables
                                [default: 600]. Env: GIGASTT_INFERENCE_TIMEOUT_SECS.
  --max-session-secs <S>        Wall-clock session cap [default: 3600]. 0 = disabled.
                                Env: GIGASTT_MAX_SESSION_SECS.
  --shutdown-drain-secs <S>     Max wait for in-flight sessions on SIGTERM [default: 10].
                                Env: GIGASTT_SHUTDOWN_DRAIN_SECS.
  --skip-quantize               Skip auto-quantization step on first run.
                                Env: GIGASTT_SKIP_QUANTIZE.

gigastt download [OPTIONS]
  --model-dir <DIR>      Model directory [default: ~/.gigastt/models]
  --model-variant <V>    Head to download: rnnt (default) | e2e_rnnt. Env: GIGASTT_MODEL_VARIANT.
  --skip-diarization     Skip downloading the speaker diarization model
  --skip-quantize        Skip auto-quantization after download (FP32 only)

gigastt transcribe [OPTIONS] <FILE>
  --model-dir <DIR>           Model directory [default: ~/.gigastt/models]
  -f, --format <FORMAT>       Export format: json, txt, srt, vtt, md [default: txt]
  -o, --output <FILE>         Write rendered output to file instead of stdout
  --max-chars-per-line <N>    Max chars per subtitle line (SRT/VTT) [default: 80]
  --max-words-per-line <N>    Max words per subtitle line (SRT/VTT) [default: 14]
  --word-timestamps           Include per-word timestamps in Markdown output
  Supports: WAV, M4A, MP3, OGG, FLAC (mono or auto-mixed)

  Examples:
    gigastt transcribe recording.wav
    gigastt transcribe recording.wav -f srt -o recording.srt
    gigastt transcribe recording.wav -f md --word-timestamps -o notes.md

gigastt quantize [OPTIONS]          # always available since v0.9.0
  --model-dir <DIR>      Model directory [default: ~/.gigastt/models]
  --force                Re-quantize even if INT8 model exists
```
