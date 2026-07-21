# CLI Reference

> **Recipes:** the [GigaSTT Workbook](https://ekhodzitsky.github.io/gigastt/) holds scenario-driven guides (EN + RU); this document stays the canonical CLI reference.

Complete command-line interface for `gigastt`.

All flags have corresponding environment variables (see individual options below).

```
gigastt [OPTIONS] <COMMAND>

Options:
  --log-level <LEVEL>    Log level [default: info]
  --offline              Air-gapped mode (env: GIGASTT_OFFLINE=1): refuse every
                         network fetch — model download, punctuation/VAD
                         auto-fetch — with an error naming the missing file
                         instead of a connect timeout

Commands:
  serve        Start STT server
  download     Download model (~850 MB) and auto-generate INT8 encoder
  transcribe   Transcribe audio file (offline)
  transcribe-batch  Transcribe every audio file in a directory (offline)
  watch        Watch a directory and transcribe new/changed audio files
  quantize     Quantize encoder to INT8 (always available since v0.9.0)

gigastt serve [OPTIONS]
  --port <PORT>             Listen port [default: 9876]
  --host <HOST>             Bind address [default: 127.0.0.1]
  --model-dir <DIR>         Model directory [default: ~/.gigastt/models]
  --model-variant <V>       Recognition head: rnnt | e2e_rnnt | ml_ctc | ml_ctc_large.
                            Omit to use the model already installed; fresh installs
                            default to rnnt (lower WER, no punctuation). e2e_rnnt keeps
                            punctuation/casing/ITN. ml_ctc / ml_ctc_large are the GigaAM
                            Multilingual charwise-CTC heads (220M / 600M encoder,
                            ru/en/kk/ky/uz, bare lowercase). Env: GIGASTT_MODEL_VARIANT.
  --punctuation <MODE>      Restore punctuation/casing on output: auto | on | off
                            [default: auto = on for rnnt, off for e2e_rnnt].
                            Optional ONNX pass; absent model → text unchanged.
                            Env: GIGASTT_PUNCTUATION.
  --punct-model-dir <DIR>   Punctuation model directory [default: ~/.gigastt/models/punct].
                            Auto-downloaded from ekhodzitsky/rupunct-small-onnx when
                            the pass is enabled and the files are absent.
                            Env: GIGASTT_PUNCT_MODEL_DIR.
  --itn <MODE>              Inverse text normalization (number-words → digits):
                            auto | on | off [default: auto = on for rnnt, off for
                            e2e_rnnt]. Runs before punctuation. Env: GIGASTT_ITN.
  --pool-size <N>           Concurrent inference sessions [default: 2]
  --pool-checkout-timeout-secs <S>  Seconds a handler waits for a free session triplet
                            before returning 503 [default: 30].
                            Env: GIGASTT_POOL_CHECKOUT_TIMEOUT_SECS.
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
  --model-variant <V>    Head to download: rnnt (default) | e2e_rnnt | ml_ctc | ml_ctc_large.
                         Env: GIGASTT_MODEL_VARIANT.
  --skip-diarization     Skip downloading the speaker diarization model
  --skip-quantize        Skip auto-quantization after download (FP32 only)
  --prequantized         Fetch the pre-quantized INT8 bundle from the pinned
                         GitHub Release (no FP32 download, no on-device quantize)
  --progress <FORMAT>    Progress output: human (default) | json.
                         Env: GIGASTT_DOWNLOAD_PROGRESS.

  Machine-readable progress (--progress=json)
    stdout carries one NDJSON event per line and nothing else (the human
    `\r`-progress renderer is disabled and tracing logs go to stderr), so a
    sidecar can drive an exact progress bar:

      {"phase":"download","file":"v3_rnnt_encoder.onnx","bytes_done":N,"bytes_total":M}
      {"phase":"verify","file":"v3_rnnt_encoder.onnx"}
      {"phase":"quantize","file":"v3_rnnt_encoder.onnx"}
      {"phase":"done","model_dir":"/home/u/.gigastt/models"}
      {"phase":"error","kind":"network|disk|checksum|interrupted|other","message":"..."}

    download events fire on the first chunk, then at most once per ~200 ms per
    file, and always once at 100% (bytes_total is 0 when the server does not
    send a length). verify fires per SHA-256 check; quantize marks the start
    of the ~2-minute on-device INT8 pass. done is emitted once, last, on
    success; error is emitted right before a non-zero exit.

  Exit codes (sysexits-flavored; 2 is deliberately unused — clap exits 2 on
  argument/usage errors before any NDJSON event can be emitted, so a code-2
  exit always means a misconfigured invocation, never a download failure)
    0    success
    1    other error
    65   checksum mismatch (corrupt or tampered download)
    69   network error (unreachable host, broken stream, HTTP error status)
    74   disk error (cannot create/write/rename model files)
    130  interrupted (Ctrl-C / SIGINT)

gigastt transcribe [OPTIONS] <FILE>
  --model-dir <DIR>           Model directory [default: ~/.gigastt/models]
  --model-variant <V>         Recognition head: rnnt | e2e_rnnt | ml_ctc | ml_ctc_large.
                              Omit to auto-detect. Env: GIGASTT_MODEL_VARIANT.
  --punctuation <MODE>        Restore punctuation/casing: auto | on | off
                              [default: auto = on for rnnt, off for e2e_rnnt].
                              Env: GIGASTT_PUNCTUATION.
  --punct-model-dir <DIR>     Punctuation model directory [default: ~/.gigastt/models/punct].
                              Auto-downloaded from ekhodzitsky/rupunct-small-onnx when
                              enabled and absent. Env: GIGASTT_PUNCT_MODEL_DIR.
  --itn <MODE>                Inverse text normalization (number-words → digits):
                              auto | on | off [default: auto = on for rnnt, off for
                              e2e_rnnt]. Runs before punctuation. Env: GIGASTT_ITN.
  -f, --format <FORMAT>       Export format: json, txt, srt, vtt, md [default: txt]
  -o, --output <FILE>         Write rendered output to file instead of stdout
  --max-chars-per-line <N>    Max chars per subtitle line (SRT/VTT) [default: 80]
  --max-words-per-line <N>    Max words per subtitle line (SRT/VTT) [default: 14]
  --word-timestamps           Include per-word timestamps in Markdown output
  --codec <CODEC>             Decode a headerless raw stream instead of a container:
                              pcmu | pcma | g722 (aliases: ulaw, alaw). Requires
                              --sample-rate. Env: GIGASTT_CODEC.
  --sample-rate <HZ>          Sample rate of a raw --codec stream (8000 or 16000
                              for g722). Env: GIGASTT_SAMPLE_RATE.
  Supports: WAV (incl. G.711 A-law/μ-law and G.722 payloads), M4A, MP3, OGG,
            FLAC (mono or auto-mixed); raw .ulaw/.alaw/.g722 via --codec

  Examples:
    gigastt transcribe recording.wav
    gigastt transcribe recording.wav -f srt -o recording.srt
    gigastt transcribe recording.wav -f md --word-timestamps -o notes.md
    gigastt transcribe call.ulaw --codec pcmu --sample-rate 8000

gigastt transcribe-batch [OPTIONS] <INPUT_DIR> <OUTPUT_DIR>
  Recursively transcribe every audio file (WAV, MP3, M4A, OGG, FLAC) under
  INPUT_DIR, writing one `<stem>.<ext>` file per format into OUTPUT_DIR.
  Files are processed in parallel (--pool-size workers). Files already inside
  a --move-to directory are excluded from the scan.
  --model-dir <DIR>           Model directory [default: ~/.gigastt/models]
  --model-variant <V>         Recognition head: rnnt | e2e_rnnt | ml_ctc | ml_ctc_large.
                              Omit to auto-detect. Env: GIGASTT_MODEL_VARIANT.
  --punctuation <MODE>        Restore punctuation/casing: auto | on | off.
                              Env: GIGASTT_PUNCTUATION.
  --punct-model-dir <DIR>     Punctuation model directory. Env: GIGASTT_PUNCT_MODEL_DIR.
  --itn <MODE>                Inverse text normalization: auto | on | off. Env: GIGASTT_ITN.
  -f, --format <LIST>         Export formats, comma-separated: txt, json, md, srt, vtt
                              [default: txt,json]. Env: GIGASTT_FORMAT.
  --pool-size <N>             Concurrent transcription workers [default: 2]
  --retries <N>               Extra attempts per file after a failure [default: 0].
                              Env: GIGASTT_BATCH_RETRIES.
  --move-to <DIR>             Move each successfully transcribed source into DIR
                              (e.g. --move-to done/). Env: GIGASTT_BATCH_MOVE_TO.
  --delete-source             Delete each successfully transcribed source
                              (exclusive with --move-to). Env: GIGASTT_BATCH_DELETE_SOURCE.
  --max-chars-per-line <N>    Max chars per subtitle line (SRT/VTT) [default: 80]
  --max-words-per-line <N>    Max words per subtitle line (SRT/VTT) [default: 14]
  --word-timestamps           Include per-word timestamps in Markdown output
  Exit codes: 0 = all files done · 1 = at least one file failed · 130 = interrupted
  (Ctrl-C finishes in-flight files, skips the rest).

  Examples:
    gigastt transcribe-batch samples/ out/
    gigastt transcribe-batch samples/ out/ --format txt,json,srt --pool-size 4
    gigastt transcribe-batch inbox/ out/ --move-to inbox/done/

gigastt watch [OPTIONS] <INPUT_DIR> <OUTPUT_DIR>
  Poll INPUT_DIR and transcribe new/changed audio files as they appear. A file
  is scheduled only after its size+mtime is unchanged for --settle-polls
  consecutive polls, so partially-copied files are never picked up. Files
  already present at startup are registered but NOT transcribed (use
  transcribe-batch for the backlog). Ctrl-C stops polling and waits for
  in-flight files before exiting.
  Same options as transcribe-batch, plus:
  --poll-interval-ms <MS>     Poll interval [default: 1000].
                              Env: GIGASTT_WATCH_POLL_INTERVAL_MS.
  --settle-polls <N>          Identical polls required before scheduling a file
                              [default: 2]. Env: GIGASTT_WATCH_SETTLE_POLLS.
  --retries <N>               Extra attempts per file after a failure [default: 2].
                              Env: GIGASTT_BATCH_RETRIES.

  Examples:
    gigastt watch inbox/ out/ --move-to inbox/done/
    gigastt watch inbox/ out/ --format txt --delete-source

gigastt quantize [OPTIONS]          # always available since v0.9.0
  --model-dir <DIR>      Model directory [default: ~/.gigastt/models]
  --force                Re-quantize even if INT8 model exists
```
