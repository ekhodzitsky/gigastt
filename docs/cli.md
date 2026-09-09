# CLI Reference

> **Recipes:** the [GigaSTT Workbook](https://ekhodzitsky.github.io/gigastt/) holds scenario-driven guides (EN + RU); this document stays the canonical CLI reference.

Complete command-line interface for `gigastt`.

Most flags have a matching `GIGASTT_*` environment variable (noted on each
option). A few are CLI-only — notably `--pool-size`.

```
gigastt [OPTIONS] <COMMAND>

Options:
  --log-level <LEVEL>    Log level [default: info]
  --offline              Air-gapped mode (env: GIGASTT_OFFLINE=1): refuse every
                         network fetch — model download, punctuation /
                         diarization / VAD auto-fetch — with an error naming
                         the missing file instead of a connect timeout

Commands:
  serve        Start STT server
  download     Download lean INT8 model (~225 MB; default)
  transcribe   Transcribe audio file (offline)
  transcribe-batch  Transcribe every audio file in a directory (offline)
  watch        Watch a directory and transcribe new/changed audio files
  quantize     Quantize encoder to INT8 (always available since v0.9.0)
  cache-gc     Prune stale ORT optimized graphs and stale CoreML compiled-model
               caches; optional content-hash dedupe

gigastt serve [OPTIONS]
  --port <PORT>             Listen port [default: 9876]
  --host <HOST>             Bind address [default: 127.0.0.1]
  --model-dir <DIR>         Model directory [default: ~/.gigastt/models]
  --optimized-cache-dir <PATH>  ORT optimized-graph cache directory
                            [default: <model-dir>/optimized_cache]. Where the
                            CPU encoder writes/reads its *_optimized.ort graphs.
                            If the directory cannot be created or is not
                            writable, the server logs a warning and starts
                            without the cache (slower cold start, higher
                            per-session RAM) instead of failing.
                            Env: GIGASTT_OPTIMIZED_CACHE_DIR.
  --model-variant <V>       Recognition head: rnnt | e2e_rnnt | ml_ctc | ml_ctc_large.
                            Omit to use the model already installed; fresh installs
                            default to rnnt (lower WER, no punctuation). e2e_rnnt keeps
                            punctuation/casing/ITN. ml_ctc / ml_ctc_large are the GigaAM
                            Multilingual charwise-CTC heads (220M / 600M encoder,
                            ru/en/kk/ky/uz, bare lowercase). ml_ctc is a speed SKU
                            (~1.3× faster than rnnt, RTF 0.032 vs 0.043), not a
                            lean-RAM SKU (ready RSS ≈ rnnt).
                            Env: GIGASTT_MODEL_VARIANT.
  --punctuation <MODE>      Restore punctuation/casing on output: auto | on | off
                            [default: auto = on for rnnt, off for e2e_rnnt].
                            Optional ONNX pass; absent model → text unchanged.
                            When present, ready RSS tax ~+4…28 MiB; edge hosts may
                            leave off. Env: GIGASTT_PUNCTUATION.
  --punct-model-dir <DIR>   Punctuation model directory [default: ~/.gigastt/models/punct].
                            Auto-downloaded from ekhodzitsky/rupunct-small-onnx when
                            the pass is enabled and the files are absent.
                            Env: GIGASTT_PUNCT_MODEL_DIR.
  --itn <MODE>              Inverse text normalization (number-words → digits):
                            auto | on | off [default: auto = on for rnnt, off for
                            e2e_rnnt]. Runs before punctuation. Env: GIGASTT_ITN.
  --hotwords-file <FILE>    Contextual hotword biasing: file of phrases to boost
                            (one per line, optional `\t<weight>` suffix). Off when
                            unset. Env: GIGASTT_HOTWORDS_FILE.
  --hotwords-default        Also bias the built-in Russian brand/acronym lexicon.
                            Env: GIGASTT_HOTWORDS_DEFAULT.
  --hotwords-boost <N>      Additive logit boost for hotword continuation tokens
                            [default: 5.0]. Env: GIGASTT_HOTWORDS_BOOST.
  --vad                     Voice activity detection: skip silence before decoding
                            and finalize streaming segments on trailing silence.
                            Recommended for pause-rich long files (meetings/podcasts);
                            RTF up to ~×2.6 on silence-rich audio. Downloads the
                            Silero VAD model on first use. Env: GIGASTT_VAD.
  --vad-threshold <N>       VAD speech-probability threshold in [0,1]
                            [default: 0.5]. No effect unless --vad.
                            Env: GIGASTT_VAD_THRESHOLD.
  --vad-min-silence-ms <N>  Minimum trailing silence (ms) to close a speech region
                            [default: 500]. No effect unless --vad.
                            Env: GIGASTT_VAD_MIN_SILENCE_MS.
  --vad-model-dir <DIR>     Silero VAD model directory [default: ~/.gigastt/models/vad].
                            Env: GIGASTT_VAD_MODEL_DIR.
  --endpoint-mode <MODE>    WS utterance end: auto|assistant|manual [default: auto].
                            Env: GIGASTT_ENDPOINT_MODE. Window cap never emits final.
  --stream-max-window-secs <N>  Max retained streaming encoder window (seconds)
                            [default: 2.5], clamped to 2.4–30. Longer windows
                            (e.g. 7.5) improve WER on phrases longer than the
                            window, at a linear per-stride encoder-cost increase.
                            Env: GIGASTT_STREAM_MAX_WINDOW_SECS.
  --stream-stable-prefix    Commit only hypothesis-stable prefixes at the window
                            cap (two consecutive decodes must agree, edge words
                            wait): fewer lost/replaced words at slide boundaries
                            without widening the window [default: true].
                            Opt out with --stream-stable-prefix=false.
                            Env: GIGASTT_STREAM_STABLE_PREFIX.
  --profile <P>             Deploy profile: default | edge [default: default].
                            edge applies --pool-size 1 and --vad when those
                            flags are left at defaults. Env: GIGASTT_PROFILE.
  --pool-size <N>           Concurrent inference sessions [default: 2].
                            CLI-only (no matching env var). Edge / low-RAM:
                            `--pool-size 1` (~46 MB resident / ~277 MB `ps` RSS).
                            Default 2 is ~66 MB resident / ~510 MB `ps` RSS;
                            the 215 MB encoder is memory-mapped and shared
                            (~20 MB resident per extra slot). `ps` RSS overstates
                            because it counts the mapping. Pool > 1 can cost
                            ~10–20% single-job RTF (encoder threads split).
  --encoder-intra-threads <N>  Intra-op threads for the encoder session (CPU build
                            only). Unset: logical CPUs divided across the pool.
                            Avoid `1` on multi-core (~3× slower than auto); explicit
                            `1` is still allowed for debugging.
                            Env: GIGASTT_ENCODER_INTRA_THREADS.
  --file-window-concurrency <N>  Max pooled triplets one file transcription may
                            hold to decode overlapping 24 s windows in parallel
                            [default: 1]. `1` is serial. `2` with `--pool-size 2`
                            uses an idle extra slot (try-checkout, never waits).
                            File path only; WebSocket is unchanged.
                            Env: GIGASTT_FILE_WINDOW_CONCURRENCY.
  --pool-checkout-timeout-secs <S>  Seconds a handler waits for a free session triplet
                            before returning 503 + retry_after_ms [default: 30].
                            Longer = queue under saturation; shorter = fail-fast.
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
  --trust-proxy             Trust X-Forwarded-For / X-Real-IP for rate-limit IP
                            extraction (only when the direct peer is loopback,
                            RFC1918, IPv6 unique-local, or IPv6 link-local).
                            Env: GIGASTT_TRUST_PROXY.
  --metrics                 Expose Prometheus metrics at GET /metrics.
                            Off by default. Env: GIGASTT_METRICS.
  --metrics-listen <ADDR>   Bind address for the separate metrics listener
                            [default: 127.0.0.1:9090]. Only used with --metrics.
                            Env: GIGASTT_METRICS_LISTEN.

  --pool-min-size <N>           Minimum session triplets that must load for the server to
                                boot; degraded-pool boot floor, clamped to 1..=pool_size
                                [default: 1]. Env: GIGASTT_POOL_MIN_SIZE.
  --batch-pool-size <N>         Triplets reserved for batch REST file transcription, split
                                off from --pool-size (not additive — total loaded sessions
                                stay at --pool-size) so a long file job can't starve
                                WebSocket/SSE streaming. 0 disables the split [default: 0].
                                Env: GIGASTT_BATCH_POOL_SIZE.
  --enable-jobs                 Enable the asynchronous /v1/jobs API for long-file and
                                batch transcription (off by default; routes 404 when
                                disabled). Env: GIGASTT_ENABLE_JOBS.
  --jobs-ttl-secs <N>           TTL for finished/failed/cancelled jobs before eviction
                                [default: 3600]. Env: GIGASTT_JOBS_TTL_SECS.
  --max-audio-secs <N>          Reject audio longer than N seconds on every path; 0 leaves
                                the streaming file path unlimited [default: 0], with or
                                without --vad. Paths that still need the whole buffer
                                (diarization, channels=split, telephony) keep their own
                                1800 s ceiling regardless. Over the limit returns
                                413 audio_too_long. Env: GIGASTT_MAX_AUDIO_SECS.
  --jobs-max <N>                Max jobs kept in memory; POST /v1/jobs returns 429 when
                                full [default: 100]. Env: GIGASTT_JOBS_MAX.
  --jobs-max-bytes <N>          Max total bytes of buffered job uploads kept in memory;
                                bounds RAM independently of --jobs-max, a submission over
                                budget returns 429 [default: 536870912 = 512 MiB].
                                Env: GIGASTT_JOBS_MAX_BYTES.
  --jobs-retry <N>              Max retries for a job that panics (a deterministic
                                inference_timeout is not retried) [default: 3].
                                Env: GIGASTT_JOBS_RETRY.
  --inference-timeout-secs <N>  Per-request inference timeout; a run exceeding it returns
                                inference_timeout (REST 504 / WS close). 0 disables
                                [default: 600]. Env: GIGASTT_INFERENCE_TIMEOUT_SECS.
  --max-session-secs <S>        Wall-clock session cap [default: 3600]. 0 = disabled.
                                Env: GIGASTT_MAX_SESSION_SECS.
  --shutdown-drain-secs <S>     Max wait for in-flight sessions on SIGTERM [default: 10].
                                Env: GIGASTT_SHUTDOWN_DRAIN_SECS.
  --config <FILE>               Path to a TOML config file for runtime limits
                                (reloaded on SIGHUP).

gigastt download [OPTIONS]
  --model-dir <DIR>      Model directory [default: ~/.gigastt/models]
  --model-variant <V>    Head to download: rnnt (default) | e2e_rnnt | ml_ctc | ml_ctc_large.
                         Env: GIGASTT_MODEL_VARIANT.
  --skip-diarization     Skip downloading the speaker diarization model
  --progress <FORMAT>    Progress output: human (default) | json.
                         Env: GIGASTT_DOWNLOAD_PROGRESS.

  Always fetches the lean **INT8** bundle (~225 MB). There is no FP32 download
  path and no on-device quantize step for runtime.

  Machine-readable progress (--progress=json)
    stdout carries one NDJSON event per line and nothing else (the human
    `\r`-progress renderer is disabled and tracing logs go to stderr), so a
    sidecar can drive an exact progress bar:

      {"phase":"download","file":"v3_rnnt_encoder_int8.onnx","bytes_done":N,"bytes_total":M}
      {"phase":"verify","file":"v3_rnnt_encoder_int8.onnx"}
      {"phase":"done","model_dir":"/home/u/.gigastt/models"}
      {"phase":"error","kind":"network|disk|checksum|interrupted|other","message":"..."}

    download events fire on the first chunk, then at most once per ~200 ms per
    file, and always once at 100% (bytes_total is 0 when the server does not
    send a length). verify fires per SHA-256 check. There is **no** `quantize`
    phase on the product path (INT8 is pre-shipped). done is emitted once,
    last, on success; error is emitted right before a non-zero exit.

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
  --hotwords-file <FILE>      Hotword biasing: file of phrases to boost (one per
                              line, optional `\t<weight>`).
                              Env: GIGASTT_HOTWORDS_FILE.
  --hotwords-default          Also bias the built-in brand/acronym lexicon.
                              Env: GIGASTT_HOTWORDS_DEFAULT.
  --hotwords-boost <N>        Logit boost for hotword tokens [default: 5.0].
                              Env: GIGASTT_HOTWORDS_BOOST.
  --vad                       Voice activity detection: skip silence before decoding
                              (recommended for pause-rich long files; RTF up to ~×2.6
                              on silence-rich audio). Downloads the Silero VAD model
                              on first use. Env: GIGASTT_VAD.
  --vad-threshold <N>         VAD speech-probability threshold [default: 0.5].
                              Env: GIGASTT_VAD_THRESHOLD.
  --vad-min-silence-ms <N>    Minimum trailing silence (ms) to close a speech region
                              [default: 500]. Env: GIGASTT_VAD_MIN_SILENCE_MS.
  --vad-model-dir <DIR>       Silero VAD model directory [default: ~/.gigastt/models/vad].
                              Env: GIGASTT_VAD_MODEL_DIR.
  --encoder-intra-threads <N>  Intra-op threads for the encoder session (CPU build
                              only). Unset: logical CPUs (single triplet). Avoid
                              `1` on multi-core (~3× slower than auto); explicit
                              `1` still allowed for debugging.
                              Env: GIGASTT_ENCODER_INTRA_THREADS.
  --file-window-concurrency <N>  Max triplets this file decode may hold to run
                              overlapping 24 s windows in parallel [default: 1].
                              `2` loads two triplets and splits encoder threads.
                              Env: GIGASTT_FILE_WINDOW_CONCURRENCY.
  -f, --format <FORMAT>       Export format: json, txt, srt, vtt, md [default: txt].
                              Env: GIGASTT_FORMAT.
  -o, --output <FILE>         Write rendered output to file instead of stdout.
                              Env: GIGASTT_OUTPUT.
  --max-chars-per-line <N>    Max chars per subtitle line (SRT/VTT) [default: 80].
                              Env: GIGASTT_MAX_CHARS_PER_LINE.
  --max-words-per-line <N>    Max words per subtitle line (SRT/VTT) [default: 14].
                              Env: GIGASTT_MAX_WORDS_PER_LINE.
  --word-timestamps           Include per-word timestamps in Markdown output.
                              Env: GIGASTT_WORD_TIMESTAMPS.
  --stereo-speakers           Transcribe left/right channels as separate speakers
                              (speaker_0 / speaker_1); falls back to mono when the
                              input is not stereo. Env: GIGASTT_STEREO_SPEAKERS.
  --codec <CODEC>             Decode a headerless raw stream instead of a container:
                              pcmu | pcma | g722 (aliases: ulaw, alaw). Requires
                              --sample-rate. Env: GIGASTT_CODEC.
  --sample-rate <HZ>          Sample rate of a raw --codec stream (8000 or 16000
                              for g722). Env: GIGASTT_SAMPLE_RATE.
  Supports: WAV (incl. G.711 A-law/μ-law and G.722 payloads, GSM 06.10 / wav49, MS/IMA ADPCM, RF64), M4A, MP3,
            OGG/Vorbis, OGG/Opus (.opus), WebM/Opus, FLAC (mono or
            auto-mixed); raw .ulaw/.alaw/.g722 via --codec

  Examples:
    gigastt transcribe recording.wav
    gigastt transcribe recording.wav -f srt -o recording.srt
    gigastt transcribe recording.wav -f md --word-timestamps -o notes.md
    gigastt transcribe call.ulaw --codec pcmu --sample-rate 8000

gigastt transcribe-batch [OPTIONS] <INPUT_DIR> <OUTPUT_DIR>
  Recursively transcribe every audio file (WAV, MP3, M4A, OGG, FLAC, WebM) under
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
  --pool-size <N>             Concurrent transcription workers [default: 2].
                              CLI-only. Edge: `--pool-size 1` (~46 MB resident).
                              Default 2 ~66 MB resident; `ps` RSS is higher
                              (mapped encoder). Pool > 1 can cost ~10–20%
                              single-job RTF (thread split).
  --retries <N>               Extra attempts per file after a failure [default: 0].
                              Env: GIGASTT_BATCH_RETRIES.
  --move-to <DIR>             Move each successfully transcribed source into DIR
                              (e.g. --move-to done/). Env: GIGASTT_BATCH_MOVE_TO.
  --delete-source             Delete each successfully transcribed source
                              (exclusive with --move-to). Env: GIGASTT_BATCH_DELETE_SOURCE.
  --max-chars-per-line <N>    Max chars per subtitle line (SRT/VTT) [default: 80]
  --max-words-per-line <N>    Max words per subtitle line (SRT/VTT) [default: 14]
  --word-timestamps           Include per-word timestamps in Markdown output
  Also accepts the recognition flags of `transcribe`: --hotwords-file,
  --hotwords-default, --hotwords-boost, --vad, --vad-threshold,
  --vad-min-silence-ms, --vad-model-dir, --encoder-intra-threads.
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

gigastt quantize [OPTIONS]          # packaging only (needs local FP32 source)
  --model-dir <DIR>      Model directory holding the FP32 encoder
  --force                Re-quantize even if INT8 model exists

gigastt cache-gc [OPTIONS]
  --model-dir <DIR>      Model directory [default: ~/.gigastt/models]
  --optimized-cache-dir <PATH>  ORT optimized-graph cache directory to prune
                         [default: <model-dir>/optimized_cache]. Pass the same
                         path `serve` uses (e.g. /var/cache/gigastt under the
                         shipped systemd unit) so stale *_optimized.ort graphs
                         are pruned from the right place.
                         Env: GIGASTT_OPTIMIZED_CACHE_DIR.
  --dry-run              Report reclaimable files without deleting / hardlinking
  --dedupe               Also hardlink content-identical files (SHA-256 groups)

  Removes *_optimized.{ort,onnx} graphs (from optimized_cache/, or the
  --optimized-cache-dir directory when given) that no installed head can
  load, keeping the graph for the preferred encoder of every head whose
  weights are present in the directory (INT8 preferred). Also prunes stale
  CoreML compiled-model caches under coreml_cache/: keeps only the current
  ort-<minor>/ version dir, removes dirs left by other ONNX Runtime builds
  and legacy unversioned hash dirs.
  Safe on accuracy: leftovers are pure disk waste from FP32 runs, head
  switches, or ONNX Runtime upgrades.
```
