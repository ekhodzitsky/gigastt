# CLI and batch processing

Turn a folder of recordings into transcripts: one-off sweeps with
`transcribe-batch`, a continuously watched drop folder with `watch`, and an
asynchronous queue with the jobs API. Every recipe is copy-paste runnable and
ends with a check that it worked.

## Scenario

You have a directory of audio — call-center recordings, podcast episodes, an
archive of voice notes — and you want text files out the other side. Sometimes
it is a one-time archive conversion; sometimes new recordings keep arriving
and the pipeline must run unattended, retry failures, and never get stuck on
a half-copied file.

## Prerequisites

- gigastt installed and the model downloaded (`gigastt download`) — see
  [Getting started](01-getting-started.md).
- A folder of audio files: WAV, MP3, M4A, OGG, FLAC (subfolders are scanned
  recursively).
- Nothing else: `transcribe`, `transcribe-batch`, and `watch` are offline
  commands — no server, no network.

One thing to know before scripting: **every CLI invocation loads the model**
(~1–2 s warm). `transcribe-batch` amortizes that cost across the whole
folder, so prefer it over a shell `for` loop around single-file `transcribe`
calls.

## Recipe: one-off folder run — `transcribe-batch`

The workhorse. Point it at an input directory and an output directory:

```sh
gigastt transcribe-batch calls/ transcripts/
```

It scans `calls/` recursively, transcribes every supported audio file with
`--pool-size` workers (default 2), and writes `transcripts/<name>.txt` and
`transcripts/<name>.json` per input (default `--format txt,json`).

A production-shaped run — more formats, more workers, and a source policy:

```sh
gigastt transcribe-batch calls/ transcripts/ \
  --format txt,json,srt \
  --pool-size 4 \
  --retries 2 \
  --move-to calls/done/
```

- `--format` — comma-separated list of `txt,json,md,srt,vtt`; one output file
  per format per input.
- `--pool-size` — parallel workers; each costs ~0.4 GB RAM (INT8 encoder), so
  scale with memory, not just cores (see the throughput recipe below).
- `--retries` — extra attempts per file with a short backoff (200 ms, 400 ms,
  …). Default 0 for batch, 2 for watch.
- `--move-to` — move each *successfully* transcribed source into the given
  directory. Failed files are always left in place. The move-to directory is
  excluded from the scan, so putting it inside the input folder
  (`calls/done/`) is safe and is the recommended layout.
- `--delete-source` — alternative to `--move-to`: delete sources after
  success. Mutually exclusive with `--move-to`.

Reading the run report. Each file logs a line, and the run ends with a
summary:

```text
INFO gigastt::batch: done /calls/alpha.wav processed=1 failed=0
WARN gigastt::batch: failed /calls/broken.mp3 error=invalid audio: Unsupported audio format: ...
INFO gigastt: batch finished processed=12 failed=1 skipped=0
```

Exit codes (script on these, not on the log text):

| Code | Meaning |
|---|---|
| `0` | every file transcribed |
| `1` | at least one file failed after all retries |
| `130` | interrupted with Ctrl-C — in-flight files finish, the rest are skipped (`skipped=N` in the summary) |

An empty input folder is not an error: it logs `no audio files found` and
exits 0.

### Verify the result

```sh
gigastt transcribe-batch calls/ transcripts/ --move-to calls/done/
echo "exit code: $?"        # 0 = clean sweep, 1 = some files failed
ls transcripts/             # one <name>.txt + <name>.json per source
ls calls/done/              # sources that succeeded
ls calls/*.wav 2>/dev/null  # anything still here failed — check the WARN lines
```

## Recipe: a live folder — `watch`

`watch` polls a directory and transcribes files as they arrive:

```sh
gigastt watch inbox/ transcripts/ --format txt,json --move-to inbox/done/
```

How it differs from `transcribe-batch`:

- **Backlog is skipped.** Files already present at startup are registered but
  *not* transcribed (`watching /inbox backlog=3 poll_ms=1000`). Sweep the
  existing pile with `transcribe-batch` first (see the wrapper recipe), then
  let `watch` handle new arrivals.
- **Settle protection.** A file is scheduled only after its size + mtime are
  unchanged for `--settle-polls` consecutive polls (default 2) spaced
  `--poll-interval-ms` apart (default 1000 ms). A recording still being copied
  or written is never picked up half-finished. Slow network mounts → raise
  both.
- **Changes are picked up.** Overwriting a file resets the settle counter and
  the new version is transcribed (a change mid-transcription re-queues it
  after the current run finishes).
- **Failures are sticky.** A file that exhausts its retries (default 2 for
  watch) is marked failed and left alone until its content changes.
- **Graceful stop.** Ctrl-C stops scheduling new files, waits for the ones in
  flight, prints `watch stopped processed=N failed=M`, and exits 0 (1 if
  anything failed).

### Verify the result

```sh
# terminal 1
gigastt watch inbox/ transcripts/ --move-to inbox/done/

# terminal 2
cp ~/recordings/sample.wav inbox/
# wait settle-polls x poll-interval plus transcription time, then:
ls transcripts/sample.txt        # appeared
ls inbox/done/sample.wav         # source archived
# back in terminal 1: Ctrl-C → "watch stopped processed=1 failed=0"
```

## Recipe: inbox pipeline wrapper (shell)

The standard "drop folder" service: audio lands in `inbox/`, transcripts come
out, successes are archived to `done/`, failures are collected in `failed/`
and automatically retried on the next sweep. Save as `transcribe-inbox.sh`:

```bash
#!/usr/bin/env bash
# Usage: transcribe-inbox.sh [INBOX] [OUT]
set -uo pipefail

INBOX="${1:-inbox}"
OUT="${2:-transcripts}"
DONE="$INBOX/done"
FAILED="$INBOX/failed"
mkdir -p "$OUT" "$DONE" "$FAILED"

# Requeue previous failures for another attempt.
find "$FAILED" -maxdepth 1 -type f \
  \( -name '*.wav' -o -name '*.mp3' -o -name '*.m4a' -o -name '*.ogg' -o -name '*.flac' \) \
  -exec mv -n {} "$INBOX/" \;

gigastt transcribe-batch "$INBOX" "$OUT" --format txt,json --move-to "$DONE"
rc=$?

# Successes were moved to done/; whatever audio remains at the inbox top
# level failed all retries — collect it for inspection and future requeue.
if [ "$rc" -eq 1 ]; then
  find "$INBOX" -maxdepth 1 -type f \
    \( -name '*.wav' -o -name '*.mp3' -o -name '*.m4a' -o -name '*.ogg' -o -name '*.flac' \) \
    -exec mv -n {} "$FAILED/" \;
  echo "some files failed — collected in $FAILED" >&2
fi
exit "$rc"
```

Run it on a schedule. A cron entry is enough for most inboxes:

```cron
*/15 * * * * /usr/local/bin/transcribe-inbox.sh /srv/stt/inbox /srv/stt/transcripts >> /var/log/stt-batch.log 2>&1
```

For a systemd timer + service unit instead of cron, see
[Deployment & ops](06-deployment-ops.md).

**Watch + batch catch-up.** The two commands compose into an always-on
pipeline: `watch` handles trickling arrivals with low latency, a periodic
`transcribe-batch` drains the startup backlog and anything the watcher
marked failed. Both honor the same `--move-to` exclusion, so they do not
double-process the backlog. One caveat: a catch-up sweep started *while* the
watcher is live can pick up a file the watcher has just scheduled but not yet
moved — schedule sweeps for quiet hours, or accept that a file is
occasionally transcribed twice (its outputs are simply overwritten).

```sh
# once, and then periodically (quiet hours): drain backlog + retry failures
./transcribe-inbox.sh /srv/stt/inbox /srv/stt/transcripts

# always on: new arrivals
gigastt watch /srv/stt/inbox /srv/stt/transcripts \
  --format txt,json --move-to /srv/stt/inbox/done/
```

### Verify the result

```sh
chmod +x transcribe-inbox.sh
cp ~/recordings/*.wav inbox/ && printf 'junk' > inbox/broken.mp3
./transcribe-inbox.sh inbox transcripts; echo "exit: $?"   # 1 — broken.mp3 failed
ls transcripts/   # transcripts for the good files
ls inbox/done/    # good sources archived
ls inbox/failed/  # broken.mp3 collected here
./transcribe-inbox.sh inbox transcripts   # re-runs the failure, exits 1 again
```

## Recipe: queued pipeline — the jobs API

`watch` covers one machine with a shared folder. Switch to the **jobs API**
when producers are on other machines, when files are long enough that holding
a synchronous HTTP request is awkward, or when you need progress reporting
and cancellation. It is the same engine, fronted by an in-memory FIFO queue
inside `gigastt serve`.

Jobs are off by default. Enable them (and reserve inference slots so queued
work cannot starve WebSocket/REST streaming):

```sh
gigastt serve --enable-jobs --batch-pool-size 1
```

Submit → poll → fetch:

```sh
# submit (accepts the same query params as /v1/transcribe, e.g. ?format=srt)
curl -s -X POST http://127.0.0.1:9876/v1/jobs \
  --data-binary @episode.wav
# {"job_id":"019f858a-...","status":"queued","created_at":1784651881.9}

# poll status
curl -s http://127.0.0.1:9876/v1/jobs/019f858a-...
# {"job_id":"...","status":"processing","processed_seconds":12.5,"percent":42}

# fetch the result once status is "done"
curl -s http://127.0.0.1:9876/v1/jobs/019f858a-.../result
# {"text":"...","words":[...],"duration":3512.4}
```

`status` walks `queued` → `processing` → `done` | `failed` | `cancelled`.
Fetching `/result` before `done` returns `409 job_not_finished`. Other
endpoints: `DELETE /v1/jobs/{id}` cancels a queued/processing job (`204`),
and `GET /v1/jobs/{id}/events` streams SSE progress
(`data: {"type":"progress","percent":42,...}` then `done`/`failed`).

A minimal driver script:

```bash
#!/usr/bin/env bash
# submit-and-wait.sh AUDIO_FILE — submit a job and print its transcript.
set -euo pipefail
BASE="${GIGASTT_BASE:-http://127.0.0.1:9876}"

job=$(curl -sf -X POST "$BASE/v1/jobs" --data-binary "@$1" \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')
echo "job: $job" >&2

while true; do
  status=$(curl -sf "$BASE/v1/jobs/$job" \
           | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')
  case "$status" in
    done)               break ;;
    failed|cancelled)   echo "job $status" >&2; exit 1 ;;
  esac
  sleep 2
done

curl -sf "$BASE/v1/jobs/$job/result" | python3 -c 'import json,sys; print(json.load(sys.stdin)["text"])'
```

Queue behavior to plan around:

- `--jobs-retry` (default 3) — retries only *transient* failures: inference
  timeouts and worker panics. A file that cannot be decoded fails immediately,
  no retries.
- `--jobs-max` (default 100) — when the store is full, submit returns
  `429 queue_full` with `Retry-After`. Back off and resubmit.
- `--jobs-ttl-secs` (default 3600) — finished/failed/cancelled jobs are
  evicted after the TTL. **Fetch and persist results promptly** — the store
  is in-memory, so a server restart drops queued jobs and unfetched results.

### Verify the result

```sh
curl -s http://127.0.0.1:9876/ready     # {"status":"ready",...} before submitting
./submit-and-wait.sh episode.wav        # prints the transcript text
# disabled API check: without --enable-jobs every /v1/jobs call returns 404
```

## Recipe: choosing output formats

Five formats, one file per format per input. Pick by the consumer:

| Format | Use it for | Notes |
|---|---|---|
| `txt` | humans, grep, downstream text tools | transcript text only |
| `json` | machines | `{"text", "words": [{"word","start","end","confidence"}], "duration"}` — per-word timings and confidence |
| `srt` | video editors, YouTube upload | SubRip cues grouped from word timings |
| `vtt` | web players | WebVTT variant of the same cues |
| `md` | notes, archives | YAML frontmatter (`duration`, `language`, `speakers`) + transcript |

```sh
gigastt transcribe-batch episodes/ out/ --format txt,json        # default pair
gigastt transcribe recording.wav -f srt -o recording.srt         # single file
```

Subtitle shaping (SRT/VTT): `--max-chars-per-line` (default 80) and
`--max-words-per-line` (default 14) control cue grouping; `0` disables the
limit. Broadcast-style captions usually want shorter lines:

```sh
gigastt transcribe recording.wav -f vtt --max-chars-per-line 42 -o recording.vtt
```

Markdown extras: `--word-timestamps` appends a per-word table with timings
and confidence — handy for manual review, noisy for archives.

Scripting caveat for single-file `transcribe`: logs go to **stdout** at the
default `info` level, mixed with the transcript. Use `-o` to write the
transcript to a file, or silence the logs with the global flag placed before
the subcommand:

```sh
gigastt --log-level error transcribe recording.wav          # stdout = transcript only
```

Extracting text from a folder of JSON results:

```sh
jq -r '.text' transcripts/*.json
```

### Verify the result

```sh
gigastt transcribe recording.wav -f srt -o /tmp/check.srt
head -4 /tmp/check.srt
# 1
# 00:00:00,480 --> 00:00:02,160
# Привет, как дела?
jq -r '.duration' transcripts/episode.json    # JSON parses and has fields
```

## Recipe: unusual inputs — telephony WAV, Opus, raw streams

**G.711 / G.722 inside WAV — just works.** A-law/μ-law (8 kHz telephony
exports) and G.722 ADPCM (Asterisk/Cisco/Teams, format tags `0x0064`/`0x028F`)
are decoded automatically; the batch walker picks them up like any other
`.wav`.

**OGG/Opus and `.opus` (Telegram voice notes, browser MediaRecorder).** The
container is sniffed from content, so single-file transcription works as-is:

```sh
gigastt transcribe voice.opus
```

But the batch/watch walkers scan by extension (`wav,mp3,m4a,ogg,flac`) and
**do not pick up `.opus` files**. Rename them to `.ogg` before sweeping —
the content is already an OGG container, so a plain rename suffices:

```sh
for f in inbox/*.opus; do mv "$f" "${f%.opus}.ogg"; done
gigastt transcribe-batch inbox/ transcripts/
```

**Raw headerless streams** (RTP dumps, Asterisk Monitor raw) carry no
container to sniff — declare the codec and rate explicitly:

```sh
gigastt transcribe call.ulaw --codec pcmu --sample-rate 8000
gigastt transcribe call.alaw --codec pcma --sample-rate 8000
gigastt transcribe call.g722 --codec g722 --sample-rate 8000   # 16000 also accepted
```

`--codec` accepts `pcmu` (alias `ulaw`), `pcma` (alias `alaw`), `g722`, and
requires `--sample-rate`. Anything else — WebM, AMR, MP4 video, a corrupt
file — fails with `invalid audio: Unsupported audio format: ...` (REST:
`422 invalid_audio`).

### Verify the result

```sh
file recording.wav                 # confirms the container type
gigastt --log-level error transcribe call.ulaw --codec pcmu --sample-rate 8000
echo "exit: $?"                    # 0 = decoded and transcribed
gigastt transcribe call.ulaw --codec pcmu 2>&1 | head -2
# error: the following required arguments were not provided: --sample-rate
```

## Recipe: throughput and memory

The `rnnt` head on INT8 runs at **RTF ≈ 0.10** on an M1 CPU — one worker
processes an hour of audio in about 6 minutes. A 100-hour archive at
`--pool-size 4` finishes in roughly `100 h × 0.10 / 4 ≈ 2.5 h` of wall time.
Full measurements, other hardware, and WER numbers:
[docs/benchmarks.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/benchmarks.md).

Budget memory before raising `--pool-size`:

- Each worker loads its own encoder copy: **~0.4 GB resident** with the
  default INT8 encoder, **~1.7 GB** with FP32. Default pool of 2 ≈ 790 MB RSS.
- The engine refuses to let the pool eat more than half of total RAM: an
  oversized `--pool-size` is **clamped with a warning** at load, so check the
  log instead of assuming you got the parallelism you asked for.
- Stay on INT8 (the default after the first-run auto-quantization): the
  encoder shrinks 844 MB → 215 MB on disk with ~0% WER degradation, and FP32
  quadruples the per-worker memory cost for no batch-speed win.
- On CPU builds, `--encoder-intra-threads` defaults to logical CPUs divided
  across the pool — the right value for a dedicated batch box; tune only for
  shared machines.

### Verify the result

```sh
# clamp warning, if any, appears at load time:
gigastt transcribe-batch calls/ transcripts/ --pool-size 8 2>&1 | grep -i "pool" | head -3
# per-file throughput in the log: "transcribe complete audio_s=... wall_s=... rtf=0.129"
time gigastt transcribe-batch calls/ transcripts/ --pool-size 4 --move-to calls/done/
```

## Common pitfalls

- **Half-copied files.** `transcribe-batch` transcribes whatever is in the
  folder *now*, including a file still being copied — you get a decode failure
  or a truncated transcript. Producers should write to a temp name and
  `mv` into the inbox (rename is atomic on the same filesystem). `watch`
  protects itself with settle polls; batch relies on a quiet folder.
- **Accidental reprocessing.** Without `--move-to`/`--delete-source`, every
  rerun redoes the whole folder. Always set a source policy for repeated
  sweeps. Related: `--move-to` flattens subdirectories — `a/week1/call.wav`
  and `a/week2/call.wav` collide as one `done/call.wav` (and the run warns
  `duplicate output ... inputs with equal file stems overwrite each other`
  for the transcripts). Keep source filenames unique.
- **Expecting parallelism you did not get.** `--pool-size 16` on 8 GB RAM is
  silently clamped at load (warning logged). Check the startup log, and
  remember FP32 quadruples per-worker memory.
- **`invalid audio` / 422 on an unsupported container.** WebM, AMR, MP4 video,
  or a corrupt upload fails decoding. Convert first
  (`ffmpeg -i in.webm -ar 16000 -ac 1 out.wav`) or, for raw telephony streams,
  declare `--codec` + `--sample-rate`. A `.opus` file is decodable but
  invisible to batch/watch — rename to `.ogg`.
- **Watch "forgets" failures.** A file that failed all retries is not retried
  until its content changes, and restarting the watcher registers it as
  backlog (never processed). Fix or replace the file, or point
  `transcribe-batch` at it — that is what the catch-up sweep in the wrapper
  recipe is for.
- **Job results evaporate.** The jobs store is in-memory with a 1 h TTL on
  terminal jobs: a restart loses queued jobs, and unfetched results are
  evicted. Persist results client-side; treat `429 queue_full` as
  backpressure, not an error worth alerting on.

## Links

- [docs/cli.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/cli.md) —
  full flag reference for `transcribe`, `transcribe-batch`, `watch`
- [docs/api.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/api.md) —
  jobs API endpoints, query parameters, error codes
- [docs/benchmarks.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/benchmarks.md) —
  RTF, memory, and WER measurements
- [docs/troubleshooting.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/troubleshooting.md) —
  decode and format error catalog
- [Getting started](01-getting-started.md) — install and model download
- [Deployment & ops](06-deployment-ops.md) — systemd units and timers for
  always-on pipelines
