# Long recordings

What happens to a long file on the existing file and jobs APIs: the upload
cap, how much RAM the server holds, and what survives a disconnect, a cancel,
or a restart. This is not a new queue. Live WebSocket truncation is a
different path; see the [runbook](runbook.md#pool-exhaustion--backpressure).

## Measured case

Host: AMD Ryzen AI 9 HX 370, 24 logical CPUs, 76 GiB RAM. Release binary
`gigastt 2.21.0` built from `06a367a`, `--offline serve --host 127.0.0.1
--port 9894 --pool-size 1 --enable-jobs --punctuation off --itn off
--shutdown-drain-secs 10`. Model: installed rnnt INT8, CPU execution provider,
existing optimized-graph cache. Server memory is `VmRSS` / `VmHWM` from
`/proc/<pid>/status`, sampled every 0.25 s. The curl process is not included.

Input: 20 minutes of 16 kHz mono PCM16 silence, `long20.wav`, 38,400,044
bytes (36.6 MiB). Silence measures cost and recovery, not word error.

| Call | Result |
|---|---|
| `POST /v1/transcribe` | HTTP 200 in 32.8 s wall. `duration` 1200.0, empty text. RTF 0.027 on this host. |
| `POST /v1/jobs` | `202` with `queued`, then `done` at `processed_seconds` 1200 and `percent` 100. `GET /result` HTTP 200, same duration. |
| Events client killed after the first progress events | The job still reached `done`. Closing the SSE stream does not cancel work. |

Idle `VmRSS` after warmup: 279,112 KiB (273 MiB). While that one file was
transcribed synchronously, `VmRSS` peaked at 397,876 KiB (389 MiB), 116 MiB
above idle. The jobs run of the same file peaked at 428,368 KiB while the
upload was still buffered and the decode was in progress. After `done`,
`VmRSS` was 336,660 KiB: the upload bytes are dropped, and the process does
not return all the way to the idle figure (ORT arenas stay warm). A second
36.6 MiB submit on top of that warm process pushed `VmRSS` to 472,924 KiB.
That is two bodies overlapping a warm engine, not the cost of one file.

## Limits that already exist

| Knob | Default | What it bounds |
|---|---|---|
| `--body-limit-bytes` | 50 MiB | One REST or jobs upload |
| `--jobs-max-bytes` | 512 MiB | Store-held upload bytes; one newly admitted body can cross this threshold |
| `--jobs-max` | 100 | How many job records are kept, not their size |
| `--jobs-ttl-secs` | 3600 | How long a finished, failed, or cancelled record stays |
| `--max-audio-secs` | 0 | Duration. `0` means no duration cap |
| `--inference-timeout-secs` | 600 | Time since the last completed window, not file length |

A 16 kHz mono PCM16 WAV crosses 50 MiB at about 27 minutes. Opus and other
compressed uploads hit the same byte cap much later in the recording.
`--jobs-max` alone does not bound RAM: 100 jobs times 50 MiB is about 5 GiB.
The byte budget is what stops that. A finished, failed, or cancelled job
drops its upload. The transcript (or the provisional partial) stays until
the TTL.

`Content-Length` over the cap returns **413** before the handler runs, with
the plain text `Failed to buffer the request body: length limit exceeded`.
A chunked body with no `Content-Length` is cut off once the buffered amount
passes the cap; the client in this run saw a broken pipe. Either way the
file is not transcribed. Raising the cap is `--body-limit-bytes` together
with `--jobs-max-bytes` if jobs are on.

## Aggregate upload admission

The server admits at most **N active uploads**, where N is the number of
successfully loaded sessions in the batch pool at boot (the shared pool when
`--batch-pool-size` is zero). Admission happens before reading the body on
`/v1/transcribe`, `/v1/transcribe/stream`, `/v1/audio/transcriptions` (including
multipart streaming), and `/v1/jobs`. There is no queue of uploads waiting for
admission: excess requests receive **503 `upload_busy`**, with `Retry-After` and
`retry_after_ms`. Retry after that delay. Health, readiness, WebSocket and admin
requests do not consume upload permits.

Together with the per-request body limit B, this bounds the total admitted
encoded input to **N × B**: normally 2 × 50 MiB = 100 MiB. These are boot-time
limits; restart to resize admission after changing the model pool or body cap.
For file transcription, the permit stays with the input through pool waiting
and blocking work, including raw-codec/channel expansion and SSE production.
A timeout or disconnect cannot free admission while detached work still holds
its input. Malformed and interrupted uploads release their permits on cleanup.

Jobs transfer accounting to the existing `--jobs-max-bytes` threshold once
store insertion completes, so queued jobs do not occupy upload permits. The
store checks its current total before insertion: one body can cross the
threshold, giving a store-held bound of the threshold plus one body cap
(up to 562 MiB with defaults), separately from the 100 MiB ingress bound.
Job count limits and queue-full 429 responses still apply separately.
Cancelled or timed-out job workers may retain input after the store clears its
reference; budget those outstanding worker/pool-held inputs separately until
cooperative cancellation finishes. The jobs threshold is not a strict cap on
all live payload references.

This is **not a process RSS cap**. Multipart/parser buffering, allocator
capacity, raw-codec rewrapped audio, decoded PCM, model memory, native inference scratch, transcript
results and socket buffers need additional memory. No temporary files are
created. Slow clients occupy admission until their request ends; an operator
who needs an upload read deadline should configure it at the reverse proxy.
The inference watchdog starts after upload and pool checkout.

The Linux model-free test `test_upload_admission_bounds_server_memory_at_pool_saturation`
starts a separate mock server process, saturates its inference pool with a
WebSocket, and reports that server's `VmRSS`/`VmHWM` while rejecting concurrent
uploads. Its client allocations stay in the parent process; the measurement
validates admission behavior and is not a model inference RAM benchmark.

## What is not resumed

Jobs live in the process. Nothing is written for a later restart.

| Event | What remains | What the client must do |
|---|---|---|
| Upload killed before the response | No job id. RSS returned to the idle band (281,712 KiB after a rate-limited upload was killed at 2 s). | Send the file again. |
| Client disconnects after `202`, or drops `GET /events` | The job keeps running. | Poll `GET /v1/jobs/{id}`. |
| `DELETE /v1/jobs/{id}` while `processing` | HTTP 204. Status stays `cancelled`. The empty-silence partial was still readable (`text` empty, `is_final` false). `GET /result` is **409** `job_not_finished`. The audio is not kept. | Submit the file again if you need a transcript. The partial is not a resume cursor. |
| `SIGTERM` during `processing` | This process exited in 1.02 s (the 10 s drain was not exhausted). Queued jobs are cancelled; an in-flight run is aborted. A native encoder call already inside ONNX still has to return first. | After `/ready` is 200, the old id is **404** `job_not_found`. Submit again. |
| `SIGKILL` | Process gone in 0.01 s. | Same 404 after the new process is ready. |
| Process still starting | `/health` can be 200 with `"model":"loading"`. `/ready` and `/v1/jobs` return **503** `initializing`. | Wait for `/ready`, then submit. A 503 here is not a saved job. |

## What to do

Chunk the recording under `--body-limit-bytes`, or raise that limit and
`--jobs-max-bytes` together, and submit with `POST /v1/jobs`. Poll or listen
for `done`, then `GET /result`. On cancel, kill, or restart, send the whole
file again. Do not treat `partial` as recovered audio, and do not raise
`--inference-timeout-secs` to allow a longer file.

No durable queue is required for these cases. The server already refuses an
oversized body, drops the upload when a job finishes or is cancelled, and
keeps running a job after the events client leaves.

## Experimental jobs zero context

Enable with `serve --enable-jobs --jobs-file-zero-context` (or set
`GIGASTT_JOBS_FILE_ZERO_CONTEXT=true`) to try a limited mitigation for unwanted
RNN-T words near the ends of file windows. This is **off by default**. The
resolved engine must be CPU INT8 `rnnt`, with `--file-window-concurrency 1`,
VAD off and no configured hotwords. Incompatible startup and model reloads
fail rather than silently enabling another path. A job requesting
`channels=split`, diarization, VAD or nonempty hotwords fails explicitly.
Ordinary mono routing, including the existing channel downmix, is supported.
REST, synchronous file transcription and WebSocket recognition are unchanged.

The option appends two seconds of zero-valued PCM **only for inference** on
native 24-second windows. Shorter windows and single-pass windows of other
lengths are unchanged. File duration, progress, timestamps, overlap geometry
and stitching are based on the original input. A raw nonblank RNN-T token at
or beyond local frame 600 invalidates the entire padded result, even if it
would disappear during word formatting. In that case the server logs a warning
and runs the original window once with a fresh decoder; no synthetic suffix
is clipped or published. Other inference failures are not retried by this
mechanism. On cancellation during an opted-in full window, only preceding
completed window publications are retained, including if cancellation occurs
during the fallback. There is no unbounded retry.

The padded path increases encoder input from 24 to 26 seconds, needs an
additional bounded PCM buffer of about 1.6 MiB, and uses more encoder and
decoder work. A rejected padded result costs up to two inference runs for
the same physical window (26 seconds then 24 seconds of model input).
The duration and progress limits still apply to the original samples; the
no-progress watchdog must allow the additional inference work.

Local CPU experiments on a reporter's 8 kHz stereo recording through ordinary
mono downmix, 16 kHz mono affirmative/negation controls and an independently
selected continuous recording found improvements in provisional boundary text
without observed loss of the checked affirmative or negation words. The
evidence is limited to those recordings; this is not a general WER improvement,
an elimination of all false words, or a mitigation for INT8 phrase omissions
in a different model head. The frame guard prevents emitting labels in the
synthetic region, but it cannot recognize hallucinated words timed within
the physical audio. Leave the option off unless its tradeoff is acceptable.
