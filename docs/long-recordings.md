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
| `--jobs-max-bytes` | 512 MiB | Sum of upload bytes still queued or processing |
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
