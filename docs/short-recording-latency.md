# Short-note latency on a two-CPU pin

A report against gigastt 2.21.0 said four real Telegram notes took 1.00–1.47 s
locally and 0.45–0.78 s on a cloud service, on a two-vCPU Xeon VM. Punctuation
and inverse text normalization were wanted locally. The notes, their lengths,
and a distribution were not published. This page measures a stand-in, not that
VM.

## What was measured

Fifteen labelled Golos voice commands already in this repo
(`crates/gigastt/tests/fixtures/golos_*.wav`, references in
`manifest.json`). They are 2.04–5.94 s of Russian, closer to a short command
than to a meeting. Each clip was also encoded as:

- Telegram-style Ogg/Opus: 48 kHz mono, `libopus` voip, 32 kb/s
- Browser-style WebM/Opus: 48 kHz mono, 60 ms frames, live WebM (no container
  duration)

Bytes, SHA-256, and the source duration are in
[`benchmark/short_notes/corpus.json`](../benchmark/short_notes/corpus.json).
The encoded files are
[`benchmark/short_notes/encoded/`](../benchmark/short_notes/encoded/).
Regenerate them with `python3 benchmark/measure_short_notes.py` (that script
also repeats the timing run).

Every number below is an HTTP `POST /v1/transcribe` of the whole file against
a release binary built from commit `824bf8ae094c` (`gigastt --version` still
prints `gigastt 2.21.0`). Server: `taskset -c 0,1`, `--host 127.0.0.1
--port 9891 --pool-size 1 --model-variant rnnt`, INT8 encoder, CPU execution
provider, optimized-graph cache already on disk. `taskset` is what the process
sees as two CPUs (`available_parallelism`).

**This machine is an AMD Ryzen AI 9 HX 370 (24 logical CPUs), not the
reported Xeon VM.** Do not read these times as that host.

Date: 2026-09-24. Raw rows:
[`benchmark/short_notes/measurements.json`](../benchmark/short_notes/measurements.json).
Load average was 6.95 at the start of the run and 21.5 at the end, because
other builds were in progress. The default configuration (punctuation and ITN
on) ran first, while the per-clip real-time factor stayed inside 0.053–0.061.
The punctuation-off pass ran as the load rose; treat its wall-clock tail as
an upper bound.

## Startup, first request, warm requests

Default `rnnt` with `--punctuation auto --itn auto` (the punctuation model
was already installed):

| Phase | n | p50 | p95 |
|---|--:|--:|--:|
| Process start → `GET /ready` | 6 | 0.86 s | 1.01 s |
| First request after ready (`golos_00.wav`, 4.0 s) | 6 | 0.63 s | 0.66 s |
| Later requests, WAV, all 15 clips × 8 rounds | 120 | 0.25 s | 0.51 s |
| Same, Ogg/Opus | 120 | 0.25 s | 0.52 s |
| Same, WebM/Opus | 120 | 0.26 s | 0.62 s |

Real-time factor (wall time ÷ source duration), same warm samples:

| Encoding | RTF p50 | RTF p95 |
|---|--:|--:|
| WAV | 0.057 | 0.149 |
| Ogg/Opus | 0.058 | 0.152 |
| WebM/Opus | 0.058 | 0.151 |

The container does not move the cost. Median WAV time tracks duration, not a
fixed overhead:

| Clip | Duration | Median wall | Median RTF |
|---|--:|--:|--:|
| `golos_07.wav` | 2.04 s | 0.12 s | 0.060 |
| `golos_02.wav` | 3.06 s | 0.18 s | 0.058 |
| `golos_00.wav` | 4.00 s | 0.23 s | 0.058 |
| `golos_06.wav` | 5.00 s | 0.27 s | 0.054 |
| `golos_12.wav` | 5.94 s | 0.32 s | 0.053 |

The other ten clips sit on the same line (RTF 0.053–0.061). One logged 4 s
request spent 216 ms in the encoder and 4 ms in greedy decode. That is one
observation, not a percentile, and it matches the table: the encoder is the
request.

Ready time includes the built-in one-second silence warmup (about 80 ms of
that second on this run) and an optimized graph loaded from cache. It is a
process restart with the model pages already hot, not a first boot off a cold
disk.

## Punctuation and ITN off

Same binary, `--punctuation off --itn off`.

| Phase | n | p50 | p95 |
|---|--:|--:|--:|
| Process start → `GET /ready` | 6 | 0.38 s | 0.51 s |
| First 4.0 s request | 6 | 0.22 s | 0.26 s |
| Warm WAV wall | 120 | 0.26 s | 0.64 s |
| Warm WAV RTF | 120 | 0.060 | 0.147 |

Warm RTF does not improve. Startup is shorter because the punctuation model
is not loaded (one log line showed that load at about 55 ms; the half-second
gap between the two ready times is larger than that and includes whatever
else the busier second half of the run was doing). On the warm path the
post-processing is lost in the encoder time.

Corpus quality on the 15 WAV references (71 words after the benchmark
normalizer, which maps number words and digits onto each other):

| Configuration | Normalized WER | Verbatim WER |
|---|--:|--:|
| Punctuation and ITN on | 0% (0/71) | 8% (6/75) |
| Both off | 0% (0/71) | 0% (0/75) |

The verbatim gap is writing, not different words. `golos_00.wav` came back as
`60000 тенге, сколько будет стоить?` with post-processing and as
`шестьдесят тысяч тенге сколько будет стоить` without it. The reference is
the second string. Both are the same recognition.

## What to do with the report

Keep one server process. On this substitute, ready plus the first 4 s note is
about 1.5 s, which overlaps 1.00–1.47 s. A later note of the same length is
about 0.23 s, which is under the cloud range quoted above. If a deployment
starts gigastt per note, that startup is the bill, not the recognizer.

Do not turn punctuation or ITN off to chase that report. They are what the
comparison wanted, they do not lower warm real-time factor here, and the
acoustic transcript on this set is already identical to the reference.

Do not treat Ogg or WebM as more expensive than WAV for these lengths.

No code change follows from this run. The only measurement that can confirm
the Xeon numbers is a run on a two-vCPU Xeon, with the same split between
process start, first request, and a warm server. This page is not that run.
