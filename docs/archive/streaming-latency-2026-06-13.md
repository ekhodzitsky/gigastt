# Streaming latency — measurement note (gigastt)

> **Historical.** Smoke note from 2026-06-13 on a single 4 s clip. The live
> protocol, 100-clip WER Δ, and TTFP p50/p95 now live in
> [`docs/benchmarks.md`](../benchmarks.md#streaming-measurement-protocol)
> (`STREAM_PROTOCOL_VERSION` 1.0). Do not quote the numbers below as current
> product figures.

**Date:** 2026-06-13 · **Scope:** smoke, single file (`golos_00.wav`, 4.0 s) · **Status:** harness
fixed, remeasured on CPU + CoreML, and a streaming-quality problem surfaced.

## TL;DR

1. The old `time_to_first_partial_ms: 4099` was a **harness artifact** (the client sent the whole
   clip before reading any reply). Fixed with a concurrent reader.
2. Honest TTFP (time from audio start to first partial): **~735 ms on CPU**, **~681 ms on
   CoreML** — sub-second but **NOT sub-200 ms**. It is dominated by *where the first word falls in
   the clip* + real-time pacing, not by engine compute.
3. Per-chunk engine **compute** latency (the number actually comparable to "incremental streaming
   latency"): **~100 ms CPU**, **~70 ms CoreML** (from server `encoder_inference` logs). Both are
   already sub-200 ms. CoreML is only **~1.4× faster on streaming chunks** (small tensors
   under-utilise the Neural Engine; the docs' ~3–5× is for batch on large files).
4. **CRITICAL — streaming recognition is broken on quality:** for the same clip, batch REST
   returns the full sentence `«60 000 тенге — сколько будет стоить?»` (7 words), but the WebSocket
   streaming path returns a single partial+final `«И»`. So the latency above is "time to the only
   word streaming bothered to emit", not a representative number. This dwarfs the latency question.

## What was wrong (the harness)

`benchmark_latency.py` used to send the *entire* clip first (real-time paced, ~clip length) and
only *then* start reading server messages, so `first_partial_at` could not be stamped before the
send finished → TTFP pinned to ~clip length (4099 ms on a 4.0 s clip).

## What changed (the harness)

- Reader runs in a concurrent `asyncio` task started before the first chunk; `started_at` is
  stamped at the first audio chunk (after ready + configure).
- New fields: `first_partial_after_audio_ms`, `audio_duration_ms`, `total_audio_sent_ms`, and
  `partial_response_lag_ms` (per-partial delay vs the most recently sent chunk — approximates
  compute lag, isolated from pacing/word-position).
- Back-compat keys `time_to_first_partial_ms` / `finalization_lag_ms` kept. No server/protocol/WAV
  changes — the harness bug was purely client-side.

## Measurement configuration

- Host: Apple M1 Pro (arm64, 10 cores), INT8 encoder, `pool_size=4`, release build.
- EPs: **CPU** (`cargo run --release -p gigastt -- serve`) and **CoreML / Neural Engine**
  (`--features coreml`; log confirms "Using CoreML execution provider", not a CPU fallback).
- Client: `benchmark_latency.py`, `chunk_ms=100`, real-time-paced send. Server warm (post-warmup).
- `results_latency.json` / `results_latency_coreml.json` are local, git-ignored artifacts.

## Results (4 runs each, ms)

| metric | CPU EP | CoreML EP |
|---|---|---|
| time_to_first_partial (TTFP, from audio start) | 720–741 (≈735) | 677–693 (≈681) |
| finalization_lag (from audio start) | 4082–4141 | 4041–4045 |
| total_audio_sent (real-time paced) | ~4040 | ~4042 |
| **per-chunk encoder compute** (server log `encoder_inference`) | mode ~100–101 (96–122) | mode ~70 (66–89) |
| partial_response_lag (harness, first partial) | n/a (pre-metric run) | 70–88 |
| **partials emitted per clip** | **1** | **1** |
| streaming final text | «И» | «И» |
| batch REST text (same file) | «60 000 тенге — сколько будет стоить?» | (same) |

## Interpretation (honest)

- **TTFP decomposition:** TTFP ≈ (time until the first word is spoken in the real-time stream) +
  (per-chunk compute). Batch timestamps put the first real word ("60") at ~0.52 s, so even a
  zero-compute engine could not beat ~0.5 s TTFP on this clip. Hence ~681–735 ms is mostly
  "word position + pacing", and the engine's marginal contribution is ~70–100 ms.
- **"sub-200 ms"** is TRUE for per-chunk *compute* (CPU ~100 ms, CoreML ~70 ms) but FALSE for
  TTFP-from-clip-start. README conflates the two (and also a third number: ~700 ms *batch* time on
  a 16 s file, which is RTF ≈ 0.044, not streaming latency).
- **CoreML on streaming = ~1.4×, not 3–5×:** 100 ms chunks are tiny tensors; ANE dispatch/copy
  overhead eats the speedup. The ~3–5× figure is batch-on-large-file.
- **Streaming quality is the real issue:** per-chunk windows with no cross-chunk encoder
  left-context collapse a 7-word sentence to a single hallucinated «И». Any "real-time streaming
  with sub-Xms latency" claim is moot until streaming actually transcribes — fixing that is real
  engineering (bounded left-context window), not a harness tweak. Base model e2e_rnnt is offline
  RNN-T.

## Competitor latency context (orientation only — verify before publishing)

Streaming TTFP / algorithmic latency, from general knowledge + repo README (NOT measured here):
sherpa-onnx zipformer-streaming ~150–320 ms (local); Vosk ~100–300 ms (local); NVIDIA
Parakeet/Canary ~80–480 ms (GPU, lookahead-dependent); cloud (Deepgram/Soniox/Yandex
SpeechKit/SaluteSpeech) ~100–400 ms + network; whisper.cpp / faster-whisper are offline (seconds,
not natively streaming). gigastt's measured strengths are local-only operation, small INT8
footprint (210 MB at the time of writing; the current prequantized INT8 bundle is ~225 MB), RTF on batch, and Russian accuracy (far-field lead; Vosk leads clean speech) —
streaming latency is not a demonstrated advantage, and streaming quality currently regresses.

## Caveats

- Smoke: ONE 4 s clip, 4 repeats per EP — not statistically representative.
- `partial_response_lag_ms` is an upper-bounded approximation, under-estimated when compute ≥
  `chunk_ms`; the server `encoder_inference` log is the authoritative per-chunk compute source.
- README / public positioning wording intentionally untouched here. The streaming-quality
  problem was fixed the same day in the sliding-context-window commit (see update below).

## Update (2026-06-13) — streaming-quality fix landed

The streaming-quality problem is fixed (commit `8daa9a9`, "fix(inference): decode streaming on a
sliding context window"): the streaming path now decodes on a **sliding
context window** — accumulate audio and re-run the encoder on the whole retained (≤5 s) window
each chunk (fresh decoder state), then slide + dedup on endpoint/cap. Verified by model-gated
integration tests (`crates/gigastt-core/tests/streaming_quality.rs`: streaming ≈ batch on
`golos_00`, plus a >5 s long-audio slide test). Re-measured on the same fixture (CPU EP, INT8,
release):

| metric | before fix (broken) | after fix |
|---|---|---|
| streaming text (golos_00) | «И» | «60 000 — сколько будет стоить?» (full phrase, across segments) |
| partials emitted / clip | 1 | 11 |
| TTFP (from audio start) | ~735 ms | ~782–792 ms |
| per-chunk encoder compute | flat ~100 ms (isolated) | **432 → 712 ms, grows with window** |

**The tradeoff (honest):** correctness is bought with compute. Re-decoding the growing window
every 100 ms chunk costs ~432–712 ms of encoder time on CPU — **~4–7× slower than real-time**
near the 5 s window cap. So on the CPU EP the streaming path now transcribes correctly but
**cannot keep up with a live real-time stream** (it falls behind / accumulates backpressure).
Mitigations (not yet done): CoreML/GPU EP, re-decoding less often than every chunk (e.g. every
~250–500 ms of new audio), and/or a smaller context window — the stride-decode change below
(commit `3c120a1`) addresses the middle one.

**Implication for public positioning:** streaming is now **accurate** and — after the
stride-decode change (below)
— **real-time on CPU**. "sub-200ms" remains unsupported for end-to-end TTFP (TTFP is dominated by
word position in the stream + compute, ~0.8 s on this clip).

## Update (2026-06-13) — streaming real-time perf (stride decode)

The stride-decode change (commit `3c120a1`, "perf(inference): stride streaming decodes to keep
real-time on CPU") removed the compute tradeoff above: the encoder no longer re-runs on
every ~100 ms chunk. A decode **stride** re-runs it only after ~0.8 s of new audio (or at the window
cap), then resets; `finish_stream` decodes the sub-stride remainder at end-of-stream (Stop/EOF) so
batching never drops trailing words (wired into the WS stop handler and the SSE flush). Re-measured
(CPU EP, INT8, release, `golos_00`, fast-feed + drain-to-close):

| metric | sliding window (re-decode every chunk) | stride |
|---|---|---|
| decode calls / clip | 27 | **5** |
| encoder time sum | 10.7 s | 1.94 s |
| **RTF (CPU)** | **2.68** | **0.49** |
| streaming text | «60 000» (truncated) | «60 000 тенге — сколько будет стоить?» (= batch) |

RTF 0.49 → real-time with ~2× headroom on CPU; worst case at the 5 s window cap ≈ 0.75. Quality is
guarded by the `streaming_quality` integration tests (streaming ≈ batch + a >5 s slide test).
