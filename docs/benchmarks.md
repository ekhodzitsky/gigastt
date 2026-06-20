# Benchmarks

Honest, reproducible comparison of gigastt against current Russian-ASR engines.
Measured on an **Apple M1, CPU** execution provider (INT8 / greedy where applicable),
1000 samples per domain, failures counted as 100% WER, 95% bootstrap confidence
intervals. Competitor numbers come from the committed artifacts in
[`benchmark/results_full/`](../benchmark/results_full/); the **gigastt** rows are the
v2.3 default **`rnnt`** head, re-measured through the *same* Python harness, manifests,
and normalization as the competitors — so they are like-for-like. Methodology and
dataset prep are in [`benchmark/README.md`](../benchmark/README.md).

> **Contamination caveat.** GigaAM v3 (gigastt) is a SberDevices model whose training is
> dominated by Golos, and OpenSTT-style corpora are common in Russian ASR training mixes.
> The Golos / OpenSTT slices here **very likely overlap GigaAM v3's training
> distribution** — treat gigastt's in-domain numbers as a best-case upper bound, not WER
> on unseen data. (Golos ships an official train/test split, so this is distribution
> overlap, not row-level leakage.)

## Accuracy by domain — WER % (95% CI)

Domains: **Clean read** `golos_crowd_1k` · **Far-field** `golos_farfield` ·
**Phone** `openstt_calls` · **YouTube** `openstt_youtube`.

| Engine | Clean read | Far-field | Phone calls | YouTube |
|---|---|---|---|---|
| **gigastt** (GigaAM v3 `rnnt`, INT8) | 3.55 (2.9–4.2) | **4.08 (3.4–4.8)** | **18.50 (17.1–19.9)** | **10.91 (9.9–11.8)** |
| Vosk 0.54 (Zipformer2) | **2.97 (2.4–3.6)** | 6.29 (5.4–7.3) | 22.74 (21.3–24.2) | 17.24 (16.0–18.4) |
| Vosk 0.42 | 4.82 (4.0–5.6) | 13.93 (12.5–15.5) | 38.57 (36.7–40.6) | 20.65 (19.4–22.0) |
| T-one (beam+LM) | 6.61 (5.4–7.9) | 14.62 (12.5–17.0) | 21.73 (20.0–23.7) | 23.23 (21.5–25.1) |
| T-one (greedy, no LM) | 7.85 (6.7–9.2) | 17.22 (15.0–19.6) | 22.37 (20.6–24.2) | 26.54 (24.7–28.5) |
| whisper.cpp (Large v3) | 15.26 (13.7–16.7) | 17.91 (16.3–19.6) | 32.73 (30.7–34.9) | 22.61 (21.0–24.2) |
| faster-whisper (Large v3) | 15.53 (13.9–17.1) | 17.34 (15.6–19.1) | 24.93 (23.3–26.6) | 15.45 (14.2–16.6) |
| faster-whisper-turbo ¹ | 14.45 (11.5–18.0) | 18.30 (16.7–20.0) | 26.58 (24.9–28.2) | 15.45 (14.2–16.6) |

¹ turbo clean read is a 300-sample slice (wider CI); the rest are 1000.

> The pre-v2.3 default was the `e2e_rnnt` head (clean read 8.60%, far-field 5.90,
> phone 19.28, YouTube 11.35); the `rnnt` head above more than halves clean-read WER
> and edges the others. Both heads share the encoder — `rnnt` emits bare lowercase
> text (pair with `--punctuation` / `--itn` for readable output), `e2e_rnnt` bakes in
> punctuation/casing. WER is identical whether `rnnt` is run with `--itn` or not: the
> harness normalizes number-words ↔ digits symmetrically on every engine, so word vs
> digit output is neither rewarded nor penalized.

**Honest reading:**

- **Clean read** → a **statistical tie**: gigastt-rnnt (3.55%) vs **Vosk 0.54 (2.97%)** —
  the CIs overlap (2.9–4.2 vs 2.4–3.6) and Vosk's point estimate is slightly ahead.
  (The old `e2e` head trailed badly here at 8.60%.)
- **Far-field** → **gigastt wins** (4.08 vs Vosk 0.54 6.29) — CI-separated.
- **Phone calls** → **gigastt wins** (18.50): beats Vosk 0.54 (22.74) and even T-one's
  production beam+LM (21.73). Note the contamination caveat — and that T-one's
  *published* telephony strength is on its own call-center set, not this one.
- **YouTube** → **gigastt wins** (10.91 vs all; next best faster-whisper 15.45).

So gigastt-rnnt is **the most accurate engine on three of four domains** (far-field,
phone, YouTube — CI-separated) and **statistically ties the best (Vosk 0.54) on clean
read**. It is not a runaway leader on clean read — Vosk's point estimate still edges it —
but the head switch turned the old "concedes clean read" story into a near-tie. The
durable advantage remains the packaging — see Footprint and the
[README](../README.md#where-it-fits).

## Speed — RTF (processing ÷ audio; lower = faster; M1 CPU)

| Engine | Clean | Far-field | Phone | YouTube |
|---|---|---|---|---|
| Vosk 0.42 / 0.54 | ~0.03 | ~0.03 | ~0.03 | ~0.04 |
| **T-one (beam+LM)** | 0.056 | 0.060 | 0.065 | 0.065 |
| gigastt (`rnnt`, INT8) | 0.103 | 0.095 | 0.096 | 0.097 |
| whisper.cpp | 0.357 | 0.556 | 0.624 | 0.765 |
| faster-whisper / turbo | >1.0 (slower than real-time on CPU) | | | |

The CTC/transducer engines (Vosk, T-one, gigastt) are all comfortably real-time;
the Whisper engines are **slower than real-time** on CPU. gigastt is real-time but not
the fastest — Vosk and T-one are quicker. (The `rnnt` head's RTF above is slightly
better than the old `e2e` head's ~0.157, since the char-vocab joiner is cheaper than
the 1025-token BPE one.)

## Footprint

| Engine | Deployable model on disk | Peak RSS (cold) | Cold-start |
|---|---|---|---|
| **gigastt** | **~225 MB** (INT8) | 790 MB ¹ | **0.94 s** |
| T-one (greedy) | 138 MB | 672 MB | 1.87 s |
| T-one (beam+LM) | 138 MB + 5.5 GB KenLM | — | — |
| Vosk 0.54 | 966 MB | **560 MB** | 1.16 s |
| Vosk 0.42 | 3.5 GB | 1100 MB | 29.8 s |
| faster-whisper-turbo | 1.6 GB | 2154 MB | 6.8 s |
| faster-whisper (Large v3) | 2.9 GB | 2619 MB | 8.2 s |

¹ gigastt RSS is at the v2.3 default `--pool-size 2` (2 model copies, INT8 `rnnt`);
a single session is roughly half (~400 MB). The pre-v2.3 default was `--pool-size 4`
(~1502 MB); v2.3 lowered it to 2 plus a RAM-aware auto-cap.

gigastt wins **on-disk size** (4–13× smaller than the Whisper/Vosk engines) and
**cold-start** (0.94 s; Vosk 0.42 is a dreadful ~30 s). It is honestly **not** the
absolute smallest — T-one greedy is 138 MB — but T-one's *production* config adds a
5.5 GB KenLM, so gigastt is the smallest model **with no language-model trade-off**.
gigastt does **not** win peak RAM at the default pool size; Vosk 0.54 and T-one are
leaner (single-session gigastt ~400 MB is competitive).

## Streaming latency

Only gigastt exposes genuine incremental WebSocket streaming. Measured on `golos_00.wav`
(4 s, fed in real time, timer from connect): **TTFP ~782 ms (CPU) / ~693 ms (CoreML)**.
This is *buffered/chunked over an offline RNN-T*, not a natively streaming acoustic model
— the win is "true incremental partials from a single embedded binary" vs Whisper's
no-streaming, **not** a sub-second-latency claim. Vosk-server and T-one (300 ms chunks)
are also genuine streaming designs.

## Headline single-engine metrics

All gigastt numbers are the v2.3 default **`rnnt`** head (INT8), measured through the
cross-engine Python harness so they line up with the table above.

| Metric | Value |
|---|---|
| **WER — clean read** | **3.55%** (`golos_crowd_1k`, 992 samples, 95% CI 2.9–4.2%) |
| WER — other domains | far-field **4.08%** · phone **18.50%** · YouTube **10.91%** |
| Verbatim → normalized WER | clean 9.73→3.55 · far-field 4.69→4.08 · phone 19.39→18.50 · YouTube 12.19→10.91. The gap is number/filler formatting, normalized **symmetrically for every engine** (so it neither helps nor hurts gigastt relative to competitors). |
| RTF (`rnnt` INT8, M1 CPU) | ~0.10 |
| Peak RSS (default `--pool-size 2`) | 790 MB (single session ~400 MB) |
| INT8 encoder | 844 MB → 215 MB (**3.9×**), ~0% WER degradation |

## Reproduce

```sh
cd benchmark
pip install -r requirements.lock.txt
python benchmark.py --runners gigastt --dataset golos_crowd_1k --max-samples 0 --no-cache
```

New competitor runners (Vosk 0.54, faster-whisper-turbo, T-one) live under
[`benchmark/runners/`](../benchmark/runners/); each gracefully skips if its optional
dependency/model is absent. T-one beam+LM needs the 5.5 GB KenLM (`BENCHMARK_TONE_KENLM`).
