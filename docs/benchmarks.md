# Benchmarks

Honest, reproducible comparison of gigastt against current Russian-ASR engines.
Measured on an **Apple M1, CPU** execution provider (INT8 / greedy where applicable),
1000 samples per domain, failures counted as 100% WER, 95% bootstrap confidence
intervals. Numbers come from the committed artifacts in
[`benchmark/results_full/`](../benchmark/results_full/); methodology and dataset prep are
in [`benchmark/README.md`](../benchmark/README.md).

> **Contamination caveat.** GigaAM v3 (gigastt) is a SberDevices model whose training is
> dominated by Golos, and OpenSTT-style corpora are common in Russian ASR training mixes.
> The Golos / OpenSTT slices here **very likely overlap GigaAM v3's training
> distribution** — treat gigastt's in-domain numbers as a best-case upper bound, not WER
> on unseen data. (Golos ships an official train/test split, so this is distribution
> overlap, not row-level leakage.)

## Accuracy by domain — WER % (95% CI)

Domains: **Clean read** `golos_crowd_1k` · **Far-field** `golos_farfield` ·
**Phone** `openstt_calls` · **YouTube** `openstt_youtube`.

> **Default head changed in v2.3.** The cross-engine tables below were measured on
> the **`e2e_rnnt`** head (the pre-v2.3 default), under one identical pipeline for
> all engines. v2.3 makes the **`rnnt`** head the default, which scores a much lower
> **2.6% acoustic WER** on the full Golos crowd set (see *Headline metrics*). A
> like-for-like cross-engine rerun of the `rnnt` head across all four domains is
> pending, so the comparison rows below remain the `e2e_rnnt` measurement and should
> be read as such.

| Engine | Clean read | Far-field | Phone calls | YouTube |
|---|---|---|---|---|
| **gigastt** (GigaAM v3 `e2e_rnnt`, INT8) | 8.60 (7.5–9.7) | **5.90 (5.1–6.8)** | **19.28 (17.9–20.7)** | **11.35 (10.3–12.3)** |
| Vosk 0.54 (Zipformer2) | **2.97 (2.4–3.6)** | 6.29 (5.4–7.3) | 22.74 (21.3–24.2) | 17.24 (16.0–18.4) |
| Vosk 0.42 | 4.82 (4.0–5.6) | 13.93 (12.5–15.5) | 38.57 (36.7–40.6) | 20.65 (19.4–22.0) |
| T-one (beam+LM) | 6.61 (5.4–7.9) | 14.62 (12.5–17.0) | 21.73 (20.0–23.7) | 23.23 (21.5–25.1) |
| T-one (greedy, no LM) | 7.85 (6.7–9.2) | 17.22 (15.0–19.6) | 22.37 (20.6–24.2) | 26.54 (24.7–28.5) |
| whisper.cpp (Large v3) | 15.26 (13.7–16.7) | 17.91 (16.3–19.6) | 32.73 (30.7–34.9) | 22.61 (21.0–24.2) |
| faster-whisper (Large v3) | 15.53 (13.9–17.1) | 17.34 (15.6–19.1) | 24.93 (23.3–26.6) | 15.45 (14.2–16.6) |
| faster-whisper-turbo ¹ | 14.45 (11.5–18.0) | 18.30 (16.7–20.0) | 26.58 (24.9–28.2) | 15.45 (14.2–16.6) |

¹ turbo clean read is a 300-sample slice (wider CI); the rest are 1000.

**Honest reading** (of the `e2e_rnnt`-head table above; the v2.3 default `rnnt` head changes the clean-read line — see below):

- **Clean read** → on the `e2e` head **Vosk 0.54 wins** (2.97%); gigastt-e2e (8.60%) and T-one (6.61%) trail it. The v2.3 default `rnnt` head reaches **2.6%** on the full Golos crowd set, effectively closing this gap (a like-for-like golos_crowd_1k rerun is pending).
- **Far-field** → a **tie** between gigastt (5.90) and Vosk 0.54 (6.29) — CIs overlap.
  gigastt's old far-field "lead" was only against the outdated Vosk 0.42 (13.93).
- **Phone calls** → **gigastt holds** (19.28): it beats Vosk 0.54 (22.74) and ties/leads
  even T-one's production beam+LM (21.73). Note the contamination caveat — and that
  T-one's *published* telephony strength is on its own call-center set, not this one.
- **YouTube** → **gigastt's only CI-separated win** (11.35 vs all).

On the `e2e` head, gigastt was **not** a general WER leader: *wins YouTube, holds noisy
phone calls, ties far-field, concedes clean read to Vosk 0.54.* The v2.3 `rnnt` default
materially improves clean read (2.6% on the full set), and the durable advantage remains
the packaging — see Footprint and the [README](../README.md#where-it-fits).

## Speed — RTF (processing ÷ audio; lower = faster; M1 CPU)

| Engine | Clean | Far-field | Phone | YouTube |
|---|---|---|---|---|
| Vosk 0.42 / 0.54 | ~0.03 | ~0.03 | ~0.03 | ~0.04 |
| **T-one (beam+LM)** | 0.056 | 0.060 | 0.065 | 0.065 |
| gigastt (`e2e_rnnt`) | 0.157 | 0.164 | 0.212 | 0.158 |
| whisper.cpp | 0.357 | 0.556 | 0.624 | 0.765 |
| faster-whisper / turbo | >1.0 (slower than real-time on CPU) | | | |

The CTC/transducer engines (Vosk, T-one, gigastt) are all comfortably real-time;
the Whisper engines are **slower than real-time** on CPU. gigastt is real-time but not
the fastest — Vosk and T-one are quicker.

The gigastt row above is the `e2e_rnnt` head measured over HTTP (the cross-engine
methodology). The v2.3 `rnnt` head's INT8 RTF on the full Golos crowd set, measured
in-process (no HTTP transport overhead), is **0.109** — slightly faster, since the
encoder is shared and the char-vocab joiner is cheaper than the 1025-token BPE one.

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

| Metric | Value |
|---|---|
| **WER — `rnnt` head (v2.3 default)** | **2.6%** (full Golos crowd, 9 994 samples, 50 394 words, 95% CI 2.4–2.8%) |
| Verbatim/acoustic WER (`rnnt`) | 2.6% — **Δ 0.0** vs normalized: the WER is genuinely acoustic, not normalization-inflated (contrast `e2e`: naive 14.40% / ITN 8.60%, Δ −5.80) |
| WER — `e2e_rnnt` head (prior default) | 8.60% renorm flagship (golos_crowd_1k) · 11.37% raw full set |
| RTF (`rnnt` INT8, full pipeline, M-series CPU) | **0.109** (4 401 s of compute for 11.2 h of audio, in-process harness) |
| Peak RSS (default `--pool-size 2`) | 790 MB (single session ~400 MB) |
| INT8 encoder | 844 MB → 215 MB (**3.9×**), ~0% WER degradation |

## Reproduce

```sh
cd benchmark
pip install -r requirements.lock.txt
python benchmark.py --max-samples 100 --dataset golos_crowd
```

New competitor runners (Vosk 0.54, faster-whisper-turbo, T-one) live under
[`benchmark/runners/`](../benchmark/runners/); each gracefully skips if its optional
dependency/model is absent. T-one beam+LM needs the 5.5 GB KenLM (`BENCHMARK_TONE_KENLM`).
