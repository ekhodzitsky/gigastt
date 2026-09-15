# Benchmarks

Honest, reproducible comparison of gigastt against current Russian-ASR engines.
Measured on an **Apple M1, CPU** execution provider (INT8 / greedy where applicable),
1000-sample manifests per domain (992 scored on `golos_crowd_1k` after dropping empty
references), failures counted as 100% WER, 95% bootstrap confidence
intervals. Competitor numbers come from the committed artifacts in
[`benchmark/results_full/`](../benchmark/results_full/); the **gigastt** rows are the
default **`rnnt`** head (since v2.3), re-measured through the *same* Python harness,
manifests, and normalization as the competitors — so they are like-for-like. Methodology
and dataset prep are in [`benchmark/README.md`](../benchmark/README.md).

> **Provenance (gigastt rows).** The committed `results_full/*_gigastt*.json` artifacts
> are the pre-v2.3 `e2e_rnnt` run (gigastt 2.0.13, 2026-06; normalized WER 8.60 / 5.90 /
> 19.28 / 11.35 across the four domains — the `e2e_rnnt` numbers quoted further down).
> The `rnnt` re-measurement behind the headline gigastt rows (3.55 / 4.08 / 18.50 /
> 10.91) is **not committed** to `results_full/`; the competitor rows are.

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
| gigastt (GigaAM Multilingual `ml_ctc_large`, 600M, INT8) | 4.44 (3.7–5.2) | 5.70 (4.9–6.6) | — ² | — ² |
| gigastt (GigaAM Multilingual `ml_ctc`, 220M, INT8) | 6.15 (5.4–7.0) | 8.28 (7.3–9.4) | — ² | — ² |
| Vosk 0.54 (Zipformer2) | **2.97 (2.4–3.6)** | 6.29 (5.4–7.3) | 22.74 (21.3–24.2) | 17.24 (16.0–18.4) |
| Vosk 0.42 | 4.82 (4.0–5.6) | 13.93 (12.5–15.5) | 38.57 (36.7–40.6) | 20.65 (19.4–22.0) |
| T-one (beam+LM) | 6.61 (5.4–7.9) | 14.62 (12.5–17.0) | 21.73 (20.0–23.7) | 23.23 (21.5–25.1) |
| T-one (greedy, no LM) | 7.85 (6.7–9.2) | 17.22 (15.0–19.6) | 22.37 (20.6–24.2) | 26.54 (24.7–28.5) |
| whisper.cpp (Large v3) | 15.26 (13.7–16.7) | 17.91 (16.3–19.6) | 32.73 (30.7–34.9) | 22.61 (21.0–24.2) |
| faster-whisper (Large v3) | 15.53 (13.9–17.1) | 17.34 (15.6–19.1) | 24.93 (23.3–26.6) | 15.45 (14.2–16.6) |
| faster-whisper-turbo ¹ | 14.45 (11.5–18.0) | 18.30 (16.7–20.0) | 26.58 (24.9–28.2) | 15.45 (14.2–16.6) |

¹ turbo clean read is a 300-sample slice (wider CI); the rest are 1000.

² the Multilingual CTC heads were measured only on the two Russian domains whose audio was
locally available (clean read `golos_crowd_1k`, far-field `golos_farfield`); the OpenSTT
phone / YouTube sets were not on hand. Their Kazakh / Kyrgyz / Uzbek accuracy is not
measured here (no reference set).

## Held-out / additional public sets — WER % (95% CI)

Same harness and machine (Apple M1, CPU, `rnnt` INT8). These are **not** the
Golos/OpenSTT slices above (still may overlap train mixes in general — see the
contamination caveat). Protocol:
[`specs/held-out-datasets-roadmap.md`](../specs/held-out-datasets-roadmap.md).
Prep commands and per-dataset notes:
[`benchmark/README.md` § Datasets](../benchmark/README.md#datasets).

### Comparison (lower is better)

| Dataset | Domain | n | **gigastt** | Vosk 0.54 | faster-whisper L3 |
|---|---|--:|--:|--:|--:|
| Common Voice RU (CV 21.0, seed=42) | crowd read | 1000 | **2.63 (2.2–3.2)** | 6.10 (5.4–6.9) | 5.22 (4.5–5.9) |
| FLEURS `ru_ru` test (full) | clean read | 775 | 5.26 (4.7–5.8) | 6.14 (5.6–6.8) | **3.84 (3.4–4.3)** |
| RuLS (OpenSLR 96 / HF mirror) | audiobook | 1000 | **4.21 (3.8–4.6)** | 9.18 (8.6–9.7) | 9.65 (9.0–10.2) |
| SOVA RuDevices | device / command | 1000 | 10.30 (9.4–11.2) | **6.28 (5.5–7.0)** | 14.79 (13.6–16.1) |
| Podlodka Speech (train, full) | podcast / conversational | 67 | **7.33 (5.6–9.2)** | 9.96 (8.0–12.0) | 7.27 (5.9–8.9) |
| ToneWebinars (val, seed=42) | webinar / lecture | 1000 | 13.02 (12.3–13.8) | 14.87 (14.2–15.6) | **8.33 (7.7–9.0)** |

RTF (M1 CPU): gigastt ~0.04–0.09 · Vosk ~0.04–0.05 · faster-whisper ~0.7–1.3.

**Takeaways**

- **vs Vosk 0.54:** gigastt wins on CV, FLEURS, RuLS, Podlodka, ToneWebinars;
  **loses on SOVA device/command** (10.30 vs 6.28) — Zipformer/command domain.
- **vs faster-whisper Large-v3:** gigastt ahead on **CV** (2.63 vs 5.22), **RuLS**
  (4.21 vs 9.65), and **SOVA** (10.30 vs 14.79); **FLEURS** and **ToneWebinars**
  Whisper leads (3.84 vs 5.26; 8.33 vs 13.02); **Podlodka** is a statistical tie
  (7.33 vs 7.27, wide CI). Domain-dependent — Whisper stronger on long lecture speech.
- Podlodka n=67 is thin (HF only ~87 utts total); CI is wide.
- ToneWebinars: validation slice of first 2500 RU rows, seed=42 → n=1000; ~7.1 h audio;
  mostly Russian webinar segments (Cyrillic majority filter).

### Provenance

| Dataset | License | Prep | Artifacts |
|---|---|---|---|
| Common Voice RU | CC0-1.0 | `scripts/prepare_common_voice_ru.py` (mirror `artyomboyko/common_voice_21_0_ru`) | `results_full/common_voice_ru_{gigastt,vosk054,baselines}.json` |
| FLEURS `ru_ru` | CC BY 4.0 | `scripts/prepare_fleurs.py --config ru_ru` | `results_full/fleurs_ru_{gigastt,vosk054,baselines}.json` |
| RuLS | Public Domain (USA) / LibriVox | HF `istupakov/russian_librispeech` test (seed=42) | `results_full/ruls_{gigastt,vosk054,faster_whisper}.json` |
| SOVA RuDevices | see HF card | HF `bond005/sova_rudevices` (seed=42, n=1000 of 5k+) | `results_full/sova_rudevices_{gigastt,vosk054,faster_whisper}.json` |
| Podlodka | see HF card | HF `bond005/podlodka_speech` train (n=67 = full train) | `results_full/podlodka_{gigastt,vosk054,faster_whisper}.json` |
| ToneWebinars | Apache-2.0 | `scripts/prepare_tone_webinars.py` (val, max-scan 2500, seed=42 → n=1000) | `results_full/tone_webinars_{gigastt,vosk054,faster_whisper}.json` |

Manifests under `benchmark/manifests/`. License notes:
[`benchmark/DATA_LICENSE`](../benchmark/DATA_LICENSE).

> The pre-v2.3 default was the `e2e_rnnt` head (clean read 8.60%, far-field 5.90,
> phone 19.28, YouTube 11.35); the `rnnt` head above more than halves clean-read WER
> and edges the others. Both heads share the encoder — `rnnt` emits bare lowercase
> text (pair with `--punctuation` / `--itn` for readable output), `e2e_rnnt` bakes in
> punctuation/casing. WER is identical whether `rnnt` is run with `--itn` or not: the
> harness normalizes number-words ↔ digits symmetrically on every engine, so word vs
> digit output is neither rewarded nor penalized.

> **Multilingual heads.** `ml_ctc` (220M) and `ml_ctc_large` (600M) are the opt-in GigaAM
> Multilingual charwise-CTC heads (ru/en/kk/ky/uz). On Russian they trade some accuracy for
> language coverage: the 600M head (4.44% clean / 5.70% far-field) approaches the
> Russian-specialized `rnnt` and comfortably beats the old `e2e_rnnt` (8.60% clean), while
> the 220M head (6.15% / 8.28%) is the smaller, faster option. Measured through the same
> harness, manifests, and normalization as the rows above; bare lowercase output, so pair
> with `--punctuation` / `--itn` for readable text.

### Punctuation quality — `e2e_rnnt` vs `rnnt` + RuPunct restore

The low-WER `rnnt` head is bare lowercase, so readable Russian comes two ways: bake it in
with the `e2e_rnnt` head (one pass), or restore it on top of `rnnt` with the `--punctuation`
RuPunct model plus `--itn` (two passes). Measured on **775 punctuated FLEURS-ru references**
(the `raw_transcription` field; numbers are written as digits, so both configs run `--itn on`
to match), position-based F1 with the same metric as
[`benchmark/benchmark_punctuation.py`](../benchmark/benchmark_punctuation.py):

| Config | Punctuation F1 | Capitalization F1 |
|---|---|---|
| `e2e_rnnt` (one pass, baked in) | **0.540** | **0.726** |
| `rnnt` + RuPunct restore (two passes) | 0.355 | 0.656 |

`e2e_rnnt` wins on both — and the gap is a **lower bound**: the metric is position-based, so
`e2e_rnnt`'s higher WER (more misrecognized words shift downstream positions) handicaps *its*
own score, yet it still leads. This is why both heads are kept: `rnnt` for lowest WER on raw
text, `e2e_rnnt` as the single-pass path to punctuated / cased / ITN'd Russian whose
punctuation is better than restoring it after the fact.

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
[README](../README.md#performance).

## English — WER % (LibriSpeech test-clean)

The **Multilingual CTC heads** (`ml_ctc` / `ml_ctc_large`) also transcribe English. Measured
on a 1000-sample seed-42 slice of **LibriSpeech `test-clean`** (read English, the standard
clean-English ASR benchmark; CC BY 4.0), verbatim WER — the Russian words-to-digits ITN /
anglicism normalization used in the Russian table does not apply to English (and here
normalized vs verbatim agree to within 0.2 pp anyway).

| Engine | WER % (95% CI) |
|---|---|
| **gigastt** (GigaAM Multilingual `ml_ctc_large`, 600M, INT8) | **4.63 (4.4–5.1)** |
| gigastt (GigaAM Multilingual `ml_ctc`, 220M, INT8) | 6.67 (6.4–7.3) |
| gigastt (GigaAM v3 `rnnt` / `e2e_rnnt`, Russian-only) | 100 |

The 600M head (4.63%) is only ~0.2 pp behind its own Russian clean-read WER (4.44%), so the
model card's "moderate on English" understates it on clean read; the 220M head is at 6.67%.
The Russian-specialized `rnnt` / `e2e_rnnt` heads have a **Cyrillic-only** vocabulary and
cannot produce English at all (100% WER), so the Multilingual heads are the only option for
English — and for Kazakh / Kyrgyz / Uzbek, which are not measured here for lack of a
reference set (Common Voice 16.1 was removed from the Hub).

> Same caveat as the Russian table: GigaAM Multilingual is pre-trained on 2M hours across
> 70+ languages and LibriSpeech is a common English ASR corpus, so read this as a best-case
> in-distribution upper bound, not WER on unseen English.

## Kazakh / Kyrgyz / Uzbek — WER % (FLEURS)

The Multilingual CTC heads' other three supported languages, on FLEURS test splits (read
speech, CC BY 4.0; Kazakh 856 · Kyrgyz 977 · Uzbek 862 utterances). WER is computed with a
Unicode-complete verbatim normalizer ([`scripts/wer_unicode.py`](../scripts/wer_unicode.py)) —
the Russian harness normalizer keeps only `[a-zа-я0-9]` and would strip the Turkic Cyrillic
letters (`ә ғ қ ң ө ұ ү һ і`) these languages need.

Two normalization confounds are removed so the number reflects recognition, not writing
convention: **(1)** the charwise-CTC heads spell numbers out while ~19% of FLEURS references
keep digits, and there is no reliable words↔digits ITN for these languages (`num2words` has
no support) — so the headline **digit-free** WER excludes number-bearing sentences; **(2)**
apostrophe variants are folded, so Uzbek `oʻ` / `gʻ` (U+02BB) and the model's `o'` / `g'`
(U+0027) compare equal. The **full** figure is the upper bound over all utterances.

| Head | Kazakh | Kyrgyz | Uzbek |
|---|---|---|---|
| **gigastt** (`ml_ctc_large`, 600M, INT8) | **6.52 (5.9–7.1)** | **7.39 (6.7–8.0)** | **9.21 (8.5–9.9)** |
| gigastt (`ml_ctc`, 220M, INT8) | 7.21 (6.6–7.9) | 8.82 (8.1–9.5) | 11.96 (11.1–12.8) |

*Full-set upper bounds (all utterances, incl. the digit-format mismatch): 600M — kk 11.35 /
ky 12.50 / uz 14.04; 220M — kk 12.14 / ky 13.88 / uz 17.06.*

*Provenance: the committed `results_full/fleurs_{kk,ky,uz}_gigastt_ml_ctc*.json` artifacts
were scored with the older normalizer and do not reproduce these numbers (e.g. uz 600M
reads 19.85 there vs 9.21 here); the digit-free / apostrophe-folded recompute via
[`scripts/wer_unicode.py`](../scripts/wer_unicode.py) is not committed.*

Across all five supported languages the 600M head lands at **4.4–9.2% clean-read WER**
(Russian 4.44 · English 4.63 · Kazakh 6.52 · Kyrgyz 7.39 · Uzbek 9.21) — a genuinely strong
multilingual result. Same caveat as above: FLEURS overlaps common multilingual ASR training
data, so read these as in-distribution upper bounds.

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

| Engine | Deployable model on disk | Peak RAM | Cold-start |
|---|---|---|---|
| **gigastt** | **~225 MB** (INT8) | **~510 MB RSS / ~66 MB resident** ¹ | **0.94 s** |
| T-one (greedy) | 138 MB | 672 MB | 1.87 s |
| T-one (beam+LM) | 138 MB + 5.5 GB KenLM | — | — |
| Vosk 0.54 | 966 MB | 560 MB | 1.16 s |
| Vosk 0.42 | 3.5 GB | 1100 MB | 29.8 s |
| faster-whisper-turbo | 1.6 GB | 2154 MB | 6.8 s |
| whisper.cpp (Large v3) | 2.9 GB | — | — |
| faster-whisper (Large v3) | 2.9 GB | 2619 MB | 8.2 s |

¹ gigastt memory is measured on Apple M1 Pro 16 GB (macOS), INT8 `rnnt`,
`--punctuation off --itn off`, steady state after 5 warm decodes. Two metrics,
because they now diverge: **resident footprint** (dirty + compressed pages,
`/usr/bin/footprint`) is ~46 MB at `--pool-size 1` and ~66 MB at the default
`--pool-size 2` (~35 / ~57 MB at `/ready`) — the honest "RAM you actually
need" figure, since the 215 MB model is memory-mapped and file-backed and the
OS reclaims those clean pages under pressure. **`ps` RSS** — what
`top`/Activity Monitor shows — reads ~277 MB (pool 1) / ~510 MB (pool 2)
because it counts the shared model mapping per mapping. The server's own
`memory_after_load rss_mb=` startup log samples before the mapping is touched
and reads only ~55 / ~83 MB — a known under-read, not a number to quote. The
committed `benchmark/results_footprint_gigastt.json` predates the memory-mapped
encoder: its cold-start (0.94 s) still matches the table, but its ~1501 MB peak
RSS is the pre-mmap figure and contradicts the rows above — do not quote it. The
pre-v2.3 default was `--pool-size 4`; v2.3 lowered it to 2 plus a RAM-aware
auto-cap.

gigastt wins **on-disk size** (4–13× smaller than the Whisper/Vosk engines) and
**cold-start** (0.94 s; Vosk 0.42 is a dreadful ~30 s). It is honestly **not** the
absolute smallest — T-one greedy is 138 MB — but T-one's *production* config adds a
5.5 GB KenLM, so gigastt is the smallest model **with no language-model trade-off**.
gigastt now also wins **peak RAM**: ~46 MB resident / ~277 MB `ps` RSS at
`--pool-size 1` makes it the lightest engine in this table, and even the
default `--pool-size 2` (~66 MB resident / ~510 MB RSS, ~20 MB marginal per
extra slot) sits below Vosk 0.54 (560 MB) and T-one greedy (672 MB). The
resident figure is what to budget: RSS counts the shared memory-mapped model,
whose pages the OS reclaims under pressure.

## Streaming measurement protocol

Streaming is **buffered/chunked over an offline RNN-T**, not a native streaming AM.
Encoder geometry (do not change without a new protocol version): stride **0.8 s**,
max window **2.5 s** by default (configurable via `--stream-max-window-secs`,
clamped to 2.4–30; longer windows improve long-phrase WER at a linear per-stride
encoder-cost increase), left context **1.5 s**. Slide commits are
hypothesis-stable by default (`--stream-stable-prefix`, opt out with
`--stream-stable-prefix=false`; see cli.md). The first decode
cannot run before
~0.8 s of new audio, so end-to-end TTFP cannot honestly be “sub-200 ms” on this path.

**Client (canonical, `STREAM_PROTOCOL_VERSION = 1.0`):**

1. Connect `GET /v1/ws`. Wait for `Ready`.
2. Send `{"type":"configure","sample_rate":16000}` **before** any audio.
3. Feed **16 kHz mono PCM16**, `chunk_ms=100`, real-time `sleep` between frames.
4. Start the TTFP clock on the **first audio frame** (after Ready + configure), not on connect.
5. Ignore empty / whitespace `partial`s. They do not start the clock.
6. Send `{"type":"stop"}` at EOF. Keep every `final` until the socket ends (mid-stream utterances + Stop flush); join them in order, then append a live partial only if it is not the last final. The Stop handler may drop TCP without a close frame — that is session end, not a dropped clip.
7. A clip with no counted partial before timeout stays in the corpus as `n` / `n_timeout` / `n_no_partial`. **p50/p95 are over observed TTFPs only** (a missing partial is not imputed). Quote `n_timeout` next to p95.
8. Warm server, INT8, CPU. Published latency rows use `--pool-size 1`. WER `--mode both` starts `serve` with the default `--pool-size 2`; do not mix those rows with the latency table. Stream RTF in `benchmark.py` is paced wall-clock (~1.0+), not encoder compute.

Commands:

```sh
# Streaming WER vs the same REST batch path (same files, same normalizer)
cd benchmark
python benchmark.py --mode both --runners gigastt --dataset golos_crowd --max-samples 100 \
  --output results_stream_wer.json

# TTFP / TTFS p50–p95 (server must already be up)
gigastt serve --port 9877 --pool-size 1
python benchmark_latency.py --dataset golos_crowd --max-samples 100 \
  --port 9877 --output results_latency_corpus.json
```

`--mode batch` is the historical REST table (default). `--mode stream` is WebSocket only.
`--mode both` prints **Δ = WER_stream − WER_batch** with a bootstrap 95% CI on the paired clips.

### Streaming vs batch WER

First **100** clips of each committed manifest (not the 1000-row competitor table
above). Apple M1 Pro, CPU INT8, `rnnt`. WER `--mode both` uses default
`--pool-size 2`. Same files, same normalizer. Measured 2026-08-14.
Summary artifact:
[`benchmark/results_full/stream_protocol_v1_100.json`](../benchmark/results_full/stream_protocol_v1_100.json).

| Dataset | n | WER_batch | WER_stream | Δ pp (stream − batch) | 95% CI on Δ |
|---|--:|--:|--:|--:|---|
| `golos_crowd_1k` | 100 | 4.97 | 19.46 | **+14.49** | [10.91, 18.24] |
| `golos_farfield` | 100 | 4.82 | 15.42 | **+10.60** | [6.53, 15.09] |

Crowd: 2 stream clips produced no transcript (counted as 100% WER). Farfield: 0
timeouts. Typical stream errors are dropped / truncated words (`сколько` →
`сколь`, long commands collapsed to a prefix), not substitutions of a full
sentence.

This 100-clip batch WER (4.97 / 4.82) is a **different n** from the 1000-row
table (3.55 / 4.08). Do not splice them. The Δ is the number that matters:
**streaming currently costs about 11–15 pp** on these slices.

A single-file guard still exists: `crates/gigastt-core/tests/streaming_quality.rs` (`golos_00`, word overlap ≥ 0.5). That is not a corpus WER.

### Streaming latency (p50 / p95)

Older single-clip smoke (`golos_00.wav`, 4 s, real-time, timer from first audio):
**TTFP ~782 ms (CPU) / ~693 ms (CoreML)**. That number is dominated by *where the first word
falls* plus the 0.8 s stride — not by encoder compute (~70–100 ms/chunk).

Corpus (same protocol, Apple M1 Pro, CPU INT8, warm `--pool-size 1`, 2026-08-14).
p50/p95 are over **observed** values only. `n_timeout` / `n_no_partial` /
`n_error` stay in the experiment count.

**`golos_crowd_1k`** (n=100; 2 clips no partial / harness error):

| Metric | n | p50 | p95 | max | notes |
|---|--:|--:|--:|--:|---|
| TTFP (first audio → first non-empty partial) | 98 | 1653 | 2628 | 3045 | clip-start; 0.8 s stride buckets |
| TTFS (energy onset → first partial) | 89 | 803 | 2514 | 3045 | 11 clips had no onset |
| Partial lag (send → partial) | 452 | 51 | 100 | 298 | compute + queue |
| Finalization lag (first audio → final) | 98 | 4284 | 7325 | 10754 | includes clip duration |

**`golos_farfield`** (n=100; 0 timeouts):

| Metric | n | p50 | p95 | max | notes |
|---|--:|--:|--:|--:|---|
| TTFP | 100 | 820 | 829 | 1704 | almost all first-stride |
| TTFS | 42 | 500 | 656 | 731 | 45 no onset; 13 dropped (onset after partial) |
| Partial lag | 310 | 41 | 114 | 443 | compute + queue |
| Finalization lag | 100 | 2787 | 5443 | 7454 | includes clip duration |

TTFP p50 is **0.82 s (far-field) / 1.65 s (crowd)** — first-word position plus
the 0.8 s stride, not encoder compute. Per-partial lag p50 is **41–51 ms**,
p95 ~100–114 ms. Negative TTFS (energy onset after the first partial) is
dropped from the percentile, not imputed.

Vosk-server and T-one (300 ms chunks) are also genuine streaming designs. Whisper engines are offline. gigastt’s streaming win vs Whisper is incremental partials from one binary, **not** a lowest-latency claim, and **not** batch-equal WER on the live path.

## Edge / Raspberry Pi

No Raspberry Pi measurements exist yet — every cell below is a placeholder, and
nothing on this page is extrapolated from the Apple M1 numbers above. The full
measurement protocol (boards, storage variants, warm-up, metrics) lives in
[`specs/edge-raspberry-pi-roadmap.md`](../specs/edge-raspberry-pi-roadmap.md);
operators run it on-device with
[`scripts/bench_edge_pi.sh`](../scripts/bench_edge_pi.sh), which wraps
[`benchmark/bench_edge.py`](../benchmark/bench_edge.py) (cold-start, RSS@ready,
warm RTF, RSS after decode, WebSocket time-to-first-partial).

| Platform | Head | RTF | Peak RSS | Cold-start | TTFP |
|---|---|---|---|---|---|
| Raspberry Pi 4 (microSD) | `rnnt` INT8 | — | — | — | — |
| Raspberry Pi 4 (microSD) | `ml_ctc` INT8 | — | — | — | — |
| Raspberry Pi 4 (USB SSD) | `rnnt` INT8 | — | — | — | — |
| Raspberry Pi 4 (USB SSD) | `ml_ctc` INT8 | — | — | — | — |

"—" = not measured. Pi rows are filled in only from on-device runs of the
protocol above.

**Apple M1 reference (same protocol, `--pool-size 1`).** For orientation while
Pi hardware is pending: the same harness on the M1 development machine, per
head. RTF and TTFP are from the committed `benchmark/results_edge_m1.json`
(gigastt 2.16.0); RAM and cold-start come from a 2026-08-04 re-measurement
after the memory-mapped ORT-cache change and are **not** in that artifact —
it is pre-mmap (it reads ~1.0 s cold start, ~747 MB RSS@ready). **Not** a Pi
prediction.

| Head | RTF | Peak RSS | Cold-start | TTFP |
|---|---|---|---|---|
| `rnnt` INT8 | 0.043 (0.041–0.045) | ~277 MiB RSS / ~46 MB resident | ~0.4 s warm boot | 766 ms |
| `ml_ctc` INT8 | 0.032 (0.030–0.036) | ~261 MiB RSS / ~28 MB resident | ~0.3 s warm boot | 749 ms |

RTF and TTFP measured 2026-08-03, gigastt 2.16.0, M1 16 GB, 5 warm
`golos_0{0..4}` fixtures; RTF is mean (min–max). TTFP is time to first partial
on a real-time-paced 4 s stream; finalization lag ≈ audio duration + ~150 ms
for both heads. RAM and cold-start re-measured 2026-08-04 on M1 Pro after the
memory-mapped ORT-cache change (`--pool-size 1`): RSS is process `ps`
RSS after warm decodes, resident is the macOS `footprint` dirty+compressed
figure; the warm boot reads the cached `.ort` file, and the first boot after a
model update pays a one-time ~2.7 s `.onnx`→`.ort` conversion. `ml_ctc` is the
lighter head — a single encoder-only session, no decoder/joiner pair.

> **RAM note.** The encoder weights now load from a memory-mapped ORT-format
> cache (`.ort`) with zero-copy initializers and ORT prepacking disabled: the
> 215 MB model is file-backed, shared across pool sessions, and the OS
> reclaims those clean pages under memory pressure. Hence two honest metrics:
> **resident footprint** (dirty + compressed pages) — ~46 MB pool-1 / ~66 MB
> pool-2 after warm decodes (~35 / ~57 MB at `/ready`), ~20 MB marginal per
> extra slot — and **`ps` RSS**, which counts the shared mapping per mapping
> (~277 / ~510 MB). Budget the resident figure. The server's own
> `memory_after_load rss_mb=` startup log samples before the mapping is
> touched and reads only ~55 / ~83 MB — a known under-read, not a number to
> quote.

**Caveats (read before quoting any of this):**

- **Vosk 0.54 vs Vosk small must not be conflated.** The Vosk rows elsewhere on
  this page are the 966 MB Zipformer2 model; the ~45 MB Vosk-small that makers
  actually run on Pi is a different, much weaker model. "Vosk WER + small size"
  is never one row.
- **Default `rnnt` output is bare lowercase.** Readable text needs
  `--punctuation` / `--itn` (an extra model plus extra CPU — relevant on
  constrained devices) or the `e2e_rnnt` head, which trades WER for baked-in
  punctuation.
- **Diarization model is skippable.** Speaker diarization is opt-in at request
  time; `download --skip-diarization` skips downloading the speaker model on
  constrained devices.

## Headline single-engine metrics

All gigastt numbers are the default **`rnnt`** head (since v2.3; INT8), measured through the
cross-engine Python harness so they line up with the table above. As noted in the
provenance note up top, this `rnnt` re-measurement is not committed to
`benchmark/results_full/` — the committed gigastt artifacts there are the older
`e2e_rnnt` run.

| Metric | Value |
|---|---|
| **WER — clean read** | **3.55%** (`golos_crowd_1k`, 992 samples, 95% CI 2.9–4.2%) |
| WER — other domains | far-field **4.08%** · phone **18.50%** · YouTube **10.91%** |
| Verbatim → normalized WER | clean 9.73→3.55 · far-field 4.69→4.08 · phone 19.39→18.50 · YouTube 12.19→10.91. The gap is number/filler formatting, normalized **symmetrically for every engine** (so it neither helps nor hurts gigastt relative to competitors). |
| RTF (`rnnt` INT8, M1 CPU) | ~0.10 |
| RAM (default `--pool-size 2`) | ~66 MB resident / ~510 MB `ps` RSS (single session ~46 MB / ~277 MB — RSS counts the shared memory-mapped model; resident is the honest figure) |
| INT8 encoder (only runtime path) | ~215 MB on disk |

## Held-out queue

Full 3-engine table (gigastt · Vosk 0.54 · faster-whisper L3) is above. Status:

| # | Status | Dataset | Domain |
|---|--------|---------|--------|
| 1 | **done** (+ FW) | Mozilla Common Voice RU | clean / crowd read |
| 2 | **done** (+ FW) | FLEURS `ru_ru` (WER) | clean read |
| 3 | **done** (+ FW) | Russian LibriSpeech (RuLS) | audiobook |
| 4 | **done** (+ FW) | SOVA RuDevices | device / command |
| 5 | **partial** (n=67 only, + FW) | Podlodka Speech | conversational |
| 6 | **done** (+ FW) | ToneWebinars | webinar / lecture |
| 7 | optional | Phone-sim on a held-out set | telephony proxy |

Full queue, prep scripts, protocol, and definition of done:
[`specs/held-out-datasets-roadmap.md`](../specs/held-out-datasets-roadmap.md).

## Reproduce

```sh
cd benchmark
pip install -r requirements.lock.txt
python benchmark.py --runners gigastt --dataset golos_crowd_1k --max-samples 0 --no-cache
```

New competitor runners (Vosk 0.54, faster-whisper-turbo, T-one) live under
[`benchmark/runners/`](../benchmark/runners/); each gracefully skips if its optional
dependency/model is absent. T-one beam+LM needs the 5.5 GB KenLM (`BENCHMARK_TONE_KENLM`).
