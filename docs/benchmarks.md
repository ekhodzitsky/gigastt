# Benchmarks

> **Model revision:** Unless explicitly linked to the [2026-10-09 model
> comparison](rnnt-quantization.md), performance and footprint measurements on
> this page describe the older ConvInteger bundle. They are historical evidence,
> not current-model memory budgets. The current encoder is 305 MiB; its M1
> footprint and latency have not been remeasured.

Historical comparisons of gigastt against Russian-ASR engines. The primary
comparison used Apple M1 CPU, 1,000-sample manifests per domain (992 nonempty
clean-read references), failures included in scoring, and bootstrap intervals.
Competitor and older `e2e_rnnt` results are retained in
[`benchmark/results_full/`](../benchmark/results_full/). Methodology and dataset
preparation are in [`benchmark/README.md`](../benchmark/README.md).

> **Primary `rnnt` scores withheld.** The previously reported WER values
> 3.55 / 4.08 / 18.50 / 10.91 and their intervals cannot currently be traced to
> the original per-sample results and complete run configuration. They are
> historical reports, not substantiated headline measurements or a basis for
> ranking engines. The committed `*_gigastt*.json` files for these four domains
> describe the older gigastt 2.0.13 `e2e_rnnt` run, not that `rnnt` measurement.

### Evidence required to restore the primary scores

Each score must link to a retrievable raw result and a content checksum, together
with the exact corpus manifest and audio revision/checksums, code revision and
binary build configuration, model filenames and SHA-256 hashes, hardware,
ONNX Runtime version/provider, thread/pool settings, and all transcription
options (including punctuation, ITN, VAD and hotwords). Record per-sample
hypotheses, references, failures and exclusions; version the normalization and
scoring code, aggregation rule, bootstrap method, seed and number of resamples.
Publish verbatim and normalized scores together, with failure counts and timing.

The local September Linux clean-read result also reports 3.55%, but lacks that
complete provenance and is not the original M1 run. Its far-field result is
4.32%, not the former 4.08% headline. Neither restores the four-domain comparison.
The public benchmark-results branch contains a different 15-fixture smoke run.
A new fully documented run may replace the missing evidence; older `e2e_rnnt`,
multilingual, held-out and streaming measurements must remain separately labelled.

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
| **gigastt** (GigaAM v3 `rnnt`, INT8) | pending evidence | pending evidence | pending evidence | pending evidence |
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
phone / YouTube sets were not on hand for that comparison. Separate verified Kazakh,
Kyrgyz and Uzbek measurements are in [Languages](languages.md).

## Held-out / additional public sets — WER % (95% CI)

Same harness and machine (Apple M1, CPU, `rnnt` INT8). These are **not** the
Golos/OpenSTT slices above (still may overlap train mixes in general — see the
contamination caveat). Protocol:
[`docs/held-out-datasets-roadmap.md`](held-out-datasets-roadmap.md).
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

### Author-package oracle

The table above is gigastt against other engines. It is not a check against
the author `gigaam` package. That check is separate:
[`benchmark/tolerances/gigaam-v3-rnnt.json`](../benchmark/tolerances/gigaam-v3-rnnt.json).

gigastt loads the INT8 encoder only, so there is no FP32 engine number. The
FP32 reference is `gigaam.load_model("v3_rnnt", fp16_encoder=False, device="cpu")`
(package revision `7447938`, greedy, no external language model). The shipped
ONNX encoder is one graph, so the author package's per-block tensors are not
outputs we can diff.

On `crates/gigastt/tests/fixtures/golos_00.wav` (4 s, 16 kHz) the INT8
transcript and the author FP32 transcript are the same string. Log-mel is
`[64, 399]` (HTK, `center=false`, hop 160). The author clamps energy at
`1e-9` before the log and this frontend clamps at `1e-10`, so silent bins
differ by `ln(10)` (max abs 2.303). On bins above that clamp the mean
absolute gap is 0.0088 and the max is 0.307. Encoder output is `[768, 100]`:
INT8 versus the author FP32 activation has cosine 0.9985 and max abs 0.145
on the measured Ryzen host. The GitHub Ubuntu runner measured 0.99750 and
0.22470 with the checksum-verified model. The CPU oracle requires cosine
above 0.997, max abs below 0.25, and the exact reference transcript; the
runner observation and model hash are recorded in the tolerance JSON.

FLEURS `ru_ru` test, n=775, same greedy recipe, punctuation and ITN off.
Word error here is raw `jiwer.wer` (whitespace tokens, no number
normalization). That is not the harness figure in the table above (gigastt
5.26% on an Apple M1). The author package refuses five clips longer than one
pass; those count as empty hypotheses. On the 770 clips both sides decode,
INT8 is 8.918% and the author FP32 package is 8.816% (+0.102 pp). On all
775, counting those five as deletions for the author only, INT8 is 8.939%
and the author is 10.168%. Measured with gigastt 2.21.0 on an AMD Ryzen AI
9 HX 370, CPU, 2026-09-27. Scoring these same INT8 hypotheses with the
harness normalizer on this machine gives 4.65%. That does not replace the
table, and it was not computed for the author package.

```sh
python3 scripts/oracle_gigaam.py
cargo test -p gigastt-core --lib test_golos_00_int8_encoder_near_author_fp32 -- --ignored
cargo test -p gigastt --test oracle_gigaam -- --ignored
```

The mel comparison runs in ordinary `cargo test --lib` and does not need the
model. The script skips when `gigaam` or the release binary is missing. The
ignored tests skip when `~/.gigastt/models/v3_rnnt_encoder_int8.onnx` is
absent. FLEURS-ru is skipped until `scripts/prepare_fleurs.py --config ru_ru`
has written the manifest and the wavs.

> The pre-v2.3 default was the `e2e_rnnt` head (clean read 8.60%, far-field 5.90,
> phone 19.28, YouTube 11.35). The missing primary `rnnt` artifacts prevent a
> supported comparison with those scores. Both heads share the encoder — `rnnt` emits bare lowercase
> text (pair with `--punctuation` / `--itn` for readable output), `e2e_rnnt` bakes in
> punctuation/casing. The harness applies the same normalization to references
> and hypotheses, but this does not imply equal benefit across engines; see the
> normalization caveat in the benchmark guide.

> **Multilingual heads.** `ml_ctc` (220M) and `ml_ctc_large` (600M) are the opt-in GigaAM
> Multilingual charwise-CTC heads (ru/en/kk/ky/uz). The 600M head reports
> 4.44% clean / 5.70% far-field against the old `e2e_rnnt` 8.60% clean result, while
> the 220M head (6.15% / 8.28%) is the smaller, faster option. Measured through the same
> harness, manifests, and normalization as the rows above; bare lowercase output, so pair
> with `--punctuation` / `--itn` for readable text.

### Punctuation quality — `e2e_rnnt` vs `rnnt` + RuPunct restore

The `rnnt` head is bare lowercase, so readable Russian comes two ways: bake it in
with the `e2e_rnnt` head (one pass), or restore it on top of `rnnt` with the `--punctuation`
RuPunct model plus `--itn` (two passes). Measured on **775 punctuated FLEURS-ru references**
(the `raw_transcription` field; numbers are written as digits, so both configs run `--itn on`
to match), position-based F1 with the same metric as
[`benchmark/benchmark_punctuation.py`](../benchmark/benchmark_punctuation.py):

| Config | Punctuation F1 | Capitalization F1 |
|---|---|---|
| `e2e_rnnt` (one pass, baked in) | **0.540** | **0.726** |
| `rnnt` + RuPunct restore (two passes) | 0.355 | 0.656 |

The reported `e2e_rnnt` scores are higher on both metrics. Position-based F1
also depends on word alignment; these measurements do not establish a lower
bound on the punctuation-quality gap or validate the missing four-domain WER
comparison. `rnnt` provides bare text with optional restoration; `e2e_rnnt`
provides punctuation, casing and ITN in one pass.

**Interpretation:** the primary `rnnt` evidence gap prevents a supported
four-domain ranking or a clean-read equivalence claim. Overlapping marginal
confidence intervals alone would not establish statistical equivalence either.
Packaging and resource measurements are discussed separately below and in the
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
English, Kazakh, Kyrgyz and Uzbek. See [Languages](languages.md) for the later
1,000-recording FLEURS evaluation and separate conversation measurements.

> Same caveat as the Russian table: GigaAM Multilingual is pre-trained on 2M hours across
> 70+ languages and LibriSpeech is a common English ASR corpus, so read this as a best-case
> in-distribution upper bound, not WER on unseen English.

## Kazakh / Kyrgyz / Uzbek — WER % (FLEURS)

The published fixed evaluation uses 1,000 distinct recordings per language and
model, with all eligible FLEURS test rows followed by a seeded validation top-up.
These are complete-set micro WER values with Unicode letters preserved,
apostrophe variants removed and no number conversion. Number-bearing references
remain in the primary metric. This replaces the earlier digit-free headline;
the populations and scoring scopes must not be mixed.

| Head | Kazakh | Kyrgyz | Uzbek |
|---|---:|---:|---:|
| `ml_ctc_large` INT8 | 11.47% | 12.34% | 13.89% |
| `ml_ctc` INT8 | 12.32% | 13.68% | 16.75% |

[Full five-language report, raw outputs and independent audit](../benchmark/results/multilingual_1000_20261001/README.md).
This is reading, not telephone or conversational accuracy, and not the official
test-only FLEURS benchmark. Training overlap is unknown.

Separate natural-speech measurements for `ml_ctc_large` are **36.91% WER** on
887 short Kazakh conversational utterances, **16.54%** on 31 Kazakh media clips,
and **22.16%** on 745 Uzbek voice messages. Kyrgyz conversation remains unmeasured.
Whisper large-v3 and Omnilingual CTC1B v2 did not improve WER on the matched
eligible sets; Omnilingual had lower CER on Kyrgyz reading despite higher WER.

[Language selection and interpretation](languages.md) ·
[Matched comparison and CPU resource measurements](../benchmark/results/multilingual_model_comparison_20261001/README.md).
The latter is a separate Linux CPU experiment, not the historical M1 speed table below.

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
`--stream-stable-prefix=false`; see [cli.md](cli.md)). That default
shipped in 2.19.0 (2026-08-31) and is **not** what the published corpus
table measured — see below. The first decode cannot run before
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

Commands below measure **today's** defaults, including stable-prefix on.
They did not produce the historical table.

```sh
# New streaming WER vs the same REST batch path (same files, same normalizer).
# Not the 2026-08-14 table — that run predates stable-prefix.
cd benchmark
python benchmark.py --mode both --runners gigastt --dataset golos_crowd --max-samples 100 \
  --output results_stream_wer.json

# TTFP / TTFS / finalization lag (server must already be up). Latency, not WER.
gigastt serve --port 9877 --pool-size 1
python benchmark_latency.py --dataset golos_crowd --max-samples 100 \
  --port 9877 --output results_latency_corpus.json
```

`--mode batch` is the file/REST competitor table (the default mode). `--mode stream` is WebSocket only.
`--mode both` prints **Δ = WER_stream − WER_batch** with a paired bootstrap 95% CI
(1000 resamples, 2.5th and 97.5th percentiles; [`benchmark/streaming.py`](../benchmark/streaming.py)).

### Streaming vs batch WER

**Historical (2026-08-14, pre-stable-prefix). Not a current accuracy claim.**
`--stream-stable-prefix` did not exist on this run; it shipped in 2.19.0
(2026-08-31) and is on by default now. No corpus rerun under that policy is
in this repository, so there is no supported current stream-minus-file WER.
Do not quote the Δ column as what streaming costs today.

First **100** clips of each committed manifest (not the 1000-row competitor table
above). Summary artifact (rollup only — no per-clip hypotheses):
[`benchmark/results_full/stream_protocol_v1_100.json`](../benchmark/results_full/stream_protocol_v1_100.json).

| Dataset | n | WER_batch | WER_stream | Δ pp (stream − batch) | 95% CI on Δ |
|---|--:|--:|--:|--:|---|
| `golos_crowd_1k` | 100 | 4.97 | 19.46 | **+14.49** | [10.91, 18.24] |
| `golos_farfield` | 100 | 4.82 | 15.42 | **+10.60** | [6.53, 15.09] |

Recovered configuration, from that artifact plus the tree that added it
([`7049118da4fdce3d9fa4297e274eb4426195ddf0`](https://github.com/ekhodzitsky/gigastt/commit/7049118da4fdce3d9fa4297e274eb4426195ddf0),
`v2.18.0-4-g7049118`, four commits after tag `v2.18.0`):

- **Policy.** Fixed 2.5 s window, 0.8 s stride, 1.5 s left context
  (`--stream-max-window-secs` did not exist yet). On the window cap the
  engine committed every live word and slid. That is not hypothesis-stable
  prefixes (two agreeing decodes, 1.0 s edge horizon), which landed later in
  [`b012f8a72ebde7581a7181c23f8aac13239bc3aa`](https://github.com/ekhodzitsky/gigastt/commit/b012f8a72ebde7581a7181c23f8aac13239bc3aa).
- **Head / machine.** INT8 `rnnt`, Apple M1 Pro, CPU, as the artifact's
  `machine` field records. No encoder checksum and no `gigastt --version`
  string are in the artifact.
- **Harness.** The summary was added in that commit, from
  `benchmark.py --mode both` on the first 100 clips of
  [`golos_crowd_1k`](../benchmark/manifests/golos_crowd_1k.json) and
  [`golos_farfield`](../benchmark/manifests/golos_farfield.json) (those
  manifest blobs are unchanged since the commit). The artifact note says WER
  used default `--pool-size 2` and latency used `--pool-size 1`. The runner
  in that commit starts `serve` with no punctuation, ITN, variant, or VAD
  flag, so the CLI defaults of that tree apply: punctuation `auto`, ITN
  `auto`, variant auto-detected (artifact: `rnnt`), VAD off, endpoint mode
  `auto` (decoder blank-run). The artifact does not record `gigastt --version`
  or whether a punctuation model was on disk.
- **Normalization.** Reported WER is the harness `compute_wer` pass
  (`normalize_for_wer` in [`benchmark/common.py`](../benchmark/common.py)):
  symmetric lowercase, `ё`→`е`, words-to-digits ITN, anglicism map, on both
  reference and hypothesis. The summary does not include the verbatim
  (`naive`) pass.
- **CI.** Paired bootstrap in [`benchmark/streaming.py`](../benchmark/streaming.py)
  (1000 resamples, 2.5th / 97.5th percentiles). The JSON stores rolled-up
  bounds only. Far-field `delta_pp` is `10.6` there (two-decimal rounding);
  the table shows that published value as +10.60.

Crowd: 2 stream clips produced no transcript (counted as 100% WER). Farfield: 0
timeouts. On this run, typical stream errors were dropped / truncated words
(`сколько` → `сколь`, long commands collapsed to a prefix), not substitutions
of a full sentence.

This 100-clip batch WER (4.97 / 4.82) is a **different n** from the 1000-row
table, whose `rnnt` scores are now withheld. Do not splice them.

**No current corpus number.** A replacement claim needs the same protocol
with stable-prefix **on**, and a retrievable raw artifact naming the code
revision, model version, corpus manifest, normalization, window/policy, and
confidence-interval method. This revision does not include that rerun. The
2.19.0 changelog note on 10 labelled Golos fixtures
(`crates/gigastt-core/tests/streaming_quality.rs`) is a different, smaller
regression guard — not this table, and not a substitute for it. The same
file's `golos_00` word-overlap check (≥ 0.5) is not a corpus WER either.

### Streaming latency (p50 / p95)

Latency only. These rows are not a stream-minus-file WER, and they are not
mixed into the accuracy claim above.

Older single-clip smoke (`golos_00.wav`, 4 s, real-time, timer from first audio):
**TTFP ~782 ms (CPU) / ~693 ms (CoreML)**. That number is dominated by *where the first word
falls* plus the 0.8 s stride — not by encoder compute (~70–100 ms/chunk). It is
not a corpus figure and not from the 2026-08-14 run.

Corpus latency from the **same 2026-08-14 measurement** as the historical WER
table (revision `7049118`, pre-`--stream-stable-prefix`, Apple M1 Pro, CPU
INT8, warm `--pool-size 1`; rollup in the same
[`stream_protocol_v1_100.json`](../benchmark/results_full/stream_protocol_v1_100.json)).
Not a remeasurement with stable-prefix on. p50/p95 are over **observed**
values only. `n_timeout` / `n_no_partial` / `n_error` stay in the experiment
count. TTFP, TTFS, partial lag, and finalization lag stay separate from the
WER table above; finalization lag includes clip duration (see the notes on
each row).

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

Vosk-server and T-one (300 ms chunks) are also genuine streaming designs. Whisper engines are offline. gigastt’s streaming win vs Whisper is incremental partials from one binary, **not** a lowest-latency claim. Live WER is not the file-transcription table; the only published corpus Δ is the historical table above, and it is not a current figure.

## Edge / Raspberry Pi

No Raspberry Pi measurements exist yet — every cell below is a placeholder, and
nothing on this page is extrapolated from the Apple M1 numbers above. The full
measurement protocol (boards, storage variants, warm-up, metrics) lives in
[`docs/edge-raspberry-pi-roadmap.md`](edge-raspberry-pi-roadmap.md);
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

## Pull-request model smoke gate

`Model smoke` runs on every pull request (including forks) and main push,
with read-only repository permissions and no secrets. It caches only the four
pinned `rnnt` recognition files; cache misses download them, and engine loading
verifies the checksums from `model/variant.rs` on every run. The workflow checks
file presence before invoking the ignored oracle so a missing model cannot
silently skip the gate. Exact test names are also checked against the test list
to reject a zero-test run after renaming. It does not cache optimized graphs or
optional sidecars.

The serial tests compare the encoder to the author activation fixture and the
exact expected transcript (see the oracle tolerance JSON), then exercise real
speech through WebSocket partials, window slides and finalization under all
commit policies. The WebSocket test checks text consistency between policies,
not equality with batch recognition or corpus WER. These are correctness gates;
no shared-runner latency threshold is enforced. The entire job has a 25-minute
limit. The full custom WER harness is never invoked by these explicit targets.

Repository administrators must require the `Model smoke` check for `main`
after this workflow is available. Full E2E and quality suites remain main-push
checks; the Criterion regression job remains advisory because runner timing is
noisy. Forks use the ordinary `pull_request` event, not a privileged target event;
GitHub may require a maintainer to approve a first-time contributor's run.

## Headline single-engine metrics

The primary four-domain `rnnt` WER claims are withheld pending the evidence
specified above. Resource measurements below have their own measurement context;
they do not validate the missing accuracy run.

| Metric | Value |
|---|---|
| WER — primary four-domain comparison | Pending complete raw results and provenance |
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
[`docs/held-out-datasets-roadmap.md`](held-out-datasets-roadmap.md).

## Reproduce

```sh
cd benchmark
pip install -r requirements.lock.txt
python benchmark.py --runners gigastt --dataset golos_crowd_1k --max-samples 0 --no-cache
```

New competitor runners (Vosk 0.54, faster-whisper-turbo, T-one) live under
[`benchmark/runners/`](../benchmark/runners/); each gracefully skips if its optional
dependency/model is absent. T-one beam+LM needs the 5.5 GB KenLM (`BENCHMARK_TONE_KENLM`).

## Stereo container decode work

For `channels=split`, the full recording determines whether stereo channels
are duplicates. Non-Opus formats use a bounded streaming scan. Mono inputs need
only a header probe before transcription; dual-mono inputs keep the original
container for mono mixing, preserving mix-before-resample behavior.

Opus channel detection already materializes the channels under the existing
whole-buffer duration ceiling. With VAD enabled, genuine stereo now reuses that
PCM instead of decoding and resampling the entire container again. This reduces
two whole-channel Opus passes to one without adding a buffer or changing the
ceiling. Dual-mono still uses its original mono fallback. WAV/FLAC VAD splitting
keeps its scan plus channel decode; non-VAD genuine stereo keeps a scan plus one
windowed pass per channel. Avoiding those passes would require a different
memory/storage policy, so this optimization does not change them.

The ignored `benchmark_split_channel_decode` unit test measures container work
without ASR or VAD inference. Supply `GIGASTT_STEREO_BENCH_FILE` and
`GIGASTT_STEREO_BENCH_VAD=0|1`; with VAD, `GIGASTT_STEREO_BENCH_REUSE=1` selects
the retained scan path and `0` selects the previous scan-then-decode path. Run
one test process per case with `/usr/bin/time -v` for peak RSS, including the
encoded input and test runtime. Use mono, duplicate-channel and distinct-channel
fixtures at the same rate/duration across WAV, FLAC and Opus. The test reports
sample counts, scan/total time and whole-channel Opus decoder invocations.
These measurements isolate container work and do not establish an end-to-end
transcription speedup. Without an explicit fixture, the manual test prints a
skip message so the full ignored model-test suite can still run in CI.

```sh
GIGASTT_STEREO_BENCH_FILE=/path/to/stereo.opus \
GIGASTT_STEREO_BENCH_VAD=1 GIGASTT_STEREO_BENCH_REUSE=1 \
cargo test -p gigastt-core --lib \
  inference::audio::tests::stream::benchmark_split_channel_decode \
  -- --ignored --exact --nocapture
```
