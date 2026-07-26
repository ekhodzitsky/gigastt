# Held-out / public datasets roadmap

**Goal.** Publish WER (and RTF where cheap) on public Russian speech sets
**outside** the current Golos + OpenSTT matrix, so the contamination caveat in
[`docs/benchmarks.md`](../docs/benchmarks.md) is paired with a second column of
evidence. Measure **one dataset at a time**; ship results before starting the next.

**Not a guarantee of zero train overlap.** GigaAM / competitors may still have
seen similar corpora. The point is: *not the same Golos/OpenSTT val slices we
already publish*, different collection pipelines where possible, fixed protocol.

## Protocol (every item)

1. Prepare a **deterministic** slice (prefer `seed=42`, ~1000 utt when the set is large enough; smaller OK with wider CI).
2. Manifest in `benchmark/manifests/<name>.json` (same schema as existing).
3. Run the cross-engine Python harness:
   ```sh
   cd benchmark
   # Vosk 0.54 runner CLI id: vosk-0.54 (normalized vosk_0_54)
   python benchmark.py --dataset <name> --max-samples 0 \
     --runners gigastt --output results_full/<name>_gigastt.json
   python benchmark.py --dataset <name> --max-samples 0 \
     --runners vosk-0.54 --output results_full/<name>_vosk054.json
   python benchmark.py --dataset <name> --max-samples 0 \
     --runners faster_whisper --output results_full/<name>_faster_whisper.json
   ```
4. Attach or keep local `results_full/<name>*.json` artifacts (`results_full/` is
   gitignored; published numbers go into docs).
5. Update [`docs/benchmarks.md`](../docs/benchmarks.md) with a **Held-out / additional public sets** table row + provenance + license note in [`benchmark/DATA_LICENSE`](../benchmark/DATA_LICENSE) / `NOTICE` if new upstream license appears.
6. Update [`benchmark/README.md`](../benchmark/README.md) dataset section.
7. Mark the row below **done** with date and short note.

Default engines: **gigastt (`rnnt` INT8)**, **Vosk 0.54**, **faster-whisper large-v3**.
Same hardware note as the main table (Apple M1 CPU unless stated otherwise).

## Queue

| # | Status | Dataset | Domain | License (check card) | Source | Prep | Headline (gigastt / Vosk / FW) |
|---|--------|---------|--------|----------------------|--------|------|--------------------------------|
| 1 | **done (2026-07-25)** · FW same day | Common Voice RU | crowd read | CC0-1.0 | mirror `artyomboyko/common_voice_21_0_ru` | `prepare_common_voice_ru.py` | **2.63** / 6.10 / 5.22 (n=1000) |
| 2 | **done (2026-07-25)** · FW 2026-07-26 | FLEURS `ru_ru` | clean read | CC BY 4.0 | [google/fleurs](https://huggingface.co/datasets/google/fleurs) | `prepare_fleurs.py --config ru_ru` | 5.26 / 6.14 / **3.84** (n=775) |
| 3 | **done (2026-07-25)** · FW 2026-07-26 | RuLS | audiobook | PD USA / LibriVox | [openslr.org/96](https://www.openslr.org/96/) | `prepare_rulslib.py` | **4.21** / 9.18 / 9.65 (n=1000) |
| 4 | **done (2026-07-25)** · FW 2026-07-26 | SOVA RuDevices | device / command | see HF | `bond005/sova_rudevices` | committed manifest | 10.30 / **6.28** / 14.79 (n=1000) |
| 5 | **partial (n=67)** · FW 2026-07-26 | Podlodka Speech | podcast | see HF | [podlodka_speech](https://huggingface.co/datasets/bond005/podlodka_speech) | `prepare_podlodka.py` | 7.33 / 9.96 / 7.27 (thin CI) |
| 6 | **done (2026-07-26)** · 3-engine | ToneWebinars | webinar / lecture | Apache-2.0 | [ToneWebinars](https://huggingface.co/datasets/Vikhrmodels/ToneWebinars) | `prepare_tone_webinars.py` | 13.02 / 14.87 / **8.33** (n=1000) |
| 7 | optional | Phone-sim on held-out set | telephony proxy | inherits source | reuse CV or FLEURS | *need* codec recipe | 8 kHz + μ-law/A-law proxy only |

### Explicitly out of this queue

| Dataset | Why |
|---------|-----|
| Golos crowd / farfield | Already published; GigaAM train-family |
| OpenSTT calls / YouTube | Already published; common mix |
| Random YouTube without refs | No ground truth |
| Paid/NDA call corpora | Commercial proof track — separate from public roadmap |
| ToneBooks | Deferred; webinars preferred for in-the-wild acoustics |

## Already published (baseline matrix)

Tracked for orientation only — not part of the new queue.

| Dataset | Role | Docs |
|---------|------|------|
| `golos_crowd_1k` | clean read (in-domain upper bound) | `docs/benchmarks.md` |
| `golos_farfield` | far-field | same |
| `openstt_calls` | phone | same |
| `openstt_youtube` | YouTube / noisy | same |
| `fleurs_ru_punct` | punctuation F1 only | same |
| `librispeech_test_clean` | English (ml_ctc heads) | same |
| FLEURS kk/ky/uz | multilingual heads | same |

## Definition of done (per dataset)

For items **1–4 and 6** the checklist is satisfied (manifest + 3 engines + CI +
docs row + license note + this roadmap). Item **5** is partial (n=67 only).

- [x] Manifest committed (or reproducible prep script + committed sample list).
- [x] Full run: gigastt + Vosk 0.54 + faster-whisper L3 (items 1–6).
- [x] Bootstrap 95% CI reported.
- [x] Row in `docs/benchmarks.md` under **Held-out / additional public sets**.
- [x] License note in `benchmark/DATA_LICENSE` + root `NOTICE`.
- [x] This roadmap: status dated.
- [ ] Item 5: larger Podlodka split if upstream grows.
- [ ] Item 7: phone-sim recipe (optional).

## Progress log

| Date | Dataset | Action |
|------|---------|--------|
| 2026-07-25 | — | Roadmap created; queue ordered; start with Common Voice RU. |
| 2026-07-25 | Common Voice RU | n=1000 seed=42 (CV 21.0 mirror); gigastt 2.63 · Vosk 6.10 · FW 5.22. |
| 2026-07-25 | FLEURS `ru_ru` | Full test n=775; gigastt 5.26 · Vosk 6.14. |
| 2026-07-25 | tooling | `vosk_054` runner: MP3 + IEEE-float WAV via soundfile/av. |
| 2026-07-25 | RuLS / SOVA / Podlodka | gigastt vs Vosk: RuLS 4.21/9.18 · SOVA 10.30/6.28 · Podlodka 7.33/9.96 (n=67). |
| 2026-07-26 | FLEURS faster-whisper | FW 3.84 (3.4–4.3) n=775 RTF 0.73 — leads FLEURS. |
| 2026-07-26 | RuLS / SOVA / Podlodka FW | FW: RuLS 9.65 · SOVA 14.79 · Podlodka 7.27. |
| 2026-07-26 | ToneWebinars | val RU n=1000 (~7.1 h); FW **8.33** · gigastt 13.02 · Vosk 14.87. Docs + NOTICE locked. |

## Links

- Methodology: [`benchmark/README.md`](../benchmark/README.md)
- Published tables: [`docs/benchmarks.md`](../docs/benchmarks.md)
- Data licenses: [`benchmark/DATA_LICENSE`](../benchmark/DATA_LICENSE)
- Contamination caveat: `docs/benchmarks.md` § Dataset contamination / README caveats
