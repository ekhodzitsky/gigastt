# Edge / Raspberry Pi roadmap

**Goal.** Make gigastt an honest, documented option on Raspberry Pi–class
hardware: publish real RTF / RAM / cold-start / TTFP numbers, then only invest
in speed work that those numbers justify.

**Non-goal (for this roadmap).** Winning every real-time scenario on Pi 4.
Batch / file transcription and optional near-real-time streaming are the
realistic first targets. “Vosk-quality-at-Vosk-speed with GigaAM WER” is a
longer product bet (P3), not a config flag.

**Status key:** `todo` · `in_progress` · `blocked (need hardware)` · `done`

---

## Why this exists

- All published RTF / footprint / TTFP numbers today are **Apple M1 CPU**
  ([`docs/benchmarks.md`](../docs/benchmarks.md)).
- Pi 4 is `aarch64` Linux with Cortex-A72, no CoreML/CUDA; arm64 Docker and
  release artifacts exist, but **no published Pi numbers**.
- Maker default is still Vosk because Vosk has years of Pi tutorials and
  expected RTF — not because quality wins on noisy speech.

Until Pi (and ideally Pi 5) numbers land, every EP / profile / “edge head”
decision is extrapolation.

---

## Scenario matrix (product truth)

| Scenario on Pi 4 | Realistic choice today | After P0–P2 (if numbers support it) |
|---|---|---|
| Voice commands / short phrases, quiet room | Vosk **small** | gigastt only if RTF ≪ 1 and SOVA-like domain OK |
| File → text, quality matters | **gigastt** (`rnnt`, pool 1) | same + published RTF |
| Overnight batch | **gigastt** | same |
| Continuous live dictation | Pi 5 / x86 / remote server | gigastt on Pi 4 only if measured RTF ≲ 0.5 with headroom |
| Far-field / phone / YouTube-like audio | gigastt **wins WER** on M1 matrix | same quality; speed is the open question |

**Vosk caveats (do not mix models):**

- **Vosk 0.54** (benchmarked): ~966 MB disk, ~560 MB RAM, RTF ~0.03 on M1;
  strong clean/command domains (e.g. SOVA), weaker far-field / phone / YouTube
  vs gigastt `rnnt`.
- **Vosk small (~45 MB):** much lighter and faster on Pi — **not** the model in
  our published WER tables. Do not claim “Vosk WER + small size” as one row.

**gigastt caveats on edge:**

- Default `rnnt` is bare lowercase; readable text needs `--punctuation` /
  `--itn` (extra model + CPU) or `e2e_rnnt` (worse WER tradeoff).
- `--vad` (Silero) and `--encoder-intra-threads` **already exist** — profile
  work is packaging, not invention.
- Diarization is opt-in at request time; skip downloading the speaker model
  on constrained devices (`download --skip-diarization`).

---

## Priority lanes

### P0 — Evidence, not code  ·  **in_progress**

**Definition of done:** committed methodology + fillable tables; at least one
real Pi 4 run artifact (or explicit “blocked: no hardware” with volunteer call).

| # | Item | Status | Notes |
|---|------|--------|-------|
| P0.1 | Protocol + host metadata schema | **done** | this doc + `benchmark/bench_edge.py` |
| P0.2 | One-shot edge harness (RTF, RSS, cold-start, TTFP) | **done** | `benchmark/bench_edge.py`, wrapper `scripts/bench_edge_pi.sh` |
| P0.3 | Run on **Pi 4 4GB and/or 8GB** | **blocked (need hardware)** | `rnnt` INT8 + `ml_ctc` INT8, `--pool-size 1` |
| P0.4 | Optional Pi 5 same protocol | todo | for “upgrade path” narrative |
| P0.5 | microSD vs USB3 SSD cold-start / RTF (same board) | **blocked (need hardware)** | label via `--storage-label` |
| P0.6 | Fill tables in `docs/benchmarks.md` | **in_progress** | Edge section + Pi placeholders + M1 reference landed 2026-08-03; Pi cells await hardware |
| P0.7 | One honest README / README_RU line + link | **done** | 2026-08-03: both READMEs link the Edge section, no invented claims |

**Metrics to collect (per board × head × storage label):**

| Metric | How | Success framing |
|---|---|---|
| RTF (mean over warm fixtures) | REST `/v1/transcribe`, pre-warmed server | RTF < 1.0 ⇒ faster than real-time on files |
| Peak RSS | `ps` RSS **and** resident (`footprint` dirty+compressed) after ready + one decode | M1 reference after mmap: ~46 MB resident / ~277 MB `ps` RSS at `--pool-size 1`. Pi still unmeasured — do not publish the copy-era ~400 / ~750 MB figures. |
| Cold-start | wall time binary start → HTTP 200 `/ready` | microSD vs USB comparison |
| TTFP | WS real-time paced stream (`golos_00.wav`) | report + note buffered/chunked nature |
| Host metadata | model string, RAM, cores, kernel, binary source | reproducibility |

**Minimum audio set:** `crates/gigastt/tests/fixtures/golos_0{0..4}.wav`
(already in-tree). Optional later: 50–100 Golos crowd samples for stabler RTF.

**Do not publish:** extrapolated “RTF 0.4–0.8 on Pi 4” as measured fact.
Guesses may live only in discussion / this roadmap under *Expectations*.

**Expectations (unmeasured — for planning only):**

- Pi 4 Cortex-A72 is several× slower single-thread than M1; RTF for `rnnt`
  INT8 may land near real-time or slightly under it — **measure**.
- `ml_ctc` (220M CTC) may be faster with higher WER (M1 clean 6.15 vs `rnnt`
  3.55) — candidate **edge head**, not default accuracy head.

---

### P1 — Package what already exists  ·  todo (after P0 numbers)

| # | Item | Status | Depends |
|---|------|--------|---------|
| P1.1 | `--profile edge` (or `rpi`) preset | todo | P0.3 optional but preferred |
| P1.2 | Preset meaning: `pool-size=1`, diarization off, sensible `encoder-intra-threads` (2–3), optional `--vad on`, punct/ITN off unless requested | todo | P1.1 |
| P1.3 | Docs: “Raspberry Pi / edge install” (Docker arm64 + prebuilt binary preferred; avoid `cargo install` on-device) | todo | P0.6 |
| P1.4 | Hint on first download (aarch64): prefer USB3/SSD for model dir over microSD | todo | P1.3 |
| P1.5 | Confirm release assets: `aarch64-unknown-linux-gnu` binary discoverability | todo | — |
| P1.6 | Auto-detect Pi model (`/proc/device-tree/model`) for preset | optional | P1.1 |

Preset defaults (proposed — confirm after P0 thread sweeps):

```text
--pool-size 1
--encoder-intra-threads 3   # leave 1 core for OS / capture; re-bench 2/3/4
--punctuation off
--itn off
# no diarization model required for lean deploy
# --vad optional for pause-heavy capture
```

---

### P2 — Speed (only if P0 says “near real-time, need headroom”)  ·  todo

| # | Item | Status | Risk / note |
|---|------|--------|-------------|
| P2.1 | Thread matrix on Pi: intra-op 2 / 3 / 4 | todo | cheap; may beat desktop defaults |
| P2.2 | Document `ml_ctc` as recommended **edge head** if RTF wins enough vs WER loss | todo | WER already known on M1; re-check quality subjectively on Pi |
| P2.3 | XNNPACK EP experiment on aarch64 INT8 | todo | needs ort feature support + bench; no merge without win |
| P2.4 | `target-cpu=cortex-a72` (or `native` on-device) release artifact | todo | separate CI target; measure before advertising |
| P2.5 | Adaptive streaming chunk size under measured RTF | todo | protocol/tests cost |
| P2.6 | VAD-first gate as default in edge profile when capture is continuous | todo | Silero already optional |

**Out of scope for P2:** NNAPI (no Android stack on Pi OS), VideoCore OpenCL
EP (not a supported ORT path worth investing in here), aggressive 4-bit quant
without WER validation.

---

### P3 — Product bet: best Russian STT for edge  ·  deferred

| # | Item | Status |
|---|------|--------|
| P3.1 | Distilled / smaller encoder trained for GigaAM-like quality | deferred (ML) |
| P3.2 | Structural quant / shorter encoder windows | deferred (ML + decode design) |
| P3.3 | Positioning campaign: “Pi RTF X, WER better than Vosk 0.54 on noisy domains” | deferred until P0+P2 |

---

## Implementation order

```text
P0.1–P0.2  harness + docs stubs          ← current
P0.3–P0.5  real hardware runs
P0.6–P0.7  publish numbers (or “blocked”)
P1.*       edge profile + install docs
P2.*       only levers justified by P0
P3.*       only if edge is a multi-quarter bet
```

---

## How to run P0 (operator)

### Preferred path: prebuilt arm64

```sh
# On the Pi (64-bit OS only):
docker pull ghcr.io/ekhodzitsky/gigastt:latest
# or install the aarch64 release tarball from GitHub Releases

# Model once (prefer USB SSD path if available):
gigastt download --model-variant rnnt
# optional second head for comparison:
gigastt download --model-variant ml_ctc
```

### Measurement

From a checkout (or copy of `benchmark/` + fixtures):

```sh
# Label storage honestly: microSD | usb-ssd | nvme | unknown
./scripts/bench_edge_pi.sh \
  --storage-label microSD \
  --variants rnnt,ml_ctc \
  --output benchmark/results_edge_pi4.json
```

Or call the Python harness directly:

```sh
python3 benchmark/bench_edge.py \
  --binary "$(command -v gigastt)" \
  --pool-size 1 \
  --variants rnnt \
  --storage-label usb-ssd \
  --output benchmark/results_edge.json
```

### After a run

1. Keep the JSON artifact (even if gitignored under local names).
2. Paste numbers into the **Edge / Raspberry Pi** tables in
   [`docs/benchmarks.md`](../docs/benchmarks.md).
3. Replace the README “unmeasured” note with one measured RTF line + link.
4. Tick P0.3–P0.7 in this file with date and board model string.

---

## What not to claim until measured

- Exact Pi 4 RTF ratios vs M1.
- “Works great for real-time dictation on Pi 4.”
- “Twice as accurate as Vosk at the same speed” (speed unknown; accuracy is
  domain-dependent — see SOVA where Vosk 0.54 wins).
- That `ml_ctc` is always the right Pi default (speed vs WER unproven on device).

---

## Links

| Artifact | Path |
|---|---|
| Edge harness | [`benchmark/bench_edge.py`](../benchmark/bench_edge.py) |
| Shell wrapper | [`scripts/bench_edge_pi.sh`](../scripts/bench_edge_pi.sh) |
| Latency reference (M1) | [`benchmark/benchmark_latency.py`](../benchmark/benchmark_latency.py), [Historical latency note](../docs/archive/streaming-latency-2026-06-13.md) |
| Published M1 matrix | [`docs/benchmarks.md`](../docs/benchmarks.md) |
| Held-out WER queue | [`specs/held-out-datasets-roadmap.md`](held-out-datasets-roadmap.md) |
| CLI flags | [`docs/cli.md`](../docs/cli.md) |

---

## Changelog (this roadmap)

| Date | Note |
|---|---|
| 2026-07-26 | Initial roadmap + P0 harness/docs stubs. Hardware runs blocked pending Pi access. |
| 2026-07-26 | Smoke-ran `bench_edge.py` on Apple M1 (`rnnt`, `--pool-size 1`): RTF mean ~0.058 on golos_00–04 fixtures; cold-start ~1.3 s. **RSS@ready ~780 MiB** on that host — higher than the README “~400 MB single-session” figure; treat published RAM as approximate until re-measured with a clear method (process RSS after ready + one decode, pool 1, INT8 only) on both M1 and Pi. TTFP step needs `pip install websockets`. |
| 2026-08-03 | Full M1 baseline under the protocol, both heads (`benchmark/results_edge_m1.json`, gigastt 2.16.0, TTFP included): `rnnt` RTF 0.043 / RSS 755 MiB / cold-start 1.01 s / TTFP 766 ms; `ml_ctc` RTF 0.032 / RSS 752 MiB / cold-start 0.83 s / TTFP 749 ms. RSS@ready ~744–747 MiB at `--pool-size 1` reproduces the 2026-07-26 anomaly — the “~400 MB single-session” headline figure does **not** match this protocol; flagged in `docs/benchmarks.md` Edge section, headline RAM claims not yet rewritten. `docs/benchmarks.md` gained the Edge / Raspberry Pi section (Pi placeholders + this M1 reference); README / README_RU gained the honest “not yet measured on Pi” line (P0.7 done). Pi cells still blocked on hardware. |
| 2026-08-03 | RAM anomaly **resolved**: the “~400 MB single-session” figure traced to v2.3.0 docs arithmetic (pool-4 ÷ 4), never a pool-1 measurement. Measured composition of the ~750 MB pool-1 footprint: ~215 MB INT8 weights (copied, not mmap’d) + ~300 MB ORT runtime/protobuf per session + ~180 MB base; pool 2 = ~1.3 GB; ~550–570 MB per extra slot. Ruled out by experiment: encoder intra-threads (743 MB at `--encoder-intra-threads 1`), ORT 1.24→1.28 bump, warmup, FP32 encoder. Published figures corrected in README / README_RU / AGENTS.md / docs/benchmarks.md. Edge implication: **Pi 4 4GB fits pool 1 only, and per-session cost (~570 MB) is the lever that matters** — candidate follow-ups: reuse the optimized-model cache instead of re-serializing 224 MB every boot; investigate per-session ModelProto retention (~300 MB). |
| 2026-08-04 | Both candidate follow-ups landed (optimized-cache reuse + memory-mapped ORT-format cache with zero-copy initializers, prepacking off) and the RAM situation is **resolved far below all prior numbers**: pool 1 ~46 MB resident / ~277 MB `ps` RSS, pool 2 ~66 MB / ~510 MB (steady state after 5 warm decodes, M1 Pro, INT8 `rnnt`; resident = macOS `footprint` dirty+compressed, RSS counts the shared mmap’d model). Per-slot marginal cost is now ~20 MB resident, not ~570 MB. RTF unchanged (0.037–0.056 on golos fixtures), transcription byte-identical, warm boot ~0.4 s (one-time ~2.7 s `.ort` conversion after a model update). Consequence for this roadmap: **RAM stops being the binding constraint on Pi 4 (4 GB)** — pool 2 fits trivially; the open Pi questions are RTF / thread tuning / storage, not memory. The P1 preset keeps `--pool-size 1`, but now for thread/RTF reasons rather than RAM. Published figures updated in README / README_RU / AGENTS.md / docs/benchmarks.md (both metrics quoted, with the mmap methodology). |
| 2026-08-17 | Peak-RSS success row updated: measure both `ps` RSS and resident; quote the mmap M1 reference (~46 / ~277 at pool 1), not the copy-era ~400 / ~750 MB class. Pi still unmeasured. User-facing docs (hub, workbook, CLI, crate README) now use the same resident-vs-RSS split. |
