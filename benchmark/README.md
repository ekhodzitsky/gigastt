# gigastt Cross-ASR Benchmark

Reproducible benchmark comparing **gigastt** against popular open-source ASR engines on Russian speech.

## Supported Engines

| Engine | Backend | Language | Installation |
|--------|---------|----------|--------------|
| gigastt | ONNX Runtime / Rust | Russian | Built from source or `cargo install` |
| whisper.cpp | GGML / C++ | Multilingual | Auto-downloaded on first run |
| faster-whisper | CTranslate2 / Python | Multilingual | `pip install faster-whisper` |
| Vosk | Kaldi / C++ | Russian | `pip install vosk` (model auto-downloaded) |

## Metrics

- **WER** (Word Error Rate) — lower is better. Computed after symmetric text normalization applied identically to the reference and the hypothesis for every engine.
- **RTF** (Real-Time Factor) — `processing_time / audio_duration`. Lower is better; < 1.0 means faster than real-time.

## Methodology

### Timing

RTF is measured against a **pre-warmed engine** so that model-load time is not unfairly charged to any runner:

- **gigastt** is measured via HTTP POST to a `gigastt serve` process that stays up for the whole benchmark. WebSocket streaming was evaluated but abandoned for WER benchmarking: when inference is slower than real-time, the streaming endpoint finalizes on incomplete audio and returns truncated transcripts.
- **faster-whisper** and **Vosk** load their models once in `is_available()` and reuse them for every sample.
- **whisper.cpp** runs in **server mode** (`whisper-server`). The model is loaded once when the server starts; each sample is sent as an HTTP POST to `/inference` and the wall-clock request latency is used as `processing_time`. This replaces the previous per-sample `whisper-cli` invocation that re-loaded the ~3 GB model on every file and produced an artificially high RTF.

WER is unchanged by this switch: whisper.cpp still uses the same `large-v3` Russian model and the same text normalization pipeline as the other engines.

### Word-error normalization

WER is computed after symmetric text normalization so that Russian number words and Arabic digits become comparable tokens. The same pipeline is applied to the reference and the hypothesis for every engine; there are no per-engine branches.

**Caveat — the benefit of normalization is not engine-neutral.** Although the *same* function runs on reference and hypothesis with no per-engine branches, the words-to-digits ITN and the anglicism map can only fire on hypotheses that contain Arabic digits or Latin tokens. Engines that emit digits/Latin (gigastt, whisper) get a large WER reduction from normalization; a word-only engine like Vosk gets none (and even loses a little). Measured on `golos_crowd_1k`, recomputed from the committed `results_full/*.json` (not a new run):

| Engine | naive WER | ITN WER | Δ |
|---|---|---|---|
| gigastt | 14.40 | 8.60 | **−5.80** |
| whisper.cpp | 20.63 | 15.26 | **−5.37** |
| faster-whisper | 20.51 | 15.53 | **−4.97** |
| Vosk | 4.57 | 4.82 | **+0.25** |

Normalization is therefore the single largest lever on the reported WER gap, and it rewards gigastt's output style: much of gigastt's lead over Vosk on clean read speech in the ITN-digit numbers is produced by the normalization itself, not by acoustics. (*naive* = lowercase + `ё`→`е` + strip everything outside `[a-zа-я0-9\s]` + split; *ITN* = the pipeline below.)

**This split is now reported on every run, not just the offline table above.** `benchmark.py` computes both passes per sample and emits, for each engine, the verbatim WER (`naive_wer`, `naive_ci_low`, `naive_ci_high`) alongside the normalized `wer`, plus `naive_delta = wer - naive_wer` (a negative delta means normalization — number style, punctuation, transliteration — closed the gap, not acoustics). The results table prints `naive %` and `Δ pp` columns next to the WER column. Because the verbatim pass applies no words-to-digits ITN or anglicism mapping and strips to the exact same `[a-zа-я0-9\s]` character class in both harnesses, the `naive_wer` numbers are directly comparable across the Python and Rust harnesses even though their ITN passes differ.

The normalization steps are:

1. Lowercase and replace `ё` with `е`.
2. Convert dashes/hyphens to spaces.
3. Tokenize letters (Latin or Cyrillic), digit sequences, and symbols/punctuation as separate tokens.
4. Convert Russian number-word sequences into Arabic digits, including cardinals, ordinals, compound numbers ("две тысячи двадцать" → `2020`), and scale words ("тысяча", "миллион").
5. Merge adjacent short digit groups (each ≤ 3 digits) into a single token, so phone numbers and chunked digit strings align.
6. Drop symbols (`+`, `№`, `%`, `$`, `-`, `€`, `₽`) and their spoken equivalents (`плюс`, `минус`, `номер`, `процент`, currency words, and wake-word artifacts such as `джой`).
7. Map common anglicisms to Russian tokens (e.g. `youtube` → `ютуб`).

Empty or whitespace-only references are skipped at load time by `load_manifest()`; `results.json` reports the count as `skipped_empty_refs`.

### Decode parameters

The following decode parameters are used so readers can reproduce the comparison exactly:

| Engine | Parameter | Value | Notes |
|---|---|---|---|
| gigastt | greedy beam search | beam width 1 | RNN-T greedy decode via ONNX Runtime |
| whisper.cpp | default CLI/server defaults | — | temperature 0, prompt none, language `ru` |
| faster-whisper | `beam_size` | 5 | CTranslate2, `language="ru"`, `compute_type="int8"` |
| Vosk | default Kaldi graph | — | `SetWords(False)`, 16 kHz mono 16-bit input |

### Failure handling

If a runner crashes or fails on a sample, that sample is counted as a 100% WER deletion of the reference (all reference words marked as errors). The per-runner `failures` counter and the top-level `total_failures` field in `results.json` make these cases visible instead of silently dropping them from the denominator.

### Confidence intervals

WER is reported with a bootstrap 95% confidence interval computed by resampling per-sample `(ref_words, errors)` pairs with replacement 1 000 times and taking the 2.5th and 97.5th percentiles. This mirrors the Rust CI implementation in `crates/gigastt/tests/benchmark.rs`.

### CI / Rust harness divergence note

The Rust CI harness in `crates/gigastt/tests/benchmark.rs` uses a simpler digit-to-words normalization. Its normalized WER numbers may therefore diverge from the Python benchmark on samples with digits, dates, or currency; this is tracked separately and is not part of this fix. The Rust harness also emits the verbatim (`naive_wer`, `naive_delta`) split in its JSON output and stderr summary; since the verbatim pass strips to the identical `[a-zа-я0-9\s]` character class in both harnesses, those naive numbers do line up.

### Dataset contamination

GigaAM v3 is a SberDevices model whose fine-tuning is dominated by Golos, and Common Voice / OpenSTT-style corpora are commonly part of Russian ASR training mixes. The Golos / OpenSTT / Common Voice slices used here therefore very likely overlap GigaAM v3's training distribution, so the in-domain WER should be read as a **best-case upper bound**, not a WER on unseen data. Golos ships an official train/test split (distribution overlap, not row-level leakage); the renormalized matrix below still shows Vosk ahead on clean read speech.

## Quick Start

```bash
cd benchmark
pip install -r requirements.lock.txt

# Run on 100 samples (default). First run transcribes; later runs read from cache.
python benchmark.py

# Run on full Golos crowd dataset (slow on first run, ~seconds once cached)
python benchmark.py --max-samples 0 --output results_full.json

# Run only specific engines
python benchmark.py --runners gigastt,whisper_cpp

# Use environment variable for limit
GIGASTT_BENCHMARK_MAX_SAMPLES=50 python benchmark.py

# Force a fresh run without using the cache
python benchmark.py --no-cache

# Clear cached transcription results
python benchmark.py --clear-cache

# Profile where time is spent (writes benchmark.prof)
python benchmark.py --profile --max-samples 10
python -m pstats benchmark.prof
```

On a 2024 MacBook Pro, 3 Golos crowd samples through `gigastt` take ~85 s on a cold cache and ~0.35 s once cached. Most of the wall-clock time is model inference; the cache eliminates that on repeat runs.

### Lockfile

`requirements.lock.txt` pins the full transitive dependency tree used by CI.
Regenerate it from `requirements.txt` with [uv](https://docs.astral.sh/uv/):

```bash
uv pip compile requirements.txt \
  --python-version 3.12 \
  --python-platform x86_64-manylinux_2_31 \
  --output-file requirements.lock.txt
```

## Tests

Run the benchmark unit tests with:

```bash
python -m pytest tests/ -v
```

## Docker (fully isolated)

If you prefer not to install Python dependencies locally, use the provided Dockerfile:

```bash
# Build image
docker build -f benchmark/Dockerfile -t gigastt-benchmark .

# Run benchmark with mounted model caches
docker run -v ~/.gigastt/models:/root/.gigastt/models:ro \
           -v ~/.gigastt/benchmarks:/root/.gigastt/benchmarks:ro \
           -v $(pwd)/benchmark/results:/workspace/benchmark/results \
           gigastt-benchmark \
           --max-samples 100 --runners all
```

Or use Docker Compose:

```bash
cd benchmark
GIGASTT_BENCHMARK_MAX_SAMPLES=100 docker-compose up
```

> **Note:** On macOS, Docker Desktop must be running. On Linux with NVIDIA GPUs, add `runtime: nvidia` to `docker-compose.yml` and use `--gpus all` with `docker run`.

## Datasets

The benchmark supports multiple Russian speech datasets. Use `--dataset <name>` to select one (default: `golos_crowd`).

### Golos crowd

The default **Golos crowd** test set (9 994 samples of Russian speech).

- **Source:** SberDevices
- **Repository:** https://github.com/sberdevices/golos
- **Paper:** Karpov et al., *Golos: Russian Dataset for Speech Research*, arXiv:2106.10161 (2021)
- **License:** Sber Public License (attribution/non-commercial/share-alike) — https://github.com/sberdevices/golos/blob/master/license/en_us.pdf

```bash
# Download and extract (one-time)
python ../scripts/extract_golos.py
```

### Golos crowd 1k

A deterministic 1 000-sample slice (`random.seed(42)`) of the Golos crowd test
set. Use this for cross-dataset comparisons so all domains have the same sample
size and comparable confidence intervals.

```bash
python benchmark.py --dataset golos_crowd_1k --max-samples 0
```

### Golos farfield

The **Golos farfield** test set (1 916 samples) recorded at a distance from the microphone.

- **Source:** SberDevices
- **Repository:** https://github.com/sberdevices/golos
- **Paper:** Karpov et al., *Golos: Russian Dataset for Speech Research*, arXiv:2106.10161 (2021)
- **License:** Sber Public License (attribution/non-commercial/share-alike) — https://github.com/sberdevices/golos/blob/master/license/en_us.pdf

```bash
# Download and extract (one-time), then create the committed 1 000-sample manifest
python ../scripts/extract_golos_farfield.py
```

Run the benchmark on the farfield slice:

```bash
python benchmark.py --dataset golos_farfield --max-samples 0
```

### Common Voice Russian

An alternative benchmark slice can be prepared from **Mozilla Common Voice** Russian (`ru`) test split.

- **Source:** Mozilla Common Voice contributors
- **Dataset:** https://huggingface.co/datasets/mozilla-foundation/common_voice_16_1
- **Project page:** https://commonvoice.mozilla.org/ru
- **License:** CC0-1.0

```bash
# Prepare a deterministic 1000-sample slice (one-time)
python ../scripts/prepare_common_voice_ru.py
```

Run the benchmark on the Common Voice slice:

```bash
python benchmark.py --dataset common_voice_ru --max-samples 0
```

### OpenSTT phone calls

An **OpenSTT** `asr_calls_2_val` validation slice (1 000 manually-annotated phone-call samples).

- **Source:** snakers4 / OpenSTT
- **Repository:** https://github.com/snakers4/open_stt
- **License:** CC BY-NC 4.0 — https://creativecommons.org/licenses/by-nc/4.0/

```bash
# Prepare a deterministic 1000-sample slice (one-time).
# The full archive is ~0.8 GB; use --use-unpacked-source to fetch only the
# selected 1000 wav+txt pairs instead.
python ../scripts/prepare_openstt_calls.py --use-unpacked-source
```

Run the benchmark on the OpenSTT phone-calls slice:

```bash
python benchmark.py --dataset openstt_calls --max-samples 0
```

### OpenSTT YouTube

An **OpenSTT** `public_youtube700_val` validation slice (1 000 manually-annotated YouTube samples).

- **Source:** snakers4 / OpenSTT
- **Repository:** https://github.com/snakers4/open_stt
- **License:** CC BY-NC 4.0 — https://creativecommons.org/licenses/by-nc/4.0/

```bash
python ../scripts/prepare_openstt_youtube.py --use-unpacked-source
```

Run the benchmark on the OpenSTT YouTube slice:

```bash
python benchmark.py --dataset openstt_youtube --max-samples 0
```

### Common Voice Russian

An alternative benchmark slice can be prepared from **Mozilla Common Voice** Russian (`ru`) test split.

- **Source:** Mozilla Common Voice contributors
- **Dataset:** https://huggingface.co/datasets/mozilla-foundation/common_voice_16_1
- **Project page:** https://commonvoice.mozilla.org/ru
- **License:** CC0-1.0

```bash
# Prepare a deterministic 1000-sample slice (one-time).
# Hugging Face may require accepting the dataset terms or setting HF_TOKEN.
python ../scripts/prepare_common_voice_ru.py
```

Run the benchmark on the Common Voice slice:

```bash
python benchmark.py --dataset common_voice_ru --max-samples 0
```

If the external dataset is missing, the benchmark falls back to the bundled fixtures (15 samples) from `crates/gigastt/tests/fixtures/`.

## Renormalized WER results

Existing result files were recomputed with the new symmetric words-to-digits normalization (`benchmark/recompute_wer.py`). The full 4×4 matrix below now includes the previously missing `openstt_calls` and `openstt_youtube` pairs, generated with the new normalization.

| Dataset | Engine | Old WER | Old CI | New WER | New CI | Δ WER |
|---|---|---|---|---|---|---|
| golos_crowd_1k | faster-whisper | 15.54 | 14.06–16.96 | 15.53 | 13.94–17.10 | -0.01 |
|  | gigastt | 10.77 | 9.17–12.16 | 8.60 | 7.51–9.66 | -2.17 |
|  | vosk | 4.57 | 3.82–5.33 | 4.82 | 4.03–5.60 | +0.25 |
|  | whisper.cpp | 15.80 | 14.34–17.26 | 15.26 | 13.74–16.71 | -0.54 |
| golos_farfield | faster-whisper | 16.31 | 14.71–17.89 | 17.34 | 15.62–19.07 | +1.03 |
|  | gigastt | 5.84 | 5.05–6.71 | 5.90 | 5.09–6.83 | +0.06 |
|  | vosk | 13.93 | 12.49–15.47 | 13.93 | 12.49–15.47 | -0.00 |
|  | whisper.cpp | 16.94 | 15.40–18.51 | 17.91 | 16.29–19.57 | +0.97 |
| openstt_calls | faster-whisper | 24.93 | 23.32–26.57 | 24.93 | 23.32–26.57 | -0.00 |
|  | gigastt | 19.28 | 17.88–20.67 | 19.28 | 17.88–20.67 | -0.00 |
|  | vosk | 38.57 | 36.72–40.64 | 38.57 | 36.72–40.64 | +0.00 |
|  | whisper.cpp | 32.73 | 30.69–34.91 | 32.73 | 30.69–34.91 | -0.00 |
| openstt_youtube | faster-whisper | 15.45 | 14.15–16.62 | 15.45 | 14.15–16.62 | +0.00 |
|  | gigastt | 11.35 | 10.32–12.31 | 11.35 | 10.32–12.31 | -0.00 |
|  | vosk | 20.65 | 19.38–21.98 | 20.65 | 19.38–21.98 | +0.00 |
|  | whisper.cpp | 22.61 | 20.97–24.20 | 22.61 | 20.97–24.20 | -0.00 |

### Residual errors

On `golos_crowd_1k` gigastt reaches 8.60% WER after renormalization (down from 10.77%) — the flagship number used in the README (1000 samples, 95% CI [7.51%, 9.66%]). The residual errors are dominated by:

- **Foreign brand / artist / product names** output in original Latin spelling by gigastt (and whisper) while the reference uses Russian transliteration, e.g. "Fashion TV" vs "фэшн ти ви", "Okko" vs "окко", "Bon Jovi" vs "бона джови". Roughly 45–50% of remaining error tokens fall in this category.
- **Real ASR errors or partial hypotheses**, including mis-heard words, substitutions, and truncated outputs on long digit strings. About half of the residual errors are genuine recognition mistakes rather than normalization mismatches.
- **Date/year format mismatches**, e.g. "двадцатый год" vs "2020". A small share (~1–2%).
- **Decimal/fraction numbers** not normalized, e.g. "три и два" vs "3,2". A small share (<1%).

No further normalization rules were added specifically to tailor results to gigastt; the pipeline remains symmetric across all engines. Concrete examples of the top residual errors are in [`results_full/residual_errors_gigastt_crowd.md`](results_full/residual_errors_gigastt_crowd.md).

## Output Format

`results.json` contains run metadata, per-engine summaries with failures and 95% CI, and per-sample details:

```json
{
  "manifest_samples": 100,
  "total_failures": 0,
  "runners": [
    {
      "name": "gigastt",
      "samples": 100,
      "failures": 0,
      "cached_hits": 100,
      "wer": 11.40,
      "ci_low": 10.9,
      "ci_high": 11.9,
      "rtf": 0.045,
      "total_errors": 57,
      "total_ref_words": 500,
      "details": [
        {
          "file": "00001.wav",
          "reference": "...",
          "hypothesis": "...",
          "wer": 0.0,
          "errors": 0,
          "ref_words": 5,
          "audio_sec": 3.5,
          "proc_sec": 0.15,
          "failed": false,
          "cached": true
        }
      ]
    }
  ],
  "metadata": {
    "collected_at": "2026-06-12T14:32:00+00:00",
    "host": { "cpu": "...", "ram_bytes": ..., "os": "...", "python_version": "..." },
    "dataset": { "name": "golos", "source": "...", "license": "...", "manifest_path": "..." },
    "engines": [ { "name": "gigastt", "version": "...", "model_sha256": "..." }, ... ]
  }
}
```

## Histograms

Each runner result includes WER breakdown histograms in `runners[*].histograms`:

| Dimension | Buckets | What it tells you |
|---|---|---|
| `audio_duration` | `0-5s`, `5-15s`, `15-30s`, `30s+` | WER by clip length — reveals whether the engine struggles with long-form audio. |
| `ref_words` | `1-5`, `6-15`, `16-30`, `30+` | WER by utterance complexity — short commands vs. long sentences. |
| `wer` | `0%`, `1-10%`, `10-20%`, `20-50%`, `50-100%`, `100%+` | Distribution of per-sample WER — shows how many samples are perfect, how many are catastrophic. |

Each bucket contains:

```json
{
  "bucket": "5-15s",
  "samples": 42,
  "ref_words": 315,
  "errors": 23,
  "wer": 7.30,
  "low_inclusive": 5.0,
  "high_exclusive": 15.0
}
```

Failed samples are counted in the `100%+` bucket because they are treated as 100% WER for that sample.

Example CLI output:

```text
--- Histograms: gigastt ---

audio_duration:
  Bucket            Samples    Words   Errors    WER %
  0-5s                   45      312       12     3.85
  5-15s                  42      315       23     7.30
  15-30s                 10       89        9    10.11
  30s+                    3       28        8    28.57
```

## CI / Automation

A GitHub Action runs the benchmark weekly (Sunday at 04:00 UTC) on `ubuntu-latest` and commits `results.json` to the `benchmark-results-local` branch. See `.github/workflows/benchmark.yml`.

### Badges

Add to your README:

```markdown
![WER](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fekhodzitsky%2Fgigastt%2Fbenchmark-results-local%2Fresults.json&query=%24.runners%5B0%5D.wer&suffix=%25&label=WER&color=blue)
![RTF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fekhodzitsky%2Fgigastt%2Fbenchmark-results-local%2Fresults.json&query=%24.runners%5B0%5D.rtf&suffix=x&label=RTF&color=green)
```
