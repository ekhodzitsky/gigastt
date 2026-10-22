# Current GigaAM recognition heads: accuracy, speed and formatting

Compare current Russian v3 RNN-T / E2E RNN-T / CTC / E2E CTC and the two
multilingual CTC heads. Obsolete v1/v2 heads are excluded. Model selection must
consider recognition errors and resources, independently of whether punctuation
is built in. Product defaults are unchanged.

## Artifact identity is a prerequisite

The accepted RNN-T assets are the active `models-v3-2026-10-09` quality bundle:

| Encoder | Active SHA-256 | Size, bytes |
|---|---|---:|
| `v3_rnnt_encoder_int8.onnx` | `1d5a6f580b692e38ba35ed98d0f54648808f27b0a147dc11bd8978c15dc1a18f` | 319665270 |
| `v3_e2e_rnnt_encoder_int8.onnx` | `16805903b102d8b27bb0253de044f25e33a8a5bf07fce0c4da4ac1a5a7544091` | 319665270 |

An initial local attempt used older cached encoders (`c52665e9` / `cf51b300`,
225250603 bytes each). That attempt is excluded. The earlier RUPunct reports'
audio/Engine measurements also used those cached encoders; their isolated text
restoration measurements are independent of ASR weights, but their integrated
latency and actual-recognition formatting scores are historical.

Successful Engine loading does not establish release identity: custom locally
quantized weights are intentionally supported, including canonical file names.
The comparison now obtains the four registered heads' authoritative checksum
catalog from the compiled probe and verifies every asset before preparing data
and before starting each run phase. A real stale-cache attempt is rejected before
creating an output directory. The regression test covers matching and mismatching
assets. Runtime custom-pack compatibility remains intact.

The two additional CTC models are provisioned through existing model manifests
with the generic CTC architecture, not new named CLI/download variants. Their
upstream snapshot is `istupakov/gigaam-v3-onnx` at revision
`322c3b29492673eb7d0b434bfa9dfb8653e34d02`; INT8 model hashes match the snapshot's
LFS SHA-256 values. The comparison validates the single-output logits layout and
preserves text in BPE word-boundary tokens. Independent `onnx-asr 0.12.0` agrees
on words and punctuation for nine smoke recordings per new head; E2E CTC has one
quotation-spacing difference.

## Protocol

Every head receives the same frozen local recordings:

| Corpus | Included / manifest rows | Audio seconds | Domain |
|---|---:|---:|---|
| FLEURS Russian test | 775 / 775 | 8993.82 | Read speech, human punctuation |
| Golos crowd | 992 / 1000 | 3927.57 | Close-microphone read/command speech |
| Golos far-field | 1000 / 1000 | 2623.30 | Distant microphone / commands |
| Podlodka | 67 / 67 | 1494.80 | Conversational podcast excerpts |

All 2834 included files have distinct audio hashes. Eight missing crowd files
are explicitly enumerated, shared exclusions for every head; they are not guessed
from transcripts or replaced. This is not the historical 1000-file crowd cohort.
The missing files, source manifest hashes, dataset provenance and licenses are
retained in the protocol. No audio is republished. This campaign does not qualify
telephone audio, full meetings, live WebSocket quality or other languages.

All inference uses the same Rust release probe, CPU ONNX Runtime 1.28.0
(`ort = 2.0.0-rc.13`), one pool slot, four encoder threads, and default serial file
windowing. VAD, hotwords, lexicon, external punctuation and product ITN are off.
Raw hypotheses are frozen; additional ITN annotations are produced after the
recognition timer ends. E2E formatting is intrinsic to those heads. The timer
wraps `Engine::transcribe_request(Path)`, including file reading, audio decode,
resampling, features and ASR decoding; it is not encoder-only kernel time.

Speed is measured **serially on CPUs 0–3**, the four fast physical cores of the
AMD Ryzen AI 9 HX 370. Forty fixed files cover ten duration quantiles per corpus
(430.69 seconds total). Each process loads and warms on two speech files before
measurement. Round zero conditions model/optimized caches; six subsequent rounds
rotate model order through all six positions. Report median weighted RTF
(`sum inference seconds / sum audio seconds`) and its observed min–max, separate
warm-cache load times, RSS after warmup and process peak RSS. The first
conditioning round is retained but excluded from steady-state summaries. These
are single-request CPU results, not concurrent server capacity, GPU or macOS
measurements. Unrelated host activity is not completely excluded.

Full-corpus accuracy can run concurrently on disjoint physical CPU sets
`0–3`, `4–7`, `8–11`; these include different core classes. **Its observed wall
times are not used for speed ranking.** All timing/accuracy inputs retain exact
hashes, and transcripts shared with the serial timing cohort are checked for
consistency. Process CPU time is deliberately omitted for parallel quality jobs
because child-process aggregate counters would double-count overlapping jobs.

The active RNN-T packages keep 50 convolutions in floating point and dynamically
quantize 128 matrix multiplications. Available upstream CTC packages quantize
convolutions too: 51 integer convolutions for the three smaller CTC exports and
75 for multilingual large. Therefore results compare available INT8 packages,
not pure decoder architecture at identical quantization. Do not infer that CTC
intrinsically has the measured speed/accuracy ranking. A matched selective-
quantization ablation is needed before ruling out a head's potential.

## Metrics and interpretation

Lexical WER lowercases, folds `ё` to `е`, splits at non-alphanumeric boundaries,
and retains digits. CER uses the corresponding space-separated character stream.
Neither counts punctuation/casing as recognition errors. Also report the existing
benchmark's symmetric normalized WER: it canonicalizes Russian number words,
digit groups/ordinals, anglicisms and its existing ignorable wakeword/symbol set.
That convention can hide real delivered-text differences, so raw lexical WER/CER
remain visible. Neither score alone isolates acoustics from writing conventions.
Both digit-free and numeral-free sensitivity subsets are reference-selected,
not selected by model success. The latter excludes digits and any token in the
existing benchmark numeral inventory (including case forms and ordinals); it is
a diagnostic slice, not a complete detector of semantic quantities. Missing/duplicate/unknown outputs and incomplete reports fail
validation; failed inference requests remain as empty hypotheses/deletions.

Confidence intervals are descriptive 1000-replicate bootstraps grouped by identical
lexical reference text (seed 42), not proof of independent speakers or unseen-domain
performance. Corpora have different domains and normalization effects; they are
not pooled into a universal accuracy percentage. Small score differences do not
by themselves establish model superiority.

Built-in punctuation/casing on FLEURS uses the same tested word-alignment metric
as the earlier head audit, separately from WER. External RUPunct restoration does
not repair recognition errors. Its active-weight replay is reported below. Earlier cached-weight integrated
numbers cannot justify a release decision.

## Results on the verified active packages

All six heads completed every included recording with **zero inference errors**.
All 40 serial timing transcripts are stable across seven rounds and match their
full-cohort accuracy transcripts. Training overlap with these public corpora is
not established; this is a reproducible local comparison, not a held-out claim.

The four Russian ASR heads and the E2E formatting distinction follow the
[upstream model card](https://huggingface.co/ai-sage/GigaAM-v3).
The SSL pretraining encoder is not a ready-to-use ASR head and is excluded.

### Recognition, independently of punctuation

Lexical WER (%), lower is better. E2E number/brand spelling affects these scores:

| Head | FLEURS | Crowd | Far-field | Podcast |
|---|---:|---:|---:|---:|
| rnnt | 7.13 | 2.77 | 3.96 | 7.62 |
| e2e_rnnt | 4.51 | 14.20 | 6.35 | 6.51 |
| v3_ctc | 7.35 | 2.75 | 4.76 | 7.86 |
| v3_e2e_ctc | 4.76 | 14.80 | 6.72 | 7.92 |
| ml_ctc | 8.17 | 5.77 | 8.12 | 8.95 |
| ml_ctc_large | 6.74 | 4.31 | 5.71 | 8.16 |

Existing benchmark-normalized WER (%) on the **same complete cohorts**:

| Head | FLEURS | Crowd | Far-field | Podcast |
|---|---:|---:|---:|---:|
| rnnt | 4.63 | 2.92 | 4.23 | 7.11 |
| e2e_rnnt | 4.29 | 8.43 | 5.72 | 6.32 |
| v3_ctc | 4.86 | 2.94 | 5.04 | 7.36 |
| v3_e2e_ctc | 4.50 | 9.19 | 6.19 | 7.82 |
| ml_ctc | 5.65 | 6.09 | 8.67 | 8.41 |
| ml_ctc_large | 4.25 | 4.54 | 6.09 | 7.63 |

Reference-selected numeral-free lexical WER (%), with common counts
**524 / 818 / 967 / 34** respectively:

| Head | FLEURS | Crowd | Far-field | Podcast |
|---|---:|---:|---:|---:|
| rnnt | 3.54 | 3.31 | 4.01 | 7.12 |
| e2e_rnnt | 3.25 | 7.75 | 5.35 | 6.47 |
| v3_ctc | 3.57 | 3.15 | 4.84 | 7.18 |
| v3_e2e_ctc | 3.30 | 8.49 | 5.82 | 8.09 |
| ml_ctc | 4.34 | 6.45 | 8.43 | 7.83 |
| ml_ctc_large | 2.95 | 4.84 | 5.89 | 7.76 |

This control reduces the E2E crowd gap, but does not remove it. Brand spelling,
transliteration and compound boundaries still differ. Podcast's 34-file control
is small; do not treat it as another independent benchmark. Full lexical and
normalized CER, S/D/I counts, empty outputs, digit-free slices and descriptive
intervals are in [summary.json](measurements/current-asr-20261022/summary.json).

#### Why E2E crowd WER needs inspection

The 14.20% E2E RNN-T lexical crowd score is a **delivered-string error rate**,
not a count of wrongly heard words. The fixed normalizer reduces it to 8.43%,
but cannot canonicalize every valid brand/unit/currency rendering. No aliases
were added to favor a particular model. Examples from the diagnostic slice:

| Reference excerpt | E2E RNN-T output excerpt | Interpretation |
|---|---|---|
| `пепси два с половиной литра` | `Pepsi 2,5 л.` | Valid normalization/Latin spelling inflates lexical WER |
| `зэ смокерс мэгазин` | `The Smokers Magazine` | Transliteration versus English name |
| `один два два три девять два восемь шесть три ноль` | `1 2 39 286 3 0` | Number grouping plus an omitted `2`; not all differences are cosmetic |
| `фэшн ти ви четыре ка` | `Fashion TV четыре копейки.` | Semantic error: 4K interpreted as currency |

An independent `onnx-asr 0.12.0`/ORT 1.30.0 decoder reproduces E2E RNN-T's words
on all **16 deliberately difficult crowd cases**, using both the native greedy
budget of 10 and upstream default of 3. Exact strings match 14/16: one currency
spacing difference and one quotation difference remain. This supports the
word-level diagnosis; it does not establish full-corpus implementation parity.
For E2E CTC the same difficult slice agrees lexically in 14/16, with two small
word/digit discrepancies. Their cause (frontend/runtime numerical differences
versus adapter behavior) is unresolved; do not attribute every CTC error to the
weights. Both runtimes reproduce the `4 копейки` semantic error. The initial nine
CTC smoke cases remain as an additional, less adversarial check.

### Single-request CPU speed and memory

RTF 0.0165 means about 1.65 seconds of timed inference per 100 seconds of audio
for this workload. It excludes model loading, external formatting and transport.
Warm load is reported separately. RSS counts mapped memory; it is not macOS
physical footprint. Peak is the maximum process high-water RSS of steady runs,
including startup and warmup, not an incremental inference allocation.

| Head | Median RTF | Observed range | Ready RSS, MiB | Peak RSS, MiB | Warm load, s |
|---|---:|---:|---:|---:|---:|
| rnnt | 0.01655 | 0.01633–0.01731 | 355.9 | 503.4 | 0.325 |
| e2e_rnnt | 0.01691 | 0.01654–0.02019 | 363.3 | 489.9 | 0.320 |
| v3_ctc | 0.02308 | 0.02289–0.02352 | 277.3 | 439.6 | 0.302 |
| v3_e2e_ctc | 0.02286 | 0.02159–0.02358 | 266.7 | 440.9 | 0.297 |
| ml_ctc | 0.03018 | 0.02794–0.03120 | 266.0 | 505.1 | 0.294 |
| ml_ctc_large | 0.06650 | 0.06556–0.07341 | 636.3 | 1009.1 | 0.656 |

E2E RNN-T's median is about 2% above plain RNN-T, with overlapping observed
ranges; this is not evidence of a reliable 2% penalty. Available v3 CTC packs
cost about 38–40% more inference time but save roughly 79–89 MiB ready RSS.
Multilingual large costs about 4.0 times the RNN-T inference time. Its FLEURS
advantage does not generalize to the other Russian domains measured here.
Multilingual usefulness for other languages is outside this Russian-only test.

### Active-weight RUPunct replay

FLEURS, 775 active-weight hypotheses, rupunct-rs 0.19.1. Ordinary RNN-T inputs
receive existing product ITN before restoration; E2E keeps its built-in output.
Punctuation F1 is end-to-end word-aligned, not a recognition score:

| Path | Punctuation F1 | Casing F1 | Reconstructed RTF |
|---|---:|---:|---:|
| RNN-T + current ONNX | 0.7733 | 0.7510 | 0.01660 |
| RNN-T + RUPunct small | 0.7731 | 0.7495 | 0.01674 |
| RNN-T + RUPunct big | 0.8306 | 0.7841 | 0.02258 |
| E2E RNN-T builtin | 0.8355 | 0.7770 | 0.01691 (ASR only) |

Reconstructed RTF adds independently timed ASR and restoration on the **same 40
files**, across six steady rounds. It excludes ITN cost, model load, API handling and
export; it is not an integrated-server benchmark. Punctuation uses one Rayon
thread and one window worker on CPUs 0–3. Its own process peak RSS is about
91 MiB for ONNX, 59 MiB for small and 198 MiB for big; these cannot simply be
added to ASR RSS to predict an integrated process.

Big improves RNN-T punctuation F1 from 0.7733 to 0.8306, near builtin E2E RNN-T's
0.8355, while raising the reconstructed ASR+punctuation time about 36% relative
to ONNX. Small's F1 is essentially unchanged and the total time is about 0.8%
higher than ONNX in this configuration. No restoration path changed its input's
alphanumeric content. The standalone ONNX hyphen-token alignment defect tracked
in the earlier audit is not fixed here and remains a separate limitation.

Passing already punctuated E2E output directly through RUPunct is harmful here:
F1 drops to 0.5846 (small) / 0.6320 (big). Stripping formatting first reaches
0.7753 / 0.8294, below keeping builtin punctuation, and the cleaning step changes
lexical token boundaries (FLEURS lexical WER 4.51% to 4.73%). This does not justify
repunctuating E2E output by default.

### Selection outcome and remaining work

Keep ordinary RNN-T as the general default for Russian recognition: it is the
fastest measured package, competitive on commands and strongest on the measured
far-field cohort. E2E RNN-T
is a useful alternative when normalized readable text matters: lower delivered
WER on FLEURS/podcast, strong builtin punctuation and similar CPU cost, with
specific command/numeric semantic errors requiring evaluation on the intended
workload. Do not replace it with E2E CTC merely because both offer punctuation.

RUPunct big is an explicit quality/resource trade-off for unpunctuated heads,
not a demonstrated universal improvement. There is no evidence here to promote
small as the default. The CTC comparison should next isolate selective
quantization before interpreting its architecture's potential. None of these
results changes product defaults or starts automatic CI.

All raw transcripts, per-request timings, model/audio hashes, exclusions,
operator counts and oracle outputs are frozen under
[measurements/current-asr-20261022](measurements/current-asr-20261022).
No model weights, optimized graph caches or audio are committed. The excluded
stale-cache attempt is described in validation metadata, not mixed into results.

## Verification

Local workspace unit/bin tests pass in both configurations: **1240** default and
**1243** with `rupunct`. Strict workspace/all-target Clippy passes for both.
The new scorer/gate tests pass (10), as do the existing punctuation-head (8) and
gold-fixture tests (4); formatting and documentation drift checks pass. Regression
coverage includes BPE text retention/alignment, single-output CTC shape/length
validation, active-model preflight before inference, failed-request denominators,
incomplete evidence and reference-only numeral controls. Exported measurements
recompute to the byte-identical summary. No GitHub Actions run was requested.

## Reproduction

Provision the exact pinned source assets and dataset files locally. The scripts
do not download models or run GitHub Actions. Use an empty output directory.

```sh
cargo build --release --locked -p gigastt-core --features rupunct --example current_asr_probe
python3 scripts/prepare-current-asr.py \
  --probe target/release/examples/current_asr_probe \
  --models /path/to/verified-active-models --v3 /path/to/pinned-v3-ctc-snapshot \
  --output /tmp/current-asr
python3 scripts/run-current-asr.py /tmp/current-asr \
  --probe target/release/examples/current_asr_probe --phase timing
python3 scripts/run-current-asr.py /tmp/current-asr \
  --probe target/release/examples/current_asr_probe --phase quality \
  --quality-cpus 0-3 4-7 8-11
python3 scripts/score-current-asr.py /tmp/current-asr
python3 scripts/test-current-asr.py
cargo run -p gigastt-quantize --example inspect_ops -- /path/to/encoder.onnx
```

Omit `--quality-cpus` for serial accuracy on the timing CPU allocation. The
snapshot directory includes `download-protocol.json`, frozen model files and
vocabularies; its per-file hashes and upstream revision are retained with the
measurements. No new production dependency was introduced. Scoring/asset tests
join the existing manually dispatched CI workflow; automatic triggers remain off.
