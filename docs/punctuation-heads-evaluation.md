# Punctuation after real RNN-T recognition

This follow-up to the [native adapter experiment](rupunct-evaluation.md) compares
complete text results on the same 775 FLEURS Russian test recordings, including
built-in `e2e_rnnt` punctuation. It leaves product defaults unchanged: automatic
punctuation and ITN are enabled for `rnnt`, and disabled for `e2e_rnnt`.

## Results and recommendation

Do not replace the standard backend with native small on quality grounds from
this corpus: its actual `rnnt` output is essentially tied with ONNX. Native big
improves bare-head punctuation by **5.79 F1 percentage points**, at about **15.6%
more reconstructed serial time** than `rnnt` + ONNX. Keep it an explicit
quality/compute option. The earlier experiment's RAM advantage can still motivate
small for memory-constrained callers, but this run does not show better ASR-input
punctuation from small.

For `e2e_rnnt`, retain built-in formatting. Direct native restoration produces
duplicate marks, such as `времени..` and `семья,,`. Stripping punctuation first
avoids duplication but does not improve punctuation F1 here, even with big.
Big raises casing F1 slightly (77.72% to 78.50%) while adding about 19.1% serial
time and reducing punctuation F1. Reprocessing is not justified as an automatic
policy. Existing boot-policy tests already ensure `auto` skips the sidecar for
`e2e_rnnt`, including when no punctuation model is installed.

| Delivered text path | Punctuation F1, % | Casing F1, % | Added punctuation time, s | Reconstructed serial RTF |
|---|---:|---:|---:|---:|
| `rnnt` + ITN + ONNX small | 77.29 | 74.85 | 0.385 | 0.03164 |
| `rnnt` + ITN + native small | 77.21 | 74.73 | 1.419 | 0.03176 |
| `rnnt` + ITN + native big | 83.08 | 78.33 | 44.744 | 0.03658 |
| `e2e_rnnt`, built-in | **83.40** | 77.72 | 0 | 0.02744 |
| `e2e_rnnt` → native small directly | 58.37 | 76.36 | 1.918 | 0.02766 |
| `e2e_rnnt` → native big directly | 63.11 | 77.99 | 48.912 | 0.03288 |
| `e2e_rnnt` → clean → ONNX small | 73.71 | 71.61 | 0.388 | 0.02749 |
| `e2e_rnnt` → clean → native small | 77.43 | 75.21 | 1.434 | 0.02760 |
| `e2e_rnnt` → clean → native big | 82.97 | **78.50** | 47.173 | 0.03269 |

Every postprocessor preserves the alphanumeric content of its exact input on all
775 files. These are formatting comparisons; native punctuation does not repair
missing or misrecognized words. Small differs from ONNX on only 47 of the bare
head's 775 delivered sentences. Its extra punctuation time is about 1.03 seconds
across the 284-second decode pass, around 0.36% of reconstructed total time.
This does not supersede the previous concurrency/CPU-contention measurements.

| Recognition head | Raw WER, % | Common existing ITN WER, % | Common ITN WER on 620 digit-free references, % | Decode time, s | Decode peak / final RSS, MiB |
|---|---:|---:|---:|---:|---:|
| `rnnt` | 7.16 | 5.55 | 3.44 | 284.22 | 876.6 / 497.0 |
| `e2e_rnnt` | 4.51 | 4.36 | 3.22 | 246.83 | 876.7 / 498.6 |

`e2e_rnnt` wins delivered punctuation and written-form WER in this specific
read-speech run. Much of the WER gap is sensitive to number conventions; on
references without digits it is much smaller. This is not evidence for changing
the global default head or for a four-domain/spontaneous-speech WER ranking.
Both heads peak near 877 MiB in this run. The samples include startup; the
location of that peak was not isolated. They cannot be compared to the earlier
steady-state pool-1 figures as evidence of a model RAM regression. A larger live request/concurrency campaign is required
before changing release defaults.

The operational rule should follow model capabilities: apply an optional external
punctuator to output lacking punctuation; retain native formatting for models that
provide it. This run qualifies only the two Russian RNN-T heads on this corpus.
Multilingual CTC languages, conversational speech, ARM/macOS and concurrent full-ASR
workloads need their own evidence before enabling a new backend there.

Raw hypotheses, commands and per-file timing:
[measurement directory](measurements/punctuation-heads-20261022/),
[structured results](measurements/punctuation-heads-20261022/summary.json),
[protocol and audio/executable hashes](measurements/punctuation-heads-20261022/protocol.json),
[model/source identities](measurements/punctuation-heads-20261022/models-and-source.json),
[local checks](measurements/punctuation-heads-20261022/validation.json).

## Why the earlier small-model gain does not transfer directly

On the gold-input experiment, native small's gain is concentrated in the 98
hyphen-bearing inputs: punctuation F1 is 62.60% versus ONNX 35.58%. On the other
677 inputs it is 80.91% versus 80.45%. **None of the actual `rnnt` + ITN inputs
in this run contains a hyphen or any other non-word punctuation character.**
Restoring punctuation on a normalized human reference is therefore a different
input task from restoring this recognizer's output.

There is also an existing baseline alignment limitation: ONNX restoration uses
whitespace-word spans, whereas its BERT tokenizer assigns word IDs to punctuation
splits as well. Applying those IDs to whitespace-word labels can shift labels
after internal punctuation. This needs a focused correction and regression test;
these gold-input differences must not be interpreted as native arithmetic alone
improving accuracy. The bare `rnnt` inputs in this comparison avoid that boundary
mismatch. The optional ONNX reprocessing of cleaned `e2e_rnnt` can still encounter
hyphens, so its score is a historical-baseline diagnostic rather than proof of an
intrinsic native quality advantage.

## Protocol and interpretation

The frozen manifest is `benchmark/manifests/fleurs_ru_punct.json`, from
[google/fleurs, ru_ru test](https://huggingface.co/datasets/google/fleurs),
CC BY 4.0, Conneau et al. (2022). The 775 files contain 8,993.82 seconds of
read speech. This is not a qualification on spontaneous conversations, and
checkpoint training overlap cannot be ruled out.

Both heads are selected explicitly from the same model directory. Each file is
decoded once per head, with pool size 1, four encoder threads, and CPUs 0–3 on
the same AMD Ryzen AI 9 HX 370 Linux host. No VAD, lexicon, hotwords or external
punctuation participates in this decode. The resulting hypotheses are frozen.
`rnnt` text then receives gigastt's existing ITN, as the product default does,
before ONNX small or native small/big restoration. `e2e_rnnt` is evaluated as-is,
then with native restoration applied directly, and with restoration after
lowercasing/removing sentence punctuation. The latter uses the same `asr_text`
normalizer as the earlier experiment, preserving internal hyphens and decimal
dots. Native GEMM has `RAYON_NUM_THREADS=1`; adapter window workers remain 1.

Word Levenshtein alignment uses lowercase alphanumeric tokens and normalizes
`ё` to `е`. Punctuation events are anchored after their preceding word, including
repeated marks and initial quotation marks. Substituted words can retain aligned
word-end punctuation; marks on deleted/inserted words count as unmatched.
Capitalization events match at character offsets inside lexically equal aligned
words. Missing, duplicate and unexpected hypotheses fail evaluation. This avoids
the global character-position shifts in the older head-comparison benchmark.

Full punctuation/casing F1 therefore includes recognition failures. Additional
`matched_*` metrics condition on lexically equal aligned word anchors; they are
diagnostics, since each head can have a different set of correct words. Neither
metric is directly interchangeable with the earlier gold-input character-boundary
F1 or the library README's word-label metric. Alignment ties use deterministic
diagonal, deletion, insertion precedence. Orthographic variants, hyphenation,
numbers and ambiguous alignment can still affect the result.

WER ignores case and punctuation. Report raw WER and a sensitivity comparison
applying **the same existing gigastt ITN to both reference and hypothesis**.
ITN is heuristic, not an authoritative spoken/written-number normalizer: for
example, a recognized ordinal can become `1900 девяностом` rather than `1990`.
These WER values measure delivered textual conventions as well as recognition;
they cannot attribute every numeric difference to acoustic accuracy. Postprocessor
lexical preservation is checked separately against its exact frozen input.

Decode timing includes file decode and recognition. Punctuation timing is measured
in separate sequential processes, with warm filesystem caches; the first call is
included. Adding the two components gives a **reconstructed serial RTF**, not a
full-server latency or concurrency measurement. One corpus pass per head does not
establish a statistically significant speed ordering. Decode peak RSS and
punctuation-only peak RSS are reported separately and must not be added to infer
combined Engine memory. Prior integrated memory measurements remain in the
[adapter report](rupunct-evaluation.md).

## Reproduction

Provision the same model assets and the FLEURS audio locally; these commands do
not download models or dispatch CI. Use an empty output path.

```sh
cargo build --release --locked -p gigastt-core --features rupunct \
  --example punctuation_heads_probe --example punctuation_probe
python3 scripts/compare-punctuation-heads.py \
  --decode-probe target/release/examples/punctuation_heads_probe \
  --restore-probe target/release/examples/punctuation_probe \
  --models ~/.gigastt/models --big ~/.rupunct/RUPunct_big \
  --audio-root ~/.gigastt/benchmarks/fleurs_ru --output /tmp/punctuation-heads
python3 scripts/summarize-punctuation-heads.py /tmp/punctuation-heads
python3 scripts/test-punctuation-heads.py
```

Both Python scripts use the standard library. The comparison runner records
manifest/audio/executable hashes and commands before executing them, refuses to
overwrite an existing output directory, and fails on a failed inference process.
ASR rows are flushed individually so a partial run is visibly incomplete.
Scoring tests are included in the existing manual CI workflow; no automatic
trigger was introduced.
