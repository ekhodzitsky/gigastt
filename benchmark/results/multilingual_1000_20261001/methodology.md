# Five-language, 1,000-recording evaluation methodology

This evaluation measures the two **original, unadapted INT8 multilingual
models through gigastt** on English, Russian, Kazakh, Kyrgyz and Uzbek read
speech. It does not train, adapt, select or replace model weights. Earlier
experimental CTC-head checkpoints are excluded.

## Models and language scope

The [publisher's model card](https://huggingface.co/ai-sage/GigaAM-Multilingual)
identifies these five ASR languages. Its 70+ language statement concerns
self-supervised pretraining, not ready-to-use ASR support for 70+ languages.
The card distinguishes the smaller and larger CTC models and describes
English quality as moderate. Those are publisher claims; this evaluation
measures the actual product paths independently.

The following installed file hashes were independently verified before the
benchmark and match the project's model identity. The baseline is the shipped
INT8 model, not an FP32 source checkpoint or a trained head from the preceding
research pilot.

| Variant / file | SHA-256 |
|---|---|
| `ml_ctc` / `multilingual_ctc.int8.onnx` | `e08e27ae5669b39f0c378fae101bbbb9a80505f74f9b66719c309bf5b894a480` |
| `ml_ctc_large` / `multilingual_large_ctc.int8.onnx` | `b2ad9c38fc04197ba758105d33f7404fd13d977958722e0f49e3f3e22521f1c6` |
| Shared `multilingual_vocab.txt` | `4d130287892e1099fedfb3f93c4b4cf8a263151158801680b28977d1be4133f4` |

The vocabulary contains 71 classes, including blank and a space marker.
It includes basic Latin letters, ASCII apostrophe and Cyrillic letters needed
for Russian, Kazakh and Kyrgyz. It has no decimal digit tokens. No automatic
transliteration, number expansion, spelling repair or language-specific text
replacement is applied to improve the scores. Characters outside the
vocabulary remain in reference scoring.

## Corpus and selection

Use the official [Google FLEURS dataset](https://huggingface.co/datasets/google/fleurs),
attributed to Conneau et al., *FLEURS: Few-shot Learning Evaluation of Universal
Representations of Speech* (2022), CC BY 4.0. It is human read speech with
parallel sentence material across languages; several recordings may read the
same sentence. Recording count must not be described as speaker count or
unique-sentence count. The publisher distinguishes training speakers from
development/test speakers and cautions that read-speech results may not
represent noisy production speech.

Configurations are `en_us`, `ru_ru`, `kk_kz`, `ky_kg` and `uz_uz`. Each language
uses 1,000 distinct recordings from **test plus validation only**. No training
split is needed or permitted for this evaluation. The original test partitions
contain fewer than 1,000 recordings, so this is not an official test-only score.
The pinned parquet revision is `168de341b3db6859a9bac1c50a2ef5e3b47647e0`.
Selection takes all eligible test rows in source order, then tops up to 1,000
from eligible validation rows shuffled by `random.Random(42)`. Reject empty
references, nonpositive durations and duplicate audio hashes. For Kazakh,
exclude sentence IDs and original-audio hashes used in the preceding
adaptation pilot's validation set before selection. This rule is fixed before
transcription and identical for both models; manifests preserve split, source
row, sentence ID, audio hash, dataset revision and reference text.

The official dataset viewer size API reported these raw counts when checked;
the extracted manifests are authoritative for selected counts and exclusions:

| Language | Test available | Validation available | Validation needed after all test rows |
|---|---:|---:|---:|
| English | 647 | 394 | 353 |
| Russian | 775 | 356 | 225 |
| Kazakh | 856 | 369 | 144 |
| Kyrgyz | 977 | 422 | 23 |
| Uzbek | 862 | 363 | 138 |

Use configuration/split/source-row identifiers for sample uniqueness: the
FLEURS `id` field repeats across recordings of the same sentence. Verify audio
hash uniqueness and disclose any invalid source rows rather than silently
replacing recognition failures. Report selected test and validation counts
separately, along with both split-level and combined metrics. Validation is
included to meet the fixed sample count; it is not used to tune either model.
Some recordings were previously inspected in the preceding research, so the
combined set must not be called a newly blinded benchmark.

## Reference and scoring protocol

The primary reference is the dataset's `transcription` field. Retain
`raw_transcription` as provenance if desired; do not select whichever reference
scores better. `benchmark_multilingual_public.score` and
`scripts/wer_unicode.py::normalize` supply the established, fixed scorer.
Apply the same transformation to reference and hypothesis:

1. Lowercase.
2. Delete the existing list of apostrophe-like Unicode characters.
3. Retain Unicode letters/digits and whitespace; delete other characters.
4. Collapse whitespace by tokenization and rejoin tokens with a single space.

This preserves Kazakh/Kyrgyz-specific letters, unlike an ASCII/Russian-only
regular expression. It is nevertheless **orthography-normalized scoring**,
not raw verbatim accuracy. Important limitations are disclosed rather than
silently changing the legacy scorer during the run:

- Apostrophes are removed entirely, including the distinction between Uzbek
  `oʻ` and `o`, and between `gʻ` and `g`. Thus apostrophe typography does not
  create errors, but missing distinctions can also be forgiven. English
  contraction apostrophes are removed too.
- Hyphens/dashes are deleted, potentially joining words. For example,
  `бір-екі` becomes one token while `бір екі` remains two.
- Russian `ё` is not folded to `е`.
- Unicode NFC normalization is not performed. A precomposed letter and a
  decomposed equivalent can therefore differ after combining-mark deletion.
  Reference character auditing should disclose non-NFC inputs if present.
- Digits remain digits. A spoken-number hypothesis can differ from a
  digit-bearing reference even when the acoustic content is correct.

Primary **micro WER** is total substitutions + deletions + insertions divided
by total reference words; **micro CER** is the analogous ratio for characters.
CER includes inter-word spaces after normalized tokens are rejoined. Do not
average individual recording percentages. WER may exceed 100% due to insertions.
Empty successful hypotheses remain in the denominator and are counted.

Report a secondary **digit-free-reference subset** for each language/model:
exclude from that subset any reference containing a character for which
`str.isdigit()` is true, before scoring. This matches the existing scorer's
definition. Do not substitute Unicode category `Nd` without recording a method
change, and do not drop digit-bearing samples from the primary 1,000-recording
score. Give the subset size and denominator. No number-to-word conversion is
introduced after examining recognition outputs.

Also report word-error components, total reference words/characters, duration,
empty transcripts and request failures. The runner's fixed policy performs no
retries: a failed request is retained with `status=failed`, its error recorded,
and an empty hypothesis contributes deletions to the operational full score.
That is not a successful empty transcription. Report failures separately and
do not claim 1,000 successful transcriptions or a model-only accuracy score if
requests failed. No recognition result is silently excluded.

The requested 95% intervals use **utterance bootstrap**: 1,000 resamples with
replacement of the 1,000 scored recording pairs `(reference_words, errors)`,
seed 42, recomputing micro WER in each resample. The interval uses the 2.5th
and 97.5th percentile positions as in `wer_unicode._wer_ci`. These are
conditional descriptive intervals for this selected set, not independent-
speaker confidence intervals or estimates covering the language generally.
FLEURS repeats sentence content and does not provide a simple independent-
speaker identifier in these rows. Compare the two models on matched audio;
do not draw strong conclusions from a small numerical difference alone.

## Comparability and runtime scope

The publisher's [reported WER protocol](https://huggingface.co/ai-sage/GigaAM-Multilingual#model-performance)
excludes digit-bearing references and utterances over 30 seconds and uses its
own normalization. This benchmark retains the fixed 1,000-recording selection,
uses the product's INT8 path and the existing scorer, and combines test with
validation. It is **not a reproduction of the published model-card numbers**.
The digit-free subset alone does not remove all those differences.

Evaluate original files through the file/REST product path with punctuation
and ITN off and with the variant, model hashes and actual server configuration
recorded. Native product preprocessing is part of the measured pipeline.
No trained head, FP32 control or alternate recognizer belongs in these results.
VAD, threading, pool size and runtime/backend choices must be the same for the
paired models unless a difference is explicitly reported.

These are read-speech accuracy results. They do not establish telephone,
code-switching, spontaneous-conversation or live WebSocket accuracy. Timing
under other concurrent work is not a controlled throughput comparison;
report the environment and contention if giving RTF. Code/scorer hashes,
binary identity, manifests and the complete per-recording results provide
the basis for independent recomputation.
