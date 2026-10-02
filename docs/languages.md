# Languages and measured accuracy

The default `rnnt` model recognizes Russian. Select a multilingual CTC head for
English, Kazakh, Kyrgyz or Uzbek. Among the evaluated configurations,
`ml_ctc_large` is the recommended starting point for accuracy. This is a model
selection recommendation, not a guarantee of transcript quality.

```sh
gigastt download --model-variant ml_ctc_large
gigastt transcribe recording.wav --model-variant ml_ctc_large --punctuation off --itn off
gigastt serve --model-variant ml_ctc_large --punctuation off --itn off
```

CTC recognizes the supported languages without a language selector. The
Russian punctuation and number-normalization stages are disabled in these
examples, matching the quality evaluation. The global Russian default has
not changed. Both multilingual heads use existing INT8 models; no training
or fine-tuning was performed.

## Reading: 1,000 recordings per language

Micro word error rate (WER), lower is better. Each model receives the same
1,000 distinct FLEURS recordings per language: eligible test rows followed
by a seeded validation top-up. This is not the official test-only benchmark.

| Language | `ml_ctc` WER | `ml_ctc_large` WER |
|---|---:|---:|
| Russian | 9.97% | 8.78% |
| English | 16.21% | 13.62% |
| Kazakh | 12.32% | 11.47% |
| Kyrgyz | 13.68% | 12.34% |
| Uzbek | 16.75% | 13.89% |

[Reading report and per-recording results](../benchmark/results/multilingual_1000_20261001/README.md).
These results support language recognition on reading; they do not establish
accuracy on telephone calls, spontaneous conversation or live WebSocket input.

## Natural speech and ready-model comparison

All entries in a row use identical source recordings and fixed human references.
Models and inference settings were frozen before evaluation. WER is the primary
metric; these are complete-corpus results, not selected favorable examples.

| Corpus | Clips | GigaAM `ml_ctc_large` | Whisper large-v3 | Omnilingual CTC1B v2 |
|---|---:|---:|---:|---:|
| Kazakh conversational utterances | 887 | 36.91% | 95.19% | 83.36% |
| Kazakh interview and standup | 31 | 16.54% | 49.25% | 26.50% |
| Uzbek Telegram voice messages | 745 | 22.16% | 101.10% | 59.78% |
| Kyrgyz reading, same FLEURS subset as above | 1,000 | 12.34% | Unsupported language token | 15.01% |

The two alternatives do not improve WER on these matched corpora. GigaAM also
uses less CPU time and sampled memory in the small, separate resource test;
timings were not isolated from unrelated workloads. On Kyrgyz reading,
Omnilingual has lower character error rate (3.44% versus 4.69%), despite higher
word error rate. Results compare these deployed configurations, including their
different runtimes, preprocessing and precision.

The remaining limitation is conversational accuracy: 36.91% WER on the acquired
Kazakh utterances and 22.16% on Uzbek messages still require transcript review
where correctness matters. **Kyrgyz conversation remains unmeasured.** A bigger
model is not demonstrated to solve these errors by this comparison.

[Conversation sources and selection](../benchmark/results/multilingual_conversation_20261001/README.md)
· [Full model comparison, resource measurements and audits](../benchmark/results/multilingual_model_comparison_20261001/README.md).

## How to interpret the numbers

- WER counts substitutions, deletions and insertions divided by reference words.
  Insertions can make it exceed 100%. It is not the percentage of failed files,
  and `100 − WER` is not a calibrated accuracy probability.
- Reading and conversation are different corpora. Do not combine them into one
  accuracy percentage for a language or compare their rows as language rankings.
- The 887 Kazakh clips are human-segmented, non-overlapping intonation units
  averaging 1.55 seconds from four recordings, not 887 independent conversations.
  This does not evaluate full-conversation segmentation or diarization. The
  31 media clips come from only two videos.
- Uzbek uses all 745 publisher-verified messages, including 18 duplicate rows.
  Publisher provenance is not independent authentication of every recording;
  speaker identities and model pretraining overlap are unknown.
- Scoring preserves language letters and digits, removes punctuation/apostrophe
  variants, and applies no number conversion, transliteration or best-reference
  selection. Both recognition and written-form differences contribute to WER.
- All failed requests remain in scoring as empty outputs. There were no technical
  request failures in the published complete runs.

## Evidence and reproduction

The three linked reports retain manifests, hypotheses, per-recording edit counts,
source revisions, model/binary/source hashes and the independent audits. The
measurement snapshot predates this documentation commit; it is not a claim that
the current `main` binary was re-benchmarked.

Check the published text scores without downloading audio or model weights:

```sh
python -m pip install rapidfuzz
python scripts/verify_multilingual_publication.py
```

This checks the publication's content hashes, manifest/result identities and
all 20 complete result files (15,989 recognitions), then recomputes WER and CER.
It does not independently authenticate upstream labels or re-run inference.
The original full audit scripts additionally require the original source-audio
and model caches. Recorded absolute paths are preserved as historical evidence;
they are not portable installation paths. Exact inference replay also requires
the recorded frozen binary; its hash is retained, but the binary is not bundled
or reconstructed by this publication.

No audio or model binaries are included. The references retain their upstream
licenses, including MCSKL's CC BY-NC-SA terms; see
[data attribution](../benchmark/DATA_LICENSE).
