# Conversational speech recognition in Kazakh Kyrgyz and Uzbek

The existing standard `ml_ctc_large` model was evaluated through gigastt on human-reference natural speech. Kazakh conversational utterances produced **36.91% WER**; Uzbek voice messages produced **22.16% WER**. Kyrgyz conversational accuracy remains unmeasured because the inspected sources did not establish an accessible qualifying corpus. No training, model changes or processing sweeps were performed.

## Measured results

Lower word error rate (WER) and character error rate (CER) are better. These are separate corpora, not a cross-language ranking.

| Language and speech type | Clips | Minutes | WER | CER | Empty outputs | Failed requests |
|---|---:|---:|---:|---:|---:|---:|
| Kazakh, conversational utterances | 887 | 22.98 | 36.91% | 17.37% | 11 | 0 |
| Kazakh, interview and standup | 31 | 3.70 | 16.54% | 4.26% | 0 | 0 |
| Uzbek, Telegram voice messages | 745 | 99.91 | 22.16% | 7.66% | 3 | 0 |
| Kyrgyz, conversational speech | Not acquired | — | Not measured | Not measured | — | — |

The Kazakh conversational result contains 1,366 word edits over 3,701 reference words (910 substitutions, 356 deletions, 100 insertions). Uzbek contains 2,404 edits over 10,849 words. Empty outputs remain in the denominator and count as deletions. All 1,663 requests completed without technical failure.

WER counts substitutions, deletions and insertions divided by reference words. It is not the percentage of incorrect sentences or a calibrated probability that a word is correct. CER can be much lower when a word differs by only a few letters; it must not replace WER as the headline.

## Reading and conversation are different measurements

The earlier [1,000 recordings per language benchmark](../multilingual_1000_20261001/README.md) used FLEURS read speech and the same standard large model: Kazakh **11.47%**, Kyrgyz **12.34%**, Uzbek **13.89% WER**. Those results establish recognition on that reading corpus. They do not predict telephone or spontaneous conversation quality.

The natural-speech measurements above show materially higher error rates on the acquired Kazakh and Uzbek corpora. Corpus, speakers, recording conditions, segmentation and reference conventions differ, so the difference cannot be attributed to any one cause. The findings do not establish a single accuracy percentage for an entire language.

## Sources and selection

**Kazakh conversations:** [MCSKL](https://osf.io/6zjdq/), by Giorgia Troiani, Andrey Filchenko and the annotation team; [human annotation description](https://reference-global.com/article/10.5334/johd.529?tab=article). CC BY-NC-SA 4.0. Four recordings, MCSKL030–033, include friends, students and an online interview. Native-speaker ELAN orthographic references retain spoken fragments and Russian/English insertions. This research use does not confer a commercial data license.

We selected up to 250 eligible intonation units per recording with seed 42, then restored time order: 250/250/137/250. Units had to last 1–30 seconds, have usable human text and no overlap with another speaker's annotated unit, including empty annotations. Original PCM, sample rates and channels were preserved. The resulting 887 units contain 22.98 minutes from only four recordings; their average duration is about 1.55 seconds. This measures recognition with human-provided boundaries on isolated non-overlapping utterances, not end-to-end transcription or diarization of full conversations. It is not 887 independent conversations.

The initial 100-unit selection contained only 151 seconds and was superseded before inference. Independent inspection also caught empty-annotation overlaps and residual transcription markup. The final expanded manifest was corrected, frozen and independently verified before any MCSKL inference; no model outcomes informed selection. The [provenance](kk_conversations_expanded_provenance.json) records the amendment and exclusions; the [independent source audit](kk_conversations_independent_audit.json) confirms the final SHA and exact PCM alignment.

**Kazakh media:** all 31 clips from [Timur Seidalin's Kazakh codeswitch benchmark](https://github.com/Tim2190/Kaz-ASR-codeswitch-benchmark), pinned revision `6575c5cd9e9e481edcb4d9928fda2b2f60d3e5ad`. Sixteen standup clips and fifteen interview clips come from just two videos. Publisher annotations/code are MIT, source audio CC BY per publisher attribution. Primary scoring uses the verbatim human reference, removing only literal `[false_start]` editorial markers while retaining spoken fragments. This previously evaluated small corpus is separate from MCSKL and is not a blind holdout.

**Uzbek:** all 745 published rows of [Bobur Amirov's ASR evaluation set](https://huggingface.co/datasets/BoburAmirov/asr_evaluate_set), pinned revision `c6e1b4f66d45e9d7644003d1564837786295850a`, Apache-2.0 per publisher. The publisher describes manually annotated, human-verified Telegram voice messages. Conversational provenance and human verification are publisher claims, not independent per-recording authentication. Speaker identities are unavailable. Source audio arrays were saved as float32 WAV without sample changes, filtering or resampling. The published split is named `train`; it was used only for evaluation here, with no training or tuning. Model pretraining overlap is unknown.

There are 727 distinct Uzbek audio hashes: 18 duplicated rows remain in the primary metric, including two pairs with conflicting normalized references. Keeping the first source-order row per hash gives **22.30% WER** on 727 recordings, a secondary diagnostic. Full [provenance](uz_provenance.json) and [duplicate audit](uz_duplicate_audit.json) are retained.

**Kyrgyz:** [source search evidence](ky_sources.json) records 13 inspected source families. Reading corpora, synthetic telephone/noise augmentations, unverified labels and inaccessible material were not substituted for natural human-reference conversation. This bounded search does not establish that such data do not exist or that the model lacks Kyrgyz support.

## Fixed inference and scoring

CPU, standard INT8 `ml_ctc_large`, pool size 2, three encoder threads, punctuation/ITN/VAD off. Original audio only; no telephone simulation or external preprocessing. The binary and model are the same frozen identities used for the earlier standard-model benchmark. The encoder SHA is `b2ad9c38fc04197ba758105d33f7404fd13d977958722e0f49e3f3e22521f1c6`.

Primary scoring is micro WER/CER with the existing Unicode normalizer: lowercase, retain Unicode letters/digits, remove punctuation and apostrophes, no number conversion. CER includes normalized spaces. Every row uses one fixed reference; there is no best-of-reference selection. On the Kazakh media clips, using the publisher's separate normalized-written reference gives **6.62% WER** for the same hypotheses instead of primary 16.54%; this illustrates reference sensitivity and is not a replacement headline. Uzbek digit-free references yield 20.62% WER as a separate diagnostic.

## Artifacts and verification

Final results contain hypotheses and per-recording edit counts: [Kazakh conversations](kk_conversations_expanded_large.json), [Kazakh media](kk_large.json), [Uzbek messages](uz_large.json). Each has a durable `.jsonl` ledger, metadata, frozen protocol and server log. Manifests retain local audio paths, SHA hashes, references and source identities; audio remains in the local cache.

[Independent numeric validation](validation.json) recomputes word distances with independent dynamic programming, recomputes character scores, verifies all summary groups and checks manifest/result/ledger, audio, model, binary and scorer identities. Corpora remain separate. Repository verification: 1,119 unit tests passed, 39 ignored, zero failed; clippy and formatting checks passed. New Python helpers compile and `git diff --check` passes.

To reproduce using the pinned local binary, model and acquired audio cache, run from the repository root:

```sh
benchmark/.venv/bin/python scripts/benchmark_conversational_languages.py --manifest benchmark/results/multilingual_conversation_20261001/kk_conversations_expanded_manifest.json --output benchmark/results/multilingual_conversation_20261001/kk_conversations_expanded_large.json --port 20023
```

Use the corresponding `kk_manifest.json` or `uz_manifest.json` and output prefix for the other corpora. The wrapper verifies hashes and preserves the frozen protocol; existing complete ledgers are audited/resumed rather than silently replaced. Preparation scripts are `prepare_conversation_kk.py`, `prepare_conversation_kk_mcskl.py` and `prepare_conversation_uz.py` under `scripts/`.
