# Telephone corpus availability and evaluation expansion

Search and acquisition on 2026-10-01 added one real Kazakh telephone recording with a publisher-supplied human transcript. This expands the available public telephone evidence from one recording to two, not to a representative call-center benchmark; independent conversations and speakers cannot be certified from the public metadata. No additional openly downloadable Uzbek telephone corpus with verified human references was found in the checked sources. This is a search result, not a claim that none exists.

## Acquired Kazakh telephone sample

The [MATERIAL Kazakh-English catalog](https://catalog.ldc.upenn.edu/LDC2025S03) publishes a 120-second, two-channel, 8 kHz A-law WAV demonstration and a Kazakh transcript covering approximately its first minute. The transcript differs from the previously tested Babel sample. MATERIAL incorporates Babel material, so independence from the complete Babel collection and speaker independence cannot be established from the public files.

Both channels are cut at 56.140 seconds, before the final exchange in the partial transcript. Channel 0 maps to `inLine`, channel 1 to `outLine`; speech/silence at the annotated intervals confirms the mapping without choosing it using ASR output. Non-speech markup is removed; lexical content, filled pauses and punctuation are preserved for the existing scorer. The result is 112.28 channel-seconds from **one** conversation. An additional 24 excerpts, totaling 47.403 channel-seconds, use human timestamps for diagnostic segmentation. They are not independent calls, and results must be labeled as oracle segmentation.

The sample is conversational telephone speech, not a customer-service call, and Kazakh-Russian code-switching has not been established. Keep the audio, transcript, references and generated hypotheses locally because the catalog demonstration is not an openly licensed corpus for redistribution. Source hashes and preparation details are in [material_sample_provenance.json](material_sample_provenance.json).

Reproduce with the existing benchmark Python environment:

```sh
benchmark/.venv/bin/python scripts/prepare_telephone_corpora.py \
  --cache /tmp/gigastt-telephone-expansion \
  --summary benchmark/results/telephone_corpora_20261001/material_sample_provenance.json
```

The local manifests are `telephone_expansion_manifest.json` (two complete channels within the annotated interval) and `telephone_segments_manifest.json` (24 diagnostic excerpts). The script checks fixed source SHA-256 values, sample rate, channel count and full decoded length before preparation.

## Additional sources checked

| Source | Evidence and access | Decision |
|---|---|---|
| [Babel Kazakh LDC2018S13](https://catalog.ldc.upenn.edu/LDC2018S13) | Approximately 203 hours of telephone speech with transcripts; full access requires an LDC agreement. Public sample already tested. | Potential real telephone adaptation data after licensed access; no new public recordings acquired. |
| [NIST OpenKWS16](https://www.nist.gov/itl/iad/mltg/openkws16-evaluation) | Babel packs, including Kazakh, were distributed to registered participants with signed agreements and passwords. | Not an unrestricted alternative download route for Babel training data. |
| [IICT Kazakh telephone research corpus](https://www.nature.com/articles/s41598-022-12260-y) | The 2022 paper describes 200 hours of telephone speech supplied by a telecom for scientific use; data availability says not applicable. | Relevant evidence that a corpus was collected, but no downloadable benchmark or training data in the paper. |
| [MATERIAL Kazakh LDC2025S03](https://catalog.ldc.upenn.edu/LDC2025S03) | Approximately 57 hours; only about 17% transcribed. Full collection requires an agreement. | Acquired public demonstration only; not a new 57-hour training set. |
| [dev-muhammad/call_center_voice_uz_1](https://huggingface.co/datasets/dev-muhammad/call_center_voice_uz_1) | Hub metadata declares Uzbek ASR and MIT; revision `c1a2333bb0e9717736840e2c1f33a94c58944c3d`. README download returned an authentication/access restriction; public file listing contains only attributes and README. | Not acquired. The title alone does not verify real calls, human references or available hours. |
| [KSC2](https://issai.nu.edu.kz/kz-speech-corpus/) | About 1,200 hours, including Kazakh-Russian switching, read material, TV/radio, senate and podcasts. Official site specifies CC BY 4.0. | Useful multilingual adaptation source, not demonstrated telephone audio. HF card says MIT, a metadata discrepancy; retain official attribution and resolve provenance before bulk reuse. |
| [Uzbek Speech Corpus](https://issai.nu.edu.kz/issai-datasets/) | 105 hours from 958 speakers, CC BY 4.0 on the official site; [author recipe](https://github.com/IS2AI/Uzbek_ASR) provides download and training instructions. | Human read-speech adaptation candidate, not real telephone evaluation. Official train/dev/test must remain separated. |
| [Uzbek Telegram evaluation set](https://huggingface.co/datasets/BoburAmirov/asr_evaluate_set) | Already acquired; publisher describes manual references, Apache-2.0. One published split named train, with no verified speaker-disjoint partition. | Keep our existing evaluation material out of training. Voice messages are not telephone calls. |
| [Uzbek call-center Whisper](https://huggingface.co/Abduqayum/whisper-uzbek-medium-callcenter) | Publisher describes audiobook/podcast/tech-talk training with Gemini-generated labels and telephone augmentation. | Alternative model candidate; its data is not human-labeled real telephone speech despite the model name. |
| [Kazakh OpenSLR 140](https://openslr.org/140/) | Large Kazakh speech corpus, CC BY-SA 3.0; download shards of 12–15 GB. | No evidence on the catalog page of real telephone recordings. Not downloaded as a call corpus. |
| [shunyalabs Kazakh](https://huggingface.co/datasets/shunyalabs/kazakh-speech-dataset) | Public audio/transcript parquet with train/validation/test, but card contains only structural metadata and no source, license or telephone description. | Excluded from defensible telephone benchmark and training selection. |

The [Habr article](https://habr.com/ru/articles/1088214/) and [publisher version](https://sanatel.kz/paper_local_asr_server.htm) describe private Asterisk calls and compare confidence/coverage on three Uzbek conversations. Neither inspected article provides downloadable evaluation calls with reference transcripts. Those metrics are not WER and cannot be compared numerically with our WER. The existing 31-clip YouTube/interview code-switching benchmark is a different source, not the article's call dataset.

## Conditional adaptation data and separation

For a small reproducible adaptation experiment, use the official Kazakh FLEURS **train** partition for fitting and **validation** for model selection. FLEURS is human read speech under CC BY 4.0, not a telephone corpus. Its [publisher card](https://huggingface.co/datasets/google/fleurs) states that training speakers differ from development/test speakers. Narrowband filtering and codec augmentation create a proxy experiment and cannot establish real-call quality by themselves. Pinned parquet revision: `168de341b3db6859a9bac1c50a2ef5e3b47647e0` in [google/fleurs](https://huggingface.co/datasets/google/fleurs/tree/168de341b3db6859a9bac1c50a2ef5e3b47647e0/kk_kz).

| Partition | Parquet SHA-256 |
|---|---|
| train | `7f5d4a6c58db465dadfef9718dc4a61f1137f13196c3634cf803acd6eb859457` |
| validation | `35f8de4d59af07b236a9174e64093f64cfa74e5a47bf488f909e4138a8d24315` |

Both files were acquired and their complete hashes verified: 3,200 training rows and 369 validation rows, approximately 3.00 GB combined. Local paths, exact sizes and pinned URLs are in [fleurs_training_source_provenance.json](fleurs_training_source_provenance.json).

The two LDC demonstrations, all 31 existing Kazakh natural-speech evaluation clips, Uzbek FLEURS test clips, and Uzbek Telegram evaluation clips must stay out of training. Treat the already repeatedly inspected telephone samples as development diagnostics, not a pristine final holdout. Preserve original conversation and speaker identity when new data becomes available; never split turns of one call between training and evaluation. Any speaker-disjoint claim needs actual identities or publisher-guaranteed partitions, not an arbitrary random split of utterances.

A production-quality fine-tuning result remains dependent on additional representative real calls, human references, and an independent held-out set. No commercial license was purchased, restricted data accessed, or publisher contacted during this search.
