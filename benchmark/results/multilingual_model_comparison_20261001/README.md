# Ready multilingual speech recognition comparison

**Complete: all seven quality runs and 22 resource runs independently verified.** All seven alternative-model corpus runs passed independent validation. GigaAM has lower word error rate than both alternatives on both Kazakh corpora and Uzbek voice messages. On Kyrgyz read speech, GigaAM has lower WER while Omnilingual has lower CER. These are corpus-specific results, not language-wide accuracy estimates.

The experiment compares existing models without training, adaptation or product integration. Frozen source recordings and human references are identical to the earlier GigaAM evaluation.

## Recommendation

Keep the existing standard `ml_ctc_large` configuration among the three evaluated choices. Neither ready alternative improves corpus-level WER on any eligible matched corpus, and GigaAM also has the lowest observed CPU time and sampled memory footprint. This does not establish superiority over every available model or application.

The remaining product limitation is accuracy on spontaneous speech: GigaAM still has 36.91% WER on short Kazakh conversational utterances and 22.16% on the acquired Uzbek messages. These outputs require review when transcript correctness matters. Kyrgyz conversation remains unmeasured; its 12.34% result is for reading. No training, fine-tuning, model-default change or product integration was performed.

## Quality results

WER counts substitutions, deletions and insertions divided by reference words; lower is better. It is not the percentage of incorrect sentences. Each corpus stays separate.

| Corpus | Recordings | GigaAM Large WER | Whisper large-v3 WER | Omnilingual CTC1B v2 WER |
|---|---:|---:|---:|---:|
| Kazakh interview and standup | 31 | 16.54% | 49.25% | 26.50% |
| Kazakh conversational utterances | 887 | 36.91% | 95.19% | 83.36% |
| Uzbek Telegram voice messages | 745 | 22.16% | 101.10% | 59.78% |
| Kyrgyz read speech | 1,000 | 12.34% | Unsupported language token | 15.01% |

The completed Kazakh media comparison uses 532 reference words: GigaAM has 88 word edits, Whisper 262 and Omnilingual 141. CER is 4.26%, 14.54% and 6.26%, respectively. Both alternatives completed all 31 files without failed or empty results. This corpus comes from only two source videos and is not a representative estimate of everyday conversation.

On the 887 short Kazakh conversational utterances, all models were scored against the same 3,701 reference words. GigaAM has 1,366 word edits, Whisper 3,523 and Omnilingual 3,085; CER is 17.37%, 53.45% and 61.19%, respectively. Empty outputs are 11, 0 and 9; no requests failed. Whisper improves on GigaAM on 25 clips, worsens on 729 and ties on 133; Omnilingual improves on 29, worsens on 692 and ties on 166. These are isolated intonation units averaging 1.55 seconds from four recordings, not a general estimate for full conversations. Both larger alternatives perform worse in this exact configuration and corpus. [Whisper conversational results](whisper_kk_conversations_expanded.json) and [Omnilingual conversational results](omni_kk_conversations_expanded.json) are independently verified.

Completed alternative outputs: [Whisper Kazakh media](whisper_kk.json) and [Omnilingual Kazakh media](omni_kk.json). [Independent validation](independent_validation.json) is complete for all seven runs. Prior baseline reports: [natural speech](../multilingual_conversation_20261001/README.md) and [read speech](../multilingual_1000_20261001/README.md).

On all 745 Uzbek messages, Whisper has 10,968 word edits over 10,849 reference words: 8,648 substitutions, 1,046 deletions and 1,274 insertions. WER is 101.10% and CER is 43.94%, compared with GigaAM 22.16% WER and 7.66% CER. WER may exceed 100% because insertions count as extra errors; it is not a percentage of recordings failed. Whisper has eight empty outputs and no failed requests. It improves on GigaAM on six messages, worsens on 714 and ties on 25. [Complete Whisper Uzbek results](whisper_uz.json) passed independent scoring and identity checks.

Omnilingual completed the same 745 Uzbek messages with 6,486 word edits (4,930 substitutions, 1,349 deletions, 207 insertions), 59.78% WER and 29.01% CER. It has one empty output and no failed requests. It improves on GigaAM on 21 messages, worsens on 657 and ties on 67. [Complete Omnilingual Uzbek results](omni_uz.json) passed the independent audit. Neither ready alternative improves the primary Uzbek score in this comparison.

On the 1,000 Kyrgyz read recordings, Omnilingual has 2,602 word edits over 17,333 reference words, 15.01% WER and 3.44% CER. GigaAM has 2,139 word edits, 12.34% WER and 4.69% CER. Both have zero empty outputs and failed requests. Omnilingual improves word edit count on 198 recordings, worsens on 504 and ties on 298. Its lower CER despite higher WER means the ranking depends on the error unit: smaller character changes may still invalidate whole words. [Complete Kyrgyz results](omni_ky.json) passed independent verification. This does not measure spontaneous Kyrgyz conversation.

The [complete-corpus output diagnostic](kk887_output_diagnostics.json) finds 34 Whisper outputs longer than twice their references and 48 with runs of at least three identical tokens; substitutions account for most Whisper word errors. Omnilingual produces fewer words overall and 519 outputs contain a script absent from their corresponding reference, including Arabic or Han characters. Script detection is not language identification. These observations describe text differences without establishing their acoustic or model-internal cause. Word boundaries also differ across scripts, so WER alone does not capture the full character-level damage. [Five Kazakh media examples](kk31_error_examples.json) illustrate the separate 31-clip result.

The [Uzbek script diagnostic](whisper_uz_output_diagnostics.json) verifies all 745 source references against the acquired Arrow tables and finds all references use Latin script. Whisper emits Cyrillic in 63 outputs, but 10,148 of its 10,968 word edits occur in the 670 Latin-only outputs. The large raw WER is therefore not concentrated in a Cyrillic-versus-Latin mismatch. This exhaustive diagnostic does not change primary metrics or establish a purely acoustic error rate; spelling, word boundaries, number formatting and recognition differences remain mixed.

The corresponding [Omnilingual Uzbek diagnostic](omni_uz_output_diagnostics.json) finds script mismatches in 204 of 745 outputs. Those outputs account for 1,796 of 6,486 word edits; most edits occur in outputs without a foreign-script mismatch. This is co-occurrence, not a count of errors causally attributable to script selection. No transliteration or revised scoring was applied.

## CPU time and memory

The fixed resource sample contains the first five source-order recordings per corpus, with a separate sixth recording for warmup. Each model/corpus starts in a fresh process. Two rounds reverse model order, giving 22 sequential jobs; Whisper Kyrgyz is omitted. All 110 measured hypotheses exactly match the corresponding quality-run hypotheses, with zero failed requests. [Resource validation](resource_validation.json) verifies identities, timings, order and absence of overlapping resource jobs.

Measured wall RTF is processing time divided by audio duration; below 1 means faster than real time on this sample. These are the observed two-round ranges, not confidence intervals or full-corpus latency statistics.

| Five-recording sample | GigaAM RTF | Whisper RTF | Omnilingual RTF |
|---|---:|---:|---:|
| Kazakh conversational utterances | 0.089–0.100 | 4.094–4.524 | 0.562–0.606 |
| Kazakh interview and standup | 0.072–0.081 | 1.216–1.217 | 0.476–0.516 |
| Uzbek messages | 0.090–0.092 | 0.653–0.692 | 0.503–0.507 |
| Kyrgyz read speech | 0.074–0.098 | Unsupported | 0.476–0.482 |

Memory below is sampled process-tree PSS, including the benchmark harness and owned child processes. Ranges span the measured phases of all eligible corpus/round jobs. The all-phase peak also includes model startup and warmup.

| Model | Measured-phase peak PSS range | Maximum all-phase PSS |
|---|---:|---:|
| GigaAM Large INT8 | 0.66–0.80 GiB | 0.80 GiB |
| Whisper large-v3 int8_float32 | 2.45–2.67 GiB | 2.96 GiB |
| Omnilingual CTC1B v2 FP32 | 4.16–4.62 GiB | 7.46 GiB |

This Linux AMD Ryzen AI 9 HX 370 host was not isolated from unrelated user workloads. Our quality jobs had ended and resource jobs did not overlap, but these timings do not establish a hardware-independent or fully isolated speed ratio. GigaAM uses pool size 2; each alternative uses one model instance. All use three compute threads. GigaAM's measurement includes its local HTTP boundary; the alternatives use Python adapters. Model precision, native preprocessing and runtime differ.

Memory is sampled every 100 ms and can miss short peaks. PSS/USS and RSS are read from different Linux accounting sources at different instants; seven observed PSS-above-RSS discrepancies are preserved in the [memory sampling audit](resource_memory_sampling_audit.json). No raw values were repaired. PSS is used for the memory table; RSS is not treated as a precise resident-memory bound. Startup time includes identity validation and loading, with uncontrolled filesystem caches.

## Frozen configurations

| Model | Local implementation | Precision | Language input |
|---|---|---|---|
| GigaAM multilingual large | Frozen gigastt binary and ONNX model | INT8 | None |
| Whisper large-v3 | faster-whisper 1.2.1 and CTranslate2 4.8.2 | CPU int8_float32, weights stored FP16 | Known kk or uz |
| Omnilingual CTC1B v2 | Official unchanged Meta pipeline, fairseq2 0.6, torch 2.8.0 CPU | FP32 | None for CTC |

Whisper uses beam size 5, temperature 0, transcription rather than translation, no previous-recording context and no external VAD. Its standard internal no-speech gate remains enabled. This is the standard [Whisper large-v3 conversion](https://huggingface.co/Systran/faster-whisper-large-v3/blob/edaa852ec7e145841d8ffdb056a99866b5f0a478/README.md), not the previously piloted Kazakh/Russian turbo adaptation. The [Whisper language inventory](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py) contains Kazakh and Uzbek but no Kyrgyz token; no neighboring-language substitution is tested.

[Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr/tree/81f51e224ce9e74b02cc2a3eaf21b2d91d743455) uses the official CTC1B v2 checkpoint and written-v2 tokenizer, greedy CTC and no external language model. All recordings fit its 40-second limit. The declared but source-unused KenLM dependency could not be linked in this environment; the unmodified official CTC pipeline imports and runs without it. This omission, installed source hashes and full package versions are recorded in [setup evidence](omni_setup.json). Whisper is MIT; the Omnilingual repository is Apache-2.0.

Weight and source identities: [Whisper provenance](whisper_provenance.json), [Omnilingual acquisition](omni_acquisition.json), [baseline pairing audit](baseline_pairing_audit.json). No model files or decoder settings were selected by evaluation outcomes.

## Input and scoring limits

Alternatives receive the same source files through FFmpeg float32 mono 16 kHz decoding, followed by each model's fixed native preprocessing. There is no denoising, telephone simulation, VAD sweep, post-hoc transliteration or output correction. The [frontend audit](whisper_frontend_sanity.json) verifies all 745 Uzbek mono arrays are bit-identical to their original float32 samples. For Kazakh stereo audio, FFmpeg's downmix has approximately sqrt(2) gain relative to an arithmetic channel average. Identical source files therefore do not establish identical decoded PCM across all native pipelines. These are complete configuration comparisons, not an isolated test of model architecture.

The established Unicode normalizer lowercases and removes punctuation/apostrophe variants, retaining language letters and digits; it does not convert numbers. Primary micro WER and CER use one fixed human reference. CER includes normalized spaces. Every committed failed request remains an empty hypothesis and contributes deletion errors. Uzbek duplicates remain in the primary 745-row population, as in the baseline.

Kazakh MCSKL consists of short, human-segmented non-overlapping utterances from four recordings, not 887 independent conversations. Uzbek speaker identities are unavailable. Kyrgyz is read speech only; natural conversation quality is still unmeasured. Training overlap with public corpora is unknown. See the baseline provenance reports for source licensing, human-annotation evidence, selection and duplicate-reference limits.

Whisper also emitted text on one synthetic one-second silence smoke input; [the observation](whisper_runtime_smoke.json) is retained without threshold tuning. It is a single implementation check, not a hallucination-rate benchmark.

## Execution and verification

[Protocol](protocol.json) fixes settings before quality inference. Each instance uses three CPU compute threads. Four independent identical Whisper instances process separate files for quality throughput; Omnilingual uses one instance per corpus run. Outputs commit in source order to durable JSONL ledgers. Five preselected Whisper outputs were identical between sequential and parallel execution: [consistency check](whisper_parallel_consistency.json).

The user requested a pause, then explicitly resumed work. All five active processes stopped during the pause. [Resume integrity](resume_integrity_20261002.json) verifies the preserved prefixes, new process suffixes, unchanged identities and absence of duplicate or partial records. Complete results are retained rather than rerun.

Quality-run latencies are not used for speed rankings. The [20-recording resource selection](resource_subset.json) was fixed before candidate outcomes. Preparatory timing runs informed only the independent-worker execution amendment and are excluded from final resource results. Final runs are recorded in [the order ledger](resource_order.jsonl). Unrelated user workloads were not stopped.

The runner and audit tools are `scripts/benchmark_ready_asr.py` and `scripts/audit_ready_asr.py`; model adapters are `scripts/ready_asr_whisper.py` and `scripts/ready_asr_omni.py`. They reject changed audio, model, source, scorer or execution identities on resume. Independent scoring recomputes word edit distances, character distances, aggregate metrics and exact baseline pairing. All quality and resource runs are complete. Repository checks passed: 1,126 unit tests, 39 ignored, zero failed; clippy and formatting passed. Runner, adapter and audit integrity tests passed, including non-atomic Linux memory accounting validation.


To reproduce audits against the retained local models and source-audio cache, from the repository root:

```sh
benchmark/.venv/bin/python scripts/audit_ready_asr.py --output benchmark/results/multilingual_model_comparison_20261001/independent_validation.json
benchmark/.venv/bin/python scripts/audit_ready_resources.py --output benchmark/results/multilingual_model_comparison_20261001/resource_validation.json
```

Quality execution uses `scripts/benchmark_ready_asr.py` with the backend, model directory, manifest and output paths recorded in each `.meta.json`; Whisper uses `--workers 4`, Omnilingual the default one worker. Existing complete outputs are verified and retained. Resource orchestration is `scripts/run_ready_resources.py`; it verifies and skips successfully committed jobs. The pinned cache is required; model weights and audio are not embedded in this report directory.
