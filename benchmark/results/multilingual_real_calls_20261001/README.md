# Multilingual models on the available real telephone recordings

**The new small quantization does not fix recognition of these complete calls.** It has the same primary full-channel word-error totals as the original small model. Default VAD improves some results, particularly large on Babel, but all three models still have high errors on these two examples. The earlier improvement on simulated FLEURS telephone audio does not establish reliable real-call accuracy.

No training, model replacement, audio preprocessing sweep or threshold tuning was performed. All 162 measured HTTP requests completed successfully. The installed models remain unchanged.

## Fixed primary comparison

The inputs are two repeatedly inspected Kazakh telephone recordings: the Babel LDC2018S13 public demonstration (110.04 seconds, one channel) and the MATERIAL LDC2025S03 demonstration (two channels, each cut at the pre-existing 56.140-second annotation boundary). There are **two recordings, three channels and 170 reference words**, not three independent calls. MATERIAL contains Babel material; independent speakers cannot be certified. These are conversational demonstrations, not a representative customer-service corpus.

The three models use the same frozen server binary, original prepared files, CPU, pool size two, three encoder threads, two concurrent HTTP workers, punctuation/ITN off and VAD off. The candidate is the exact previously selected S8 per-channel reduced-range model with FP32 convolution weights. One unmeasured warmup is excluded for each model.

WER (%), errors/reference words in parentheses:

| Recording | Original small | Candidate small | Large |
|---|---:|---:|---:|
| Babel, complete channel | 96.10 (74/77) | 96.10 (74/77) | 100.00 (77/77) |
| MATERIAL, both retained channels | 100.00 (93/93) | 100.00 (93/93) | 98.92 (92/93) |
| Aggregate | 98.24 (167/170) | 98.24 (167/170) | 99.41 (169/170) |

There are no empty primary responses or failed requests. Equal error totals do not imply identical transcripts. These poor results are specific to this small, repeatedly examined call set and pipeline; they do not contradict the separate 1,000-recording-per-language read-speech measurements or establish language-wide model quality.

## Separate automatic VAD control

After completing the primary comparison, a separate protocol fixed nine additional requests: the same three complete channels and all three models with existing Silero v5.1.2 VAD enabled at defaults (threshold 0.5, minimum trailing silence 500 ms). No threshold or segmentation search was performed. The cached VAD model matches the repository's pinned SHA-256.

| Recording | Original small + VAD | Candidate small + VAD | Large + VAD |
|---|---:|---:|---:|
| Babel | 100.00 (77/77) | 97.40 (75/77) | 62.34 (48/77) |
| MATERIAL | 94.62 (88/93) | 93.55 (87/93) | 95.70 (89/93) |
| Aggregate | 97.06 (165/170) | 95.29 (162/170) | 80.59 (137/170) |

VAD helps large substantially on Babel, while MATERIAL remains difficult. This is not a consistent or sufficient repair. There are no empty VAD responses or failed HTTP requests.

The frozen health/model endpoints do not expose a VAD flag. Activation is instead verified by the explicit CLI configuration, pinned model hash, attachment logs and request-scoped `transcribe complete (streaming windows, vad)` messages for every call and excluded warmup. Detected regions are 24 for Babel and 7/5 for the MATERIAL channels. There are no request-scoped VAD fallback warnings. Two synthetic-silence engine warmups before the server starts produce expected no-speech fallback warnings; these are not call inference.

## Oracle segmentation diagnostics

The existing human reference timestamps define 24 Babel and 24 MATERIAL excerpts. These reuse the same recordings and words; they are **privileged diagnostic segmentation**, not automatic segmentation or 48 extra calls. VAD is off in this group.

| Recording | Original small WER | Candidate small WER | Large WER |
|---|---:|---:|---:|
| Babel oracle excerpts | 79.22 (61/77) | 75.32 (58/77) | 74.03 (57/77) |
| MATERIAL oracle excerpts | 51.61 (48/93) | 51.61 (48/93) | 40.86 (38/93) |
| Aggregate | 64.12 (109/170) | 62.35 (106/170) | 55.88 (95/170) |

Empty excerpt outputs are 3/2/6 for original small/candidate small/large, respectively; all remain scored. The candidate's three-error improvement on Babel excerpts is not evidence of a complete-call solution. Oracle segmentation and full-channel scores must not be pooled.

## Provenance, privacy and limits

- [protocol.json](protocol.json) froze 51 inputs per model before primary inference: three channels plus 48 oracle excerpts. [vad_protocol.json](vad_protocol.json) separately froze the nine VAD requests before their inference.
- [original_small.json](original_small.json), [candidate_small.json](candidate_small.json), [large.json](large.json) and their `_vad.json` counterparts contain metrics, input/reference hashes and model/server identities; restricted reference text and generated transcripts are excluded.
- [validation.json](validation.json) independently recomputes word-edit distances, character scores and aggregate metrics for all 162 saved results, verifies paired input/reference identities, and checks VAD execution evidence. This is an output audit, not a second inference run.
- Restricted audio, transcripts, oracle cuts and hypotheses remain under `~/.cache/gigastt-real-calls/` and `~/.cache/gigastt-official-telephony/`. No demonstration audio or text is redistributed.
- Source and license context: [telephone corpus provenance](../telephone_corpora_20261001/README.md). The current [source search](source_search.json) found no additional verified, openly downloadable real-call examples with human references. This is not a claim that none exist.
- The earlier 31 Kazakh natural code-switch clips and 100 Uzbek Telegram clips are absent from persistent local caches; they were not silently substituted or downloaded for this run. They are not real-call evidence in any case. No usable Uzbek real-call corpus was evaluated.
- Timings overlap other host work and are not performance measurements. No speaker-independent confidence interval is warranted by two recordings.

One candidate startup preflight was rejected before any request because the preceding server's port was still unavailable; model runs now use separate ports. A VAD output validator initially counted startup silence warnings as request fallbacks; examining request-scoped logs resolved the distinction, and already completed outputs were retained without repeating inference. Neither orchestration correction changed inputs, model weights, recognition settings or scoring. All diagnostic servers were stopped after completion.

## Reproduction

Run the existing `scripts/prepare_telephone_segments.py` against the persistent Babel cache to create `~/.cache/gigastt-real-calls/babel_oracle/`. The persistent MATERIAL full-channel and oracle manifests are reused unchanged.

```sh
benchmark/.venv/bin/python scripts/benchmark_small_real_calls.py --mode prepare
benchmark/.venv/bin/python scripts/benchmark_small_real_calls.py --mode run --model original_small
benchmark/.venv/bin/python scripts/benchmark_small_real_calls.py --mode run --model candidate_small
benchmark/.venv/bin/python scripts/benchmark_small_real_calls.py --mode run --model large
benchmark/.venv/bin/python scripts/benchmark_small_real_calls.py --mode prepare-vad
benchmark/.venv/bin/python scripts/benchmark_small_real_calls.py --mode run-vad --model original_small
benchmark/.venv/bin/python scripts/benchmark_small_real_calls.py --mode run-vad --model candidate_small
benchmark/.venv/bin/python scripts/benchmark_small_real_calls.py --mode run-vad --model large
benchmark/.venv/bin/python scripts/benchmark_small_real_calls.py --mode audit
```

The new quantization remains an experimental local option supported by its earlier simulated-channel benchmark, not a demonstrated solution for these real calls.
