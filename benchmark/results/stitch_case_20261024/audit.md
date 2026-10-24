# Follow-up seam-casing audit

The follow-up review covers the changes through `7662973`, the stitching caller,
regression tests, and the local comparison harness. It found three gaps:

| Finding | Consequence | Resolution |
|---|---|---|
| Ambiguity was checked only in the older overlap | An older `да` could lend lowercase to the first `Да` in a newer `Да да` hypothesis despite competing matches | Require a unique match in both bounded overlap slices; preserve the selected spelling otherwise |
| Input and response validation used Python `assert` | `python -O` or `PYTHONOPTIMIZE` disabled checksum and transcription checks | Use explicit validation exceptions, with an optimized-interpreter regression test |
| Timing-only runs compared each response only to its own warmup | A consistently incorrect candidate could produce apparently valid latency results | Compare warmups across arms before collecting timings; retain both responses before validation and record allowed casing differences |

The new repetition regression failed before the runtime fix. A separate control
checks that repetitions outside the allowed timestamp drift do not suppress a
valid correction. Existing tests cover the actual first decoded word, proper
names without lowercase evidence, sentence boundaries, acronyms, Latin text,
punctuation, ambiguous older repetitions, bounded fallback, empty inputs, the
older winning hypothesis, retained-prefix length and complete word metadata.

The runtime refactoring shares one private uniqueness check between both
hypotheses. It retains the existing alignment, cut selection, 64-word overlap
limit and 0.4-second timestamp tolerance. It introduces no dependency, model,
configuration, public API or decoder change. The added check can only suppress a
previously permitted casing correction. It cannot introduce a new correction.
The caller still receives the exact unchanged prefix length for incremental
publication; correction only touches the first appended word's spelling.

## Evidence and remaining limits

The 42 saved historical response pairs were revalidated with the hardened
comparison helper, and their WER/casing scores were recomputed from the frozen
manifest. This is a replay of saved evidence, **not a new inference run**.
The measured runtime remains `36ddb57`; historical JSON, timings and checksums
are unchanged. The timing table does not measure this follow-up revision.
The original campaign used normal Python and the full quality/timing phase,
so neither harness gap invalidates its published comparisons.

The original author's recording is still unavailable and the public corpus
exercised no positive casing correction. The older hypothesis can itself have
incorrect casing, so alignment is supporting evidence, not a proper-name
recognizer or a proof of sentence boundaries. Keep the PR open pending a real
positive reproduction; do not claim a demonstrated quality gain. The historical
latency spread also prevents a strict 5% non-inferiority claim.

## Local validation

- Workspace unit/bin tests: 1,248 passed, 48 ignored, zero failures.
- Nine comparison-helper tests pass under normal Python and `python -O`.
- Strict workspace/all-target Clippy, formatting, documentation drift and
  publication checks pass.
- No GitHub workflow was dispatched and the PR was not merged.
