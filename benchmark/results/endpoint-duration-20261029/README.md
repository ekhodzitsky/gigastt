# Decoder endpoint duration: local regression evidence

This is a targeted streaming regression probe, not a release-wide WER or latency
benchmark. The reporter's recording was unavailable. No universal threshold is
selected from these observations; the default remains 600 ms.

## Setup

Baseline: main `143b9f2920a65fe4b466c012cfaaa8e0fb302bda`. Candidate source hashes,
model hashes, fixture hashes and raw-output hashes are in [summary.json](summary.json).
Both use the published models-v3-2026-10-09 bundles, CPU ONNX Runtime, one pool slot,
four encoder threads, no VAD, no external punctuation or ITN, 100 ms input chunks,
800 ms decode stride and the default 2.5-second window. E2E model formatting is
inherent and remains enabled. These are debug test builds on Ryzen AI 9 HX 370.

Three existing natural Golos fixtures (`golos_00.wav` through `golos_02.wav`) are
replayed, plus two explicitly synthetic stress variants of the first fixture:
an 800 ms midpoint pause (which can split a word), and a pitch-changing half-speed
version made by repeating samples. Neither is natural conversational slow speech.
Every input gets three seconds of trailing zeros and an explicit final flush.

The opt-in `benchmark_endpoint_duration` test records all partial/final text and
endpoint reasons. `at_ms` is **input-audio time**, not wall-clock or network latency;
input is replayed without real-time sleeps. Build and recognition timings are not
used to claim a performance improvement. Raw hypotheses remain in local evidence;
this directory publishes counts, timings and hashes only. WER uses the unchanged
`benchmark/common.py` normalizer, including its existing numeric/spelling rules.

## Results

40 stream replays: baseline two heads at 600 ms (10), candidate two heads at
600/1000/1500 ms (30). Counts below separate natural and synthetic inputs.

| Head / implementation | Threshold ms | Natural word errors | Stress word errors | Nonempty blank-triggered finals |
|---|---:|---:|---:|---:|
| RNNT baseline | 600 | 1 | 0 | 7 |
| RNNT candidate | 600 | 1 | 0 | 7 |
| RNNT candidate | 1000 | 1 | 0 | 6 |
| RNNT candidate | 1500 | 1 | 0 | 5 |
| E2E baseline | 600 | 3 | 4 | 16 |
| E2E candidate | 600 | 0 | 1 | 6 |
| E2E candidate | 1000 | 0 | 1 | 5 |
| E2E candidate | 1500 | 0 | 1 | 5 |

A lower number of finals is not automatically better: one recording can contain
multiple legitimate utterances. Here the directed pause/slowdown probes illustrate
how the threshold changes segmentation. Increasing it generally delays the final
by decode strides; 1500 ms is rounded up to 1520 ms. Natural examples at 1500 vs
600 ms have final-decision shifts of 0–1600 ms. These are not forced-alignment
measurements of delay since the last spoken word.

Two separate bugs were reproduced and fixed with initially failing unit tests:

- An internal blank run set a sticky endpoint flag even if speech resumed later
  in the same re-decoded window. A new token (including the dense-token cap path)
  now clears that flag. File decoding still consumes every frame.
- After committed speech slid out, a blank-only window could not close the existing
  utterance because the decoder had emitted no local tokens. Streaming now also
  considers trailing blanks when the assembled utterance is nonempty. Empty
  streams still do not emit empty utterances, cancellation is respected, and
  VAD/manual/assistant endpoint ownership remains unchanged.

The second bug surfaced in the intermediate 1500 ms E2E slowdown probe, which had
waited for explicit stop; the final implementation produces an automatic endpoint.
The slowdown still contains a numerical recognition error. Neither a larger pause
threshold nor these fixes establish general recognition accuracy or resolve every
reported truncation. Broader conversational evaluation needs the reporter's input.

## Reproduction

Use a fresh output filename each time; the test refuses to overwrite evidence:

```sh
GIGASTT_TEST_MODEL_DIR=/path/to/published/models \
GIGASTT_ENDPOINT_PROBE=/private/output/rnnt.json \
cargo test -p gigastt-core --lib benchmark_endpoint_duration -- --ignored --nocapture

GIGASTT_TEST_MODEL_DIR=/path/to/published/models \
GIGASTT_ENDPOINT_VARIANT=e2e_rnnt \
GIGASTT_ENDPOINT_PROBE=/private/output/e2e.json \
cargo test -p gigastt-core --lib benchmark_endpoint_duration -- --ignored --nocapture
```

For the baseline, the same test-only harness was appended to the baseline's
`inference/engine/live_probe/experiment.rs`, with the duration assignment removed
and the loop restricted to `[600]`. No baseline production code was modified.
The complete baseline harness file is hash-pinned in the summary. For each row,
`compute_wer(reference, text)` supplies the word-error count and reference count;
final events are retained by filtering `event.final`. Summaries include text hashes
so equal error counts cannot conceal changed hypotheses.

Additional local verification: workspace unit/bin tests, strict all-target Clippy,
37 model-backed WebSocket tests (including reconfigure and unchanged file JSON),
JS unit tests/typecheck, Go SDK tests and documentation drift checks. Hosted CI
was not dispatched. The unit tests also cover exact thresholds, frame rounding,
validation, per-session isolation and protocol 1.0–1.3 compatibility.
