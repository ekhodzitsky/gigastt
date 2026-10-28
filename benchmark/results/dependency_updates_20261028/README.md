# Dependency update validation — 2026-10-28

Integration implementation: `a0d38c4190ae6e5fe51dd9179842ea97c6c3e4fb`, against
accepted main `29c42e71606e25c539b5303c3aa9da00528b70d1`.
The saved baseline executable was built at `e9fa93b0d4705c4a25c044f899b708557f81d01f`;
its `crates/`, root manifest and lockfile exactly match that main revision.
[Report](report.json) records hashes, commands, model identity, every measurement
and limitations. [Full paired responses](responses.json.gz) retain all 84 bodies.

## Decisions and integration changes

| PR | Update | Reason and validation |
|---|---|---|
| #458 | Tokio 1.54.1 | Runtime/I/O fixes and timer-handle compatibility correction; server shutdown, saturation, REST, SSE, WebSocket and jobs tests pass. |
| #459 | ort rc.15 | Memory-safety fixes require explicit environment ownership. Shared process-lifetime ownership and an explicit host handoff are covered by lifecycle tests. Desktop native Runtime changes from **1.28.0 to 1.31.0**; Android's separate pin changes from **1.27.0 to 1.31.0**. Recognition/resource results are below. |
| #460 | opus-rs 0.1.39 | Decoder mode/channel transitions, SILK alignment and FMA dispatch fixes. Existing independent libopus reference tests pass; speech and timing checks below show no material regression in the sampled cases. |
| #461 | setup-python v7 | Update the remaining v6 step; it uses only supported `python-version` input. Workflow triggers remain manual-only. |
| #462 | napi 3.15.0 | Lifetime/GC/teardown fixes. Node 18.20.8 and 22.19.0 pass model-backed async, streaming, concurrent, event-loop, GC and outstanding-task ownership smoke tests. |
| #463 | uuid 1.28.0 | Parsing/ordering fixes and optional serialization additions; existing v4 job/request IDs and API tests pass. |

The unrelated itertools/toml resolver changes carried by the bot PRs are restored
to main's edges. Fuzz uses the same ORT and Opus versions; its stale anyhow and
yanked der are also aligned with the fixed workspace versions. No advisory
ignores are added. [Workspace audit](audit-workspace.json) has no vulnerability
findings and retains the pre-existing `paste` maintenance warning;
[fuzz audit](audit-fuzz.json) has no findings or warnings. Cargo Deny passes.

Upstream references: [ORT rc.15](https://github.com/pykeio/ort/releases/tag/v2.0.0-rc.15),
[ORT rc.14 migration](https://github.com/pykeio/ort/releases/tag/v2.0.0-rc.14),
[Tokio](https://github.com/tokio-rs/tokio/releases/tag/tokio-1.54.1),
[Opus changes](https://github.com/restsend/opus-rs/compare/b7bae9c42e3bcb4511a018a1f38930034dc71238...0f1728014d512f56eadcbfee58ce2ac4d9ffab9a),
[NAPI](https://github.com/napi-rs/napi-rs/releases/tag/napi-v3.15.0),
[UUID](https://github.com/uuid-rs/uuid/releases/tag/v1.28.0).

## Recognition and resources

Seven frozen public recordings (19.2 minutes), both heads, offsets 0/3/22 seconds:
42 paired cases. On the original recordings, error counts are unchanged:
197/2,714 words for rnnt and 158/2,714 for e2e_rnnt. Across all offsets:

| Head | Baseline WER | Candidate WER | Word errors |
|---|---:|---:|---:|
| rnnt | 7.2587% | 7.2709% | 591 → 592 / 8,142 |
| e2e_rnnt | 5.7357% | 5.7357% | 467 → 467 / 8,142 |

These are correlated repetitions, not 42 independent recordings. Some words,
confidence values and timestamps change; **response equality and improved
recognition accuracy are not claimed**. The one additional rnnt error does not
establish a general degradation or statistical equivalence. Broader corpora may
behave differently. The initial quality executable predates the additive host
initialization API and test/documentation changes; all four final-binary warmup
responses exactly match their corresponding quality-run responses. Both binary
hashes and this distinction are preserved in the report.

Seven alternating pairs for each head and duration (56 requests), CPU affinity
2/3, pool 1, two encoder threads, file concurrency 1, punctuation/ITN off. Each
runtime independently primes its own optimized cache before measurement.

| Head / recording | Median paired latency change | Paired RSS change | Paired HWM change |
|---|---:|---:|---:|
| rnnt / long_example | -9.25% | +1.38% | +1.16% |
| rnnt / podlodka-306 | -11.87% | -0.04% | -0.03% |
| e2e_rnnt / long_example | -11.62% | -0.79% | -0.73% |
| e2e_rnnt / podlodka-306 | -8.47% | -1.18% | -1.03% |

Executable size: 51,323,224 → 53,084,624 bytes (+3.43%). These observations are
from one Linux host, not a cross-platform speed guarantee. Background host load
is recorded. GPU/mobile execution and extended Node resource soak are not
measured locally; feature compilation is not device validation.

## Opus

Encode the in-tree `golos_00.wav` at 48 kHz using ffmpeg/libopus: 16/32 kb/s with
`-application voip`, and 96 kb/s with `-application audio`. Decode each container
through gigastt and compare with libopus decoded to float WAV, then passed
through the same gigastt resampler. Archived probe and Python harnesses preserve
the exact local experiment; their local paths need adapting for another host.
Hashes and all samples' metrics are recorded in the report.

The 96 kb/s decoded PCM is identical across revisions. At 16/32 kb/s the updated
SILK alignment shifts by one 16 kHz sample; aligned signal agreement with libopus
is essentially unchanged (about 13.54/13.71 dB for these fixtures). The common
container trimming/delay discrepancy remains; this update does not fix it.
On both ASR heads all three Opus encodings and their libopus WAV references have
the same normalized error counts before and after the update.

Initial decoder timings overlapped other checks and suggested roughly 7% slower
16/32/96 kb/s paths in some cases. The separate seven-pair confirmation after
ASR/Node checks gives +0.30% (96 kb/s), -2.36% (32 kb/s), +1.49% (16 kb/s).
Both runs are retained; the initial noisy slowdown is not treated as a confirmed
regression. These short fixtures do not cover every Opus mode or transition.

## Checks

- Workspace unit/bin tests, strict all-target Clippy and formatting pass.
- Rust 1.94 workspace and default/lean/CoreML/CUDA/NNAPI compile checks pass.
- 126 model-backed API/CLI tests pass across the serial E2E suites. An obsolete
  automatic-quantization test was replaced with a passing model-free test of
  the current offline/INT8 contract. The first diarization attempt lacked the
  speaker asset; the affected suite passed with the verified speaker model.
- Android AAR SHA-256, header API 31 and all three ELF ABIs pass validation.
- Fuzz dependency graph compiles; workspace/fuzz audits and Cargo Deny pass.
- Both supported Node test versions pass again against the final addon.

This runtime upgrade is deferred after the paired CoreML failure below. Existing strict
comparison gates are not relaxed: the archived exploratory collection wrapper
records runtime numerical differences for review, while production CI retains
its recognition, speaker and performance checks.

## First full CI findings

The first full manual run ([40670427842](https://github.com/ekhodzitsky/gigastt/actions/runs/40670427842))
exposed a macOS static CoreML link failure: `__isPlatformVersionAtLeast` was
unresolved in ONNX Runtime 1.31's CoreML objects. A target/feature-gated core build
script now links the selected Apple Clang compiler runtime; SDK discovery,
missing-archive/error handling and unaffected configurations have local tests.
This changes macOS linking only; Linux measurement binaries above are unaffected.
The semver job separately lost its download of the **baseline** ORT 1.28 archive
(`Peer disconnected`); it did not report an API incompatibility. The semver
action now skips native ORT downloads for its rustdoc-only comparison. Runtime
build and inference jobs still download, link and execute the real library.
These failures are retained rather than treated as a successful CI run. The
updated head requires another manual validation before merge.

The manual CI also exposes `coreml_smoke` for a focused follow-up: it runs the
normal required checks plus real CoreML inference, without repeating all full
E2E/coverage/quality/Docker jobs after a macOS-only link correction. `full=true`
continues to include the same CoreML smoke. Neither input enables automatic CI.

## Follow-up validation

[The second manual CI run](https://github.com/ekhodzitsky/gigastt/actions/runs/40673817532)
passed the non-CoreML checks, including semver, model smoke, speaker performance
and the benchmark regression gate. The 14 microbenchmark mean changes range
from -4.91% to +1.47%; the earlier incomplete run's mel slowdown did not repeat.
Seven paired speaker measurements give median candidate/base ratios of 1.00643
for embedding, 1.00592 for offline diarization, 1.00613 for streaming, and 0.99465
for RSS/peak RSS. These small changes do not establish a speed improvement.

Additional local `ml_ctc` and `ml_ctc_large` CPU smoke checks on `golos_00.wav`
succeed for both revisions and produce the same reference text. Word-level
numeric fields differ. This is one short fixture, not a multilingual WER study;
raw responses and their hashes are retained alongside `followup-summary.json`.
The Linux binaries are the same baseline/final binaries described above.

CoreML now links, but its encoder probe fails at inference on macOS 14.8.9.
[The focused diagnostic run](https://github.com/ekhodzitsky/gigastt/actions/runs/40678427152)
retains the error: `Unable to compute the prediction using a neural network model`.
Its attempted CLI CPU control was invalid (`transcribe` has no execution-provider
flag) and supplies no CPU evidence. Subsequent comparisons must use the explicit
provider API. The smoke harness now prints errors under `bash -e`, checks both
output streams for fallback warnings, and retains failure artifacts.

The warning about requiring macOS 14.4 is misleading upstream text: the native
[availability macro](https://github.com/microsoft/onnxruntime/blob/402979ac1bef89dcd55add41eaa7277c407599eb/onnxruntime/core/providers/coreml/model/host_utils.h)
actually requires macOS 15 for CoreML 8 optimization hints. The warning alone
is not evidence that the compiler-runtime availability implementation is broken.
The paired investigation confirms an introduced regression, so the ORT update
is deferred. The other five updates are isolated in [PR #465](https://github.com/ekhodzitsky/gigastt/pull/465).

## Confirmed CoreML regression

[Paired run 40681440536](https://github.com/ekhodzitsky/gigastt/actions/runs/40681440536)
uses the same macOS runner, release RNNT model and `golos_00.wav`. The small
public-API probe requests each execution provider explicitly, preventing an
automatic CPU fallback from counting as CoreML success. It uses one pool slot
and two encoder threads; punctuation/ITN are not initialized.

| Revision | Explicit CPU | Explicit CoreML |
|---|---|---|
| main `29c42e7`, ort rc.13 / native 1.28 | Pass | Pass |
| candidate runtime code, ort rc.15 / native 1.31 | Pass | Encoder prediction fails |

All successful arms recognize `шестьдесят тысяч тенге сколько будет стоить`.
Logs, exit statuses and the probe source are retained in `coreml-comparison/`.
The diagnostic branch `10f3323` contains the candidate runtime code from
`e7e4896` plus only a workflow/probe addition; the baseline is checked out at
its exact SHA and compiled using the same probe. Each native runtime gets its
version-scoped CoreML cache. The baseline passes after the candidate fails on
the same runner, ruling out a generally unusable CoreML runner as the cause.

The exact failing CoreML operation is not yet isolated. Linking is repaired,
but that does not make the native runtime safe to upgrade. The Linux speed
improvement does not justify introducing a failure on a supported backend.
No CPU fallback, relaxed quality threshold or newer-macOS-only gate is used to
make this update pass. PR #464 remains a draft investigation, not a merge candidate.
