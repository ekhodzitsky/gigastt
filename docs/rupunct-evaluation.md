# Native RUPunct evaluation

> Measurement correction: audio/Engine results below used older locally cached
> RNN-T encoders, not the active October quality bundle. They are historical
> and must not determine current-head selection. See the
> [release-pinned comparison](current-asr-comparison.md) for the identity check
> and corrected campaign. Isolated text-restoration measurements are independent
> of ASR weights, with the documented ONNX alignment caveat.


This experiment compares the accepted gigastt ONNX punctuation path with
[rupunct 0.19.1](https://github.com/ekhodzitsky/rupunct-rs/releases/tag/v0.19.1).
Standard model selection/downloads remain unchanged. The native adapter is
available only with the optional core `rupunct` Cargo feature and explicit API
attachment:

```rust,ignore
let model = gigastt_core::punctuation::NativePunctuator::load(model_dir)?;
let engine = engine.with_native_punctuator(Some(model));
```

Attach this adapter to bare-text heads under an explicit caller policy. It does
not infer the CLI auto policy; feeding already punctuated text directly can
duplicate punctuation, as the [head comparison](punctuation-heads-evaluation.md) shows.

The library consumes locally provisioned `model.safetensors`, `tokenizer.json`
and `config.json`. It neither fetches weights nor selects a model implicitly.
The adapter preserves gigastt's fail-soft policy: restore failures log a warning,
increment `failed_restores()`, and return the original input. Request punctuation
overrides, ITN ordering, and cancellation boundaries remain in the shared Engine
postprocessing path. Native restore calls are synchronous and cannot be interrupted
inside inference. The native library caps input at 8 MiB and returns a whole-call
error, whereas the existing ONNX restorer can retain successful individual windows.

For actual audio recognition, explicit head selection and reprocessing of
`e2e_rnnt`, see the [head comparison](punctuation-heads-evaluation.md). Its
alignment-aware metrics differ from the gold-input metrics below.

## Results and recommendation

Keep ONNX small as the standard low-latency path. Native small reduces RAM, but
its gold-input quality gain below does not transfer to the actual `rnnt` output
in the [audio comparison](punctuation-heads-evaluation.md). Big improves actual
bare-head punctuation, but is an explicit accuracy/compute trade-off;
it is not a speed upgrade. The experiment adds an opt-in Engine API, not a CLI
backend switch or automatic replacement in releases.

All numbers below are local Linux x86_64 release-build measurements on an AMD
Ryzen AI 9 HX 370. Four fast physical cores are allocated equally to every
process. [Models, crate/executable hashes and hardware](measurements/rupunct-20261022/models-and-protocol.json),
[metric definitions and input hash](measurements/rupunct-20261022/input-protocol.json),
and [structured summary](measurements/rupunct-20261022/summary.json) accompany them.

| Path | Punctuation F1 | Capitalization F1 | Exact sentences | Required model files, MiB |
|---|---:|---:|---:|---:|
| Existing ONNX small | 0.7357 | 0.8379 | 34.45% | 30.21 |
| Native small Q8 | 0.7773 | 0.8938 | 35.87% | 32.59 |
| Native big Q8 | 0.8384 | 0.9330 | 49.55% | 184.03 |

No lexical mutations were found in any of the 775 outputs per backend. The
native small gain is **4.16 percentage points of punctuation F1**; big gains
10.27 points. Big changes the checkpoint as well as the execution path. These
numbers include an existing ONNX word-alignment limitation on internal punctuation.
Most of the small-model gain concentrates in hyphen-bearing gold inputs, absent
from actual bare-head outputs here; see the [stratified audio audit](punctuation-heads-evaluation.md#why-the-earlier-small-model-gain-does-not-transfer-directly).
They do not establish that the native compute kernel alone improves accuracy.
They also are not directly comparable with the library README's different
corpus/word-label metric. Per-sentence outputs:
[ONNX](measurements/rupunct-20261022/onnx-small-quality.jsonl),
[native small](measurements/rupunct-20261022/native-small-quality.jsonl),
[native big](measurements/rupunct-20261022/native-big-quality.jsonl).

With one Rayon worker for native GEMM, median **punctuation-only** request latency
and peak RSS across the four text lengths are:

| Path | 5 words, ms | 30 words, ms | 1,000 words, ms | Peak RSS: 1 / 2 / 4 requests, MiB |
|---|---:|---:|---:|---:|
| Existing ONNX small | 0.28 | 0.80 | 41.40 | 109 / 135 / 186 |
| Native small Q8 | 1.31 | 2.51 | 146.89 | 61 / 70 / 88 |
| Native big Q8 | 37.66 | 76.61 | 2,179.99 | 189 / 218 / 273 |

Native small saves about **44–53% peak RSS** in this punctuation-only workload,
while its model files are about 8% larger. Big's files are about 6.1 times the
standard model pack size. RSS includes allocator retention and mapped pages;
these are not full-server RAM reduction percentages. Native Q8 uses packed
weights with tiled dequantization/F32 GEMM; ONNX uses dynamic INT8. Source review
identifies that implementation difference, but no kernel profile was used to
attribute the slowdown to one specific operation.

Automatic Rayon helps a single large request: small's 1,000-word median falls
from 147 to 127 ms, big's from 2,180 to 1,193 ms. It still loses to ONNX's 41 ms.
At the standard **two-request concurrency**, a pair of 30-word restores takes
about 2.6 ms with native small/one Rayon worker and 3.3 ms with automatic Rayon;
the initial ONNX bracket takes 1.9 ms. Single and two-request performance do
not justify default replacement.

At four requests, native small with one Rayon worker completes a 250-word batch
in 30 ms versus ONNX's 38 ms, an example of a conditional throughput advantage.
Automatic Rayon makes that native batch take 41 ms instead. Across the identical
four-request mixed-length suite, measured CPU time is 1.7 s for ONNX, 6.0 s for
native small, and 92–94 s for big. Those figures include warmups, model load and
all suite calls, not the ASR encoder or a per-request CPU guarantee. Observed
peak thread counts range from 5 for ONNX to 6/9 for native one/auto configurations.
More CPU consumption can compete with recognition even when a punctuation batch
finishes sooner. The adapter therefore does not silently set a global Rayon
policy on behalf of other users of the process.

Loading with a warm filesystem cache takes approximately 42 ms for ONNX,
32–37 ms for native small and 56–58 ms for big. First inference then takes
0.4 ms, 4–5 ms and 42–43 ms respectively. A faster mmap/load call must not be
reported as faster model readiness without accounting for that first inference.
[Process resources and both ONNX brackets](measurements/rupunct-20261022/timing-processes.json)
and [all seven-round measurements](measurements/rupunct-20261022/timings.jsonl)
are retained; these are exploratory observations, not a platform-wide release gate.

## Integrated recognition

All fifteen repository speech recordings were transcribed through the actual
Engine with each of the three punctuation paths. Word strings, confidence values
and timestamps matched, including their canonical JSON hashes; alphanumeric text content
also matched. [Per-recording outputs and word-array hashes](measurements/rupunct-20261022/asr-contracts.json)
record these checks. Golos lacks punctuation gold, so this verifies the ASR
contract rather than assigning punctuation accuracy to those recordings.
The complete Engine process with pool=1 peaks at **330.5 MiB** for ONNX small,
**295.9 MiB** for native small and **384.9 MiB** for native big. Native small's
saving in this integrated workload is therefore about 10.5%, rather than the
44–53% punctuation-only reduction; big increases the integrated peak by 16.5%.
These processes do not run the HTTP server or concurrent speech recognition.


A second check concatenates those recordings in order into 61.765 s of PCM16
mono audio at 16 kHz. Three independent processes per backend alternate/reverse
order, using identical RNN-T models and pool=1. The twelve runs again preserve
words/timestamps. Total transcription ranges overlap:

| Path | Median total, s | Min–max, s | Median RTF |
|---|---:|---:|---:|
| Existing ONNX small | 7.43 | 6.93–7.59 | 0.120 |
| Native small, one Rayon worker | 7.17 | 6.60–7.80 | 0.116 |
| Native big, one Rayon worker | 7.12 | 6.98–7.14 | 0.115 |
| Native big, automatic Rayon | 6.98 | 6.63–7.01 | 0.113 |

With only three runs and visible host variability, these medians **do not prove
an end-to-end speed improvement**. They do show that punctuation-stage slowdown
must not be multiplied directly into whole-transcription RTF. The faster isolated
ONNX punctuation remains the lower-overhead choice. [Every long-run result](measurements/rupunct-20261022/long-asr.json)
is included. No claim is made about live-server p99 latency or concurrent ASR +
punctuation interference; the concurrent benchmark isolates the shared punctuation
component. macOS/Accelerate, ARM, Windows, cold filesystem cache and representative
spontaneous-conversation quality were not measured.

## Integration and dependency costs

The default feature set adds no enabled rupunct dependencies. Enabling it adds
22 packages to this Linux normal-dependency graph, including four proc-macro
packages, gemm and Rayon dependencies that were already present only in the
lockfile for other optional backends. The crate itself has three direct
dependencies; that does not make the activated transitive graph three packages.
No ONNX Runtime dependency can be removed from gigastt because speech inference
still uses it. Server distribution-size and clean-build-time deltas were not
measured; this API experiment does not expose native model selection in the CLI.

The native adapter preserves newlines/tabs, while the legacy restorer normalizes
whitespace. That is useful for direct text consumers, but ordinary ASR output is
already space-separated. Its 8 MiB input cap and whole-call fallback differ from
the legacy partially successful window behavior. Those limits must be considered
before a production migration. Local model-backed tests exercise fail-soft size
rejection, formatting, punctuation overrides, ITN order, cancellation and builder
replacement. Workspace tests and strict Clippy pass with the feature both on and
off. Scoring tests verify position stability, lexical-mutation handling and rejection
of missing/duplicate outputs. No automatic workflow trigger was introduced.

## Protocol

Base: gigastt `def4f223eac28cbdc5372f06c74f942cddfa34bd`. Pin crates.io rupunct
exactly to `0.19.1`; no local path patch is used. Compare the existing
`RUPunct_small` dynamic-INT8 ONNX model, locally provisioned small Q8 safetensors,
and big Q8 safetensors. Small shares the same tokenizer/config files across both
paths, but quantization and long-text windowing differ: this is not a pure
compute-kernel benchmark or proof of floating-point equivalence.

Quality uses all 775 existing [FLEURS](https://huggingface.co/datasets/google/fleurs) ru_ru test references in
`benchmark/manifests/fleurs_ru_punct.json` (CC BY 4.0, Conneau et al., 2022).
Lowercase and remove sentence punctuation, preserving internal hyphens and decimal
dots. Score punctuation characters and uppercase letters at alphanumeric character
boundaries, so inserting a comma cannot shift subsequent matches. Lexical changes
fail the row conservatively; missing/duplicate outputs fail evaluation. Exact match
normalizes whitespace only. These are text restoration tests on read-speech gold,
not WER on noisy recognition, spontaneous conversation or unseen-domain accuracy.
The corpora may overlap checkpoint training. Golos has no punctuation gold and is
used only for recognition/word/timestamp regression checks.

On Linux the native library's GEMM uses a global Rayon pool **independently of**
`RestoreOpts.threads`. The adapter sets window workers to 1, but does not change
Rayon's global pool or process environment. Measure native with
`RAYON_NUM_THREADS=1` and with that variable absent. On this 24-logical-CPU hybrid
host, pin the timing processes to CPUs 0–3 (four equal-class physical cores).
Automatic Rayon therefore sees a four-CPU allocation. This does not qualify
unrestricted default scheduling or the Accelerate backend on macOS.

Timing compares 5, 30, 250 and 1,000 words, one, two and four simultaneous requests
sharing one loaded model, two warmups, then seven rounds. The 1,000-word input is
an artificial concatenation for window/resource testing, not a quality reference.
Record individual request latency, round wall time, process CPU time and observed
thread count. Baseline runs bracket each concurrency campaign. Do not run builds
concurrently with timing. Model load is measured with warm filesystem cache;
first inference is a separate field, not an OS-cold-start guarantee.

Memory snapshots distinguish loaded/active models from post-scope drops. Linux
VmRSS/VmHWM include file-backed pages and allocator caches; they are not physical
memory attributable exclusively to model weights. Timing/quality processes use
punctuation only; integrated ASR probes also load the same RNN-T model with pool=1.

## Reproduction

```sh
cargo build --release --locked -p gigastt-core --features rupunct --example punctuation_probe
python3 scripts/compare-punctuation.py --prepare /tmp/punct-comparison
python3 scripts/compare-punctuation.py --measure /tmp/punct-comparison \
  --probe target/release/examples/punctuation_probe \
  --small ~/.gigastt/models/punct --big ~/.rupunct/RUPunct_big --cpus 0-3
RAYON_NUM_THREADS=1 taskset -c 0-3 target/release/examples/punctuation_probe \
  quality onnx ~/.gigastt/models/punct /tmp/punct-comparison/quality-inputs.json /tmp/punct-comparison/onnx-quality.json
RAYON_NUM_THREADS=1 taskset -c 0-3 target/release/examples/punctuation_probe \
  quality native ~/.gigastt/models/punct /tmp/punct-comparison/quality-inputs.json /tmp/punct-comparison/native-small-quality.json
RAYON_NUM_THREADS=1 taskset -c 0-3 target/release/examples/punctuation_probe \
  quality native ~/.rupunct/RUPunct_big /tmp/punct-comparison/quality-inputs.json /tmp/punct-comparison/native-big-quality.json
python3 scripts/compare-punctuation.py --score /tmp/punct-comparison/quality-inputs.json /tmp/punct-comparison/native-big-quality.json
python3 scripts/test-compare-punctuation.py
GIGASTT_NATIVE_PUNCT_MODEL=~/.gigastt/models/punct RAYON_NUM_THREADS=1 \
  cargo test --locked -p gigastt-core --features rupunct --lib \
  test_native_punctuation_request_controls_and_lexical_preservation -- --ignored
```

`--prepare` also generates the concatenated WAV and `long-asr-inputs.json`.
The probe's `asr` mode takes either `asr-inputs.json` or `long-asr-inputs.json`,
followed by the ASR model directory as its
final argument. It attaches each punctuation backend to the actual Engine and
records full results, including words/timestamps, for all fifteen repository
speech fixtures. Raw outputs are checked against the ONNX run; a mismatch is a
regression, not silently normalized away.

CI remains manual-only. Model-free optional adapter tests/lints and scoring tests
run when CI is explicitly dispatched. Models and model-backed evaluations stay
local; no GitHub Actions run is required by this experiment.

Local verification: **1,237** default and **1,240** native-feature workspace
unit/bin tests passed; both strict workspace/all-target Clippy configurations,
standalone optional-core Clippy, rustfmt, Docker cache target checks and docs
drift checks passed. The explicitly requested model-backed contract test passed
separately; it is ignored in ordinary CI and fails if weights are missing.
[Validation counts](measurements/rupunct-20261022/validation.json) exclude ignored
tests from passed totals. GitHub Actions were not dispatched.
