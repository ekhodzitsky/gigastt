# Performance and dependency acceptance

A dependency update that changes an inference backend, model, frontend, codec,
thread policy or enabled Cargo features is a product behavior change. A matching
API, successful compilation, passing unit tests, or unchanged speaker labels
are insufficient evidence that it is safe to ship.

Before accepting such a change, identify the affected workloads and compare the
base revision with the candidate using the same assets, hardware, release
profile, thread/pool settings and options. Record cold load time, warm latency
and throughput, peak memory, quality and failures. Check concurrent requests
when pools or threads change. Include the dependency/feature delta and its
transitive build and distribution costs in the review. Unmeasured changes to a
runtime backend must not be merged as routine dependency maintenance.

Keep model and corpus checksums, code revisions, raw measurements, warmup and
aggregation rules with the results. A measured regression needs a fix before
release; do not remove a gate, widen a threshold, or replace the baseline just
to turn the checks green. New benchmarks need a comparable base measurement;
missing, renamed and removed benchmark results require investigation.

## Automated checks

- **Release Model** rebuilds both RNN-T heads from verified FP32 packaging
  sources and compares the pinned previous and candidate INT8 bundles on
  seven frozen long-form recordings. Aggregate WER, per-file phrase deletions,
  cold/warm latency, peak RSS and encoder size must pass before publication.
  The explicitly approved memory/size trade-off for the exact ConvInteger to
  float-Conv transition is recorded by model hash, not a blanket relaxation.
  See [the quantization protocol](rnnt-quantization.md).

- **Benchmarks (regression gate)** runs on every PR, including Dependabot, and
  every main push. It measures the immutable PR base or previous main commit
  and the candidate on the same runner with locked dependencies. It fails if
  either build/run fails, any comparison is absent or invalid, or the lower
  confidence bound of a Criterion mean increase exceeds 5%. It uses structured
  reports rather than interpreting the absence of a log message as success.
- **Speaker performance** downloads and verifies the 26 MB WeSpeaker model.
  Its release-mode test exercises real Golos speech prefixes of 0.5, 1.5 and
  3 seconds, short inputs, silence, concurrent pool users, offline turns and
  streaming feeds/finalization. Embeddings must match both the CPU oracle and
  frozen vectors (maximum component error below 0.001). Seven alternating
  timing rounds must have median candidate/oracle ratio at most 1.20.
  It also builds the same test-only production-path probe in clean base and
  candidate checkouts, each with its own locked dependencies. Seven alternating
  pairs of separate Linux processes must preserve embeddings and offline/streaming
  turns. Median paired increases above 20% for load, embedding, offline or streaming
  time, or above 10% for RSS/peak RSS, fail the check. This catches changes shared
  by the live oracle and candidate, including an ONNX Runtime upgrade.
- **Model smoke** explicitly requires both performance jobs to succeed. It
  runs even if a dependency fails, and fails itself in that case. This is
  necessary because a skipped required GitHub job can otherwise permit merge.
  Keep `Model smoke` required in branch protection.
- **Quality Gate (WER/RTF/RSS)** continues to exercise the bundled recognition
  fixtures on main pushes. A release requires this gate as well as the checks
  above, Format, Clippy and Unit Tests to succeed.

The release validator checks the latest `ci.yml` main-push run for the exact
release commit, including its latest attempt and named jobs. A successful run
for another SHA, an earlier green run followed by failure, or missing/skipped
jobs cannot authorize publication. GitHub binary/Docker releases and requested
npm/PyPI publication use this validator. Wait for main CI before tagging;
if CI is still pending, the release fails safely and can be rerun after success.
Direct manual publication remains an operator responsibility; run the same
validator before `cargo publish` or any publication outside these workflows.

## Running the speaker check locally

Provision the default model directory with `gigastt download` (without
`--skip-diarization`), then run:

```sh
cargo test -p gigastt-core --release --locked --lib \
  inference::diarization::model_tests::test_speaker_quality_and_latency_against_cpu_oracle \
  -- --ignored --exact --test-threads=1 --nocapture
python3 scripts/test-performance-gates.py
```

On Linux, run the mandatory comparison with an immutable base commit:

```sh
python3 scripts/compare-speaker-revisions.py --base <base-commit-sha>
```

Both builds complete before timing. The script adds the identical test-only
probe to temporary checkouts, without replacing either production implementation,
and retains raw process logs and `target/speaker-revisions/report.json`.
Embeddings use the same nine prefixes; offline and streaming timings include the
full pipeline on the concatenated clips with silence gaps. Each path is warmed
before timing. Model hashing warms the filesystem cache before measuring load.
All seven pairs must use matching assets, CPU counts and four speaker pool slots;
missing or invalid measurements fail. The same-source probe must compile against
both revisions; an incompatible test seam needs review, never a skipped comparison.

For an additional same-build diagnostic of the legacy backend:

```sh
for backend in production legacy-tract; do
  GIGASTT_SPEAKER_RESOURCE_BACKEND="$backend" cargo test -p gigastt-core \
    --release --locked --lib \
    inference::diarization::model_tests::test_speaker_resource_probe \
    -- --ignored --exact --test-threads=1 --nocapture
done
```

This measures the test process after nine three-second embeddings with four
pool slots. It includes the loaded native libraries, excludes the recognition
model and is not a full server memory measurement. Load time depends on the
filesystem cache; do not label a warm-cache process start as cold disk loading.

Missing or changed model assets fail the test. Do not run other CPU-heavy work
concurrently with timing. CI retains the speaker timings and Criterion reports.
The frozen vectors in
`crates/gigastt-core/src/inference/diarization/speaker-embeddings.f32le` contain
three consecutive 256-component little-endian f32 embeddings for the first
24,000 samples of `golos_00.wav`, `golos_01.wav`, and `golos_02.wav`. They were
computed with CPU ONNX Runtime (`ort` 2.0.0-rc.13), polyvoice 1.0's fbank/CMVN
frontend (byte-identical to 0.19 and 0.21), and the pinned WeSpeaker SHA-256
`3955447b0499dc9e0a4541a895df08b03c69098eba4e56c02b5603e9f7f4fcbb`.
Regenerating them requires an independently reviewed quality comparison.

## What these checks establish

The speaker oracle uses the pre-migration CPU budget (available CPUs divided
by four pool slots, at least one intra-op thread). Unset
`POLYVOICE_SESSION_POOL_SIZE` and `POLYVOICE_INTRA_THREADS` when running the
reference test; the production adapter continues to honor these existing
upstream tuning variables. Idle speaker workers do not spin while another
session or ASR inference uses the CPU.

The speaker oracle detects replacing the production CPU implementation with a
substantially slower backend, without depending on the speed of a particular
runner. The 20% allowance is for measurement variability; it is not permission
to introduce known slowdowns. Fixed vectors detect changes shared by the live
reference and candidate frontend. Offline and streaming comparisons preserve
the existing pipeline behavior on these fixtures.

These short clips are not a labeled meeting corpus and do not establish DER on
long meetings, overlap quality, all execution providers, or all CPU models.
Revision comparisons cover speaker latency and memory on these Linux fixtures;
they do not establish full-server peak memory with simultaneous ASR requests.
Changes affecting these areas require representative measurements in addition
to the automated gates. Timing tests cannot prove absence of every regression.

The Engine uses gigastt's CPU runtime for speaker inference. The historical
public `SpeakerEncoder` alias still names polyvoice's extractor, so its tract
feature remains for Rust source compatibility. Engine inference must not use
that alias; removing the public dependency surface belongs in a major release.
