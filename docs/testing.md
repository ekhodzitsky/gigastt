# Testing and validation

Run commands from the repository root after installing the
[native prerequisites](../.github/CONTRIBUTING.md#development).
All Actions workflows are manual; the commands below run locally. The Docs Drift
job fails manual CI when a checked documentation contract or relative link drifts.

## Focused development loop

Use Cargo's package, target and module filters for an ordinary edit. Do not
download a model or run the workspace suite on every iteration.

| Change | Model-free examples |
|---|---|
| Private tokenizer implementation | `cargo test -p gigastt-core --lib inference::tokenizer::tests` |
| WebSocket message contract | `cargo test -p gigastt-core --lib protocol::tests` and `cargo test -p gigastt --lib server::ws::` |
| Mel frontend hot path | `cargo test -p gigastt-core --lib inference::features::tests` and the focused benchmark below |

Select direct consumers as well when a public contract changes; for WebSocket
wire changes include the Go and JS SDK tests listed below and the relevant
model-backed E2E tests. Verify the selected filter runs tests, not just a
successful invocation with zero matches. Use `-- --list` to discover names.

For changes to the reused mel buffers, run the existing deterministic
synthetic-input Criterion benchmark:

```sh
cargo bench -p gigastt-core --bench mel -- 'mel_spectrogram/1s_16khz_reused'
```

A focused measurement supports local iteration, not a release performance
claim. Compare immutable revisions with equivalent workloads and record
same-binary controls where noise is material. When claiming inference
performance, also run the relevant
[whole-ASR comparison](performance-regressions.md#running-the-asr-check-locally).
An unstable timing result is inconclusive, not evidence of improvement or
permission to widen a threshold.

## Full model-free merge checks

```sh
make check
python3 scripts/check-docs-drift.py
python3 scripts/test_docs_drift.py
python3 scripts/check-publication.py
```

`make check` runs formatting, strict workspace/all-target Clippy and
`cargo test --workspace --lib --bins`. The custom WER benchmark defaults to
model-free self-tests; explicit `--run` is needed for corpus execution.
Unit tests use synthetic inputs;
model-dependent tests are explicitly ignored until selected below.

Before committing, run the full workspace checks above; the pre-commit hook
enforces the same Rust suite. Manual CI additionally checks protected statuses,
risk-specific model smoke and performance gates on the merge candidate.
A failing required check blocks merging. Enable `.githooks` with `git config core.hooksPath .githooks`. The publication
check inspects the staged tree; stage intended changes before the final check.

## Model-backed HTTP and WebSocket tests

Provision the required models before invoking ignored tests:

```sh
cargo run -- download
cargo test -p gigastt \
  --test e2e_rest --test e2e_ws --test e2e_errors \
  --test e2e_shutdown --test e2e_rate_limit --test e2e_jobs \
  --test e2e_cli --test e2e_admin_reload --test e2e_http_cov \
  -- --ignored --test-threads=1
```

Tests load real models; serial execution avoids multiplying memory use.
The shared server/readiness helpers live in
[`tests/common`](../crates/gigastt/tests/common). The
[transcription contract matrix](transcription-contracts.md) maps source types,
cancellation and output behavior to targeted regression tests.

Model smoke and quality gates are separate from a full WER experiment. See
[benchmark gates](benchmarks.md#pull-request-model-smoke-gate) and the
[performance acceptance policy](performance-regressions.md) for exact protocols.
The [whole-ASR comparison](performance-regressions.md#running-the-asr-check-locally)
measures both RNN-T heads against the accepted revision on the same host.
Its model-free failure-contract tests run with
`python3 scripts/test-asr-performance.py`.
A test that skips for missing assets is not evidence of model correctness.

## Load, soak and long-form quality

```sh
cargo test -p gigastt --test load_test -- --ignored --test-threads=1
GIGASTT_SOAK_DURATION_SECS=300 \
  cargo test -p gigastt --test soak_test -- --ignored --test-threads=1
```

Long-form stitch quality requires the RuLS corpus (a multi-gigabyte download):

```sh
python3 scripts/prepare_rulslib.py
cargo test --release -p gigastt --test longform_quality -- --ignored --test-threads=1
```

Keep missing-corpus skips visible. Do not substitute a different corpus and
retain the old acceptance claim. The stitch-cost limit is controlled by
`GIGASTT_LONGFORM_MAX_STITCH_PP`; changing it needs an explained quality decision.
See [long-form stitching](longform-stitching.md).

## Feature checks

```sh
cargo check --no-default-features
cargo check --features coreml
cargo check --features cuda
cargo check --features ane
cargo check --features candle
cargo audit
cargo deny check
```

Run feature builds separately: several backend combinations are intentionally
mutually exclusive. Compilation on a different OS does not prove the accelerator
runs there. Use the runtime smoke tests on the target hardware before claiming
provider compatibility or speed.

### Lean core tests

Check the raw-PCM-only core independently of the default server feature set:

```sh
cargo test -p gigastt-core --lib --no-default-features
cargo clippy -p gigastt-core --lib --tests --no-default-features -- -D warnings
cargo test -p gigastt-core --no-default-features --test lean_build --test corpus_duration_scan
```

The lean suite retains raw PCM and resampling, channel mixing, blocking pool,
model path and checksum, progress schema, and pure manifest resolution tests.
Tests that exercise file containers or native file wrappers require
`file-decode`; network download and reporting implementations require `net`;
async checkout tests require `async-pool`, independently of whether another
feature also brings in Tokio. Speaker download tests need both `net` and
`diarization`; ANE package extraction tests need both
`net` and `ane`. File-backed ignored integration probes require their actual
file-decode feature; the model-free manifest parser remains available without
it. The default-feature configuration retains every original test and assertion.

When changing these feature boundaries, repeat the lean library test and
strict library/test lint commands with `--features` set separately to
`file-decode`, `net`, `async-pool`, `diarization`, and `net,diarization`.
Run the corresponding `ane` and `net,ane` cases on macOS (the Objective-C
dependency does not compile on Linux). Keep `--no-default-features` in each command
so the required feature is tested without incidental support from the default
set. Do not invoke ignored model or corpus tests for this feature check.
For optional Candle backend tests, compile the `candle`-only test targets
without file decoding; actual file-based parity probes require both features.
Run the full default workspace checks as usual.

## Client SDKs

```sh
cd sdks/js
npm ci
npm test
npm run typecheck
npm run build
```

For Go, run `go test ./...` from `sdks/go`. Both SDK READMEs describe their opt-in
live tests. Native bindings have separate build and packaging requirements in
[quickstarts](quickstarts.md); they are not these WebSocket clients.

## Benchmarks and manual CI

For the intentionally long Golos WER harness:

```sh
cargo test -p gigastt --release --test benchmark -- --run
```

The local run prefers an installed external Golos corpus, falling back to the
bundled fixtures. Set `GIGASTT_BENCHMARK_MAX_SAMPLES` for a local diagnostic
subset (0 means unlimited); a missing model is explicitly reported as skipped,
not passed. To require complete, pinned bundled evidence and the downloaded
RNNT INT8 model, with no truncation or baseline updates:

```sh
cargo test -p gigastt --release --test benchmark -- --run --required
```

The hosted gate additionally checks the rounded WER against
`.github/wer-threshold.txt`. Only intentional local full-corpus runs may refresh
the baseline using `GIGASTT_BENCHMARK_UPDATE_BASELINE=1`.

Use [benchmark/README.md](../benchmark/README.md) for cross-engine evaluation,
normalization, datasets and reproducibility. Record exact model hashes before
comparing results; a canonical filename can refer to older locally cached weights.

Manual branch/full-main CI, protected statuses and publication checks follow
[manual CI operation](performance-regressions.md#manual-ci-operation).
Pushing a commit or tag does not start a workflow.

## Generated files and cleanup

Keep local Cargo output in `target/` (or set `CARGO_TARGET_DIR` outside the
checkout). Python bytecode, test caches, coverage output and virtual environments
are disposable local files; do not commit them. Stop builds before running
`cargo clean` to reclaim the current target directory.

Published files under `benchmark/results*/` are measurement evidence, not build
artifacts. Preserve them when cleaning a checkout. Completed series may move to
checksummed public archives only after retrieval and offline verification:

- [Russian FLEURS reading evidence](languages.md#reading-1000-recordings-per-language)
  is archived, with the originals retained for the multilingual publication audit.
- [Sidecar contention sweep](sidecar-contention.md#reproduction) is archived;
  its original raw results no longer occupy the current source tree.

Keep active publication-index inputs, model-pack acceptance evidence, test
fixtures and replay inputs until their consumers have a verified replacement.
Do not republish datasets with unresolved redistribution terms. Archive cleanup
reduces a fresh checkout, not existing Git history. Docker excludes these results
and local caches from its build context; benchmark containers use mounted datasets
and write fresh results at runtime. Source fixtures and benchmark manifests remain
in the context. Avoid `git clean -fdx`: it can delete unpublished research, local
configuration and downloaded inputs along with generated files.
