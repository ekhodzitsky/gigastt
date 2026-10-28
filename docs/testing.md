# Testing and validation

Run commands from the repository root after installing the
[native prerequisites](../.github/CONTRIBUTING.md#development).
All Actions workflows are manual; the commands below run locally. The Docs Drift
job fails manual CI when a checked documentation contract or relative link drifts.

## Fast checks without model files

```sh
make check
python3 scripts/check-docs-drift.py
python3 scripts/test_docs_drift.py
python3 scripts/check-publication.py
```

`make check` runs formatting, strict workspace/all-target Clippy and
`cargo test --workspace --lib --bins`. Do not use bare `cargo test` or
`cargo test --workspace`: the custom WER benchmark has `harness = false`, so
`--ignored` does not prevent its long run. Unit tests use synthetic inputs;
model-dependent tests are explicitly ignored until selected below.

Enable `.githooks` with `git config core.hooksPath .githooks`. The publication
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
cargo test -p gigastt --test benchmark -- --ignored
```

Use [benchmark/README.md](../benchmark/README.md) for cross-engine evaluation,
normalization, datasets and reproducibility. Record exact model hashes before
comparing results; a canonical filename can refer to older locally cached weights.

Manual branch/full-main CI, protected statuses and publication checks follow
[manual CI operation](performance-regressions.md#manual-ci-operation).
Pushing a commit or tag does not start a workflow.
