# Dependency security validation — 2026-10-25

Five vulnerability advisories and the `basic-toml` maintenance failure are removed
without adding advisory ignores. [Before](audit-before.json) and
[after](audit-after.json) share RustSec database commit
`21d909e4c8bd4e14c7aea58bf980c881a0053ab1`; both resolve 582 packages.
Workspace and fuzz audits and all Cargo Deny categories pass.
See [patch scope and provenance](../../../vendor/README.md).

The candidate is `e9fa93b0d4705c4a25c044f899b708557f81d01f`. The saved baseline
binary was built at `f29d057515a7a399a546f875e686f194b8c40b4d`; its `crates/`,
`Cargo.toml` and `Cargo.lock` match accepted main
`06ce99e680a915a10c947cd07d65281bb5e83a29` exactly.

## Recognition and resources

Seven frozen public recordings (19.2 minutes), three boundary offsets, both
RNN-T heads: **42 full JSON pairs exactly equal**, including word timestamps and
confidence. WER is unchanged: 7.2587% `rnnt`, 5.7357% `e2e_rnnt` (the repeated
offsets are not independent recordings). Both arms use the same verified model
files and existing optimized graphs, CPU, pool 1, two encoder threads, file
concurrency 1, punctuation/ITN off.

Seven alternating timing pairs per head and duration (56 requests), after warmup:

| Head / recording | Median paired latency change | Median RSS change | Median HWM change |
|---|---:|---:|---:|
| rnnt/long_example | +0.11% | +0.21% | -0.20% |
| rnnt/podlodka-306 | +0.02% | -0.80% | -0.20% |
| e2e_rnnt/long_example | -2.10% | +0.39% | -1.36% |
| e2e_rnnt/podlodka-306 | -0.15% | +0.04% | -0.45% |

Release binary size: 51,318,872 → 51,323,224 bytes (+4,352 bytes).
These are observed single-host measurements, not a speedup claim or a statistical
upper bound on regression. Background host load is recorded. Metal runtime
performance was not measured; its tensor kernels and model-loading code are unchanged.

[Report](report.json) includes provenance, every timing, quality counts and limits.
[Compressed paired responses](responses.json.gz) preserve all 84 complete bodies;
the report records compressed and uncompressed SHA-256 hashes.
Runner: [benchmark_stitch_case.py](../../../scripts/benchmark_stitch_case.py),
`--phase all --rounds 7`, using the pinned long-form manifest.

## Compatibility

- 1,256 workspace unit/bin tests passed, 48 model-dependent tests ignored.
- Strict workspace/all-target Clippy, formatting and documentation checks passed.
- Rust 1.94 checks passed for the workspace and patched Candle CPU crate.
- 17 Askama configuration tests and 6 Candle NPZ/PyTorch/safetensors tests passed.
- Python/Swift/Kotlin binding generation passed; Python syntax and native ABI import
  passed. Swift/Kotlin compilation/formatting was not tested locally.
- Actual HTTPS download and SHA-256 verification of the missing vocabulary succeeded.
- Docker patch availability regression failed before adding vendor COPY instructions
  and passed afterwards; two Docker-cache and seven manual-workflow tests passed.
- Vendored source trees were independently reconstructed from verified registry
  archives plus their declared patches and compared byte for byte.

GitHub CI status is recorded on the PR, not inferred from these local checks.
