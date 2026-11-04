# Final local acceptance comparison

The full frozen corpus and resource comparison was repeated after the seam-case
audit and real-audio reproductions. This run supports accepting the scoped change
on the measured CPU configuration. It is not a universal performance guarantee.

## Results

All **42 complete baseline/candidate responses are byte-identical**. Normalized
WER remains 7.2587% for `rnnt` and 5.7357% for `e2e_rnnt`; case, punctuation,
word timings, confidence and response metadata match. The three demonstrated
improvements belong to the separate [targeted reproduction set](../seam_reproduction_20261024/README.md).
Its candidate binary SHA-256 is identical to this run's rebuilt candidate.

All **168 timed responses** exactly match their corresponding warmup response.
Each group contains 21 alternating AB/BA pairs, fixed before the run began.
The paired change is the median of candidate/baseline ratios, not a ratio of
marginal medians. Every measurement, including outliers, is retained.

| Head / input | Baseline median seconds | Candidate median seconds | Median paired change | Paired bootstrap 95% interval |
|---|---:|---:|---:|---:|
| rnnt/long_example | 1.843 | 1.855 | +0.67% | -0.46% to +1.91% |
| rnnt/podlodka-306 | 6.721 | 6.870 | +0.65% | -0.92% to +2.54% |
| e2e_rnnt/long_example | 1.884 | 1.902 | +0.52% | -1.01% to +1.54% |
| e2e_rnnt/podlodka-306 | 7.405 | 7.351 | +0.38% | -2.93% to +2.59% |

Every interval includes zero change and has an upper bound below +5%.
The point estimates do not demonstrate a speed improvement or a reliable slowdown.
Intervals use 10,000 paired-median resamples, seed 455, with sorted indices 250
and 9750; they assume exchangeable pairs and describe this run's variability.
These are per-group descriptive intervals, not a simultaneous cross-platform bound.
The earlier seven-pair run remains published with its wider uncertainty.

Median retained RSS is 1.98–2.46% higher; the largest relative HWM increase is
2.13%. No growth trend appeared during repeated requests. RSS/HWM are process
resident/high-water values, not unique physical footprint; shared mappings and
allocator retention are not isolated. This modest observed cost is recorded,
not hidden or presented as a memory improvement.

## Protocol and provenance

- Baseline: `e3ba93e838712b93551cca6037f52c2a37cf95da` (`main`).
- Measured candidate: `4b31c958fe3717375beb6aeb0befe7784d69acb5`.
- Candidate rebuilt with `cargo build --release -p gigastt --bin gigastt`;
  SHA-256 `f15c72f44076e3c55a2a4a29371ff93528b8e3f318a15b71eed29aa40b48400c`.
- Subsequent changes only move a test fixture into its published crate, fix
  manual publication validation for fork PRs, and add this report; inference
  code, dependencies, model assets and runtime options are unchanged.
- Both default-feature CPU release binaries use identical Cargo.lock/model
  assets, pinned encoder checksums, two encoder threads, pool size 1, serial
  file windows, external punctuation and ITN off.
- Same seven frozen recordings, leading silence 0/3/22 seconds, both heads.
  Distinct source audio totals 19.23 minutes; shifted copies are not independent.
- AMD Ryzen AI 9 HX 370, two equal-type physical fast cores (2,3). Primed ORT
  cache followed by fresh measured processes; cache-hit log evidence retained.
- No heavy local builds/tests were run alongside timing. This was still a
  desktop workstation, without exclusive CPU isolation; system load is recorded.
- Timing: 71.25-second `long_example` and 243.657-second `podlodka-306`, 21 pairs
  after warmup. Warmups compare both arms; requests are sequential.
- systemd user service: MemoryMax=6G, MemorySwapMax=0, RuntimeMaxSec=2400, Nice=5.
  Completed successfully in 20m34s. No inference request failed.
- Startup measurements are descriptive warm-filesystem process starts; they
  are not cold-disk startup comparisons.

See [the original protocol and corpus limitations](../stitch_case_20261024/README.md)
for input sources, joins, references and normalization. No source audio is
redistributed. Original-author audio is still unavailable; the public examples
establish the mechanism independently, not the author's exact recording.

## Reproduction and evidence

Run `scripts/benchmark_stitch_case.py` with the two binaries and revisions above,
the frozen audio/model directories, a fresh output directory, `--rounds 21` and
`--cpus 2,3`. The exact script checksum, binary/model checksums and commands are
in `metadata.json` and `servers.json`. Model-free helper tests remain available.

`quality.json`, `timings.json`, `summary.json`, `paired-analysis.json`,
`response-checksums.json` and `cache-load-evidence.json` retain the evidence.
Full responses, logs and binaries are kept locally under
`~/.cache/gigastt/seam-final-20261024/`. To recompute intervals and response hashes,
run `python3 benchmark/results/seam_final_20261024/analyze.py /path/to/full-run`.
This requires the saved full responses as well as the published timing rows.
