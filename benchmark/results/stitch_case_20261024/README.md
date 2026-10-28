# Local seam-casing validation

This comparison evaluates the runtime change in PR #455 against its `main`
ancestor. It does not qualify unrelated model, decoder or punctuation changes.

A [follow-up audit](audit.md) tightened repetition guards and benchmark validation.
The measurements below remain tied to runtime revision `36ddb57`; they are not
new measurements of that follow-up.

A later [targeted real-audio search](../seam_reproduction_20261024/README.md)
reproduced three corrected casing failures on individual public source excerpts.
That enriched diagnostic set is separate from the unchanged 42-pair corpus below.

A [final 21-pair-per-group comparison](../seam_final_20261024/README.md)
repeated the full corpus on the audited runtime and reduced latency uncertainty.
The historical measurements below remain unchanged.

## Quality result

All **42 baseline/candidate responses were identical**, including case,
punctuation, word count, timestamps, confidence and other metadata. No real-audio
case correction was exercised by this corpus. This supports no observed regression
on these inputs; it does **not** demonstrate that the reported window-capitalization
problem is fixed in practice. A real positive reproduction is still needed before
presenting this PR as a demonstrated quality improvement.

| Head | Reference words across shifts | Baseline WER | Candidate WER | Changed words |
|---|---:|---:|---:|---:|
| rnnt | 8,142 | 7.2587% | 7.2587% | 0 |
| e2e_rnnt | 8,142 | 5.7357% | 5.7357% | 0 |

On the cased podcast references, `e2e_rnnt` has 234 initial-case mismatches on
6,412 lexically aligned words in both arms. Its capitalization TP/FP/FN counts
are 473/120/114 in both arms. The bare `rnnt` control deliberately has external
punctuation/casing disabled. These scores compare revisions **within each head**;
they are not a new model-selection benchmark or the project's headline WER.

## Timing and memory result

All 56 measured warm responses matched their warmup responses exactly. The
following are medians of seven observations per arm. Paired change is the median
of seven candidate/baseline ratios, **not** the ratio of the two marginal medians.
These estimators can differ when background load changes between requests.

| Head / input | Baseline seconds | Candidate seconds | Median paired change | Paired bootstrap interval |
|---|---:|---:|---:|---:|
| rnnt/long_example | 2.148 | 2.127 | -0.11% | -4.35% to +9.04% |
| rnnt/podlodka-306 | 7.886 | 6.964 | -7.72% | -14.56% to +2.35% |
| e2e_rnnt/long_example | 2.405 | 2.507 | +0.50% | -0.14% to +5.97% |
| e2e_rnnt/podlodka-306 | 8.157 | 8.054 | -0.52% | -8.33% to +10.34% |

The paired point estimates do not show a material slowdown. They also do not
establish a strict 5% non-inferiority bound: background work produced substantial
spread, and several descriptive bootstrap intervals extend above +5%. The
intervals resample the seven observed pairs 10,000 times with seed 455 and assume
exchangeable pairs; they are not a substitute for a quiet dedicated runner.
**No speed improvement or unconditional performance acceptance is claimed.**

| Head / input | Baseline RSS / HWM, MiB | Candidate RSS / HWM, MiB |
|---|---:|---:|
| rnnt/long_example | 526.7 / 656.0 | 553.7 / 647.3 |
| rnnt/podlodka-306 | 528.7 / 656.0 | 569.2 / 666.4 |
| e2e_rnnt/long_example | 560.5 / 662.8 | 562.1 / 665.2 |
| e2e_rnnt/podlodka-306 | 560.5 / 662.8 | 564.5 / 665.2 |

Maximum observed HWM increase is 1.58%; retained RNN-T RSS is about 5–8% higher
in the candidate processes. This single process pair per head does not isolate
allocator retention or shared-mapping effects from a code effect. These are
reported measurements, not evidence of a memory optimization.

## Evidence

`quality.json` records all 42 paired scores; `response-checksums.json` confirms
byte-identical saved JSON responses. `timings.json` contains every wall/CPU/RSS
measurement and system load, including outliers. `paired-analysis.json` preserves
the raw paired ratios and descriptive intervals. `metadata.json`, `servers.json`,
`cache-load-evidence.json` and `cpu-topology.txt` record assets, commands and host.
`excluded-initial/` retains the preliminary measurements and their exclusion reason.
Full responses, logs and binaries remain in the local measurement directory
`~/.cache/gigastt/stitch-case-20261024/`; no source audio is copied into this report.

## Protocol

- Baseline: `e3ba93e838712b93551cca6037f52c2a37cf95da`.
- Candidate runtime: `36ddb578a72d933171db428fe1da754e80d61529`.
- Both binaries: default-feature CPU release builds, workspace release profile
  (LTO, one codegen unit), identical Cargo.lock and model assets.
- AMD Ryzen AI 9 HX 370, Linux x86_64, affinity to physical fast cores 2 and 3;
  two encoder threads, pool size 1, serial file windows, external punctuation
  and ITN off. `e2e_rnnt` retains its built-in formatting.
- Current October release INT8 encoders are checked against their pinned
  SHA-256 values before running. Every model and binary checksum is recorded.
- Prime the ORT optimized disk cache, stop that server, then start **both**
  measured servers from the same existing cache. Fail if either server's log
  does not confirm the cache fast path. Wait for `/ready`, not `/health`.
- Quality: the seven files in the [frozen manifest](../../manifests/longform_stitch.json),
  with 0, 3 and 22 seconds of leading silence, on both RNN-T heads: 42 matched
  input/configuration pairs. The unshifted source material totals 1,154.03
  seconds (19.23 minutes); shifted versions are not independent recordings.
- Latency: 71.25-second `long_example` and 243.657-second `podlodka-306`;
  seven paired AB/BA rounds per head/file after one discarded warmup per arm.
  Requests are sequential. Wall time covers the REST request through parsed
  JSON response. CPU time is the server process's user + system time.
- Every repeated timing response must exactly match its warmup response.
  Every baseline/candidate quality pair must preserve word count, lowercase
  text including punctuation, timestamps, confidence, speaker fields and other
  response metadata. All changed word spellings are retained for review.
- Normalized WER uses lowercase Unicode words and maps `ё` to `е`. Casing is
  additionally scored on lexically aligned matching podcast-reference words,
  so insertion errors do not shift the remaining capitalization comparisons.
  Case-sensitive word edit counts are also retained. Poetry references are
  excluded from sentence-capitalization scoring because line capitalization
  is an editorial convention.

This is a local workstation comparison, not an isolated benchmark appliance.
Other desktop work may run; raw paired measurements, CPU time and system load
are retained. Small differences should not be presented as a speed improvement.
RSS/HWM are Linux process resident/high-water values, not physical footprint.
Startup times are descriptive: the OS page cache is not controlled.

## Corpus limitations

The manifest records the sources and checksums. Podcast files concatenate
in-order excerpts from one episode; their joins are not guaranteed acoustically
continuous. No source audio is redistributed. Dataset transcriptions and poem
references are not new independent human annotations. The original recording
reported by the PR author is unavailable. No claim about all long-form speech,
other platforms, concurrent load, GPU backends or live WebSocket quality follows
from this experiment.

## Reproduction

Build the two exact revisions with `cargo build --release -p gigastt --bin gigastt`
and save the binaries separately. Prepare the frozen corpus using
[`scripts/longform_stitch.py`](../../../scripts/longform_stitch.py). Run
[`scripts/benchmark_stitch_case.py`](../../../scripts/benchmark_stitch_case.py)
with `--baseline`, `--candidate`, `--models`, `--audio`, a fresh `--output`,
`--baseline-revision`, `--candidate-revision`, `--cpus 2,3` and `--rounds 7`.
Choose two physical cores of the same type on another host. This run used a
systemd user unit with `MemoryMax=6G`, `MemorySwapMax=0`, `RuntimeMaxSec=1800`
and `Nice=5`. The script uses the Python standard library and local assets only.

The measurement helper's alignment, difference detection and readiness behavior
are checked by `python3 scripts/test_benchmark_stitch_case.py`.

## Excluded preliminary timing

The first campaign used cores 2 and 4 (different core types). Its RNN-T baseline
created an optimized graph from source, while the candidate loaded the resulting
cache. A warmed request does not equalize these session/kernel-prepacking paths.
The preliminary short-file median paired ratio was 1.0698. That is not a valid
estimate of this code change's overhead. The run was stopped and retained, not
silently replaced or averaged into the qualified results. Its 21 RNN-T quality
pairs had identical complete responses. The measurement helper was corrected to
prime and restart, verify cache hits for both arms, and use equal-type cores.
