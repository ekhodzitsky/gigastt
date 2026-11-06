# Published syom 0.10.1 and bounded pull-adapter qualification

**Decoder acceptance passes; overall acceptance remains incomplete.** This
refresh uses the published registry crate plus the nine-file downstream handoff
from syom. Seven runtime Rust files were verified against the handoff manifest;
the manifest/lockfile instead use the published registry checksum. The bounded
M4A/ADTS pull and buffered reader changes are part of this result: this is not
a claim that changing only the dependency version fixes consumer overhead.

The base is main `adf4db0d7b06a150534734b69b925e3136f0a631`; default Symphonia
is retained. Compare the failed [0.10.0 qualification](../aac-20261104/README.md).
No old result was removed or threshold widened.

## Fresh local results

- 1305 optional-AAC tests and 1269 default tests pass, with 49 model tests
  ignored in each ordinary suite; strict Clippy passes for both configurations.
  The upstream qualification tool suite also passes all 36 offline tests.
- All 29 [PCM cases](pcm-parity.json) match the preserved 0.10.0 adapter output.
- [Two](campaign-1.json) [separate](campaign-2.json) six-case decoder campaigns
  pass the unchanged 1.05 median paired latency limit. CPU 1 was selected before
  measurement to reproduce the upstream protocol; [registration](preregistered.json)
  fixes two campaigns and retains every outcome. Worst ratio is 1.048 in the
  first and 1.040 in the second. Each uses seven alternating process pairs,
  two warmups and seven timed decodes per process. No concurrent builds ran.
- Split peak RSS remains higher: campaign two is 51724 versus 38440 KiB,
  about 35% (+13 MiB). This backend remains opt-in; no memory improvement is claimed.
- The raw [32-case HTTP equality canary](http-canary.json) remains false, as
  expected for corrected presentation. Fresh [independent evidence](independent.json)
  and [reference qualification](reference.json) pass on the active 305 MiB RNNT
  model. This preserves exact presentation-duration checks and fresh WAV/FFmpeg
  recognition rather than accepting incorrect baseline priming/padding.
- The [load/fault campaign](http-load.json) completes 224 requests per backend.
  Eleven of twelve gates pass: responses, memory, client latency with retries,
  admission/backoff, cancellation, watchdog and recovery. All owned processes
  exit cleanly. The separate service-latency gate FAILS in the final 8-client
  burst: candidate p95 495.23 ms versus baseline 201.84 ms (2.45x; limit 1.5x).
  Previous bursts and client latency including backoff pass. Cause is not
  established; do not discard this result or call overall qualification green.

The binaries use identical compiler/release profiles and the same static ORT
archive. [Metadata](validation.json) pins registry, handoff and validation scope;
[commands](commands.json) retain the fresh HTTP sequence. Local path fields in
reference files describe this run and require rebinding when replayed elsewhere.
No audio, PCM or model weights are published. This is Linux x86_64 evidence only.
No hosted CI, release publication or deployment was performed for this refresh.

## Load-latency localization

The original two slow final responses spend 457/495 ms inside logged encoder
inference. AAC setup begins about 1–2 ms before inference starts. Thus the
observed half-second delay is in ASR inference, rather than AAC input/handoff
or the upload-admission retry interval.

A fixed diagnostic (four rounds, four server conditions, 8 clients, the same
32 mixed WAV/AAC cases) adds per-thread CPU observations. It does not replace
acceptance. [Summary](latency-localization.json) retains every round. Main's
unpinned p95 ranges 155–747 ms; candidate 204–702 ms. Slow WAV controls occur
in both binaries: maximum logged WAV encoder time is 716/639 ms. Both binaries
were also tested with requested affinity on the same four fast physical cores
0–3; sampled CPU IDs confirm this placement. Main p95 ranges 174–236 ms, while
candidate ranges 173–561 ms. Affinity reduces typical variability but neither
reserves cores nor eliminates the candidate outlier. No stable candidate-only
cause has been established.

The host also carries indexing/rendering/compilation work. A separate
[post-experiment ten-second observation](process-load-snapshot.json) measures
actual CPU deltas; it is not telemetry from the original spike. The diagnostic
observer itself adds overhead (recorded in the summary) and is pinned to CPU8.
Kernel sched_schedstats was disabled: sampled wait counters are not a reliable
live starvation measure. No other user's processes were stopped or modified.

Conclusion: the slow stage is localized to encoder inference and the phenotype
also occurs without syom, on main and WAV. Shared-host CPU contention/placement
is a supported explanation to investigate, not a proven historical root cause.
The original load failure remains valid and unresolved. Next qualification must
use a quiet or isolated runner, keep identical placement/configuration for both
binaries, and retain CPU/pressure context around requests. No production thread
limits or acceptance thresholds were changed to make this result pass.
