# Published syom 0.8.1 qualification

This follow-up updates the optional AAC integration to published syom 0.8.1
and incorporates upstream main `143b9f2920a65fe4b466c012cfaaa8e0fb302bda`.
Integration commit `d577b9ee3ca5a15f2eb848db899db1d1e598d22b` preserves the
audited bounded adapter, cancellation/budget checks and Symphonia fallbacks.
Standard builds retain their existing decoder; `extended-aac` stays opt-in.
No service deployment or GitHub CI dispatch is part of this work.

## Controlled sources and local checks

The compared consumer snapshots have **292 identical Rust files**. Their
locked dependency entries are identical except for syom 0.7.1 versus 0.8.1.
Both syom versions are registry packages with zero runtime dependency edges;
the 0.8.1 checksum matches the published registry artifact. The new version
fixes fill-padding allocations and PNS seek history in the library.

[Protocol](protocol.json) records source trees, checksums, toolchain, release
profile, selected/resolved features and limits. Rust is 1.98.1 on Linux x86_64.
Release probes/servers use the existing dynamic ORT installation; manifests
keep their normal ORT defaults. Building the example together with the server
unifies server development features into the core, so the default decoder
probe explicitly matches that feature list except for `extended-aac`.
No diarization inference is requested. Build times use warmed caches and are
not controlled cold-build or startup measurements.

Default workspace unit/bin tests: **1259 passed, 48 ignored**. With
`extended-aac`: **1268 passed, 48 ignored**. Strict workspace/all-target
Clippy passed for both. Formatting, documentation drift and its tests,
the existing acceptance-harness tests and normal Git hooks passed.
The merge retains current main's lock versions. Cargo also normalized four
pre-existing Prost/UniFFI dependency-edge references; no existing package
version changed relative to main, and syom is the only added package.

## PCM parity

[All 29 cases](pcm-parity.json) are **bit-identical** between syom 0.7.1 and
0.8.1 through the refactored adapter. The comparison includes 15 independent
48 kHz AAC-LC speech M4As, HE/PS ADTS/M4A/LATM, LD LOAS, fragmented MP4,
AAC Matroska and MP3-in-MP4 fallback, plus 80-second stereo file/bytes/split
paths. Every output has finite, nonempty PCM with matching plane lengths and
SHA256. This establishes parity on these inputs; no listening or new ISO
conformance claim is made. The older HE/LD fixture provenance limitations in
the [original audit](../../aac-audit.md) still apply.

The [manifest](pcm-manifest.json) retains input paths; hashes are in the result.
Inputs and PCM remain outside Git. Reproduce with the same release probes:

```sh
python3 scripts/compare-aac-pcm.py \
  --base-probe /path/to/syom-0.7.1-probe \
  --candidate-probe /path/to/syom-0.8.1-probe \
  --manifest /path/to/local-pcm-manifest.json \
  --output /tmp/aac-pcm-parity.json
```

The runner deletes its temporary PCM, returns nonzero on differences/errors,
and verifies dump lengths against the probe's geometry.
[Controlled invalid-output checks](pcm-harness-controls.json) show it accepts
identical PCM and rejects changed samples or a mismatched dump length.

## Handoff overhead and final decoder acceptance

The first pure-version comparison used CPUs 2/3. Five cases passed, but the
short bytes case failed at 1.124x; individual pairs ranged from 0.843x to
1.417x. QEMU was observed afterward on SMT sibling 15 of CPU 3, then exited
before automated follow-up sampling. This suggests scheduling interference,
not proven causation. The [failure](upgrade-paired.json) and
[diagnostic](scheduling-diagnostic.json) are retained. Two complete controls on
equal-class physical CPUs 0/2 kept the same binaries, fixtures and 5% limit;
[control 1](upgrade-controlled-1.json) and [control 2](upgrade-controlled-2.json)
passed all six cases. The bytes ratios were 0.975x and 0.990x.

The wider [standard-versus-extended comparison](default-versus-extended.json)
then failed three gates at 8 Ki-frame batches: short file 1.121x, ADTS 1.073x,
long stereo file 1.114x. Passing the version-only comparison was not enough
to accept the whole feature. A [resource profile](scheduling-profile.json)
found median 3777 voluntary context switches per extended probe process versus
one for standard, and higher aggregate CPU cost. Each process decodes the
80-second file nine times; these are process totals, not per-request counts.

The only additional Rust change increases the bounded producer batch from
8192 to **24576 source frames**. Queue capacity remains two. This amortizes
handoffs and increases bounded queued PCM, without retaining duration-sized
native audio. It does not change a public option, pool setting or timeout.
The final source differs from the pure-version snapshot in that constant only.

The [post-change PCM comparison](batched-pcm-parity.json) is exact on all 29
original cases plus two repeated LC960 mono/stereo LOAS controls. The latter
cross several resampling boundaries; [recipes](frame960-recipes.json) retain
the source hashes and repeat counts. Optional tests/Clippy passed again.

Both final standard-decoder comparisons pass the existing **1.05x** limit:

| Workload | Final / standard | Confirming repeat | Final / old syom 0.7.1 adapter |
|---|---:|---:|---:|
| 4 s mono M4A, file | 0.914x | 0.920x | 0.934x |
| 4 s mono M4A, bytes | 0.926x | 0.931x | 0.912x |
| ADTS, file | 0.969x | 0.906x | 0.897x |
| 80 s stereo M4A, file | 0.652x | 0.669x | 0.740x |
| 80 s stereo M4A, split | 1.022x | 1.030x | 1.030x |
| 80 s stereo M4A, scan | 1.032x | 1.024x | 1.016x |

[Final comparison](batched-default-versus-extended.json),
[confirming repeat](batched-default-repeat.json),
[old-adapter comparison](final-upgrade-paired.json).
Each uses seven alternating paired processes, two warmups and seven timed
decodes per process on CPUs 0/2. No concurrent local builds ran during timing.
Other applications remained present. No failing measurements were removed,
no baseline binary was replaced after a failure, and no threshold was widened.

The long file is about 33–35% faster than standard in these two passes, but
scan/split are 2–3% slower. First-pass median peak RSS is 10300/9544 KiB for
standard/extended short file, 20732/20356 KiB for long file,
31232/40916 KiB for split and 11980/10736 KiB for scan. Split remains about
31% higher than standard; this existing opt-in tradeoff is not waived or
presented as a memory win. The feature remains non-default. These are local
decode measurements, not a universal codec ranking or deployment capacity.

## Actual HTTP inference, load and faults

The [paired canary](http-canary.json) passes all 32 normalized transcript and
duration comparisons, matching 422 responses for malformed inputs and clean
shutdown. It uses the same 15 speech recordings as WAV/AAC-LC plus HE v1/v2
derivatives. The baseline is the refactored 0.7.1 adapter; the final candidate
uses published 0.8.1 and 24 Ki-frame batches. This is a separate ASR check,
not the 5% decoder-only timing gate above.

The first [load attempt](http-load-legacy-expectations.json) used the old
consumer's queueing expectations and failed on **both versions**. Current
main has [aggregate upload admission](../../long-recordings.md#aggregate-upload-admission):
excess uploads get `503 upload_busy` before body collection. They do not all
wait for inference checkout. This run also capped encoder threads to one when
two pool slots were confined to two logical CPUs. It is retained as a failed
protocol/configuration diagnostic, not a passing load qualification.

The [final load](http-load.json) uses the explicit `--upload-admission` mode
of the [versioned local runner](https://github.com/ekhodzitsky/syom/blob/a91ef93/lab/release-audit/run_http_load.py).
It leaves upload admission enabled, honors `Retry-After` on the typed rejection
and also gates client latency including retries. Any other error fails the
normal-load check. One/two-client waves must succeed without admission retries;
four/eight-client waves exercise overload while ultimately verifying all
32 successful results. Both servers use the same unrestricted host affinity,
two slots and two encoder threads per slot, without thread clamping.
Checkout timeout is explicitly one second to provide a one-second backoff;
the inference watchdog is 600 seconds. Fault profiles use one slot/thread.

| Check | Baseline | Final candidate |
|---|---:|---:|
| Successful measured requests, seven 32-input waves | 224 | 224 |
| Typed upload-busy responses retried after backoff | 105 | 102 |
| Sampled peak process RSS, KiB | 816564 | 839292 |
| Post-wave RSS growth across four concurrency-8 waves, KiB | 604 | 832 |
| Last wave client p95 / p99 including backoff, ms | 5152.7 / 5189.1 | 5265.2 / 5342.1 |
| Longest pool recovery after a cancelled REST client, ms | 1669.5 | 1759.4 |
| Pool recovery after inference watchdog response, ms | 850.1 | 756.6 |

All 448 successful results match the serial reference, including duration
within one 16 kHz sample. Every wave passes service and client p95/p99 ratios
<=1.5; sampled peak RSS passes baseline x1.25 + 32 MiB, and each server's
steady RSS growth stays below 64 MiB. These load-regression gates were defined
separately from the unchanged 5% paired decoder gate. They are not production
SLOs or statistically precise p99 estimates.

Fault checks independently exercise eight immediate `upload_busy` rejections
per binary while a long REST request occupies admission. Then a submitted job
releases its upload permit and holds the sole inference slot: eight sequential
REST requests return actual `503 timeout`, verified by the pool-timeout counter.
The holding job completes with the same text hash/duration as the long REST
request. Three disconnects and three 1 ms client read timeouts each recover
the pool and pass an exact control inference; ten probes after each fault
profile also pass. A separate one-second no-progress watchdog returns
`504 inference_timeout` and increments its counter on both binaries.
All six final-load server processes drain and exit 0; no service was installed.

The [protocol](protocol.json) pins binaries, runtime and models. Inference uses
the existing **legacy 215 MiB RNN-T INT8 encoder**, ORT 1.28.0 CPU, punctuation
and ITN off. This is not qualification of current main's newer 305 MiB release
bundle. Model files were verified unchanged afterward. No diarization,
WS/SSE recognition, alternate provider, physical mobile or multi-hour soak
claim is made. WebSocket was not needed: the holding job isolates pool waiting
without bypassing upload admission. Prometheus snapshots and filtered lifecycle
logs accompany the JSON; text is retained only as normalized hashes.

Reproduce using the runner at the linked syom commit and the same two server
binaries, model/runtime and matching private manifest:

```sh
python3 /path/to/syom/lab/release-audit/run_http_load.py \
  --upload-admission \
  --baseline /path/to/old-adapter-server \
  --candidate /path/to/final-server \
  --model-dir /path/to/legacy-models \
  --ort /path/to/libonnxruntime.so.1.28.0 \
  --manifest /path/to/local-manifest.json \
  --reference /path/to/paired-canary-result.json \
  --long-input /path/to/repeated-180s.m4a \
  --out /path/to/fresh-results
```

Run from the manifest's intended working directory so its paths resolve.
The maintainer explicitly requested **no deployment**. The deliverable is
local qualification and the existing PR update; merging/publishing remains
subject to the repository's required checks. No GitHub workflow was dispatched.
