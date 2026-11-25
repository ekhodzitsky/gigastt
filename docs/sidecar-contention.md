# Shared sidecar contention

This opt-in investigation measures existing synchronization without changing
model sharing, runtime defaults or recognition behavior. A main-pool size is
not a sidecar-pool size: Silero VAD and punctuation each share one session,
while the speaker extractor has four pooled sessions.

## Protocol

`benchmark/sidecar_contention.py` runs five isolated configurations (none, VAD,
punctuation, speaker diarization, all) at main-pool sizes 1, 2 and 4. Each child
process loads the rnnt INT8 model and selected installed sidecars. Construction,
first batch call and first streaming call are recorded separately. The first
batch triggers lazy speaker loading; the subsequent first streaming call does
not represent another cold model load. Serial reference calls warm all three
fixtures before mixed measurement. Cold and serial-reference interactive calls
are unpaced; only the warm mixed interactive worker uses real-time pacing.
Their whole-call elapsed times must not be compared as equivalent workloads.

The mixed workload has one interactive worker and three batch workers, each
processing Golos clips 00, 01 and 02 once. Interactive input is paced in 100 ms
chunks, keeps its main-pool checkout for the clip, and does not drop audio when
late. Batch workers call the same core request API used by REST. This is a
WS-core-style/REST-core-style workload, excluding transport, resampling and
HTTP admission. The workers begin together; scheduling and pool checkout order
are not controlled. At pool size one the requests necessarily serialize, and
an interactive checkout holds capacity while waiting for the next audio chunk.

The harness asserts equality against serial references within each configuration:
full text, confidence, word timing, endpoints and speaker annotations. Only
streaming wall-clock event timestamps are excluded. These comparisons establish
state isolation for this workload, not corpus-level diarization or ASR quality.

## Measurement boundaries

- VAD wait measures acquisition of its shared input-tensor mutex. Execution then
  includes input copies, acquiring the nested session mutex and native inference;
  output/state extraction occurs after the measured scope.
- Punctuation wait measures its session mutex. Execution includes native inference
  and borrowed-logit argmax reduction, excluding tokenization and label-to-text
  assembly outside that scope.
- Offline diarization and streaming speaker embedding report total elapsed time.
  The upstream extractor pool does not expose a separate wait observation. Its
  raw wait is null and its wait distribution has zero samples, not zero contention.
- Main-pool checkout, whole-request elapsed time, interactive per-chunk work,
  first partial (including checkout), and stop flush are measured independently.
- CPU is aggregate process time across all native threads, using the recorded
  Linux tick rate. It is not exclusive attribution to sidecars. RSS is sampled
  after calls/chunks and includes shared mapped pages; transient peaks can be missed.

Percentiles use nearest rank for p95 and median for p50, with explicit sample
counts. Three interactive requests and nine batch requests are too few for a
stable request-tail estimate; per-call sidecar distributions have more samples.
Probe recording itself uses a short test-only mutex and grows an observation
vector, so instrumentation can perturb scheduling.
All timings are exploratory debug measurements on a shared host. Configuration
order is fixed, not randomized, so ratios do not establish causal speedups.

Four encoder threads per main slot imply up to sixteen encoder threads at pool
four, in addition to decoder, sidecar and application work. The upstream tract
speaker backend ignores its ort-compatible intra-thread setting and uses the
tract default executor. Duplicating sidecars would add model/session memory and
may worsen CPU oversubscription; reducing a mutex wait alone is not enough to
justify it.

## Observed results

Measured on 2026-10-01 on AMD Ryzen AI 9 HX 370 (24 logical CPUs), Linux,
Rust 1.98.1 and debug profile. The [recorded artifacts (archive v1)](https://github.com/ekhodzitsky/gigastt/releases/download/benchmark-sidecar-contention-20261001-v1/benchmark-sidecar-contention-20261001-v1.zip)
include raw observations, recognition outputs, cold measurements, source/model
hashes and tick units. All fifteen cases passed concurrent-versus-serial output
equivalence. These exploratory runs are not release RTF claims.

| Case | Pool | Interactive p50 / p95 ms | Batch p50 / p95 ms | Requests/s | CPU s | Sampled RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| none | 1 | 4253 / 4748 | 4605 / 5306 | 0.856 | 14.14 | 427.2 |
| none | 2 | 3685 / 4736 | 849 / 986 | 1.020 | 14.19 | 797.2 |
| none | 4 | 3678 / 4173 | 380 / 457 | 1.077 | 14.08 | 1502.8 |
| vad | 1 | 4300 / 4538 | 4579 / 5050 | 0.876 | 13.73 | 446.2 |
| vad | 2 | 3918 / 4168 | 750 / 894 | 1.017 | 13.16 | 811.6 |
| vad | 4 | 3738 / 4298 | 452 / 637 | 1.046 | 13.99 | 1496.1 |
| punctuation | 1 | 4508 / 4963 | 4573 / 5286 | 0.856 | 13.74 | 474.7 |
| punctuation | 2 | 3984 / 4660 | 857 / 956 | 1.000 | 14.30 | 828.1 |
| punctuation | 4 | 3709 / 4259 | 487 / 578 | 1.050 | 13.95 | 1531.0 |
| speaker | 1 | 19905 / 25045 | 19872 / 26102 | 0.188 | 62.89 | 675.9 |
| speaker | 2 | 12158 / 13108 | 9108 / 12137 | 0.357 | 63.62 | 1018.8 |
| speaker | 4 | 8241 / 9336 | 5838 / 6807 | 0.474 | 71.80 | 1723.1 |
| all | 1 | 21312 / 21997 | 22459 / 27107 | 0.175 | 65.84 | 718.3 |
| all | 2 | 12897 / 14592 | 10867 / 17227 | 0.309 | 69.46 | 1064.0 |
| all | 4 | 9510 / 10245 | 5933 / 6408 | 0.440 | 66.14 | 1765.2 |

Each row has three interactive and nine batch requests, totaling 41.96 seconds
of input audio. Interactive wall latency includes deliberate input pacing.
The raw summary also contains first-partial, stop-flush, per-chunk and checkout
distributions, so admission delay can be separated from recognition work.

| Case | Pool | VAD wait / execution p95 ms | Punctuation wait / execution p95 ms | Offline diarization total p95 ms | Streaming embedding total p95 ms |
|---|---:|---:|---:|---:|---:|
| vad | 1 | 0.0027 / 0.736 | — | — | — |
| vad | 2 | 0.0029 / 0.746 | — | — | — |
| vad | 4 | 1.7409 / 0.971 | — | — | — |
| punctuation | 1 | — | 0.0025 / 1.475 | — | — |
| punctuation | 2 | — | 0.0042 / 1.409 | — | — |
| punctuation | 4 | — | 0.0055 / 1.900 | — | — |
| speaker | 1 | — | — | 7556 | 1395 |
| speaker | 2 | — | — | 5197 | 1601 |
| speaker | 4 | — | — | 6176 | 1830 |
| all | 1 | 0.0012 / 0.836 | 0.0029 / 2.045 | 6530 | 1884 |
| all | 2 | 0.0014 / 1.083 | 0.0053 / 1.980 | 6651 | 2151 |
| all | 4 | 1.8503 / 0.999 | 0.0044 / 6.772 | 5916 | 2652 |

VAD distributions contain 1,314 calls per applicable case. Punctuation and
diarization counts are in the artifacts; short utterances produce few calls.
Speaker totals include opaque upstream waits and are not directly comparable
to the mutex-separated VAD/punctuation observations.

| Case | Pool | Construction ms | First batch ms | First stream ms |
|---|---:|---:|---:|---:|
| none | 1 | 1735 | 378 | 1122 |
| none | 2 | 1261 | 293 | 974 |
| none | 4 | 2201 | 333 | 981 |
| vad | 1 | 1955 | 317 | 1072 |
| vad | 2 | 1748 | 282 | 1122 |
| vad | 4 | 3432 | 358 | 1905 |
| punctuation | 1 | 1967 | 398 | 1230 |
| punctuation | 2 | 1782 | 398 | 1123 |
| punctuation | 4 | 3029 | 376 | 1287 |
| speaker | 1 | 2940 | 13412 | 6166 |
| speaker | 2 | 2542 | 11936 | 6697 |
| speaker | 4 | 2982 | 11020 | 8350 |
| all | 1 | 2835 | 21435 | 7962 |
| all | 2 | 2011 | 11574 | 7440 |
| all | 4 | 3643 | 17206 | 6509 |

## Decision and bounded follow-up

Keep existing sidecar sharing. Punctuation lock waits are tiny in this workload;
it does not justify another resident punctuation model. VAD exhibits measurable
millisecond waits with more concurrent requests, but the measurements do not
establish a material end-to-end benefit from duplication. Pool-one admission
delays are much larger and cannot be repaired by changing a sidecar mutex.

Speaker-enabled workloads have substantial debug execution costs and increased
RSS. Pool four removes most checkout delay here but leaves slow interactive
embedding calls and higher CPU/memory consumption. Without exclusive upstream
pool-wait measurements and controlled release runs, these totals cannot identify
a speaker-lock bottleneck or justify a larger speaker pool.

A bounded follow-up is a randomized release run on an idle target machine with
longer utterances, more request samples and explicit CPU budgets. If speaker
latency remains material, measure upstream pool acquisition separately before
changing concurrency. Any implementation must retain per-request VAD state and
exact same-input recognition/speaker outputs, and report incremental resident
memory and CPU rather than only shorter lock waits. No such implementation is
included in this research change.

## Reproduction

To verify the complete historical sweep without installing any models, download
and check the versioned archive in a temporary directory:

```sh
curl -fLO https://github.com/ekhodzitsky/gigastt/releases/download/benchmark-sidecar-contention-20261001-v1/benchmark-sidecar-contention-20261001-v1.zip
echo "17a883d99bbfd6712afe4c9fcf0cd532ca43dfe5d4f023ae58e1b4dcf5660cd8  benchmark-sidecar-contention-20261001-v1.zip" | sha256sum -c -
unzip benchmark-sidecar-contention-20261001-v1.zip -d sidecar-contention
python3 sidecar-contention/verify.py
```

The pinned SHA-256 identifies the intended bytes even though GitHub does not
force release immutability for this repository. The verifier recomputes all
15 summaries from 180 recorded requests, not fresh inference or an independent
serial-output comparison (the original harness asserted the latter). All raw
measurements and negative findings are preserved. Golos-derived outputs retain
[Sber's attribution, noncommercial, share-alike license](https://github.com/sberdevices/golos/blob/master/license/en_us.pdf);
no audio or model weights are bundled. The original source remains at the
[pinned evidence revision](https://github.com/ekhodzitsky/gigastt/tree/60600b0ae6b2bd2350efc2188d9edd0939267dd7/benchmark/results/sidecar-contention).

For a new inference run, install rnnt, punctuation, Silero VAD and WeSpeaker models first. The harness
never downloads models. Run from the repository root:

```sh
python3 benchmark/sidecar_contention.py --output /absolute/local/results
python3 benchmark/sidecar_contention.py --output /absolute/local/results --summarize-only
PYTHONPATH=benchmark python3 -m pytest benchmark/tests/test_sidecar_contention.py -q
```

`--cases` and `--pools` select smaller experiments. The ignored Rust harness
explicitly skips unless `GIGASTT_SIDECAR_PROBE` specifies an output path, keeping
normal model coverage bounded.
