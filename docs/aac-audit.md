# AAC integration audit

The syom integration is useful as an **opt-in format/profile extension**. It is
not yet justified as the default decoder replacement. Build it explicitly with:

```sh
cargo build --release -p gigastt --features extended-aac
```

Standard builds keep the existing Symphonia AAC/MP4 path and do not compile
syom. No download, runtime flag or release workflow enables this feature.
The default and optional paths retain AAC in Matroska and non-AAC MP4 tracks.

The follow-up [published syom 0.8.1 qualification](measurements/aac-20261029/README.md)
updates this integration against current main. The measurements below retain
their original syom 0.7.1 provenance; they are not relabeled as 0.8.1 results.

## What the dependency adds

Review covered syom 0.7.1 and Symphonia 0.6.1, their locked Cargo features,
file/bytes ingestion, channel scanning/splitting, cancellation, budgets and
worker lifecycle. syom adds one optional MIT crate with no runtime dependency
edges. Its encoder is not used in production; roundtrip tests use it only to
exercise the adapter. Symphonia's AAC and ISO MP4 components cannot be removed:
they remain necessary for containers outside syom's AAC transport handling.
The release diagnostic executable grew by about 0.87 MiB with the refactored
optional path; the default executable size matched base. These are probe
artifacts, not a claim about the server distribution size. Clean build time
and cold-start cost were not measured.

On the local fixtures, syom adds LATM/LOAS and AAC-LD decoding and accepts an
HE-AAC v2 fragmented MP4 rejected by the baseline. HE-AAC M4A uses the decoded
presentation rate instead of the core rate: two fixtures that previously
produced roughly half the expected output now have the correct duration.
AAC-LC fragmented MP4 already worked in the baseline. This is not a claim that
every AAC profile, container layout, damaged input or fragmented stream works.
AAC in Matroska continues to use Symphonia, including its profile limitations.

## Confirmed integration regressions and fixes

The original integration removed AAC/MP4 support from Symphonia without
checking containers that still depended on it. Three tests were added first
and failed against that version:

- AAC in Matroska lost its decoder. Restore the Symphonia AAC feature.
- MP3 in MP4 lost duration probing because the AAC probe claimed the container
  and returned a format error. Preserve the MP4 demuxer and let non-AAC MP4
  metadata fall through to it.
- Channel splitting checked cancellation only before and after full AAC
  decoding. Check cancellation between frames and resampling steps.

The original channel scan/split also materialized the entire native-rate PCM.
The replacement resamples each plane incrementally; scans retain correlation
state only, while splitting retains the requested 16 kHz output. It inspects
the actual decoded channel layout because implicit HE-AAC v2 can advertise
mono in ADTS but emit stereo. Scan/split parity and cancellation have tests.

The callback-to-window adapter previously handed every AAC frame across a
thread boundary. It now transfers batches of at least 8192 source frames with
a two-slot bounded queue. Worker ownership covers initialization/error exits;
dropping the receiver unblocks a waiting producer before joining. File decoding
reuses the opened handle. Transport input, metadata probing and channel
processing are separated into focused modules. Existing duration and parser
memory limits remain; streaming does not collect duration-sized native PCM.

These changes pass the controlled local latency comparison. The new path is
nevertheless behind the non-default `extended-aac` feature: the format extension
is useful to users of HE/LD/LATM, while replacing the standard AAC-LC path is
not required. Split memory remains higher, scheduling on mixed-speed cores
needs further qualification, and concurrent/platform coverage is incomplete.
Enabling it by default requires the same acceptance evidence for those workloads.

## Measurements

Linux x86_64, AMD Ryzen AI 9 HX 370, Rust release profile with LTO, locked
dependencies. Base: `def4f223eac28cbdc5372f06c74f942cddfa34bd`;
original integration: `ca3d29029c0460a8323ad6609da68a6b586f5229`.
[Protocol, versions, CPU details and asset/binary hashes](measurements/aac-20261016/protocol.json)
accompany the results. Seven alternating base/candidate processes per case;
each process performs two warmups and seven timed decodes. Ratios are the
median of paired within-process medians. RSS is Linux process high-water RSS,
not an allocator-only measure. The probe retains compressed bytes and the
previous output in both versions.

The **controlled comparison pins all processes to CPUs 2 and 3**, distinct
physical cores of the same fast class. This matters on this hybrid CPU:
cores 0–3 advertise a 5.16 GHz maximum; cores 4–11 advertise 3.29 GHz.
There were no concurrent builds during measurements. Other applications were
present, so these results remain local evidence, not cross-machine guarantees.

| Workload | Original/base time | Refactored opt-in/base time | Final default/base time | Base / original / refactored peak RSS, MiB |
|---|---:|---:|---:|---:|
| 4 s mono M4A, file | 1.092 | 0.989 | 0.995 | 15.5 / 14.1 / 14.3 |
| 4 s mono M4A, bytes | 1.088 | 0.975 | 1.003 | 15.5 / 13.9 / 14.2 |
| Same speech, ADTS file | 1.111 | 0.975 | 1.005 | 15.4 / 14.3 / 14.3 |
| 80 s stereo M4A, file | 1.188 | 0.854 | 0.994 | 25.6 / 24.7 / 24.9 |
| 80 s stereo M4A, split | 1.097 | 1.012 | 1.010 | 36.1 / 75.0 / 45.4 |
| 80 s stereo M4A, scan | 1.074 | 1.016 | 0.976 | 17.3 / 56.1 / 16.0 |

The original integration fails the 5% latency threshold on all six workloads.
Both the refactored optional build and the standard build pass on these cores.
Splitting still uses more peak memory than base; scans no longer retain the
native PCM recording. RSS medians come from the respective campaigns (base
shown from the refactored comparison). These are decode-path measurements, not
cold ASR load or concurrent-request benchmarks.

Full timing arrays, RSS, PCM summaries and binary/fixture hashes:
[original](measurements/aac-20261016/original-pinned.json),
[final refactored opt-in](measurements/aac-20261016/final-pinned.json),
[final default](measurements/aac-20261016/default-pinned.json).
The earlier [refactor repeat](measurements/aac-20261016/candidate-pinned.json)
also passed all cases. The final rebuild retained identical speech PCM in all
15 cases ([hashes](measurements/aac-20261016/final-pcm.json)).
The [acceptance harness](../scripts/compare-aac-decoders.py) fails on missing
measurements, decode errors, invalid/silent speech PCM or a median paired
latency increase above 5%. Memory is reported separately; waveform/ASR quality
requires the checks below.

### Why the first measurements were repeated

Initial campaigns allowed scheduling across all 24 logical CPUs. They showed
original/base latency ratios of 1.20–1.27 on mono/file workloads and refactored
ratios of 1.09–1.13 on short inputs. However, even the **unchanged default path**
then measured 1.53 on the byte case, with individual process medians ranging
from 4.3 to 7.1 ms. Inspection revealed the mixed core classes above. Those
numbers cannot be treated as isolated decoder cost or evidence of a confirmed
remaining short-file regression.

All original observations are retained:
[original process medians](measurements/aac-20261016/original-paired.json),
[refactored process medians](measurements/aac-20261016/candidate-paired.json),
[default full run](measurements/aac-20261016/default-paired.json).
The response was to control affinity and repeat **all three builds**, not to
relax the threshold or discard inconvenient cases. Behavior under unrestricted
scheduling and concurrent load still needs qualification before a default change.

## Audio and recognition quality

Fifteen existing Golos fixtures were independently encoded with FFmpeg native
AAC at 48 kHz, mono, 64 kbit/s. All fifteen refactored outputs are byte-identical
to the original syom path. Standard-build outputs are independently verified
[byte-identical to main](measurements/aac-20261016/default-pcm.json).

The same CPU RNN-T engine, INT8 model hashes, pool size 1, encoder thread count 1,
ITN off and no punctuation/VAD processed all three PCM sets. Base had one word
error out of 75 reference words (`тиви` versus `тв`, an abbreviation rather
than a missing phrase); both syom paths had zero. See the
[per-file transcripts, errors and model hashes](measurements/aac-20261016/asr-quality.json).
This small, likely training-overlapping slice only checks regressions; it does
not establish general WER, long-form completeness or HE-AAC recognition quality.

An independent FFmpeg decode at the **native presentation rate**, followed by
the same gigastt mono mix/resampler, agrees with syom at 115–138 dB SNR over
aligned interior samples on five speech M4As and five HE-AAC transport fixtures.
[Exact results](measurements/aac-20261016/native-pcm-quality.json) include lengths
and offsets. The comparison drops 256 output samples at each overlap boundary,
uses no amplitude fitting, and therefore does not certify those boundaries.
M4A output can be shorter than FFmpeg by up to 192 output samples here because
syom applies the declared final edit, while FFmpeg retains some coded padding.
Comparing against an already-resampled FFmpeg file gives misleading SNR from
different resampler phase/filter choices; that exploratory result is not used
as decoder-quality evidence.

The HE/LD capability fixtures were read locally from syom's goldens, with
hashes recorded. Historical source/encoder provenance is incomplete for some
of them, so they are not redistributed here and are not a substitute for an
independent representative corpus. The local FFmpeg could not decode the
AAC-LD LOAS fixture; its result is **capability-only**, not an independently
verified fidelity result. [Capability outcomes](measurements/aac-20261016/capabilities.json)
retain failures and unknown durations. No macOS, ARM, mobile, cold-start or
multi-request performance qualification was performed.

## Reproduce and maintain the acceptance check

Use isolated clean worktrees for the base and candidate. Copy the identical
`crates/gigastt-core/examples/aac_probe.rs` from this change into the base core
examples directory. The base can auto-discover it. Build with the same toolchain,
Cargo release profile, target settings and native libraries:

```sh
# In the baseline worktree; save the executable outside Cargo's target directory.
cargo build --release --locked -p gigastt-core --example aac_probe
cp target/release/examples/aac_probe /tmp/aac-base

# In the candidate worktree; omit extended-aac to check the standard build.
cargo build --release --locked -p gigastt-core --example aac_probe --features extended-aac
cp target/release/examples/aac_probe /tmp/aac-candidate
# Select two equal-class physical cores appropriate to the host.
taskset -c 2,3 python3 scripts/compare-aac-decoders.py --prepare \
  --base-probe /tmp/aac-base --candidate-probe /tmp/aac-candidate \
  --fixtures /tmp/aac-fixtures --output /tmp/aac-comparison.json
python3 scripts/test-compare-aac-decoders.py
```

The fixture generator records commands and FFmpeg version and fails on encoder
errors. For a repeat use the same fixtures without `--prepare`. The harness
records complete timing arrays, executable/fixture hashes, RSS and channel
verdicts. Run without concurrent builds or other heavy workloads. Investigate
noise and report all runs; never choose a newer failing baseline to hide cost.

For recognition, encode each `golos_00.wav` through `golos_14.wav` with the same
command as `speech0.m4a`. Use `aac_probe file INPUT 1 PREFIX` to dump each
16 kHz `.0.f32` result, then run `aac_asr_probe MODEL_DIR PCM...` once over all
sets. References are in `crates/gigastt/tests/fixtures/manifest.json`.
The recorded normalization is lowercase, `ё` to `е`, word tokenization, and
Levenshtein distance divided by reference words. For the PCM oracle, use
`ffmpeg -i INPUT -c:a pcm_f32le native.wav` without `-ar`, then dump `native.wav`
through the same probe.

Local workspace tests must pass both with default features and with
`--features gigastt/extended-aac`; run strict Clippy for both. Manual CI now
also runs the optional AAC integration tests. Workflows remain
`workflow_dispatch` only; this audit did not dispatch GitHub Actions.

## Local validation

- Default workspace unit/bin suite: 1,240 passed; optional AAC suite enabled
  across the workspace: 1,249 passed. Model-requiring ignored tests were not
  counted as passed; the explicit 45-transcription check is described above.
- Strict workspace/all-target Clippy passed for both configurations; rustfmt,
  documentation drift, Docker cache targets and publication checks passed.
- The comparison harness tests reject missing rounds/cases, decode errors,
  nonfinite timings, absent RSS, invalid/silent PCM and latency regressions.
- Lean `gigastt-core --no-default-features` compiles, with two existing warnings
  in unchanged engine code (`WindowSpec` and `max_audio_secs` unused).
- GitHub workflows remain manual-only. No Actions run or merge was performed.

Merge recommendation: review as an optional capability extension with the
compatibility fixes and evidence above. Do not market it as a universally
faster decoder or enable it in default releases based on this one-host audit.
