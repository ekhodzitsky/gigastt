# Edge and Raspberry Pi validation

Linux ARM64 builds are available, but there are no published Raspberry Pi
measurements in this repository. Select a model using on-device recognition
quality, latency and memory measurements. Historical M1 results are not Pi
capacity estimates. The filename is retained for existing links.

## Install and configure

Use a 64-bit OS and the `aarch64-unknown-linux-gnu` archive from
[Releases](https://github.com/ekhodzitsky/gigastt/releases), or the ARM64
[Docker image](deployment.md#docker). Follow the [installation guide](../README.md)
and [model guide](models.md); pin the binary version and model hashes for comparisons.

```sh
gigastt download --model-variant rnnt --skip-diarization
gigastt serve --profile edge --punctuation off --itn off
```

The existing `edge` profile defaults to one pool slot and enables VAD when those
settings are not explicitly supplied. It does not select a different ASR model
or establish that live recognition keeps up on the device. VAD requires its own
model. Punctuation, VAD and speaker models have separate provisioning and runtime
costs; see [model setup](models.md#punctuation-numbers-and-speaker-labels).

Leave encoder thread selection at its default for the first measurement, then
compare explicit thread counts on the same board. Measure any change in model,
quantization, backend, VAD or pool size before recommending it.

## Measurement protocol

The [edge harness](../benchmark/bench_edge.py) starts a local server, measures
startup to `/ready`, warm file RTF on five fixtures, RSS snapshots and optional
WebSocket time to first partial. It disables punctuation and ITN for its ASR
comparison; it does not reproduce every setting of `serve --profile edge`.
Python 3 is required; install `websockets` in a virtual environment for TTFP.

From a checkout with the chosen binary installed and models provisioned:

```sh
./scripts/bench_edge_pi.sh \
  --binary "$(command -v gigastt)" \
  --storage-label microSD \
  --variants rnnt \
  --pool-size 1 \
  --output benchmark/results_edge_pi4.json
```

Use `--model-dir` for a separate model directory. Repeat with `usb-ssd` or `nvme`
only when that is where the model and optimized cache actually reside. Use
`--variants rnnt,ml_ctc` only after provisioning both models. The wrapper forwards
`--encoder-intra-threads N` for thread comparisons.

Record alongside the JSON:

- Board model, RAM, OS, CPU governor, cooling, temperature and throttling state.
- Binary version/commit, build features, model SHA-256 hashes and configuration.
- Storage location and whether the optimized cache and OS file cache were warm.
- Warm-up count, repeated runs, audio manifest and enabled postprocessing.

A new process is not necessarily a cold disk/cache start. RSS snapshots are not
peak memory, and macOS dirty/compressed footprint is not a Linux memory metric.
Collect peak memory separately when evaluating device capacity; include side
models, concurrent requests, the OS and other services.

## Quality and publication

The five short fixtures are a speed smoke test, not a recognition-quality
benchmark. Run a representative corpus with reference transcripts using the
[dataset protocol](held-out-datasets-roadmap.md). Report WER/CER, deletions and
empty outputs as well as RTF; report punctuation separately when enabled.
For live use, also measure finalization lag and streaming WER under the
[streaming protocol](benchmarks.md#streaming-measurement-protocol).

Publish raw output, hashes, settings and limitations with the
[benchmark results](benchmarks.md#edge--raspberry-pi). An RTF below one on short
files does not establish that continuous live recognition meets its latency or
quality target. No Pi default or capacity claim should be based on M1 figures.

Historical M1 values are described in the benchmark report with their
provenance limitations. The [dated latency note](archive/streaming-latency-2026-06-13.md)
records a separate historical experiment.
