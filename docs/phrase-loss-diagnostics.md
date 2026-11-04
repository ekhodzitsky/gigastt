# Local diagnosis of residual phrase loss

This bounded protocol investigates [issue #376](https://github.com/ekhodzitsky/gigastt/issues/376).
It does not claim to fix the private meeting, reproduce the author's unavailable
scorer, or measure WER. No audio or transcript is uploaded.

This protocol uses the **CLI file path**, while the original report uses REST.
First check whether `full-cold.json` / `full-warm.json` reproduce the reported
missing phrases. If CLI and REST differ, retain that result and investigate the
input/API paths before attributing the difference to window stitching.

## Requirements

- Python 3.11 or later; only the standard library is used.
- A current `main` **CPU-only** gigastt build on Linux (for example,
  `cargo build --release --locked -p gigastt`). No runtime upgrade is required.
  Validated runtime source: `143b9f2920a65fe4b466c012cfaaa8e0fb302bda`.
  Record the exact source commit of your binary; the version string alone does
  not identify the native runtime. Older binaries without the provider/cache
  diagnostic log lines fail explicitly.
- The existing published `e2e_rnnt` INT8 bundle, including decoder, joiner and
  vocabulary. The runner is offline and never downloads or quantizes a model.
- The original mono 16 kHz PCM16 WAV, with its original time origin. The runner
  rejects other formats instead of silently resampling or changing the input.
- Local free space for an isolated optimized model cache and results. Model
  files are symlinked, not copied. Run on a filesystem that supports symlinks.

## Run the three disputed windows

From the repository root:

```sh
python3 scripts/phrase_loss_diagnostic.py \
  --binary ./target/release/gigastt \
  --audio /private/meeting.wav \
  --model-dir "$HOME/.gigastt/models" \
  --at 823.5 1087.8 1172.7 \
  --threads 6 \
  --output /private/diagnostic-threads6
```

The selected windows are **814–838**, **1078–1102** and **1166–1190** seconds.
Their PCM bytes are copied exactly from the input; the source file is untouched.
The geometry is the current CPU file path: 24-second windows, two-second overlap,
VAD disabled. Positions near the adjusted final window are deliberately rejected.
The `--at` values select windows, not the duration of a reference phrase.

The runner performs five recognitions: the full file with a fresh cache, the
full file with the reused cache, then the three standalone windows with that
same cache. It fixes the model variant, one file-window slot, encoder threads,
punctuation off and ITN off. Ambient `GIGASTT_*` settings are removed from the
child process so VAD/hotwords and other environment defaults cannot alter it.
Logs must confirm CPU execution and the expected source/cached graph path.
The timeout is 300 seconds **per invocation**, adjustable with `--timeout`.
This is diagnostic wall time, not a controlled performance benchmark.

For the thread-count control, repeat with `--threads 1` and a **new** output
directory. The original installed graph cache is not changed. CPU scheduling
may cap the requested thread count; record the host's available CPUs too.
Do not substitute FP32 bytes under an INT8 filename. This protocol initially
uses the published INT8 graph; an FP32 diagnostic is a separate experiment.

## What to inspect and share

The output directory is created with owner-only permissions on Unix. All files
remain local. **Do not attach the directory or an archive of it.**

- `summary.json`: counts, equality flags, model/binary hashes and architecture.
  It contains no recognized words, source audio hash, local paths or audio.
  Review it before sharing it. Zero words in both arms is not proof of correctness.
- `comparison-private.json`: token disagreements and timestamp coverage,
  including private text. Local timestamps are shifted to the original timeline.
- `full-cold.json`, `full-warm.json`, `window-*.json`: complete private hypotheses.
- `*.log`, `provenance.json`: commands, paths, source hash, versions and timings;
  these may contain private information. Partial runs retain their diagnostics.
- `windows/`: private audio clips; keep these local.
- `models/`: links to the installed models and the isolated optimized cache.

A failed command, unexpected provider/cache path or modified source/model stops
the run without a completed `summary.json`. An existing output directory is never
reused or overwritten. A failed run must not be reported as a zero-loss result.

## Interpret the comparison conservatively

Text comparison uses lowercase Unicode word tokens; it does not expand numbers
or equate different spellings. It aligns standalone-window tokens against full-file
tokens within `[window start + 1, window start + 23)`, excluding the overlap edges.
The saved raw hypotheses retain those edges. Differences at the comparison bounds
can themselves result from shifted timestamps, so inspect the nearby raw words.
The edit operations are diagnostic sequence alignment, **not true deletion labels**.

Timestamp coverage separately counts local word starts without *any* full-file
word start within ±0.5 seconds. It ignores word identity. This compares two
hypotheses, not the author's PyTorch reference; it is not the original lost-run
metric. A different word at the same time can satisfy coverage, and the same word
at a shifted time can fail it. No three-word lost-run count is claimed here.

Use the results to choose the next investigation:

1. A phrase absent from both the full file and the exact standalone window points
   toward recognition inside the window; inspect encoder/decode outputs next.
2. A phrase present in the standalone window but absent from the full file needs
   inspection of in-process pre-stitch hypotheses. This alone does **not** prove a
   stitcher defect: independently re-decoding a clip can change numerical paths.
3. Matching text with failing timestamp coverage points toward timing/alignment.
4. Cold/warm differences must be separated from window/full-file differences.

The useful first reply is the reviewed summary for six threads, then optionally
one thread, the binary source commit and CPU model, whether CLI reproduces the
REST losses, and whether the three phrases are audible/present in the local raw
outputs. The offered standalone scorer and a synthetic reference-format example
would let us reproduce the author's scoring without access to the private audio.
