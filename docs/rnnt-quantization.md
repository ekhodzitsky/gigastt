# RNN-T model quantization and phrase loss

The default packaging recipe keeps convolutions in FP32 and quantizes constant
MatMul/Gemm weights per output channel to INT8. Runtime remains INT8-only;
FP32 encoders are packaging inputs, not a second runtime mode. The encoder is
about 305 MiB instead of 215 MiB. This trade-off prioritizes recognition quality
and latency; it requires more memory. General Gemm operations with bias,
transposes or scaling retain their original float semantics; only plain
MatMul-equivalent Gemm operations use the integer chain.

`gigastt quantize --model-dir SOURCE --model-variant e2e_rnnt --force` rebuilds
that head explicitly even in a directory containing both heads. `--skip-conv`
remains a compatibility alias for the default recipe. `--quantize-conv` is an
experimental opt-in to the old dynamic ConvInteger recipe. Existing output is
never silently accepted as a successful conversion: replacement needs `--force`.
A graph with no quantized weights is rejected, leaving any existing output intact.

## Evidence and limits

The [phrase-loss report](https://github.com/ekhodzitsky/gigastt/issues/376)
contains controlled comparisons of the same private 2040.8-second meeting.
On Ryzen 9950X the shipped encoder lost six runs / 33 words, the per-channel
MatMul-only encoder lost three / 14, and FP32 lost one / seven. Processing time
was approximately 48, 23 and 32 seconds respectively. On Ryzen 5600X the
MatMul-only recipe also improved both phrase loss and speed. VNNI did not remove
the regression. These are reporter measurements, not independently reproduced
results on that private recording.

The [local public evidence](../benchmark/results/rnnt-quantization/2026-10-09/summary.json)
uses Linux x86_64 on Ryzen AI 9 HX 370: 2714 reference words across 1154 seconds.
RNN-T errors decrease from 200 to 197, e2e RNN-T from 160 to 158. Median paired
warm wall time is 0.547x and 0.560x baseline; peak RSS is 1.233x and 1.221x.
There are no three-word deletion runs in either public arm. On one e2e podcast
excerpt the candidate has one more word error; the corpus total improves.
Full transcripts are content-addressed beside each report. These observations
cover this host and corpus, not every CPU or utterance.

The additional [Golos short-form evidence](../benchmark/results/rnnt-quantization/2026-10-09/golos-report.json)
uses the canonical `golos_crowd_1k` slice and existing benchmark normalization:
992 non-empty references, 4732 normalized words; eight empty references are
excluded by the existing corpus policy. RNN-T errors decrease from 139 to 138
(2.937% to 2.916% WER). E2E RNN-T errors increase from 397 to 399 (8.390% to
8.432%). Five e2e utterances worsen and three improve; some differences concern
English name spelling or abbreviations, but there are also word substitutions.
Raw scoring is retained too (RNN-T 139 to 139; e2e 756 to 762 errors on 5013
words). The normalizer has not been changed to erase these differences.
The measured e2e trade-off was explicitly accepted on 2026-10-09 in exchange
for fewer long-form phrase losses and approximately twice the CPU speed.
These Linux measurements do not replace the historical macOS benchmark numbers.

The remaining errors have distinct boundaries: dynamic MatMul quantization
still changes predictions relative to FP32, while the phrase missing at 1087.8
seconds is also missing in FP32. Removing Conv quantization cannot establish a
fix for that residual. The private input is unavailable, so it is not valid to
claim that all losses are eliminated. A Gemm-only recipe does not provide a
useful intermediate model: this encoder contains no Gemm nodes and the old
quantizer silently wrote the unmodified FP32 graph.

Our public comparison uses all seven hash-pinned recordings in
`benchmark/manifests/longform_stitch.json`, including four podcast excerpts.
It evaluates both RNN-T heads with the same native CPU binary, six encoder
threads, serial file windows, no VAD, punctuation or ITN, one cold process and
two cache-warm processes for each recording and model. Order alternates between
baseline and candidate. Separate model directories isolate optimized caches.
The CLI uses one inference slot. Raw transcripts, GNU time wall/RSS samples,
model/fixture/binary hashes and machine information are preserved.

The quality gate requires aggregate word error rate not to increase and checks
consecutive reference-word deletion runs of at least three words on each file.
Individual substitution differences are retained in the report. This text
alignment metric differs from the reporter's timestamp-only lost-run metric;
neither should be described as the other. Public data cannot prove correctness
on the private meeting or on every language/domain.

## Model release checks

`benchmark/model-release.json` pins the FP32 source files, the immutable old
bundle and the exact candidate encoder hashes. The release workflow rebuilds
both heads from these sources; runtime `download` is never used to produce a
candidate. Every source and output is verified. The old workflow called
`download`, which had become INT8-only, so its supposed rebuild just copied the
already published bundle. It also only printed hashes and had no model quality
or latency gate.

`Release Model` now runs the public long-form comparison before making a bundle
available for publication. A missing, invalid or incomplete measurement fails.
It also validates the saved Golos evidence against every proposed model file,
the frozen reference manifest and complete transcript coverage, then recomputes
both score variants from the transcripts. A short-form WER increase blocks
publication independently of long-form improvement unless an explicit approval
matches the exact model files, reference manifest and all measured transcripts.
The accepted e2e transition permits 397 to 399 errors only for that evidence;
changing models or transcripts does not inherit the exception.
Publication also requires successful main CI for the exact source commit and
refuses an existing model tag. Optional minisign signatures are retained.
The publication baseline must match the runtime download pin. After promotion,
a future candidate must be compared with the newly active bundle; it cannot
reuse an older, slower reference to conceal a regression. The historical recipe
remains usable with `publish: false` to reproduce its original comparison.
Candidates can be built with `publish: false` before promotion. Model activation
in runtime pins follows publication, so clients never reference an unavailable
bundle and old installed versions retain their immutable download source.

Standard limits are 5% warm latency, 10% cold latency, and 10% peak RSS/model
size. The explicitly approved transition from the old ConvInteger model to the
exact new MatMul-only hashes permits up to 50% peak RSS and 45% encoder-size
increase; this exception cannot apply to other model bytes. Accuracy and latency
checks remain active. Size and memory changes must be reported, not hidden by
an overall speed improvement.

Run locally on an otherwise idle Linux host:

```sh
python3 scripts/prepare-model-bundle.py --binary target/release/gigastt --root /tmp/model-bundle
python3 scripts/longform_stitch.py prepare --audio /tmp/model-audio
python3 scripts/model-quality.py --binary target/release/gigastt --head e2e_rnnt \
  --baseline /tmp/model-bundle/baseline/e2e_rnnt \
  --candidate /tmp/model-bundle/candidate/e2e_rnnt \
  --audio /tmp/model-audio --output /tmp/model-e2e-results
```

Repeat for `rnnt`. Do not run compilation or other inference jobs concurrently
with resource measurements. Keep the complete evidence when a candidate fails.

To reproduce short-form evidence with the existing Golos WAV corpus:

```sh
python3 scripts/measure-model-short-quality.py --binary target/release/gigastt \
  --models /tmp/model-bundle --audio "$HOME/.gigastt/benchmarks/golos_wav" \
  --output /tmp/model-golos-results
python3 scripts/model-short-quality.py
```

The second command checks the committed report selected by
`benchmark/model-release.json`, including the narrowly scoped e2e approval.
Batch wall time and peak RSS in the short-form report are informative;
the repeated cold/warm long-form measurements enforce the resource limits.

## Upgrading existing installations

Fresh downloads use the immutable `models-v3-2026-10-09` bundle. Existing
complete model directories are deliberately reused, including locally quantized
models; installing a new binary or running `download` over a complete directory
does not replace its encoder.

Use a binary containing the updated model pins, download into a new directory,
and select it explicitly. Each head needs about 310 MiB (rnnt 309.4 MiB,
e2e_rnnt 311.8 MiB), plus its generated optimized cache and optional sidecars:

```sh
gigastt download --model-dir "$HOME/.gigastt/models-v3-2026-10-09" --model-variant rnnt
gigastt download --model-dir "$HOME/.gigastt/models-v3-2026-10-09" --model-variant e2e_rnnt
gigastt transcribe recording.wav --model-dir "$HOME/.gigastt/models-v3-2026-10-09" --model-variant rnnt
gigastt serve --model-dir "$HOME/.gigastt/models-v3-2026-10-09" --model-variant rnnt
```

For e2e output choose `--model-variant e2e_rnnt` on `transcribe` or `serve`.
Stop an existing server before starting another on the same port; update its
service configuration to use the new directory. Keep the previous directory to
roll back by selecting it again. No model or recording needs to be deleted.
The old immutable release remains available for older binaries.

The encoder hashes are:

- `rnnt`: `1d5a6f580b692e38ba35ed98d0f54648808f27b0a147dc11bd8978c15dc1a18f`
- `e2e_rnnt`: `16805903b102d8b27bb0253de044f25e33a8a5bf07fce0c4da4ac1a5a7544091`
