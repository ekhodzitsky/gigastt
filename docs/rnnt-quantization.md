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
Publication also requires successful main CI for the exact source commit and
refuses an existing model tag. Optional minisign signatures are retained.
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
