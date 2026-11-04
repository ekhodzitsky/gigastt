# Telephony fixtures

`g711_alaw_benchmark.alaw` and `g711_mulaw_benchmark.ulaw` freeze the exact
payloads used by the original decoder benchmark. Their source, preparation and
hashes are recorded in `g711-provenance.json`. Freezing the inputs keeps both
revisions on the same workload when the development-only fixture encoder is
updated. The old mu-law encoder had a bit-complement error; these remain valid
mu-law decoder inputs, but are **not** an encoder correctness oracle.

`pcmu_codepoints_ffmpeg.pcm` is an independent PCM16 reference for every mu-law
code from 0 through 255. Regenerate it with
`python3 scripts/generate_g711_reference.py` from the repository root. The unit
test checks that the fixture encoder preserves all canonical levels (both zero
codes canonicalize to `0xff`) and that production decoding matches the same
reference after resampling. This checks codepoint preservation; it does not
assert that every 16-bit PCM quantization boundary matches FFmpeg.
