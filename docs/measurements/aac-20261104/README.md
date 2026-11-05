# syom 0.10.0 qualification on current main

**Not accepted for merge yet.** The optional adapter builds against the published
0.10.0 crate, but decoder and HTTP parity checks against the standard decoder
on main do not pass. No thresholds were relaxed and no service was deployed.

The base is main `adf4db0d7b06a150534734b69b925e3136f0a631`. Both servers and
codec probes were built separately with the same toolchain/release profile.
The candidate enables `extended-aac`; the standard decoder does not. Codec
probes use the same source and feature union apart from the optional backend.
Historical 0.8.1/215 MiB measurements remain unchanged in the earlier report.

## Results

- Default workspace: 1269 tests passed; optional AAC: 1278 passed. Both skip
  49 model tests in the ordinary suite. Strict Clippy passes in both builds.
- All 29 [PCM cases](pcm-parity.json) are bit-identical between the preserved
  current-consumer syom 0.9.0 probe and syom 0.10.0. This is version parity,
  not parity with Symphonia or a new conformance claim.
- Seven alternating decoder pairs pinned to one allowed CPU retain the 1.05
  latency limit. Against [standard decoding](decoder-timing.json), short file,
  bytes and ADTS ratios are 1.083/1.093/1.092; long stereo file is 1.136.
  Split and scan pass at 1.027/1.017. The gate fails.
- The [version-only comparison](version-timing.json) passes all six cases
  (0.9.0 versus 0.10.0 on the same refreshed consumer). The default-comparison
  failure is not isolated to the new library version by this evidence.
- The [HTTP canary](http-canary.json) uses the active 305 MiB RNNT encoder,
  pool 1, two encoder threads, CPU, punctuation/ITN off. All 32 requests return
  200; 31 normalized text hashes match. `golos_04-aac` differs. M4A durations
  also differ, so duration parity fails. Priming/padding is a hypothesis to
  investigate, not an established cause or a reason to bypass the check.
- HTTP latency and peak RSS limits pass; invalid/truncated input checks pass.
  The servers drain normally. Load/fault qualification was not rerun after
  these prerequisite failures; it is not claimed as qualified on this model.

[Validation metadata](validation.json) pins the active encoder and registry
checksum. No audio or PCM is redistributed. The local HTTP runner copies only
remove the unused dynamic-library override and label the actual static ORT
archive fingerprint correctly; acceptance assertions remain unchanged.
The HTTP corpus contains WAV controls, AAC-LC speech and HE controls. It is
not a representative population WER measurement or cross-platform acceptance.

## Follow-up: the changed transcript is correct

A [targeted independent check](golos04-presentation.json) resolves the changed
`golos_04-aac` output. The human reference, original WAV, FFmpeg-decoded M4A
bounded to its 5.220-second presentation, and syom all produce
`ищи телеканал про бизнес на тиви`. The standard main AAC path instead produces
`ищи телеканал про бизнес на тв`.

The standard path returns 84,992 samples, syom 83,520 (16 kHz). Aligning the
standard PCM by 1,024 samples reduces RMS difference from 0.0943 to 5.52e-8;
the remaining excess tail is 448 samples. This is evidence of unremoved
priming/padding in the standard path for this recording. It explains the
5.312-versus-5.220-second duration difference; the latter matches the container
presentation duration. It does not establish the cause for every other file.

Therefore the old transcript/duration equality assertion fails on a demonstrable
correction in this example. It must not be interpreted as an ASR quality
regression. Corpus-level reference scoring and the still-failing decoder speed
gate remain necessary. The recorded failed canary is retained unchanged.
