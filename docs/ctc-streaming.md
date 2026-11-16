# CTC streaming on Linux CPU

`ml_ctc_large` is an explicit alternative to the default RNN-T head. It uses the
existing WebSocket endpoint and INT8 model; no training or GPU is required.
This guide describes the tested Linux CPU configuration, not a change to the
default model or a general solution to recognition errors.

To keep the RNN-T transcript head and use CTC only to confirm streaming speech
and automatic endpoints, see [RNN-T with CTC confirmation](stream-ctc-confirmation.md).
That opt-in mode loads both models and has different resource and latency costs.

## Download and run

```sh
gigastt download --model-variant ml_ctc_large --skip-diarization
gigastt --offline serve --model-variant ml_ctc_large --execution-provider cpu \
  --pool-size 1 --encoder-intra-threads 1 --punctuation off --itn off
```

The download command requires network access. The server command uses cached
files under `~/.gigastt/models/`; pass `--model-dir` consistently to both commands
for another location. The head needs `multilingual_large_ctc.int8.onnx` and
`multilingual_vocab.txt`, from the pinned
[`istupakov/gigaam-multilingual-large-ctc-onnx` catalog](models.md).
Their total size is approximately **592 MB (564 MiB) on disk**, not process RAM;
optimized graph caches use additional storage. No custom manifest is required.

Connect to `ws://127.0.0.1:9876/v1/ws`. Follow the existing
[WebSocket protocol](api.md#websocket--streaming) to configure the source sample rate, send
PCM16 mono audio and send `stop` when the audio ends. The validation below used
8 kHz input in 100 ms packets, with Stop immediately after the final packet.
VAD is off by default and was not enabled. The explicit single-thread setting
reproduces the measured setup; it is not a throughput recommendation.

## Partial and final results

CTC does not provide the RNN-T blank-run endpoint signal. In this VAD-off
configuration, use Stop to finish a recording; do not assume that a pause will
produce a speech final. Window-cap processing can commit stable prefixes while
continuing the stream. Read `committed`, `tentative` and `is_final` according to
the protocol rather than treating every partial as a separate utterance.

Re-decoding overlapping audio can move a word's predicted time beyond the
committed boundary. CTC stable-prefix reconciliation matches the fully covered
committed suffix against the new hypothesis, allowing leading words that the
existing time-coverage filter already rejects. It consumes that prefix once,
then preserves the normal coverage filter without a second same-word seam cut.
This addresses repeated publication of words such as “вас” and “по”. RNN-T and
legacy commit-mode behavior are unchanged.

## Bounded validation and limits

A local Linux CPU WebSocket evaluation used ONNX Runtime 1.31.0 with the pinned
large CTC encoder and unchanged vocabulary. It completed **31 conditions**:
a 96-second reported problem recording and 30 human-transcribed speech
conditions containing **220 reference words**. Compared with the unpatched CTC
baseline, final-transcript word errors decreased from **12 to 4**, with no
condition's final word-error count increasing. Scoring used case folding,
punctuation separation (preserving interior hyphens) and `ё` → `е`. The
conditions include correlated transforms
and previously examined examples; this is not a general accuracy benchmark.

On the problem recording, the earlier RNN-T run had **97 message-level false
“Да” events**; the tested CTC route had zero. Those are repeated partial/final
message events, not 97 distinct spoken words. Switching heads already removed
these events before prefix reconciliation. The reconciliation patch removed
the extra “вас” and “по” in the CTC output. The recording has no complete human
reference, so this does not establish its full transcript accuracy.

Matching text and predicted intervals do not prove acoustic occurrence identity.
If a new decode omits an old occurrence and substitutes a genuine new identical
phrase, reconciliation can suppress real speech. Genuine repetitions, quiet
speech and negation therefore remain important regression controls. Four
reference-word errors remain in this evaluation. These results neither repair
the RNN-T model nor establish a general fix for file-mode phrase omissions.

See [Models](models.md) for asset selection and [Languages](languages.md) for
separate multilingual file-transcription measurements.
