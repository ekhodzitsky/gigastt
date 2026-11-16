# RNN-T streaming with CTC confirmation

`serve --stream-ctc-model-dir` optionally loads a second, large CTC model to
confirm speech and automatic endpoints while keeping the RNN-T transcript head.
It is **off when the flag and `GIGASTT_STREAM_CTC_MODEL_DIR` are absent**.
This mode needs no training or GPU. It trades additional CPU, memory and
sometimes later partial results for fewer unsupported streaming words in the
bounded Linux CPU evaluations below.

## Prepare cached models and run

Download the two model packs separately. The server does not download the CTC
sidecar automatically:

```sh
gigastt download --model-variant rnnt --skip-diarization
gigastt download --model-variant ml_ctc_large --skip-diarization \
  --model-dir "$HOME/.gigastt/models-ctc-stream"

gigastt --offline serve --model-variant rnnt --execution-provider cpu \
  --pool-size 1 --pool-min-size 1 --batch-pool-size 0 \
  --encoder-intra-threads 1 --endpoint-mode auto \
  --stream-stable-prefix --punctuation off --itn off \
  --stream-ctc-model-dir "$HOME/.gigastt/models-ctc-stream"
```

The first command uses `~/.gigastt/models/`; pass `--model-dir` to both the RNN-T
download and server commands if using another main model directory. The CTC
directory must contain the pinned `multilingual_large_ctc.int8.onnx` and
`multilingual_vocab.txt` files. They add approximately **592 MB on disk**,
before optimized graph caches. Downloading requires network access; the server
command above uses only cached models. See [Models](models.md) for provenance.

Startup requires CPU, INT8 `rnnt`, pool size/minimum one, no batch
pool, file-window concurrency one, one encoder thread, automatic endpoints,
stable-prefix streaming and VAD
off. Incompatible startup options are rejected. This restriction reproduces the
qualified configuration; it is not a general recommendation to use one thread.
Existing WebSocket endpoint-mode configuration remains available; confirmation
applies only to automatic decoder endpoints. A missing or invalid sidecar
prevents boot or reload from succeeding; readiness is not reported for a
partially loaded engine.

Connect to `/v1/ws` and send raw PCM16 audio according to the
[WebSocket protocol](api.md#websocket--streaming). Send Stop after the final audio
packet. The evaluations used mono 8 kHz audio in 100 ms packets, the default
2.5-second retained window and 600 ms endpoint duration. File transcription,
REST, SSE and jobs do not use this confirmation step. This is not a VAD gate for
file audio, and does not address E2E file-mode quantization omissions.

## What changes in streaming

When the streaming assembler has no hypothesis, CTC scores the same real audio
window and native mel features. If every valid CTC frame selects blank, the
current RNN-T proposal is deferred. The audio remains subject to the bounded
streaming buffer policy; a blank result does not mark it as published speech or
create a permanent silence boundary. No word blacklist, confidence threshold,
artificial right padding or future audio is used.

An automatic RNN-T blank-run endpoint also requires CTC's trailing blank run to
reach the configured endpoint duration. The default 600 ms corresponds to 15
encoded frames. A CTC observation from the same window is reused; otherwise one
CTC forward is added when an automatic endpoint is proposed. RNN-T is not run
a second time. Explicit Stop does not wait for this endpoint confirmation;
normal current-window admission and flushing still apply. Model errors are
reported rather than silently switching off confirmation.

A real word can therefore appear later. In one genuine “да не” clip, ordinary
RNN-T first displayed “да” after 1.7 seconds of audio; confirmation first displayed
the complete phrase after the 2.78-second clip and Stop. The final words were
correct in both modes. Applications needing immediate acknowledgments should
account for this delay and continue to distinguish tentative, committed and
final text.

## Evidence and costs

An unchanged research implementation completed two paired Linux CPU streaming
screens using the same binary with confirmation off and on:

- A previously examined 32-condition screen reduced human-reference word errors
  from 17 to 6 over 214 words. On its 96-second reported problem recording,
  97 repeated partial/final false “Да” message events became zero while the
  baseline-supported “не молчи” and two “привет” occurrences remained. There is
  no complete human transcript for that recording. A separate control changed
  a baseline omission of “а не” into unsupported “они”, still two word errors;
  the aggregate improvement does not erase this ordinary lexical error.
- A prospectively selected 24-source Golos screen reduced errors from 8 to 5
  over 133 words, with three improved clips and none worse. Six unsupported
  “Да” messages across four sources, including two finals, became zero. Genuine
  “да”, baseline-correct negations and repeated words survived. The remaining
  shared errors include lexical/inflection and word-segmentation errors.

The second cohort excluded recorded prior audio/reference identities before
recognition. It does not establish different speakers, dataset independence or
absence from model training. Low-level recordings were selected from original
PCM by a fixed level rule, not certified as whispers. Every changed partial was
retained for review; final word-error counts do not measure response latency.
These short, bounded screens are not population accuracy or long-session
concurrency guarantees, and they do not establish a universal RNN-T repair.

Cold research-process high-water RSS was approximately **2.9 GiB with
confirmation versus 1.1 GiB without it**. Sampled CPU time rose about **2.3×** on
the short fresh screen and **1.2×** on the longer mixed screen. These include
startup and optimized-cache work in a research build; they are not steady-state
production benchmarks. Real-time packet delivery makes wall-clock duration an
unsuitable inference-speed comparison. Measure memory, CPU and response latency
on the deployment host before enabling this option.

For a CTC transcript head instead of RNN-T plus a sidecar, see
[CTC streaming](ctc-streaming.md). For the separate optional E2E file profile and
its known omissions, see [FP32 on CPU](fp32-cpu.md).
