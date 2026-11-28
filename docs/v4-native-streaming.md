# Native GigaAM v4 streaming CPU core

The opt-in `v4-mkl` feature on GNU/Linux x86-64 exposes
`gigastt_core::inference::v4::V4StreamingEngine`. It is separate from the v4
full-context `V4Engine` and the default v3 inference path. This core API does
not by itself enable a network route; see the separate
[native v4 WebSocket profile](v4-native-websocket.md).

Load a [qualified cached-streaming pack](v4-streaming-pack.md) and the explicit
[qualified local oneMKL runtime](v4-pack-validation.md). The engine verifies
role-specific artifacts, loads the exact bytes into CPU-only, single-threaded
ONNX sessions, and does not download, reuse legacy optimized graph caches, or
fall back to older models. The current profile supports one stream per graph
invocation, FP32 acoustic computation, and the `raw` and `e2e` RNNT heads.

```rust,ignore
use std::{path::Path, sync::Arc};
use gigastt_core::{
    inference::v4::{V4StreamingEngine, V4StreamingStyles},
    model::V4StreamContext,
};

let engine = Arc::new(V4StreamingEngine::load(
    Path::new("/path/to/v4-streaming-pack"),
    Path::new("/path/to/qualified-mkl-runtime"),
)?);
let mut stream = engine.start_session(V4StreamContext::Ms320, V4StreamingStyles::Raw)?;
let packet = stream.push_pcm(&mono_f32_16k_packet, false, None)?;
let end = stream.flush(None)?;
```

Choose 160, 320, or 640 ms future context explicitly. These correspond to
4, 8, or 16 encoded frames, with 32 left-context frames. `push_pcm` accepts
canonical finite 16 kHz mono float32 PCM; it does not decode, resample, split
or insert waveform padding. Ordinary calls consume complete feature windows
and only ready context blocks. `flush` or `push_pcm(..., true, ...)` is EOF.
Only pending mel frames are zero-padded to the graph block; an incomplete
waveform window is not padded. The head sees only the valid acoustic prefix.
Repeated flushes and later audio on a finished session return empty deltas;
call `reset(context)` to start from a fresh state.

Each stream owns its waveform carry, pending mel, four frontend caches, 54
encoder caches, position, and selected head histories, including recurrent
state and cached predictions. The loaded engine allows two live state owners
and serializes their graph calls through one execution slot. Sessions can be
interleaved, but their states never share caches. Drop a session to release its
owner reservation. A packet is limited to 16,000 samples and a session to
57,600,000 samples (one hour); the latter is a memory bound, not a measured
continuous-serving limit. At most 319 waveform samples and fewer than one
context block of mel frames are retained between calls. Graph caches total
3,866,496 bytes per state; token history is also bounded by the session limit
and the maximum of ten labels per valid encoder frame. Returned deltas are
owned by the caller.

Raw tokens are newly emitted nonblank IDs with absolute encoder frames; text
and words may lag while the current word is incomplete. Word timestamps use
the fixed 40 ms grid; newly finalized words at EOF are clamped to the accepted
audio duration. Word confidence is derived from the model's selected token
scores, not a calibrated probability of transcript correctness. There is no
speaker, language-hint or extra text-normalization support. If SentencePiece
would rewrite earlier published text instead of appending, the packet fails
rather than reporting a corrupt suffix.

Invalid input and cancellation before entering packet processing leave the
state unchanged. Cancellation is cooperative between stages and while
waiting for the execution slot; an in-flight native call is not preempted.
A cancellation or inference error during processing returns no partial
packet, releases the execution slot when the call returns, and discards the
state. Explicit reset is required before that session can resume. A previous
successful result already returned to the caller remains unchanged.

This is a pinned CPU profile, not an assertion that arbitrary packet
partitions have identical recognition. The official helper's greedy decoder
receives local valid acoustic blocks, so the exact arrival schedule can matter.
