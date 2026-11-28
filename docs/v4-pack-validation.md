# Versioned GigaAM v4 model pack validation

This page describes the **full-context** v4 validation contract. For the
explicit file and native WebSocket profile, see [v4 RNNT delivery](v4-rnnt-delivery.md).
Existing v3 variants, download behavior, precision defaults, and unversioned
manifests are unchanged. Do not replace an installed v3 pack or add v3 files
to a v4 directory.

The first profile covers the bilingual GigaAM v4 RNN-T FP32 split encoder and
both independent `raw` and `e2e` heads on an explicitly selected CPU, using
full-context inference with batch size one. Cached streaming, INT8, CTC,
Multilingual-v2, non-CPU providers, automatic provider fallback, and external
ONNX data are not qualified by this profile.

## Descriptor

A versioned `manifest.toml` requires both `schema_version` and `family`.
Either top-level key is reserved: a legacy loader rejects the declaration even
if legacy `architecture`/`files` keys and matching v3 files also coexist.
Malformed or unusable manifest files are errors, not missing manifests. Other
unversioned extension metadata remains accepted as before. The legacy download,
quantization, model-load and cache-maintenance paths do not act on a versioned
pack.

Unlike the legacy naming manifest, this descriptor is an integrity and
compatibility contract. The following example identifies the exact locally
qualified artifacts by role, byte size, and SHA-256, from the official
[upstream source](https://github.com/salute-developers/GigaAM/tree/e9bbf2e773eedd217c6f0d3b89b9728c41332b7d)
and [checkpoint](https://huggingface.co/ai-sage/GigaAM-v4/tree/9fe3ebf50c9e0cce5eed60d546938632238b6a96).
The [combined bundle installer](v4-rnnt-delivery.md) supports local archives
and explicit HTTPS URLs; it does not advertise a release URL. Preserve the
official MIT license and notice when packaging derived artifacts.

```toml
schema_version = 1
family = "gigaam_v4"
head = "rnnt"
styles = ["raw", "e2e"]
mode = "full_context"
precision = "fp32"
provider = "cpu"
abi = "v4_rnnt_split_v1"
external_data = []

[provenance]
source_repository = "https://github.com/salute-developers/GigaAM"
source_revision = "e9bbf2e773eedd217c6f0d3b89b9728c41332b7d"
model_repository = "ai-sage/GigaAM-v4"
model_revision = "9fe3ebf50c9e0cce5eed60d546938632238b6a96"
checkpoint_sha256 = "2c38690c38edbd9dfc2190f1a2dacdcfbea2a692078457fc145a6034c031840a"
license = "MIT"

[artifacts.encoder]
file = "v4_rnnt_encoder.onnx"
bytes = 951218063
sha256 = "62fd1735d1a3cf45d83d525c1857fd5df807be9c5d38cd61a5af52f9de2888b6"

[artifacts.raw_decoder]
file = "v4_rnnt_raw_decoder.onnx"
bytes = 4599970
sha256 = "2881cef81aa7f1e559293306bdbc2059be209065f6ce4ac45140b1c818cc8879"

[artifacts.raw_joint]
file = "v4_rnnt_raw_joint.onnx"
bytes = 2712938
sha256 = "38815775659e398cf9b0d977b92707557f2dcaea73c482233b7d5c6fb86fe95f"

[artifacts.raw_tokenizer]
file = "raw_tokenizer.model"
bytes = 15374
sha256 = "dcbe2598d1b374065091832970593a06065d498a43fbb1787de73e180d0556bc"

[artifacts.e2e_decoder]
file = "v4_rnnt_e2e_decoder.onnx"
bytes = 4599970
sha256 = "087ea392f825547ecc707e37f091d31ab211ec3ca2ee23dd86298d07621d4545"

[artifacts.e2e_joint]
file = "v4_rnnt_e2e_joint.onnx"
bytes = 2712938
sha256 = "4964b0a350f35c78da1d100705b83fe05dfdf95e8535b970192985385f366ed7"

[artifacts.e2e_tokenizer]
file = "e2e_tokenizer.model"
bytes = 14686
sha256 = "f547bcb38e071694ef1ddb25d9d40c86b4ab88cd0b16ab9fd52f27dd52c48e39"
```

Use safe, distinct basenames and regular files; symlink artifacts and
nonempty external-data inventories are rejected. All seven roles are required,
including the unselected head. The trusted ABI allowlist pins the actual
qualified bytes for each role, not merely a user-declared checksum: a different
export, a swapped same-shaped head, or a graph with incompatible dimensions is
rejected even if its own manifest is internally consistent. Extra unreferenced
files are not used. A later qualified artifact set requires a separate profile
or an explicit profile update; this is not a generic ONNX inspection API.

`VersionedPackDescriptor::parse` validates the descriptor and its supported
semantic metadata without accessing model files. `QualifiedV4Pack::load`
additionally validates the declared pins and streams every required file through
SHA-256. It neither creates ONNX sessions nor accesses the network.

```rust
use gigastt_core::model::{PackSelection, QualifiedV4Pack, TranscriptStyle};
use std::path::Path;

let pack = QualifiedV4Pack::load(Path::new("/path/to/local-v4-pack"))?;
let raw = pack.select(PackSelection::cpu_full_context(TranscriptStyle::Raw))?;
let e2e = pack.select(PackSelection::cpu_full_context(TranscriptStyle::E2e))?;
assert_eq!(raw.encoder, e2e.encoder);
assert_ne!(raw.cache_identity, e2e.cache_identity);
# Ok::<(), anyhow::Error>(())
```

The whole-pack identity is domain-separated, includes provenance and all
role-specific sizes and digests, and is independent of directory, basename,
TOML presentation, and style ordering. Selection identity additionally binds
the style. Neither is the legacy optimized-encoder source-file hash. Validation
is a snapshot: future inference must still consume the unchanged verified model
bytes, not assume that mutable files remain valid forever.

## Semantic split ABI

The exact admitted graphs use ONNX opset 17. In the following table `B=1`, `M`
is the mel sequence length, `T` the encoder sequence length, and `W` the joiner
scoring window. The qualified profile uses 16 kHz PCM, 64 mel bins,
320-sample FFT/window, 160-sample hop, `center=false` and subsampling by four.

| Graph | Inputs | Outputs |
|---|---|---|
| Shared encoder | `audio_signal` FP32 `[B,64,M]`, `length` INT64 `[B]` | `encoded` FP32 `[B,768,T]`, `encoded_len` INT64 `[B]` |
| Either predictor | `x` INT64 `[B,1]`, `hi`/`ci` FP32 `[1,B,320]` | `dec` FP32 `[B,1,320]`, `ho`/`co` FP32 `[1,B,320]` |
| Either joiner | `enc` FP32 `[B,768,W]`, `dec` FP32 `[B,320,1]` | `joint` FP32 `[B,W,1,1025]` log-softmax |

Each head has its own 1024-piece SentencePiece model. Blank is label 1024;
initial predictor state and the blank embedding are zero. The author greedy
contract permits at most 10 symbols per encoder frame and scores windows of
eight frames. The full-context encoder takes no streaming cache state. The
encoder's ONNX output dimensions are symbolic in the graph metadata; actual
width and valid lengths are not inferred from those symbolic strings.

The admitted graph set was compared with the pinned FP32 CPU author reference
on four whole Russian/English short and longer recordings; both heads matched
token IDs, frames and text, and instrumentation checked log-softmax and selected
token confidence parity. This is not an independent WER claim, an arbitrary
length or batching qualification, nor proof of Rust frontend/tokenizer or
streaming parity. The author fused ONNX export produces token IDs only and does
not meet the metadata-preserving split contract. The optional core engine uses the split
contract below; a fused graph is not a substitute.

## Qualified tokenization and timestamp helpers

The `inference::v4::V4Tokenizer` library helper loads a selected raw or e2e
SentencePiece tokenizer from the validated pack. It rehashes the exact small
model bytes before parsing them; changing the file after pack validation is an
error. It does not use the legacy text vocabulary, greedy longest-match
encoding, or filename-based selection.

The two qualified BPE models use their embedded identity normalization and
whitespace rules. Each has 1024 pieces, unknown ID 0, no BOS/EOS/PAD IDs, and no
byte fallback. RNNT appends blank ID 1024 outside the SentencePiece model; blank
and other out-of-range IDs are rejected by token decoding and literal-piece
lookup. The raw and e2e unknown surfaces differ, so unknown IDs must not be
silently dropped or treated as blank. `id_to_piece` returns the literal piece;
`decode` applies SentencePiece's sequence-level whitespace and unknown semantics.

```rust
use gigastt_core::inference::v4::V4Tokenizer;
use gigastt_core::model::{QualifiedV4Pack, TranscriptStyle};
use std::path::Path;

let pack = QualifiedV4Pack::load(Path::new("/path/to/local-v4-pack"))?;
let tokenizer = V4Tokenizer::from_pack(&pack, TranscriptStyle::Raw)?;
let ids = tokenizer.encode("привет world")?;
let text = tokenizer.decode(&ids)?;
assert_eq!(tokenizer.blank_id(), 1024);
# Ok::<(), anyhow::Error>(())
```

The exact pinned `sentencepiece-rs` 0.2.2 dependency is a pure-Rust,
Apache-2.0-licensed runtime with no transitive dependencies, native build, or
system tokenizer requirement. Both specific model files were checked against
the official SentencePiece processor: all 2048 literal pieces and singleton
decodes, 12,288 boundary/unknown contextual decodes, 54 curated text encodes,
20 direct sequences (including invalid IDs), and eight word-timing cases.
This qualifies these exact models, not arbitrary SentencePiece files or an
unbounded equivalence claim for every possible string.

`V4FrameGrid` tracks physical and valid lengths separately for the pinned
center=false ABI: `M = floor((samples - 320) / 160) + 1`, and expected encoder
length `T = ceil(M / 4)`. Its full-context timestamp shift is valid sample count
/ 16,000 / valid encoder length, not a fixed 40 ms interval. `V4Tokenizer::words`
groups literal pieces on that grid, retaining the author's bare-boundary
frame semantics and exclusive frame ends. The optional core engine validates the
actual returned encoder length before using this geometry; it is not a native
streaming clock or a source of confidence scores.

The default build has **no v4 acoustic inference path**. The portable
RustFFT frontend experiment failed the fixed parity gate for both 8 kHz-resampled
inputs. The discrepancy came from float32 FFT cancellation near the log floor;
neither a real-input Rust FFT nor a higher-precision mathematical FFT met the
same author tolerance. These unsuccessful experiments do not justify fallback
or changing the existing v3 frontend.

## Optional experimental CPU acoustic frontend

On GNU Linux x86-64 only, building `gigastt-core` with the opt-in `v4-mkl`
feature exposes `inference::v4::V4Frontend`. It remains off by default,
including in lean builds. On other targets the feature does not expose this
API. This feature also enables the explicit core engine below, but does not
add PyTorch, implicit provider fallback, server routing, or automatic
runtime installation.

The initial profile uses Intel oneMKL 2024.2 build 20240605, sequential LP64,
with the exact 12 libraries and SHA-256/size pins recorded in
[`mkl-profile.json`](../crates/gigastt-core/src/inference/v4/frontend/mkl-profile.json).
The complete vendor CPU/VML closure is about 350 MiB on disk. It was chosen
after an isolated seven-library loader experiment showed a genuine missing
VML-library search; all five documented CPU VML kernels are now explicit
members of the link group. This profile is numerically qualified on the
measured Linux x86-64 host, not every CPU, glibc, compiler, or MKL release.

Supply the exact local [Intel MKL 2024.2.0 wheel](https://pypi.org/project/mkl/2024.2.0/),
obtained separately by the operator. The helper neither downloads nor installs
anything system-wide. It verifies the wheel, extracts only the listed CPU
libraries and message catalog, retains the Intel Simplified Software License
and third-party notices, and compiles the project's small public DFTI bridge:

```sh
python3 scripts/build-v4-mkl-runtime.py \
  mkl-2024.2.0-py2.py3-none-manylinux1_x86_64.whl /path/to/new-runtime
```

The first profile pins the exact bridge binary as well as the vendor files.
The measured bridge was built with GCC 15.2.0 (Ubuntu 15.2.0-16ubuntu1), GNU
Binutils 2.46 and glibc 2.43 using the flags in that helper. A different
compiler/link result is rejected, even if the source is identical; it needs
separate qualification. The directory must remain operator-owned, trusted
and immutable while the application runs. Model validation is not a sandbox
for native executable code. Unexpected runtime-directory entries (including
loader hardware-capability overlays), symlinks, wrong hashes or version,
missing symbols and incompatible namespace paths are errors.

```rust
use gigastt_core::inference::v4::V4Frontend;
use gigastt_core::model::QualifiedV4Pack;
use std::path::Path;

let pack = QualifiedV4Pack::load(Path::new("/path/to/local-v4-pack"))?;
let frontend = V4Frontend::from_pack(&pack, Path::new("/path/to/new-runtime"))?;
let samples = vec![0.0_f32; 3200];
let features = frontend.compute(&samples)?;
assert_eq!(features.grid.valid_mel_frames(), 19);
# Ok::<(), anyhow::Error>(())
```

The bridge is an explicit isolated GNU dynamic-loader namespace, with its
pinned libraries in the same directory and `$ORIGIN` search. The process
never sets global loader environment variables or MKL thread settings.
Explicit environment overrides that can change the qualified MKL arithmetic
are rejected. Each transform uses one local MKL thread, checked aligned,
disjoint input/output buffers, and a bounded, separately freed descriptor;
the calling thread's setting is restored. One selected namespace remains for
the process lifetime: do not hot-reload or replace it, and restart after a
native status, namespace, or ownership failure. Calling again with the same
qualified runtime reuses it. A different runtime is not substituted. The
vendor libraries retained mappings after a successful `dlclose` in the
qualification probe, so no unload guarantee is made.

The frontend accepts finite mono 16 kHz float PCM, from one complete 320-sample
window up to 30 seconds, the bounded qualification scope. It does not
resample, center, implicitly pad or truncate. `compute_padded` distinguishes
explicit physical padding from a separate valid prefix; the returned
`V4FrameGrid` reports physical/valid mel lengths and expected encoded
lengths from causal subsampling. Actual encoder output is validated by the optional core inference
runtime below. There is no v4 streaming or batching claim.

The 42,496-byte coefficient asset contains the exact pinned author's FP32
Hann window (320 values) followed by the HTK mel-major 64×161 matrix, with
no precision reduction. `scripts/extract-v4-frontend.py` reconstructs it from
the pinned safetensors checkpoint after its exact SHA-256 check, reading
only `model.preprocessor.featurizer.0.spectrogram.window` and `mel_scale.fb` and
transposing the latter. Asset SHA-256 is
`15c9da5c673fcfe5f6fa5dbfd28b8e19b658c14f15fd4c7ff656405764a417c0`.
Custom distributors enabling `v4-mkl` must accompany the embedded coefficient
asset with its [upstream MIT notice](../crates/gigastt-core/src/inference/v4/frontend/gigaam_v4_frontend.LICENSE).
The separately supplied Intel binary libraries remain governed by their own
license and notices; gigastt does not bundle them by default.

The actual optional Rust frontend API passed the frozen author comparison on
18 supported inputs and rejected all three sub-window inputs. This included
short/odd/long inputs, explicit padding with a separate valid prefix, and the
same fixed float PCM from both the author and native resampling paths at
8/16/48 kHz. Across the supported feature arrays the maximum error was
1.91e-6 and the worst active-bin mean error was 4.91e-8. Three frozen
spectrum/power/mel/log controls were bit-identical. The 8 and 48 kHz waveform
resampling algorithms themselves are different; this is feature parity on
matched input PCM, not a claim of identical FFmpeg and Rubato samples or
inference transcript quality.


## Explicit experimental core CPU engine

On the same opt-in GNU Linux x86-64 profile, `inference::v4::V4Engine`
loads the qualified seven-artifact local pack and the explicitly supplied
MKL runtime. The explicit core API does not implement streaming or batching.
The opt-in [file CLI and server profile](cli.md#explicit-v4-full-context-file-profile)
uses this engine for bounded full-context requests; no automatic v4 routing
or native WebSocket streaming is enabled by this pack. The default legacy
engine is unchanged. The separate [native v4 WebSocket profile](v4-native-websocket.md)
requires its own cached-streaming pack.

```rust
use gigastt_core::inference::v4::{V4Engine, V4Pcm};
use gigastt_core::model::TranscriptStyle;
use std::path::Path;

let engine = V4Engine::load(
    Path::new("/path/to/local-v4-pack"),
    Path::new("/path/to/new-runtime"),
)?;
let samples = vec![0.0_f32; 3200];
let result = engine.transcribe(V4Pcm::full(&samples), TranscriptStyle::E2e, None)?;
# Ok::<(), anyhow::Error>(())
```

Exactly one shared acoustic encoder and two independent style-specific
predictor, joiner and SentencePiece heads are loaded. Each ONNX graph is
rehashed into an owned byte buffer, and that same buffer is parsed by ONNX
Runtime with the explicit CPU provider and one intra/inter-op thread.
No optimized graph cache or legacy variant fallback is used. Keep the
operator-owned pack immutable; this loader does not monitor files after
session creation. The graph load temporarily needs memory for the full
FP32 encoder byte buffer as well as ONNX Runtime's allocations.

Each synchronous request owns a single engine slot, with fresh recurrent
prediction state. The explicit `transcribe_both` path runs the acoustic
encoder once and decodes both styles independently. Cancellation is
cooperative while waiting for the slot and around each native stage;
an in-flight native call is not preempted. Cancelled requests return no
partial state or transcript. Only finite mono 16 kHz float PCM within the
frontend's stated physical/valid length limits is accepted.

The v4 greedy schedule scores an eight-frame joiner window, clamping the
gather to physical encoder length and searching only its valid prefix. It
commits recurrent state only on nonblank emissions and limits each frame
to ten symbols. Returned IDs, frames, and complete decoded text remain
separate from the author's literal word grouping. Token confidence is the
selected model log-softmax exponent, and the word confidence is the mean
token confidence within that group. The overall confidence is
duration-weighted across words (or absent for empty groups). These are
gigastt's derived model scores, not author-provided word confidences or
calibrated correctness probabilities. Timestamps have the valid full-context
utterance's exclusive-frame-end convention, not a streaming clock.

The core API is restricted to the pinned FP32 split profile, not arbitrary
v4 weights, other execution providers or quantization. Numerical qualification
uses matched PCM/feature inputs, not arbitrary audio quality or a WER claim.
