# Versioned GigaAM v4 model pack validation

This is a **validation-only** library contract. GigaAM v4 inference is not yet
available through the engine, CLI, HTTP, WebSocket, or bindings. Existing v3
variants, download behavior, precision defaults, and unversioned manifests are
unchanged. Do not replace an installed v3 pack or add v3 files to a v4 directory.

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
There is not yet a downloadable gigastt v4 bundle. Preserve the official MIT
license and notice when packaging any derived artifacts.

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
not meet the metadata-preserving split contract. Native inference and word
metadata integration remain subsequent work.

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
frame semantics and exclusive frame ends. Future inference must validate the
actual returned encoder length before using this geometry; it is not a native
streaming clock or a source of confidence scores.

There is **no v4 acoustic frontend implementation or inference path** in this
validation slice. A local frontend experiment matched the short, longer, odd,
padded, native 16 kHz, and 48 kHz-resampled examples, but failed the fixed
numerical parity gate for both 8 kHz-resampled inputs. The discrepancy was
localized to float32 FFT backend cancellation near the log floor, not to
SentencePiece or the model-pack contract. Those results do not justify
replacing the v3 frontend, introducing an implicit fallback, or claiming v4
acoustic parity.
