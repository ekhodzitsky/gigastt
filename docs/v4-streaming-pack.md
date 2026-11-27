# GigaAM v4: explicit cached streaming pack

The cached-streaming profile is separate from the existing
[full-context v4 pack](v4-pack-validation.md) and legacy v3 model variants.
It is not a way to load the full-context encoder as a streaming encoder.
`QualifiedV4StreamingPack::load` verifies all local role-specific bytes and
rejects a missing or malformed manifest, unsupported context, provider,
precision, schema, checkpoint, or mismatched artifact. It never downloads a
model, changes the installed default, or falls back to another family or mode.

The initial profile pins the same upstream GigaAM source revision and bilingual
v4 RNN-T weights as the full-context profile, using the official dynamic native
streaming export instead. Both `raw` and `e2e` heads are required, even when a
session runs only one style. The measured profile is FP32, explicit CPU,
batch size one, mono 16 kHz PCM, four 10ms mel frames per 40ms encoder frame,
with supported future contexts of 160, 320 and 640 ms and a fixed 32-frame left
cache. Qualification is arithmetic and trajectory parity to the upstream
streaming implementation, not full-context equivalence or a WER claim.

Install both local directories with the [v4 RNNT bundle installer](v4-rnnt-delivery.md).
The bundle page also explains observed word-publication delay and runtime limits.

## Descriptor

The local directory must provide a bounded regular `manifest.toml`:

```toml
schema_version = 1
family = "gigaam_v4"
head = "rnnt"
styles = ["raw", "e2e"]
mode = "streaming"
precision = "fp32"
provider = "cpu"
abi = "v4_rnnt_streaming_split_v1"
future_context_ms = [160, 320, 640]
left_context_frames = 32
state_schema = "v4_zapformer_cache_v1"
external_data = []

[provenance]
source_repository = "https://github.com/salute-developers/GigaAM"
source_revision = "e9bbf2e773eedd217c6f0d3b89b9728c41332b7d"
model_repository = "ai-sage/GigaAM-v4"
model_revision = "9fe3ebf50c9e0cce5eed60d546938632238b6a96"
checkpoint_sha256 = "2c38690c38edbd9dfc2190f1a2dacdcfbea2a692078457fc145a6034c031840a"
license = "MIT"
```

Add one `[artifacts.<role>]` table for every role below, each containing a safe
unique basename `file`, its exact `bytes` and lowercase `sha256`. The verifier
checks each declaration against the pinned profile in source, then hashes the
actual file; a self-declared digest is not qualification. All artifacts must be
regular files (no symlinks). The official JSON files and YAML are provenance
artifacts only, checked as exact bytes. Their internal whole/advance graph
locators are not resolved or executed. The pack excludes those fused/token-only
outputs, which do not expose the required streaming frame and confidence state.

| Role | Official export basename | Bytes | SHA-256 |
|---|---|---:|---|
| `frontend` | `v4_rnnt_dynamic_stream_frontend.onnx` | 3,228,802 | `5fd2b06f0a83b1206d88b2a32c5785bcdf261005cc176aa65b74c4d94ca3ac84` |
| `step` | `v4_rnnt_dynamic_stream_step.onnx` | 948,135,265 | `75e57be63d3ed747d5e25985c86fbba406ad94b7dda7240659fa299598c3fac4` |
| `e2e_decoder` | `v4_rnnt_dynamic_e2e_decoder.onnx` | 4,599,970 | `087ea392f825547ecc707e37f091d31ab211ec3ca2ee23dd86298d07621d4545` |
| `e2e_joint` | `v4_rnnt_dynamic_e2e_joint.onnx` | 2,712,938 | `4964b0a350f35c78da1d100705b83fe05dfdf95e8535b970192985385f366ed7` |
| `raw_decoder` | `v4_rnnt_dynamic_raw_decoder.onnx` | 4,599,970 | `2881cef81aa7f1e559293306bdbc2059be209065f6ce4ac45140b1c818cc8879` |
| `raw_joint` | `v4_rnnt_dynamic_raw_joint.onnx` | 2,712,938 | `38815775659e398cf9b0d977b92707557f2dcaea73c482233b7d5c6fb86fe95f` |
| `e2e_tokenizer` | `v4_rnnt_dynamic_e2e_tokenizer.model` | 14,686 | `f547bcb38e071694ef1ddb25d9d40c86b4ab88cd0b16ab9fd52f27dd52c48e39` |
| `raw_tokenizer` | `v4_rnnt_dynamic_raw_tokenizer.model` | 15,374 | `dcbe2598d1b374065091832970593a06065d498a43fbb1787de73e180d0556bc` |
| `official_state_manifest` | `streaming.json` | 6,328 | `879597716d08ec7ad2631a87e5c5e18fc265cfc7fb33c729cb32223e69cfb372` |
| `official_manifest_alias` | `v4_rnnt_dynamic.streaming.json` | 6,328 | `879597716d08ec7ad2631a87e5c5e18fc265cfc7fb33c729cb32223e69cfb372` |
| `official_saved_config` | `v4_rnnt_dynamic.yaml` | 1,789 | `f7d32c94c7de7fe0e414232cf80caf0f16aaf3ff3581980cc73760737aa9185a` |

The complete required artifact set is approximately 966 MB. As with the
full-context profile, consumption re-verifies graph and tokenizer bytes after
pack admission; the validation snapshot does not certify future disk contents.
The stable whole-pack identity is independent of local names and includes both
heads, the streaming mode and context/cache ABI. Runtime graph metadata and
physical tensor geometry must be checked separately; symbolic ONNX dimension
names are not actual cache or frame lengths.

## Stateful tensor contract

For a context of `C` encoded frames (4, 8 or 16), the input mel block is `4*C`.
A normal packet admits complete blocks only; EOF admits the valid remainder,
with physical mel padding to a full block and valid lengths passed explicitly.
The streaming frontend graph accepts `[1,64,Mphysical]`, one valid mel length,
and four initial-zero caches. It returns `[1,Tphysical,768]`, one valid encoded
length and four next caches. The cached step graph accepts those features,
valid lengths, explicit context, position and valid cache length plus 54
initial-zero layer caches. It returns `[1,768,Tphysical]`, valid length,
next position, next cache length and the next caches. Their fixed physical
shapes are part of the code-level ABI; combined acoustic state is 3,866,496
bytes per stream before decoder state or other buffers. Only valid encoder
rows reach the selected heads; no padded frame is decoded into a transcript.

Each head has its own recurrent state, previous token, frame offset and cached
prediction including its next recurrent state. The streaming timestamp grid
is fixed at 40ms, unlike the full-context duration-dependent grid. The upstream
word-publication policy retains the incomplete trailing word until a boundary
or final flush. The EOF contract does not invent missing waveform samples or
consume incomplete STFT windows; reset clears acoustic, decoder and publication
state. These are native streaming semantics, not a rolling full-context
approximation.
