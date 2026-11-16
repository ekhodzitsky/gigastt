# Experimental WebM/Matroska demuxing

Build with `cargo build --release --features unlace` to opt into the published
`unlace` 0.2.4 sequential demuxer for WebM/Matroska Opus files. This feature is
**off by default**. It enables file decoding and is forwarded by the server crate
to the core crate. It does not change raw PCM WebSocket recognition.

The adapter keeps the existing opus-rs packet splitter and decoder. It supports
one enabled mono/stereo Opus audio track with mapping family zero and zero output
gain. Other container formats and non-Opus WebM audio use the existing decoder.
The existing demuxer identifies the audio codec before strict Opus parsing;
routing is decided before emitting PCM. Malformed recognized WebM, unsupported
Opus parameters, discontinuous timelines and truncated blocks return errors;
there is no fallback after decoding starts.

Unlike the legacy WebM path, this adapter applies signed DiscardPadding to the
whole decoded block, then removes Opus pre-skip once across the stream, before
48 kHz mono PCM is resampled. CodecDelay is not subtracted a second time. The
container clock is checked against the untrimmed decoded frame count within one
container timestamp tick; later lace timestamps are not guessed. These choices
change output lengths and can reject files accepted by the default demuxer.

After codec identification, Opus input is read sequentially without constructing
an encoded-file index or retaining complete PCM in the file-window path. The
initial legacy probe may seek to existing SeekHead/Cues metadata, which is
discarded before strict decoding. Codec identification and strict header parsing share an 8 MiB input-read budget
(and strict header position limit). This opt-in ceiling can reject unusual
non-Opus Matroska headers before the codec is identified. Encoded
blocks to 1 MiB, packets to 64 KiB and lacing to 256 frames. The decoded mono block
has a separate ceiling of 256 × 5,760 samples; stereo decoder scratch holds at
most 5,760 frames per channel. Resampler staging and the requested output window
are additional bounded buffers. The existing `channels=split` API still returns
whole channel vectors and retains its separate whole-buffer duration ceiling.

Duration limits count decoded frames before trimming. Cooperative cancellation
is checked between packet decoding, block completion and resampling calls;
it cannot interrupt a blocking local read or an active decoder call. A successful
unknown-size Segment EOF is not proof that its producer intended to finish.

The native tests exercise signed padding across laces, pre-skip across blocks,
mono/stereo file/bytes/channel paths, bounded window retention, duration limits,
cancellation and truncated tails. Existing fixture comparisons establish packet
and untrimmed same-decoder equality on the shared admitted subset, but strict
trim changes PCM and the old independent libopus numeric threshold still fails on some files. In the
reconstructed corpus, maximum absolute differences were about half a PCM16 LSB;
reference output quantization may contribute, so these threshold failures do
not establish a severe decoder defect. This feature
is experimental: no ASR quality, performance superiority or transparent migration
guarantee is implied.
