# Long-form stitching

Files longer than 30 seconds use independent 24-second windows with 2 seconds
of overlap on CPU, CoreML EP and CUDA (30-second windows on ANE). The encoder
and RNN-T predictor start afresh for each window. This bounds inference memory;
it also changes acoustic and label context compared with a single pass.

The stitch aligns words within the overlap by normalized text, sequence order
and start time. It maximizes matching words, then minimizes timestamp drift
(at most 400 ms per pair). Matching ignores case, punctuation and `ё`/`е`, but
the selected word retains its original text, timing and confidence. A matching
word near the overlap midpoint supplies a shared cut: both copies belong to
the same side, so timestamp jitter cannot independently retain or discard both.
Sequence alignment preserves repeated words such as “да да”.

Without an eligible match the midpoint rule remains: words starting at or
before the midpoint come from the earlier window; later words come from the
next window. An empty next hypothesis preserves the available preceding tail.
Alignment examines at most 64 words per side, independent of file length.
It preserves increasing start times and does not rewrite word timestamps.

VAD applies the same merge to the compressed timeline and maps timestamps back
to the original recording. Short pauses and 100 ms padding can survive VAD;
compressed audio is not necessarily continuous speech. Window parallelism and
the single-pass path for short files retain their existing behavior.

## Measurement protocol

The frozen [manifest](../benchmark/manifests/longform_stitch.json) contains the
official 71.25-second GigaAM example and two consecutive RuLS chapter excerpts
(69.62 and 83.43 seconds). These are clean read Russian verse, 455 reference
words before shifting. RuLS excerpts preserve segment order within a chapter;
they are not shuffled unrelated utterances. Original between-segment pauses
may have been trimmed by the dataset. References and audio SHA-256 values are
recorded in the manifest. The official example uses the published Pushkin poem,
including its spoken role heading; it is a literary-text reference, not an
independent time-aligned annotation.

RuLS attribution: [OpenSLR 96](https://www.openslr.org/96/) describes the source
as public domain in the USA; the [istupakov Hugging Face mirror](https://huggingface.co/datasets/istupakov/russian_librispeech)
used here declares CC BY 4.0. The selected literary texts are by Alexander Pushkin.

Compare the same binaries/configuration on `rnnt`, `e2e_rnnt` and `ml_ctc`, with
0, 3 and 22 seconds of leading silence. All three recordings run serially;
the official example also runs with VAD and with two concurrent windows. A
20-second prefix checks that single-pass outputs remain identical. This makes
48 runs per binary. Punctuation restoration and ITN are disabled; normalization
lowercases, folds `ё` to `е`, and splits on non-alphanumeric Unicode characters.
Report reference WER and substitution/deletion/insertion counts separately
from disagreement between shifted hypotheses. Shifts are repeated views of
the same audio, not independent corpus samples.

The pause ablation decodes each window once, verifies midpoint replay against
the production baseline, then moves each cut to the quietest 100 ms region in
the overlap, keeping 250 ms from either edge. It requires RMS at most 0.01
(-40 dBFS); otherwise the midpoint remains. Both alternatives use exactly the
same decoded words. This tests cut selection, not decoder-state propagation.

## Results (2026-09-17)

CPU/INT8, ORT `2.0.0-rc.13`, four encoder threads. Full provenance, hashes,
per-case counts and changed words with timestamps are in the
[measurement artifact](../benchmark/results/longform_stitch_cpu.json).
The serial matrix has 1,365 scored word occurrences per head (455 words at
three shifts):

| Head | Midpoint WER | Aligned WER | Substitution / deletion / insertion counts |
|---|---:|---:|---|
| `rnnt` | 1.83% | 1.83% | 21 / 0 / 4 → 21 / 0 / 4 |
| `e2e_rnnt` | 2.64% | 2.64% | 33 / 0 / 3 → 33 / 0 / 3 |
| `ml_ctc` | 6.01% | 5.49% | 69 / 3 / 10 → 69 / 3 / 3 |

On the official sample's VAD runs, one duplicate was removed from `e2e_rnnt`
and one from `ml_ctc`. All seven changed cases improve their reference edit
count; no tested head/mode regresses. Parallel-window texts retain their
baseline results. All three short-file JSON outputs, including word metadata,
are identical. Unit tests also cover loss of both copies, fast legitimate
repetitions, punctuation, partial words, empty hypotheses and bounded fallback.

The official sample reproduces four shifted-hypothesis substitutions for
`e2e_rnnt` and one for `ml_ctc` at a 3-second shift; the 22-second control has
zero. These substitutions remain with the aligned stitch. They are sensitivity
to window context, not proof of four additional errors against the reference.

Moving the timestamp-only cut to the quietest qualifying region was deferred:
on the same 27 serial cases it adds one error to `e2e_rnnt` and one to `ml_ctc`,
with `rnnt` unchanged. For example, it deletes a word in an otherwise correct
`e2e_rnnt` recording. A low-energy interval alone is insufficient evidence of a
safe word boundary. This does not rule out a better pause-selection method.

Separate paired resource runs (three per head/binary, alternating order,
71.25-second input) include CLI startup and model loading:

| Head | Median wall seconds, before → after | Peak RSS MiB, before → after |
|---|---:|---:|
| `rnnt` | 4.21 → 4.09 | 366.5 → 366.0 |
| `e2e_rnnt` | 4.11 → 4.00 | 388.8 → 396.6 |
| `ml_ctc` | 4.28 → 4.29 | 469.6 → 467.9 |

These small timing differences are not a speedup claim. RSS includes mapped
model pages and is not unique resident memory. All inference runs completed
under the 4 GiB memory ceiling with swap disabled; no OOM or timeout occurred.

## Reproduce

Requirements: Linux with user systemd/cgroup v2, Python 3 standard library,
GNU time, and the INT8 `rnnt`, `e2e_rnnt`, `ml_ctc` and Silero VAD models. The
models use the download pins and checksums in `model/variant.rs`. `run` is
offline; download required models before starting it. Save a baseline binary
from the parent revision before building the candidate. Both must use the
same feature set and build profile.

```sh
audio="$HOME/.cache/gigastt/longform-stitch"
python3 scripts/longform_stitch.py prepare --audio "$audio"

# Repeat with each binary and a new output directory. Runs are sequential;
# the harness refuses an unlimited or Orca-owned cgroup and stops on failure.
systemd-run --user --wait --pipe \
  -p MemoryMax=4G -p MemorySwapMax=0 -p RuntimeMaxSec=1200 \
  python3 "$PWD/scripts/longform_stitch.py" run \
  --audio "$audio" --binary /absolute/path/to/gigastt-baseline \
  --output "$audio/baseline"

python3 scripts/longform_stitch.py compare \
  --baseline "$audio/baseline" --candidate "$audio/candidate"

systemd-run --user --wait --pipe \
  -p MemoryMax=4G -p MemorySwapMax=0 -p RuntimeMaxSec=300 \
  python3 "$PWD/scripts/longform_stitch.py" performance \
  --audio "$audio" --baseline /absolute/path/to/gigastt-baseline \
  --candidate /absolute/path/to/gigastt-candidate --output "$audio/performance"

systemd-run --user --wait --pipe \
  -p MemoryMax=4G -p MemorySwapMax=0 -p RuntimeMaxSec=1200 \
  python3 "$PWD/scripts/longform_stitch.py" pause \
  --audio "$audio" --binary /absolute/path/to/gigastt-baseline \
  --baseline "$audio/baseline" --output "$audio/pause"
```

Each run saves raw word timestamps, edit counts, model/binary hashes, per-run
wall time and peak RSS. The comparison fails on a WER regression or changed
short-file output. Inspect changed words as well as aggregate scores. For VAD,
the reported word times are on the original recording; fixed-grid seams are
on the compressed timeline, so do not infer seam locations from word index.

## Limits and further work

This small set cannot establish accuracy on meetings, noisy or spontaneous
speech, other languages or execution providers. Word times are model estimates,
not forced alignments. A lower shift disagreement alone does not prove improved
accuracy, and this stitch does not make offline decoding shift-invariant.

Carrying RNN-T predictor state across windows remains a separate research
direction. It needs alignment/state snapshots at the chosen cut, protection
against replaying overlap labels, and a policy for parallel window decoding.
Simply passing the final state of the preceding window into the next one
would include labels from audio that the next window decodes again.
