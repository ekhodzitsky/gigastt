# Long-form stitching

Files longer than 30 seconds use independent 24-second windows with 2 seconds
of nominal overlap on CPU, CoreML EP and CUDA (30-second windows on ANE). No
window of a long input is shorter than the window length: when a window can
reach the end of the input within the 30-second single-pass ceiling it absorbs
the remainder (a 71.25-second file is decoded as 0–24, 22–46 and 44–71.25 s);
when the remainder is longer than that allows but shorter than a window, the
trailing window is anchored to the end, starting at `length − window` rounded
up to the 40 ms encoder frame grid, and so overlaps its predecessor by more than
2 seconds. The seam between any two windows is the midpoint of their actual
overlap. The encoder and RNN-T predictor start afresh for each window. This
bounds inference memory; it also changes acoustic context compared with a
single pass.

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
offline; download required models before starting it. `prepare` fetches the
Podlodka parts through the Hugging Face datasets-server rows API and verifies
every part and every assembled recording against the manifest hashes. Save a baseline binary
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

# Geometry sweeps: serial cases only, compared against the full baseline.
systemd-run --user --wait --pipe \
  -p MemoryMax=4G -p MemorySwapMax=0 -p RuntimeMaxSec=3600 \
  python3 "$PWD/scripts/longform_stitch.py" run --modes serial \
  --audio "$audio" --binary /absolute/path/to/gigastt-variant --output "$audio/variant"
python3 scripts/longform_stitch.py compare --subset \
  --baseline "$audio/baseline" --candidate "$audio/variant"

# Adjudicate every shifted-vs-unshifted disagreement on the official example
# against the reference; `--windows` adds the per-window copies from `pause`.
python3 scripts/longform_stitch.py residuals \
  --run "$audio/candidate" --windows "$audio/pause" --output "$audio/residuals.json"

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

## Boundary-dependent readings (2026-09-17, follow-up)

The aligned stitch left the disagreement reported in issue #349 in place: on
the official example, 3 seconds of leading silence still changed four
`e2e_rnnt` words and one `ml_ctc` word, while the 22-second control changed
none. Every reported item was adjudicated against the reference, with
original-audio times and the per-window copies decoded before stitching
(`scripts/longform_stitch.py residuals`, window dumps from `pause`):

| Head | Time (s) | 0 s shift | 3 s shift | Reference | Window that produced each copy |
|---|---:|---|---|---|---|
| `rnnt` | 21.56 | темну ✓ | темную ✗ | темну | [0, 24] / [19, 43] |
| `e2e_rnnt` | 21.52 | тёмну ✓ | тёмную ✗ | темну | [0, 24] / [19, 43] |
| `e2e_rnnt` | 23.44 | возглащённые ✗ | возлащённые ✗ | позлащенные | [22, 46] / [19, 43] |
| `e2e_rnnt` | 64.08 | Раскаянье ✗ | раскаянья ✓ | раскаянья | [44, 68] / [63, 71.25] |
| `e2e_rnnt` | 68.36 | сложу ✗ | сложи ✓ | сложи | [66, 71.25] / [63, 71.25] |
| `ml_ctc` | 64.24 | раскаянье ✗ | раскаянья ✓ | раскаянья | [44, 68] / [63, 71.25] |

`rnnt` was not in the report but shows the same темну/темную flip. Each side
of every disagreement is one window's reading; the stitch only selects a copy
and never rewrites one, so merge effects are excluded. Three controls then
separated the remaining causes (all runs sequential in a memory-limited unit,
`crates/gigastt-core/src/inference/engine/tests/predictor_probe.rs`):

- **Predictor reset is not a cause.** The same encoded window was decoded
  cold, cold from the seam frame, and from a predictor state warmed with the
  preceding window's labels before the seam (no replay). In all 14 RNN-T
  cases the target word was identical in every variant; only one copy of
  позлащенные changed between two wrong spellings.
- **Full-context decoding flips the same words.** One encoder pass over the
  whole 71.25-second recording (no windows) reads темну at a 3-second shift
  and темную at 0 for `rnnt` and `e2e_rnnt`, and the reverse for `ml_ctc`;
  `e2e_rnnt` also reads *Слажи* there. Leading silence alone changes the
  encoder's decision on these acoustically ambiguous inflections.
- **Context map.** Sliding a cold 24-second window across each span in 0.5–2 s
  steps (288 decodes, three heads) shows non-monotonic dependence on window
  content. `rnnt` reads темну only when the window contains the recording's
  first ~3.5 seconds; `e2e_rnnt` reads позлащённые in one placement out of
  eighteen; `ml_ctc` never reads темну in a 24-second window. There is no
  amount of left or right context that stabilises these words.

The one structural defect the map did expose is the trailing window: at a 0 s
shift the official example ends with a 5.25-second window ([66, 71.25]), and
`e2e_rnnt` reads сложу only there. Short windows are the regime where the
encoder is measurably worse (see `tests/longform_quality.rs`: one-pass WER
6.25% at 10 s versus 2.21% at 30 s), and any input length produces one.

### Corpus

The frozen manifest gained four continuous conversational recordings:
in-order excerpts of one Podlodka podcast episode each
([bond005/podlodka_speech](https://huggingface.co/datasets/bond005/podlodka_speech),
train split, rows recorded per part with SHA-256), 182–267 seconds, 2 259
normalized reference words, multiple speakers, fed to the engine at their
native 16/44.1/48 kHz stereo layout. Joins between excerpts are not guaranteed
to be continuous audio. Together with the three read recordings the serial
matrix scores 2 714 reference words per head per shift (8 142 per head).
Shifts remain repeated views of one recording.

### Geometry sweep

Seven geometries were decoded on the serial matrix (63 runs each, same
binaries otherwise, sequential in the memory-limited unit). Counts are
substitution + deletion + insertion errors against the reference over 8 142
reference words per head; wall time is the sum over the 63 runs including
model load, relative to the current-main run (613 s):

| Geometry | `rnnt` | `e2e_rnnt` | `ml_ctc` | Windows | Wall |
|---|---:|---:|---:|---:|---:|
| main: 24 s window, 2 s overlap, short trailing window | 602 | 472 | 742 | 528 | — |
| **absorb-or-anchor trailing window, 2 s overlap (shipped)** | 603 | 474 | 735 | 528 | +1% |
| anchored trailing window only (always a full 24 s tail) | 604 | 473 | 730 | 528 | +8% |
| 4 s overlap | 591 | 485 | 732 | 567 | +4% |
| 6 s overlap | 583 | 478 | 725 | 612 | +10% |
| 8 s overlap | 589 | 466 | 721 | 684 | +28% |
| anchored + 4 s overlap | 590 | 488 | 725 | 567 | — |
| anchored + 6 s overlap | 585 | 474 | 711 | 612 | — |
| anchored + 8 s overlap | 590 | 471 | 710 | 684 | +34% |

The trailing-window rules change the window count by at most one per file
and add at most one window's worth of encoder input per file. "Anchored
only" always decodes a full 24 s tail, which on the 71.25-second example turns
a 5.25 s tail into a 24 s one and costs 23–28% wall time on that file (paired
runs: 3.99 → 4.90 s `rnnt`, 4.11 → 5.28 s `e2e_rnnt`, 4.33 → 5.30 s `ml_ctc`),
while costing 4% on the whole matrix. The shipped rule absorbs such a tail
into the previous window instead (44–71.25 s, 27.25 s, within the 30 s
single-pass ceiling) and anchors only remainders between 6 and 24 s.

Per-recording deltas are mixed in sign for both RNN-T heads under every
geometry (for example the official example gets one to two errors *worse* for
`rnnt` at any overlap above 2 s), while `ml_ctc` improves under every variant.
A wider overlap moves every seam, which re-reads every ambiguous word near a
seam; on this corpus that is a small net gain bought with 10–34% more encoder
time. It is not adopted as the default. `CHUNK_OVERLAP_SAMPLES` in
`crates/gigastt-core/src/inference/windows.rs` is the single constant to change
for deployments that prefer the 6–8 s trade.

### Result

The shipped change is the trailing-window rule alone. Against the current-main
baseline under identical settings (120 runs per binary, full provenance in the
[measurement artifact](../benchmark/results/longform_trailing_window_cpu.json)):

| Head / mode | Reference words | Before S/D/I | After S/D/I | Errors |
|---|---:|---|---|---:|
| `rnnt` serial | 8 142 | 441 / 107 / 54 | 442 / 107 / 54 | 602 → 603 |
| `rnnt` VAD | 2 460 | 184 / 40 / 23 | 182 / 40 / 25 | 247 → 247 |
| `rnnt` parallel | 2 460 | 180 / 51 / 24 | 178 / 51 / 24 | 255 → 253 |
| `e2e_rnnt` serial | 8 142 | 318 / 87 / 67 | 317 / 90 / 67 | 472 → 474 |
| `e2e_rnnt` VAD | 2 460 | 163 / 25 / 24 | 159 / 24 / 24 | 212 → 207 |
| `e2e_rnnt` parallel | 2 460 | 153 / 28 / 17 | 151 / 28 / 17 | 198 → 196 |
| `ml_ctc` serial | 8 142 | 581 / 105 / 56 | 574 / 105 / 56 | 742 → 735 |
| `ml_ctc` VAD | 2 460 | 232 / 40 / 30 | 234 / 40 / 31 | 302 → 305 |
| `ml_ctc` parallel | 2 460 | 236 / 43 / 25 | 233 / 43 / 25 | 304 → 301 |

Every change is a re-reading of an ambiguous word inside the last window of a
recording; no head/mode moves by more than 0.5% of its error count, and the
signs are mixed. This is not a WER improvement claim for the RNN-T heads: the
serial matrix gets one (`rnnt`) and two (`e2e_rnnt`) errors worse, VAD and
window-parallel decoding of the same recordings get two to five better. All
three short-file outputs, word metadata included, are byte-identical. Word
times remain monotone and on the original clock in every VAD and parallel
run. On the official example the shipped rule fixes сложи at the 0 s and 22 s
shifts for `e2e_rnnt`; раскаянья at 0 s (`e2e_rnnt`, `ml_ctc`) stays wrong
because the absorbed 44–71.25 s window reads it the way the 44–68 s window
did, and темну at 3 s (`rnnt`, `e2e_rnnt`) and позлащенные (`e2e_rnnt`) stay
wrong under every geometry measured, including the full-context decode. The
"anchored only" variant additionally fixes раскаянья at 0 s, at the cost
above.

Paired resource runs (three per head and binary, alternating order,
71.25-second input, CLI start-up and model load included):

| Head | Median wall seconds, before → after | Peak RSS MiB, before → after |
|---|---:|---:|
| `rnnt` | 4.31 → 4.24 | 353.5 → 392.0 |
| `e2e_rnnt` | 4.22 → 4.23 | 378.6 → 414.3 |
| `ml_ctc` | 4.67 → 4.89 | 469.2 → 508.0 |

The RSS increase is the absorbed 27.25 s last window's encoder activation,
the same size a 27-second file already needs on the single-pass path; it is
bounded by the 30 s ceiling. Over the whole matrix the median RTF moved from
0.0560 to 0.0579 and the peak RSS from 914 to 860 MiB. Every run completed
under the 4 GiB ceiling with swap disabled.

## Limits and further work

This small set cannot establish accuracy on meetings, noisy or spontaneous
speech, other languages or execution providers. Word times are model estimates,
not forced alignments. A lower shift disagreement alone does not prove improved
accuracy, and this stitch does not make offline decoding shift-invariant.

Carrying RNN-T predictor state across windows was measured, not assumed: a
predictor warmed with the preceding window's labels and started at the seam
frame changed none of the disputed words (see above). It remains available as
an experiment (`predictor_probe`), but the evidence does not support it as a
correction. The words that still change with a shift on the official example
(темну for `rnnt` and `e2e_rnnt`, позлащенные for `e2e_rnnt`) change under a
full-context decode as well; they need a language-model prior or a different
acoustic model, not a different window grid. Offline decoding is not
shift-invariant, and no window geometry tried here makes it so.
