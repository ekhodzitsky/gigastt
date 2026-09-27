# Speaker attribution on short public meetings

Whether the current WeSpeaker + polyvoice path over-segments a meeting the
way one downstream lab reported (14 clusters for five speakers). This is not
a new diarizer. The audio is not in the repository.

## What was measured

Corpus: VoxConverse development labels, version 0.3, CC BY 4.0. The RTTM
files are the dataset authors' reference (they describe them as manually
checked to 0.1 s). This run did not repeat that listening audit. The Oxford
archive did not transfer; the KAIST mirror did, until the connection died
after about 1.0 GB. Four clips from the planned short subset were extracted
before that:

| Clip | Reference speakers | Duration | Reference overlap |
|---|---:|---:|---:|
| `syiwe` | 3 | 69.1 s | 0 s |
| `wjhgf` | 5 | 84.7 s | 19.0 s |
| `exymw` | 5 | 90.9 s | 0.2 s |
| `nnqfq` | 5 | 145.3 s | 21.0 s |

Host: AMD Ryzen AI 9 HX 370, 24 logical CPUs. Release binary built from this
tree, `gigastt --offline serve --pool-size 1 --model-variant ml_ctc
--punctuation off --itn off`. Speaker model:
`~/.gigastt/models/wespeaker_resnet34.onnx`. Clustering is polyvoice 0.21.0
defaults: cosine threshold 0.45, `min_cluster_size` 2, `max_speakers` 64.
Diarization is `POST /v1/transcribe?diarization=true&word_timestamps=true`.

The default `rnnt` head emitted no words on `syiwe`, so the client saw no
`speaker` fields. The multilingual CTC head is what produced the words below.
English meetings are outside the Russian head.

No second diarizer ran. FluidAudio is an Apple stack, and pyannote's
pretrained pipeline needs a Hugging Face token this machine does not have.

## Speaker count

The clusterer reports one count. Words can drop a cluster when no word
midpoint falls inside that turn.

| Clip | Reference | Diarization turns / speakers | Distinct `words[].speaker` | Count error (turns) |
|---|---:|---:|---:|---:|
| `syiwe` | 3 | 6 / 4 | 4 | +1 |
| `exymw` | 5 | 9 / 6 | 6 | +1 |
| `nnqfq` | 5 | 11 / 5 | 5 | 0 |
| `wjhgf` | 5 | 13 / 7 | 6 | +2 |

Five speakers did not become 14 clusters. The largest miss on this subset is
two extra speakers.

## Attribution, separate from the cluster count

On a 10 ms grid, keep frames where the reference has exactly one speaker and
that are more than 0.25 s from a reference boundary. A frame with no covering
word is an ASR gap, not a speaker error. A frame with one word is correct
when that word's speaker matches the reference speaker under a one-to-one map
chosen to maximize agreement.

| Clip | Single-speaker frames | No word | Word, wrong speaker | Word, correct |
|---|---:|---:|---:|---:|
| `syiwe` | 6076 | 2700 | 16 | 3360 |
| `exymw` | 8214 | 3492 | 0 | 4722 |
| `nnqfq` | 11534 | 5094 | 3 | 6437 |
| `wjhgf` | 5850 | 2870 | 158 | 2822 |

Where a word exists, the label matches the reference on three of the four
clips (0–16 wrong frames). `wjhgf` is the exception: 158 of 2980 labeled
frames disagree, and one of the seven clusters never lands on a word. Overlap
frames are not in this table (`nnqfq` 852, `wjhgf` 1580). Reference turns
shorter than 1 s: one on `syiwe`, one on `exymw`, ten on `nnqfq`, none on
`wjhgf`. Each short turn that exists overlaps at least one word.

This is not NIST DER. Missed ASR time is not charged as a speaker error, and
overlap is reported beside the score instead of inside it.

## Time and memory

Server `VmRSS` / `VmHWM` from `/proc/<pid>/status`. The client is not included.

| | |
|---|---|
| Idle after warmup | 272,504 KiB (266 MiB) |
| Peak after the four files | 669,120 KiB (654 MiB) |

Request wall clock, ASR plus diarization: `syiwe` 10.7 s (RTF 0.15), `wjhgf`
14.2 s (0.16), `exymw` 19.1 s (0.16), `nnqfq` 23.3 s (0.16). On `exymw` the
speaker encoder loaded after recognition had already started; clustering then
took about 15 s of that 19 s. Later files reuse the loaded encoder.

## What not to change

Do not replace WeSpeaker or the polyvoice clusterer from the 14-versus-5
report. That figure was one private meeting whose reference was inferred from
the transcript, and it does not show up on these four public clips. Do not
retune the threshold on a four-clip English sample either.

Two limits remain. The Russian `rnnt` head produces no words here, so
speakers never reach the client. And this subset is one to two minutes; a
longer meeting was not in the part of the archive that transferred.
