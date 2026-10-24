# Real-audio reproduction of seam capitalization

This is a targeted reproduction search for PR #455, not a random quality sample
or a speed benchmark. The original author's recording is still unavailable.

## Search method

The initial frozen corpus comparison exercised no corrections. This search used
its word timings to locate pauses before lowercase words and move those words
just beyond the first 23-second seam. Twenty-five 46-second excerpts were tested
on the baseline with `e2e_rnnt`; each was also decoded as separate 24-second
windows starting at 0 and 22 seconds. `search-inputs.json` records every selection,
including cases without a casing error. Seven candidates had a capital after a
nonterminal preceding word near the seam. This is an enriched diagnostic set:
its hit rate is not an estimate of real-world error frequency.

Because the long-form corpus concatenates dataset excerpts, the final
reproductions use **individual source recordings**, never a join of unrelated
excerpts. Leading/trailing digital silence places the target near the seam;
where needed the source is cropped at its end. No internal pause is inserted,
no word is moved, and sample rate, channels and recorded PCM remain unchanged.
Each prepared file is exactly 46 seconds: two 24-second windows with a two-second
overlap. The `speech` control decodes the same recorded PCM without added silence
in one pass (all selected speech spans are shorter than 30 seconds).

## Verified results

The current PR fixes **three reproducible capitalization errors** on these
individual-source inputs:

| Case | Baseline fragment | Candidate fragment | Result |
|---|---|---|---|
| 005 | `много Вот таких` | `много вот таких` | Corrected |
| 006 | `таких Широты знаний` | `таких широты знаний` | Corrected |
| 011 | `по книжке И никакого` | `по книжке и никакого` | Corrected |
| 008 | Changed decoder hypothesis after isolating the source | Identical to baseline | Control; not counted as a casing fix |
| 013 | `это Недооценённые языки` | `это Недооценённые языки` | Unresolved; older window says `недооценённо.`, so exact-match guard correctly abstains |

The unpadded speech controls use lowercase at the investigated positions.
Three corrections agree with the dataset references; word spelling apart from
initial case, punctuation, timestamps, confidence and other response metadata
are identical between baseline and candidate. Each full response was repeated
three times per binary and was identical on all repetitions (30 full responses;
60 total requests including separate windows and unpadded controls).

Cases 005 and 006 are different seam placements in the **same** source excerpt;
case 011 comes from another excerpt. They are three failure configurations, not
three independent recordings. This demonstrates practical benefit for the
specific first-word-after-pause scenario. It does not establish an aggregate
quality gain or solve all seam capitalization. Case 013 explicitly remains
unfixed. No speed or memory conclusions follow from this targeted run.

`summary.json` records every full-response difference, `response-checksums.json`
records all 60 saved responses, and `provenance.json` identifies the measured
binaries and script. `stitch-fixtures.json` captures four real window pairs
(three fixes and the unmatched control) for a model-free Rust regression test.
The newer timestamps are offset by 550 encoder frames of 0.04 seconds using the
same frame-grid arithmetic as production. Only the first three newer words are
needed in those fixtures; full raw responses remain in the local cache.

## Source and reproduction

Source: [bond005/podlodka_speech](https://huggingface.co/datasets/bond005/podlodka_speech),
train rows 39, 40, 42 and 43, episode 200, “Как учить языки программирования”.
The reference transcriptions use lowercase `вот`, `широты`, `сказывается`, `и`
and `недооценённые` at the investigated positions. Audio is used for measurement
only and is not redistributed. Source file checksums are frozen in the existing
[long-form manifest](../../manifests/longform_stitch.json).

`inputs.json` specifies source SHA-256, frame-exact source ranges, added silence
and prepared WAV SHA-256. Prepare the source corpus with
[`longform_stitch.py`](../../../scripts/longform_stitch.py), then run:

```sh
python3 scripts/reproduce_seam_case.py \
  --baseline /path/to/baseline-gigastt \
  --candidate /path/to/candidate-gigastt \
  --models /path/to/models \
  --parts /path/to/longform-corpus/ruls-parts \
  --output /path/to/fresh-output-directory
```

Baseline runtime: `e3ba93e838712b93551cca6037f52c2a37cf95da`.
Candidate runtime: `968f3e6e957ba568f708e8c4675d4eb0f1187577`.
Both are default-feature CPU release binaries. The script pins the encoder
checksum, records binary checksums and commands, disables external punctuation
and ITN, uses pool size 1 / two encoder threads / serial file windows, and waits
for `/ready`. For each arm it saves the full response three times, the two
individual window responses, and the unpadded speech response. Repeated full
responses must be identical. Cross-arm full responses may differ only in case;
word metadata and all other response fields must match.

Raw audio, full responses and logs remain in the local reproduction cache under
`~/.cache/gigastt/seam-reproduction-20261024/`. Public evidence contains the input
recipe, checksums, focused word differences and regression fixtures.

## Local checks

The captured-window regression passes on the candidate. Workspace unit/bin tests:
1,249 passed, 48 ignored, zero failures. Strict all-target Clippy, formatting,
documentation drift, publication checks and nine benchmark-helper tests pass.
No GitHub workflow was dispatched and the PR was not merged.
