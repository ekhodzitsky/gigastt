# Phrase-loss diagnostic validation

This validates the local diagnostic workflow, not a fix for issue #376.
The private meeting and the author's standalone scorer are unavailable.

The public `podlodka-306` fixture (243.657 seconds) from the frozen long-form
manifest was explicitly converted from stereo 48 kHz to mono PCM16 16 kHz for
this check. The diagnostic itself does not resample. Input hashes, conversion
command, binary source and exact window PCM hashes are in `validation.json`.
The real binary uses the final dependency integration tree now on main; neither
inference code nor the model is modified by this diagnostic PR.

Two independent output directories were used for one and six encoder threads.
Each runs the entire file twice (source graph then verified reused graph), then
three exact windows at 22–46, 44–68 and 66–90 seconds with the reused graph.
All ten recognitions succeeded. Each full-file output has 677 words. Cold/warm
full responses are identical, and the one/six-thread warm full responses also
match completely. For all three windows, the compared interior tokens match
the full-file tokens and no local word start fails the ±0.5-second coverage
check. These are hypothesis comparisons, not ground-truth WER or the reporter's
original timestamp-only lost-run metric.

The summaries contain no audio, transcripts or local paths. Full public raw
responses remain in the local diagnostic cache; their hashes are retained.
Synthetic tests independently cover nonzero-offset byte-exact extraction,
deduplication, invalid/tail inputs, unsupported WAV formats, shifted timestamps
without text changes, changed words without coverage gaps, malformed response
records, timeout/subprocess failure, provider/cache verification, environment
isolation, output reuse rejection and the content of the shareable summary.

Run `python3 scripts/test_phrase_loss_diagnostic.py` from the repository root.
See [the author protocol](../../../docs/phrase-loss-diagnostics.md) for commands.
