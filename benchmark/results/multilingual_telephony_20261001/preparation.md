# Paired telephone degradation

The ten manifests preserve all 1,000 baseline recordings per language (`ru`, `en`, `kk`, `ky`, `uz`), their order, IDs, human references, and split assignments. Each original has separate G.711 A-law and mu-law conditions: 10,000 prepared recordings derived from 5,000 originals. These are controlled degraded read-speech inputs, not recordings of actual calls.

Preparation uses FFmpeg 7.0.2-static: default high-pass at 300 Hz, default low-pass at 3,400 Hz, mono output, 8,000 Hz resampling, and `pcm_alaw` or `pcm_mulaw` in WAV. The filters are not brick-wall filters. No gain adjustment, denoising, loudness normalization, or clipping normalization is applied. Exact command and tool versions are in `telephony_dataset_provenance.json`.

Every original SHA-256 is checked against the baseline. Every output is independently decoded to PCM16 with libsndfile, verifying WAV format, G.711 subtype, channel count, 8,000 Hz sample rate, frame count, and duration within one output sample of the original. Prepared SHA-256 hashes are stored in each manifest and guarded on resume. Published manifests are atomic and contain no audio. Audio resides in the persistent local cache.

Preparation:

```sh
benchmark/.venv/bin/python scripts/prepare_multilingual_telephony.py \
  --baseline benchmark/results/multilingual_1000_20261001 \
  --output benchmark/results/multilingual_telephony_20261001 \
  --cache ~/.cache/gigastt-multilingual-telephony --workers 4
```

Guard verification passed for unchanged-cache reuse without rewriting, corrupt prepared-file regeneration, and rejection of a changed source file. The fixed source IDs make both codecs and original conditions directly pairable. Baseline reference-number formatting and repeated-sentence limitations continue to apply; no training is performed.
