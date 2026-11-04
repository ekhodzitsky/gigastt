# Resampler delay and finalization audit

No production code was changed for this audit. The deterministic standalone probe imports the current production `audio/resample.rs` directly, uses rubato 5.0.0, and compares its whole-buffer output with cached calls split into 127-source-sample chunks. Results are in [resampling_probe.json](resampling_probe.json). Synthetic inputs are mono 8 kHz, zero except for one amplitude-0.5 impulse at the beginning, midpoint, or last sample; lengths are 160, 8000, and 48001 samples.

## Proven behavior

All nine probes produce `2 * input_len - 4` samples, and cached output is bit-identical to whole-buffer output. For the 8000-sample input, a midpoint impulse at index 4000 peaks at output index 8255, rather than the nominal 8000. A last-sample impulse at index 7999 disappears completely after the existing PCM16 rounding: every output sample is zero. The same tail disappearance occurs at the other two lengths.

The four-sample duration deficit is therefore **not the full signal loss**. Startup filter delay remains in the output while the delayed end of the waveform is never drained. `ResampleTo16k::finish` and `finish_into` drain staged real input but never pump the cached filter with silence or remove its startup delay. This is a file-finalization/alignment defect. Preserving delay between calls is normal for an ongoing streaming resampler; a live stream must not reset or flush the filter after every frame.

The upstream initial sinc index is -255. Its fixed-input output-count formula, at ratio 2 and sinc length 256, gives `floor((N - 257 + 255) / 0.5) = 2N - 4`. The reported `output_delay()` is 256 output samples (16 ms). Measured impulse peaks are delayed by 255 samples; using the documented 256-sample trim leaves a one-sample phase offset. This offset should be recorded, not tuned against recognition results.

Appending 512 zero source samples to the same resampler and selecting output `[256 : 256 + 2N]` restores a full-length output and a strong final impulse (approximately 0.475). This does not recreate the missing tail with arbitrary zeros: it drains the actual filter state. An equivalent experiment using the production decoder must first prove that the extended-file output prefix is bit-identical to the original output.

## Upstream API caution

Rubato documents `output_delay()` as event delay and recommends complete-clip processing with startup-delay removal. However, the exact pinned 5.0.0 `process_all_into_buffer` implementation does not remove startup delay when the clip fits in one configured chunk. The probe confirms this: setting the chunk to clip length produces a nominal-length output with the same delayed midpoint and near-zero final impulse. Consequently, replacing the current call with `process_all` without independent tests is insufficient. The implementation also copies only `frames_to_trim` frames when trimming in its multi-chunk loop, which warrants separate validation before adopting that helper.

Primary source: [rubato 5.0.0 Resampler API](https://docs.rs/rubato/5.0.0/rubato/trait.Resampler.html), and locally resolved source files `rubato-5.0.0/src/asynchro.rs`, `asynchro_sinc.rs`, and `lib.rs`. Relevant points are `init_last_index`, `calculate_output_size`, `output_delay`, and `process_all_into_buffer`.

## Scope of inference

This proves audio alignment and tail loss, **not** that these defects account for the observed word deletions. Four samples alone may change a feature-frame count at a hop boundary; removing filter delay also shifts the entire waveform relative to the STFT grid. The production multilingual path receives the Rust `FeatureExtractor` log-mel output, using `center=false`: `run_inference` copies those features into the encoder tensor with shape `[1, 64, num_frames]`. Direct inspection of the installed INT8 graph confirms inputs `features: [batch_size, 64, seq_len]` and `feature_lengths: [batch_size]`; this is not a raw-waveform frontend. Length and time shifts change sampled STFT frames and downstream encoder behavior. A nonlocal encoder can react beyond the few affected edge frames. Recognition comparisons must distinguish length-only padding, time shift, proper filter drain, filter differences, and INT8 precision.

A minimal file-only implementation can leave `resample_with_cache` unchanged: have `ResampleTo16k` count original source frames, discard startup delay across its first emitted chunks, and drain the cached filter once at end of input, keeping exactly the remaining target duration. Count output already returned by `drain_ready_into` so streaming file readers work too; finalization must be idempotent. Do not flush or recreate the shared live-stream resampler for every frame.

Before any production change, tests should fail for final-impulse survival, complete duration, impulse alignment within the documented sample tolerance, staged-versus-flat equality, repeated finalization, and empty/short clips. Recognition acceptance additionally requires the fixed diagnostic corpus and subsequent regression benchmark, including native 16 kHz passthrough.

## Reproduction

The published probe source is `scripts/resampling_length_probe.rs`; its runner copies that source and the current production resampler into a persistent cache crate, keeping the repository Cargo configuration unchanged:

```sh
python3 scripts/run_resampling_length_probe.py > /tmp/resampling_probe.json
```

The runner uses only existing cached dependencies (`cargo --offline`); its default cache is `~/.cache/gigastt-small-diagnosis/resampling-probe-repro/`. A later production change can alter the output, so compare it with the recorded JSON and the repository revision under investigation. This synthetic audit does not substitute for the repository's required unit/clippy checks when implementing a fix.
