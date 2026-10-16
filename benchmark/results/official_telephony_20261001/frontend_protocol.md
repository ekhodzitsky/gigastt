# Fixed matched frontend control protocol

Protocol specified before running the diagnostic:

- Select the first **one** entry of each original ru/en/kk/ky/uz manifest.
- Match the exact same ID in original, A-law, and mu-law manifests: 15 inputs.
- Do not replace cases, tune preprocessing, select by errors, or train models.
- Dump Rust decoded PCM and mel arrays using the existing `inspect_audio`
  example, rebuilt against the unchanged current source if necessary.
- Run the official FFmpeg PCM16 loader and original checkpoint frontend.
- Evaluate three chains: official PCM + official mel; Rust PCM + official mel;
  Rust PCM + Rust mel. The middle chain isolates resampling/input changes from
  frontend changes. Compare PCM lengths and aligned sample differences; compare
  mel shape and aligned differences, clearly reporting unequal shapes.
- For each chain run the unchanged original small source checkpoint FP32,
  cached FP32 ONNX, and production INT8 ONNX on **exactly identical feature
  arrays**. Save arrays/hashes and compare frame argmax and logit differences.
- Total: 15 overlapping inputs, 45 feature arrays, 135 encoder executions.
  This is a small paired diagnostic, not a new independent evaluation corpus.
- Score with the same Unicode normalization and micro WER; aggregate by
  condition and preserve per-input counts. Do not interpret 1 sample/language
  as a language-quality estimate. No throughput claims: other runs overlap.
- Two inference threads, CPU; all checkpoints already cached. No download,
  training, source mutation, model deployment, or Rust inference change.

Report source/FP32 matching separately from INT8 effects. A discrepancy between
two chains is evidence for the changed stage on these inputs only; it does not
prove that every telephone error has the same cause.
