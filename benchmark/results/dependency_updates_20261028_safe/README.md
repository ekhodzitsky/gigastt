# Dependency updates with ONNX Runtime held unchanged

## Decision and scope

Integrate #458 (Tokio 1.54.1), #460 (opus-rs 0.1.39), #461
(setup-python v7), #462 (napi 3.15.0 / napi-sys 3.4.1) and #463
(uuid 1.28.0). Preserve the original PR commits as merge parents.
Restore unrelated prost/UniFFI resolver changes from the bot lockfiles.
Synchronize fuzz Opus and update its stale anyhow and yanked der resolutions.

**Exclude #459 (ort rc.15).** The combined candidate needs an environment/API
migration and a macOS compiler-runtime link correction. It then fails CoreML
encoder inference on a macOS 14.8.9 runner. The paired baseline investigation
is tracked in [PR #464](https://github.com/ekhodzitsky/gigastt/pull/464), with
[retained failure diagnostics](https://github.com/ekhodzitsky/gigastt/actions/runs/40678427152).
This integration keeps ort rc.13, native desktop ORT 1.28, and the existing
Android pin. It makes no claim to resolve the separate CoreML investigation.
The earlier candidate's 8–12% Linux latency improvement is **not** a measurement
of this accepted graph.

## Local validation

Base: `29c42e71606e25c539b5303c3aa9da00528b70d1`. `comparison.json` records
binary, lockfile and fixture hashes. The release binary was built from the
final code/lockfile before adding these result artifacts; documentation does
not change its inference implementation.

Compare `transcribe --format json --punctuation off --itn off
--encoder-intra-threads 2` with explicit `rnnt` and `e2e_rnnt`, offline verified
release models, and logging disabled. Both revisions use the same model files.

- 15 bundled WAV fixtures (61.765 seconds total), each decoded with both heads:
  all 30 complete JSON responses are identical to the baseline.
- Three Opus encodings of `golos_00.wav` (16/32/96 kbit/s), both heads:
  all six texts are identical. The 16/32 kbit/s cases change numeric word fields;
  96 kbit/s responses are identical. These are correlated codec smoke cases,
  not independent speech samples or a broad quality benchmark.
- All 36 paired texts match. Raw responses are in `responses.json.gz`.
- Model-backed Node 18.20.8 and Node 22.19.0 smoke checks pass, including
  concurrent recognition, streaming, event-loop and GC/ownership checks.
- All 126 model-backed E2E tests across nine suites pass serially, plus the
  model-free offline FP32 rejection regression test.
- Workspace unit/bin tests, strict all-target Clippy and Cargo Deny pass.
- Workspace and fuzz audits report no vulnerabilities; the workspace retains
  its pre-existing `paste` unmaintained warning. Audit JSON is included.

The obsolete e2e expectation that download automatically quantizes FP32 is
replaced by a model-free test that offline FP32-only installations are rejected
without producing an INT8 model. The rustdoc-only semver check skips native
library downloads; actual native build/runtime checks remain enabled.
All GitHub workflows continue to require manual dispatch.
