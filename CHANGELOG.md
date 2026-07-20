# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.12.0] - 2026-07-20

### Added

- **Streaming `final` segments now carry ITN + punctuation/casing restoration.** Both the
  WebSocket and SSE streaming paths post-process each segment at its finalization boundary
  (endpoint flush / stop flush) with the same policy as file transcription: inverse text
  normalization first, then punctuation/casing restoration when a punctuation model is
  attached (`--punctuation` / `--itn`, default `auto` = on for the bare `rnnt` head, off
  for `e2e_rnnt`, which is already punctuated by the model). `partial` messages stay the
  raw decoder hypothesis, and `words[]` payloads keep the raw output — only the joined
  `text` is rewritten, exactly like the file path. Live-dictation clients no longer have
  to choose between streaming previews and readable transcripts.
  - Additive WS `configure` fields `punctuation` / `itn` override the server policy per
    session (sent before the first audio frame; an omitted field keeps the server
    default; `punctuation: true` on a server without a punctuation model is a graceful
    no-op). The overrides survive mid-session state recreation (diarization
    reconfiguration, post-panic session reset). SSE follows the server boot policy —
    per-request parameters for `/v1/transcribe/stream` are not part of this release.
  - Measured cost at the finalization boundary: `restore` on 1–10-word segments is
    p95 ≈ 0.5–1.0 ms (debug build, Apple Silicon) — roughly two orders of magnitude under
    a 100 ms budget, so enrichment runs unconditionally regardless of segment length.
  - The `ClientMessage::Configure` struct variant is now `#[non_exhaustive]` (match it
    with a `..` rest pattern): the wire protocol grows by additive optional fields, and
    the marking lets future ones ship as minor releases.

## [2.11.3] - 2026-07-19

### Fixed

- **Offline file transcription no longer diarizes unless asked.** After the 2.11.2 WeSpeaker
  fix made the speaker encoder actually work, every `/v1/transcribe` (and SSE stream) began
  attaching speaker labels whenever the diarization model was present — ignoring the opt-in
  `?diarization=true` parameter and, in particular, labelling the `channels=split` dual-mono
  fallback. Offline speaker diarization is now gated on the request: a plain transcript carries
  no `speaker` fields, and only `?diarization=true` runs the speaker pass. Streaming (WebSocket)
  was already correctly gated. Adds the additive `Engine::transcribe_bytes_shared_with_overrides_diarized`
  entry point.

## [2.11.2] - 2026-07-19

### Fixed

- **Speaker diarization now feeds WeSpeaker the expected fbank features.** The previous legacy
  extractor passed rank-2 raw waveform tensors to a rank-3 feature model, so offline and streaming
  diarization logged `Got: 2 Expected: 3` and returned transcripts without speaker labels.

## [2.11.1] - 2026-07-19

### Fixed

- **`--model-variant` is now honored when a model directory holds more than one head.**
  Previously the engine re-detected the head from the files on disk (with `rnnt` precedence)
  at load time, so `--model-variant e2e_rnnt` (or any non-default head) was silently ignored
  whenever a directory contained more than one head's files — the highest-precedence head
  loaded instead. The resolved variant is now threaded through to the engine loader, so the
  requested head is the one that loads (and the load fails with a clear error if that head's
  files are absent). With no `--model-variant`, on-disk auto-detection is unchanged. Adds the
  additive `Engine::load_with_pools_threads_variant` entry point.

## [2.11.0] - 2026-07-17

### Added

- **GigaAM Multilingual charwise-CTC recognition heads (`--model-variant ml_ctc` /
  `ml_ctc_large`).** Two opt-in heads alongside `rnnt` and `e2e_rnnt`, built on Salute's
  GigaAM Multilingual encoders (220M and 600M, released June 2026; MIT). Trained across 70+
  languages with best-in-class WER on Russian, Kazakh, Kyrgyz, and Uzbek (moderate on
  English); the recognition vocabulary is a 71-class multilingual character set (Latin +
  Cyrillic + Kazakh/Turkic letters). Output is bare lowercase (no punctuation / casing /
  ITN), so pair it with `--punctuation` / `--itn` for readable Russian text, the same as the
  `rnnt` head.
  - `gigastt download --model-variant ml_ctc` (220M, ~225 MB) or `ml_ctc_large` (600M,
    ~592 MB) fetches the pre-quantized INT8 encoder and vocabulary directly from
    `istupakov/gigaam-multilingual-ctc-onnx` / `-large-ctc-onnx` on HuggingFace — no FP32
    download and no on-device quantization (the upstream INT8 encoder is used as-is). Both
    files are SHA-256-verified. The two heads share a byte-identical vocabulary.
  - The heads share the existing 64-mel / FFT 320 / hop 160 frontend, so no audio-pipeline
    changes are needed; they are encoder-only (a single ONNX with a CTC projection, greedy
    CTC decoding — no prediction network / joiner).
  - `serve`, `download`, and `transcribe` accept `ml_ctc` / `ml_ctc_large` (with `-` aliases)
    via `--model-variant` (or `GIGASTT_MODEL_VARIANT`); the engine also auto-detects the head
    from the encoder present on disk. REST `/v1/models` reports them as
    `gigaam-multilingual-ctc` / `gigaam-multilingual-large-ctc`.
- **Benchmark WER for the Multilingual CTC heads across all five supported languages**
  (`docs/benchmarks.md`). Clean-read WER for the 600M head: Russian 4.44% (`golos_crowd_1k`),
  English 4.63% (LibriSpeech `test-clean`), Kazakh 6.52% / Kyrgyz 7.39% / Uzbek 9.21% (FLEURS,
  digit-free) — 4.4–9.2% across languages; the 220M head is 6.15–11.96%. The Russian-only
  `rnnt` / `e2e_rnnt` heads are Cyrillic-only (100% WER on English). Adds the `gigastt-ml-ctc`
  / `gigastt-ml-ctc-large` benchmark runners and `scripts/prepare_librispeech.py`,
  `scripts/prepare_fleurs.py`, `scripts/wer_unicode.py` (a Unicode-complete WER with the
  number / apostrophe normalization these languages need). Russian phone / YouTube WER are not
  measured (no local reference audio).

### Fixed

- **The SSE streaming endpoint (`POST /v1/transcribe/stream`) now reserves a batch-pool
  slot before decoding the upload**, capping the number of concurrent audio decodes at the
  pool size. Previously the handler expanded each compressed upload to f32 PCM before
  checking out a pool slot, so a burst of large uploads (up to the body-size limit) could
  each inflate into a multi-hundred-megabyte buffer simultaneously and exhaust memory. The
  endpoint now returns `503` + `Retry-After` under pool saturation, matching the synchronous
  `/v1/transcribe` handler.

## [2.10.0] - 2026-07-15

### Added

- **Asynchronous Job API for long-file and batch transcription (`/v1/jobs`).** Disabled by
  default; enable with `--enable-jobs` or `GIGASTT_ENABLE_JOBS=1`. When disabled the
  `/v1/jobs` routes are not registered and return 404.
  - `POST /v1/jobs` accepts the same body and query parameters as `/v1/transcribe`
    (`format`, `word_timestamps`, `segments`, `channels`, `punctuation`, `itn`, `vad`, etc.)
    and returns `202 {"job_id","status":"queued","created_at"}`.
  - `GET /v1/jobs/{id}` returns the job lifecycle (`queued|processing|done|failed|cancelled`),
    an estimate of processed audio seconds, and a `percent` derived from `processed/total`.
    Errors are sanitized and never leak paths or model internals.
  - `GET /v1/jobs/{id}/result` returns the finished transcript in the format requested at
    submission; `409 job_not_finished` while running, `404` for unknown/expired jobs.
  - `DELETE /v1/jobs/{id}` cancels queued or processing jobs and broadcasts a `cancelled`
    event to any SSE listeners.
  - `GET /v1/jobs/{id}/events` is an SSE stream of `progress`, `done`, `failed`, and
    `cancelled` events; the stream closes automatically after a terminal event.
  - The in-memory FIFO queue respects `--batch-pool-size` (default 0, clamped to at least 1
    job worker), limits concurrent batch jobs so they cannot starve WebSocket / synchronous
    REST traffic, and retries up to `--jobs-retry` (default 3) on `inference_timeout` or panic.
    Finished/failed/cancelled jobs are evicted after `--jobs-ttl-secs` (default 3600) or when
    `--jobs-max` (default 100) is reached, at which point `POST /v1/jobs` returns
    `429 queue_full` with `Retry-After`.
  - Graceful shutdown cancels all queued jobs; any in-flight job is allowed to finish within
    the existing drain window and its triplet is returned to the pool.

## [2.9.0] - 2026-07-14

### Changed

- **`?segments=true` groups words at natural boundaries and carries speaker labels.**
  Segment grouping on `POST /v1/transcribe` moves from the cue-sized windows shared with
  the `srt` / `vtt` exports to natural boundaries: a new segment starts on an inter-word
  pause (> 0.9 s), after sentence-ending punctuation (`.` `!` `?`), on a speaker change,
  or at a 30 s duration cap. When diarization or the stereo channel-speaker mode supplied
  labels, each segment gains an additive `speaker` field; `format=md&segments=true`
  section headers follow the same boundaries. The `srt` / `vtt` caption exports keep
  their cue-sized grouping unchanged, and the default response without `segments=true`
  stays byte-for-byte identical.

## [2.8.0] - 2026-07-10

### Added

- **Per-request recognition-knob overrides on `POST /v1/transcribe`.** Three additive,
  optional query params let a single running server vary the post-processing knobs per
  request instead of only at boot: `?punctuation=true|false` (punctuation / casing
  restoration pass), `?itn=true|false` (inverse text normalization, number-words →
  digits), and `?vad=true|false` (VAD silence-skipping). An absent param falls back to
  the server's boot policy, so the default response is byte-for-byte unchanged. A knob
  can only be turned *on* per-request when its backing resource is loaded: `?vad=true`
  on a server started without `--vad` returns `409 vad_not_loaded`, and
  `?punctuation=true` with no punctuation model returns `409 punctuation_not_available`
  — both validated before any pool checkout (fail-fast). Turning a knob *off*, and ITN
  in either direction, is always accepted. A forward-compatibility `?variant=<head>`
  guard returns `409 variant_not_loaded` when the requested head differs from the loaded
  one (a single-model server can't switch heads; matching or absent proceeds). Scope:
  `POST /v1/transcribe` only — the SSE `/v1/transcribe/stream` and WebSocket paths do not
  run the punctuation / ITN passes, so the punctuation / ITN knobs don't apply there.
  Deferred: per-request hotword biasing and multi-model variant switching.

## [2.7.0] - 2026-07-10

### Added

- **Segment-level output on `POST /v1/transcribe` (`?segments=true`).** The default JSON
  response gains an additive `segments` array — `[{start, end, text, words:[…]}]` — that
  groups words into cue-sized segments, sharing the exact boundaries used by the `srt` /
  `vtt` exports (one grouping source), while the top-level `text` / `words` / `duration`
  stay unchanged. Combined with `format=md`, `segments=true` switches Markdown to
  `### [mm:ss]` (widening to `[hh:mm:ss]` past an hour) section headers per segment instead
  of the flat `# Transcript` blob. Fully opt-in: the default response and plain `format=md`
  are byte-unchanged, and `txt` / `srt` / `vtt` ignore the flag (already flat / cue-based).
- **Model hot-reload without a restart.** A new loopback-only `POST /v1/admin/reload`
  rebuilds the inference engine from the exact boot recipe (model dir, pool sizes,
  encoder threads, and the punctuation / ITN / VAD / hotword chain) and atomically
  swaps it in. The new engine is warmed before the swap so the first request after a
  reload pays no cold-start cost, and a build failure leaves the currently-serving
  engine untouched (the server is never left without a model). In-flight requests keep
  the engine they started on and finish against its pool. The endpoint is restricted to
  loopback callers by an explicit peer-IP check that holds even under `--bind-all` /
  `--cors-allow-any`; a second concurrent reload is rejected with `409`. The endpoint
  is also exempt from the per-IP rate limiter — an operator triggering a reload is
  gated by the 409/403 logic, not the token bucket. Returns
  `200 {"reloaded":true,"variant":…,"encoder":"int8"|"fp32"}` on success.

## [2.6.0] - 2026-07-09

### Added

- Per-request RTF `info!` log on every completed file transcription (audio seconds, wall seconds, RTF, encoder label `int8/cpu` etc.); covers CLI `transcribe`, REST `/v1/transcribe`, and SSE — not the streaming WebSocket path.
- `warn!` when the INT8 quantized encoder is missing and the engine falls back to the FP32 encoder on the default ORT path; names the one-line fix (`gigastt download` or `gigastt quantize`). Suppressed for candle and ANE builds, which have their own model formats.

### Changed

- **CPU encoder now uses all cores by default (`--encoder-intra-threads`).** When the
  flag / `GIGASTT_ENCODER_INTRA_THREADS` env is left unset, the encoder intra-op thread
  count now defaults to the machine's logical CPU count divided across the
  concurrently-running pool triplets (`serve`: `pool_size + batch_pool_size`; offline
  `transcribe`: a single triplet), instead of the previous fixed `1`. A default install
  on an N-core box now decodes across the available cores rather than pinning one. An
  explicit value (flag or env, including `1`) is still honoured verbatim, and the
  resolved count continues to be auto-clamped so `pool_size * threads` can't oversubscribe
  the logical CPUs. No effect on the CoreML / CUDA / ANE builds, where the accelerator owns
  scheduling. The Docker `CMD` needs no override to benefit.

### Fixed

- **Integer-overflow panic on a crafted APEv2 tag header.** A 36-byte APEv2 tag
  (APE tags can ride on MP3 uploads) with an unbounded `size` field made
  `symphonia-metadata`'s `size + 32` overflow and panic with "attempt to add
  with overflow", reachable from the audio-upload path. A vendored one-line guard
  (`saturating_add`) in `symphonia-metadata` turns the crafted header into a clean
  decode error instead of a panic, and the SSE `/v1/transcribe/stream` decode is
  now wrapped in `catch_unwind` (matching the REST handler) so any future
  decode-path panic is absorbed as a 422 rather than surfacing as a 500.

## [2.5.0] - 2026-06-24

### Added

- **Optional native ANE (Core ML) encoder backend (`--features ane`, macOS ARM64).**
  Runs the GigaAM v3 `rnnt` **encoder** on the Apple **Neural Engine** via per-bucket
  fixed-shape Core ML `.mlpackage`s; the decoder/joiner stay on the `ort` CPU path.
  Strictly opt-in and additive — the default `ort` build is unchanged. File-mode:
  each encoder window is padded up to the nearest fixed bucket (ladder `512/768/1536/3000`
  mel frames), run FP16 on the ANE, and trimmed back; windows below the 50% fill
  floor — including every streaming window — transparently fall back to the `ort`
  encoder (no crash, no ANE benefit). Warm end-to-end ≈ 10× over the CPU build
  (encoder ~15× on the Neural Engine; the RNN-T greedy decode and feature extraction
  stay on the CPU and become the larger share once the encoder is offloaded). A
  compiled-model disk cache cuts the ~20 s first-load to ~0.1 s on later starts.
  WER vs `ort` ≈ 1.11% (one borderline FP16-pad-up token flip on a 15-clip Golos
  set, else byte-identical).
  The smallest (512) bucket serves typical 3–5 s clips at higher fill, cutting the
  pad-up waste — and the encoder latency — versus routing them up to the 768 bucket.
  Distinct from `--features coreml` (the ort CoreML EP); mutually exclusive with
  `coreml`/`cuda`/`nnapi`/`candle`; `rnnt` head only (`e2e_rnnt` falls back to `ort`).
  Bucket packages download via `gigastt download --ane` (published at the
  `ane-v3-2026-06-24` release) or convert locally via `scripts/convert_gigaam_ane.py`.
  See [`docs/ane-backend.md`](docs/ane-backend.md).

## [2.4.0] - 2026-06-23

### Added

- **Optional Candle / Metal inference backend (`--features candle`, experimental).**
  A pure-Rust alternative to the default ONNX Runtime path that runs the GigaAM v3
  `rnnt` encoder, decoder (LSTM prediction network), and joiner on the Apple Silicon
  **Metal GPU** (Candle's CPU backend elsewhere). Strictly opt-in and additive — the
  default `ort` build is unchanged and remains the default. Transcription is
  **byte-for-byte identical to the `ort` backend** on the benchmark fixtures
  (per-stage numeric parity ~1e-6; whole-file and streaming transcripts identical).
  Weights convert from the local ONNX models via `scripts/convert_gigaam_candle.py`
  (no PyTorch or extra download). Mutually exclusive with `coreml`/`cuda`/`nnapi`;
  `rnnt` head only (e2e_rnnt falls back to `ort`). See
  [`docs/candle-backend.md`](docs/candle-backend.md).
- **Prebuilt Docker images on GitHub Container Registry.** Each tagged release
  now publishes `ghcr.io/ekhodzitsky/gigastt:{X.Y.Z,latest}` (CPU, multi-arch
  linux/amd64 + linux/arm64) and `:{X.Y.Z-cuda,cuda}` (CUDA, linux/amd64), so
  consumers can `docker pull` a versioned image instead of building from
  `cargo install` in their own Dockerfile. The publish job is independent of the
  binary GitHub release (a registry hiccup never blocks the tarballs and vice
  versa). See [`docs/deployment.md`](docs/deployment.md).
- **Non-blocking first-run boot.** The TCP listener now binds *before* the
  first-run model download + INT8 quantization (which can take minutes). During
  that window a minimal bootstrap responder answers `GET /health` with `200`
  (`model: "loading"`) and `GET /ready` with `503 {"reason":"initializing"}`,
  then the *same* bound socket is handed to the full server with no rebind — so
  `curl --fail /health` and Docker `HEALTHCHECK` no longer see connection-refused
  (indistinguishable from a crash) while the model loads. A shutdown signal
  during loading exits cleanly without serving. The heavy load work runs on a
  blocking thread so the bootstrap responder stays responsive.

### Changed

- **`/health`, `/v1/models`, and the WebSocket `Ready` frame now report the head
  actually loaded.** They previously hardcoded `model: "gigaam-v3-e2e-rnnt"`
  regardless of which head was running, so a default `rnnt` server misreported
  itself as `e2e_rnnt`. `model`/`id` are now derived from the loaded variant
  (`gigaam-v3-rnnt` vs `gigaam-v3-e2e-rnnt`), and `/health` + `/v1/models` gain
  additive `variant` (`rnnt`/`e2e_rnnt`), `punctuation`, and `itn` fields so a
  client can confirm the effective output style from a single probe. New fields
  are additive (existing clients unaffected); OpenAPI updated to match.

## [2.3.0] - 2026-06-20

This release makes the lower-WER `rnnt` head the default and lands the INT8
integer-compute speed fix, voice activity detection, contextual hotword biasing,
punctuation/ITN restoration for the `rnnt` head, export formats, dual-WER benchmark
scoring, and a ~2× lower idle footprint. Re-measured through the cross-engine harness
(same manifests/normalization as the competitors), the `rnnt` head is **the most
accurate engine on 3 of 4 Russian domains** — far-field **4.08%**, phone **18.50%**,
YouTube **10.91%** — and a **statistical tie with Vosk 0.54 on clean read** (**3.55%**
vs 2.97%, CIs overlap), down from the old `e2e` default's 8.60%. **RTF ~0.10**,
**790 MB** RSS at the default `--pool-size 2`, INT8 encoder 844 MB → 215 MB (3.9×).
See [`docs/benchmarks.md`](docs/benchmarks.md).

### Changed

- **Greedy decode loop no longer allocates per token.** `run_decoder` / the joiner
  step now reuse buffers (`copy_from_slice`) instead of allocating fresh `Vec`s per
  non-blank token, removing millions of per-token heap allocations on long clips. The
  blank-run decoder-output cache is preserved and decode output is bit-identical. (A
  joiner encoder-projection precompute and ort IoBinding are possible future work.)
- **INT8 encoder now uses dynamic integer compute (`MatMulInteger`/`ConvInteger`).**
  `gigastt-core::quantize` previously emitted weight-only `DequantizeLinear`, which ONNX
  Runtime constant-folds back to FP32 at load — so the "INT8" encoder ran as FP32 (no
  speedup, full FP32 memory). It now emits `DynamicQuantizeLinear` on activations plus
  `MatMulInteger`/`ConvInteger` integer kernels (matching the upstream pre-quantized
  format), so the CPU EP runs true INT8 compute. Measured on an M-series CPU: encoder
  RTF drops from ~1.3 (slower than real time) to well below 0.1, a ~13–30× single-stream
  speedup, with the transcript unchanged. The `gigastt quantize` CLI and the
  auto-quantize-on-first-run behavior are unchanged.
- **Lower idle memory footprint: default `--pool-size` is now 2 (was 4), plus an
  automatic RAM-aware pool cap.** Each pooled session triplet deserializes its
  own copy of the encoder weights — ORT's shared `PrepackedWeights` container
  shares prepacked kernel buffers, not the raw initializer tensors, and this ORT
  version exposes no stable cross-session initializer-sharing path (the
  in-memory `use_ort_model_bytes_for_initializers` route was measured *worse*,
  not better, for our self-contained INT8 graph). A pooled INT8 triplet costs
  ~0.4 GB resident, so the default server footprint drops from ~2.0 GB to
  ~1.0 GB. The server now also clamps `--pool-size` at load so the pooled
  encoders stay under half of total system RAM, logging a warning when it has to
  reduce concurrency — this prevents a large `--pool-size` from OOM-ing a small
  host. The cap never *raises* the requested size and is a no-op on hosts with
  ample memory. Raise `--pool-size` for higher concurrency when RAM allows.

### Added

- **Voice activity detection (`--vad`, opt-in, off by default).** Optional Silero
  v5 VAD (MIT, ~2 MB ONNX) loaded through the existing `ort` runtime — no extra
  dependency. Enables two things: file transcription **skips silence** (decodes
  only detected speech regions, then remaps word timestamps back to the original
  timeline) for a speedup proportional to the silence fraction; and WebSocket
  streaming gains **VAD endpointing** (finalizes a segment on detected trailing
  silence, augmenting the decoder's blank-run heuristic). The model auto-downloads
  to `--vad-model-dir` (default `~/.gigastt/models/vad/`) with SHA-256
  verification on first use. Tunables: `--vad-threshold` (default 0.5),
  `--vad-min-silence-ms` (default 500); env `GIGASTT_VAD`, `GIGASTT_VAD_THRESHOLD`,
  `GIGASTT_VAD_MIN_SILENCE_MS`, `GIGASTT_VAD_MODEL_DIR`. VAD is strictly
  non-blocking: a missing model or inference error logs a warning and proceeds
  without VAD. **Default OFF: with no `--vad`, decoding is byte-for-byte
  unchanged** (full buffer decoded; streaming endpointing unchanged).
- **Verbatim ("naive") WER reported alongside normalized WER in the benchmark.**
  Both the Python (`benchmark.py`) and Rust (`crates/gigastt/tests/benchmark.rs`)
  harnesses now compute a second WER per sample with verbatim rules only
  (lowercase + `ё`→`е` + strip non-word characters, but no words-to-digits ITN,
  digit-group merging, or anglicism mapping) and surface `naive_wer`, its 95% CI,
  and `naive_delta = wer - naive_wer`. A negative delta means writing convention
  (number style, punctuation, transliteration) — not acoustics — produced the WER
  gap, which separates genuine recognition error from formatting. The Python
  results table gains `naive %` / `Δ pp` columns. The Rust regression gate still
  decides pass/fail on the normalized WER only; the verbatim numbers are
  reported, never gated.
- **Configurable encoder threads (`--encoder-intra-threads`).** The CPU encoder runs
  single-threaded by default; on weak CPUs or long single-file jobs it can now take more
  intra-op threads (env `GIGASTT_ENCODER_INTRA_THREADS`, default 1 — unchanged), clamped so
  `pool_size × threads` stays within the logical CPU count. Decoder/joiner and the
  CoreML/CUDA paths are untouched.
- **Chunked long-form file decoding (bounded peak memory).** File transcription now
  splits long audio into overlapping ~24 s windows, decodes each independently, and
  stitches the words (overlap de-dup, monotonic timestamps) — peak encoder activation
  memory is O(chunk) instead of O(file), removing the OOM risk on long inputs. Short
  files keep the single-pass path unchanged; the hard duration cap is raised accordingly.
- **Contextual hotword biasing (`--hotwords-file` / `--hotwords-boost`).** Optional
  token-level shallow-fusion biasing inside the existing greedy RNN-T loop (no beam
  search): a trie over hotword phrases — each tokenized to the active vocabulary —
  boosts the joiner logits of tokens that extend an active hotword prefix, so brands,
  names, and domain terms are recognized more reliably. Works for both heads (char and
  BPE vocabularies). A curated Russian brand/acronym lexicon ships as the default pack.
  Default OFF: with no hotwords configured, decoding is byte-for-byte unchanged. Env
  `GIGASTT_HOTWORDS_FILE` / `GIGASTT_HOTWORDS_BOOST`.
- **Inverse text normalization for the `rnnt` head (`--itn`).** An optional
  post-processing pass converts spelled-out Russian numbers into digits
  (e.g. `шестьдесят тысяч` → `60000`, `две тысячи двадцать` → `2020`). It runs
  **before** the punctuation pass, so the combined pipeline turns
  `шестьдесят тысяч тенге сколько будет стоить` into
  `60 000 тенге, сколько будет стоить?`. `--itn <auto|on|off>`
  (env `GIGASTT_ITN`, default `auto` = on for `rnnt`, off for `e2e_rnnt` which
  already digitizes numbers). The number-word table and the words→digits state
  machine are a verbatim port of the WER benchmark normalizer, so online and
  offline number handling stay symmetric.
- **Punctuation model auto-download.** The RUPunct ONNX artifact is now published
  at `ekhodzitsky/rupunct-small-onnx` (public, MIT) and downloads automatically
  into `--punct-model-dir` (with SHA-256 verification + atomic rename) the first
  time the punctuation pass is enabled, so `--punctuation on` works out of the
  box. A download failure is logged and swallowed — transcription is never
  blocked.
- **Punctuation + capitalization restoration for the `rnnt` head
  (`--punctuation`).** An optional post-processing pass turns the plain `rnnt`
  head's bare lowercase output into properly cased, punctuated Russian
  (e.g. `шестьдесят тысяч тенге сколько будет стоить` →
  `Шестьдесят тысяч тенге, сколько будет стоить?`). Backed by an INT8 ONNX export
  of `RUPunct/RUPunct_small` (MIT, ~29 MB) run via ONNX Runtime with the
  `tokenizers` crate for WordPiece — no Python at runtime. `--punctuation
  <auto|on|off>` (env `GIGASTT_PUNCTUATION`, default `auto` = on for `rnnt`, off
  for `e2e_rnnt` which already punctuates) and `--punct-model-dir`
  (env `GIGASTT_PUNCT_MODEL_DIR`, default `~/.gigastt/models/punct/`). The pass is
  fully optional: if the model is absent it logs once and returns the text
  unchanged.
- **Selectable recognition head (`--model-variant rnnt|e2e_rnnt`).** gigastt can now
  run either GigaAM v3 head. The plain `rnnt` head is the default for fresh installs:
  it scores markedly lower WER on bare normalized text (measured ~3.3% vs ~9.6% for
  `e2e_rnnt` on a golos_crowd_1k subset) but emits lowercase text without punctuation.
  `e2e_rnnt` keeps native punctuation, casing, and inverse text normalization. The
  engine auto-detects the installed variant from the files on disk (real filenames per
  variant; the `rnnt` vocab is `v3_vocab.txt`), and the downloader fetches the matching
  set with per-file SHA-256 verification. Existing installs are respected: running
  `serve`/`transcribe` without `--model-variant` uses whatever model is already present
  and never silently re-downloads; an explicit `--model-variant` switches (and never
  mixes variants). Env var `GIGASTT_MODEL_VARIANT`.
- **Dedicated batch pool (`--batch-pool-size`).** Both REST file-transcription
  paths — `/v1/transcribe` and the SSE `/v1/transcribe/stream` (which also
  transcribes a whole upload, holding its triplet for the file's duration) —
  can now draw from a pool of triplets split off from `--pool-size` (env
  `GIGASTT_BATCH_POOL_SIZE`, default `0` = off; clamped to leave at least one
  interactive triplet), so a long batch job no longer starves real-time
  WebSocket streaming, which keeps the interactive pool. New
  `Engine::load_with_pools` / `Engine::pool_for_batch`. When enabled, the batch
  pool exports its own `gigastt_batch_pool_available` / `gigastt_batch_pool_waiters`
  gauges so its saturation is observable separately from the interactive pool.
- **Per-request inference timeout (`--inference-timeout-secs`).** A
  `spawn_blocking` ONNX run that exceeds the timeout (env
  `GIGASTT_INFERENCE_TIMEOUT_SECS`, default 600, `0` disables) now returns a
  typed `inference_timeout` to the client (REST: HTTP 504; WebSocket: error +
  close) instead of hanging the request, with a `gigastt_inference_timeouts_total`
  counter. The default of 600 s comfortably covers a worst-case in-spec
  10-minute (600 s audio) file on the CPU EP, so the advertised upload ceiling
  still works out of the box. The SSE streaming path is not wrapped — each 1 s
  chunk is a small bounded unit and shutdown is handled by a per-chunk
  cancellation check. Note: `spawn_blocking` can't be
  cancelled, so a hung run's triplet returns to the pool only when it
  eventually completes (or at restart) — the timeout unblocks the *client*, not
  the stuck slot.
- **Degraded pool boot (`--pool-min-size`).** When some session triplets fail
  to load (e.g. low memory) the server can now start on a partial pool instead
  of failing outright. `--pool-min-size` (env `GIGASTT_POOL_MIN_SIZE`, default
  1, clamped to `1..=pool_size`) sets the floor: `min <= loaded < pool_size`
  boots with a `degraded pool` warning; fewer than `min` still errors. New
  `Engine::load_with_pool_size_min`; `load_with_pool_size` keeps the strict
  all-or-nothing behavior.
- **Server-side WebSocket keepalive.** The server now pings each WebSocket
  every 30 s and closes the socket after two consecutive unanswered pings
  (any inbound frame resets the counter). Detects half-open TCP sessions and
  keeps connections alive through idle-dropping proxies — far faster than the
  300 s idle timeout.
- **Benchmark regression gate.** `tests/benchmark.rs` loads a committed
  `tests/benchmark_baseline.json` and now **fails** (non-zero exit) when WER
  regresses past `tolerance_pp` — with a printed diff table and a
  `GIGASTT_BENCHMARK_UPDATE_BASELINE=1` refresh path. The absolute `MAX_WER`
  ceiling is a hard failure too; previously it only warned and `pass` was
  hardcoded `true`.
- **Export formats for transcription results.** The REST endpoint
  `/v1/transcribe` now accepts `?format=txt|json|srt|vtt|md` (JSON remains the
  default) and optional formatter controls (`max_chars_per_line`,
  `max_words_per_line`, `word_timestamps`, `download`). The CLI command
  `gigastt transcribe` gained `--format`/`-f`, `--output`/`-o`,
  `--max-chars-per-line`, `--max-words-per-line`, and `--word-timestamps`.
  New `gigastt-core::export` module provides pure formatters; SRT/VTT are
  speaker-aware and Markdown includes YAML frontmatter with `duration`,
  `language: ru`, and `speakers`.

### Changed

- **`/metrics` is now served on a separate loopback listener** (default
  `127.0.0.1:9090`, configurable via `--metrics-listen` / `GIGASTT_METRICS_LISTEN`)
  instead of the primary port. Previously it sat behind the primary CORS
  allowlist and per-IP rate limiter, so any allowlisted browser origin could
  read request telemetry and a 15 s Prometheus scraper could be throttled.
  **Breaking for metrics scrapers:** point Prometheus at the new port (loopback
  by default; expose it deliberately). Requires `--metrics`. A non-loopback
  `--metrics-listen` now requires the same `--bind-all` (or
  `GIGASTT_ALLOW_BIND_ANY=1`) opt-in the primary port does, so telemetry can't
  be exposed network-wide without an explicit, warned acknowledgment.
- **Bounded the Prometheus `path` label** to the known route set (unmatched
  paths collapse to `other`) so scanners hitting arbitrary URLs can no longer
  explode metric label cardinality.

- **polyvoice** 0.6.8 → 0.7.0. Our streaming diarization path (`StreamingPipeline`)
  is still bound on polyvoice's legacy `EmbeddingExtractor` trait, which 0.7.0
  deprecated in favour of the v1.0 `polyvoice::embedder::Embedder` API; the legacy
  usage is annotated `#[allow(deprecated)]` (mirroring polyvoice's own crate-level
  suppression) until upstream wires `Embedder` into the streaming pipeline.

### Fixed

- **INT8 quantization axis.** The quantizer chose `axis=0` for every
  weight; for `MatMul`/`Gemm` weights the per-output-channel axis is the `N`
  dimension, so per-channel scales were grouped along the wrong axis — silently
  inflating INT8 weight error (and WER). The axis is now derived from the
  consuming op (`Conv`→0, `MatMul`→last dim, `Gemm`→`transB`-dependent) and the
  weight is quantized along it with a strided gather. Regenerated INT8 verified
  on the bundled Golos set (1.3% WER, model loads and transcribes cleanly).
- **Monotonic wire timestamps.** `now_timestamp()` is anchored once to a
  process-start `Instant`, so segment timestamps stay epoch-aligned but advance
  monotonically — immune to NTP steps / wall-clock jumps mid-process.
- **AsyncAPI `WordInfo`** now documents `confidence` (required) and
  `speaker` (optional), matching the Rust struct.
- **SSE error parity with WebSocket.** The SSE streaming endpoint
  (`/v1/transcribe/stream`) used to collapse every failure into one generic
  `inference_error` event and stayed silent on a panic. It now emits a stable
  per-variant code (`GigasttError::code()` — `inference_error` / `invalid_audio`
  / …) and a distinct `inference_panic` event when the inference task panics,
  matching the WebSocket error contract.
- **Minor hardening:** single `tokenizer::WORD_BOUNDARY` const for the U+2581
  marker; `MelSpectrogram: Default`; `/health` is liveness-only
  and no longer touches engine state; server shutdown logs a `warn!`
  instead of silently swallowing a oneshot `RecvError`.

## [2.2.1] - 2026-06-18

### Added

- **Disk cache for benchmark transcription results.** Repeated benchmark runs now
  reuse cached hypotheses keyed by runner configuration and audio file SHA-256,
  dramatically reducing iteration time during benchmark development.
- **WER histograms in benchmark output.** Results now include per-runner
  histograms broken down by audio duration, reference word count, and WER bucket.
- **Benchmark profiling flag.** `python benchmark.py --profile` dumps cProfile
  stats to `benchmark.prof` for performance investigation.

### Changed

- **Benchmark runner lifecycle and cache config.** Runner selection now uses a
  module-level class registry to avoid import-time side effects, and the
  `gigastt` runner cache config uses a documented schema-version constant.

## [2.1.0] - 2026-06-14

A large correctness + honesty pass (a full project audit), an honest cross-ASR
benchmark against current Russian-ASR engines, and a reworked README.

### Added

- **Honest cross-ASR benchmark — 7 engines × 4 Russian domains.** gigastt, Vosk 0.42,
  **Vosk 0.54** (Zipformer2 via sherpa-onnx), whisper.cpp, faster-whisper,
  faster-whisper-turbo, and **T-one** (greedy + production beam+LM) on clean read,
  far-field, phone calls and YouTube — with WER + 95% CI, RTF, footprint and an explicit
  training-contamination caveat. See [`docs/benchmarks.md`](docs/benchmarks.md).
- **Release platform matrix expanded** to `x86_64-pc-windows-msvc` and
  `aarch64-unknown-linux-gnu` (cross-compiled), with matching Homebrew coverage.
  (Prebuilt `x86_64-apple-darwin` Intel-Mac tarballs were dropped: GitHub's
  `macos-13` Intel runners are being retired and no longer schedule. Intel-Mac
  users install via `cargo install gigastt`.)
- **Benchmark data licensing** — `benchmark/DATA_LICENSE` + root `NOTICE` carve the
  benchmark transcripts (Open STT CC BY-NC 4.0; Golos Sber Public License) out of the
  project's MIT grant.
- **CI builds the Docker images** (`docker-build` jobs): the CPU image plus a `--version`
  smoke run on every main push and on docker-relevant PRs; the CUDA and benchmark images
  on main push.

### Changed

- **README reworked (EN + RU)** — niche-led, honest positioning (an embeddable on-device
  Russian STT, not a WER-leaderboard claim), ~80 lines, 4 badges, with detail split into
  `docs/{api,benchmarks,architecture,troubleshooting}.md`. Removed unmeasured/false
  claims (sub-200 ms streaming, "only Rust-native"); streaming reframed honestly as
  buffered/chunked over an offline RNN-T (~0.7 s time-to-first-partial).
- **API + docs aligned to the code** — REST / OpenAPI / AsyncAPI error contracts, a
  single flagship WER (8.60% renorm), and consistent test counts / model sizes / MSRV
  across README, CLAUDE.md and AGENTS.md.
- **Supply-chain transparency** — documented the build-time onnxruntime fetch (ort's
  default `download-binaries`), and pinned `cargo-ndk` and the benchmark lockfile.
- **MSRV bumped 1.87 → 1.88** (`ort`/`ort-sys` 2.0.0-rc.12 require it). `ort`/`ort-sys`
  remain pinned to the pre-release `2.0.0-rc.12` — no stable `ort` 2.0.x exists yet.

### Fixed

- **Streaming recognition quality** — decode on a sliding context window over the offline
  RNN-T (fixes the isolated-chunk regression that collapsed partials to a single token),
  with a decode stride to stay real-time on CPU; streaming word-timestamp units corrected.
- **Model downloads** now use connect/read timeouts and a bounded redirect policy.
- **All three Docker images build again** (builder pins behind the workspace MSRV, stale
  dummy-rlib relinking, missing `[[bench]]`/`[[test]]` stubs, `pkg-config`/`libssl-dev`
  for `openssl-sys`, and base images too old for ort's prebuilt onnxruntime). Cargo
  invocations now run `--locked`.

### Security

- **Audio sample-rate DoS bounded** — a hostile WAV header can no longer drive an
  unbounded allocation; the decode sample budget is clamped to a fixed ceiling.
- **FFI use-after-free guarded** — the C-ABI hot paths reject already-disposed handles
  and use acquire/release ordering on the dispose flag.

## [2.0.14] - 2026-06-11

### Fixed

- **CoreML EP no longer fails at runtime on the GigaAM Conformer encoder
  (issue #42).** CoreML cannot execute partitions compiled with dynamic
  shapes — every prediction failed with `error code: -1` regardless of
  compute units or model format. The EP is now configured with
  `RequireStaticInputShapes` + `MLProgram`: heavy conv/matmul blocks run on
  the Neural Engine, dynamic-shape ops stay on CPU. Measured ~3x faster
  encoder inference on a 4 s WAV and ~5.6x on a 2-minute file vs the
  pure-CPU build (M1 Pro, INT8, release).

### Added

- **`Engine::warmup()` + automatic CoreML→CPU runtime fallback.**
  `Engine::load` now probes CoreML with ~1 s of silence and transparently
  rebuilds all sessions on the CPU EP if the probe fails (covers both
  session-load and first-`Run()` failures), logging
  `falling back to CPU execution provider` instead of crashing. The server
  warms every pooled session triplet before `axum::serve` accepts traffic,
  removing the first-request cold start.
- **CoreML runtime smoke in CI** (`build-coreml`, macos-14, main push only):
  transcribes `golos_00.wav` with the release binary and fails on inference
  error, missing reference text, or silent CPU fallback.
- **Full cross-ASR benchmark on 9 994 Golos crowd samples.**
  - Vosk: 4.27% WER / 0.107x RTF (1.3 GB)
  - gigastt: 11.37% WER / 0.335x RTF (230 MB)
  - whisper.cpp: 14.96% WER / 1.108x RTF (~3 GB)
  - faster-whisper: 15.73% WER / 1.224x RTF (~3 GB)
  - Results published on `benchmark-results-local` branch with shield.io badges.
  - Added sequential runner script and monitor for long-running benchmarks.

## [2.0.13] - 2026-06-08

### Dependencies

- **prost** 0.14.3 → 0.14.4
- **prost-build** 0.14.3 → 0.14.4
- **polyvoice** 0.6.7 → 0.6.8

### CI

- **actions/cache** 4 → 5
- **codecov/codecov-action** 4 → 7

## [2.0.12] - 2026-06-03

### Added

- **CI coverage expanded to 90 %+.**
  - `coverage-e2e` job now runs ignored lib+bin unit tests (model-dependent)
    alongside integration tests, merging all coverage into a single Cobertura
    report.
  - CLI command coverage: `--help`, `serve --help`, `download`, `quantize --force`,
    and `transcribe` are exercised via `cargo llvm-cov run` to cover the `main()`
    command-dispatch paths.
  - New E2E WebSocket tests: unrecognized text message (`Ok(_) => Continue`),
    client Close frame handling, and max-session pre-check branch.
  - New unit tests for `build_limits` with valid/invalid TOML config files.

### Changed

- `.gitignore` now also ignores `lcov-*.info` and `lcov-*.xml` (local coverage
  merge artifacts).

## [2.0.11] - 2026-06-01

### Added

- **Comprehensive unit-test coverage expansion across all crates.**
  - `gigastt-core` — pool blocking checkout paths, ONNX error wrapping,
    engine load failure modes, feature-extractor defaults, tokenizer edge
    cases (empty lines, bare integers, no-whitespace fallback), quantizer
    end-to-end matmul and shared-weight paths.
  - `gigastt` — HTTP handler unit tests for readiness (shutdown, pool
    exhaustion, metrics), transcribe (payload too large, pool closed, invalid
    audio), SSE stream (payload too large, pool closed), JSON serialization
    fallback, server startup/shutdown, runtime-limit clamping, origin and
    request-id middleware integration.
  - `gigastt-ffi` — null-engine guards, invalid-UTF8 paths, path traversal
    rejection, `stream_new`/`process_chunk`/`flush` round-trip, idempotent
    quantize, `string_free_null` safety.
  - E2E tests refined: deduplicated shared helpers, cleaner shutdown
    sequencing, expanded rate-limit and error-path coverage.
- **`__internals` required feature for `tokenizer` benchmark.** Prevents
  `cargo clippy --all-targets` from failing on the private tokenizer module.

### Fixed

- Flaky `test_readiness_with_metrics` caused by shared engine singleton
  contention (pool size 1). Switched to isolated `fresh_engine()` per test.

## [2.0.10] - 2026-05-31

### Changed

- **rubato 0.16 → 3.0** — migrated resampling (`resample` / `resample_with_cache`)
  for the breaking 3.0 API: `SincFixedIn` replaced by `Async::new_sinc` with
  `FixedAsync::Input`, and the new `audioadapter` buffer model
  (`SequentialSliceOfVecs`). Behaviour-preserving (same sinc parameters).
- **polyvoice 0.5 → 0.6.7** — migrated speaker diarization for the new 0.6 API:
  `OfflineDiarizer` / `OnlineDiarizer` replaced by `Pipeline` / `StreamingPipeline`
  that take an explicit `EnergyVad`. The `EmbeddingExtractor` is now owned per
  pipeline, so the shared ONNX speaker encoder is wrapped in `Arc` behind a
  `SharedExtractor` adapter and cloned into each session. `DiarizationConfig`
  flat fields moved under nested `ClusterConfig` / `VadConfig` (streaming
  clustering threshold pinned to 0.5).
- **toml 0.9 → 1.1** — TOML config parsing update.
- **dashmap 6.1 → 6.2** — concurrent map update (per-IP rate limiter).
- **serde_json 1.0.149 → 1.0.150** — JSON serialization patch update.

## [2.0.9] - 2026-05-24

### Changed

- **symphonia 0.5 → 0.6** — migrated audio decoding (`decode_audio_inner`) for
  breaking changes in Symphonia 0.6.0: `SampleBuffer` replaced with
  `GenericAudioBufferRef` copy methods, `next_packet()` now returns `Ok(None)` on
  EOF, `default_track()` requires `TrackType::Audio`, `CodecParameters` is now an
  enum with audio params accessed via `.audio()`.
- **thiserror 1.0 → 2.0** — error derive macro update.
- **toml 0.8 → 1.1** — TOML config file parsing update.
- **tokio 1.52.2 → 1.52.3** — async runtime patch update.
- **cbindgen 0.28 → 0.29** — FFI header generation build-dependency update.
- **codecov-action 5 → 6** — CI coverage action update.

## [2.0.8] - 2026-05-07

### Changed

- **WER benchmark expanded to full Golos crowd dataset** — `benchmark.rs` now
  reads from `~/.gigastt/benchmarks/golos_wav/manifest.json` (9 994 samples) and
  falls back to the bundled 15-file smoke test when the external set is missing.
  Full run: **11.4% WER** (5 729 errors / 50 394 words, 95% bootstrap CI
  [10.9%, 11.9%]) on Apple M1 CPU (~105 min).  Added bootstrap confidence
  intervals, progress reporting every 50 samples, and `translit_anglicisms`
  normalization for loanwords (`tv` → `тв`, `synergy` → `синергия`, etc.).

## [2.0.7] - 2026-05-08

### Fixed

- **PCM16 carry byte loss** — `parse_pcm16_with_carry_into` no longer drops the
  last byte of an odd-length chunk when no carry is pending. The previous
  tuple-pattern loop `while let (Some(b0), Some(b1))` consumed the trailing byte
  into `b0` and then discarded it when `b1` was `None`.

### Added

- **Pre-commit hook** — `.githooks/pre-commit` enforces `fmt`, `clippy`, and
  `cargo test --workspace` before every commit.
- **Makefile** — `make check` runs the full local validation suite;
  `make fix` auto-formats and applies clippy suggestions.

### Changed

- **AGENTS.md** — documents hook setup and required GitHub branch protection
  rules to prevent broken code from reaching `main`.

## [2.0.6] - 2026-05-08

### Security / Reliability

- **Pool slot leak on cancelled checkout** — `PoolInner::checkin` now retries dead
  waiters instead of silently dropping the item when a `checkout` future is
  cancelled (timeout, `select!`, or abort).  Three new unit tests verify the fix.
- **SIGTERM handling** — the server now traps `SIGTERM` on Unix in addition to
  `SIGINT`, enabling graceful shutdown in Docker and Kubernetes environments.
- **FFI panic safety** — `gigastt_transcribe_file` wraps inference in
  `catch_unwind`, preventing undefined behavior when an ONNX panic crosses the
  C ABI boundary.
- **FFI path traversal via symlinks** — `gigastt_transcribe_file` now resolves
  symlinks with `canonicalize()` before the working-directory boundary check,
  blocking symlink attacks that escape the sandbox.
- **Tokenizer blank-id underflow** — `Tokenizer::load` guards against empty
  vocabularies and uses `saturating_sub` instead of `tokens.len() - 1`,
  eliminating a subtraction underflow on corrupted `vocab.txt`.
- **SIGHUP registration failure** — replaces `expect()` with graceful degradation;
  the server no longer aborts when signal fds are exhausted in containers.
- **Decode loop panic surface** — replaces `unwrap()` in the RNN-T greedy decode
  blank-run cache with `anyhow::bail!` and `expect` with clear messages.
- **Resampler cache invariant** — replaces `unreachable!()` in
  `resample_with_cache` with `anyhow::bail!`, and surfaces resampler failures
  instead of silently returning empty output.

### Protocol / API

- **WebSocket idle timeout Close frame** — idle timeouts now emit a
  `ServerMessage::Error` with code `"idle_timeout"` followed by `Close(1001)`,
  making the reason distinguishable from a network partition.
- **WebSocket empty-frame spam Close frame** — exceeding
  `MAX_EMPTY_FRAMES_PER_SESSION` now sends `Error` + `Close(1008)` before
  dropping the connection.
- **CORS preflight for all protected routes** — `OPTIONS` handlers added to
  `/v1/models`, `/v1/ws`, and `/metrics` so browser preflight requests no
  longer receive 405.
- **Rate limiter `Retry-After`** — header is now computed from the actual refill
  interval instead of the hard-coded `60` seconds.  The 429 JSON body now
  includes `retry_after_ms` for consistency with the 503 pool-saturation response.
- **SSE final segment on shutdown** — the SSE streaming task always flushes a
  final segment (even during shutdown), matching WebSocket behavior.

### Concurrency / Reliability

- **Pool lost-wakeup race** — `checkout` and `checkout_blocking` now re-check
  the `items` queue under the `waiters` lock before registering a waiter,
  closing the race where a concurrent `checkin` could return an item to the
  pool while the caller needlessly sleeps.
- **Rate limiter O(n) eviction** — replaced the global `iter().min_by_key()`
  scan with a bounded 100-entry sample, eliminating the DoS vector where an
  attacker fills the map and forces a 100 k-entry scan on every new IP.

### Performance

- **Eliminated encoder output copy** — `run_inference` passes the encoder
  tensor borrow directly to `greedy_decode` instead of `to_vec()`-copying it.
  Saves ~1–4 MB of transient allocation per chunk (scales with audio duration).
- **Reusable mel-output buffer** — `StreamingState` now owns a `mel_output`
  `Vec<f32>` that is resized in-place by `compute_with_buffers`, removing one
  ~4–20 KB allocation per `process_chunk` call.
- **Metrics interning** — metric family names are now `Arc<str>` keys in the
  Prometheus registry, eliminating `String` allocation on every `counter_inc`,
  `gauge_set`, and `histogram_record`.  The public API accepts
  `&[(&str, &str)]` labels so callers no longer allocate `Vec<(String, String)>`
  on the hot path.
- **Flattened mel filterbank** — `MelSpectrogram.mel_filterbank` changed from
  `Vec<Vec<f32>>` to a contiguous `Vec<f32>` (row-major), improving cache
  locality during the per-frame mel dot-product and removing one
  pointer-indirection per bin.
- **Zero-copy audio buffer preparation** — `prepare_audio_buffer` now returns
  `Option<usize>` (usable sample count) instead of allocating a new `Vec<f32>`.
  The caller borrows `&buffer[..usable]` for feature extraction and then shifts
  leftovers in-place, removing one ~0.5–5 KB allocation per `process_chunk`.
- **Reusable PCM decode buffer** — new `parse_pcm16_with_carry_into` writes
  decoded f32 samples into a caller-provided `&mut Vec<f32>`. The WebSocket
  handler reuses a single buffer across binary frames, eliminating the per-frame
  `Vec<f32>` allocation from PCM16 decoding.
- **Resampler zero-alloc path** — `resample_with_cache` now sanitizes non-finite
  samples in-place (no extra `Vec`), takes owned input, and writes output into a
  reusable buffer via `process_into_buffer`, removing rubato's internal output
  allocation on every resampling call.

### Fixed

- **Dependency version mismatch** — `gigastt/Cargo.toml` and
  `gigastt-ffi/Cargo.toml` now specify `gigastt-core = "2.0.5"` instead of the
  stale `"2.0.2"`.
- **ONNX cache dir creation error swallowed** — `create_dir_all` failure now
  propagates with context instead of being silently ignored.
- **Pre-epoch clock warning** — `now_timestamp()` logs a `warn!` when the
  system clock is before Unix epoch instead of silently returning `0.0`.
- **`libc::flock` safety comment** — added `// SAFETY:` documentation for the
  only `unsafe` block outside the FFI crate.
- **AGENTS.md updated** — removed stale `/ws` alias references and documented
  the `nnapi` execution-provider feature.

## [2.0.5] - 2026-05-07

### Fixed

- **Model download lock file** — `ensure_model` now creates the target directory
  *before* acquiring the advisory `flock`, preventing a "No such file or directory"
  error when `~/.gigastt/models` does not yet exist (regression in v2.0.4).

## [2.0.4] - 2026-05-07

### Security / Reliability

- **Pool lock poisoning eliminated** — replaced `std::sync::Mutex` with `parking_lot::Mutex` in `PoolInner`. A panic in any inference thread no longer poisons the mutex and permanently disables the pool.
- **Pool session leak fixed** — `OwnedReservation` now owns the `SessionTriplet` and implements `Drop`, guaranteeing the slot is returned even when `spawn_blocking` panics or is cancelled.
- **Rate limiter memory cap** — `DashMap` now has a hard limit of 100 000 buckets. Oldest entries are evicted when the cap is hit, preventing unbounded growth under rotating-IP botnets.
- **WS empty-frame spam protection** — connections are closed after 1 000 empty binary frames to prevent CPU/queue exhaustion.
- **Model download TOCTOU fixed** — `ensure_model` uses advisory `flock()` and unique `.partial` filenames so concurrent processes cannot corrupt partial downloads.
- **CLI validation** — `--rate-limit-burst 0` with `--rate-limit-per-minute > 0` is now rejected at startup.
- **SSE keep-alive** — explicit 15-second heartbeat comments keep proxies (nginx, ALB) from silently closing idle streams.
- **Metrics lock poisoning fixed** — `std::sync::RwLock` replaced with `parking_lot::RwLock` in the Prometheus registry.
- **Distributed tracing spans** — `tracing::Span::current()` is propagated across `spawn_blocking` boundaries so the full request path (pool wait → inference → output) is traceable.

### Added

- New Prometheus gauge `gigastt_pool_waiters` exposing the number of tasks blocked on pool checkout.
- Unit tests for `OwnedReservation` panic recovery and `Option<OwnedReservation>` round-trips.

## [2.0.3] - 2026-05-07

### Fixed

- **Documentation sync** — updated Russian README, AGENTS.md, CHANGELOG,
  SECURITY.md, OpenAPI spec, release verification docs, and production-readiness
  spec to reflect the v2.0 workspace split accurately.

## [2.0.2] - 2026-05-07

### Added

- **`crates/gigastt-core/README.md`** — dedicated README for the `gigastt-core`
  crate on crates.io.

## [2.0.1] - 2026-05-07

### Fixed

- **crates.io publishing** — added `readme` path to all crate manifests for
  proper crates.io display.
- **Path dependency versions** — added explicit versions to `gigastt-core` path
  dependencies for crates.io publish compatibility.
- **Workspace README** — updated top-level README for the workspace split.

## [2.0.0] - 2026-05-07

### Changed

- **Workspace split** — monolith refactored into a 3-crate Cargo workspace:
  - `gigastt-core` — inference engine, model download, quantization, audio
    decoding (library crate)
  - `gigastt-ffi` — C-ABI bindings for Android (cdylib)
  - `gigastt` — CLI + axum server (binary crate)
  This enables embedding `gigastt-core` as a standalone library in other Rust
  projects.

## [1.0.1] - 2026-05-06

### Changed

- **polyvoice** dependency bumped from `0.4.3` to `0.5.2` (VAD, AHC
  clustering, DER evaluation, CLI tooling; no API changes for gigastt).

## [1.0.0] - 2026-05-06

First stable release. All P0 blockers and P1 ship-before-v1.0 items from the
production-readiness review are closed. Public API (REST, WebSocket, CLI) is
now covered by semver guarantees.

### Added

- **Configurable pool checkout timeout** — `--pool-checkout-timeout-secs`
  (env `GIGASTT_POOL_CHECKOUT_TIMEOUT_SECS`, default 30). `Retry-After`
  headers and `retry_after_ms` JSON fields derive from the same value.
- **WebSocket protocol version negotiation** — `Configure` accepts an
  optional `protocol_version` field; the server rejects unsupported versions
  with error code `unsupported_protocol_version`. `Ready` includes
  `min_protocol_version` for client-side discovery.
- **Extended Prometheus metrics** — `gigastt_pool_available` (gauge),
  `gigastt_pool_checkout_duration_seconds` (histogram),
  `gigastt_pool_timeouts_total` (counter), `gigastt_ws_active_connections`
  (gauge), `gigastt_inference_duration_seconds` (histogram),
  `gigastt_rate_limit_rejections_total` (counter).
- **`SECURITY.md`** — responsible disclosure policy (90-day timeline).
- **`docs/privacy.md`** — auditable privacy-first claim documentation.
- **`docs/observability/`** — example Prometheus alerting rules and a
  starter Grafana dashboard JSON.
- **`cargo-semver-checks`** in CI — catches accidental breaking changes on PRs.
- **`cargo-tarpaulin`** coverage job — uploads to Codecov on main push.

### Changed

- **Idle timeout e2e test** uses a 3 s server-side timeout instead of the
  default 300 s, reducing wall-clock from ~310 s to ~5 s.

### Removed

- **`POOL_RETRY_AFTER_MS` / `POOL_RETRY_AFTER_SECS` constants** — replaced
  by functions that derive the hint from `pool_checkout_timeout_secs`.

## [0.10.0] - 2026-05-05

Speaker diarization is now a default feature powered by the polyvoice crate
from crates.io. Breaking CLI change: `--diarization` replaced by `--skip-diarization`.

### Changed

- **Diarization default feature** — `diarization` added to default Cargo features;
  speaker model downloads automatically with `gigastt download`.
- **polyvoice from crates.io** — replaced local path dependency (`../polyvoice`)
  with published `polyvoice = "0.4.3"` from crates.io.
- **CLI flag renamed** — `gigastt download --diarization` replaced by
  `gigastt download --skip-diarization` (opt-out instead of opt-in).
- **API adaptation** — `DiarizationConfig` uses `SampleRate` wrapper type and
  explicit `max_gap_secs` field to match polyvoice 0.4.3 API.

### Removed

- **In-tree diarization module** (`src/inference/diarization.rs`) — `SpeakerEncoder`,
  `SpeakerCluster`, and `cosine_similarity` replaced by polyvoice's
  `OnnxEmbeddingExtractor`, `OnlineDiarizer`, and `OfflineDiarizer`.

### Fixed

- Duplicate `tokens.is_empty()` guard in `tokens_to_words`.
- Stale `diarization.rs` references in CLAUDE.md and AGENTS.md.
- Magic numbers in `OnnxEmbeddingExtractor::new()` extracted to named constants.
- `text` assembly in `transcribe_samples` moved after diarization annotation.

## [0.9.6] - 2026-05-04

Performance, security, and DX improvements. Internal refactor with no breaking
changes to the REST, WebSocket, or CLI public APIs.

### Added

- **OpenAPI REST spec** (`docs/openapi.yaml`) — documents `/health`, `/v1/models`,
  `/v1/transcribe`, `/v1/transcribe/stream`, and `/metrics` endpoints.
- **Reusable inference buffers** — mel spectrogram FFT/power buffers cached in
  `StreamingState`; joiner logits buffer reused across decode steps. Reduces
  per-chunk allocations by ~60 % in the streaming hot path.
- **Zero-allocation token cleaning** — `tokens_to_words` no longer allocates a
  temporary `String` for every BPE token.
- **`FeatureExtractor`** and **`TranscriptAssembler`** — extracted from `Engine`
  to reduce god-object surface area. `Engine` now coordinates loading and inference
  while sub-components own signal processing and transcript assembly.
- **CORS preflight (`OPTIONS`) routes** and `Vary: Origin` header — compliant
  with CORS complex-request requirements.
- **Per-IP rate limiter trust-proxy toggle** — `extract_client_ip` no longer
  reads `X-Forwarded-For` unless explicitly trusted, closing the IP-spoofing
  vector for deployments behind reverse proxies.
- **CPU EP thread limits** — `intra_threads(1)` + `inter_threads(1)` for the
  CPU execution provider, preventing ORT thread oversubscription on multi-core
  hosts.
- **Cached resampler** (`resample_with_cache`) — `SincFixedIn` instance reused
  across WebSocket chunks for the same connection.
- **WER gate in benchmark suite** — `tests/benchmark.rs` asserts `wer < 12.0`
  so regressions fail CI.
- **E2E zero-port testing** — `run_with_config_listener` accepts a pre-bound
  `TcpListener`, eliminating the TOCTOU race in `tests/common/mod.rs`.

### Changed

- **WebSocket examples** (`examples/*`) migrated to canonical `/v1/ws` path,
  added `ready`-wait and explicit `stop` signal.
- **`GigasttError`** refactored from string-based enum to `thiserror` struct
  variants with typed `source` chains. Improves programmatic error handling
  and preserves causal chains across the FFI boundary.
- **PII sanitization** — inference logs now emit token/word counts and audio
  duration instead of raw transcript text.

### Fixed

- **Pool deadlock** — reverted `Pool<T>` back to `async-channel` (the partial
  `tokio::sync::mpsc` migration deadlocked `close()` vs `blocking_recv()`).
- **`soak.yml`** stdout redirect and artifact retention fixed.
- **CI** migrated to `nextest` for faster, more reliable unit-test runs.

## [0.9.5] - 2026-04-23

Production-hardening release: security fix, Android FFI support, and lean builds.

### Security

- **RUSTSEC-2026-0104** — `rustls-webpki` 0.103.12 → 0.103.13. Reachable panic in CRL parsing fixed.

### Added

- **Android FFI layer** (`src/ffi.rs`, feature `ffi`). C-ABI exports:
  `gigastt_engine_new`, `gigastt_engine_new_with_pool_size`,
  `gigastt_transcribe_file`, `gigastt_stream_new`,
  `gigastt_stream_process_chunk`, `gigastt_stream_flush`,
  `gigastt_stream_free`, `gigastt_string_free`, `gigastt_engine_free`.
  Enables embedding gigastt as `libgigastt.so` in Android apps.
- **`ort/nnapi`** via feature `ffi` — Android NPU/DSP acceleration when available.
- **`Pool::checkout_blocking()`** — synchronous pool checkout for FFI callers.
- **`Serialize` derive on `TranscriptSegment`** — JSON serialization for FFI streaming.
- **Android default pool size = 1** — reduces mobile RSS from ~560 MB to ~350 MB.
- **`server` Cargo feature** (enabled by default) — gates `axum`, `tokio-stream`,
  `tokio-util`. Building `--no-default-features --features ffi` strips server
  dead code from mobile `.so` binaries.
- **Binary target** (`gigastt` bin) requires `server` feature via
  `required-features` in `Cargo.toml`.
- **Android CI workflow** (`.github/workflows/android.yml`) — builds
  `libgigastt.so` for `arm64-v8a` on every PR/push.

### Changed

- **README refactor** — added terminal demo block, Troubleshooting section,
  HTTP error codes table, extracted full CLI reference to `docs/cli.md`,
  fixed "unlimited concurrent" comparison, added Contributing link.

### Fixed

- Clippy warnings: `clippy::single_match`, `clippy::items_after_test_module`,
  `clippy::manual_is_multiple_of`.

## [0.9.4] - 2026-04-21

Dependency-bump rollup. No functional source changes; every entry here is
a Dependabot PR that landed green on the polish-green main from 0.9.3.

### Dependencies

- `reqwest` 0.12.28 → 0.13.2 (new TLS backend pulls `aws-lc-rs`).
- `prost-build` 0.13.5 → 0.14.3 (`petgraph` 0.7 → 0.8 transitively).
- `axum` 0.8.8 → 0.8.9 + `tokio-tungstenite` 0.28 → 0.29 in dev-deps
  (wire-protocol unchanged).
- `tokio`-ecosystem group bump (tokio + tokio-* minors).

### CI / workflow actions

- `actions/checkout` 4 → 6 across ci.yml, release.yml, soak.yml, homebrew.yml.
- `actions/upload-artifact` 4 → 7 in release.yml, soak.yml.
- `actions/attest-build-provenance` 3 → 4 in release.yml.
- `softprops/action-gh-release` 2 → 3 in release.yml.

### Not landed

- `rubato` 0.16 → 2.0 (Dependabot PR #9) closed: the 2.0 release removes
  `SincFixedIn` in favour of a new `audioadapter` API; migrating
  `src/inference/audio.rs::resample` is a code change Dependabot can't
  generate automatically. Pinned at 0.16.2 until someone ports the
  resampler.

## [0.9.3] - 2026-04-21

Polish-before-production release. No functional behaviour changes for existing
clients; server, CLI, REST, and WebSocket surfaces are wire-compatible with
v0.9.2. Dockerfile was broken since v0.9.0 — this release fixes it.

### Fixed

- **Docker images now actually build** (`Dockerfile`, `Dockerfile.cuda`). Both
  builders gained `protobuf-compiler` (required by `build.rs` since v0.9.0's
  `prost-build` migration) and now `COPY proto/` + `build.rs` before `src/`.
  The 0.9.0 / 0.9.1 / 0.9.2 images failed at `cargo build` with
  `prost-build failed to compile proto/onnx.proto`; the published Docker
  recipes in README only worked if the reader had protoc in their base image.
- **`tests/e2e_rest.rs::test_rest_large_body_rss_within_budget` removed.** The
  test asserted `RSS_after - RSS_before < wav.len() * 3 + 40 MiB` after
  POSTing a 300 s WAV. Every main-push CI run since v0.9.0 observed a delta
  of ~320 MiB regardless of whether the REST upload path was zero-copy or
  4×-copy, because ONNX Runtime's encoder scratch for 5 minutes of 16 kHz
  audio allocates ~90+ MiB by itself. The test could neither catch the
  zero-copy regression it was designed for nor pass reliably. The zero-copy
  contract is covered by the `BytesMediaSource` impl in
  `src/inference/audio.rs` and the unit tests around it.
- **`src/inference/tokenizer.rs`**: skip vocab lines that parse as a bare
  integer (e.g. a legacy `1025\n` size header). Such a line has no trailing
  id column, so the existing `rfind([' ', '\t'])` fallback would push the
  integer string as a ghost token and poison the ID space.
- **`src/server/metrics.rs::fmt_f64_prom`**: drop the empty-body
  `if v == v.trunc() && v.abs() < 1e15 { }` branch that existed only to
  document that it was a no-op. The `format!("{v}")` tail always ran.

### Changed

- **`Engine::transcribe_file` / `transcribe_bytes_shared` share a
  `transcribe_samples(&[f32], &mut SessionTriplet)` tail.** Previously the
  two bodies duplicated the mel → encoder → decode → word-join sequence
  byte-for-byte. Same public API, same behaviour, one implementation.
- **`src/server/mod.rs`**: `MAX_RPM` clamp + warn moved into
  `RateLimiter::new`; the startup log line calls `limiter.interval_ms()`
  instead of duplicating the math. Dropped the write-only
  `RateLimiter::last_evict_ms` field — eviction already runs on a tokio
  interval, nothing read the stored timestamp. Exposed the public
  constant `server::rate_limit::MAX_RPM` (= 60 000) for external callers.
- **`src/server/mod.rs`**: single source of truth for `SUPPORTED_RATES`
  (`pub(crate)`). The REST `/v1/models` handler used to inline
  `vec![8000, 16000, 24000, 44100, 48000]` — it now reuses the same
  const the WS `Ready` payload reads.
- **`src/server/http.rs`**: `vocab_size` in `/v1/models` comes from
  `engine.vocab_size()` (new public `Engine::vocab_size()`), not a `1025`
  literal. If the upstream model rev ever resizes its BPE vocabulary the
  REST surface no longer lies.
- **`src/model/mod.rs`**: extracted `stream_to_partial_then_finalize(url,
  final_dest, expected_sha256, label)`. The per-file GigaAM download
  loop and the single-file speaker-diarization download now share one
  implementation of URL fetch, progress, stream-to-partial, and
  SHA-256 + atomic-rename finalize. Drops ~50 duplicated lines.
- **`src/quantize.rs`**: staging file suffix switched from `.onnx.tmp` to
  `.partial` so the in-tree INT8 quantizer uses the same convention as
  the HuggingFace download pipeline in `src/model/mod.rs`.
- **`tests/server_integration.rs` removed** (367 LoC, 6 tests). Every case
  is covered by the v0.4.3+ `tests/e2e_ws.rs` / `tests/e2e_rest.rs` /
  `tests/load_test.rs` suites; the legacy file used `sleep(200ms)` race
  gates and the long-deprecated `server::run(engine, port, host)`
  signature.
- **Deprecation headers on `/ws`**: the upgrade response now carries
  RFC 8594 `Deprecation: true` plus `Link: </v1/ws>; rel="successor-version"`
  so client libraries can surface the migration warning before v1.0
  drops the alias. Server-side warn log was already in place.
- **Docs + specs housekeeping:**
  - `specs/design-v1.0-{pool-and-rate-limit,rest-streaming,ws-lifecycle}.md`
    → `specs/archive/design-v1.0/` (all three shipped in v0.9.0-rc.1).
  - `docs/superpowers/` (v0.4 pre-ship plans) → `docs/archive/superpowers-v0.4/`.
  - `missions/gigastt-wer/` scratchpad deleted.
  - `specs/prod-readiness-v1.0.md` now carries a v0.9.0 rollup banner
    listing the closed IDs; detail rows left for historical trail.
  - `README_RU.md` synced to English README (`/v1/ws`, `/metrics` row,
    `125 unit tests`, INT8 section rewritten for the no-feature-flag
    behaviour shipped in v0.9.0).
  - `docs/deployment.md` rate-limiter version string corrected
    (v0.8.0, not v0.7.3).
  - `CLAUDE.md` drops references to the now-deleted
    `tests/server_integration.rs`.

### CI

- **`cargo audit` job** switched from `cargo install cargo-audit --locked`
  (rebuilt on every PR, ~90 s) to the prebuilt `rustsec/audit-check@v2`
  action — same checker, same advisory source.
- **`soak.yml` cache key** scoped by profile (`-release`) so nightly soak
  runs don't evict the `target/` that `ci.yml` populates.

### Dependencies

- Removed redundant `tracing-subscriber` entry from `[dev-dependencies]`
  (already declared in `[dependencies]`; cargo exposes it to integration
  tests automatically).

## [0.9.2] - 2026-04-21

### Fixed

- **CI: minisign signing step accepts password non-interactively** (`.github/workflows/release.yml`). v0.9.1 release job got through the build + SBOM + provenance steps but failed at `Sign tarballs + SHA256SUMS with minisign` because `rsign2 sign -W` interprets `-W` as "write signature" (not "password"), so the process still prompted for a passphrase and rejected the key with `Wrong password for that key`. Switched to the apt-installed `minisign` binary, which reads the passphrase on stdin when stdout is non-TTY — a well-supported CI pattern.

## [0.9.1] - 2026-04-21

### Fixed

- **CI: install `protoc` on every cargo-build job** (`.github/workflows/ci.yml`, `.github/workflows/release.yml`, `.github/workflows/soak.yml`). v0.9.0 rollout failed in the release workflow because `prost-build` shells out to `protoc` and the GitHub-hosted `macos-14` + `ubuntu-latest` runners don't carry it. Every cargo-build-facing job now runs `arduino/setup-protoc@v3` right after `rust-toolchain`. No source change — the v0.9.0 binaries would have been bit-identical if the CI had succeeded; v0.9.1 is purely a rebuild.

## [0.9.0] - 2026-04-21

_Stable release promoting `0.9.0-rc.2` + the follow-up supply-chain
lockdown (vendored ONNX protobuf, in-tree token-bucket rate limiter,
in-tree Prometheus encoder, CycloneDX SBOM, SLSA provenance, minisign
release signing). See the [Unreleased] rows moved in below for the
full rollup; no functional regressions since rc.2._


### Added

- **Vendored ONNX protobuf schema + native codegen** (`proto/onnx.proto`, `build.rs`, `src/onnx_proto.rs`). `proto/onnx.proto` is copied verbatim from github.com/onnx/onnx (MIT-licensed, 1 000 LoC) and regenerated on every build by `prost-build 0.13` — replacing the unmaintained `onnx-pb 0.1.4` crate (last published 2020, transitively pinned to `prost 0.6`). Requires `protoc` in `PATH` at build time (`brew install protobuf`, `apt install protobuf-compiler`); see `build.rs` for the friendly failure message. Closes `RUSTSEC-2021-0073`: the advisory targeted `prost-types 0.6`'s `From<Timestamp> for SystemTime` path, which no longer ships in our dependency graph — the ignore block is gone from `deny.toml` and `.cargo/audit.toml`.
- **Custom Prometheus text encoder** (`src/server/metrics.rs`, replaces `metrics-exporter-prometheus`). ~280-line `MetricsRegistry` that serialises counters + histograms in Prometheus 0.0.4 exposition format. We only expose two metrics (`gigastt_http_requests_total`, `gigastt_http_request_duration_seconds`), so a full Recorder-trait registry was overkill — the new `HttpMetricsMiddleware` calls `registry.counter_inc(...)` / `registry.histogram_record(...)` directly, no global recorder, no `metrics` / `metrics-util` / `indexmap` / `quanta` transitives. 125 lib tests pass (+ 7 new metrics tests covering counter increment, histogram bucket cumulativity, label ordering, label escaping, empty labels, sum tracking, and empty-registry rendering).
- **Nightly soak + load CI**. New `.github/workflows/soak.yml` runs `cargo test --test soak_test -- --ignored` at 03:17 UTC daily (plus `workflow_dispatch` for on-demand checks), reusing the main-CI model cache so regressions in pool drift / descriptor leaks / RSS growth surface outside the fast-feedback envelope.
- **`docs/deployment.md`: rate-limiter & X-Forwarded-For section**. The published nginx recipe used `$proxy_add_x_forwarded_for`, which appends client-supplied headers, and the Caddy recipe did not forward the real peer at all — operators who turned on `--rate-limit-per-minute` were running a defence attackers could trivially bypass. Both recipes now overwrite the header with `$remote_addr` / `{remote_host}`, and a new section explains why it's not optional.

### Changed

- **Custom per-IP token-bucket rate limiter (drops `tower_governor`)** (`src/server/rate_limit.rs`, `src/server/mod.rs`, `Cargo.toml`). Replaced the `tower_governor = "0.7"` dependency with a focused ~150-line implementation tailored to gigastt's single middleware hook. Drops `tower_governor`, `governor`, `forwarded-header-value`, and `nonzero_ext` from `Cargo.lock`; `dashmap` is promoted from transitive to direct so the lock-free shard map stays explicit. Refill math preserves the per-IP refill formula (`refill_per_ms = rpm / 60_000`) — covered by both the existing `test_rate_limit_interval_formula` and the new `test_rate_limiter_refill_formula_matches_v1_06`. IP extraction still honours the X-Forwarded-For trust boundary (first hop of `X-Forwarded-For`, then `X-Real-IP`, then `ConnectInfo`). Memory is bounded by a `tokio::spawn` eviction task that tracks `shutdown_root`, not the old `std::thread::spawn` GC thread that leaked on shutdown.
- **`Engine::create_state` accepts `diarization_enabled` unconditionally** (`src/inference/mod.rs`). The parameter used to be gated behind `#[cfg(feature = "diarization")]`, so the same public API mutated between feature builds — `src/lib.rs`'s doctest compiled only with `--features diarization`, and external consumers had to wrap every call site in their own gate. The bool is now always present; without the feature a `warn!` is emitted if the caller asked for diarization so the contract mismatch stays observable.
- **INT8 quantization is now always available and auto-invoked** (`src/main.rs`, `src/quantize.rs`, `src/lib.rs`, `Cargo.toml`). The native Rust quantization pipeline no longer hides behind the `quantize` Cargo feature — `onnx-pb` and `prost` are now unconditional dependencies and `pub mod quantize` is always compiled in. `gigastt download` and `gigastt serve` both call `ensure_int8_encoder` after `model::ensure_model`, producing the `v3_e2e_rnnt_encoder_int8.onnx` artifact on first run (~2 min one-time). The `quantize` feature is retained as a documented no-op so existing `cargo install gigastt --features quantize` invocations keep working.
- **New `--skip-quantize` flag on `serve` and `download`** (env `GIGASTT_SKIP_QUANTIZE=1`, default off). Opt out of the automatic quantization step when debugging against the FP32 encoder.

### Fixed

- **Model download TOCTOU** (`src/model/mod.rs`). `download_file` used to stream each `v3_e2e_rnnt_*.onnx` blob directly into its final path, compute SHA-256 afterwards, and `remove_file` on mismatch. Between the last `write` and the hash comparison another process (or a second `ensure_model` call on restart) could observe an unverified file under the canonical name — and a crash in that window left a corrupt artefact that `model_files_exist()` would later accept, skipping re-download on next boot. Downloads now stream into `<filename>.partial`, SHA-256 is computed against the partial, and only after verification does `std::fs::rename` (atomic on the same filesystem) promote it to the final path. Mismatch or crash leaves nothing under the final name. Stale `.partial` files from previous crashed runs are deleted before the new download begins.
- **Speaker diarization model lacked SHA-256 verification** (`src/model/mod.rs`, `--features diarization`). `ensure_speaker_model` streamed `wespeaker_resnet34.onnx` (26 535 549 bytes, from `onnx-community/wespeaker-voxceleb-resnet34-LM`) straight to its final path with no integrity check, so a tampered mirror or corrupted redirect was loaded into `ort::Session` without complaint — the same failure class as the model download TOCTOU. The downloader now stages into `<name>.partial`, verifies SHA-256 against the new `SPEAKER_MODEL_SHA256 = "3955447b0499dc9e0a4541a895df08b03c69098eba4e56c02b5603e9f7f4fcbb"` constant (pinned to the 2026-04-20 HuggingFace copy), and only then atomically renames.
- **Odd-length PCM16 WebSocket frames corrupted subsequent frames** (`src/server/mod.rs`). `handle_binary_frame` called `chunks_exact(2)` directly, silently dropping a trailing odd byte whenever a client split their PCM16 stream on an odd boundary. The dropped byte put the following frame 1 sample out of phase with the audio decoded so far — subtle in the waveform, measurable in the inference output, hard to diagnose. A per-connection `pending_byte: Option<u8>` now carries the remainder across frames (prepended before the next `chunks_exact`, re-stashed if the combined length is again odd).
- **`tests/e2e_rest.rs::test_rest_large_body_rss_within_budget` was mis-sized**. The helper call `generate_wav(150, 16000)` produced a 4.6 MiB WAV but the test asserted `> 30 MiB` and panicked before running. Regenerated at `generate_wav(300, 16000)` (9.6 MiB) with a budget that now accounts for the PCM16 → f32 expansion (2× wav.len() + 40 MiB slack).
- **`deny.toml` / `.cargo/audit.toml` justification for `RUSTSEC-2021-0073`**. `onnx-pb` has no newer release on crates.io (0.1.4 is the only published version) and the closest modern replacement (`onnx-protobuf` on the `protobuf` crate family) is broken at its current release. The advisory stays ignored with a refreshed rationale documenting that the affected `From<Timestamp> for SystemTime` code path is unreachable from our quantization pipeline.

## [0.9.0-rc.2] - 2026-04-20

### Fixed

- **`test_rest_oversized_body_rejected` e2e assertion** (`tests/e2e_errors.rs`). The rc.1 assertion insisted on a JSON body with `code="payload_too_large"`, but `axum::DefaultBodyLimit` returns a plain-text 413 when `Content-Length` exceeds the cap — the middleware layer fires before the handler's defence-in-depth guard. The strict 413 status contract is unchanged; the JSON-body check is now conditional on the handler-layer guard being the one that fires. The rc.1 binaries are functionally correct.

## [0.9.0-rc.1] - 2026-04-20

_Release candidate for v0.9.0 — bundles five P0 fixes plus two supporting items (`PoolGuard` Drop, strict 413 assertion) from `specs/prod-readiness-v1.0.md`. RuntimeLimits gained two fields (`max_session_secs`, `shutdown_drain_secs`) — external callers constructing the struct literally must update their call sites. SessionPool checkout API replaced (`checkout() -> PoolGuard`)._

### Added

- **Graceful WebSocket / SSE drain on shutdown** (closes `specs/prod-readiness-v1.0.md` P0). `axum::serve.with_graceful_shutdown` only tracks the HTTP router — WebSocket upgrades and SSE `spawn_blocking` tasks used to outlive the signal, so clients lost their `Final` frame on deploy. New `CancellationToken` + `TaskTracker` cascade through every handler; on SIGTERM each live session flushes, emits an empty-if-needed `Final`, and closes with `Close(1001 Going Away)`. After `axum::serve` returns, `run_with_config` waits up to `shutdown_drain_secs` for the tracker to drain.
- **Wall-clock max-session cap** (closes `specs/prod-readiness-v1.0.md` P0). `idle_timeout` is reset on every frame, so a client that streams silence every 100 ms held a `SessionTriplet` forever. New `max_session_secs` limit closes the session with `Close(1008 Policy Violation)` + `Error { code: "max_session_duration_exceeded" }`. `0` disables the cap (not recommended).
- **CLI flags.**
  - `--max-session-secs` / `GIGASTT_MAX_SESSION_SECS` (default `3600`).
  - `--shutdown-drain-secs` / `GIGASTT_SHUTDOWN_DRAIN_SECS` (default `10`, clamped to `>= 1`).
- **`tests/e2e_shutdown.rs` re-enabled in CI** with four additional assertions: `test_shutdown_ws_emits_final_and_close`, `test_shutdown_sse_stream_terminates_cleanly`, `test_max_session_duration_cap`, and `test_shutdown_during_pool_saturation_returns_503_not_500`. The main-push e2e job now runs the full `--test e2e_rest --test e2e_ws --test e2e_errors --test e2e_shutdown` matrix.
- **`docs/runbook.md`** — rollback + on-call guidance for the new knobs.
- **`docs/deployment.md`** — `terminationGracePeriodSeconds` recommendation for k8s / docker-compose.

### Fixed

- **Per-IP rate-limiter math (`src/server/mod.rs`).** `(rate_limit_per_minute / 60).max(1)` truncated every value below 60 rpm to a 1 rps refill (= 60 rpm), so a defender setting `--rate-limit-per-minute 10` actually allowed 60 rpm — 6× weaker than declared. Switched to `tower_governor`'s `per_millisecond(60_000 / rpm)`, which preserves sub-second precision down to 1 rpm and clamps the upper bound at 60 000 rpm with a `warn!`. The startup log now includes the resolved `interval_ms` alongside `rpm` for diagnostics.
- **Session pool panic + unfairness (`src/inference/mod.rs`).** Replaced the `tokio::sync::mpsc::Receiver` behind a `tokio::sync::Mutex` with a lock-free `async_channel`. The new `Pool<T>` (alias `SessionPool = Pool<SessionTriplet>`) is FIFO under contention, exposes `close()` so graceful shutdown wakes every waiter with `PoolError::Closed` instead of panicking via `.expect("Pool sender dropped")`, and returns a `PoolGuard` whose `Drop` impl auto-checks-in the triplet on panic unwind. Server shutdown now wires `engine.pool.close()` into the shutdown future, and the REST handlers translate `PoolError::Closed` into a distinct 503 `pool_closed` response (separate from the 503 `timeout` for the 30 s checkout deadline).
- **REST oversized-body rejection is now strict 413** (`tests/e2e_errors.rs::test_rest_oversized_body_rejected`). Handlers in `src/server/http.rs` now add an explicit `body.len() > limits.body_limit_bytes → 413 payload_too_large` guard as defence-in-depth behind `DefaultBodyLimit`, and the e2e assertion upgrades from `!= 200` to a strict `== 413` with `code="payload_too_large"`.

### Changed

- `RuntimeLimits` gained `max_session_secs: u64` and `shutdown_drain_secs: u64`. External callers constructing `RuntimeLimits` literally will need to add the new fields (pre-1.0 minor bump — acceptable).
- `http::AppState` carries `shutdown: CancellationToken` and `tracker: TaskTracker`.
- `handle_ws_inner` switches from a bare `timeout(idle, source.next())` to a `biased;` `select!` with explicit cancel + deadline branches.
- `/v1/transcribe/stream` SSE task now runs on `TaskTracker::spawn_blocking` and polls the shutdown token between chunks so SIGTERM aborts long transcriptions instead of waiting them out.
- `SessionPool::{checkout, checkin, blocking_checkin}` replaced by `SessionPool::checkout() -> Result<PoolGuard, PoolError>`. The guard `Deref`s to `SessionTriplet` and auto-checks-in on drop. For `'static` consumers (`spawn_blocking`), call `guard.into_owned()` to get a `(SessionTriplet, OwnedReservation)` pair and return the triplet via `OwnedReservation::checkin(triplet)`.
- **Zero-copy REST upload decode path** (`src/inference/audio.rs`, `src/inference/mod.rs`, `src/server/http.rs`). The `/v1/transcribe` and `/v1/transcribe/stream` handlers used to call `body.to_vec()` on the incoming `axum::body::Bytes`, then `decode_audio_bytes` cloned that `Vec<u8>` into a `std::io::Cursor`, and symphonia decoded the PCM into another `Vec<f32>` — four concurrent copies of the upload were in RAM at peak. A 4× concurrent upload of a 10-minute WAV held ~1 GiB transiently and could OOM on a 1 GiB container. New path: `bytes::Bytes` flows end-to-end via a crate-private `BytesMediaSource` that implements `Read + Seek + MediaSource` directly on the refcounted buffer; new `decode_audio_bytes_shared(Bytes)` and `Engine::transcribe_bytes_shared(Bytes, _)` entry points. The legacy `decode_audio_bytes(&[u8])` / `Engine::transcribe_bytes(&[u8], _)` functions remain as thin shims (one `Bytes::copy_from_slice` for non-REST callers), so no public API breakage.
- **Incremental 10-minute duration cap** inside the decode loop. The check used to fire only after the full PCM buffer was assembled, so a malformed or hostile upload could still allocate hundreds of MiB before being rejected. Now each packet's samples are accumulated against a precomputed sample budget and the decoder bails out on the first packet that breaks the cap.

### Dependencies

- Promoted `tokio-util = { version = "0.7", features = ["rt"] }` from transitive to direct. Dev-deps gained `tracing-subscriber` so integration tests can surface server logs on failure.
- `async-channel = "2"` (transitive pieces — `concurrent-queue`, `event-listener` — were already in the graph).
- Added explicit `bytes = "1"` pin (previously transitive via `axum` / `tokio`) — makes the zero-copy contract between axum and symphonia visible in `Cargo.toml`.

## [0.8.1] - 2026-04-17

### Fixed

- **CoreML / CUDA startup crash on macOS 26+ (`Unable to serialize model as it contains compiled nodes`)** — `src/inference/mod.rs` previously called `.with_optimized_model_path(...)` after registering the CoreML / CUDA execution providers. Those EPs replace parts of the graph with compiled nodes that cannot be re-serialized as ONNX, so ORT aborted session creation before the server could bind. Regression introduced in v0.5.0. The optimized-ONNX cache path is removed from both EP paths; the CoreML block keeps its dedicated `coreml_cache/` (compiled-model cache) and the CUDA EP keeps its internal caches. Cost: ~1–2 s additional cold start. Benefit: `gigastt serve --features coreml` works again on macOS 14+.

## [0.8.0] - 2026-04-17

### Added

- **Prometheus `/metrics` endpoint** (closes `specs/todo.md` item 7). Enabled via `--metrics` (env `GIGASTT_METRICS=1`); off by default. Exposes
  - `gigastt_http_requests_total{method,path,status}` (counter)
  - `gigastt_http_request_duration_seconds{method,path}` (histogram).
  The endpoint sits behind the Origin allowlist and (when configured) the per-IP rate limiter. Recorder install is tolerant of double-install: emits a warning and keeps the server running instead of failing.
- **Per-IP rate limiting** (closes `specs/todo.md` item 17). `--rate-limit-per-minute N` (env `GIGASTT_RATE_LIMIT_PER_MINUTE`) + `--rate-limit-burst N` (env `GIGASTT_RATE_LIMIT_BURST`). Off by default. Applies to `/v1/*` and `/v1/ws`; `/health` is exempt. Implemented with `tower_governor` using `SmartIpKeyExtractor`. Returns 429 on violations. A background task evicts expired token buckets every 60 s.
- **`docs/deployment.md`** (closes `specs/todo.md` item 20). Reverse-proxy recipes for Caddy and nginx (certbot + `auth_basic`), Origin header behaviour, Docker binding strategy, health-check target, and a hardening checklist for remote deployments.

### Changed

- `ServerConfig` gained a `metrics_enabled: bool` field; `RuntimeLimits` gained `rate_limit_per_minute` + `rate_limit_burst`.
- `http::AppState` now carries `metrics_handle: Option<PrometheusHandle>`.
- The axum router splits into `/health` (public) and a `protected` sub-router for `/v1/*`, `/ws` alias, `/v1/ws`, and `/metrics` — rate limiter is layered on the protected branch only.

### Dependencies

- `tower_governor = "0.7"`
- `metrics = "0.24"`
- `metrics-exporter-prometheus = "0.17"` (default-features off)

## [0.7.2] - 2026-04-17

### Fixed

- **`cargo-deny` licenses + advisories** (`deny.toml`).
  - Added `CDLA-Permissive-2.0` to the license allowlist — `webpki-root-certs` (the Mozilla CA bundle) publishes under CDLA; behaves like MIT for our use.
  - Added `RUSTSEC-2021-0073` to `ignore` in `[advisories]`. `prost-types 0.6.1` is a build-time transitive of `onnx-pb` under the `quantize` feature; the affected `From<Timestamp> for SystemTime` path is not reached by our pipeline.

## [0.7.1] - 2026-04-17

### Fixed

- **`cargo-deny` CI job** (`.github/workflows/ci.yml`) — removed the trailing `arguments: licenses advisories bans sources`. The installed `cargo-deny` on stable-musl interpreted them as subcommands and failed with `unrecognized subcommand 'licenses'`. Default `check` already covers licenses, advisories, bans, and sources.

## [0.7.0] - 2026-04-17

### Added

- **Configurable runtime limits** (`gigastt::server::RuntimeLimits`, closes `specs/todo.md` item 6). Three knobs exposed via CLI + environment variables:
  - `--idle-timeout-secs` / `GIGASTT_IDLE_TIMEOUT_SECS` — WebSocket idle timeout (default 300).
  - `--ws-frame-max-bytes` / `GIGASTT_WS_FRAME_MAX_BYTES` — max WS frame / message (default 512 KiB).
  - `--body-limit-bytes` / `GIGASTT_BODY_LIMIT_BYTES` — max REST body (default 50 MiB).
  Delivered via a new `RuntimeLimits` field on `ServerConfig` and `http::AppState`; TOML config file support stays for a follow-up.
- **Canonical WebSocket path `/v1/ws`** (closes `specs/todo.md` item 11). Versioned path aligned with REST; legacy `/ws` remains as an alias with a warn-level deprecation log on every upgrade. Removal planned for v1.0.
- **`diarization` capability in `GET /v1/models`** (closes `specs/todo.md` item 12). Mirrors the WebSocket `Ready` field so clients can probe capabilities without opening a WS.
- **Docker `GIGASTT_BAKE_MODEL=1` build-arg** (closes `specs/todo.md` item 10). When set, a dedicated `model-fetcher` stage runs `gigastt download` during image build and the runtime stage copies the model into `/home/gigastt/.gigastt/models/`. Default (`0`) preserves the slim image.
- **`cargo deny check` in CI + `deny.toml`** (closes first half of `specs/todo.md` item 14 — SBOM stays for later). Enforces license allowlist + advisory scan + crates.io-only source + wildcard ban on every PR via `EmbarkStudios/cargo-deny-action@v2`.

### Changed

- `http::AppState` now carries `limits: RuntimeLimits` alongside `engine`; `handle_ws_inner` takes `&RuntimeLimits` so the idle timeout is no longer hard-coded.
- `DefaultBodyLimit::max` in the Axum router reads from `config.limits.body_limit_bytes` instead of the old `50 * 1024 * 1024` literal.
- `ws_handler` reads `ws_frame_max_bytes` from `AppState` instead of baking 512 KiB into the code path.

### Notes

- `RuntimeLimits`, `ServerConfig::local`, and `run_with_config` are public — downstream embedders can construct a fully customised server without CLI.
- Murmur remains on v0.6.0: no wire protocol break, new limits are additive + opt-in.

## [0.6.1] - 2026-04-17

### Changed

- **`handle_ws_inner` refactor** (`src/server/mod.rs`) — extracted three frame handlers (`handle_binary_frame`, `handle_configure_message`, `handle_stop_message`) and a `send_server_message` helper; the session loop is now ~60 lines with a single `FrameOutcome` dispatch. Behavior unchanged (same tests + clippy clean); reduces future risk when touching the hot path.

### Added

- **Integration test for origin middleware** (`src/server/mod.rs::tests::test_origin_middleware_integration`) — spins a minimal axum router with `origin_middleware` on a real port and verifies: `/health` is exempt, cross-origin `/v1/*` returns 403 `origin_denied`, loopback Origin is echoed into `Access-Control-Allow-Origin`, no-Origin requests pass through, and `localhost.evil.example.com` DNS-continuation attempts are denied.

## [0.6.0] - 2026-04-17

### Added

- **Origin allowlist middleware.** Cross-origin requests from non-loopback pages are denied by default across `/v1/*` and `/ws`; loopback origins (`localhost`, `127.0.0.1`, `[::1]`) always pass. New CLI flags:
  - `--allow-origin <URL>` (repeatable) — exact-match, case-insensitive Origin allowlist.
  - `--cors-allow-any` — legacy `Access-Control-Allow-Origin: *` behaviour, opt-in.
  `/health` remains free of Origin checks for monitoring / Docker `HEALTHCHECK`.
- **`--bind-all` guard.** `gigastt serve` refuses non-loopback `--host` values (`0.0.0.0`, LAN IPs, …) unless `--bind-all` is passed or `GIGASTT_ALLOW_BIND_ANY=1` is set. Both `Dockerfile` and `Dockerfile.cuda` now pass `--bind-all` in their `CMD` line.
- **`Retry-After` on pool saturation.**
  - REST `/v1/transcribe` and `/v1/transcribe/stream` return HTTP 503 with a `Retry-After: 30` header (RFC 9110 §10.2.3) and `retry_after_ms: 30000` in the JSON body.
  - WebSocket `ServerMessage::Error` gained an optional `retry_after_ms` field (omitted from JSON when absent, preserving backward compatibility); the pool-timeout path at connect emits `retry_after_ms: 30000`.
- **`gigastt::server::{ServerConfig, OriginPolicy, run_with_config}`** — new public API for programmatic startup with explicit origin policy.

### Changed

- **Default cross-origin posture is now deny.** Previous behaviour (wildcard CORS + non-local Origin only warned about) is preserved behind `--cors-allow-any`. Browser integrations hitting the server from a non-loopback page must either add their origin via `--allow-origin` or run the server with `--cors-allow-any`.

### Security

- Closes `specs/todo.md` P1 items 4, 5, 8, 9. Reduces the risk that a malicious webpage can drive-by-connect to the local transcription server and exfiltrate microphone audio.

## [0.5.3] - 2026-04-17

### Security

- **`rustls-webpki` 0.103.10 → 0.103.12** (`Cargo.lock`) — resolves RUSTSEC-2026-0098 (name constraints for URI names incorrectly accepted) and RUSTSEC-2026-0099 (name constraints accepted for certificates asserting a wildcard name). Pulled in transitively via `reqwest → hyper-rustls → rustls`.

## [0.5.2] - 2026-04-17

### Fixed

- **CI clippy** (`src/model/mod.rs:29`) — replaced manual `if self.total > 0` division guard with `checked_div`, satisfying Rust 1.95's new `clippy::manual_checked_ops` lint that broke CI on v0.5.1.
- **Release workflow** (`.github/workflows/release.yml`) — removed the `linux-x86_64-cuda` matrix entry: `Jimver/cuda-toolkit@v0.2.19` cannot resolve the `cuda-nvcc-12-4` / `cuda-cudart-12-4` packages on `ubuntu-latest`. Tracked for re-enabling in `specs/todo.md`. Until then CUDA users build from source.

## [0.5.1] - 2026-04-17

### Added

- **Release automation** (`.github/workflows/release.yml`) — tag-triggered matrix workflow that produces `gigastt-<ver>-aarch64-apple-darwin.tar.gz` (coreml), `gigastt-<ver>-x86_64-unknown-linux-gnu.tar.gz` (cpu), `gigastt-<ver>-x86_64-unknown-linux-gnu-cuda.tar.gz`, per-asset `.sha256` files, and aggregated `SHA256SUMS.txt`. Replaces ad-hoc manual uploads that previously broke SHA-pinned downstream clients.
- **`CONTRIBUTING.md`** — release checklist and contribution guidelines, including an explicit prohibition on manual `gh release upload` of binary assets.
- **`examples/bun_client.ts`, `examples/go_client.go`, `examples/KotlinClient.kt`** — WebSocket client samples in Go, Kotlin (OkHttp), and Bun-native TypeScript.
- **`specs/todo.md` + `specs/plan.md`** — 20-item follow-up list from the v0.5.0 critique, ranked P0/P1/P2 and sequenced into six phases through v1.0.0.

### Fixed

- **WebSocket pool recovery after inference panic** (`src/server/mod.rs`) — a panic inside `process_chunk` used to leak the `SessionTriplet` and permanently shrink the pool. Now the blocking task owns the state and triplet, wraps the inner call in `catch_unwind(AssertUnwindSafe(_))`, and returns both unconditionally. On panic the WS session sends an `inference_panic` error, resets its streaming state, and continues instead of tearing down.
- **`clippy::never_loop`** in `tests/e2e_errors.rs` (two occurrences) — replaced the single-iteration `while let` drains with a `tokio::time::timeout(_).await` call, unblocking stricter lint levels.

### Removed

- **`scripts/quantize.py`** — superseded by native Rust quantization (`gigastt quantize --features quantize`).
- **`examples/js_client.mjs`** — replaced by `examples/bun_client.ts`.

## [0.5.0] - 2026-04-13

### Added

- **Native Rust INT8 quantization** (`--features quantize`) — `gigastt quantize` command replaces `scripts/quantize.py`. Per-channel symmetric QDQ format, hardened against shared weights and malformed tensors.
- **Auto-quantize on download/serve** — automatically creates INT8 encoder when built with `--features quantize`. Prints hint otherwise.
- **`GET /v1/models` endpoint** — returns model info: encoder type (int8/fp32), vocab size, pool status, supported formats and sample rates.
- **`--log-level` CLI option** — global flag for all commands (`gigastt --log-level debug serve`), replaces `RUST_LOG`-only config.
- **`--pool-size` CLI option** — configurable concurrent inference sessions for `serve` command.
- **`Engine::is_int8()`** method exposes encoder quantization status.
- **PrepackedWeights** — shared ONNX Runtime weight memory across session pool (reduced memory footprint).
- **Inference instrumentation** — encoder/decoder timing logged at info level.
- **Russian README** (`README_RU.md`) with language switcher.
- **CI `cargo fmt --check`** job for format enforcement.

### Changed

- **WER benchmark** verified on 993 Golos samples (4991 words): FP32 10.5%, INT8 10.4% — 0% degradation confirmed.
- **README** updated with verified metrics: WER 10.4%, latency ~700ms, memory ~560MB. Expanded comparison table.
- **Decoder optimization** — cached decoder output during blank runs (86% decoder call reduction).
- **Optimized model cache** directory for pre-compiled ONNX models.

### Fixed

- **Server hardening** — WS pool checkout timeout (30s), REST `catch_unwind` for panic recovery, removed `unwrap`/`expect` in handlers.
- **Security** — upgraded `tokio-tungstenite` 0.24→0.28, resolving RUSTSEC-2026-0097 (`rand` 0.8.5 unsoundness).
- **CI stability** — e2e tests serialized with `--test-threads=1` (prevents OOM), shutdown tests excluded (require graceful connection termination), SSE tests resilient to non-speech audio.
- **Benchmark overflow** — `number_to_words` handles numbers > 999,999.
- **Dockerfiles** updated to Rust 1.85+ for edition 2024 support.
- **Audio decode refactor** — extracted shared inner function, eliminated ~80 line duplication.

## [0.4.3] - 2026-04-13

### Added

- **Comprehensive e2e test infrastructure** — 28 new tests across 7 files:
  - `tests/e2e_rest.rs` (8 tests): health, transcribe, SSE streaming, error paths
  - `tests/e2e_ws.rs` (9 tests): WebSocket protocol — ready, audio, stop, configure, malformed JSON, disconnect, concurrency
  - `tests/e2e_errors.rs` (5 tests): oversized body/frame rejection, pool saturation (503), idle timeout
  - `tests/e2e_shutdown.rs` (2 tests): graceful shutdown during active WS/SSE sessions
  - `tests/load_test.rs` (3 tests): 4 concurrent WS/REST, burst 20 connections
  - `tests/soak_test.rs` (1 test): continuous WS cycling (configurable via `GIGASTT_SOAK_DURATION_SECS`)
- **Shared test helpers** (`tests/common/mod.rs`): `start_server` with clean shutdown, `wait_for_ready` with exponential backoff, WAV generation, WebSocket connect helpers.
- **`server::run_with_shutdown()`** — accepts optional `oneshot::Receiver<()>` for programmatic server shutdown (used by tests; `run()` unchanged).
- **CI feature matrix** — split into 7 jobs: clippy, unit tests, build-coreml, build-cuda, build-diarization, e2e tests (main push only with cached model), security audit.

### Changed

- CI workflow restructured: PRs get fast feedback (unit + clippy + feature builds), main push adds full e2e suite with ~850MB cached model (OS-independent cache key).

## [0.4.2] - 2026-04-13

### Removed

- **`dirs` dependency** — replaced with `env::var("HOME")` / `USERPROFILE` (~10 lines).
- **`indicatif` dependency** — replaced with simple stderr progress output (~50 transitive deps removed).
- **`tempfile` from production deps** — HTTP handlers decode audio from memory via `Cursor<Vec<u8>>` (faster, no disk I/O). Kept in dev-dependencies for tests.
- **`async-stream` dependency** — replaced with `futures_util::stream::unfold`.
- **`tower-http` dependency** — replaced with axum's built-in `DefaultBodyLimit`.

### Added

- `decode_audio_bytes()` — decode audio from in-memory bytes without temp files.
- `Engine::transcribe_bytes()` — transcribe from byte buffer directly.
- Security audit job in CI workflow (`cargo audit`).
- Non-root user in Dockerfiles (hardened containers).

## [0.4.1] - 2026-04-13

### Changed

- Diarization module no longer depends on internal `ort_err()` helper — uses `anyhow::Context` instead. Module is now self-contained and ready for future crate extraction.

### Fixed

- Centroid re-normalization after running average update (prevents speaker clustering drift).
- Semaphore timeout (30s) on HTTP endpoints prevents DoS via hanging requests.
- SSE semaphore permit held for stream lifetime (was dropped before stream consumed).
- SSE inference wrapped in `spawn_blocking` (no longer blocks async runtime).
- Error messages sanitized at HTTP API boundary (no internal path/model leakage).
- Speaker count capped at 64 (`MAX_SPEAKERS`) with graceful fallback.
- Cosine similarity zero-norm check uses epsilon (1e-8) instead of exact float equality.
- Request body dropped after temp file write (reduces peak memory ~2x for large files).
- Configure message rejected after first audio frame (`configure_too_late` error).
- WebSocket idle timeout (300s) disconnects silent clients.
- Unnecessary `samples_16k_copy` allocation skipped when diarization disabled at runtime.
- Async `tokio::fs::write` replaces blocking `std::fs::write` in HTTP handlers.
- `tokio-tungstenite` moved to dev-dependencies (unused in production code).
- `hound` dependency removed (unused).
- CLAUDE.md and README.md updated: test counts, architecture tree, WebSocket URL `/ws`, REST API docs, version references.

## [0.4.0] - 2026-04-13

### Added

- **Cross-platform support** via compile-time Cargo feature flags:
  - `--features coreml`: macOS ARM64 (CoreML + Neural Engine) — existing behavior
  - `--features cuda`: Linux x86_64 (NVIDIA CUDA 12+)
  - Default (no features): CPU-only, compiles on any platform
  - `compile_error!` guard prevents enabling both `coreml` and `cuda`
- **Flexible sample rate**: `ClientMessage::Configure { sample_rate }` lets clients declare input rate (8kHz, 16kHz, 24kHz, 44.1kHz, 48kHz). Default 48kHz for backward compatibility.
- **Polyphase FIR resampler** (rubato `SincFixedIn`) replaces linear interpolation — significantly better audio quality.
- **`ServerMessage::Ready`** extended with `supported_rates` field (list of accepted sample rates).
- **HTTP REST API** via axum (single port serves HTTP + WebSocket):
  - `GET /health` — health check for monitoring and Docker HEALTHCHECK
  - `POST /v1/transcribe` — upload audio file, receive full JSON transcript
  - `POST /v1/transcribe/stream` — upload audio file, receive SSE stream of partial/final results
  - `GET /ws` — WebSocket streaming (existing protocol, new path)
- **Speaker diarization** (optional, `--features diarization`):
  - WeSpeaker ResNet34 ONNX model (26.5MB, 256-dim embeddings, 16kHz)
  - Online incremental clustering (cosine similarity, configurable threshold)
  - `WordInfo.speaker: Option<u32>` field identifies speakers per word
  - `Configure { diarization: true }` enables per-session
  - `gigastt download --diarization` fetches speaker model separately
  - MAX_SPEAKERS=64 cap with graceful fallback to closest match
- **`Dockerfile.cuda`** — multi-stage CUDA build with `nvidia/cuda:12.6.3-cudnn-runtime`
- **GitHub Actions CI** matrix: macOS (CoreML) + Linux (CPU) in parallel
- **Semaphore timeout** (30s) on HTTP endpoints prevents DoS via hanging requests
- **WebSocket idle timeout** (300s) disconnects silent clients
- **Configure guard** — server rejects `Configure` after first audio frame

### Changed

- **Server migrated from raw tokio-tungstenite to axum** — single port serves HTTP routes + WebSocket upgrade
- **WebSocket endpoint moved to `/ws`** (was root `/`). Clients must connect to `ws://host:port/ws`.
- **`ClientMessage::Configure.sample_rate`** changed from `u32` to `Option<u32>` to support partial configuration (sample rate only, diarization only, or both).
- **Dockerfile** CPU build no longer uses `--no-default-features` (default features are now empty = CPU).
- **SSE inference** runs in `spawn_blocking` (no longer blocks async runtime).
- **Error responses** in HTTP handlers sanitized — generic messages to clients, details logged server-side.
- `tokio-tungstenite` moved from production to dev-dependencies (only used in integration tests).
- `hound` dependency removed (unused; all audio decoding via symphonia).

### Fixed

- **Centroid drift in speaker clustering** — centroids re-normalized after running average update.
- **Cosine similarity** zero-norm check uses epsilon (1e-8) instead of exact float equality.
- **SSE semaphore permit** held for stream lifetime (was dropped before stream consumed).
- **HTTP body memory** — request body dropped after temp file write, reducing peak memory usage.
- **Async file I/O** — `tokio::fs::write` replaces blocking `std::fs::write` in HTTP handlers.

### Breaking Changes

- WebSocket path changed: `/` → `/ws`. Update client connection URLs.
- `Configure.sample_rate` type changed: `u32` → `Option<u32>`. Existing JSON `{"type":"configure","sample_rate":8000}` still works via `#[serde(default)]`.
- Default `cargo build` (no features) now produces CPU-only binary. macOS users must explicitly add `--features coreml`.

## [0.3.0] - 2026-04-12

### Added

- `GigasttError` enum (`error` module) with variants: `ModelLoad`, `Inference`, `InvalidAudio`, `Io` — enables `match`-based error handling.
- `#[non_exhaustive]` on all public structs and enums — future additions are non-breaking.
- Comprehensive `///` rustdoc on all public types, functions, fields, and constants.
- Crate-level documentation with quick-start examples in `lib.rs`.
- Stress tests for NaN/infinity audio samples, empty inputs, and buffer boundary conditions.

### Fixed

- Potential panic on odd-length WebSocket binary frames (`chunks_exact(2)` now drops trailing byte with warning).
- Non-finite audio samples (NaN, infinity) in `resample()` replaced with zeros instead of propagating.

### Breaking Changes

- `Engine::load()`, `Engine::process_chunk()`, and `Engine::transcribe_file()` return `Result<T, GigasttError>` instead of `anyhow::Result<T>`.
- All public structs/enums are `#[non_exhaustive]` — external struct literal construction requires constructor methods.

## [0.2.0] - 2026-04-06

### Added

- Partial transcripts with real-time streaming via WebSocket.
- Endpointing detection (~600ms silence triggers finalization).
- Per-word timestamps (`WordInfo.start`, `WordInfo.end`) relative to stream start.
- Per-word confidence scores (`WordInfo.confidence`) averaged over BPE tokens.
- CoreML execution provider for macOS ARM64 (Neural Engine + CPU).
- INT8 quantized encoder support (`v3_e2e_rnnt_encoder_int8.onnx`, ~4x smaller, ~43% faster).
- CoreML model cache directory (`~/.gigastt/models/coreml_cache/`).
- Docker multi-stage build (`Dockerfile`).
- Python quantization script (`scripts/quantize.py`).

### Changed

- Audio pipeline: accept 48kHz from WebSocket clients, resample to 16kHz internally.
- Encoder output shape handling: channels-first `[1, 768, T]` format.

## [0.1.2] - 2026-04-01

### Added

- GigaAM v3 e2e_rnnt inference engine with ONNX Runtime.
- WebSocket server (tokio + tungstenite) for streaming audio.
- CLI: `serve`, `download`, `transcribe` commands.
- HuggingFace model auto-download (`istupakov/gigaam-v3-onnx`).
- BPE tokenizer (1025 tokens).
- Mel spectrogram (64 bins, FFT=320, hop=160, HTK).
- RNN-T greedy decode loop.
- Multi-format audio support: WAV, MP3, M4A/AAC, OGG/Vorbis, FLAC (via symphonia).
- 39 unit tests (tokenizer, features, decode, inference, protocol).

[Unreleased]: https://github.com/ekhodzitsky/gigastt/compare/v2.11.3...HEAD
[2.11.3]: https://github.com/ekhodzitsky/gigastt/compare/v2.11.2...v2.11.3
[2.11.2]: https://github.com/ekhodzitsky/gigastt/compare/v2.11.1...v2.11.2
[2.11.1]: https://github.com/ekhodzitsky/gigastt/compare/v2.11.0...v2.11.1
[2.11.0]: https://github.com/ekhodzitsky/gigastt/compare/v2.10.0...v2.11.0
[2.10.0]: https://github.com/ekhodzitsky/gigastt/compare/v2.9.0...v2.10.0
[2.9.0]: https://github.com/ekhodzitsky/gigastt/compare/v2.8.0...v2.9.0
[2.8.0]: https://github.com/ekhodzitsky/gigastt/compare/v2.7.0...v2.8.0
[2.7.0]: https://github.com/ekhodzitsky/gigastt/compare/v2.6.0...v2.7.0
[2.6.0]: https://github.com/ekhodzitsky/gigastt/compare/v2.5.0...v2.6.0
[2.5.0]: https://github.com/ekhodzitsky/gigastt/compare/v2.4.0...v2.5.0
[2.4.0]: https://github.com/ekhodzitsky/gigastt/compare/v2.3.0...v2.4.0
[2.3.0]: https://github.com/ekhodzitsky/gigastt/compare/v2.1.0...v2.3.0
[2.1.0]: https://github.com/ekhodzitsky/gigastt/compare/v2.0.14...v2.1.0
[2.0.14]: https://github.com/ekhodzitsky/gigastt/compare/v2.0.13...v2.0.14
[2.0.13]: https://github.com/ekhodzitsky/gigastt/compare/v2.0.12...v2.0.13
[2.0.12]: https://github.com/ekhodzitsky/gigastt/compare/v2.0.11...v2.0.12
[2.0.11]: https://github.com/ekhodzitsky/gigastt/compare/v2.0.10...v2.0.11
[2.0.10]: https://github.com/ekhodzitsky/gigastt/compare/v2.0.9...v2.0.10
[2.0.3]: https://github.com/ekhodzitsky/gigastt/compare/v2.0.2...v2.0.3
[2.0.2]: https://github.com/ekhodzitsky/gigastt/compare/v2.0.1...v2.0.2
[2.0.1]: https://github.com/ekhodzitsky/gigastt/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/ekhodzitsky/gigastt/compare/v1.0.1...v2.0.0
[1.0.1]: https://github.com/ekhodzitsky/gigastt/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/ekhodzitsky/gigastt/compare/v0.10.0...v1.0.0
[0.10.0]: https://github.com/ekhodzitsky/gigastt/compare/v0.9.6...v0.10.0
[0.9.6]: https://github.com/ekhodzitsky/gigastt/compare/v0.9.5...v0.9.6
[0.9.4]: https://github.com/ekhodzitsky/gigastt/compare/v0.9.3...v0.9.4
[0.9.3]: https://github.com/ekhodzitsky/gigastt/compare/v0.9.2...v0.9.3
[0.9.2]: https://github.com/ekhodzitsky/gigastt/compare/v0.9.1...v0.9.2
[0.9.1]: https://github.com/ekhodzitsky/gigastt/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/ekhodzitsky/gigastt/compare/v0.9.0-rc.2...v0.9.0
[0.9.0-rc.2]: https://github.com/ekhodzitsky/gigastt/compare/v0.9.0-rc.1...v0.9.0-rc.2
[0.9.0-rc.1]: https://github.com/ekhodzitsky/gigastt/compare/v0.8.1...v0.9.0-rc.1
[0.8.1]: https://github.com/ekhodzitsky/gigastt/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/ekhodzitsky/gigastt/compare/v0.7.2...v0.8.0
[0.7.2]: https://github.com/ekhodzitsky/gigastt/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/ekhodzitsky/gigastt/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/ekhodzitsky/gigastt/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/ekhodzitsky/gigastt/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/ekhodzitsky/gigastt/compare/v0.5.3...v0.6.0
[0.5.3]: https://github.com/ekhodzitsky/gigastt/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/ekhodzitsky/gigastt/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/ekhodzitsky/gigastt/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/ekhodzitsky/gigastt/compare/v0.4.3...v0.5.0
[0.4.3]: https://github.com/ekhodzitsky/gigastt/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/ekhodzitsky/gigastt/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/ekhodzitsky/gigastt/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/ekhodzitsky/gigastt/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ekhodzitsky/gigastt/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ekhodzitsky/gigastt/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/ekhodzitsky/gigastt/releases/tag/v0.1.2
