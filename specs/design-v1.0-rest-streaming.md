# Design: V1-05 — REST body streaming path

Owner: architect-2 (2026-04-20). Addresses the P0 finding that
`/v1/transcribe` peaks at ~515 MiB per upload on a 50 MiB body cap.
Target tag: `v0.9.0`. Implements **V1-05** from
`specs/prod-readiness-v1.0.md`.

## 1. Strategy — two layers

Problem: three independent copies between `axum::body::Bytes` and symphonia
decode, plus an unbounded `Vec<f32>` for full PCM. 4 concurrent 50 MiB
uploads ≈ 1.06 GiB transient → OOM on 4 GB boxes.

### Layer A (ship now — tactical, low-risk)
Remove the three redundant copies from body to symphonia:
1. `Bytes` is already heap-allocated and ref-counted. Pass directly into
   `transcribe_bytes_shared(Bytes)`.
2. Wrap `Bytes` in a custom `BytesMediaSource { data: Bytes, pos: u64 }`
   implementing `Read + Seek + MediaSource` — zero-copy reads.
3. Drop `body.to_vec()` in both REST and SSE handlers, drop `data.to_vec()`
   in `decode_audio_bytes`.

Expected impact: 3× → 1× peak for body buffer (still `Vec<f32>`-bound).

### Layer B (deferred — streaming upload)
`tokio_util::io::StreamReader` + `SyncIoBridge` on `axum::body::Body` would
allow chunk-by-chunk symphonia decode. **Problem:** symphonia requires
`Seek` for probe; M4A (MOOV atom at EOF) and indexed FLAC break without
random access. Limit streaming to WAV/OGG or skip for v0.9.0. Recommended:
**defer to v1.x** — Layer A alone brings 4 × 50 MiB body/RSS under 220 MiB.

## 2. API surface

### New in `src/inference/audio.rs`
```rust
pub fn decode_audio_bytes_shared(data: bytes::Bytes) -> Result<Vec<f32>>;

pub(crate) fn decode_audio_from_source<S: MediaSource + 'static>(
    source: S, hint: Hint, source_label: &str,
) -> Result<Vec<f32>>;
```
Keep `decode_audio_bytes(&[u8])` as a thin shim:
`decode_audio_bytes_shared(Bytes::copy_from_slice(data))`.

### `src/inference/mod.rs`
```rust
pub fn transcribe_bytes_shared(
    &self, data: bytes::Bytes, triplet: &mut SessionTriplet,
) -> Result<TranscribeResult, GigasttError>;
```
`transcribe_bytes(&[u8])` remains as shim → minor (`0.8.1 → 0.9.0`) rather
than major. No new `GigasttError` variants; `InvalidAudio`/`Io` cover it.

## 3. `BytesMediaSource` skeleton

```rust
pub struct BytesMediaSource { data: Bytes, pos: u64 }

impl Read for BytesMediaSource { /* slice copy from data[pos..] */ }
impl Seek for BytesMediaSource { /* Start/End/Current, reject negative */ }
impl MediaSource for BytesMediaSource {
    fn is_seekable(&self) -> bool { true }
    fn byte_len(&self) -> Option<u64> { Some(self.data.len() as u64) }
}
```

## 4. REST handler (Layer A)

```rust
pub async fn transcribe(
    State(state): State<Arc<AppState>>, body: Bytes,
) -> Result<Json<TranscribeResponse>, ApiError> {
    if body.is_empty() { return Err(api_error(BAD_REQUEST, "Empty body", "empty_body")); }
    // defence-in-depth (DefaultBodyLimit already rejects Content-Length > limit)
    if body.len() > state.limits.body_limit_bytes {
        return Err(api_error(PAYLOAD_TOO_LARGE, "Body exceeds limit", "payload_too_large"));
    }
    let triplet = tokio::time::timeout(Duration::from_secs(30), state.engine.pool.checkout())
        .await.map_err(|_| api_timeout_error())?;
    let engine = state.engine.clone();
    let body_for_task = body; // cheap clone — refcounted
    tokio::task::spawn_blocking(move || {
        let mut triplet = triplet;
        let r = std::panic::catch_unwind(AssertUnwindSafe(||
            engine.transcribe_bytes_shared(body_for_task, &mut triplet)));
        // (existing panic-safe wiring)
    }).await
    // ...
}
```

## 5. Duration cap refactor

Current (`src/inference/audio.rs:131-144`): duration check after full
decode into `Vec<f32>` — late abort wastes ~138 MiB for a 12-minute file.

Change: increment `total_samples` inside the decode loop and abort when
`(total_samples / sample_rate) as f64 > MAX_DURATION_S`.

## 6. Edge cases

| Case | Behaviour |
|---|---|
| Chunked upload without Content-Length | `DefaultBodyLimit` accumulates and rejects when threshold is crossed |
| REST client disconnect mid-upload | `body: Bytes` is fully buffered before the handler runs; axum rejects without invoking us |
| SSE client disconnect mid-stream | `tx.blocking_send(...).is_err()` → early return; triplet checkin via `blocking_checkin` |
| Partial/invalid audio | symphonia returns error → 422 `invalid_audio` |
| Input > `file_max_minutes` | **Bug today**: full buffer first. Fix via loop-level abort. |
| Malformed header, no audio track | 422 `invalid_audio` |
| 4 concurrent 10 min WAV @ 48 kHz | Layer A: ≈ 660 MiB total (vs 1.06 GiB today); symphonia ring buffer ~64 KiB is negligible |

## 7. Perf/memory acceptance criteria

| Metric | Target |
|---|---|
| Peak RSS delta for 1×10 min WAV | ≤ 1.3× WAV size |
| 4 concurrent 10 min uploads | ≤ 4× single |
| Latency for 1 KiB WAV | within ± 5 % of baseline |
| Content-Length 51 MiB early reject | < 10 ms (no body read) |
| `Vec<f32>` reallocations per decode | 0 (when `n_frames` known) or ≤ 2 |

Regression gate: add `test_rest_memory_budget` under `tests/load_test.rs`.

## 8. Tests

### Unit (`src/inference/audio.rs`)
- `bytes_media_source_read_full`
- `bytes_media_source_seek_end`
- `bytes_media_source_seek_past_end_ok`
- `bytes_media_source_seek_before_start_err`
- `bytes_media_source_partial_read_progress`
- `bytes_media_source_byte_len_matches`
- `decode_audio_bytes_shared_refcount_does_not_copy` — `Bytes::strong_count`
- `decode_audio_shim_matches_shared` — equivalence oracle

### Integration (`tests/e2e_rest.rs`)
- `test_rest_large_body_rss_within_budget` (Linux: `/proc/self/status`,
  macOS: `libproc`; skip on unsupported platforms)
- `test_rest_concurrent_upload_rss_linear`

### Integration (`tests/e2e_errors.rs`)
- **Fix V1-22**: change `assert_ne!(..., 200)` to
  `assert_eq!(response.status().as_u16(), 413)` + assert body code.
- `test_rest_content_length_reject_no_body_read` — 100 GiB
  `Content-Length` → 413 in < 100 ms.
- `test_sse_client_disconnect_no_orphan_task` — confirm
  `pool.available()` recovers after disconnect.

### Regression
- `test_decode_duration_cap_streaming` — 12 min WAV, expect
  `InvalidAudio("too long")` while staying under 5 MiB of allocation.

## 9. Format compatibility (Layer B preview)

| Format | True streaming | Notes |
|---|---|---|
| WAV (PCM) | yes | header in first 44 bytes |
| OGG/Vorbis | yes | page-based |
| MP3 | partial | VBR header seeks common, usually OK |
| FLAC | partial | SEEKTABLE may request seek |
| M4A/AAC | no | MOOV atom often at EOF |

Reinforces the Layer B deferral.

## 10. Rollback

Layer A is behaviour-identical; no feature flag needed. Layer B (when we
ship it) guards via `GIGASTT_REST_STREAMING=0`.

## 11. Effort breakdown

| # | Commit | Hours |
|---|---|---|
| 1 | `feat(audio): BytesMediaSource + decode_audio_bytes_shared` | 2–3 |
| 2 | `feat(inference): Engine::transcribe_bytes_shared(Bytes)` | 1 |
| 3 | `perf(http): zero-copy body path in /v1/transcribe` | 1 |
| 4 | `perf(http): zero-copy body path in /v1/transcribe/stream` | 1 |
| 5 | `fix(audio): early duration-cap abort in decode loop` | 2 |
| 6 | `test(e2e): tighten 413 assert + memory budget test` | 2–3 |
| 7 | `docs(changelog): V1-05 memory reduction` | 0.5 |

Total ≈ 9.5–11.5 h; ~1.5 engineer-days. Layer B adds 8–12 h if taken on.

## 12. Open questions

1. Keep both `transcribe_bytes(&[u8])` and `transcribe_bytes_shared(Bytes)`
   shims, or single-sig break? Minor bump preferred.
2. Ship Layer B in this cycle? Recommended: no.
3. Early duration cap changes visible behaviour for corrupt-tail files
   (`"too long"` vs `"decode error"`) — acceptable?
4. Add `bytes = "1"` to `Cargo.toml` explicitly for API stability?
5. Fixtures for MP3/OGG/FLAC/M4A decode — needed before any Layer B work.

## 13. References

- `src/server/http.rs:163-184`, `243-273` — handlers with `body.to_vec()`.
- `src/inference/audio.rs:57-62` — `decode_audio_bytes` (`data.to_vec()`).
- `src/inference/audio.rs:91-128` — unbounded `Vec<f32>` grow.
- `src/inference/audio.rs:131-144` — late duration check.
- `src/inference/mod.rs:684-710` — `Engine::transcribe_bytes(&[u8])`.
- `src/server/mod.rs:312` — `DefaultBodyLimit::max(body_limit_bytes)`.
- `src/server/mod.rs:131` — `body_limit_bytes = 50 MiB`.
- `tests/e2e_errors.rs:37-41` — weak `!= 200` assertion.
- `Cargo.toml:26` — axum 0.8 (`Body::into_data_stream()` available).
- `Cargo.toml:59` — symphonia 0.5 feature set.
