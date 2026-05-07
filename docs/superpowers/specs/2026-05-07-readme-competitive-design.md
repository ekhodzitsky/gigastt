# README Competitive Update — Design Spec

**Date:** 2026-05-07  
**Sub-project:** 1 of 3 (README → Workspace split → Python bindings)  
**Scope:** Update README.md and README_RU.md with competitive analysis, fix stale data

---

## Goal

Update the README to reflect v1.0.2 status, expand the competitive comparison from a 3-column table (gigastt / Whisper / Cloud) to a 6-column table covering the real competitive landscape, and fix stale information (test count, version badge, removed endpoints).

## Changes

### 1. Version badge

**Current:** `v0.9.4`  
**New:** Remove hardcoded version string. The crates.io badge already shows the latest version dynamically. Only update the `<sub>Latest:` line to say `v1.0.2`.

### 2. Competitive table ("Why gigastt?")

Expand from 3 columns to 6: **gigastt**, **whisper.cpp**, **faster-whisper**, **Vosk**, **sherpa-onnx**, **Cloud APIs**.

Rows:

| Row | gigastt | whisper.cpp | faster-whisper | Vosk | sherpa-onnx | Cloud APIs |
|---|---|---|---|---|---|---|
| Model | GigaAM v3 | Whisper large-v3 | Whisper large-v3 | Vosk models | varies (Whisper, Zipformer…) | vendor |
| WER (Russian) | **10.4%** | ~18% | ~18% | ~20%+ | model-dependent | 5–10% |
| Language | Russian only | 99 languages | 99 languages | 20+ languages | 10+ languages | 100+ |
| Streaming | real-time WS | — | — | WS + gRPC | WS + TCP | varies |
| Latency (16s, M1) | ~700ms | ~4s | ~2s (CTranslate2) | ~3s | ~1.5s | network |
| Privacy | 100% local | 100% local | 100% local | 100% local | 100% local | data leaves device |
| Setup | `cargo install` | cmake + make | `pip install` | `pip install` | cmake or pip | API key + billing |
| Language (impl) | Rust | C/C++ | Python/C++ | C++/Java | C++ | N/A |
| Bindings | Rust, C FFI | C, Python, Go, JS… | Python | Python, Java, JS, Go… | C, Python, Java, Swift, Go… | SDK per vendor |
| INT8 quantization | auto, 0% WER loss | GGML quant | CTranslate2 quant | — | — | N/A |
| Concurrent sessions | configurable pool | 1 | 1 | 1 | 1 | provider limits |
| Cost | free | free | free | free | free | $0.006/min+ |

Add a note below the table:

> **Trade-off:** gigastt supports Russian only. If you need multilingual recognition, consider whisper.cpp or sherpa-onnx. If you need the best Russian accuracy locally — gigastt is the only Rust-native option built on GigaAM v3, the current SOTA for Russian ASR.

### 3. API table

**Remove:** `/ws` deprecated alias row (route already deleted in v1.0)  
**Add:** `/ready` — readiness probe (returns 200 when engine pool is initialized)  
**Update:** Remove "canonical" note from `/v1/ws` description (no longer needed without alias)

New table:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check (`{"status":"ok"}`) |
| `/ready` | GET | Readiness probe (200 when engine pool is ready) |
| `/v1/models` | GET | Model info (encoder type, pool size, capabilities) |
| `/v1/transcribe` | POST | File transcription, full JSON response |
| `/v1/transcribe/stream` | POST | File transcription with SSE streaming |
| `/v1/ws` | GET | WebSocket upgrade for real-time streaming |
| `/metrics` | GET | Prometheus metrics (enabled with `--metrics`) |

### 4. Testing section

**Current:** "125 unit tests + 30 e2e tests"  
**New:** "153 unit tests + 30 e2e tests + load & soak tests + WER benchmark"

Update the command example to show 153 instead of 125. Add mention of `e2e_rate_limit` test suite and proptest-based property tests.

### 5. No changes

These sections stay as-is:
- WER 10.4% figure (still accurate)
- Architecture diagram
- Client examples (Python, Bun, Go, Kotlin)
- Performance table
- Hardware acceleration section
- INT8 quantization section
- Security section
- Model section
- CLI reference
- Troubleshooting
- Android/FFI section

### 6. README_RU.md

Apply the same changes to the Russian version. Translate any new English text to Russian.

## Out of scope

- Workspace split (sub-project #2)
- Python bindings (sub-project #3)
- Bindings row in competitive table will say "Rust, C FFI" (current state); future bindings will update this when implemented

## Success criteria

- `v0.9.4` → `v1.0.2` in the Latest line
- 6-column competitive table renders correctly in GitHub markdown
- `/ws` row removed, `/ready` row added in API table
- Test count says 153
- Both README.md and README_RU.md updated consistently
- No broken links or formatting issues
