# Privacy

gigastt is designed to keep all speech recognition entirely on the device that
runs it. This document describes precisely what data moves where and what is
retained.

## Audio and transcript data

- All audio processing runs locally via ONNX Runtime. Audio frames never leave
  the machine.
- Transcripts are returned only to the caller of the local API. They are not
  stored, written to disk, or logged.
- Tracing logs (controlled by `RUST_LOG`) record request metadata such as
  duration and word count. They do not contain transcript text. PII sanitization
  of log output shipped in v0.9.6.

## Telemetry and analytics

- gigastt contains no telemetry, analytics, or "phone-home" code.
- No usage data, error reports, or performance metrics are transmitted to any
  external service.

## Network traffic

The only outbound network call gigastt makes is the one-time model download:

- Files are fetched from `huggingface.co` (`istupakov/gigaam-v3-onnx`).
- Each file is SHA-256 verified before use and written atomically to disk.
- After the initial download, gigastt operates fully offline.

## Server binding

- By default the server listens on `127.0.0.1` (loopback only). Traffic is
  therefore not reachable from other hosts on the network.
- Binding to a non-loopback address requires an explicit opt-in:
  `--bind-all` flag or `GIGASTT_ALLOW_BIND_ANY=1` environment variable.
- Cross-origin requests are denied by default; the origin allowlist is empty
  unless `--allow-origin` or `--cors-allow-any` is passed.

## Prometheus metrics

When `--metrics` is enabled, the `/metrics` endpoint exposes request counts and
latency histograms. These metrics contain no audio content, transcript text, or
user-identifying information — only aggregate HTTP counters and durations.

## Summary

| Data type | Leaves the device? | Stored on disk? | Logged? |
|-----------|-------------------|-----------------|---------|
| Audio frames | No | No | No |
| Transcript text | No | No | No |
| Request metadata (duration, word count) | No | Only if you redirect logs | Yes (word count only) |
| Model weights | No (downloaded once, then local) | Yes (`~/.gigastt/models/`) | No |
