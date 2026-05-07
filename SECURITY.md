# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 2.0.x   | Yes (current)  |
| 1.0.x   | Yes (previous) |
| 0.10.x  | No             |
| < 0.10  | No             |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Report vulnerabilities by email to **m1ndcoderr@gmail.com** with the subject line
`[gigastt] Security vulnerability report`.

Please include:

- A description of the vulnerability and its potential impact
- Steps to reproduce or a proof-of-concept
- Affected version(s)
- Any suggested mitigations if known

### Response timeline

| Milestone | Target |
|-----------|--------|
| Acknowledgment | Within 72 hours |
| Initial assessment | Within 7 days |
| Fix or mitigation | Within 90 days |
| Public disclosure | At or after fix release, coordinated with reporter |

If a fix cannot be delivered within 90 days, the reporter may proceed with public
disclosure at their discretion after notifying the maintainer.

## Scope

The following are in scope:

- Vulnerabilities in gigastt server code (`crates/gigastt/src/`, `crates/gigastt-core/src/`)
- ONNX model loading and the quantization pipeline (`crates/gigastt-core/src/inference/`, `crates/gigastt-core/src/quantize.rs`)
- WebSocket and REST request handlers (`crates/gigastt/src/server/`)
- Authentication bypass, origin-check circumvention, or bind-guard bypass
- Denial-of-service via crafted audio, WebSocket frames, or HTTP requests that
  bypass the documented rate-limit and pool-saturation defenses

The following are **out of scope**:

- Vulnerabilities in upstream dependencies (ONNX Runtime, axum, tokio, symphonia,
  etc.) — report those directly to their maintainers
- Social engineering or phishing attacks
- Issues in third-party model files hosted on HuggingFace
- Findings that require physical access to the machine running the server

## Security design notes

- The server binds to `127.0.0.1` by default. Non-loopback addresses require
  `--bind-all` or `GIGASTT_ALLOW_BIND_ANY=1`.
- Cross-origin requests are denied by default; loopback origins are always allowed.
- All model files are SHA-256 verified on download and written atomically.
- No audio or transcript data is sent off-device. See [`docs/privacy.md`](docs/privacy.md).
