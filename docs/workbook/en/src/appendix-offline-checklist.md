# Appendix B — Offline / air-gapped checklist

Use this when the host must not reach HuggingFace or GitHub at runtime.
Detailed install recipes live in [Getting started](01-getting-started.md) and
[Deployment & ops](06-deployment-ops.md); this page is the operator checklist.

## On a connected machine

- [ ] Resolve release tag: `TAG=$(gh api repos/ekhodzitsky/gigastt/releases/latest -q .tag_name)` (or pin `v2.14.1`)
- [ ] Download **offline bundle** for the target arch  
      (`gigastt-${VER}-offline-x86_64-unknown-linux-gnu.tar.gz` or aarch64)  
      **or** deb pair: `gigastt_…_amd64.deb` + `gigastt-model-int8_…_all.deb`
- [ ] Download `.sha256` (+ `.minisig` if you verify with minisign)
- [ ] Verify checksums (`sha256sum -c`) and optional
      [docs/verifying-releases.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/verifying-releases.md)
- [ ] If you need **speaker diarization**, also fetch `wespeaker_resnet34.onnx`
      (not always in the lean offline bundle) on a connected host and copy it
      into the model dir later — or accept mono transcripts without labels
- [ ] If you need **Silero VAD** on the air-gapped host, fetch the VAD model the
      same way (`--vad` would otherwise try the network)

## On the air-gapped host

- [ ] Install binary + model (installer script, `dpkg -i`, or unpack tarball)
- [ ] Confirm model files under `~/.gigastt/models/` (or your `--model-dir`)
- [ ] Run with offline guard when scripting: `GIGASTT_OFFLINE=1` or global
      `--offline` so a missing file **fails fast** instead of hanging on DNS
- [ ] Smoke: `gigastt transcribe sample.wav` (no network)
- [ ] Server: `gigastt serve` → `curl http://127.0.0.1:9876/ready`
- [ ] systemd: enable unit from the bundle; bind remains loopback unless you
      explicitly set `--bind-all` / `GIGASTT_ALLOW_BIND_ANY=1`

## Runtime rules that stay true offline

| Action | Network? |
|---|---|
| `transcribe` / `transcribe-batch` / `watch` with models on disk | No |
| `serve` after models present | No |
| `POST /v1/admin/reload` after files replaced on disk | No |
| First `download` / missing punct / VAD / speaker model | **Yes** (blocked by `--offline`) |

## After copy-in of new model files

```sh
# No restart required if serve is already up:
curl -s -X POST http://127.0.0.1:9876/v1/admin/reload
# {"reloaded":true,"variant":"rnnt","encoder":"int8"}
```

See [Hot-reload](06-deployment-ops.md#hot-reload-the-model-without-restart).

## Links

- [Getting started — air-gapped](01-getting-started.md)
- [Deployment — offline install](06-deployment-ops.md)
- [packaging/offline/README-OFFLINE.md](https://github.com/ekhodzitsky/gigastt/blob/main/packaging/offline/README-OFFLINE.md)
- [docs/privacy.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/privacy.md) — runtime vs build-time network
