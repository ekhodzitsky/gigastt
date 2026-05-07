# Verifying gigastt releases (SUS-03, SUS-05)

Every tagged release on GitHub ships three kinds of attestation alongside
the binary tarballs. You don't need all three — pick the one that matches
your threat model.

## 1. SHA-256 checksums (every release)

`SHA256SUMS.txt` lists the expected digest for every `*.tar.gz`. This
protects against corruption in flight but **not** against a compromised
GitHub release (an attacker with release access could publish matching
checksums alongside tampered binaries).

```sh
gh release download v2.0.3 -R ekhodzitsky/gigastt \
    -p 'gigastt-*.tar.gz' -p 'SHA256SUMS.txt'
shasum -a 256 -c SHA256SUMS.txt
```

## 2. minisign signatures (SUS-03)

When the maintainer's minisign key is loaded in CI, every tarball +
`SHA256SUMS.txt` + SBOM gets a detached `.minisig` signature. This
protects against a compromised release (the attacker would also need
the minisign private key).

Public key (save as `gigastt.pub`; the two-line `untrusted comment:`
header is part of the file format — keep it verbatim):

```
untrusted comment: minisign public key C1A4D4B7428907DA
RWTaB4lCt9SkwerVa5kINWK8Jh/I96jUQDybbvmcpQr0g3lvGnymrXfm
```

Verify with [minisign](https://jedisct1.github.io/minisign/) or
[rsign2](https://github.com/jedisct1/rsign2):

```sh
gh release download v2.0.3 -R ekhodzitsky/gigastt \
    -p '*.tar.gz' -p '*.tar.gz.minisig'
minisign -Vm gigastt-2.0.3-aarch64-apple-darwin.tar.gz -p gigastt.pub
```

## 3. SLSA build provenance (SUS-05)

Every artefact carries an in-toto attestation signed by Sigstore via
GitHub's `attest-build-provenance` action. This proves the binary was
built by the `release.yml` workflow on a specific commit in
`ekhodzitsky/gigastt` — no special public key required.

```sh
gh attestation verify gigastt-2.0.3-aarch64-apple-darwin.tar.gz \
    --repo ekhodzitsky/gigastt
```

## What to use when

| Threat | SHA256 | minisign | SLSA provenance |
|---|---|---|---|
| Mirror / in-flight tampering | ✅ | ✅ | ✅ |
| Compromised GitHub release | ❌ | ✅ | ⚠ only if attacker doesn't also control CI |
| Compromised maintainer CI token | ❌ | ✅ | ❌ |
| Rebuild reproducibility proof | ❌ | ❌ | ✅ (workflow SHA recorded) |

For privacy-conscious deployments, verify **both** minisign and SLSA —
they fail independently, so it takes two compromises to forge.
