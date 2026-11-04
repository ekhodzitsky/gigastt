# Verifying gigastt releases

Published releases provide checksums and build provenance. Detached minisign
signatures are also provided when signing is configured. Verify the assets
actually attached to the chosen release; a Git tag alone does not publish them.

## 1. SHA-256 checksums (every release)

`SHA256SUMS.txt` lists the expected digest for every `*.tar.gz` and
`*.deb`. This protects against corruption in flight but **not** against a
compromised GitHub release (an attacker with release access could publish
matching checksums alongside tampered binaries).

```sh
gh release download v2.22.0 -R ekhodzitsky/gigastt \
    -p 'gigastt-*.tar.gz' -p 'SHA256SUMS.txt'
# SHA256SUMS.txt also lists the .deb packages — filter to what you downloaded
# (on GNU coreutils, `sha256sum -c --ignore-missing SHA256SUMS.txt` works too)
grep '\.tar\.gz$' SHA256SUMS.txt | shasum -a 256 -c -
```

## 2. minisign signatures

When the maintainer's minisign key is loaded in CI, every tarball and `.deb`
package + `SHA256SUMS.txt` + SBOM gets a detached `.minisig` signature. This
protects against a compromised release (the attacker would also need
the minisign private key).

Signing is optional when the CI secret is absent. Unsigned releases still require
all binary, checksum and SBOM asset categories. When signing is enabled, upload
also requires a signature for every tarball, Debian package, checksum manifest
and SBOM; an incomplete signature set stops publication.

Public key (save as `gigastt.pub`; the two-line `untrusted comment:`
header is part of the file format — keep it verbatim):

```
untrusted comment: minisign public key C1A4D4B7428907DA
RWTaB4lCt9SkwerVa5kINWK8Jh/I96jUQDybbvmcpQr0g3lvGnymrXfm
```

Verify with [minisign](https://jedisct1.github.io/minisign/) or
[rsign2](https://github.com/jedisct1/rsign2):

```sh
gh release download v2.22.0 -R ekhodzitsky/gigastt \
    -p '*.tar.gz' -p '*.tar.gz.minisig'
minisign -Vm gigastt-2.22.0-aarch64-apple-darwin.tar.gz -p gigastt.pub
```

## 3. SLSA build provenance

Every artefact carries an in-toto attestation signed by Sigstore via
GitHub's `attest-build-provenance` action. This proves the binary was
built by the `release.yml` workflow on a specific commit in
`ekhodzitsky/gigastt` — no special public key required.

```sh
gh attestation verify gigastt-2.22.0-aarch64-apple-darwin.tar.gz \
    --repo ekhodzitsky/gigastt
```

## What to use when

| Threat | SHA256 | minisign | SLSA provenance |
|---|---|---|---|
| Mirror / in-flight tampering | ✅ | ✅ | ✅ |
| Compromised GitHub release | ❌ | ✅ | ⚠ only if attacker doesn't also control CI |
| Compromised maintainer CI token | ❌ | Depends on signing-key access | ❌ |
| Bit-for-bit reproducible rebuild proof | ❌ | ❌ | ❌ |

An attestation establishes recorded build provenance, not a bit-for-bit
reproducible build or a code audit; see [GitHub's attestation documentation](https://docs.github.com/en/actions/concepts/security/artifact-attestations).
Signature and provenance checks have different trust assumptions. If both signing
credentials and the build workflow are accessible from one compromised CI
environment, they are not independent protections.

## Release source selection

A manual Release dispatch first resolves an existing
`vMAJOR.MINOR.PATCH` tag, optionally with a SemVer prerelease suffix such as
`-rc.1`. Build metadata (`+suffix`) is rejected because it cannot be used in a
Docker tag. The tag version must match the workspace and every workspace
package's effective version. Missing tags and mismatches stop all publishing
jobs.

Every binary build, SBOM and container context uses the same resolved commit,
even when manual dispatch selects a different workflow ref. Container revision
labels identify that commit. The SLSA predicate records both the invoking
workflow's ref/commit and the resolved release tag/commit, so the workflow
revision is not mistaken for the artifact source. The existing attestation
[action supports custom predicates](https://github.com/actions/attest-build-provenance/blob/v4/action.yml).

Maintainers must select a reviewed revision with successful full main CI before
dispatching. The workflow validates the tag/source version and separately enforces
the exact-source performance/quality prerequisite. It is not a dry run: successful
Release jobs publish GitHub assets and GHCR images. A tag push does not launch it.
Local resolver regressions use temporary Git repositories and never publish.
Do not dispatch historical tags as part of history maintenance or replace
previously signed release assets.
