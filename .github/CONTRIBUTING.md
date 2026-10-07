# Contributing

## Development

Use Rust 1.94 or newer, a native C/C++ toolchain, and `protoc`. On Debian or
Ubuntu, install the native prerequisites before running Cargo:

```sh
sudo apt-get install build-essential pkg-config protobuf-compiler libssl-dev
```

On macOS, install the Xcode command-line tools and `brew install protobuf`.
The default build downloads ONNX Runtime. Its locked build dependency chain
is `ort-sys` → `ureq` → `native-tls`; on Linux this requires OpenSSL development
headers and libraries through `openssl-sys`. This is a build prerequisite even
though the application's HTTP client uses Rustls. Inspect the active chain with
`cargo tree --workspace -i openssl-sys --edges normal,build`.

For a Linux OpenSSL installation outside the system search path, put the following
exports in your local shell setup, substituting the actual development prefix:

```sh
export OPENSSL_INCLUDE_DIR=/path/to/openssl/include
export OPENSSL_LIB_DIR=/path/to/openssl/lib
test -f "$OPENSSL_INCLUDE_DIR/openssl/opensslv.h"
```

Use `lib64` or the architecture-specific library directory when that is where
the installation keeps its libraries. An unpacked Debian development package
also keeps generated headers under `include/<multiarch>`; expose that directory
to the C compiler as well, for example on x86_64:

```sh
export CFLAGS="${CFLAGS:+$CFLAGS }-I$OPENSSL_INCLUDE_DIR/x86_64-linux-gnu"
```

Only add that flag when the directory exists. Both headers and link libraries
must come from a compatible OpenSSL installation. With system packages, leave
these overrides unset and let `pkg-config --cflags --libs openssl` discover them.
The same environment must be available to Git's pre-commit hook. No temporary
helper script or dependency-feature change is required.

```sh
cargo build                            # CPU debug build
cargo build --features coreml          # macOS ARM64 with CoreML
cargo build --features cuda            # Linux x86_64 with CUDA 12+
cargo build --features ane             # macOS ARM64 native ANE (file mode)
cargo build --features candle          # experimental Candle/Metal
cargo build -p gigastt-ffi             # C-ABI FFI layer (Android / mobile)

cargo test --workspace --lib --bins    # unit tests (no model needed)
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

Pass `--lib --bins`: a bare `cargo test` builds only the default member and a
bare `cargo test --workspace` pulls in the ~2.5-hour WER benchmark, which is a
`harness = false` target and so is not skipped by `--ignored`.

Enable the repository hooks once. Pre-commit checks public content, the Rust
checks above and [`typos`](https://github.com/crate-ci/typos); commit-msg and
pre-push also check publication metadata:

```sh
git config core.hooksPath .githooks
```

Keep internal planning in the ignored local backlog. Do not publish tracker
identifiers or private planning files in commits, branches, tags or PR text.
Run `python3 scripts/check-publication.py` to check the staged tree and branch;
CI also checks commit history introduced by the PR and its title/description.

For E2E / load / soak tests see [`CLAUDE.md`](../CLAUDE.md).

## Pull requests

- One logical change per PR; rebase on `main` before opening.
- CI (`ci.yml`) must be green: clippy, unit tests, feature compile checks, audit.
- `cargo deny check` passes (license + advisory + sources).
- For user-visible changes: add a bullet under the `## [Unreleased]` section of `CHANGELOG.md`.
- Keep commit messages short and present-tense (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`).

## Dependency and runtime changes

Follow [performance and dependency acceptance](../docs/performance-regressions.md).
Backend, model, codec and thread-policy changes require measured quality, speed
and resource comparisons. Compilation and API compatibility do not establish
performance compatibility. Performance gates apply to automated dependency PRs
as well as hand-written changes.

## Release checklist

Release artifacts are produced by [`release.yml`](workflows/release.yml)
on `v*` tag push. Never upload tarballs manually — the workflow is the single
source of truth, and out-of-band uploads break SHA-pinned clients (e.g. Murmur).

1. **Bump version** in `Cargo.toml` (`version = "x.y.z"`). Run `cargo check` so `Cargo.lock` updates.
2. **Update `CHANGELOG.md`**: move the `## [Unreleased]` bullets into a new `## [x.y.z] - YYYY-MM-DD` section; leave an empty `## [Unreleased]` for the next cycle.
3. **Verify locally**: `cargo test --workspace --lib --bins && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all -- --check && python3 scripts/check-docs-drift.py`.
4. **Merge through a PR**: commit the release preparation, merge after required checks pass, and wait for the exact resulting main commit to pass the full CI, including performance and quality gates.
5. **Tag & push** (signed):
   ```sh
   git tag -s vx.y.z -m "gigastt vx.y.z"
   git push origin vx.y.z
   ```
6. **Wait for the release workflow** to finish on GitHub Actions.
   It must produce:
   - `gigastt-x.y.z-aarch64-apple-darwin.tar.gz`
   - `gigastt-x.y.z-x86_64-unknown-linux-gnu.tar.gz`
   - `gigastt-x.y.z-aarch64-unknown-linux-gnu.tar.gz`
   - `gigastt-x.y.z-x86_64-pc-windows-msvc.tar.gz`
   - `SHA256SUMS.txt`
   - Per-asset `*.sha256` files

   The CUDA Linux *tarball* is still not built (the `linux-x86_64-cuda` matrix entry is
   commented out until the CUDA install path stabilizes), so CUDA users either take the
   published `ghcr.io/ekhodzitsky/gigastt:cuda` image or build from source.

   The same workflow also publishes multi-arch GHCR images (`:x.y.z` / `:latest`, plus
   `:x.y.z-cuda` / `:cuda`), offline bundles and `.deb` packages, an SBOM, SLSA provenance
   and minisign signatures. Homebrew's Formula update is proposed by `homebrew.yml` as a PR. Approve its
   workflows when prompted, wait for green checks and merge it; see
   [Homebrew release updates](../docs/homebrew-release.md).
7. **Verify the release page** on GitHub — all assets attached, release notes generated.
8. **Publish to crates.io** (only after step 7, from the validated release commit).
   Run `GITHUB_REPOSITORY=ekhodzitsky/gigastt python3 scripts/performance-gates.py release "$(git rev-parse HEAD)"`
   with authenticated `gh` before publishing:

   ```sh
   cargo publish -p gigastt-core --dry-run
   cargo publish -p gigastt-core
   cargo publish -p gigastt --dry-run
   cargo publish -p gigastt
   ```
   Publish `gigastt-core` first (it is a dependency of `gigastt`). `gigastt-ffi` is a cdylib and not published to crates.io.
   The dry-run must succeed before the real publish. A failed `cargo publish` after the tag is pushed means the tag and crate diverge — fix forward with `vx.y.z+1`; do NOT re-tag.
9. **Language bindings ship separately**, each via `workflow_dispatch`: `node-prebuilds.yml`
   (npm), `python-wheels.yml` (PyPI), `ios-xcframework.yml`, `android-aar.yml`. They are not
   triggered by the tag — run them when a release needs to reach those ecosystems.
10. **Announce** briefly (GitHub release body already covers it; no separate post required).

### If the release workflow fails

- Re-run failed jobs via the GitHub UI. Do not re-tag.
- If the tarball layout needs a fix, land the patch on `main`, bump to `vx.y.z+1`, and re-tag. The old tag stays as-is (immutable history).

### Never

- `--no-verify` on commits, `--no-gpg-sign` on tags.
- Manual `gh release upload` of binary assets — breaks downstream SHA pinning.
- Hand-editing published release assets.
