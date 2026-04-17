# Contributing

## Development

```sh
cargo build                            # CPU debug build
cargo build --features coreml          # macOS ARM64 with CoreML
cargo build --features cuda            # Linux x86_64 with CUDA 12+
cargo build --features quantize        # adds `gigastt quantize` subcommand

cargo test                             # unit tests (no model needed)
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
```

For E2E / load / soak tests see [`CLAUDE.md`](CLAUDE.md).

## Pull requests

- One logical change per PR; rebase on `main` before opening.
- CI (`ci.yml`) must be green: clippy, unit tests, feature compile checks, audit.
- For user-visible changes: add a bullet under the `## [Unreleased]` section of `CHANGELOG.md`.
- Keep commit messages short and present-tense (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`).

## Release checklist

Release artifacts are produced by [`release.yml`](.github/workflows/release.yml)
on `v*` tag push. Never upload tarballs manually — the workflow is the single
source of truth, and out-of-band uploads break SHA-pinned clients (e.g. Murmur).

1. **Bump version** in `Cargo.toml` (`version = "x.y.z"`). Run `cargo check` so `Cargo.lock` updates.
2. **Update `CHANGELOG.md`**: move the `## [Unreleased]` bullets into a new `## [x.y.z] - YYYY-MM-DD` section; leave an empty `## [Unreleased]` for the next cycle.
3. **Verify locally**: `cargo test && cargo clippy --all-targets -- -D warnings && cargo fmt --all -- --check`.
4. **Commit**: `chore: bump version to x.y.z, update CHANGELOG`.
5. **Tag & push** (signed):
   ```sh
   git tag -s vx.y.z -m "gigastt vx.y.z"
   git push origin main --tags
   ```
6. **Wait for the release workflow** to finish on GitHub Actions.
   It must produce:
   - `gigastt-x.y.z-aarch64-apple-darwin.tar.gz`
   - `gigastt-x.y.z-x86_64-unknown-linux-gnu.tar.gz`
   - `SHA256SUMS.txt`
   - Per-asset `*.sha256` files

   The CUDA Linux build is not yet automated — see `specs/todo.md` (Phase 0 addendum). Until it lands, CUDA users build from source.
7. **Verify the release page** on GitHub — all assets attached, release notes generated.
8. **Publish to crates.io** (only after step 7):
   ```sh
   cargo publish --dry-run
   cargo publish
   ```
   The dry-run must succeed before the real publish. A failed `cargo publish` after the tag is pushed means the tag and crate diverge — fix forward with `vx.y.z+1`; do NOT re-tag.
9. **Announce** briefly (GitHub release body already covers it; no separate post required).

### If the release workflow fails

- Re-run failed jobs via the GitHub UI. Do not re-tag.
- If the tarball layout needs a fix, land the patch on `main`, bump to `vx.y.z+1`, and re-tag. The old tag stays as-is (immutable history).

### Never

- `--no-verify` on commits, `--no-gpg-sign` on tags.
- Manual `gh release upload` of binary assets — breaks downstream SHA pinning.
- Hand-editing published release assets.
