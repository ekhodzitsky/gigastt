# Temporary dependency security patches

These are crates.io sources with narrowly scoped dependency fixes. The published
Candle 0.11.0 requires vulnerable ZIP 8.x; Askama derive 0.16.0 requires the
unmaintained `basic-toml`. No advisory is suppressed for these dependencies.

- **candle-core 0.11.0:** require ZIP 9.1, propagate fallible archive-name decoding
  in NPZ/PyTorch readers. Tensor operations, kernels and backend selection are
  unchanged. ZIP default features remain disabled.
- **askama_derive 0.16.0:** replace `basic-toml` with `toml` (parse + serde only),
  already used by UniFFI. Template generation is otherwise unchanged.

The adjacent `.patch` files contain every change to the upstream sources.
`provenance.json` records SHA-256 hashes of the original crates.io archives.
Original license files, tests and `Cargo.toml.orig` are retained. Registry-local
`.cargo-ok`/`.cargo_vcs_info.json` and upstream lockfiles are omitted; standalone
test runs may create ignored lockfiles. The workspace Cargo.lock controls product
builds. Do not independently upgrade these copies without updating provenance.

Root `[patch.crates-io]` applies to this checkout, including release binaries,
Docker images and binding builds. **Cargo patches are not transitive:** consumers
of published `gigastt-core` crates do not inherit them. Until upstream releases
contain these fixes, downstream workspaces enabling Candle must apply equivalent
patches themselves. Do not describe this as fixing already published crates.
The fuzz workspace does not resolve Candle or Askama; its certificate bundle is
updated separately in its lockfile.

Remove each patch and its vendored directory after adopting a maintained upstream
release that fixes its dependency, then repeat audits and compatibility checks.

Local compatibility checks (in addition to workspace tests and Clippy):

```sh
cargo test --manifest-path vendor/askama_derive-0.16.0/Cargo.toml --lib config::
cargo test --manifest-path vendor/candle-core-0.11.0/Cargo.toml --test serialization_tests --test pth_tests
python3 scripts/test_docker_cache.py
cargo audit
cargo audit --file fuzz/Cargo.lock
cargo deny check
```
