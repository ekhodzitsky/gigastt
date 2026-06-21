//! Guards the embedded lean build: `gigastt-core` with no default features must
//! not pull tokio, reqwest/hyper, or symphonia into its production dependency
//! graph. Runs `cargo tree` over normal edges only (dev-deps like wiremock,
//! which legitimately pull tokio/hyper, are excluded).

use std::process::Command;

#[test]
fn lean_core_excludes_heavy_deps() {
    let out = Command::new(env!("CARGO"))
        .args([
            "tree",
            "-p",
            "gigastt-core",
            "--no-default-features",
            "--edges",
            "normal",
            "--prefix",
            "none",
        ])
        .output()
        .expect("cargo tree runs");
    assert!(
        out.status.success(),
        "cargo tree failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let tree = String::from_utf8_lossy(&out.stdout);
    for forbidden in ["tokio ", "reqwest ", "hyper ", "symphonia "] {
        assert!(
            !tree.lines().any(|l| l.trim_start().starts_with(forbidden)),
            "lean gigastt-core must not depend on `{}`; production tree:\n{tree}",
            forbidden.trim()
        );
    }
}
