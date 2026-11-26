//! Unit tests for model download / variant / progress.

#[cfg(feature = "ane")]
use super::download::ane::*;
#[cfg(feature = "net")]
use super::download::fetch::*;
use super::download::*;
use super::progress::*;
use super::variant::*;
#[cfg(feature = "net")]
use crate::sha256::{Sha256, hex_lower};
use std::io::Write;
use std::path::Path;
#[cfg(feature = "net")]
use tokio::io::AsyncWriteExt;

/// Compute the SHA-256 of a byte slice as a lowercase hex digest.
#[cfg(feature = "net")]
pub(super) fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_lower(&hasher.finalize())
}

/// Helper to stage a `.partial` file with arbitrary bytes, mimicking
/// the state of a fully streamed download prior to verification.
pub(super) fn stage_partial(final_path: &Path, bytes: &[u8]) -> std::path::PathBuf {
    let partial = partial_path(final_path);
    let mut f = std::fs::File::create(&partial).expect("create partial");
    f.write_all(bytes).expect("write partial");
    f.sync_all().expect("sync partial");
    partial
}

#[cfg(any(feature = "ane", feature = "net"))]
mod ane;
mod download;
mod progress;
mod variant;
