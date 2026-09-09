//! Model-directory cache hygiene: prune stale ORT optimized graphs and
//! optional content-hash hardlink dedupe.
//!
//! ONNX Runtime writes one optimized graph per encoder stem under
//! `optimized_cache/`. Switching heads or running FP32 leaves zombie graphs
//! that reclaim **hundreds of MiB to ~1 GiB** on polluted installs without
//! changing inference accuracy.

use crate::sha256::{Sha256, hex_lower};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::ModelVariant;

/// Version-scoped subdirectory name for the CoreML compiled-model cache:
/// `ort-<onnxruntime-minor>` (e.g. `ort-27`).
///
/// The CoreML EP compiles each session graph to an `.mlmodelc` bundle under
/// `coreml_cache/`, keyed only by a graph hash — never by the ONNX Runtime build
/// that produced it. A bundle compiled by one ONNX Runtime is not guaranteed
/// loadable by another; loading a stale one makes the CoreML EP fail at
/// prediction time and silently fall back to CPU on every run. Scoping the cache
/// under the ORT minor version means an ORT upgrade simply misses the old entries
/// and recompiles once, which is correct and self-healing.
pub(crate) fn coreml_cache_version_dir() -> String {
    format!("ort-{}", ort::MINOR_VERSION)
}

/// Directory the CoreML EP writes its compiled cache into for a model living in
/// `model_dir`: `model_dir/coreml_cache/ort-<minor>/`. Single source of truth for
/// both the write path (the `ort` runtime factory) and the GC keep-name
/// ([`prune_coreml_cache`]). Only the `coreml` build writes the cache, so the
/// path builder is compiled there; the GC keep-name uses
/// [`coreml_cache_version_dir`] directly and stays available on every build.
#[cfg(feature = "coreml")]
pub(crate) fn coreml_cache_dir(model_dir: &Path) -> PathBuf {
    model_dir
        .join("coreml_cache")
        .join(coreml_cache_version_dir())
}

/// Basename of the optimized graph ORT would open for `model_path`'s stem:
/// `{file_stem}_optimized.ort` (ORT flatbuffer format — loaded via mmap
/// by the encoder session loader).
pub fn optimized_cache_basename(encoder_path: &Path) -> Option<String> {
    encoder_path
        .file_stem()
        .map(|s| format!("{}_optimized.ort", s.to_string_lossy()))
}

/// Preferred encoder path on disk for `variant`: INT8 when present, else FP32
/// (RNN-T). CTC heads are INT8-only.
pub fn preferred_encoder_path(variant: ModelVariant, dir: &Path) -> Option<PathBuf> {
    let int8 = dir.join(variant.encoder_int8_file());
    if int8.exists() {
        return Some(int8);
    }
    let enc = variant.encoder_file();
    if !enc.is_empty() {
        let fp32 = dir.join(enc);
        if fp32.exists() {
            return Some(fp32);
        }
    }
    None
}

/// Result of pruning `model_dir/optimized_cache/`.
#[derive(Debug, Default, Clone)]
pub struct OptimizedCachePruneReport {
    /// Optimized graphs retained: one per installed head (may be empty).
    pub kept: Vec<PathBuf>,
    /// Paths removed (or that would be removed under `dry_run`).
    pub removed: Vec<PathBuf>,
    /// Sum of sizes of `removed` entries.
    pub freed_bytes: u64,
    pub dry_run: bool,
}

/// Result of pruning stale CoreML compiled-model caches under
/// `model_dir/coreml_cache/`.
#[derive(Debug, Default, Clone)]
pub struct CoremlCachePruneReport {
    /// Current-ORT-version cache dir retained, if it exists on disk.
    pub kept: Option<PathBuf>,
    /// Stale cache dirs removed (or that would be under `dry_run`): sibling
    /// `ort-*` version dirs and legacy pre-versioning hash dirs.
    pub removed: Vec<PathBuf>,
    /// Sum of sizes of all files under `removed`.
    pub freed_bytes: u64,
    pub dry_run: bool,
}

/// Result of content-hash hardlink dedupe under `model_dir`.
#[derive(Debug, Default, Clone)]
pub struct DedupeReport {
    /// Number of hash groups with 2+ members.
    pub groups: usize,
    /// Number of paths re-linked (or that would be) onto the keep copy.
    pub hardlinked: usize,
    /// Approximate space reclaimed: sum of sizes of redundant members.
    pub freed_bytes: u64,
    pub dry_run: bool,
}

/// Drop optimized graphs that no installed head can load from
/// `model_dir/optimized_cache/`.
///
/// Keeps `{preferred_encoder_stem}_optimized.ort` for **every** head whose
/// encoder weights are present in `model_dir` (INT8 preferred), plus the
/// encoder named by a `manifest.toml` when one is installed: every installed
/// head can be started with `serve --model-variant`, so its graph must survive
/// a GC run — even while a running server has it memory-mapped. Legacy
/// `*_optimized.onnx` graphs (written by versions before the switch to the ORT
/// flatbuffer format) and `.ort` graphs whose encoder is not installed count
/// as zombies and are pruned. When no head is detected, the cache is left
/// untouched so a half-installed tree is not wiped.
///
/// With `dry_run`, reports what would be deleted without removing files.
pub fn prune_optimized_cache(model_dir: &Path, dry_run: bool) -> Result<OptimizedCachePruneReport> {
    prune_optimized_cache_dir(&model_dir.join("optimized_cache"), model_dir, dry_run)
}

/// Like [`prune_optimized_cache`], but prunes an explicit cache directory
/// instead of the default `model_dir/optimized_cache` — for installs where
/// the cache was relocated via `--optimized-cache-dir`. Keep names are still
/// resolved from the installed heads under `model_dir`.
pub fn prune_optimized_cache_dir(
    cache_dir: &Path,
    model_dir: &Path,
    dry_run: bool,
) -> Result<OptimizedCachePruneReport> {
    let mut report = OptimizedCachePruneReport {
        dry_run,
        ..Default::default()
    };
    if !cache_dir.is_dir() {
        return Ok(report);
    }

    let mut keep_names: std::collections::HashSet<String> = ModelVariant::ALL
        .into_iter()
        .filter_map(|v| preferred_encoder_path(v, model_dir))
        .filter_map(|p| optimized_cache_basename(&p))
        .collect();

    // A manifest install may rename the encoder away from the hardcoded
    // variant basenames; the engine loads that name, so keep its graph too.
    if let Some(manifest) = super::manifest::ModelManifest::load(model_dir)?
        && let Some(name) = optimized_cache_basename(&manifest.preferred_encoder_path(model_dir))
    {
        keep_names.insert(name);
    }

    if keep_names.is_empty() {
        tracing::info!(
            "optimized_cache prune: no usable encoder in {}; leaving cache untouched",
            model_dir.display()
        );
        return Ok(report);
    }

    for entry in std::fs::read_dir(cache_dir)
        .with_context(|| format!("failed to read optimized_cache at {}", cache_dir.display()))?
    {
        let entry = entry.context("failed to read optimized_cache entry")?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        // Only touch ORT optimized graphs (current `.ort` and legacy
        // `.onnx`); leave unrelated files alone.
        if !name.ends_with("_optimized.ort") && !name.ends_with("_optimized.onnx") {
            continue;
        }
        if keep_names.contains(name.as_ref()) {
            report.kept.push(path);
            continue;
        }
        let len = entry.metadata().map(|m| m.len()).unwrap_or(0);
        if !dry_run {
            std::fs::remove_file(&path).with_context(|| {
                format!("failed to remove stale optimized graph {}", path.display())
            })?;
        }
        report.removed.push(path);
        report.freed_bytes = report.freed_bytes.saturating_add(len);
    }

    if !report.removed.is_empty() {
        tracing::info!(
            dry_run,
            kept = report.kept.len(),
            removed = report.removed.len(),
            freed_mib = report.freed_bytes / (1024 * 1024),
            "optimized_cache prune finished"
        );
    }
    Ok(report)
}

/// Remove CoreML compiled-model caches produced by a different ONNX Runtime
/// build under `model_dir/coreml_cache/`.
///
/// Keeps only the current version dir (`ort-<minor>/`, see
/// `coreml_cache_dir`); removes every other subdirectory — sibling `ort-*`
/// dirs from other ORT versions and legacy hash dirs written directly under
/// `coreml_cache/` before the cache was version-scoped. Stray files are left
/// alone (mirrors [`prune_optimized_cache`]). Not feature-gated: a CPU build may
/// still need to reclaim a `coreml_cache/` left behind by an earlier CoreML
/// build.
///
/// With `dry_run`, reports what would be deleted without removing anything.
pub fn prune_coreml_cache(model_dir: &Path, dry_run: bool) -> Result<CoremlCachePruneReport> {
    let mut report = CoremlCachePruneReport {
        dry_run,
        ..Default::default()
    };
    let cache_root = model_dir.join("coreml_cache");
    if !cache_root.is_dir() {
        return Ok(report);
    }

    let keep_name = coreml_cache_version_dir();
    for entry in std::fs::read_dir(&cache_root)
        .with_context(|| format!("failed to read coreml_cache at {}", cache_root.display()))?
    {
        let entry = entry.context("failed to read coreml_cache entry")?;
        let path = entry.path();
        // Only compiled-model cache subdirectories are ours to manage.
        if !path.is_dir() {
            continue;
        }
        if entry.file_name().to_string_lossy() == keep_name {
            report.kept = Some(path);
            continue;
        }
        let size = dir_size_bytes(&path);
        if !dry_run {
            std::fs::remove_dir_all(&path).with_context(|| {
                format!("failed to remove stale coreml cache {}", path.display())
            })?;
        }
        report.removed.push(path);
        report.freed_bytes = report.freed_bytes.saturating_add(size);
    }

    if !report.removed.is_empty() {
        tracing::info!(
            dry_run,
            kept = report.kept.is_some(),
            removed = report.removed.len(),
            freed_mib = report.freed_bytes / (1024 * 1024),
            "coreml_cache prune finished"
        );
    }
    Ok(report)
}

/// Recursively sum the sizes of regular files under `path`. Best-effort:
/// unreadable entries contribute zero rather than aborting the walk.
fn dir_size_bytes(path: &Path) -> u64 {
    let mut total = 0u64;
    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in rd.flatten() {
            match entry.file_type() {
                Ok(ft) if ft.is_dir() => stack.push(entry.path()),
                Ok(ft) if ft.is_file() => {
                    total = total.saturating_add(entry.metadata().map(|m| m.len()).unwrap_or(0));
                }
                _ => {}
            }
        }
    }
    total
}

/// Hardlink exact duplicate files under `model_dir` (content SHA-256).
///
/// For each hash group with more than one path, keeps the first path in
/// stable sort order and replaces other members with hardlinks to it. Skips
/// `.partial*` staging files and directories. Cross-device hardlink failures
/// leave the original file in place and are reported via `tracing::warn`.
///
/// With `dry_run`, only measures reclaimable bytes.
pub fn dedupe_model_dir(model_dir: &Path, dry_run: bool) -> Result<DedupeReport> {
    let mut report = DedupeReport {
        dry_run,
        ..Default::default()
    };
    if !model_dir.is_dir() {
        return Ok(report);
    }

    let mut by_hash: HashMap<String, Vec<PathBuf>> = HashMap::new();
    collect_regular_files(model_dir, &mut by_hash)?;

    for mut paths in by_hash.into_values() {
        if paths.len() < 2 {
            continue;
        }
        paths.sort();
        report.groups += 1;
        let keep = paths[0].clone();
        let size = std::fs::metadata(&keep).map(|m| m.len()).unwrap_or(0);
        for other in paths.into_iter().skip(1) {
            if same_file(&keep, &other)? {
                continue;
            }
            if dry_run {
                report.hardlinked += 1;
                report.freed_bytes = report.freed_bytes.saturating_add(size);
                continue;
            }
            // Replace the redundant copy with a hardlink to `keep`.
            let tmp = other.with_extension(format!("dedupe-tmp.{}", std::process::id()));
            // Remove destination first so hard_link can create the name.
            // Use a rename-to-tmp then hardlink then remove-tmp so a crash
            // mid-way leaves either the original or a recoverable tmp.
            std::fs::rename(&other, &tmp)
                .with_context(|| format!("failed to stage {} for hardlink", other.display()))?;
            match std::fs::hard_link(&keep, &other) {
                Ok(()) => {
                    let _ = std::fs::remove_file(&tmp);
                    report.hardlinked += 1;
                    report.freed_bytes = report.freed_bytes.saturating_add(size);
                }
                Err(e) => {
                    // Restore original name on failure (e.g. cross-device).
                    let _ = std::fs::rename(&tmp, &other);
                    tracing::warn!(
                        keep = %keep.display(),
                        other = %other.display(),
                        error = %e,
                        "hardlink dedupe skipped"
                    );
                }
            }
        }
    }

    if report.hardlinked > 0 {
        tracing::info!(
            dry_run,
            groups = report.groups,
            hardlinked = report.hardlinked,
            freed_mib = report.freed_bytes / (1024 * 1024),
            "model-dir content-hash dedupe finished"
        );
    }
    Ok(report)
}

fn collect_regular_files(root: &Path, by_hash: &mut HashMap<String, Vec<PathBuf>>) -> Result<()> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let rd = match std::fs::read_dir(&dir) {
            Ok(rd) => rd,
            Err(e) => {
                tracing::warn!(path = %dir.display(), error = %e, "skip unreadable dir");
                continue;
            }
        };
        for entry in rd {
            let entry = entry.with_context(|| format!("read_dir {}", dir.display()))?;
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            // Skip download staging and lock files.
            if name.contains(".partial") || name.ends_with(".lock") || name.starts_with('.') {
                continue;
            }
            let ft = entry
                .file_type()
                .with_context(|| format!("file_type {}", path.display()))?;
            if ft.is_dir() {
                stack.push(path);
                continue;
            }
            if !ft.is_file() {
                continue;
            }
            // Small files rarely reclaim meaningful space; still hash so exact
            // dups of vocab-scale files hardlink when present.
            let digest =
                sha256_file_streaming(&path).with_context(|| format!("hash {}", path.display()))?;
            by_hash.entry(digest).or_default().push(path);
        }
    }
    Ok(())
}

fn sha256_file_streaming(path: &Path) -> Result<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_lower(&hasher.finalize()))
}

fn same_file(a: &Path, b: &Path) -> Result<bool> {
    let ma = std::fs::metadata(a)?;
    let mb = std::fs::metadata(b)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Ok(ma.dev() == mb.dev() && ma.ino() == mb.ino())
    }
    #[cfg(not(unix))]
    {
        // Stable `MetadataExt` on Windows does not expose file index / volume
        // serial (those APIs are still `windows_by_handle`). Treat as "not
        // known to be the same" so the caller still attempts a hardlink.
        let _ = (ma, mb);
        Ok(false)
    }
}

#[cfg(test)]
mod tests;
