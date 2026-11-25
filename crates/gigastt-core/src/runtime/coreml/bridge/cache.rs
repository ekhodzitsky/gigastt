//! Compile + disk-cache a per-bucket `.mlpackage` as `.mlmodelc`.

use std::path::{Path, PathBuf};

use objc2::rc::Retained;
use objc2_core_ml::{MLComputeUnits, MLModel, MLModelConfiguration};
use objc2_foundation::{NSString, NSURL};

use super::ns_error_message;
use crate::runtime::error::RuntimeError;

/// Compile a `.mlpackage` to a `.mlmodelc` and load it as an `MLModel`, caching
/// the compiled `.mlmodelc` on disk so every restart after the first is ~instant.
///
/// A `.mlpackage` must be compiled before loading, and the Core ML compile is the
/// expensive part (~20 s per bucket on first load). `compileModelAtURL_error`
/// returns a compiled `.mlmodelc` in a TEMP directory that Core ML deletes later,
/// so without persistence every process restart re-pays the full compile.
///
/// To eliminate that cold-start this fn keeps a disk cache next to the source
/// package: `<package.parent>/compiled_cache/<package.stem>.mlmodelc`, validated
/// by a sidecar `<stem>.mlmodelc.meta` recording the source package's identity
/// (recursive total byte size + newest mtime) plus the macOS product version.
///
/// - **Cache hit** (cached `.mlmodelc` exists AND sidecar key matches): load it
///   directly, SKIPPING the compile (fast path).
/// - **Cache miss / stale**: compile, copy the temp `.mlmodelc` into a staging
///   dir under `compiled_cache/`, atomically rename it into the final cache path
///   (clearing any stale one first), write the sidecar, then load from the cache.
/// - **Any cache I/O failure**: fall back to loading directly from the temp
///   `.mlmodelc` (logged), so caching can never break loading.
///
/// The OS version is part of the key because Apple may make compiled models
/// incompatible across OS updates — bumping macOS recompiles automatically.
/// Concurrent compilers (two processes, same bucket) are safe: the staging +
/// atomic-rename is last-writer-wins on byte-identical content (mirrors
/// `model::extract_ane_tar_atomic`).
///
/// When `cpu_and_ne` is set the model is configured with
/// `MLComputeUnits::CPUAndNeuralEngine` so the Apple Neural Engine is engaged.
///
/// `cpu_and_ne` is intentionally NOT part of the cache key: the compiled
/// `.mlmodelc` is compute-unit-independent. The `CPUAndNeuralEngine` vs
/// `CPUOnly` choice is applied at LOAD time via `setComputeUnits` on the
/// `MLModelConfiguration` (see below), not baked into the compile, so a single
/// cached artifact is valid for both configs. Folding `cpu_and_ne` into the key
/// would only store two byte-identical copies.
// `compileModelAtURL_error` is the synchronous compile API; objc2 marks it
// deprecated in favor of the async completion-handler variant, but a synchronous
// compile is exactly what this blocking, once-per-bucket path wants.
#[allow(deprecated)]
#[allow(unsafe_code)] // Core ML native boundary.
pub fn compile_and_load(
    package: &Path,
    cpu_and_ne: bool,
) -> Result<Retained<MLModel>, RuntimeError> {
    // SAFETY: `MLModelConfiguration::new` allocates+initializes a fresh config;
    // `setComputeUnits` is a plain setter on that owned object.
    let config: Retained<MLModelConfiguration> = unsafe { MLModelConfiguration::new() };
    let units = if cpu_and_ne {
        MLComputeUnits::CPUAndNeuralEngine
    } else {
        MLComputeUnits::CPUOnly
    };
    // SAFETY: `config` is a live, uniquely-owned MLModelConfiguration.
    unsafe { config.setComputeUnits(units) };

    // Fast path: a valid cached `.mlmodelc` lets us skip the ~20 s compile.
    let cached = cached_model_path(package);
    if cached.is_dir() {
        match current_source_key(package) {
            Ok(key) if meta_matches(&cached_meta_path(package), &key) => {
                match load_model_from_dir(&cached, &config) {
                    Ok(model) => {
                        tracing::info!(
                            cache = %cached.display(),
                            "loaded compiled ANE model from cache"
                        );
                        return Ok(model);
                    }
                    Err(e) => {
                        // Cache is structurally bad — recompile rather than fail.
                        tracing::warn!(
                            cache = %cached.display(),
                            error = %e,
                            "cached ANE model failed to load; recompiling"
                        );
                    }
                }
            }
            Ok(_) => {
                tracing::info!(
                    cache = %cached.display(),
                    "ANE compiled-model cache stale (source or OS changed); recompiling"
                );
            }
            Err(e) => {
                tracing::warn!(error = %e, "could not compute ANE cache key; recompiling");
            }
        }
    }

    // Miss / stale: compile (the expensive step) and load from the temp result.
    tracing::info!(
        package = %package.display(),
        cache = %cached.display(),
        "compiling ANE model (cold-start ~20s), caching for fast restarts"
    );
    let compiled_url = compile_package(package)?;

    // Populate the disk cache from the temp `.mlmodelc`. Best-effort: on any I/O
    // failure we log and load directly from the temp dir so caching never breaks
    // loading.
    if let Some(temp_dir) = url_to_path(&compiled_url) {
        match populate_cache(package, &temp_dir) {
            Ok(()) => {
                // Prefer loading from the cache so the load matches what future
                // restarts will load (and so the temp dir can be reclaimed).
                match load_model_from_dir(&cached, &config) {
                    Ok(model) => return Ok(model),
                    Err(e) => tracing::warn!(
                        cache = %cached.display(),
                        error = %e,
                        "freshly cached ANE model failed to load; loading from temp"
                    ),
                }
            }
            Err(e) => tracing::warn!(
                cache = %cached.display(),
                error = %e,
                "failed to populate ANE compiled-model cache; loading from temp"
            ),
        }
    } else {
        tracing::warn!("compiled ANE model URL is not a local path; cache skipped");
    }

    // Fallback: load directly from the temp `.mlmodelc` Core ML produced.
    load_model_from_url(package, &compiled_url, &config)
}

/// Run the synchronous Core ML compile, returning the temp `.mlmodelc` URL.
// See `compile_and_load` for why the deprecated synchronous API is used.
#[allow(deprecated)]
#[allow(unsafe_code)] // Core ML native boundary.
fn compile_package(package: &Path) -> Result<Retained<NSURL>, RuntimeError> {
    let path_str = package.to_str().ok_or_else(|| RuntimeError::LoadFailed {
        path: package.to_path_buf(),
        message: "package path is not valid UTF-8".to_string(),
    })?;

    // SAFETY: `from_str` returns a valid retained NSString; `fileURLWithPath`
    // takes that NSString by reference and is a safe class constructor.
    let ns_path = NSString::from_str(path_str);
    let pkg_url: Retained<NSURL> = NSURL::fileURLWithPath(&ns_path);

    // SAFETY: `compileModelAtURL_error` is a Core ML class method that takes the
    // source-model URL by reference and returns either a Retained<NSURL>
    // pointing at the compiled `.mlmodelc` (which we own) or a Retained<NSError>.
    unsafe { MLModel::compileModelAtURL_error(&pkg_url) }.map_err(|err| RuntimeError::LoadFailed {
        path: package.to_path_buf(),
        message: format!("compileModelAtURL failed: {}", ns_error_message(&err)),
    })
}

/// Load a compiled `.mlmodelc` from a local directory path with `config`.
fn load_model_from_dir(
    compiled_dir: &Path,
    config: &MLModelConfiguration,
) -> Result<Retained<MLModel>, RuntimeError> {
    let path_str = compiled_dir
        .to_str()
        .ok_or_else(|| RuntimeError::LoadFailed {
            path: compiled_dir.to_path_buf(),
            message: "compiled model path is not valid UTF-8".to_string(),
        })?;
    // SAFETY: `from_str` returns a valid retained NSString; `fileURLWithPath`
    // takes it by reference and is a safe class constructor.
    let ns_path = NSString::from_str(path_str);
    let url: Retained<NSURL> = NSURL::fileURLWithPath(&ns_path);
    load_model_from_url(compiled_dir, &url, config)
}

/// Load a compiled `.mlmodelc` from a URL with `config`. `package` is only used
/// for the error path's reported path.
#[allow(unsafe_code)] // Core ML native boundary.
fn load_model_from_url(
    package: &Path,
    compiled_url: &NSURL,
    config: &MLModelConfiguration,
) -> Result<Retained<MLModel>, RuntimeError> {
    // SAFETY: `modelWithContentsOfURL_configuration_error` loads a compiled
    // model from the URL, using our config; both args are borrowed and the call
    // returns an owned MLModel or an NSError.
    unsafe { MLModel::modelWithContentsOfURL_configuration_error(compiled_url, config) }.map_err(
        |err| RuntimeError::LoadFailed {
            path: package.to_path_buf(),
            message: format!("modelWithContentsOfURL failed: {}", ns_error_message(&err)),
        },
    )
}

// ---- compiled-model disk cache -------------------------------------------

/// Name of the cache subdirectory holding compiled `.mlmodelc` bundles, a
/// sibling of the source `.mlpackage` files (mirrors ort's `coreml_cache/`).
const COMPILED_CACHE_DIR: &str = "compiled_cache";

/// Directory holding the compiled-model cache for `package`
/// (`<package.parent>/compiled_cache`).
pub(super) fn compiled_cache_dir(package: &Path) -> PathBuf {
    package
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(COMPILED_CACHE_DIR)
}

/// Cached compiled-model path for `package`
/// (`<package.parent>/compiled_cache/<stem>.mlmodelc`).
pub(super) fn cached_model_path(package: &Path) -> PathBuf {
    let stem = package
        .file_stem()
        .map(|s| s.to_os_string())
        .unwrap_or_else(|| std::ffi::OsString::from("model"));
    let mut name = stem;
    name.push(".mlmodelc");
    compiled_cache_dir(package).join(name)
}

/// Sidecar validity-key path for `package`'s cached model
/// (`<...>/<stem>.mlmodelc.meta`).
pub(super) fn cached_meta_path(package: &Path) -> PathBuf {
    let mut p = cached_model_path(package).into_os_string();
    p.push(".meta");
    PathBuf::from(p)
}

/// macOS product version (e.g. `26.1`), part of the cache key so a Core ML OS
/// update invalidates compiled models. Reads `sw_vers -productVersion`.
fn macos_product_version() -> Result<String, RuntimeError> {
    let out = std::process::Command::new("sw_vers")
        .arg("-productVersion")
        .output()
        .map_err(|e| RuntimeError::LoadFailed {
            path: PathBuf::from("sw_vers"),
            message: format!("failed to run sw_vers: {e}"),
        })?;
    if !out.status.success() {
        return Err(RuntimeError::LoadFailed {
            path: PathBuf::from("sw_vers"),
            message: format!("sw_vers exited with {}", out.status),
        });
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Recursively sum the total byte size and find the newest mtime (as nanoseconds
/// since the UNIX epoch) of every regular file under `root`.
fn dir_size_and_newest_mtime(root: &Path) -> std::io::Result<(u64, u128)> {
    let mut total: u64 = 0;
    let mut newest: u128 = 0;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let ft = entry.file_type()?;
            if ft.is_dir() {
                stack.push(entry.path());
            } else if ft.is_file() {
                let meta = entry.metadata()?;
                total = total.saturating_add(meta.len());
                if let Ok(mtime) = meta.modified()
                    && let Ok(dur) = mtime.duration_since(std::time::UNIX_EPOCH)
                {
                    newest = newest.max(dur.as_nanos());
                }
            }
        }
    }
    Ok((total, newest))
}

/// Build the cache validity key for `package`: source size + newest mtime + OS
/// version. Anything that changes recompiles. Factored out (and `os_version`
/// injectable) so the key build + match logic is unit-testable without Core ML.
///
/// The compute-unit choice (`cpu_and_ne`) is deliberately absent: the compiled
/// `.mlmodelc` is compute-unit-independent — that config is applied at load time
/// via `setComputeUnits`, not at compile time — so one cached artifact serves
/// both `CPUAndNeuralEngine` and `CPUOnly`.
pub(super) fn build_source_key(package: &Path, os_version: &str) -> Result<String, RuntimeError> {
    let (size, mtime) =
        dir_size_and_newest_mtime(package).map_err(|e| RuntimeError::LoadFailed {
            path: package.to_path_buf(),
            message: format!("failed to stat package for cache key: {e}"),
        })?;
    Ok(format!("size={size} mtime_ns={mtime} os={os_version}"))
}

/// Current validity key for `package` using the live macOS version.
pub(super) fn current_source_key(package: &Path) -> Result<String, RuntimeError> {
    build_source_key(package, &macos_product_version()?)
}

/// True when the sidecar at `meta_path` exists and its content equals `key`
/// (trimmed). A missing / unreadable / differing sidecar is a miss.
pub(super) fn meta_matches(meta_path: &Path, key: &str) -> bool {
    match std::fs::read_to_string(meta_path) {
        Ok(content) => content.trim() == key.trim(),
        Err(_) => false,
    }
}

/// Recursively copy directory `src` into `dst` (creating `dst`). Used to copy
/// the temp `.mlmodelc` into the cache staging dir.
pub(super) fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ft = entry.file_type()?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if ft.is_dir() {
            copy_dir_recursive(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

/// Copy the freshly-compiled temp `.mlmodelc` (`temp_dir`) into the disk cache:
/// recursively copy into a unique staging dir under `compiled_cache/`, then
/// atomically rename it into the final cache path (renaming any stale one aside
/// first), and write the sidecar validity key.
///
/// Mirrors `model::extract_ane_tar_atomic`'s staging + atomic-rename +
/// cleanup-on-error discipline: the final cache path only ever appears
/// fully-formed, and a torn copy leaves only the staging dir (removed on every
/// error path). Concurrent compilers are last-writer-wins on identical content.
///
/// Multi-process reader safety: rather than `remove_dir_all(final_dir)` (which
/// would unlink files out from under another process mid-`modelWithContentsOfURL`
/// on the old cache — one process recompiling on a stale key while another loads
/// it), a stale `final_dir` is first `rename`d aside to a unique `.trash.*` dir
/// (an atomic dir-entry swap), THEN the staging dir is renamed into `final_dir`,
/// THEN the trash is removed (best-effort). A concurrent reader keeps reading the
/// now-unlinked-but-still-open inode it already opened, so its load stays valid.
/// A leftover `.trash.*`/`.staging.*` dir (e.g. on crash) is harmless and swept
/// on the next entry.
pub(super) fn populate_cache(package: &Path, temp_dir: &Path) -> std::io::Result<()> {
    let cache_dir = compiled_cache_dir(package);
    std::fs::create_dir_all(&cache_dir)?;

    // Best-effort sweep of leftover staging/trash dirs from a prior crash.
    sweep_stale_temp_dirs(&cache_dir);

    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let pid = std::process::id();
    let staging = cache_dir.join(format!(".staging.{pid}.{stamp}"));

    let cleanup = || {
        let _ = std::fs::remove_dir_all(&staging);
    };

    if let Err(e) = copy_dir_recursive(temp_dir, &staging) {
        cleanup();
        return Err(e);
    }

    let final_dir = cached_model_path(package);
    // Rename any stale cached model ASIDE (not `remove_dir_all`) so a concurrent
    // OTHER-PROCESS reader mid-load keeps its open inode valid; `rename` also
    // requires the destination be absent (or it fails "directory not empty").
    let mut trash: Option<PathBuf> = None;
    if final_dir.exists() {
        let aside = cache_dir.join(format!(".trash.{pid}.{stamp}"));
        if let Err(e) = std::fs::rename(&final_dir, &aside) {
            cleanup();
            return Err(e);
        }
        trash = Some(aside);
    }
    if let Err(e) = std::fs::rename(&staging, &final_dir) {
        cleanup();
        if let Some(aside) = trash {
            let _ = std::fs::remove_dir_all(&aside);
        }
        return Err(e);
    }
    // Drop the old cache now that the new one is in place (best-effort; a
    // leftover trash dir is harmless and swept on the next entry).
    if let Some(aside) = trash {
        let _ = std::fs::remove_dir_all(&aside);
    }

    // Write the sidecar key LAST so a hit requires both a present model and a
    // matching key (a torn run that renamed the model but died before the
    // sidecar simply recompiles next time).
    let key = current_source_key(package).map_err(std::io::Error::other)?;
    std::fs::write(cached_meta_path(package), key)?;
    Ok(())
}

/// Best-effort removal of leftover `.staging.*` / `.trash.*` dirs in `cache_dir`
/// from a prior crashed/torn `populate_cache` run. Never fails the caller.
fn sweep_stale_temp_dirs(cache_dir: &Path) {
    let Ok(entries) = std::fs::read_dir(cache_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with(".staging.") || name.starts_with(".trash.") {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

/// Convert a `file://` `NSURL` to a local filesystem path, or `None` if it is
/// not a file URL with a usable path.
fn url_to_path(url: &NSURL) -> Option<PathBuf> {
    // `path` is a safe getter in this objc2-foundation version; it returns an
    // optional owned NSString (the file-system path of a `file://` URL).
    let ns = url.path()?;
    Some(PathBuf::from(ns.to_string()))
}
