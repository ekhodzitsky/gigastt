//! Minimal Rust <-> Core ML bridge over `objc2-core-ml`.
//!
//! Production status: this bridge is the Core ML entry point for the composite
//! ANE runtime — [`super::encoder_session::AneEncoderSession`] (one per pooled
//! production session) calls [`predict_f32`] on every ANE-path encoder run, and
//! [`super::runtime::AneRuntime`] calls [`compile_and_load`] once per bucket at
//! load time. It compiles + loads a per-bucket `.mlpackage`, runs it on the
//! Apple Neural Engine (`CPU_AND_NE`), and produces output that matches a Python
//! `coremltools` reference on the SAME package + input (verified by the
//! `#[ignore]` GO/NO-GO smoke test below). This is the only file in the crate
//! allowed to touch `objc2_core_ml` / `objc2_foundation` (the module enforces
//! the isolation).
//!
//! ISOLATION: all `objc2_*` usage stays inside `runtime/coreml/`.
//! Gated `#[cfg(all(feature = "ane", target_os = "macos"))]`.
//!
//! Every `objc2` call is `unsafe` (Objective-C messaging); `unsafe` blocks are
//! kept tight and each carries a SAFETY note. Failures map to `RuntimeError`
//! variants — never `unwrap` on an objc2 result.

use std::path::{Path, PathBuf};

use half::f16;
use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSString, NSURL};

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
fn compiled_cache_dir(package: &Path) -> PathBuf {
    package
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(COMPILED_CACHE_DIR)
}

/// Cached compiled-model path for `package`
/// (`<package.parent>/compiled_cache/<stem>.mlmodelc`).
fn cached_model_path(package: &Path) -> PathBuf {
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
fn cached_meta_path(package: &Path) -> PathBuf {
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
fn build_source_key(package: &Path, os_version: &str) -> Result<String, RuntimeError> {
    let (size, mtime) =
        dir_size_and_newest_mtime(package).map_err(|e| RuntimeError::LoadFailed {
            path: package.to_path_buf(),
            message: format!("failed to stat package for cache key: {e}"),
        })?;
    Ok(format!("size={size} mtime_ns={mtime} os={os_version}"))
}

/// Current validity key for `package` using the live macOS version.
fn current_source_key(package: &Path) -> Result<String, RuntimeError> {
    build_source_key(package, &macos_product_version()?)
}

/// True when the sidecar at `meta_path` exists and its content equals `key`
/// (trimmed). A missing / unreadable / differing sidecar is a miss.
fn meta_matches(meta_path: &Path, key: &str) -> bool {
    match std::fs::read_to_string(meta_path) {
        Ok(content) => content.trim() == key.trim(),
        Err(_) => false,
    }
}

/// Recursively copy directory `src` into `dst` (creating `dst`). Used to copy
/// the temp `.mlmodelc` into the cache staging dir.
fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
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
fn populate_cache(package: &Path, temp_dir: &Path) -> std::io::Result<()> {
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

/// Run a single prediction: feed an f32 `mel` (logical shape `shape`) as a
/// Float16 `MLMultiArray` keyed by `input_name`, and return the named output
/// (`output_name`) as `(Vec<f32>, Vec<usize>)` = (row-major data, shape).
///
/// The input mel is converted f32 -> f16 on write; the output is read f16 -> f32.
/// Both directions honor the array's reported `strides()` rather than assuming
/// C-contiguity.
// `MLMultiArray::dataPointer` is deprecated in favor of the closure-scoped
// `getBytesWithHandler` / `getMutableBytesWithHandler`, but for a fixed-shape
// array owned exclusively by this call the raw pointer (read under tight SAFETY
// notes below) is the simplest correct path; a later revision could switch to
// the handler API.
#[allow(deprecated)]
pub fn predict_f32(
    model: &MLModel,
    input_name: &str,
    mel: &[f32],
    shape: &[usize],
    output_name: &str,
) -> Result<(Vec<f32>, Vec<usize>), RuntimeError> {
    let expected_len: usize = shape.iter().product();
    if mel.len() != expected_len {
        return Err(RuntimeError::DataLengthMismatch {
            expected: expected_len,
            got: mel.len(),
        });
    }

    // Build the NSArray<NSNumber> shape for the MLMultiArray.
    let dims: Vec<Retained<NSNumber>> = shape.iter().map(|&d| NSNumber::new_usize(d)).collect();
    let ns_shape: Retained<NSArray<NSNumber>> = NSArray::from_retained_slice(&dims);

    // SAFETY: `initWithShape_dataType_error` consumes a freshly allocated
    // MLMultiArray (via `MLMultiArray::alloc()`), takes the shape by reference,
    // and returns an owned, zero-initialized Float16 array or an NSError.
    let input: Retained<MLMultiArray> = unsafe {
        MLMultiArray::initWithShape_dataType_error(
            MLMultiArray::alloc(),
            &ns_shape,
            MLMultiArrayDataType::Float16,
        )
    }
    .map_err(|err| {
        RuntimeError::InferenceFailed(format!(
            "MLMultiArray init failed: {}",
            ns_error_message(&err)
        ))
    })?;

    // Fill the input buffer honoring element strides (counts, not bytes).
    let in_strides = strides_of(&input)?;
    {
        // SAFETY: `dataPointer` returns the backing store of the array we just
        // created and exclusively own; no other reference reads/writes it while
        // this slice is live. We write exactly `mel.len()` f16 values, each at an
        // in-bounds element offset computed from the array's own strides.
        let base = unsafe { input.dataPointer() }.as_ptr() as *mut f16;
        write_strided(base, mel, shape, &in_strides);
    }

    // Wrap the input array in an MLFeatureValue, then a single-entry
    // MLDictionaryFeatureProvider keyed by `input_name`.
    // SAFETY: `featureValueWithMultiArray` borrows the array and returns an owned
    // MLFeatureValue retaining it.
    let feat: Retained<MLFeatureValue> =
        unsafe { MLFeatureValue::featureValueWithMultiArray(&input) };
    let key = NSString::from_str(input_name);
    // The dictionary is typed NSDictionary<NSString, AnyObject>; an MLFeatureValue
    // *is* an AnyObject, so re-borrow it as such for the value slice.
    let value: &AnyObject = &feat;
    let dict: Retained<NSDictionary<NSString, AnyObject>> =
        NSDictionary::from_slices(&[&*key], &[value]);

    // SAFETY: `initWithDictionary_error` consumes a freshly allocated provider,
    // borrows the dictionary, and returns an owned provider or an NSError.
    let provider: Retained<MLDictionaryFeatureProvider> = unsafe {
        MLDictionaryFeatureProvider::initWithDictionary_error(
            MLDictionaryFeatureProvider::alloc(),
            &dict,
        )
    }
    .map_err(|err| {
        RuntimeError::InferenceFailed(format!(
            "feature provider init failed: {}",
            ns_error_message(&err)
        ))
    })?;

    // Erase the concrete provider to the MLFeatureProvider protocol object that
    // `predictionFromFeatures_error` expects (safe reference cast).
    let provider_obj: &ProtocolObject<dyn MLFeatureProvider> = ProtocolObject::from_ref(&*provider);

    // SAFETY: runs synchronous inference; borrows the provider and returns an
    // owned result provider (also an MLFeatureProvider protocol object) or NSError.
    let result: Retained<ProtocolObject<dyn MLFeatureProvider>> =
        unsafe { model.predictionFromFeatures_error(provider_obj) }.map_err(|err| {
            RuntimeError::InferenceFailed(format!("prediction failed: {}", ns_error_message(&err)))
        })?;

    // Pull the named output feature value -> its MLMultiArray.
    let out_key = NSString::from_str(output_name);
    // SAFETY: `featureValueForName` borrows the name and returns an optional
    // owned MLFeatureValue from the result provider.
    let out_feat: Retained<MLFeatureValue> = unsafe { result.featureValueForName(&out_key) }
        .ok_or_else(|| {
            RuntimeError::InferenceFailed(format!("output '{output_name}' missing from result"))
        })?;
    // SAFETY: reads the multi-array payload of the output feature value.
    let out_arr: Retained<MLMultiArray> =
        unsafe { out_feat.multiArrayValue() }.ok_or_else(|| {
            RuntimeError::InferenceFailed(format!("output '{output_name}' is not a multi-array"))
        })?;

    let out_shape = shape_of(&out_arr)?;
    let out_strides = strides_of(&out_arr)?;
    let out_len: usize = out_shape.iter().product();

    // The output element type is whatever the converted model declares (this
    // package declares `encoded` as Float32, even though the input is Float16).
    // Read it from the array rather than assuming, and convert to f32.
    // SAFETY: `dataType` is a plain getter on the model-owned output array.
    let out_dtype = unsafe { out_arr.dataType() };
    // SAFETY: `dataPointer` returns the backing store of the model-owned output
    // array; we read exactly `out_len` elements, each at an in-bounds offset
    // computed from the array's own shape+strides, and the array outlives the read.
    let raw = unsafe { out_arr.dataPointer() }.as_ptr();
    let data = match out_dtype {
        MLMultiArrayDataType::Float16 => {
            read_strided_f16(raw as *const f16, &out_shape, &out_strides)
        }
        MLMultiArrayDataType::Float32 => {
            read_strided_f32(raw as *const f32, &out_shape, &out_strides)
        }
        other => {
            return Err(RuntimeError::InferenceFailed(format!(
                "unsupported output dataType {other:?}"
            )));
        }
    };

    debug_assert_eq!(data.len(), out_len);
    Ok((data, out_shape))
}

// ---- helpers --------------------------------------------------------------

/// Read the `shape()` NSArray<NSNumber> of an MLMultiArray as `Vec<usize>`.
fn shape_of(arr: &MLMultiArray) -> Result<Vec<usize>, RuntimeError> {
    // SAFETY: `shape` returns an owned NSArray<NSNumber>; element access is via
    // safe NSArray/NSNumber getters.
    let ns: Retained<NSArray<NSNumber>> = unsafe { arr.shape() };
    Ok(nsarray_usize(&ns))
}

/// Read the `strides()` NSArray<NSNumber> of an MLMultiArray as element strides.
fn strides_of(arr: &MLMultiArray) -> Result<Vec<usize>, RuntimeError> {
    // SAFETY: `strides` returns an owned NSArray<NSNumber> (element strides, not
    // byte strides); element access is via safe getters.
    let ns: Retained<NSArray<NSNumber>> = unsafe { arr.strides() };
    Ok(nsarray_usize(&ns))
}

fn nsarray_usize(ns: &NSArray<NSNumber>) -> Vec<usize> {
    let n = ns.count();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let num = ns.objectAtIndex(i);
        out.push(num.as_usize());
    }
    out
}

/// Write `data` (logical row-major over `shape`) into a strided f16 buffer.
///
/// SAFETY contract: `base` points at a writable f16 buffer large enough that
/// every `sum(idx[d] * strides[d])` offset is in bounds (true for an
/// MLMultiArray of `shape` with `strides`). Caller holds exclusive access.
fn write_strided(base: *mut f16, data: &[f32], shape: &[usize], strides: &[usize]) {
    let rank = shape.len();
    let total = data.len();
    let mut idx = vec![0usize; rank];
    for &v in data.iter().take(total) {
        let mut off = 0usize;
        for d in 0..rank {
            off += idx[d] * strides[d];
        }
        // SAFETY: `off` is in bounds per the contract above; exclusive access.
        unsafe { *base.add(off) = f16::from_f32(v) };
        // increment the row-major multi-index
        for d in (0..rank).rev() {
            idx[d] += 1;
            if idx[d] < shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
}

/// Read a strided f16 buffer into a row-major `Vec<f32>` over `shape`.
///
/// SAFETY contract: `base` points at a readable f16 buffer where every
/// `sum(idx[d] * strides[d])` offset is in bounds.
fn read_strided_f16(base: *const f16, shape: &[usize], strides: &[usize]) -> Vec<f32> {
    // SAFETY (per element): `off` is in bounds per the contract above.
    read_strided_with(shape, strides, |off| unsafe { (*base.add(off)).to_f32() })
}

/// Read a strided f32 buffer into a row-major `Vec<f32>` over `shape`.
///
/// SAFETY contract: `base` points at a readable f32 buffer where every
/// `sum(idx[d] * strides[d])` offset is in bounds.
fn read_strided_f32(base: *const f32, shape: &[usize], strides: &[usize]) -> Vec<f32> {
    // SAFETY (per element): `off` is in bounds per the contract above.
    read_strided_with(shape, strides, |off| unsafe { *base.add(off) })
}

/// Walk a row-major multi-index over `shape`, calling `read(off)` with the
/// strided element offset for each position; collects the results.
fn read_strided_with(
    shape: &[usize],
    strides: &[usize],
    mut read: impl FnMut(usize) -> f32,
) -> Vec<f32> {
    let rank = shape.len();
    let total: usize = shape.iter().product();
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; rank];
    for _ in 0..total {
        let mut off = 0usize;
        for d in 0..rank {
            off += idx[d] * strides[d];
        }
        out.push(read(off));
        for d in (0..rank).rev() {
            idx[d] += 1;
            if idx[d] < shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    out
}

/// Extract a human-readable message from an NSError without leaking it to clients
/// (used only for internal `RuntimeError` messages / test diagnostics).
fn ns_error_message(err: &objc2_foundation::NSError) -> String {
    // `localizedDescription` is a safe getter in this objc2-foundation version;
    // it returns an owned NSString describing the error.
    err.localizedDescription().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::Instant;

    fn package_path() -> PathBuf {
        let home = std::env::var("HOME").expect("HOME set");
        PathBuf::from(home).join(".gigastt/models/ane/gigaam_v3_encoder_768.mlpackage")
    }

    // ---- pure cache-logic tests (no Core ML / hardware) ------------------

    #[test]
    fn cache_paths_are_sibling_compiled_cache_dir() {
        let pkg = Path::new("/models/ane/gigaam_v3_encoder_768.mlpackage");
        assert_eq!(
            compiled_cache_dir(pkg),
            PathBuf::from("/models/ane/compiled_cache")
        );
        assert_eq!(
            cached_model_path(pkg),
            PathBuf::from("/models/ane/compiled_cache/gigaam_v3_encoder_768.mlmodelc")
        );
        assert_eq!(
            cached_meta_path(pkg),
            PathBuf::from("/models/ane/compiled_cache/gigaam_v3_encoder_768.mlmodelc.meta")
        );
    }

    #[test]
    fn build_source_key_changes_with_size_and_os() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let pkg = tmp.path().join("pkg.mlpackage");
        fs::create_dir_all(pkg.join("Data")).unwrap();
        fs::write(pkg.join("Data").join("weight.bin"), b"abc").unwrap();

        let key_a = build_source_key(&pkg, "26.1").expect("key a");

        // Same source + OS -> identical key (hit).
        let key_a2 = build_source_key(&pkg, "26.1").expect("key a2");
        assert_eq!(key_a, key_a2, "same source+OS must produce the same key");

        // Different OS version -> different key (miss after OS update).
        let key_os = build_source_key(&pkg, "27.0").expect("key os");
        assert_ne!(key_a, key_os, "OS version must be part of the key");

        // Larger source (changed byte size) -> different key (miss).
        fs::write(pkg.join("Data").join("weight.bin"), b"abcdef").unwrap();
        let key_size = build_source_key(&pkg, "26.1").expect("key size");
        assert_ne!(key_a, key_size, "changed source size must change the key");
    }

    #[test]
    fn meta_matches_only_on_exact_key() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let meta = tmp.path().join("model.mlmodelc.meta");

        // Missing sidecar -> miss.
        assert!(!meta_matches(&meta, "size=10 mtime_ns=5 os=26.1"));

        fs::write(&meta, "size=10 mtime_ns=5 os=26.1\n").unwrap();
        // Trailing newline is tolerated (trimmed) -> hit.
        assert!(meta_matches(&meta, "size=10 mtime_ns=5 os=26.1"));
        // Any difference -> miss.
        assert!(!meta_matches(&meta, "size=11 mtime_ns=5 os=26.1"));
        assert!(!meta_matches(&meta, "size=10 mtime_ns=5 os=27.0"));
    }

    #[test]
    fn copy_dir_recursive_reproduces_tree() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let src = tmp.path().join("src");
        let dst = tmp.path().join("dst");
        fs::create_dir_all(src.join("nested")).unwrap();
        fs::write(src.join("top.bin"), b"top").unwrap();
        fs::write(src.join("nested").join("inner.bin"), b"inner").unwrap();

        copy_dir_recursive(&src, &dst).expect("copy");

        assert_eq!(fs::read(dst.join("top.bin")).unwrap(), b"top");
        assert_eq!(
            fs::read(dst.join("nested").join("inner.bin")).unwrap(),
            b"inner"
        );
    }

    #[test]
    fn populate_cache_atomically_places_model_and_sidecar() {
        // No Core ML: stage a fake "compiled" dir + a fake source package, then
        // assert populate_cache mirrors it into compiled_cache/ with a sidecar
        // whose key matches the current source key (so a subsequent hit works).
        let tmp = tempfile::tempdir().expect("tempdir");
        let pkg = tmp.path().join("gigaam_v3_encoder_768.mlpackage");
        fs::create_dir_all(&pkg).unwrap();
        fs::write(pkg.join("Manifest.json"), b"{}").unwrap();

        let temp_compiled = tmp.path().join("temp.mlmodelc");
        fs::create_dir_all(temp_compiled.join("model")).unwrap();
        fs::write(temp_compiled.join("coremldata.bin"), b"compiled").unwrap();
        fs::write(temp_compiled.join("model").join("net.bin"), b"net").unwrap();

        populate_cache(&pkg, &temp_compiled).expect("populate");

        let cached = cached_model_path(&pkg);
        assert!(cached.is_dir(), "cached model dir must exist");
        assert_eq!(
            fs::read(cached.join("coremldata.bin")).unwrap(),
            b"compiled"
        );
        assert_eq!(
            fs::read(cached.join("model").join("net.bin")).unwrap(),
            b"net"
        );

        // The sidecar must match the current source key -> meta_matches hit.
        let key = current_source_key(&pkg).expect("source key");
        assert!(
            meta_matches(&cached_meta_path(&pkg), &key),
            "sidecar key must match the current source key after populate_cache"
        );

        // No staging dirs left behind.
        let leftover: Vec<_> = fs::read_dir(compiled_cache_dir(&pkg))
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with(".staging"))
            .collect();
        assert!(leftover.is_empty(), "no .staging dirs must remain");
    }

    fn ref_dir() -> PathBuf {
        PathBuf::from("/tmp/gigaam-ane-spike/bridge_ref")
    }

    fn read_f32(path: &Path) -> Vec<f32> {
        let bytes = fs::read(path).expect("read f32 file");
        assert_eq!(bytes.len() % 4, 0, "f32 file length not a multiple of 4");
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn read_shapes(path: &Path) -> (Vec<usize>, Vec<usize>) {
        let txt = fs::read_to_string(path).expect("read shapes.txt");
        let mut in_shape = Vec::new();
        let mut out_shape = Vec::new();
        for line in txt.lines() {
            let mut it = line.split_whitespace();
            match it.next() {
                Some("in") => in_shape = it.map(|s| s.parse().unwrap()).collect(),
                Some("out") => out_shape = it.map(|s| s.parse().unwrap()).collect(),
                _ => {}
            }
        }
        (in_shape, out_shape)
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            dot += x as f64 * y as f64;
            na += x as f64 * x as f64;
            nb += y as f64 * y as f64;
        }
        dot / (na.sqrt() * nb.sqrt())
    }

    /// GO/NO-GO smoke test. Touches the filesystem + ANE, so `#[ignore]`d like the
    /// e2e tests; run manually:
    ///   cargo test -p gigastt-core --features ane bridge -- --ignored --nocapture
    #[test]
    #[ignore = "requires the 768 bucket .mlpackage + Python bridge_ref/; runs on ANE"]
    fn bridge_loads_predicts_matches_python_reference() {
        let pkg = package_path();
        let refd = ref_dir();
        if !pkg.exists() {
            eprintln!("SKIP: missing package {pkg:?} (run convert_gigaam_ane.py --buckets 768)");
            return;
        }
        if !refd.join("shapes.txt").exists() {
            eprintln!("SKIP: missing {refd:?}/shapes.txt (run dump_bridge_ref.py)");
            return;
        }

        let (in_shape, ref_out_shape) = read_shapes(&refd.join("shapes.txt"));
        let mel = read_f32(&refd.join("mel_in.f32"));
        let ref_out = read_f32(&refd.join("encoded_ref.f32"));
        assert_eq!(
            in_shape,
            vec![1, 64, 768],
            "unexpected reference input shape"
        );

        let model = compile_and_load(&pkg, true).expect("compile_and_load");

        let (out, out_shape) =
            predict_f32(&model, "mel", &mel, &in_shape, "encoded").expect("predict_f32");

        println!("out_shape={out_shape:?} ref_out_shape={ref_out_shape:?}");
        assert_eq!(
            out_shape, ref_out_shape,
            "output shape mismatch vs Python ref"
        );
        assert_eq!(
            out.len(),
            ref_out.len(),
            "output length mismatch vs Python ref"
        );
        assert!(
            out.iter().all(|v| v.is_finite()),
            "output has non-finite values"
        );

        let cos = cosine(&out, &ref_out);
        let max_abs = out
            .iter()
            .zip(ref_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("cosine={cos:.6}  max_abs={max_abs:.6}");
        assert!(cos > 0.999, "cosine {cos:.6} <= 0.999 vs Python reference");

        // RTFx: warm 4x, then time ~12 predicts. audio_secs = N/100 (mel hop 10ms).
        for _ in 0..4 {
            let _ = predict_f32(&model, "mel", &mel, &in_shape, "encoded").expect("warm predict");
        }
        let iters = 12;
        let mut times_ms = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            let _ = predict_f32(&model, "mel", &mel, &in_shape, "encoded").expect("timed predict");
            times_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_ms = times_ms[times_ms.len() / 2];
        let audio_secs = in_shape[2] as f64 / 100.0;
        let rtfx = audio_secs / (median_ms / 1000.0);
        println!("median_ms={median_ms:.3}  audio_secs={audio_secs:.3}  RTFx={rtfx:.1}");
    }

    /// Cold-start cache GO/NO-GO. Loads the SAME package twice in one process:
    /// the FIRST load compiles (~20 s) and populates the cache; the SECOND load
    /// is a cache hit (no compile). Asserts the 2nd load is dramatically faster
    /// AND produces byte-identical output (caching must not change results).
    /// Touches the filesystem + ANE, so `#[ignore]`d; run manually:
    ///   cargo test -p gigastt-core --features ane bridge_disk_cache -- --ignored --nocapture
    ///
    /// Hermetic: rather than wipe the developer's real warm cache, this
    /// symlinks the real `.mlpackage` into a fresh tempdir and compiles from
    /// there, so the cache derives to `<tmp>/compiled_cache/` (the cache path is
    /// `package.parent()/compiled_cache`). The real cache is never touched; the
    /// tempdir is removed on `TempDir` drop.
    #[test]
    #[ignore = "requires the 768 bucket .mlpackage; compiles on ANE (~20s first load)"]
    fn bridge_disk_cache_skips_recompile_and_preserves_output() {
        let real_pkg = package_path();
        if !real_pkg.exists() {
            eprintln!(
                "SKIP: missing package {real_pkg:?} (run convert_gigaam_ane.py --buckets 768)"
            );
            return;
        }

        // Hermetic workspace: symlink the real package into a fresh tempdir so
        // the cache derives to <tmp>/compiled_cache/, leaving the real
        // ~/.gigastt/models/ane/compiled_cache/ untouched (TempDir cleans up).
        let tmp = tempfile::tempdir().expect("tempdir");
        let pkg = tmp.path().join("gigaam_v3_encoder_768.mlpackage");
        std::os::unix::fs::symlink(&real_pkg, &pkg).expect("symlink package into tempdir");

        let real_cache = compiled_cache_dir(&real_pkg);
        let real_cache_existed = real_cache.exists();

        // The cache must start empty inside the hermetic tempdir.
        let cache_dir = compiled_cache_dir(&pkg);
        assert!(!cached_model_path(&pkg).exists(), "cache must start empty");
        assert_eq!(
            cache_dir,
            tmp.path().join("compiled_cache"),
            "cache must derive inside the tempdir, not the real cache"
        );

        // Fixed input: a deterministic ramp over the 768 bucket's mel shape.
        let in_shape = vec![1usize, 64, 768];
        let n: usize = in_shape.iter().product();
        let mel: Vec<f32> = (0..n).map(|i| (i as f32 % 17.0) * 0.01).collect();

        // First load: cold compile + cache populate.
        let t0 = Instant::now();
        let model1 = compile_and_load(&pkg, true).expect("first compile_and_load");
        let first_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let (out1, shape1) =
            predict_f32(&model1, "mel", &mel, &in_shape, "encoded").expect("predict 1");

        assert!(
            cached_model_path(&pkg).exists(),
            "first load must populate the disk cache"
        );
        let key = current_source_key(&pkg).expect("source key");
        assert!(
            meta_matches(&cached_meta_path(&pkg), &key),
            "sidecar must match the current source key after first load"
        );

        // Second load: cache hit, no compile.
        let t1 = Instant::now();
        let model2 = compile_and_load(&pkg, true).expect("second compile_and_load");
        let second_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let (out2, shape2) =
            predict_f32(&model2, "mel", &mel, &in_shape, "encoded").expect("predict 2");

        println!("cold_start_first_ms={first_ms:.1}  cache_hit_second_ms={second_ms:.1}");

        // The cache hit must be dramatically faster than the cold compile.
        assert!(
            first_ms > 5_000.0,
            "expected cold compile > 5s, got {first_ms:.1} ms"
        );
        assert!(
            second_ms < 2_000.0,
            "expected cache-hit load < 2s, got {second_ms:.1} ms"
        );
        assert!(
            second_ms < first_ms / 2.0,
            "cache hit ({second_ms:.1} ms) must be much faster than cold ({first_ms:.1} ms)"
        );

        // Caching must not change results: byte-identical output both times.
        assert_eq!(shape1, shape2, "output shape changed across cache hit");
        assert_eq!(
            out1.len(),
            out2.len(),
            "output length changed across cache hit"
        );
        assert_eq!(
            out1.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
            out2.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
            "cache hit must produce byte-identical output"
        );

        // Hermetic guarantee: caching happened in the tempdir, so the real
        // cache must be in the exact state we found it (this test never created
        // or wiped the developer's warm ~/.gigastt cache).
        assert_eq!(
            real_cache.exists(),
            real_cache_existed,
            "the real cache dir must be untouched by this test"
        );
    }
}
