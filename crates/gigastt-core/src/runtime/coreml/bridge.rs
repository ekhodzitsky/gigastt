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

use std::path::Path;

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

/// Compile a `.mlpackage` to a `.mlmodelc` and load it as an `MLModel`.
///
/// A `.mlpackage` must be compiled before loading; for the spike we compile on
/// every call (the production path will cache the compiled URL). When
/// `cpu_and_ne` is set the model is configured with
/// `MLComputeUnits::CPUAndNeuralEngine` so the Apple Neural Engine is engaged.
// `compileModelAtURL_error` is the synchronous compile API; objc2 marks it
// deprecated in favor of the async completion-handler variant, but a synchronous
// compile is exactly what this spike (and the future blocking session) wants.
#[allow(deprecated)]
pub fn compile_and_load(
    package: &Path,
    cpu_and_ne: bool,
) -> Result<Retained<MLModel>, RuntimeError> {
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
    let compiled_url: Retained<NSURL> = unsafe { MLModel::compileModelAtURL_error(&pkg_url) }
        .map_err(|err| RuntimeError::LoadFailed {
            path: package.to_path_buf(),
            message: format!("compileModelAtURL failed: {}", ns_error_message(&err)),
        })?;

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

    // SAFETY: `modelWithContentsOfURL_configuration_error` loads a compiled
    // model from the URL we just produced, using our config; both args are
    // borrowed and the call returns an owned MLModel or an NSError.
    let model: Retained<MLModel> =
        unsafe { MLModel::modelWithContentsOfURL_configuration_error(&compiled_url, &config) }
            .map_err(|err| RuntimeError::LoadFailed {
                path: package.to_path_buf(),
                message: format!("modelWithContentsOfURL failed: {}", ns_error_message(&err)),
            })?;

    Ok(model)
}

/// Run a single prediction: feed an f32 `mel` (logical shape `shape`) as a
/// Float16 `MLMultiArray` keyed by `input_name`, and return the named output
/// (`output_name`) as `(Vec<f32>, Vec<usize>)` = (row-major data, shape).
///
/// The input mel is converted f32 -> f16 on write; the output is read f16 -> f32.
/// Both directions honor the array's reported `strides()` rather than assuming
/// C-contiguity.
// `MLMultiArray::dataPointer` is deprecated in favor of the closure-scoped
// `getBytesWithHandler` / `getMutableBytesWithHandler`, but for a fixed-shape,
// single-threaded spike the raw pointer (read under tight SAFETY notes below) is
// the simplest correct path; a Phase-2b session can switch to the handler API.
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
}
