//! Float16 ANE prediction over a compiled `MLModel`.

use half::f16;
use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_ml::{
    MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel, MLMultiArray,
    MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSString};

use super::ns_error_message;
use crate::runtime::error::RuntimeError;

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
#[allow(unsafe_code)] // Core ML native boundary.
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
#[allow(unsafe_code)] // Core ML native boundary.
fn shape_of(arr: &MLMultiArray) -> Result<Vec<usize>, RuntimeError> {
    // SAFETY: `shape` returns an owned NSArray<NSNumber>; element access is via
    // safe NSArray/NSNumber getters.
    let ns: Retained<NSArray<NSNumber>> = unsafe { arr.shape() };
    Ok(nsarray_usize(&ns))
}

/// Read the `strides()` NSArray<NSNumber> of an MLMultiArray as element strides.
#[allow(unsafe_code)] // Core ML native boundary.
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
#[allow(unsafe_code)] // Core ML native boundary.
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
#[allow(unsafe_code)] // Core ML native boundary.
fn read_strided_f16(base: *const f16, shape: &[usize], strides: &[usize]) -> Vec<f32> {
    // SAFETY (per element): `off` is in bounds per the contract above.
    read_strided_with(shape, strides, |off| unsafe { (*base.add(off)).to_f32() })
}

/// Read a strided f32 buffer into a row-major `Vec<f32>` over `shape`.
///
/// SAFETY contract: `base` points at a readable f32 buffer where every
/// `sum(idx[d] * strides[d])` offset is in bounds.
#[allow(unsafe_code)] // Core ML native boundary.
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
