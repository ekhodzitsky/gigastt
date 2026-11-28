//! Model quantization at the C ABI boundary.

use std::ffi::{CStr, c_char};

use crate::boundary::{catch_ffi_panic, to_cstring};

/// Quantize the FP32 encoder model to INT8 in-place.
///
/// Auto-detects the recognition head present in `model_dir` (the `rnnt` default
/// since v2.3, or `e2e_rnnt`) and quantizes that variant's FP32 encoder, writing
/// the matching `*_int8.onnx` beside it. If the INT8 file already exists and
/// `force` is `false`, returns immediately.
///
/// # Safety
/// `model_dir` must be NULL or a readable NUL-terminated string in one allocation,
/// shorter than `isize::MAX` bytes including the terminator and not mutated
/// for the duration of the call. Non-UTF-8 input is rejected.
///
/// Returns a newly allocated C string on both success and error.
/// The caller **must** free the returned string with `gigastt_string_free`.
#[allow(unsafe_code)] // Exported C ABI boundary.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gigastt_quantize_model(
    model_dir: *const c_char,
    force: bool,
) -> *mut c_char {
    if model_dir.is_null() {
        tracing::error!("gigastt_quantize_model: model_dir is null");
        eprintln!("gigastt_quantize_model: model_dir is null");
        return to_cstring("model_dir is null").into_raw();
    }

    catch_ffi_panic("gigastt_quantize_model", || {
        // SAFETY: non-null checked above; caller guarantees the readable C string.
        let dir_str = match unsafe { CStr::from_ptr(model_dir) }.to_str() {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("gigastt_quantize_model: model_dir is not valid UTF-8: {e}");
                eprintln!("gigastt_quantize_model: model_dir is not valid UTF-8: {e}");
                let msg = format!("model_dir is not valid UTF-8: {e}");
                return to_cstring(&msg).into_raw();
            }
        };

        let model_dir = std::path::Path::new(dir_str);
        if let Err(error) = gigastt_core::model::reject_versioned_pack(model_dir) {
            return to_cstring(&format!("invalid model pack: {error:#}")).into_raw();
        }
        // Auto-detect the head from whichever encoder is present so this works for
        // the default rnnt model as well as e2e_rnnt.
        let variant = match gigastt_core::model::ModelVariant::detect_in_dir(model_dir) {
            Some(v) => v,
            None => {
                let msg = "no recognition-head encoder found in model_dir";
                tracing::error!("gigastt_quantize_model: {msg}");
                eprintln!("gigastt_quantize_model: {msg}");
                return to_cstring(msg).into_raw();
            }
        };
        let input = model_dir.join(variant.encoder_file());
        let output = model_dir.join(variant.encoder_int8_file());

        if !force && output.exists() {
            return to_cstring("ok").into_raw();
        }

        if let Err(e) = gigastt_core::quantize::quantize_model(&input, &output) {
            tracing::error!("gigastt_quantize_model: quantization failed: {e}");
            eprintln!("gigastt_quantize_model: quantization failed: {e}");
            let msg = format!("quantization failed: {e}");
            return to_cstring(&msg).into_raw();
        }

        to_cstring("ok").into_raw()
    })
    .unwrap_or_else(|| to_cstring("panic").into_raw())
}
