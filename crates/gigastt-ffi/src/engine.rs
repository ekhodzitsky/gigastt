//! Engine creation and release at the C ABI boundary.

use std::ffi::{CStr, c_char};
use std::ptr;

use gigastt_core::inference::Engine;

use crate::boundary::catch_ffi_panic;
use crate::handles::{GigasttEngine, insert_engine, take_engine};

/// Load the ONNX models from `model_dir` and create an inference engine.
///
/// Uses the default pool size (2). For mobile devices, prefer
/// `gigastt_engine_new_with_pool_size` with `pool_size = 1` to reduce RAM.
///
/// # Safety
/// `model_dir` must be NULL or a readable NUL-terminated string in one allocation,
/// shorter than `isize::MAX` bytes including the terminator and not mutated
/// for the duration of the call. Non-UTF-8 input is rejected.
/// Returns a pointer to a `GigasttEngine` on success, or `NULL` on failure.
#[allow(unsafe_code)] // Exported C ABI boundary.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gigastt_engine_new(model_dir: *const c_char) -> *mut GigasttEngine {
    // SAFETY: the forwarded constructor has the same model_dir contract.
    unsafe { gigastt_engine_new_with_pool_size(model_dir, 2) }
}

/// Load the ONNX models with a custom session pool size.
///
/// `pool_size` controls how many concurrent inference sessions are kept in
/// memory. The INT8 encoder is memory-mapped and shared; an extra slot costs
/// on the order of tens of megabytes resident, not another full model copy:
/// - pool_size = 1: recommended for mobile
/// - pool_size = 2: default desktop/server
///
/// # Safety
/// `model_dir` must be NULL or a readable NUL-terminated string in one allocation,
/// shorter than `isize::MAX` bytes including the terminator and not mutated
/// for the duration of the call. Non-UTF-8 input is rejected.
/// Returns a pointer to a `GigasttEngine` on success, or `NULL` on failure.
#[allow(unsafe_code)] // Exported C ABI boundary.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gigastt_engine_new_with_pool_size(
    model_dir: *const c_char,
    pool_size: usize,
) -> *mut GigasttEngine {
    if model_dir.is_null() {
        tracing::error!("gigastt_engine_new_with_pool_size: model_dir is null");
        eprintln!("gigastt_engine_new_with_pool_size: model_dir is null");
        return ptr::null_mut();
    }

    catch_ffi_panic("gigastt_engine_new_with_pool_size", || {
        // SAFETY: non-null checked above; caller guarantees the readable C string.
        let dir_str = match unsafe { CStr::from_ptr(model_dir) }.to_str() {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(
                    "gigastt_engine_new_with_pool_size: model_dir is not valid UTF-8: {e}"
                );
                eprintln!("gigastt_engine_new_with_pool_size: model_dir is not valid UTF-8: {e}");
                return ptr::null_mut();
            }
        };

        match Engine::load_with_pool_size(dir_str, pool_size) {
            Ok(engine) => insert_engine(engine),
            Err(e) => {
                tracing::error!("gigastt_engine_new_with_pool_size: failed to load engine: {e}");
                eprintln!("gigastt_engine_new_with_pool_size: failed to load engine: {e}");
                ptr::null_mut()
            }
        }
    })
    .unwrap_or(ptr::null_mut())
}

/// Load the ONNX models with a custom session pool size and an explicit
/// encoder intra-op thread count.
///
/// `gigastt_engine_new` and `gigastt_engine_new_with_pool_size` always build
/// the encoder with one intra-op thread, so on a phone the dominant encoder
/// runs on a single core. `encoder_intra_threads > 1` lets ONNX Runtime spread
/// it over several cores. The count is capped by the logical CPU budget per
/// pool slot, with a minimum of one thread per slot; `0` is treated as `1`.
/// A pool larger than the CPU count still uses one thread per slot.
/// `encoder_intra_threads = 1` builds sessions
/// identical to `gigastt_engine_new_with_pool_size`.
///
/// The best value depends on the SoC (big.LITTLE layout); measure it on the
/// target device, for example by trying 1, 2, 4, 6 and 8.
///
/// # Safety
/// `model_dir` must be NULL or a readable NUL-terminated string in one allocation,
/// shorter than `isize::MAX` bytes including the terminator and not mutated
/// for the duration of the call. Non-UTF-8 input is rejected.
/// Returns a pointer to a `GigasttEngine` on success, or `NULL` on failure.
#[allow(unsafe_code)] // Exported C ABI boundary.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gigastt_engine_new_with_threads(
    model_dir: *const c_char,
    pool_size: usize,
    encoder_intra_threads: usize,
) -> *mut GigasttEngine {
    if model_dir.is_null() {
        tracing::error!("gigastt_engine_new_with_threads: model_dir is null");
        eprintln!("gigastt_engine_new_with_threads: model_dir is null");
        return ptr::null_mut();
    }

    catch_ffi_panic("gigastt_engine_new_with_threads", || {
        // SAFETY: non-null checked above; caller guarantees the readable C string.
        let dir_str = match unsafe { CStr::from_ptr(model_dir) }.to_str() {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(
                    "gigastt_engine_new_with_threads: model_dir is not valid UTF-8: {e}"
                );
                eprintln!("gigastt_engine_new_with_threads: model_dir is not valid UTF-8: {e}");
                return ptr::null_mut();
            }
        };

        // Same pool semantics as `gigastt_engine_new_with_pool_size`
        // (min_size = pool_size, no batch pool), plus the thread count.
        match Engine::load_with_pools_threads(
            dir_str,
            pool_size,
            pool_size,
            0,
            encoder_intra_threads,
        ) {
            Ok(engine) => insert_engine(engine),
            Err(e) => {
                tracing::error!("gigastt_engine_new_with_threads: failed to load engine: {e}");
                eprintln!("gigastt_engine_new_with_threads: failed to load engine: {e}");
                ptr::null_mut()
            }
        }
    })
    .unwrap_or(ptr::null_mut())
}

/// Free an inference engine previously created by `gigastt_engine_new`.
///
/// # Safety
/// `engine` must be a live opaque handle returned by an engine constructor,
/// or `NULL` (in which case this is a no-op). Concurrent free during an
/// in-flight call is safe (the call's `Arc` keeps the engine alive).
#[allow(unsafe_code)] // Exported C ABI boundary.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gigastt_engine_free(engine: *mut GigasttEngine) {
    let _dropped = take_engine(engine);
}
