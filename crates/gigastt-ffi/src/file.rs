//! File transcription at the C ABI boundary.

use std::ffi::{CStr, CString, c_char};
use std::ptr;

use crate::handles::{GigasttEngine, get_engine};

/// Transcribe an audio file and return the recognized text as a newly allocated C string.
///
/// # Safety
/// - `engine` must be a live opaque handle returned by an engine constructor.
/// - `wav_path` must be NULL or a readable NUL-terminated string in one allocation,
///   shorter than `isize::MAX` bytes including the terminator and not mutated
///   for the duration of the call. Non-UTF-8 input is rejected.
/// - A concurrent `gigastt_engine_free` during this call is safe: the table
///   holds an `Arc` for the duration of the call. A call after free returns
///   `NULL`.
///
/// Returns a pointer to a NUL-terminated UTF-8 string on success, or `NULL` on failure.
/// The caller **must** free the returned string with `gigastt_string_free`.
#[allow(unsafe_code)] // Exported C ABI boundary.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gigastt_transcribe_file(
    engine: *mut GigasttEngine,
    wav_path: *const c_char,
) -> *mut c_char {
    let Some(slot) = get_engine(engine) else {
        tracing::error!("gigastt_transcribe_file: engine is null or freed");
        eprintln!("gigastt_transcribe_file: engine is null or freed");
        return ptr::null_mut();
    };
    if wav_path.is_null() {
        tracing::error!("gigastt_transcribe_file: wav_path is null");
        eprintln!("gigastt_transcribe_file: wav_path is null");
        return ptr::null_mut();
    }

    // SAFETY: non-null checked above; caller guarantees the readable C string.
    let path_str = match unsafe { CStr::from_ptr(wav_path) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("gigastt_transcribe_file: wav_path is not valid UTF-8: {e}");
            eprintln!("gigastt_transcribe_file: wav_path is not valid UTF-8: {e}");
            return ptr::null_mut();
        }
    };

    // Path sanitization: reject absolute paths, parent-dir traversal, and
    // paths that escape the working directory.
    let path = std::path::Path::new(path_str);
    if path.is_absolute() {
        tracing::error!("gigastt_transcribe_file: absolute paths are not allowed");
        eprintln!("gigastt_transcribe_file: absolute paths are not allowed");
        return ptr::null_mut();
    }
    if path
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        tracing::error!("gigastt_transcribe_file: paths containing '..' are not allowed");
        eprintln!("gigastt_transcribe_file: paths containing '..' are not allowed");
        return ptr::null_mut();
    }
    let cwd = match std::env::current_dir() {
        Ok(d) => d,
        Err(e) => {
            tracing::error!("gigastt_transcribe_file: failed to get working directory: {e}");
            eprintln!("gigastt_transcribe_file: failed to get working directory: {e}");
            return ptr::null_mut();
        }
    };
    let resolved = cwd.join(path);
    // Resolve symlinks before the boundary check; a symlink inside cwd that
    // points outside (e.g., evil.wav → /etc/passwd) must be rejected.
    let canonical = match std::fs::canonicalize(&resolved) {
        Ok(p) => p,
        Err(e) => {
            tracing::error!(
                "gigastt_transcribe_file: path does not exist or is not accessible: {e}"
            );
            eprintln!("gigastt_transcribe_file: path does not exist or is not accessible: {e}");
            return ptr::null_mut();
        }
    };
    if !canonical.starts_with(&cwd) {
        tracing::error!("gigastt_transcribe_file: path escapes working directory");
        eprintln!("gigastt_transcribe_file: path escapes working directory");
        return ptr::null_mut();
    }

    let engine_ref = &slot.engine;

    let mut guard = match engine_ref.pool.checkout_blocking() {
        Ok(g) => g,
        Err(e) => {
            tracing::error!("gigastt_transcribe_file: failed to checkout session from pool: {e}");
            eprintln!("gigastt_transcribe_file: failed to checkout session from pool: {e}");
            return ptr::null_mut();
        }
    };

    let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine_ref.transcribe_file(path_str, &mut guard)
    })) {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            tracing::error!("gigastt_transcribe_file: transcription failed: {e}");
            eprintln!("gigastt_transcribe_file: transcription failed: {e}");
            return ptr::null_mut();
        }
        Err(_) => {
            tracing::error!("gigastt_transcribe_file: panic during transcription");
            eprintln!("gigastt_transcribe_file: panic during transcription");
            return ptr::null_mut();
        }
    };

    match CString::new(result.text) {
        Ok(cstr) => cstr.into_raw(),
        Err(e) => {
            tracing::error!("gigastt_transcribe_file: result contains interior NUL: {e}");
            eprintln!("gigastt_transcribe_file: result contains interior NUL: {e}");
            ptr::null_mut()
        }
    }
}
