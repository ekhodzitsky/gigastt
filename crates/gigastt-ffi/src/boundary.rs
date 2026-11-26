//! Shared C string ownership and panic containment.

use std::ffi::{CString, c_char};

/// Convert a Rust string to a C string, falling back to a static message
/// if the input contains interior NUL bytes.  The fallback literal is
/// guaranteed NUL-free, so this never panics.
pub(super) fn to_cstring(s: &str) -> CString {
    CString::new(s)
        .unwrap_or_else(|_| CString::new("invalid string").expect("static literal is NUL-free"))
}

/// Run `f` and swallow a panic so it cannot unwind across the C ABI.
///
/// A panic across `extern "C"` is undefined behaviour under `panic=unwind`.
/// Every exported entry point must go through this (or an equivalent
/// `catch_unwind`) and return a sentinel (`NULL` / error string) instead.
pub(super) fn catch_ffi_panic<T, F>(label: &'static str, f: F) -> Option<T>
where
    F: FnOnce() -> T,
{
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(value) => Some(value),
        Err(_) => {
            tracing::error!("{label}: panic");
            eprintln!("{label}: panic");
            None
        }
    }
}

/// Free a C string previously returned by the transcription, streaming, or
/// quantization functions.
///
/// # Safety
/// `s` must be the original pointer returned by the library and not yet freed,
/// or `NULL` (a no-op). Do not free with another allocator, change the string's
/// length, or access the allocation concurrently with or after this call.
#[allow(unsafe_code)] // Exported C ABI boundary.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gigastt_string_free(s: *mut c_char) {
    if !s.is_null() {
        // SAFETY: caller transfers the original, uniquely owned CString pointer.
        let _ = unsafe { CString::from_raw(s) };
    }
}
