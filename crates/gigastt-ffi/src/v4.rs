//! Separate optional full-context v4 ABI; legacy handles cannot alias v4 handles.

use std::{ffi::c_char, ptr};

use crate::boundary::{catch_ffi_panic, to_cstring};
use crate::handles::GigasttV4Engine;

#[cfg(all(
    feature = "v4-mkl",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
))]
use {
    crate::handles::{get_v4_engine, insert_v4_engine, take_v4_engine},
    gigastt_core::{error::GigasttError, inference::v4::V4Engine, model::TranscriptStyle},
    std::{
        ffi::CStr,
        path::{Component, Path, PathBuf},
    },
};

/// Result of the explicit v4 constructor. On success `engine` is non-null
/// and `error` is NULL; on failure `engine` is NULL and `error` is an owned
/// JSON error message. Free the error with `gigastt_string_free`.
#[repr(C)]
pub struct GigasttV4EngineResult {
    pub engine: *mut GigasttV4Engine,
    pub error: *mut c_char,
}

fn error_json(code: &str, message: &str) -> *mut c_char {
    to_cstring(&serde_json::json!({"error": {"code": code, "message": message}}).to_string())
        .into_raw()
}

fn construction_error(code: &str, message: &str) -> GigasttV4EngineResult {
    GigasttV4EngineResult {
        engine: ptr::null_mut(),
        error: error_json(code, message),
    }
}

#[cfg(not(all(
    feature = "v4-mkl",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
)))]
const UNSUPPORTED: &str = "v4 requires a Linux x86-64 GNU build with v4-mkl";

#[cfg(all(
    feature = "v4-mkl",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
))]
fn core_error(error: &GigasttError) -> *mut c_char {
    let message = match error {
        GigasttError::ModelLoad { .. } => {
            "Could not load the exact v4 pack and trusted local MKL runtime.".into()
        }
        GigasttError::Io(_) => "Could not access audio file.".into(),
        GigasttError::InvalidAudio { .. } => {
            "Failed to decode audio file. Check format and duration.".into()
        }
        _ => error.to_string(),
    };
    error_json(error.code(), &message)
}

#[cfg(all(
    feature = "v4-mkl",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
))]
#[allow(unsafe_code)]
unsafe fn read_string<'a>(raw: *const c_char) -> Result<&'a str, &'static str> {
    if raw.is_null() {
        return Err("required string is NULL");
    }
    // SAFETY: caller guarantees readable NUL-terminated storage for 'a.
    let string = unsafe { CStr::from_ptr(raw) }
        .to_str()
        .map_err(|_| "required string is not valid UTF-8")?;
    if string.is_empty() {
        return Err("required string is empty");
    }
    Ok(string)
}

#[cfg(all(
    feature = "v4-mkl",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
))]
fn validated_audio_path(value: &str) -> Result<PathBuf, &'static str> {
    let path = Path::new(value);
    if path.is_absolute() || path.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err("audio path must be relative and must not contain '..'");
    }
    let cwd = std::env::current_dir()
        .and_then(std::fs::canonicalize)
        .map_err(|_| "could not resolve working directory")?;
    let canonical = cwd
        .join(path)
        .canonicalize()
        .map_err(|_| "audio path does not exist or is inaccessible")?;
    if !canonical.starts_with(&cwd) {
        return Err("audio path escapes working directory");
    }
    Ok(canonical)
}

/// Whether this library was built with the qualified optional v4 backend.
#[allow(unsafe_code)] // Exported C ABI symbol.
#[unsafe(no_mangle)]
pub extern "C" fn gigastt_v4_supported() -> bool {
    cfg!(all(
        feature = "v4-mkl",
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "gnu"
    ))
}

/// Explicitly load the local v4 RNNT FP32 CPU pack, trusted local MKL runtime
/// and a fixed style (`raw` or `e2e`). No download, discovery or v3 fallback.
///
/// # Safety
/// Each argument must be NULL or point to readable, NUL-terminated UTF-8
/// storage in one allocation, shorter than `isize::MAX` including the NUL,
/// not mutated during the call. A returned engine belongs only to the v4 ABI.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gigastt_v4_engine_new(
    pack_dir: *const c_char,
    mkl_runtime_dir: *const c_char,
    style: *const c_char,
) -> GigasttV4EngineResult {
    catch_ffi_panic("gigastt_v4_engine_new", || {
        #[cfg(all(
            feature = "v4-mkl",
            target_os = "linux",
            target_arch = "x86_64",
            target_env = "gnu"
        ))]
        {
            // SAFETY: C caller guarantees the input string memory contracts.
            let values = unsafe {
                read_string(pack_dir).and_then(|p| {
                    read_string(mkl_runtime_dir).and_then(|m| read_string(style).map(|s| (p, m, s)))
                })
            };
            let (pack, mkl, style) = match values {
                Ok(v) => v,
                Err(e) => return construction_error("invalid_input", e),
            };
            let style = match style.parse::<TranscriptStyle>() {
                Ok(v) => v,
                Err(e) => return construction_error("invalid_input", e),
            };
            match V4Engine::load(Path::new(pack), Path::new(mkl)) {
                Ok(engine) => GigasttV4EngineResult {
                    engine: insert_v4_engine(engine, style),
                    error: ptr::null_mut(),
                },
                Err(error) => GigasttV4EngineResult {
                    engine: ptr::null_mut(),
                    error: core_error(&error),
                },
            }
        }
        #[cfg(not(all(
            feature = "v4-mkl",
            target_os = "linux",
            target_arch = "x86_64",
            target_env = "gnu"
        )))]
        {
            let _ = (pack_dir, mkl_runtime_dir, style);
            construction_error("unsupported_capability", UNSUPPORTED)
        }
    })
    .unwrap_or_else(|| construction_error("inference_error", "v4 construction failed"))
}

/// Transcribe a supported container (at most 30 seconds), returning newly
/// allocated UTF-8 JSON with either a `result` or an `error` member. Free it
/// using `gigastt_string_free`. Audio paths must remain inside the working
/// directory, with no absolute path or parent-directory component.
///
/// # Safety
/// `engine` is a live opaque v4 handle (invalid/freed IDs are rejected).
/// `audio_path` must meet the string contract of `gigastt_v4_engine_new`.
/// A concurrent free is safe after lookup; it does not cancel an in-flight call.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gigastt_v4_transcribe_file(
    engine: *mut GigasttV4Engine,
    audio_path: *const c_char,
) -> *mut c_char {
    catch_ffi_panic("gigastt_v4_transcribe_file", || {
        #[cfg(all(feature="v4-mkl", target_os="linux", target_arch="x86_64", target_env="gnu"))]
        {
            let Some(slot) = get_v4_engine(engine) else { return error_json("invalid_handle", "invalid or freed v4 engine"); };
            // SAFETY: caller guarantees readable C string storage.
            let path = match unsafe { read_string(audio_path) }.and_then(validated_audio_path) { Ok(v)=>v, Err(e)=>return error_json("invalid_input", e) };
            match slot.engine.transcribe_file(&path,slot.style,None) {
                Ok(r) => {
                    let tokens: Vec<_> = r.tokens.iter().map(|t| serde_json::json!({"id":t.id,"frame":t.frame,"confidence":t.confidence})).collect();
                    to_cstring(&serde_json::json!({"result": {"style":r.style.as_str(),"text":r.text,"words":r.words,"tokens":tokens,"duration_s":r.duration_s,"confidence":r.confidence}}).to_string()).into_raw()
                },
                Err(e)=>core_error(&e),
            }
        }
        #[cfg(not(all(feature="v4-mkl", target_os="linux", target_arch="x86_64", target_env="gnu")))]
        { let _=(engine,audio_path); error_json("unsupported_capability",UNSUPPORTED) }
    }).unwrap_or_else(|| error_json("inference_error", "v4 transcription failed"))
}

/// Free an opaque v4 engine handle. NULL, invalid and already freed IDs are
/// no-ops; the v3 and stream handle tables are never affected.
///
/// # Safety
/// Pass only an opaque handle returned by the library or NULL; do not
/// dereference the pointer. Concurrent in-flight recognition may complete.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gigastt_v4_engine_free(engine: *mut GigasttV4Engine) {
    let _ = catch_ffi_panic("gigastt_v4_engine_free", || {
        #[cfg(all(
            feature = "v4-mkl",
            target_os = "linux",
            target_arch = "x86_64",
            target_env = "gnu"
        ))]
        {
            take_v4_engine(engine);
        }
        #[cfg(not(all(
            feature = "v4-mkl",
            target_os = "linux",
            target_arch = "x86_64",
            target_env = "gnu"
        )))]
        let _ = engine;
    });
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;
    use std::ffi::{CStr, CString};

    fn take_error(p: *mut c_char) -> String {
        assert!(!p.is_null());
        // SAFETY: pointer is returned by our ABI and is freed exactly once.
        let value: serde_json::Value =
            serde_json::from_str(unsafe { CStr::from_ptr(p) }.to_str().unwrap()).unwrap();
        unsafe { crate::gigastt_string_free(p) };
        value["error"]["code"].as_str().unwrap().to_owned()
    }

    #[test]
    fn test_v4_constructor_rejects_null_and_invalid_style() {
        let unused = CString::new("unused").unwrap();
        let auto = CString::new("auto").unwrap();
        // SAFETY: test-owned C strings or null pointers.
        let r = unsafe { gigastt_v4_engine_new(unused.as_ptr(), unused.as_ptr(), auto.as_ptr()) };
        assert!(r.engine.is_null());
        assert_eq!(
            take_error(r.error),
            if gigastt_v4_supported() {
                "invalid_input"
            } else {
                "unsupported_capability"
            }
        );
        let r = unsafe { gigastt_v4_engine_new(ptr::null(), ptr::null(), ptr::null()) };
        assert!(r.engine.is_null());
        assert_eq!(
            take_error(r.error),
            if gigastt_v4_supported() {
                "invalid_input"
            } else {
                "unsupported_capability"
            }
        );
    }

    #[test]
    fn test_v4_null_handle_and_free() {
        let p = unsafe { gigastt_v4_transcribe_file(ptr::null_mut(), ptr::null()) };
        assert_eq!(
            take_error(p),
            if gigastt_v4_supported() {
                "invalid_handle"
            } else {
                "unsupported_capability"
            }
        );
        unsafe { gigastt_v4_engine_free(ptr::null_mut()) };
    }

    #[cfg(all(
        feature = "v4-mkl",
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "gnu"
    ))]
    #[test]
    fn test_v4_missing_pack_wrong_family_and_invalid_utf8() {
        let dir = tempfile::tempdir().unwrap();
        let path = CString::new(dir.path().to_str().unwrap()).unwrap();
        let raw = CString::new("raw").unwrap();
        let r = unsafe { gigastt_v4_engine_new(path.as_ptr(), path.as_ptr(), raw.as_ptr()) };
        assert!(r.engine.is_null());
        assert_eq!(take_error(r.error), "model_load_error");
        let bad = [0xffu8, 0];
        let r = unsafe { gigastt_v4_engine_new(bad.as_ptr().cast(), path.as_ptr(), raw.as_ptr()) };
        assert_eq!(take_error(r.error), "invalid_input");
        gigastt_core::test_support::write_rnnt_layout(dir.path()).unwrap();
        let legacy = crate::handles::insert_engine(
            gigastt_core::test_support::load_rnnt_engine(dir.path(), 1).unwrap(),
        );
        let p = unsafe { gigastt_v4_transcribe_file(legacy.cast(), ptr::null()) };
        assert_eq!(take_error(p), "invalid_handle");
        unsafe { gigastt_v4_engine_free(legacy.cast()) };
        assert!(crate::handles::get_engine(legacy).is_some());
        unsafe { crate::gigastt_engine_free(legacy) };
        assert!(validated_audio_path("../outside.wav").is_err());
        assert!(validated_audio_path("/outside.wav").is_err());
    }
}
