//! C-ABI FFI layer for Android / JNI integration.
//!
//! Exposes a minimal surface so that Kotlin (or any other JNI consumer) can:
//! 1. Load the inference engine (`gigastt_engine_new`).
//! 2. Transcribe a WAV file (`gigastt_transcribe_file`).
//! 3. Stream audio in real-time (`gigastt_stream_new`, `gigastt_stream_process_chunk`,
//!    `gigastt_stream_flush`).
//! 4. Free the returned C string (`gigastt_string_free`).
//! 5. Tear down the engine (`gigastt_engine_free`).
//!
//! All functions are `unsafe` by nature (raw pointers cross the FFI boundary) but
//! the implementation checks nulls and logs errors before returning sentinel values.

use std::ffi::{CStr, CString, c_char};
use std::ptr;

use crate::inference::{Engine, OwnedReservation, SessionTriplet, StreamingState, audio};

/// Opaque handle to the inference engine.
///
/// The Kotlin side sees this as a `Long` (pointer-sized integer).
pub struct GigasttEngine {
    engine: Engine,
}

/// Opaque handle to a streaming transcription session.
///
/// Holds a checked-out `SessionTriplet` and a `StreamingState`. The triplet is
/// returned to the pool when `gigastt_stream_free` is called.
pub struct GigasttStream {
    state: StreamingState,
    triplet: SessionTriplet,
    reservation: OwnedReservation<SessionTriplet>,
}

/// Load the ONNX models from `model_dir` and create an inference engine.
///
/// Uses the default pool size (4). For mobile devices, prefer
/// `gigastt_engine_new_with_pool_size` with `pool_size = 1` to reduce RAM.
///
/// # Safety
/// `model_dir` must be a valid, null-terminated UTF-8 string.
/// Returns a pointer to a `GigasttEngine` on success, or `NULL` on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gigastt_engine_new(model_dir: *const c_char) -> *mut GigasttEngine {
    unsafe { gigastt_engine_new_with_pool_size(model_dir, 4) }
}

/// Load the ONNX models with a custom session pool size.
///
/// `pool_size` controls how many concurrent inference sessions are kept in
/// memory. Each session loads the full encoder, so RAM scales linearly:
/// - pool_size = 1: ~350 MB (recommended for mobile)
/// - pool_size = 4: ~560 MB (default desktop/server)
///
/// # Safety
/// `model_dir` must be a valid, null-terminated UTF-8 string.
/// Returns a pointer to a `GigasttEngine` on success, or `NULL` on failure.
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

    let dir_str = match unsafe { CStr::from_ptr(model_dir) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("gigastt_engine_new_with_pool_size: model_dir is not valid UTF-8: {e}");
            eprintln!("gigastt_engine_new_with_pool_size: model_dir is not valid UTF-8: {e}");
            return ptr::null_mut();
        }
    };

    match Engine::load_with_pool_size(dir_str, pool_size) {
        Ok(engine) => {
            let handle = Box::new(GigasttEngine { engine });
            Box::into_raw(handle)
        }
        Err(e) => {
            tracing::error!("gigastt_engine_new_with_pool_size: failed to load engine: {e}");
            eprintln!("gigastt_engine_new_with_pool_size: failed to load engine: {e}");
            ptr::null_mut()
        }
    }
}

/// Transcribe an audio file and return the recognized text as a newly allocated C string.
///
/// # Safety
/// - `engine` must be a non-null pointer returned by `gigastt_engine_new` and not yet freed.
/// - `wav_path` must be a valid, null-terminated UTF-8 string.
///
/// Returns a pointer to a NUL-terminated UTF-8 string on success, or `NULL` on failure.
/// The caller **must** free the returned string with `gigastt_string_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gigastt_transcribe_file(
    engine: *mut GigasttEngine,
    wav_path: *const c_char,
) -> *mut c_char {
    if engine.is_null() {
        tracing::error!("gigastt_transcribe_file: engine is null");
        eprintln!("gigastt_transcribe_file: engine is null");
        return ptr::null_mut();
    }
    if wav_path.is_null() {
        tracing::error!("gigastt_transcribe_file: wav_path is null");
        eprintln!("gigastt_transcribe_file: wav_path is null");
        return ptr::null_mut();
    }

    let path_str = match unsafe { CStr::from_ptr(wav_path) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("gigastt_transcribe_file: wav_path is not valid UTF-8: {e}");
            eprintln!("gigastt_transcribe_file: wav_path is not valid UTF-8: {e}");
            return ptr::null_mut();
        }
    };

    let engine_ref = unsafe { &(*engine).engine };

    let mut guard = match engine_ref.pool.checkout_blocking() {
        Ok(g) => g,
        Err(e) => {
            tracing::error!("gigastt_transcribe_file: failed to checkout session from pool: {e}");
            eprintln!("gigastt_transcribe_file: failed to checkout session from pool: {e}");
            return ptr::null_mut();
        }
    };

    let result = match engine_ref.transcribe_file(path_str, &mut guard) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("gigastt_transcribe_file: transcription failed: {e}");
            eprintln!("gigastt_transcribe_file: transcription failed: {e}");
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

/// Free a C string previously returned by `gigastt_transcribe_file` or the
/// streaming functions.
///
/// # Safety
/// `s` must be a pointer returned by one of the transcription functions and not
/// yet freed, or `NULL` (in which case this is a no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gigastt_string_free(s: *mut c_char) {
    if !s.is_null() {
        let _ = unsafe { CString::from_raw(s) };
    }
}

/// Free an inference engine previously created by `gigastt_engine_new`.
///
/// # Safety
/// `engine` must be a pointer returned by `gigastt_engine_new` and not yet freed,
/// or `NULL` (in which case this is a no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gigastt_engine_free(engine: *mut GigasttEngine) {
    if !engine.is_null() {
        let _ = unsafe { Box::from_raw(engine) };
    }
}

// ---------------------------------------------------------------------------
// Quantization API
// ---------------------------------------------------------------------------

/// Quantize the FP32 encoder model to INT8 in-place.
///
/// Looks for `v3_e2e_rnnt_encoder.onnx` inside `model_dir` and produces
/// `v3_e2e_rnnt_encoder_int8.onnx` in the same directory.
/// If the INT8 file already exists and `force` is `false`, returns immediately.
///
/// # Safety
/// `model_dir` must be a valid, null-terminated UTF-8 string.
///
/// Returns a newly allocated C string on both success and error.
/// The caller **must** free the returned string with `gigastt_string_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gigastt_quantize_model(
    model_dir: *const c_char,
    force: bool,
) -> *mut c_char {
    if model_dir.is_null() {
        tracing::error!("gigastt_quantize_model: model_dir is null");
        eprintln!("gigastt_quantize_model: model_dir is null");
        return match CString::new("model_dir is null") {
            Ok(cstr) => cstr.into_raw(),
            Err(_) => CString::new("quantization error").unwrap().into_raw(),
        };
    }

    let dir_str = match unsafe { CStr::from_ptr(model_dir) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("gigastt_quantize_model: model_dir is not valid UTF-8: {e}");
            eprintln!("gigastt_quantize_model: model_dir is not valid UTF-8: {e}");
            let msg = format!("model_dir is not valid UTF-8: {e}");
            return match CString::new(msg) {
                Ok(cstr) => cstr.into_raw(),
                Err(_) => CString::new("model_dir is not valid UTF-8")
                    .unwrap()
                    .into_raw(),
            };
        }
    };

    let model_dir = std::path::Path::new(dir_str);
    let input = model_dir.join("v3_e2e_rnnt_encoder.onnx");
    let output = model_dir.join("v3_e2e_rnnt_encoder_int8.onnx");

    if !force && output.exists() {
        return match CString::new("ok") {
            Ok(cstr) => cstr.into_raw(),
            Err(_) => CString::new("ok").unwrap().into_raw(),
        };
    }

    if let Err(e) = crate::quantize::quantize_model(&input, &output) {
        tracing::error!("gigastt_quantize_model: quantization failed: {e}");
        eprintln!("gigastt_quantize_model: quantization failed: {e}");
        let msg = format!("quantization failed: {e}");
        return match CString::new(msg) {
            Ok(cstr) => cstr.into_raw(),
            Err(_) => CString::new("quantization failed").unwrap().into_raw(),
        };
    }

    match CString::new("ok") {
        Ok(cstr) => cstr.into_raw(),
        Err(_) => CString::new("ok").unwrap().into_raw(),
    }
}

// ---------------------------------------------------------------------------
// Streaming API
// ---------------------------------------------------------------------------

/// Create a new streaming session.
///
/// Checks out a `SessionTriplet` from the engine pool and creates a fresh
/// `StreamingState`. The triplet is held for the lifetime of the stream and
/// returned to the pool by `gigastt_stream_free`.
///
/// # Safety
/// `engine` must be a valid pointer returned by `gigastt_engine_new`.
/// Returns a pointer to a `GigasttStream` on success, or `NULL` on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gigastt_stream_new(engine: *mut GigasttEngine) -> *mut GigasttStream {
    if engine.is_null() {
        tracing::error!("gigastt_stream_new: engine is null");
        eprintln!("gigastt_stream_new: engine is null");
        return ptr::null_mut();
    }

    let engine_ref = unsafe { &(*engine).engine };

    let guard = match engine_ref.pool.checkout_blocking() {
        Ok(g) => g,
        Err(e) => {
            tracing::error!("gigastt_stream_new: pool checkout failed: {e}");
            eprintln!("gigastt_stream_new: pool checkout failed: {e}");
            return ptr::null_mut();
        }
    };

    let (triplet, reservation) = guard.into_owned();

    let state = engine_ref.create_state(false);

    let stream = GigasttStream {
        state,
        triplet,
        reservation,
    };
    Box::into_raw(Box::new(stream))
}

/// Process a chunk of PCM16 audio and return any partial/final segments.
///
/// # Safety
/// - `engine` and `stream` must be valid pointers.
/// - `pcm16_bytes` must point to at least `len` valid bytes (little-endian mono PCM16).
///
/// Returns a newly allocated JSON array string on success, or `NULL` on failure.
/// The caller **must** free the returned string with `gigastt_string_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gigastt_stream_process_chunk(
    engine: *mut GigasttEngine,
    stream: *mut GigasttStream,
    pcm16_bytes: *const u8,
    len: usize,
    sample_rate: u32,
) -> *mut c_char {
    if engine.is_null() {
        tracing::error!("gigastt_stream_process_chunk: engine is null");
        return ptr::null_mut();
    }
    if stream.is_null() {
        tracing::error!("gigastt_stream_process_chunk: stream is null");
        return ptr::null_mut();
    }
    if pcm16_bytes.is_null() {
        tracing::error!("gigastt_stream_process_chunk: pcm16_bytes is null");
        return ptr::null_mut();
    }

    let engine_ref = unsafe { &(*engine).engine };
    let stream_ref = unsafe { &mut (*stream) };

    // Convert PCM16 LE bytes → f32 samples.
    let bytes = unsafe { std::slice::from_raw_parts(pcm16_bytes, len) };
    let pcm16: Vec<i16> = bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    let mut samples_f32: Vec<f32> = pcm16.iter().map(|&s| s as f32 / 32768.0).collect();

    // Resample to 16 kHz if needed.
    if sample_rate != 16000 {
        samples_f32 = match audio::resample(&samples_f32, sample_rate, 16000) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("gigastt_stream_process_chunk: resample failed: {e}");
                return ptr::null_mut();
            }
        };
    }

    let segments = match engine_ref.process_chunk(
        &samples_f32,
        &mut stream_ref.state,
        &mut stream_ref.triplet,
    ) {
        Ok(segs) => segs,
        Err(e) => {
            tracing::error!("gigastt_stream_process_chunk: inference failed: {e}");
            return ptr::null_mut();
        }
    };

    let json = serde_json::to_string(&segments).unwrap_or_else(|_| "[]".into());
    match CString::new(json) {
        Ok(cstr) => cstr.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// Flush the streaming state and return the final segment(s).
///
/// # Safety
/// `engine` and `stream` must be valid pointers.
///
/// Returns a newly allocated JSON array string (possibly `[]`) on success,
/// or `NULL` on failure. The caller **must** free the returned string with
/// `gigastt_string_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gigastt_stream_flush(
    engine: *mut GigasttEngine,
    stream: *mut GigasttStream,
) -> *mut c_char {
    if engine.is_null() {
        tracing::error!("gigastt_stream_flush: engine is null");
        return ptr::null_mut();
    }
    if stream.is_null() {
        tracing::error!("gigastt_stream_flush: stream is null");
        return ptr::null_mut();
    }

    let engine_ref = unsafe { &(*engine).engine };
    let stream_ref = unsafe { &mut (*stream) };

    let segments: Vec<crate::inference::TranscriptSegment> = engine_ref
        .flush_state(&mut stream_ref.state)
        .into_iter()
        .collect();

    let json = serde_json::to_string(&segments).unwrap_or_else(|_| "[]".into());
    match CString::new(json) {
        Ok(cstr) => cstr.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// Free a streaming session and return its triplet to the pool.
///
/// # Safety
/// `stream` must be a pointer returned by `gigastt_stream_new` and not yet freed,
/// or `NULL` (in which case this is a no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gigastt_stream_free(stream: *mut GigasttStream) {
    if !stream.is_null() {
        let stream = unsafe { Box::from_raw(stream) };
        stream.reservation.checkin(stream.triplet);
        // `state` is dropped automatically when `stream` goes out of scope.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_new_null_engine() {
        let stream = unsafe { gigastt_stream_new(ptr::null_mut()) };
        assert!(stream.is_null());
    }

    #[test]
    fn test_stream_process_chunk_null_args() {
        let r = unsafe {
            gigastt_stream_process_chunk(ptr::null_mut(), ptr::null_mut(), ptr::null(), 0, 16000)
        };
        assert!(r.is_null());
    }

    #[test]
    fn test_stream_flush_null_args() {
        let r = unsafe { gigastt_stream_flush(ptr::null_mut(), ptr::null_mut()) };
        assert!(r.is_null());
    }

    #[test]
    fn test_stream_free_null() {
        // Should be a no-op, not a crash.
        unsafe { gigastt_stream_free(ptr::null_mut()) };
    }

    #[test]
    fn test_quantize_model_null_dir() {
        let r = unsafe { gigastt_quantize_model(ptr::null(), false) };
        assert!(!r.is_null());
        let s = unsafe { CStr::from_ptr(r) }.to_str().unwrap();
        assert!(s.contains("null"));
        unsafe { gigastt_string_free(r) };
    }

    #[test]
    fn test_quantize_model_invalid_utf8() {
        let bad = [0x80u8, 0x81, 0x82, 0];
        let r = unsafe { gigastt_quantize_model(bad.as_ptr() as *const c_char, false) };
        assert!(!r.is_null());
        let s = unsafe { CStr::from_ptr(r) }.to_str().unwrap();
        assert!(s.contains("UTF-8"));
        unsafe { gigastt_string_free(r) };
    }
}
