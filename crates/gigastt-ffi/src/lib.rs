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
use std::sync::atomic::{AtomicBool, Ordering};

/// Convert a Rust string to a C string, falling back to a static message
/// if the input contains interior NUL bytes.  The fallback literal is
/// guaranteed NUL-free, so this never panics.
fn to_cstring(s: &str) -> CString {
    CString::new(s)
        .unwrap_or_else(|_| CString::new("invalid string").expect("static literal is NUL-free"))
}

use gigastt_core::inference::{Engine, OwnedReservation, SessionTriplet, StreamingState, audio};

/// Opaque handle to the inference engine.
///
/// The Kotlin side sees this as a `Long` (pointer-sized integer).
pub struct GigasttEngine {
    engine: Engine,
    disposed: AtomicBool,
}

/// Opaque handle to a streaming transcription session.
///
/// Holds a checked-out `SessionTriplet` and a `StreamingState`. The triplet is
/// returned to the pool when `gigastt_stream_free` is called.
pub struct GigasttStream {
    state: StreamingState,
    reservation: OwnedReservation<SessionTriplet>,
    disposed: AtomicBool,
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
            let handle = Box::new(GigasttEngine {
                engine,
                disposed: AtomicBool::new(false),
            });
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
/// - NOT thread-safe (single-threaded-per-handle): no thread may call
///   `gigastt_engine_free` on `engine` concurrently with this call. The early
///   `disposed` check rejects an already-freed handle but does not close the
///   in-call race.
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
    // Early disposed check (Acquire): reject an already-freed engine before any
    // dereference. Does NOT close the in-call race — see the # Safety contract.
    if unsafe { (*engine).disposed.load(Ordering::Acquire) } {
        tracing::error!("gigastt_transcribe_file: engine is disposed");
        eprintln!("gigastt_transcribe_file: engine is disposed");
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

    let engine_ref = unsafe { &(*engine).engine };

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
/// or `NULL` (in which case this is a no-op). NOT thread-safe
/// (single-threaded-per-handle): the caller must ensure no other call using this
/// pointer runs concurrently with this free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gigastt_engine_free(engine: *mut GigasttEngine) {
    if !engine.is_null() {
        let disposed = unsafe { std::ptr::addr_of_mut!((*engine).disposed) };
        if unsafe { (*disposed).swap(true, Ordering::AcqRel) } {
            return;
        }
        let _ = unsafe { Box::from_raw(engine) };
    }
}

// ---------------------------------------------------------------------------
// Quantization API
// ---------------------------------------------------------------------------

/// Quantize the FP32 encoder model to INT8 in-place.
///
/// Auto-detects the recognition head present in `model_dir` (the `rnnt` default
/// since v2.3, or `e2e_rnnt`) and quantizes that variant's FP32 encoder, writing
/// the matching `*_int8.onnx` beside it. If the INT8 file already exists and
/// `force` is `false`, returns immediately.
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
        return to_cstring("model_dir is null").into_raw();
    }

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

    let reservation = guard.into_owned();

    let state = engine_ref.create_state(false);

    let stream = GigasttStream {
        state,
        reservation,
        disposed: AtomicBool::new(false),
    };
    Box::into_raw(Box::new(stream))
}

/// Process a chunk of PCM16 audio and return any partial/final segments.
///
/// # Safety
/// - `engine` and `stream` must be valid pointers.
/// - `pcm16_bytes` must point to at least `len` valid bytes (little-endian mono PCM16).
/// - NOT thread-safe (single-threaded-per-handle): no thread may call
///   `gigastt_engine_free`/`gigastt_stream_free` on these pointers concurrently
///   with this call. The early `disposed` check rejects already-freed handles but
///   does not close the in-call race.
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
    // Early disposed check (Acquire): reject already-freed handles before any
    // dereference. Does NOT close the in-call race — see the # Safety contract.
    if unsafe { (*engine).disposed.load(Ordering::Acquire) }
        || unsafe { (*stream).disposed.load(Ordering::Acquire) }
    {
        tracing::error!("gigastt_stream_process_chunk: engine or stream is disposed");
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
        if let Err(e) = audio::resample_with_cache(
            samples_f32,
            audio::SampleRate(sample_rate),
            audio::SampleRate(16000),
            &mut stream_ref.state.resampler,
            &mut stream_ref.state.resample_output_buf,
        ) {
            tracing::error!("gigastt_stream_process_chunk: resample failed: {e}");
            return ptr::null_mut();
        }
        samples_f32 = std::mem::take(&mut stream_ref.state.resample_output_buf);
    }

    let segments = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine_ref.process_chunk(
            &samples_f32,
            &mut stream_ref.state,
            &mut stream_ref.reservation,
        )
    })) {
        Ok(Ok(segs)) => segs,
        Ok(Err(e)) => {
            tracing::error!("gigastt_stream_process_chunk: inference failed: {e}");
            return ptr::null_mut();
        }
        Err(_) => {
            tracing::error!("gigastt_stream_process_chunk: panic during inference");
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
/// `engine` and `stream` must be valid pointers. NOT thread-safe
/// (single-threaded-per-handle): no thread may call `gigastt_engine_free`/
/// `gigastt_stream_free` on these pointers concurrently with this call. The early
/// `disposed` check rejects already-freed handles but does not close the in-call race.
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
    // Early disposed check (Acquire): reject already-freed handles before any
    // dereference. Does NOT close the in-call race — see the # Safety contract.
    if unsafe { (*engine).disposed.load(Ordering::Acquire) }
        || unsafe { (*stream).disposed.load(Ordering::Acquire) }
    {
        tracing::error!("gigastt_stream_flush: engine or stream is disposed");
        return ptr::null_mut();
    }

    let engine_ref = unsafe { &(*engine).engine };
    let stream_ref = unsafe { &mut (*stream) };

    let segments: Vec<gigastt_core::inference::TranscriptSegment> = engine_ref
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
/// or `NULL` (in which case this is a no-op). NOT thread-safe
/// (single-threaded-per-handle): the caller must ensure no other call using this
/// pointer runs concurrently with this free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gigastt_stream_free(stream: *mut GigasttStream) {
    if !stream.is_null() {
        let disposed = unsafe { std::ptr::addr_of_mut!((*stream).disposed) };
        if unsafe { (*disposed).swap(true, Ordering::AcqRel) } {
            return;
        }
        let stream = unsafe { Box::from_raw(stream) };
        stream.reservation.checkin();
        // `state` is dropped automatically when `stream` goes out of scope.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shared_test_engine() -> *mut GigasttEngine {
        use std::sync::OnceLock;
        struct SendPtr(*mut GigasttEngine);
        // SAFETY: the pointer is written exactly once by `get_or_init` and then
        // only read by test threads.  The underlying `GigasttEngine` is `Send`.
        unsafe impl Send for SendPtr {}
        unsafe impl Sync for SendPtr {}
        static ENGINE: OnceLock<SendPtr> = OnceLock::new();
        ENGINE
            .get_or_init(|| {
                let dir = CString::new(gigastt_core::model::default_model_dir()).unwrap();
                let engine = unsafe { gigastt_engine_new(dir.as_ptr()) };
                assert!(!engine.is_null(), "failed to load engine for ffi tests");
                SendPtr(engine)
            })
            .0
    }

    fn generate_test_wav(duration_s: u32, sample_rate: u32) -> Vec<u8> {
        let num_samples = sample_rate * duration_s;
        let data_size = num_samples * 2;
        let file_size = 44 + data_size;
        let mut wav = Vec::with_capacity(file_size as usize);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(file_size - 8).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());
        for i in 0..num_samples {
            let sample = (440.0_f64 * 2.0 * std::f64::consts::PI * i as f64 / sample_rate as f64)
                .sin()
                * 1000.0;
            wav.extend_from_slice(&(sample as i16).to_le_bytes());
        }
        wav
    }

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

    #[test]
    fn test_engine_new_null() {
        let engine = unsafe { gigastt_engine_new(ptr::null()) };
        assert!(engine.is_null());
    }

    #[test]
    fn test_engine_new_invalid_utf8() {
        let bad = [0x80u8, 0x81, 0x82, 0];
        let engine = unsafe { gigastt_engine_new(bad.as_ptr() as *const c_char) };
        assert!(engine.is_null());
    }

    #[test]
    fn test_engine_free_null() {
        unsafe { gigastt_engine_free(ptr::null_mut()) };
    }

    #[test]
    fn test_transcribe_file_null_engine() {
        let file = CString::new("audio.wav").unwrap();
        let r = unsafe { gigastt_transcribe_file(ptr::null_mut(), file.as_ptr()) };
        assert!(r.is_null());
    }

    #[test]
    fn test_transcribe_file_null_file() {
        let r = unsafe { gigastt_transcribe_file(ptr::null_mut(), ptr::null()) };
        assert!(r.is_null());
    }

    #[test]
    fn test_to_cstring_valid() {
        let c = to_cstring("hello");
        assert_eq!(c.to_str().unwrap(), "hello");
    }

    #[test]
    fn test_to_cstring_with_nul_fallback() {
        let c = to_cstring("hel\0lo");
        assert_eq!(c.to_str().unwrap(), "invalid string");
    }

    #[test]
    #[ignore = "requires model"]
    fn test_transcribe_file_absolute_path_rejected() {
        let engine = shared_test_engine();
        let file = CString::new("/etc/passwd").unwrap();
        let r = unsafe { gigastt_transcribe_file(engine, file.as_ptr()) };
        assert!(r.is_null());
    }

    #[test]
    #[ignore = "requires model"]
    fn test_transcribe_file_parent_dir_rejected() {
        let engine = shared_test_engine();
        let file = CString::new("../secret.wav").unwrap();
        let r = unsafe { gigastt_transcribe_file(engine, file.as_ptr()) };
        assert!(r.is_null());
    }

    #[test]
    #[ignore = "requires model"]
    fn test_transcribe_file_success() {
        let engine = shared_test_engine();
        let wav = generate_test_wav(1, 16000);
        let path = std::path::Path::new("target").join("tmp_ffi_test.wav");
        std::fs::create_dir_all("target").unwrap();
        std::fs::write(&path, &wav).unwrap();
        let c_path = CString::new(path.to_str().unwrap()).unwrap();
        let r = unsafe { gigastt_transcribe_file(engine, c_path.as_ptr()) };
        assert!(!r.is_null(), "transcribe_file returned null");
        let s = unsafe { CStr::from_ptr(r) }.to_str().unwrap();
        // A sine wave is not speech; result may be empty.  Just verify it's valid UTF-8.
        let _ = s;
        unsafe { gigastt_string_free(r) };
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    #[ignore = "requires model"]
    fn test_stream_new_success() {
        let engine = shared_test_engine();
        let stream = unsafe { gigastt_stream_new(engine) };
        assert!(!stream.is_null());
        unsafe { gigastt_stream_free(stream) };
    }

    #[test]
    #[ignore = "requires model"]
    fn test_stream_process_chunk_success() {
        let engine = shared_test_engine();
        let stream = unsafe { gigastt_stream_new(engine) };
        assert!(!stream.is_null());
        // 0.2 s of silence at 16 kHz mono PCM16 = 6400 bytes
        let pcm16: Vec<u8> = vec![0u8; 3200 * 2];
        let r = unsafe {
            gigastt_stream_process_chunk(engine, stream, pcm16.as_ptr(), pcm16.len(), 16000)
        };
        assert!(!r.is_null());
        let s = unsafe { CStr::from_ptr(r) }.to_str().unwrap();
        assert!(s.starts_with('['));
        unsafe { gigastt_string_free(r) };
        unsafe { gigastt_stream_free(stream) };
    }

    #[test]
    #[ignore = "requires model"]
    fn test_stream_process_chunk_resample() {
        let engine = shared_test_engine();
        let stream = unsafe { gigastt_stream_new(engine) };
        assert!(!stream.is_null());
        // 0.2 s of silence at 48 kHz mono PCM16 = 19200 bytes
        let pcm16: Vec<u8> = vec![0u8; 9600 * 2];
        let r = unsafe {
            gigastt_stream_process_chunk(engine, stream, pcm16.as_ptr(), pcm16.len(), 48000)
        };
        assert!(!r.is_null());
        unsafe { gigastt_string_free(r) };
        unsafe { gigastt_stream_free(stream) };
    }

    #[test]
    #[ignore = "requires model"]
    fn test_stream_flush_success() {
        let engine = shared_test_engine();
        let stream = unsafe { gigastt_stream_new(engine) };
        assert!(!stream.is_null());
        let pcm16: Vec<u8> = vec![0u8; 3200 * 2];
        let r = unsafe {
            gigastt_stream_process_chunk(engine, stream, pcm16.as_ptr(), pcm16.len(), 16000)
        };
        assert!(!r.is_null());
        unsafe { gigastt_string_free(r) };
        let r = unsafe { gigastt_stream_flush(engine, stream) };
        assert!(!r.is_null());
        let s = unsafe { CStr::from_ptr(r) }.to_str().unwrap();
        assert!(s.starts_with('['));
        unsafe { gigastt_string_free(r) };
        unsafe { gigastt_stream_free(stream) };
    }

    #[test]
    #[ignore = "requires model"]
    fn test_engine_new_with_pool_size_success() {
        let dir = CString::new(gigastt_core::model::default_model_dir()).unwrap();
        let engine = unsafe { gigastt_engine_new_with_pool_size(dir.as_ptr(), 1) };
        assert!(!engine.is_null());
        unsafe { gigastt_engine_free(engine) };
    }

    #[test]
    #[ignore = "requires model"]
    fn test_quantize_model_idempotent() {
        let dir = CString::new(gigastt_core::model::default_model_dir()).unwrap();
        let r = unsafe { gigastt_quantize_model(dir.as_ptr(), false) };
        assert!(!r.is_null());
        let s = unsafe { CStr::from_ptr(r) }.to_str().unwrap();
        assert_eq!(s, "ok");
        unsafe { gigastt_string_free(r) };
    }

    #[test]
    fn test_string_free_null() {
        unsafe { gigastt_string_free(ptr::null_mut()) };
    }
}
