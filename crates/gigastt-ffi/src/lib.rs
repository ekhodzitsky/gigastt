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

mod handles;
pub use handles::{GigasttEngine, GigasttStream};
use handles::{
    StreamSlot, get_engine, get_stream, insert_engine, insert_stream, take_engine, take_stream,
};

/// Convert a Rust string to a C string, falling back to a static message
/// if the input contains interior NUL bytes.  The fallback literal is
/// guaranteed NUL-free, so this never panics.
fn to_cstring(s: &str) -> CString {
    CString::new(s)
        .unwrap_or_else(|_| CString::new("invalid string").expect("static literal is NUL-free"))
}

/// Run `f` and swallow a panic so it cannot unwind across the C ABI.
///
/// A panic across `extern "C"` is undefined behaviour under `panic=unwind`.
/// Every exported entry point must go through this (or an equivalent
/// `catch_unwind`) and return a sentinel (`NULL` / error string) instead.
fn catch_ffi_panic<T, F>(label: &'static str, f: F) -> Option<T>
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

use gigastt_core::inference::{Engine, audio};

/// Load the ONNX models from `model_dir` and create an inference engine.
///
/// Uses the default pool size (2). For mobile devices, prefer
/// `gigastt_engine_new_with_pool_size` with `pool_size = 1` to reduce RAM.
///
/// # Safety
/// `model_dir` must be a valid, null-terminated UTF-8 string.
/// Returns a pointer to a `GigasttEngine` on success, or `NULL` on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gigastt_engine_new(model_dir: *const c_char) -> *mut GigasttEngine {
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

    catch_ffi_panic("gigastt_engine_new_with_pool_size", || {
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
/// `model_dir` must be a valid, null-terminated UTF-8 string.
/// Returns a pointer to a `GigasttEngine` on success, or `NULL` on failure.
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

/// Transcribe an audio file and return the recognized text as a newly allocated C string.
///
/// # Safety
/// - `engine` must be a non-null pointer returned by `gigastt_engine_new` and not yet freed.
/// - `wav_path` must be a valid, null-terminated UTF-8 string.
/// - A concurrent `gigastt_engine_free` during this call is safe: the table
///   holds an `Arc` for the duration of the call. A call after free returns
///   `NULL`.
///
/// Returns a pointer to a NUL-terminated UTF-8 string on success, or `NULL` on failure.
/// The caller **must** free the returned string with `gigastt_string_free`.
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
/// or `NULL` (in which case this is a no-op). Concurrent free during an
/// in-flight call is safe (the call's `Arc` keeps the engine alive).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gigastt_engine_free(engine: *mut GigasttEngine) {
    let _dropped = take_engine(engine);
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

    catch_ffi_panic("gigastt_quantize_model", || {
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
    })
    .unwrap_or_else(|| to_cstring("panic").into_raw())
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
    let Some(slot) = get_engine(engine) else {
        tracing::error!("gigastt_stream_new: engine is null or freed");
        eprintln!("gigastt_stream_new: engine is null or freed");
        return ptr::null_mut();
    };

    catch_ffi_panic("gigastt_stream_new", || {
        let guard = match slot.engine.pool.checkout_blocking() {
            Ok(g) => g,
            Err(e) => {
                tracing::error!("gigastt_stream_new: pool checkout failed: {e}");
                eprintln!("gigastt_stream_new: pool checkout failed: {e}");
                return ptr::null_mut();
            }
        };

        let reservation = guard.into_owned();
        let state = slot.engine.create_state(false);
        insert_stream(StreamSlot {
            state,
            reservation: Some(reservation),
            engine: slot.clone(),
        })
    })
    .unwrap_or(ptr::null_mut())
}

/// Process a chunk of PCM16 audio and return any partial/final segments.
///
/// # Safety
/// - `engine` and `stream` must be valid pointers.
/// - `pcm16_bytes` must point to at least `len` valid bytes (little-endian mono PCM16).
/// - Concurrent free during this call is safe. A call after free returns `NULL`.
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
    let _ = engine; // engine id is optional; the stream holds its own Arc
    let Some(stream_arc) = get_stream(stream) else {
        tracing::error!("gigastt_stream_process_chunk: stream is null or freed");
        return ptr::null_mut();
    };
    if pcm16_bytes.is_null() {
        tracing::error!("gigastt_stream_process_chunk: pcm16_bytes is null");
        return ptr::null_mut();
    }

    let mut stream_guard = stream_arc.lock().unwrap_or_else(|e| e.into_inner());
    let stream_ref = &mut *stream_guard;

    // Convert PCM16 LE bytes → f32 samples.
    let bytes = unsafe { std::slice::from_raw_parts(pcm16_bytes, len) };
    let pcm16: Vec<i16> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| i16::from_le_bytes(*c))
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

    let engine_slot = stream_ref.engine.clone();
    let Some(reservation) = stream_ref.reservation.as_mut() else {
        tracing::error!("gigastt_stream_process_chunk: stream already freed");
        return ptr::null_mut();
    };
    let segments = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine_slot
            .engine
            .process_chunk(&samples_f32, &mut stream_ref.state, reservation)
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

/// Decode pending audio and return the final segment(s).
/// Successful flushes allow further chunks. A failed flush requires a new
/// stream; previously returned text remains incomplete.
///
/// # Safety
/// `stream` must be a pointer returned by `gigastt_stream_new`. Concurrent
/// free during this call is safe. A call after free returns `NULL`.
///
/// Returns a newly allocated JSON array string (possibly `[]`) on success,
/// or `NULL` on failure. The caller **must** free the returned string with
/// `gigastt_string_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gigastt_stream_flush(
    engine: *mut GigasttEngine,
    stream: *mut GigasttStream,
) -> *mut c_char {
    let _ = engine;
    let Some(stream_arc) = get_stream(stream) else {
        tracing::error!("gigastt_stream_flush: stream is null or freed");
        return ptr::null_mut();
    };

    catch_ffi_panic("gigastt_stream_flush", || {
        let mut stream_ref = stream_arc.lock().unwrap_or_else(|e| e.into_inner());
        let engine_slot = stream_ref.engine.clone();
        let StreamSlot {
            state, reservation, ..
        } = &mut *stream_ref;
        let Some(reservation) = reservation.as_mut() else {
            return ptr::null_mut();
        };
        let segments: Vec<_> = match engine_slot.engine.try_finish_stream(state, reservation) {
            Ok(segment) => segment.into_iter().collect(),
            Err(error) => {
                tracing::error!(%error, "gigastt_stream_flush failed");
                return ptr::null_mut();
            }
        };

        let json = serde_json::to_string(&segments).unwrap_or_else(|_| "[]".into());
        match CString::new(json) {
            Ok(cstr) => cstr.into_raw(),
            Err(_) => ptr::null_mut(),
        }
    })
    .unwrap_or(ptr::null_mut())
}

/// Free a streaming session and return its triplet to the pool.
///
/// # Safety
/// `stream` must be a pointer returned by `gigastt_stream_new` and not yet freed,
/// or `NULL` (in which case this is a no-op). Concurrent free during an
/// in-flight call is safe.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gigastt_stream_free(stream: *mut GigasttStream) {
    let _dropped = take_stream(stream);
}

#[cfg(test)]
mod tests;
