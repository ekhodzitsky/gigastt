use super::*;

#[test]
fn test_catch_ffi_panic_swallows_unwind() {
    assert!(catch_ffi_panic("test", || -> i32 { panic!("boom") }).is_none());
    assert_eq!(catch_ffi_panic("test", || 7), Some(7));
}

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
        let sample =
            (440.0_f64 * 2.0 * std::f64::consts::PI * i as f64 / sample_rate as f64).sin() * 1000.0;
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
fn test_stream_new_unknown_handle_is_null() {
    let bogus = 0xFFFF_FFFFu64 as *mut GigasttEngine;
    let stream = unsafe { gigastt_stream_new(bogus) };
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
    let r =
        unsafe { gigastt_stream_process_chunk(engine, stream, pcm16.as_ptr(), pcm16.len(), 16000) };
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
    let r =
        unsafe { gigastt_stream_process_chunk(engine, stream, pcm16.as_ptr(), pcm16.len(), 48000) };
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
    let r =
        unsafe { gigastt_stream_process_chunk(engine, stream, pcm16.as_ptr(), pcm16.len(), 16000) };
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

/// An empty directory holds no encoder, so `Engine::load_*` returns Err and
/// the FFI wrapper must surface that as a NULL handle (engine-load error
/// branch), without needing the real model on disk.
#[test]
fn test_engine_new_with_pool_size_missing_model_returns_null() {
    let tmp = std::env::temp_dir().join(format!("gigastt_ffi_empty_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let dir = CString::new(tmp.to_str().unwrap()).unwrap();
    let engine = unsafe { gigastt_engine_new_with_pool_size(dir.as_ptr(), 1) };
    assert!(engine.is_null());
    let _ = std::fs::remove_dir_all(&tmp);
}

/// The default-pool wrapper funnels through the same load path, so a
/// model-less directory must likewise yield a NULL handle.
#[test]
fn test_engine_new_missing_model_returns_null() {
    let tmp =
        std::env::temp_dir().join(format!("gigastt_ffi_empty_default_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let dir = CString::new(tmp.to_str().unwrap()).unwrap();
    let engine = unsafe { gigastt_engine_new(dir.as_ptr()) };
    assert!(engine.is_null());
    let _ = std::fs::remove_dir_all(&tmp);
}

/// A directory with no recognition-head encoder makes `detect_in_dir`
/// return `None`; quantize must report that as an error string (not "ok"),
/// exercising the no-encoder-found branch.
#[test]
fn test_quantize_model_no_encoder_found() {
    let tmp = std::env::temp_dir().join(format!("gigastt_ffi_quant_empty_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let dir = CString::new(tmp.to_str().unwrap()).unwrap();
    let r = unsafe { gigastt_quantize_model(dir.as_ptr(), false) };
    assert!(!r.is_null());
    let s = unsafe { CStr::from_ptr(r) }.to_str().unwrap();
    assert!(
        s.contains("no recognition-head encoder"),
        "unexpected message: {s}"
    );
    unsafe { gigastt_string_free(r) };
    let _ = std::fs::remove_dir_all(&tmp);
}

/// `force = true` on a model-less directory still has nothing to quantize,
/// so the no-encoder-found branch fires regardless of the force flag.
#[test]
fn test_quantize_model_no_encoder_found_forced() {
    let tmp = std::env::temp_dir().join(format!(
        "gigastt_ffi_quant_empty_force_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&tmp).unwrap();
    let dir = CString::new(tmp.to_str().unwrap()).unwrap();
    let r = unsafe { gigastt_quantize_model(dir.as_ptr(), true) };
    assert!(!r.is_null());
    let s = unsafe { CStr::from_ptr(r) }.to_str().unwrap();
    assert!(
        s.contains("no recognition-head encoder"),
        "unexpected message: {s}"
    );
    unsafe { gigastt_string_free(r) };
    let _ = std::fs::remove_dir_all(&tmp);
}

/// Invalid UTF-8 in `model_dir` must yield a NULL engine handle via the
/// pool-size wrapper's UTF-8 guard.
#[test]
fn test_engine_new_with_pool_size_invalid_utf8() {
    let bad = [0x80u8, 0x81, 0x82, 0];
    let engine = unsafe { gigastt_engine_new_with_pool_size(bad.as_ptr() as *const c_char, 1) };
    assert!(engine.is_null());
}

/// Transcribing with a freed engine must be rejected by the early
/// `disposed` check, returning NULL. Loading a real engine is required to
/// build a disposable handle, so this is model-gated.
#[test]
#[ignore = "requires model"]
fn test_transcribe_file_disposed_engine_returns_null() {
    let dir = CString::new(gigastt_core::model::default_model_dir()).unwrap();
    let engine = unsafe { gigastt_engine_new_with_pool_size(dir.as_ptr(), 1) };
    assert!(!engine.is_null());
    unsafe { gigastt_engine_free(engine) };
    let file = CString::new("audio.wav").unwrap();
    let r = unsafe { gigastt_transcribe_file(engine, file.as_ptr()) };
    assert!(r.is_null());
}

/// With a live engine, a NULL stream pointer must be rejected by the
/// stream-null branch of `process_chunk`. Reaching that branch requires a
/// non-null engine, so the test is model-gated.
#[test]
#[ignore = "requires model"]
fn test_stream_process_chunk_null_stream_with_engine() {
    let engine = shared_test_engine();
    let pcm16: Vec<u8> = vec![0u8; 16];
    let r = unsafe {
        gigastt_stream_process_chunk(engine, ptr::null_mut(), pcm16.as_ptr(), pcm16.len(), 16000)
    };
    assert!(r.is_null());
}

/// With a live engine and stream, a NULL PCM buffer must be rejected by the
/// pcm16-null branch of `process_chunk`.
#[test]
#[ignore = "requires model"]
fn test_stream_process_chunk_null_pcm_with_engine() {
    let engine = shared_test_engine();
    let stream = unsafe { gigastt_stream_new(engine) };
    assert!(!stream.is_null());
    let r = unsafe { gigastt_stream_process_chunk(engine, stream, ptr::null(), 0, 16000) };
    assert!(r.is_null());
    unsafe { gigastt_stream_free(stream) };
}

/// A flush against a NULL stream (with a live engine) hits the stream-null
/// branch and returns NULL.
#[test]
#[ignore = "requires model"]
fn test_stream_flush_null_stream_with_engine() {
    let engine = shared_test_engine();
    let r = unsafe { gigastt_stream_flush(engine, ptr::null_mut()) };
    assert!(r.is_null());
}

#[test]
fn test_stream_flush_tail_failure_returns_null_and_preserves_state() {
    for joiner in [false, true] {
        for panic in [false, true] {
            let tmp = tempfile::tempdir().unwrap();
            gigastt_core::test_support::write_rnnt_layout(tmp.path()).unwrap();
            let factory = gigastt_core::test_support::FailingStreamFactory::new(joiner, panic);
            let core = gigastt_core::test_support::load_rnnt_engine_with_factory(
                tmp.path(),
                1,
                Box::new(factory.clone()),
            )
            .unwrap();
            factory.arm(1);
            let engine = insert_engine(core);
            let stream = unsafe { gigastt_stream_new(engine) };
            let pcm = [0u8; 3200];
            let result = unsafe {
                gigastt_stream_process_chunk(engine, stream, pcm.as_ptr(), pcm.len(), 16000)
            };
            assert!(!result.is_null());
            unsafe { gigastt_string_free(result) };
            assert!(unsafe { gigastt_stream_flush(engine, stream) }.is_null());
            assert!(unsafe { gigastt_stream_flush(engine, stream) }.is_null());
            assert!(
                get_stream(stream)
                    .unwrap()
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .state
                    .is_failed()
            );
            unsafe {
                gigastt_stream_free(stream);
                gigastt_engine_free(engine);
            }
        }
    }
}

#[test]
fn test_engine_new_with_threads_null() {
    let engine = unsafe { gigastt_engine_new_with_threads(ptr::null(), 1, 4) };
    assert!(engine.is_null());
}

#[test]
fn test_engine_new_with_threads_invalid_utf8() {
    let bad = [0x80u8, 0x81, 0x82, 0];
    let engine = unsafe { gigastt_engine_new_with_threads(bad.as_ptr() as *const c_char, 1, 4) };
    assert!(engine.is_null());
}

/// A model-less directory must yield a NULL handle for any thread count,
/// including the out-of-range values 0 and a very large one (both clamped).
#[test]
fn test_engine_new_with_threads_missing_model_returns_null() {
    let tmp = std::env::temp_dir().join(format!("gigastt_ffi_threads_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let dir = CString::new(tmp.to_str().unwrap()).unwrap();
    for threads in [0usize, 1, 4, usize::MAX] {
        let engine = unsafe { gigastt_engine_new_with_threads(dir.as_ptr(), 1, threads) };
        assert!(engine.is_null(), "threads = {threads}");
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
#[ignore = "requires model"]
fn test_engine_new_with_threads_success() {
    let dir = CString::new(gigastt_core::model::default_model_dir()).unwrap();
    for threads in [1usize, 2] {
        let engine = unsafe { gigastt_engine_new_with_threads(dir.as_ptr(), 1, threads) };
        assert!(!engine.is_null(), "threads = {threads}");
        unsafe { gigastt_engine_free(engine) };
    }
}
