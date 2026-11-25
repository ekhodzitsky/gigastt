use super::*;
use std::io::Write;

fn write_file(path: &Path, bytes: &[u8]) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(bytes).unwrap();
}

fn write_identity_model(path: &Path) {
    // ONNX IR 8, opset 13: Identity(x: float[1]) -> y: float[1].
    write_file(path, b"\x08\x08\x3a\x40\x0a\x10\x0a\x01x\x12\x01y\x22\x08Identity\x12\x0acache-test\x5a\x0f\x0a\x01x\x12\x0a\x0a\x08\x08\x01\x12\x04\x0a\x02\x08\x01\x62\x0f\x0a\x01y\x12\x0a\x0a\x08\x08\x01\x12\x04\x0a\x02\x08\x01\x42\x02\x10\x0d");
}

fn optimized_cache_path(cache_dir: &Path, model: &Path) -> std::path::PathBuf {
    cached_runtime(cache_dir).optimized_cache.unwrap().path(
        &crate::model::optimized_source_hash(model).unwrap().unwrap(),
        1,
        false,
    )
}

#[test]
#[cfg_attr(miri, ignore = "calls into onnxruntime FFI")]
fn test_ort_f32_workspace_errors_preserve_destinations() {
    use crate::runtime::tensor::{Shape, TensorData};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("identity.onnx");
    write_identity_model(&path);
    let runtime = cached_runtime(dir.path());
    let session = runtime.load_session(&path, false).unwrap();
    let inputs = [Tensor::new(Shape::new(vec![1]), TensorData::F32(vec![3.0])).unwrap()];
    let mut first = vec![9.0];
    let mut second = vec![8.0];
    assert!(
        session
            .run_f32_into(&inputs, &mut [&mut first, &mut second])
            .is_err()
    );
    assert_eq!(first, [9.0]);
    assert_eq!(second, [8.0]);
    assert!(session.run_f32_into(&[], &mut [&mut first]).is_err());
    assert_eq!(first, [9.0]);
    // Change both ONNX tensor element types from float to int64, while
    // preserving the single-element shapes and Identity graph.
    let mut model = std::fs::read(&path).unwrap();
    for index in 0..model.len() - 5 {
        if model[index..index + 5] == [0x0a, 0x08, 0x08, 0x01, 0x12] {
            model[index + 3] = 7;
        }
    }
    let integer_path = dir.path().join("integer.onnx");
    std::fs::write(&integer_path, model).unwrap();
    let integer = runtime.load_session(&integer_path, false).unwrap();
    let inputs = [Tensor::new(Shape::new(vec![1]), TensorData::I64(vec![3])).unwrap()];
    assert!(integer.run_f32_into(&inputs, &mut [&mut first]).is_err());
    assert_eq!(first, [9.0]);
}

fn cached_runtime(cache_dir: &Path) -> OrtRuntime {
    OrtRuntime::new(
        super::super::factory::shared_environment().unwrap(),
        1,
        OrtExecutionProvider::Cpu,
        true,
        None,
        Some(cache_dir.to_path_buf()),
    )
}

#[test]
#[cfg_attr(miri, ignore = "calls into onnxruntime FFI")]
fn test_cache_changed_contents_with_preserved_size_and_time() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("encoder.onnx");
    let cache_dir = tmp.path().join("cache");
    write_identity_model(&model);
    let runtime = cached_runtime(&cache_dir);
    drop(runtime.load_session(&model, true).unwrap());
    let modified = std::fs::metadata(&model).unwrap().modified().unwrap();
    let old = std::fs::read(&model).unwrap();
    let replacement = String::from_utf8(old.clone())
        .unwrap()
        .replace("Identity", "Softsign");
    assert_eq!(old.len(), replacement.len());
    write_file(&model, replacement.as_bytes());
    std::fs::File::options()
        .write(true)
        .open(&model)
        .unwrap()
        .set_modified(modified)
        .unwrap();
    let session = runtime.load_session(&model, true).unwrap();
    let input = Tensor::new_checked(
        crate::runtime::tensor::Shape::new(vec![1]),
        crate::runtime::tensor::TensorData::F32(vec![42.0]),
    );
    let expected = Tensor::new_checked(
        crate::runtime::tensor::Shape::new(vec![1]),
        crate::runtime::tensor::TensorData::F32(vec![42.0 / 43.0]),
    );
    assert_eq!(session.run(&[input]).unwrap(), vec![expected]);
}

#[test]
#[cfg_attr(miri, ignore = "calls into onnxruntime FFI")]
fn test_cache_same_name_different_directories_and_thread_settings() {
    let tmp = tempfile::tempdir().unwrap();
    let first = tmp.path().join("one/encoder.onnx");
    let second = tmp.path().join("two/encoder.onnx");
    let cache_dir = tmp.path().join("cache");
    write_identity_model(&first);
    let bytes = String::from_utf8(std::fs::read(&first).unwrap()).unwrap();
    write_file(&second, bytes.replace("Identity", "Softsign").as_bytes());
    let runtime = cached_runtime(&cache_dir);
    drop(runtime.load_session(&first, true).unwrap());
    drop(runtime.load_session(&second, true).unwrap());
    let mut other = cached_runtime(&cache_dir);
    other.intra_threads = 2;
    drop(other.load_session(&first, true).unwrap());
    assert_eq!(std::fs::read_dir(&cache_dir).unwrap().count(), 3);
    assert_ne!(
        optimized_cache_path(&cache_dir, &first),
        optimized_cache_path(&cache_dir, &second)
    );
}

#[test]
#[cfg_attr(miri, ignore = "calls into onnxruntime FFI")]
fn test_cache_legacy_graph_is_not_loaded() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("encoder.onnx");
    let cache_dir = tmp.path().join("cache");
    write_identity_model(&model);
    let legacy = cache_dir.join("encoder_optimized.ort");
    write_file(&legacy, b"old graph");
    drop(
        cached_runtime(&cache_dir)
            .load_session(&model, true)
            .unwrap(),
    );
    assert_eq!(std::fs::read(&legacy).unwrap(), b"old graph");
    assert_eq!(std::fs::read_dir(&cache_dir).unwrap().count(), 2);
}

#[test]
#[cfg_attr(miri, ignore = "calls into onnxruntime FFI")]
fn test_cache_stale_digest_hint_does_not_publish_source_graph() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("encoder.onnx");
    let cache_dir = tmp.path().join("cache");
    write_identity_model(&model);
    drop(
        cached_runtime(&cache_dir)
            .load_session_with_model_hash(&model, true, Some(&"f".repeat(64)))
            .unwrap(),
    );
    assert_eq!(std::fs::read_dir(&cache_dir).unwrap().count(), 0);
}

#[test]
#[cfg_attr(miri, ignore = "calls into onnxruntime FFI")]
fn test_cache_malformed_hash_hint_cannot_escape_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("encoder.onnx");
    let cache_dir = tmp.path().join("cache");
    write_identity_model(&model);
    let runtime = cached_runtime(&cache_dir);
    for hint in ["../escape", "/outside", "not-a-digest"] {
        drop(
            runtime
                .load_session_with_model_hash(&model, true, Some(hint))
                .unwrap(),
        );
    }
    assert!(optimized_cache_path(&cache_dir, &model).is_file());
    assert_eq!(std::fs::read_dir(&cache_dir).unwrap().count(), 1);
    assert_eq!(std::fs::read_dir(tmp.path()).unwrap().count(), 2);
}

fn protobuf_bytes(field: u8, data: &[u8]) -> Vec<u8> {
    let mut out = vec![field << 3 | 2];
    let mut len = data.len();
    while len >= 128 {
        out.push((len as u8 & 127) | 128);
        len >>= 7;
    }
    out.push(len as u8);
    out.extend_from_slice(data);
    out
}

#[test]
#[cfg_attr(miri, ignore = "calls into onnxruntime FFI")]
fn test_cache_external_weights_changes_are_observed() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("encoder.onnx");
    let weights = tmp.path().join("weights.bin");
    let cache_dir = tmp.path().join("cache");
    // Add(x, w) -> y with w stored outside the ModelProto.
    let mut node = protobuf_bytes(1, b"x");
    node.extend(protobuf_bytes(1, b"w"));
    node.extend(protobuf_bytes(2, b"y"));
    node.extend(protobuf_bytes(4, b"Add"));
    let mut tensor = vec![8, 1, 16, 1]; // dims=[1], FLOAT
    tensor.extend(protobuf_bytes(8, b"w"));
    let mut location = protobuf_bytes(1, b"location");
    location.extend(protobuf_bytes(2, b"weights.bin"));
    tensor.extend(protobuf_bytes(13, &location));
    tensor.extend([112, 1]); // data_location=EXTERNAL
    let mut graph = protobuf_bytes(1, &node);
    graph.extend(protobuf_bytes(2, b"external-test"));
    graph.extend(protobuf_bytes(5, &tensor));
    graph.extend(protobuf_bytes(
        11,
        b"\x0a\x01x\x12\x0a\x0a\x08\x08\x01\x12\x04\x0a\x02\x08\x01",
    ));
    graph.extend(protobuf_bytes(
        12,
        b"\x0a\x01y\x12\x0a\x0a\x08\x08\x01\x12\x04\x0a\x02\x08\x01",
    ));
    let mut bytes = vec![8, 8];
    bytes.extend(protobuf_bytes(7, &graph));
    bytes.extend([66, 2, 16, 13]);
    write_file(&model, &bytes);
    let runtime = cached_runtime(&cache_dir);
    for value in [42.0f32, 10.0] {
        write_file(&weights, &value.to_le_bytes());
        let session = runtime.load_session(&model, true).unwrap();
        let input = Tensor::new_checked(
            crate::runtime::tensor::Shape::new(vec![1]),
            crate::runtime::tensor::TensorData::F32(vec![1.0]),
        );
        let expected = Tensor::new_checked(
            crate::runtime::tensor::Shape::new(vec![1]),
            crate::runtime::tensor::TensorData::F32(vec![value + 1.0]),
        );
        assert_eq!(session.run(&[input]).unwrap(), vec![expected]);
    }
    assert!(!cache_dir.exists());
}

#[test]
#[cfg_attr(miri, ignore = "calls into onnxruntime FFI")]
fn test_cache_invalid_source_leaves_no_partial_files() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("encoder.onnx");
    let cache_dir = tmp.path().join("cache");
    write_file(&model, b"invalid ONNX");
    assert!(
        cached_runtime(&cache_dir)
            .load_session(&model, true)
            .is_err()
    );
    assert_eq!(std::fs::read_dir(&cache_dir).unwrap().count(), 0);
}

#[test]
#[cfg_attr(miri, ignore = "calls into onnxruntime FFI")]
fn test_cache_broken_graph_is_replaced_on_source_load() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("encoder.onnx");
    let cache_dir = tmp.path().join("cache");
    write_identity_model(&model);
    let cache = optimized_cache_path(&cache_dir, &model);
    write_file(&cache, b"invalid ORT");
    let runtime = cached_runtime(&cache_dir);
    assert!(
        runtime
            .try_load_cached_encoder(
                &model,
                &crate::model::optimized_source_hash(&model)
                    .unwrap()
                    .unwrap()
            )
            .is_none()
    );
    // A failed reader must not delete a path another process can replace.
    assert_eq!(std::fs::read(&cache).unwrap(), b"invalid ORT");
    drop(runtime.load_session(&model, true).unwrap());
    assert!(
        runtime
            .try_load_cached_encoder(
                &model,
                &crate::model::optimized_source_hash(&model)
                    .unwrap()
                    .unwrap()
            )
            .unwrap()
            .is_ok()
    );
    assert_eq!(std::fs::read_dir(&cache_dir).unwrap().count(), 1);
}

#[cfg(unix)]
#[test]
#[cfg_attr(miri, ignore = "calls into onnxruntime FFI")]
fn test_cache_refresh_replaces_file_without_truncating_readers() {
    use std::os::unix::fs::MetadataExt;

    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("encoder.onnx");
    let cache_dir = tmp.path().join("cache");
    write_identity_model(&model);
    let cache = optimized_cache_path(&cache_dir, &model);
    let runtime = cached_runtime(&cache_dir);
    drop(runtime.load_session(&model, true).unwrap());
    let reader = std::fs::File::open(&cache).unwrap();
    // Replace the cached graph with invalid bytes. Keep the old mapped inode
    // open; recovery must publish another file rather than truncating it.
    std::fs::remove_file(&cache).unwrap();
    write_file(&cache, b"broken cache");

    drop(runtime.load_session(&model, true).unwrap());

    assert_ne!(
        reader.metadata().unwrap().ino(),
        std::fs::metadata(&cache).unwrap().ino()
    );
    assert!(
        runtime
            .try_load_cached_encoder(
                &model,
                &crate::model::optimized_source_hash(&model)
                    .unwrap()
                    .unwrap()
            )
            .unwrap()
            .is_ok()
    );
    assert_eq!(std::fs::read_dir(&cache_dir).unwrap().count(), 1);
}

#[test]
#[cfg_attr(miri, ignore = "calls into onnxruntime FFI")]
fn test_cache_concurrent_cold_loads_leave_reusable_graph() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("encoder.onnx");
    let cache_dir = tmp.path().join("cache");
    write_identity_model(&model);
    let runtime = cached_runtime(&cache_dir);
    let barrier = std::sync::Barrier::new(4);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    barrier.wait();
                    runtime.load_session(&model, true).unwrap()
                })
            })
            .collect();
        for handle in handles {
            let session = handle.join().unwrap();
            let input = Tensor::new_checked(
                crate::runtime::tensor::Shape::new(vec![1]),
                crate::runtime::tensor::TensorData::F32(vec![42.0]),
            );
            assert_eq!(
                session.run(std::slice::from_ref(&input)).unwrap(),
                vec![input]
            );
        }
    });
    assert!(
        cached_runtime(&cache_dir)
            .try_load_cached_encoder(
                &model,
                &crate::model::optimized_source_hash(&model)
                    .unwrap()
                    .unwrap()
            )
            .unwrap()
            .is_ok()
    );
    assert_eq!(std::fs::read_dir(&cache_dir).unwrap().count(), 1);
}

#[test]
#[cfg_attr(miri, ignore = "calls into onnxruntime FFI")]
fn test_cache_publication_failure_keeps_loaded_session() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("encoder.onnx");
    let cache_dir = tmp.path().join("cache");
    write_identity_model(&model);
    // A directory at the destination makes publication fail on every OS.
    std::fs::create_dir_all(optimized_cache_path(&cache_dir, &model)).unwrap();
    assert!(
        cached_runtime(&cache_dir)
            .load_session(&model, true)
            .is_ok()
    );
    assert_eq!(std::fs::read_dir(&cache_dir).unwrap().count(), 1);
}
