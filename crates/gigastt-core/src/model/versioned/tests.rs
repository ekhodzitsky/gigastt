use super::*;
use std::fs;

fn document(pins: &[ArtifactPin<'_>]) -> String {
    let mut text = format!(
        r#"schema_version = 1
family = "gigaam_v4"
head = "rnnt"
styles = ["raw", "e2e"]
mode = "full_context"
precision = "fp32"
provider = "cpu"
abi = "v4_rnnt_split_v1"
external_data = []

[provenance]
source_repository = "{SOURCE_REPOSITORY}"
source_revision = "{SOURCE_REVISION}"
model_repository = "{MODEL_REPOSITORY}"
model_revision = "{MODEL_REVISION}"
checkpoint_sha256 = "{CHECKPOINT_SHA256}"
license = "MIT"
"#
    );
    for pin in pins {
        text.push_str(&format!(
            "\n[artifacts.{}]\nfile = \"{}.bin\"\nbytes = {}\nsha256 = \"{}\"\n",
            pin.role.as_str(),
            pin.role.as_str(),
            pin.bytes,
            pin.sha256,
        ));
    }
    text
}

fn fake_artifacts(dir: &Path) -> Vec<(ArtifactRole, u64, String)> {
    V4_SPLIT_PINS
        .iter()
        .enumerate()
        .map(|(i, pin)| {
            let path = dir.join(format!("{}.bin", pin.role.as_str()));
            let bytes = format!("unique test artifact {i}");
            fs::write(&path, bytes.as_bytes()).unwrap();
            let sha256 = crate::model::download::hash_file_sha256(&path).unwrap();
            (pin.role, bytes.len() as u64, sha256)
        })
        .collect()
}

fn test_pins(artifacts: &[(ArtifactRole, u64, String)]) -> Vec<ArtifactPin<'_>> {
    artifacts
        .iter()
        .map(|(role, bytes, sha256)| ArtifactPin {
            role: *role,
            bytes: *bytes,
            sha256,
        })
        .collect()
}

#[test]
fn test_versioned_descriptor_parse_and_derived_abi() {
    let descriptor = VersionedPackDescriptor::parse(&document(&V4_SPLIT_PINS)).unwrap();
    assert_eq!(descriptor.family, PackFamily::GigaamV4);
    assert_eq!(descriptor.head, PackHead::Rnnt);
    assert_eq!(descriptor.precision, ModelPrecision::Fp32);
    assert_eq!(descriptor.artifacts.len(), 7);
    let abi = descriptor.abi.specification();
    assert_eq!(abi.opset, 17);
    assert_eq!(abi.sample_rate, 16_000);
    assert_eq!(abi.mel_bins, 64);
    assert_eq!(abi.encoder_channels, 768);
    assert_eq!(abi.predictor_hidden, 320);
    assert_eq!(abi.blank_id, 1024);
    assert_eq!(abi.max_symbols_per_step, 10);
    assert_eq!(abi.joiner_window, 8);
    assert_eq!(abi.max_batch_size, 1);
    assert!(abi.encoder_cache.is_empty());
    assert_eq!(abi.tokenizer_format, "sentencepiece");
    assert_eq!(abi.encoder_outputs[0].name, "encoded");
    assert_eq!(
        abi.encoder_outputs[0].axes,
        &[
            TensorAxis::Batch,
            TensorAxis::Fixed(768),
            TensorAxis::Frames
        ]
    );
    assert_eq!(
        abi.joint_outputs[0].axes,
        &[
            TensorAxis::Batch,
            TensorAxis::Frames,
            TensorAxis::Fixed(1),
            TensorAxis::Fixed(1025)
        ]
    );
}

#[test]
fn test_versioned_descriptor_rejects_unsupported_declared_contracts() {
    let text = document(&V4_SPLIT_PINS);
    for (old, new) in [
        ("schema_version = 1", "schema_version = 2"),
        ("family = \"gigaam_v4\"", "family = \"gigaam_v3\""),
        ("head = \"rnnt\"", "head = \"ctc\""),
        ("styles = [\"raw\", \"e2e\"]", "styles = [\"raw\", \"raw\"]"),
        ("mode = \"full_context\"", "mode = \"streaming\""),
        ("precision = \"fp32\"", "precision = \"int8\""),
        ("provider = \"cpu\"", "provider = \"auto\""),
        ("abi = \"v4_rnnt_split_v1\"", "abi = \"v4_fused\""),
        (SOURCE_REVISION, "deadbeef00000000000000000000000000000000"),
        (MODEL_REPOSITORY, "unverified/model"),
        ("license = \"MIT\"", "license = \"Other\""),
        (
            "schema_version = 1",
            "schema_version = 1\narchitecture = \"rnnt\"",
        ),
        ("license = \"MIT\"", "license = \"MIT\"\nunknown = true"),
        (
            "file = \"encoder.bin\"",
            "file = \"encoder.bin\"\nunknown = true",
        ),
        (
            "external_data = []",
            "external_data = [{file = \"other.bin\", bytes = 1, sha256 = \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"}]",
        ),
    ] {
        assert!(
            VersionedPackDescriptor::parse(&text.replace(old, new)).is_err(),
            "accepted {new}"
        );
    }
    assert!(VersionedPackDescriptor::parse(&text.replacen("schema_version = 1\n", "", 1)).is_err());
    assert!(
        VersionedPackDescriptor::parse(&text.replacen("family = \"gigaam_v4\"\n", "", 1)).is_err()
    );
    let without_raw_joint = document(
        &V4_SPLIT_PINS
            .iter()
            .copied()
            .filter(|p| p.role != ArtifactRole::RawJoint)
            .collect::<Vec<_>>(),
    );
    assert!(VersionedPackDescriptor::parse(&without_raw_joint).is_err());
    let repeated_role = format!(
        "{text}\n[artifacts.encoder]\nfile = \"another.bin\"\nbytes = 2\nsha256 = \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"\n"
    );
    assert!(VersionedPackDescriptor::parse(&repeated_role).is_err());
}

#[test]
fn test_versioned_descriptor_rejects_invalid_artifact_declarations() {
    let text = document(&V4_SPLIT_PINS);
    for name in [
        "../encoder.bin",
        "/tmp/encoder.bin",
        "C:\\encoder.bin",
        "..",
        ".",
        "",
        "bad/name.onnx",
        "raw_decoder.bin",
        "RAW_DECODER.BIN",
        "manifest.toml",
        " space.bin",
        "trailing.",
    ] {
        let changed = text.replace("\"encoder.bin\"", &format!("{name:?}"));
        assert!(
            VersionedPackDescriptor::parse(&changed).is_err(),
            "accepted {name}"
        );
    }
    assert!(
        VersionedPackDescriptor::parse(&text.replacen(V4_SPLIT_PINS[0].sha256, "abc", 1)).is_err()
    );
    assert!(
        VersionedPackDescriptor::parse(&text.replacen(
            &format!("bytes = {}", V4_SPLIT_PINS[0].bytes),
            "bytes = 0",
            1
        ))
        .is_err()
    );
}

#[test]
fn test_versioned_pack_checks_all_artifact_bytes_and_selected_paths() {
    let dir = tempfile::tempdir().unwrap();
    let artifacts = fake_artifacts(dir.path());
    let pins = test_pins(&artifacts);
    let descriptor = VersionedPackDescriptor::parse(&document(&pins)).unwrap();
    let qualified = descriptor.validate_with_pins(dir.path(), &pins).unwrap();
    assert_eq!(qualified.identity().len(), 64);
    let raw = qualified
        .select(PackSelection {
            style: TranscriptStyle::Raw,
            mode: PackMode::FullContext,
            precision: ModelPrecision::Fp32,
            provider: ExecutionProviderChoice::Cpu,
            batch_size: 1,
        })
        .unwrap();
    let e2e = qualified
        .select(PackSelection {
            style: TranscriptStyle::E2e,
            mode: PackMode::FullContext,
            precision: ModelPrecision::Fp32,
            provider: ExecutionProviderChoice::Cpu,
            batch_size: 1,
        })
        .unwrap();
    assert_eq!(raw.encoder, e2e.encoder);
    let canonical_dir = dir.path().canonicalize().unwrap();
    assert_eq!(raw.decoder, canonical_dir.join("raw_decoder.bin"));
    assert_eq!(e2e.decoder, canonical_dir.join("e2e_decoder.bin"));
    assert_ne!(raw.cache_identity, e2e.cache_identity);
    // A same-length change to an unselected head invalidates the entire pack.
    let path = dir.path().join("e2e_joint.bin");
    let mut contents = fs::read(&path).unwrap();
    contents[0] ^= 1;
    fs::write(&path, contents).unwrap();
    assert!(
        format!(
            "{:#}",
            descriptor
                .validate_with_pins(dir.path(), &pins)
                .unwrap_err()
        )
        .contains("SHA-256")
    );
    fs::write(&path, b"wrong size").unwrap();
    assert!(descriptor.validate_with_pins(dir.path(), &pins).is_err());
    fs::remove_file(&path).unwrap();
    assert!(descriptor.validate_with_pins(dir.path(), &pins).is_err());
}

#[test]
fn test_versioned_tokenizer_is_reverified_after_pack_validation() {
    let dir = tempfile::tempdir().unwrap();
    let artifacts = fake_artifacts(dir.path());
    let pins = test_pins(&artifacts);
    let descriptor = VersionedPackDescriptor::parse(&document(&pins)).unwrap();
    let qualified = descriptor.validate_with_pins(dir.path(), &pins).unwrap();
    for (style, file) in [
        (TranscriptStyle::Raw, "raw_tokenizer.bin"),
        (TranscriptStyle::E2e, "e2e_tokenizer.bin"),
    ] {
        let path = dir.path().join(file);
        let mut bytes = fs::read(&path).unwrap();
        assert_eq!(qualified.read_tokenizer_bytes(style).unwrap(), bytes);
        bytes[0] ^= 1;
        fs::write(&path, bytes).unwrap();
        assert!(qualified.read_tokenizer_bytes(style).is_err());
    }
}

#[test]
fn test_versioned_pack_production_pins_reject_self_declared_replacements() {
    let dir = tempfile::tempdir().unwrap();
    let artifacts = fake_artifacts(dir.path());
    let pins = test_pins(&artifacts);
    let descriptor = VersionedPackDescriptor::parse(&document(&pins)).unwrap();
    assert!(
        descriptor
            .validate_in_dir(dir.path())
            .unwrap_err()
            .to_string()
            .contains("not qualified")
    );
    // Even equal-sized actual raw/e2e exports are role-specific.
    let mut swapped = VersionedPackDescriptor::parse(&document(&V4_SPLIT_PINS)).unwrap();
    let raw = swapped.artifacts[&ArtifactRole::RawDecoder].clone();
    let e2e = swapped.artifacts[&ArtifactRole::E2eDecoder].clone();
    swapped.artifacts.insert(ArtifactRole::RawDecoder, e2e);
    swapped.artifacts.insert(ArtifactRole::E2eDecoder, raw);
    assert!(
        swapped
            .validate_in_dir(dir.path())
            .unwrap_err()
            .to_string()
            .contains("not qualified")
    );
}

#[test]
fn test_versioned_pack_identity_is_location_and_spelling_independent() {
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    let artifacts = fake_artifacts(first.path());
    let pins = test_pins(&artifacts);
    let descriptor = VersionedPackDescriptor::parse(&document(&pins)).unwrap();
    let a = descriptor.validate_with_pins(first.path(), &pins).unwrap();
    let mut renamed = descriptor.clone();
    renamed.styles.reverse();
    for entry in renamed.artifacts.values_mut() {
        let old = entry.file.clone();
        entry.file = format!("renamed-{old}");
        fs::copy(first.path().join(old), second.path().join(&entry.file)).unwrap();
    }
    let b = renamed.validate_with_pins(second.path(), &pins).unwrap();
    assert_eq!(a.identity(), b.identity());
    // In the private test contract only: a new, internally consistent unselected
    // head must produce a different complete identity.
    let entry = renamed.artifacts.get_mut(&ArtifactRole::E2eJoint).unwrap();
    fs::write(second.path().join(&entry.file), b"another test artifact").unwrap();
    entry.bytes = fs::metadata(second.path().join(&entry.file)).unwrap().len();
    entry.sha256 =
        crate::model::download::hash_file_sha256(&second.path().join(&entry.file)).unwrap();
    let new_pins: Vec<_> = renamed
        .artifacts
        .iter()
        .map(|(role, a)| ArtifactPin {
            role: *role,
            bytes: a.bytes,
            sha256: &a.sha256,
        })
        .collect();
    let c = renamed
        .validate_with_pins(second.path(), &new_pins)
        .unwrap();
    assert_ne!(a.identity(), c.identity());
}

#[test]
fn test_versioned_pack_selection_rejects_unqualified_mode_precision_provider_and_batch() {
    let dir = tempfile::tempdir().unwrap();
    let artifacts = fake_artifacts(dir.path());
    let pins = test_pins(&artifacts);
    let qualified = VersionedPackDescriptor::parse(&document(&pins))
        .unwrap()
        .validate_with_pins(dir.path(), &pins)
        .unwrap();
    let base = PackSelection {
        style: TranscriptStyle::Raw,
        mode: PackMode::FullContext,
        precision: ModelPrecision::Fp32,
        provider: ExecutionProviderChoice::Cpu,
        batch_size: 1,
    };
    assert!(qualified.select(base).is_ok());
    for request in [
        PackSelection {
            mode: PackMode::Streaming,
            ..base
        },
        PackSelection {
            precision: ModelPrecision::Int8,
            ..base
        },
        PackSelection {
            provider: ExecutionProviderChoice::Auto,
            ..base
        },
        PackSelection {
            provider: ExecutionProviderChoice::Coreml,
            ..base
        },
        PackSelection {
            provider: ExecutionProviderChoice::Cuda,
            ..base
        },
        PackSelection {
            batch_size: 0,
            ..base
        },
        PackSelection {
            batch_size: 2,
            ..base
        },
    ] {
        assert!(qualified.select(request).is_err());
    }
}

#[cfg(unix)]
#[test]
fn test_versioned_pack_refuses_symlink_artifact_even_with_matching_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let other = tempfile::tempdir().unwrap();
    let artifacts = fake_artifacts(dir.path());
    let pins = test_pins(&artifacts);
    let descriptor = VersionedPackDescriptor::parse(&document(&pins)).unwrap();
    let path = dir.path().join("encoder.bin");
    let outside = other.path().join("encoder.bin");
    fs::copy(&path, &outside).unwrap();
    fs::remove_file(&path).unwrap();
    std::os::unix::fs::symlink(outside, &path).unwrap();
    assert!(descriptor.validate_with_pins(dir.path(), &pins).is_err());
}

#[test]
#[ignore = "requires a local qualified v4 model pack; hash-only, no inference"]
fn test_qualified_v4_split_pack_local_hash_only() {
    // Separate opt-in smoke, intentionally not part of normal model-free CI.
    let dir = std::env::var_os("GIGASTT_QUALIFIED_V4_PACK_DIR").expect("local qualified pack path");
    let pack = QualifiedV4Pack::load(Path::new(&dir)).unwrap();
    assert_eq!(pack.descriptor().family, PackFamily::GigaamV4);
    assert_eq!(pack.identity().len(), 64);
    assert!(crate::model::ModelManifest::load(Path::new(&dir)).is_err());
    println!("qualified v4 pack identity: {}", pack.identity());
    for style in [TranscriptStyle::Raw, TranscriptStyle::E2e] {
        let selected = pack
            .select(PackSelection {
                style,
                mode: PackMode::FullContext,
                precision: ModelPrecision::Fp32,
                provider: ExecutionProviderChoice::Cpu,
                batch_size: 1,
            })
            .unwrap();
        println!(
            "selected {style:?} v4 identity: {}",
            selected.cache_identity
        );
        assert!(selected.encoder.is_file());
        assert!(selected.decoder.is_file());
        assert!(selected.joint.is_file());
        assert!(selected.tokenizer.is_file());
    }
}

#[cfg(all(
    feature = "v4-mkl",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
))]
#[test]
fn test_versioned_graph_reverified_bytes_and_role() {
    let dir = tempfile::tempdir().unwrap();
    let artifacts = fake_artifacts(dir.path());
    let pins = test_pins(&artifacts);
    let qualified = VersionedPackDescriptor::parse(&document(&pins))
        .unwrap()
        .validate_with_pins(dir.path(), &pins)
        .unwrap();
    let path = dir.path().join("encoder.bin");
    let mut bytes = fs::read(&path).unwrap();
    let graph = qualified.read_graph_bytes(ArtifactRole::Encoder).unwrap();
    assert_eq!(graph.role(), ArtifactRole::Encoder);
    assert_eq!(graph.bytes(), bytes);
    assert!(
        qualified
            .read_graph_bytes(ArtifactRole::RawTokenizer)
            .is_err()
    );
    bytes[0] ^= 1;
    fs::write(path, bytes).unwrap();
    assert!(qualified.read_graph_bytes(ArtifactRole::Encoder).is_err());
}
