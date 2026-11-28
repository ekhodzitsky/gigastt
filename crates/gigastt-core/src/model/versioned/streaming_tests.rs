use super::*;

fn document(pins: &[StreamingPin<'_>]) -> String {
    let mut text = format!(
        "schema_version = 1\nfamily = \"gigaam_v4\"\nhead = \"rnnt\"\nstyles = [\"raw\", \"e2e\"]\nmode = \"streaming\"\nprecision = \"fp32\"\nprovider = \"cpu\"\nabi = \"v4_rnnt_streaming_split_v1\"\nfuture_context_ms = [160, 320, 640]\nleft_context_frames = 32\nstate_schema = \"v4_zapformer_cache_v1\"\nexternal_data = []\n[provenance]\nsource_repository = {SOURCE_REPOSITORY:?}\nsource_revision = {SOURCE_REVISION:?}\nmodel_repository = {MODEL_REPOSITORY:?}\nmodel_revision = {MODEL_REVISION:?}\ncheckpoint_sha256 = {CHECKPOINT_SHA256:?}\nlicense = \"MIT\"\n"
    );
    for pin in pins {
        text.push_str(&format!(
            "\n[artifacts.{}]\nfile = \"{}.bin\"\nbytes = {}\nsha256 = {:?}\n",
            pin.role.as_str(),
            pin.role.as_str(),
            pin.bytes,
            pin.sha256
        ));
    }
    text
}

#[test]
fn test_streaming_descriptor_is_separate_exact_profile() {
    let doc = document(&STREAMING_PINS);
    let parsed = StreamingPackDescriptor::parse(&doc).unwrap();
    assert_eq!(parsed.future_context_ms, [160, 320, 640]);
    assert_eq!(parsed.artifacts.len(), 11);
    assert!(super::super::VersionedPackDescriptor::parse(&doc).is_err());
    for (old, new) in [
        ("schema_version = 1", "schema_version = 2"),
        ("family = \"gigaam_v4\"", "family = \"gigaam_v3\""),
        ("mode = \"streaming\"", "mode = \"full_context\""),
        (
            "abi = \"v4_rnnt_streaming_split_v1\"",
            "abi = \"v4_rnnt_split_v1\"",
        ),
        ("precision = \"fp32\"", "precision = \"int8\""),
        ("provider = \"cpu\"", "provider = \"auto\""),
        ("styles = [\"raw\", \"e2e\"]", "styles = [\"raw\", \"raw\"]"),
        (
            "future_context_ms = [160, 320, 640]",
            "future_context_ms = [160, 160, 640]",
        ),
        (
            "future_context_ms = [160, 320, 640]",
            "future_context_ms = [160, 320, 1280]",
        ),
        ("left_context_frames = 32", "left_context_frames = 64"),
        (
            "state_schema = \"v4_zapformer_cache_v1\"",
            "state_schema = \"arbitrary\"",
        ),
        ("file = \"frontend.bin\"", "file = \"../frontend.bin\""),
        ("file = \"frontend.bin\"", "file = \"STEP.BIN\""),
        ("license = \"MIT\"", "license = \"Other\""),
    ] {
        assert!(
            StreamingPackDescriptor::parse(&doc.replace(old, new)).is_err(),
            "accepted {new}"
        );
    }
    let missing = STREAMING_PINS
        .iter()
        .copied()
        .filter(|p| p.role != StreamingArtifactRole::OfficialSavedConfig)
        .collect::<Vec<_>>();
    assert!(StreamingPackDescriptor::parse(&document(&missing)).is_err());
    assert!(
        StreamingPackDescriptor::parse(&format!(
            "{doc}\n[artifacts.whole]\nfile=\"whole.onnx\"\nbytes=1\nsha256=\"{}\"\n",
            "a".repeat(64)
        ))
        .is_err()
    );
}

#[test]
fn test_streaming_profile_cache_and_outlet_contract() {
    assert_eq!(V4_STREAMING_FRONT_CACHE_SHAPES.len(), 4);
    assert_eq!(V4_STREAMING_LAYER_CACHE_SHAPES.len(), 54);
    let bytes: usize = V4_STREAMING_FRONT_CACHE_SHAPES
        .iter()
        .chain(V4_STREAMING_LAYER_CACHE_SHAPES)
        .map(|s| s.iter().product::<usize>() * 4)
        .sum();
    assert_eq!(bytes, 3_866_496);
    let (inputs, outputs) = StreamingArtifactRole::Frontend.graph_specs().unwrap();
    assert_eq!((inputs.len(), outputs.len()), (6, 6));
    assert_eq!(inputs[0].shape, &[1, 64, -1]);
    assert_eq!(outputs[0].name, "pre_encoded");
    let (inputs, outputs) = StreamingArtifactRole::Step.graph_specs().unwrap();
    assert_eq!((inputs.len(), outputs.len()), (59, 58));
    assert_eq!(inputs[2].name, "context_frames");
    assert_eq!(inputs[2].shape, &[1]);
    assert_eq!(inputs[58].shape, &[1, 768, 2]);
    assert_eq!(outputs[57].name, "next_cache_53");
    assert!(
        StreamingArtifactRole::OfficialStateManifest
            .graph_specs()
            .is_none()
    );
    for (ms, frames) in [(160, 4), (320, 8), (640, 16)] {
        let c = V4StreamContext::from_millis(ms).unwrap();
        assert_eq!(c.future_context_ms(), ms);
        assert_eq!(c.context_frames(), frames);
        assert_eq!(c.mel_block(), frames * 4);
    }
    assert!(V4StreamContext::from_millis(0).is_err());
    assert!(V4StreamContext::from_millis(480).is_err());
}

fn tiny_pack(dir: &Path) -> (StreamingPackDescriptor, Vec<StreamingPin<'static>>) {
    let mut pins = Vec::new();
    for role in STREAMING_ROLES {
        let bytes = role.as_str().as_bytes();
        std::fs::write(dir.join(format!("{}.bin", role.as_str())), bytes).unwrap();
        let mut hash = Sha256::new();
        hash.update(bytes);
        pins.push(StreamingPin {
            role,
            bytes: bytes.len() as u64,
            sha256: Box::leak(hex_lower(&hash.finalize()).into_boxed_str()),
        });
    }
    (
        StreamingPackDescriptor::parse(&document(&pins)).unwrap(),
        pins,
    )
}

#[test]
fn test_streaming_pack_verifies_all_roles_and_rechecks_consumption() {
    let dir = tempfile::tempdir().unwrap();
    let (descriptor, pins) = tiny_pack(dir.path());
    assert!(
        descriptor.validate_in_dir(dir.path()).is_err(),
        "caller hashes must not bypass shipped pins"
    );
    let q = descriptor.validate_with_pins(dir.path(), &pins).unwrap();
    assert_eq!(
        q.read_tokenizer_bytes(TranscriptStyle::Raw).unwrap(),
        b"raw_tokenizer"
    );
    let p = dir.path().join("raw_tokenizer.bin");
    std::fs::write(&p, b"bad_tokenizer").unwrap();
    assert!(q.read_tokenizer_bytes(TranscriptStyle::Raw).is_err());
    std::fs::write(&p, b"raw_tokenizer").unwrap();
    let p = dir.path().join("official_saved_config.bin");
    std::fs::remove_file(&p).unwrap();
    assert!(descriptor.validate_with_pins(dir.path(), &pins).is_err());
    assert!(
        QualifiedV4StreamingPack::load(dir.path()).is_err(),
        "manifest required; no fallback"
    );
}

#[test]
fn test_streaming_pack_rejects_role_swaps_and_partial_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let (mut descriptor, pins) = tiny_pack(dir.path());
    let raw = descriptor.artifacts[&StreamingArtifactRole::RawJoint].clone();
    let e2e = descriptor.artifacts[&StreamingArtifactRole::E2eJoint].clone();
    descriptor
        .artifacts
        .insert(StreamingArtifactRole::RawJoint, e2e);
    descriptor
        .artifacts
        .insert(StreamingArtifactRole::E2eJoint, raw);
    assert!(descriptor.validate_with_pins(dir.path(), &pins).is_err());
    let path = dir.path().join(super::super::MANIFEST_FILE);
    std::fs::write(&path, "schema_version = 1\n").unwrap();
    assert!(QualifiedV4StreamingPack::load(dir.path()).is_err());
}

#[cfg(unix)]
#[test]
fn test_streaming_pack_rejects_symlink_artifact() {
    let dir = tempfile::tempdir().unwrap();
    let (descriptor, pins) = tiny_pack(dir.path());
    let p = dir.path().join("step.bin");
    let target = dir.path().join("elsewhere.bin");
    std::fs::rename(&p, &target).unwrap();
    std::os::unix::fs::symlink(&target, &p).unwrap();
    assert!(descriptor.validate_with_pins(dir.path(), &pins).is_err());
}

#[test]
#[ignore = "requires the exact local streaming pack; hashes bytes only"]
fn test_v4_streaming_pack_exact_local_bytes() {
    let dir = std::env::var_os("GIGASTT_V4_STREAMING_PACK").expect("explicit local pack");
    let pack = QualifiedV4StreamingPack::load(Path::new(&dir)).unwrap();
    assert_eq!(pack.descriptor().artifacts.len(), 11);
    assert!(super::super::QualifiedV4Pack::load(Path::new(&dir)).is_err());
    assert!(super::super::super::reject_versioned_pack(Path::new(&dir)).is_err());
}
