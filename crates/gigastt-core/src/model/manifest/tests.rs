use super::*;
use std::fs;

#[test]
fn test_parse_valid_rnnt_manifest() {
    let text = r#"
architecture = "rnnt"

[files]
encoder = "v3_rnnt_encoder.onnx"
encoder_int8 = "v3_rnnt_encoder_int8.onnx"
decoder = "v3_rnnt_decoder.onnx"
joint = "v3_rnnt_joint.onnx"
vocab = "v3_vocab.txt"
"#;
    let m = ModelManifest::parse(text).expect("valid manifest");
    assert_eq!(m.architecture, ModelVariant::Rnnt);
    assert_eq!(m.files.encoder, "v3_rnnt_encoder.onnx");
    assert_eq!(
        m.files.encoder_int8.as_deref(),
        Some("v3_rnnt_encoder_int8.onnx")
    );
    assert_eq!(m.files.decoder.as_deref(), Some("v3_rnnt_decoder.onnx"));
    assert_eq!(m.files.joint.as_deref(), Some("v3_rnnt_joint.onnx"));
    assert_eq!(m.files.vocab, "v3_vocab.txt");
}

#[test]
fn test_parse_valid_ctc_manifest_without_decoder_joint() {
    let text = r#"
architecture = "ml_ctc"

[files]
encoder = "multilingual_ctc.onnx"
encoder_int8 = "multilingual_ctc.int8.onnx"
vocab = "multilingual_vocab.txt"
"#;
    let m = ModelManifest::parse(text).expect("ctc manifest");
    assert_eq!(m.architecture, ModelVariant::MlCtc);
    assert!(m.files.decoder.is_none());
    assert!(m.files.joint.is_none());
}

#[test]
fn test_parse_empty_decoder_joint_treated_as_absent() {
    let text = r#"
architecture = "ml_ctc_large"

[files]
encoder = "multilingual_large_ctc.onnx"
decoder = ""
joint = ""
vocab = "multilingual_vocab.txt"
"#;
    let m = ModelManifest::parse(text).expect("empty decoder/joint ok for ctc");
    assert!(m.files.decoder.is_none());
    assert!(m.files.joint.is_none());
}

#[test]
fn test_load_missing_file_returns_none() {
    let dir = tempfile::tempdir().expect("tempdir");
    let loaded = ModelManifest::load(dir.path()).expect("missing is ok");
    assert!(loaded.is_none());
}

#[test]
fn test_load_valid_manifest_from_disk() {
    let dir = tempfile::tempdir().expect("tempdir");
    fs::write(
        dir.path().join(MANIFEST_FILE),
        r#"
architecture = "e2e_rnnt"
[files]
encoder = "custom_encoder.onnx"
decoder = "custom_decoder.onnx"
joint = "custom_joint.onnx"
vocab = "custom_vocab.txt"
"#,
    )
    .unwrap();
    let m = ModelManifest::load(dir.path())
        .expect("load")
        .expect("present");
    assert_eq!(m.architecture, ModelVariant::E2eRnnt);
    assert_eq!(m.files.encoder, "custom_encoder.onnx");
}

#[test]
fn test_invalid_architecture_rejected() {
    let text = r#"
architecture = "whisper"
[files]
encoder = "e.onnx"
decoder = "d.onnx"
joint = "j.onnx"
vocab = "v.txt"
"#;
    let err = ModelManifest::parse(text).expect_err("whisper must fail");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("architecture") || msg.contains("whisper"),
        "error should mention architecture: {msg}"
    );
}

#[test]
fn test_rnnt_requires_decoder_and_joint() {
    let text = r#"
architecture = "rnnt"
[files]
encoder = "e.onnx"
vocab = "v.txt"
"#;
    let err = ModelManifest::parse(text).expect_err("decoder required");
    assert!(
        format!("{err:#}").contains("decoder"),
        "expected decoder error, got {err:#}"
    );
}

#[test]
fn test_reject_path_separators_in_basenames() {
    let text = r#"
architecture = "rnnt"
[files]
encoder = "../escape.onnx"
decoder = "d.onnx"
joint = "j.onnx"
vocab = "v.txt"
"#;
    let err = ModelManifest::parse(text).expect_err("path sep rejected");
    assert!(
        format!("{err:#}").contains("basename"),
        "expected basename error, got {err:#}"
    );
}

#[test]
fn test_preferred_encoder_path_prefers_int8_when_present() {
    let dir = tempfile::tempdir().expect("tempdir");
    fs::write(dir.path().join("enc.onnx"), b"fp32").unwrap();
    fs::write(dir.path().join("enc_int8.onnx"), b"int8").unwrap();
    let m = ModelManifest {
        precision: super::ModelPrecision::Int8,
        architecture: ModelVariant::Rnnt,
        files: ManifestFiles {
            encoder: "enc.onnx".into(),
            encoder_int8: Some("enc_int8.onnx".into()),
            decoder: Some("d.onnx".into()),
            joint: Some("j.onnx".into()),
            vocab: "v.txt".into(),
        },
    };
    assert_eq!(
        m.preferred_encoder_path(dir.path()).file_name().unwrap(),
        "enc_int8.onnx"
    );
    assert!(m.prefers_int8(dir.path()));
}

#[test]
fn test_preferred_encoder_path_without_int8_file_is_not_int8() {
    // preferred_encoder_path still names the FP32 basename when int8 is
    // missing on disk; prefers_int8 is false so Engine load rejects it.
    let dir = tempfile::tempdir().expect("tempdir");
    fs::write(dir.path().join("enc.onnx"), b"fp32").unwrap();
    let m = ModelManifest {
        precision: super::ModelPrecision::Int8,
        architecture: ModelVariant::Rnnt,
        files: ManifestFiles {
            encoder: "enc.onnx".into(),
            encoder_int8: Some("enc_int8.onnx".into()),
            decoder: Some("d.onnx".into()),
            joint: Some("j.onnx".into()),
            vocab: "v.txt".into(),
        },
    };
    assert!(!m.prefers_int8(dir.path()));
    assert_eq!(
        m.preferred_encoder_path(dir.path()).file_name().unwrap(),
        "enc.onnx"
    );
}

#[test]
fn test_load_invalid_manifest_returns_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    fs::write(dir.path().join(MANIFEST_FILE), "architecture = [\n").unwrap();
    let err = ModelManifest::load(dir.path()).expect_err("invalid toml");
    assert!(
        format!("{err:#}").contains("manifest"),
        "expected manifest context, got {err:#}"
    );
}

#[test]
fn test_precision_rejects_typos_and_unsupported_fp32_heads() {
    let template = r#"
architecture = "e2e_rnnt"
precision = "fp32"
[files]
encoder = "custom.onnx"
decoder = "decoder.onnx"
joint = "joint.onnx"
vocab = "vocab.txt"
"#;
    assert!(ModelManifest::parse(template).is_ok());
    for precision in ["fp16", "FP32", "", "fp322"] {
        assert!(
            ModelManifest::parse(&template.replace("\"fp32\"", &format!("{precision:?}"))).is_err()
        );
    }
    for architecture in ["rnnt", "ml_ctc", "ml_ctc_large"] {
        assert!(
            ModelManifest::parse(&template.replace("\"e2e_rnnt\"", &format!("{architecture:?}")))
                .is_err()
        );
    }
}
