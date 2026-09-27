//! Author-package oracle for the shipped rnnt INT8 runtime.
//!
//! The expected text is the greedy FP32 transcript of `gigaam.load_model("v3_rnnt")`
//! on this clip. The product never loads FP32. Run with the model installed:
//!
//! ```sh
//! cargo test -p gigastt --test oracle_gigaam -- --ignored
//! ```
//!
//! Absent model: the test skips. It does not download weights and it does not
//! substitute another corpus.

use std::path::PathBuf;
use std::process::Command;

const ORACLE: &str = "шестьдесят тысяч тенге сколько будет стоить";

fn home_model() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".gigastt/models/v3_rnnt_encoder_int8.onnx"))
}

#[test]
#[ignore = "needs the installed INT8 rnnt encoder; skips when that file is absent"]
fn test_rnnt_int8_matches_author_on_golos_00() {
    let Some(encoder) = home_model() else {
        eprintln!("skip: HOME is unset");
        return;
    };
    if !encoder.is_file() {
        eprintln!(
            "skip: INT8 rnnt encoder not found at {}. Install it with `gigastt download` \
             or set the model dir to the default. This gate does not fetch weights.",
            encoder.display()
        );
        return;
    }
    let wav = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golos_00.wav");
    let output = Command::new(env!("CARGO_BIN_EXE_gigastt"))
        .args([
            "--offline",
            "transcribe",
            wav.to_str().expect("fixture path is utf-8"),
            "--model-variant",
            "rnnt",
            "--punctuation",
            "off",
            "--itn",
            "off",
            "--format",
            "json",
        ])
        .output()
        .expect("run gigastt transcribe");
    assert!(
        output.status.success(),
        "transcribe failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json_line = stdout
        .lines()
        .rev()
        .find(|line| line.starts_with('{'))
        .unwrap_or("");
    let parsed: serde_json::Value = serde_json::from_str(json_line)
        .unwrap_or_else(|err| panic!("no transcript json in output ({err}): {stdout}"));
    let text = parsed["text"].as_str().unwrap_or("");
    assert_eq!(
        text, ORACLE,
        "INT8 rnnt transcript diverged from the author FP32 greedy decode"
    );
}
