//! End-to-end CLI tests: drive the actual `gigastt` binary as a subprocess.
//!
//! The e2e_rest / e2e_ws suites exercise the server through the *library*
//! (`run_with_config_listener`), which bypasses `main()` and the CLI command
//! dispatch entirely. These tests close that gap by spawning the built binary
//! via `CARGO_BIN_EXE_gigastt` and asserting on exit status / output, so the
//! `Serve` / `Transcribe` / `Quantize` match arms and their helper wiring are
//! actually executed.
//!
//! Help / version / arg-validation tests need no model and run anywhere. The
//! transcribe / serve / quantize tests are `#[ignore]` (require the GigaAM
//! model ~850 MB at `~/.gigastt/models`). Run all with:
//! `cargo test --test e2e_cli -- --include-ignored --test-threads=1`.

mod common;

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Path to the binary cargo built for this test run (instrumented under
/// `cargo llvm-cov`, so subprocess execution is captured in coverage).
fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_gigastt")
}

/// Real speech fixture shipped with the gigastt crate.
const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/golos_00.wav");

/// Grab a free TCP port by binding to :0 and immediately releasing it.
/// A small TOCTOU window, acceptable for a test that hands the port to a
/// freshly-spawned child.
fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// Minimal blocking HTTP GET that returns just the status code. Avoids pulling
/// an async runtime into these otherwise-synchronous subprocess tests.
fn http_status(port: u16, path: &str) -> Option<u16> {
    let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).ok()?;
    s.set_read_timeout(Some(Duration::from_millis(1500))).ok()?;
    s.set_write_timeout(Some(Duration::from_millis(1500)))
        .ok()?;
    write!(
        s,
        "GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
    )
    .ok()?;
    let mut buf = Vec::new();
    let _ = s.read_to_end(&mut buf);
    let head = String::from_utf8_lossy(&buf);
    head.lines().next()?.split_whitespace().nth(1)?.parse().ok()
}

// ─── no-model: help / version / argument validation ─────────────────────────

#[test]
fn cli_help_exits_zero() {
    let out = Command::new(bin()).arg("--help").output().expect("run");
    assert!(out.status.success(), "--help should exit 0");
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("Local STT"),
        "--help should print the about text"
    );
}

#[test]
fn cli_version_exits_zero() {
    let out = Command::new(bin()).arg("--version").output().expect("run");
    assert!(out.status.success(), "--version should exit 0");
}

#[test]
fn cli_no_subcommand_fails() {
    // clap requires a subcommand; a bare invocation must error with usage.
    let out = Command::new(bin()).output().expect("run");
    assert!(!out.status.success(), "bare invocation must fail");
}

#[test]
fn cli_unknown_subcommand_fails() {
    let out = Command::new(bin()).arg("frobnicate").output().expect("run");
    assert!(!out.status.success(), "unknown subcommand must fail");
}

#[test]
fn cli_subcommand_help_exits_zero() {
    for sub in ["serve", "download", "quantize", "transcribe"] {
        let out = Command::new(bin())
            .args([sub, "--help"])
            .output()
            .expect("run");
        assert!(out.status.success(), "`{sub} --help` should exit 0");
    }
}

#[test]
fn cli_serve_non_loopback_without_bind_all_fails() {
    // Enters the `Serve` arm and hits `ensure_bind_allowed` *before* any model
    // load, so this is fast and needs no model. Without `--bind-all` (and with
    // the env opt-out cleared) a non-loopback host must be rejected.
    let port = free_port();
    let out = Command::new(bin())
        .args(["serve", "--host", "0.0.0.0", "--port", &port.to_string()])
        .env_remove("GIGASTT_ALLOW_BIND_ANY")
        .output()
        .expect("run");
    assert!(
        !out.status.success(),
        "0.0.0.0 without --bind-all must be refused"
    );
}

// ─── model-gated: transcribe ────────────────────────────────────────────────

#[ignore = "requires the GigaAM model (~850MB)"]
#[test]
fn cli_transcribe_to_stdout_txt() {
    let md = common::model_dir();
    let out = Command::new(bin())
        .args([
            "transcribe",
            FIXTURE,
            "--model-dir",
            &md,
            "--punctuation",
            "off",
            "--itn",
            "off",
        ])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "transcribe failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !out.stdout.is_empty(),
        "transcribe should print text to stdout"
    );
}

#[ignore = "requires the GigaAM model (~850MB)"]
#[test]
fn cli_transcribe_json_stdout() {
    let md = common::model_dir();
    let out = Command::new(bin())
        .args([
            "transcribe",
            FIXTURE,
            "--model-dir",
            &md,
            "--log-level",
            "error",
            "--punctuation",
            "off",
            "--itn",
            "off",
            "--format",
            "json",
        ])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "transcribe json failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    // `--log-level error` keeps stdout clean; slice from the first `{` so any
    // stray line can't break the parse.
    let body = String::from_utf8_lossy(&out.stdout);
    let json = &body[body.find('{').expect("json object on stdout")..];
    let parsed: serde_json::Value =
        serde_json::from_str(json.trim()).expect("json output should be valid JSON");
    assert!(
        parsed.get("text").is_some(),
        "json should carry a text field"
    );
}

#[ignore = "requires the GigaAM model (~850MB)"]
#[test]
fn cli_transcribe_to_file_srt_with_hotwords() {
    let md = common::model_dir();
    let tmp = tempfile::tempdir().unwrap();
    // Exercise parse_hotwords_file: a plain phrase, a weighted phrase, a comment
    // and a blank line (the latter two must be ignored).
    let hw = tmp.path().join("hotwords.txt");
    std::fs::write(&hw, "гигачат\nсбербанк\t8.0\n# comment\n\n").unwrap();
    let outp = tmp.path().join("out.srt");
    let out = Command::new(bin())
        .args([
            "transcribe",
            FIXTURE,
            "--model-dir",
            &md,
            "--punctuation",
            "off",
            "--itn",
            "off",
            "--format",
            "srt",
            "-o",
            outp.to_str().unwrap(),
            "--hotwords-file",
            hw.to_str().unwrap(),
            "--hotwords-default",
            "--hotwords-boost",
            "6.0",
        ])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "transcribe srt failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let body = std::fs::read_to_string(&outp).expect("srt output file");
    assert!(!body.trim().is_empty(), "srt file should not be empty");
}

#[ignore = "requires the GigaAM model (~850MB)"]
#[test]
fn cli_transcribe_missing_file_fails() {
    let md = common::model_dir();
    let out = Command::new(bin())
        .args([
            "transcribe",
            "/no/such/file.wav",
            "--model-dir",
            &md,
            "--punctuation",
            "off",
            "--itn",
            "off",
        ])
        .output()
        .expect("run");
    assert!(
        !out.status.success(),
        "transcribing a missing file must fail"
    );
}

#[ignore = "requires the GigaAM model (~850MB)"]
#[test]
fn cli_transcribe_bad_format_fails() {
    // The format string is validated after a successful decode, so the model
    // must load first — hence model-gated.
    let md = common::model_dir();
    let out = Command::new(bin())
        .args([
            "transcribe",
            FIXTURE,
            "--model-dir",
            &md,
            "--punctuation",
            "off",
            "--itn",
            "off",
            "--format",
            "bogus",
        ])
        .output()
        .expect("run");
    assert!(!out.status.success(), "an unknown export format must fail");
}

// ─── model-gated: quantize ──────────────────────────────────────────────────

#[ignore = "requires the GigaAM model (~850MB)"]
#[test]
fn cli_quantize_existing_is_noop() {
    let md = common::model_dir();
    // Only run the fast "already exists" path; if no INT8 encoder is present a
    // real quantize would take ~2 min, so skip rather than block the suite.
    let has_int8 = ["v3_rnnt_encoder_int8.onnx", "v3_e2e_rnnt_encoder_int8.onnx"]
        .iter()
        .any(|f| std::path::Path::new(&md).join(f).exists());
    if !has_int8 {
        eprintln!("skipping cli_quantize_existing_is_noop: no INT8 encoder present");
        return;
    }
    let out = Command::new(bin())
        .args(["quantize", "--model-dir", &md])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "quantize (noop) failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// ─── model-gated: serve boot + graceful shutdown ────────────────────────────

#[ignore = "requires the GigaAM model (~850MB)"]
#[test]
fn cli_serve_boots_and_graceful_shutdown() {
    let md = common::model_dir();
    let port = free_port();
    let metrics_port = free_port();

    // Boot the server through the real CLI: covers the bulk of the `Serve` arm
    // (bind check, model resolve, INT8 check, engine load, default-hotword
    // biasing, limits build, metrics listener, server run) plus the graceful
    // SIGTERM shutdown path. `--punctuation off --itn off` and no `--vad` keep
    // it hermetic (no auxiliary model downloads).
    let mut child = Command::new(bin())
        .args([
            "serve",
            "--port",
            &port.to_string(),
            "--model-dir",
            &md,
            "--punctuation",
            "off",
            "--itn",
            "off",
            "--pool-size",
            "1",
            "--hotwords-default",
            "--metrics",
            "--metrics-listen",
            &format!("127.0.0.1:{metrics_port}"),
        ])
        .env_remove("GIGASTT_VAD")
        .env_remove("GIGASTT_ALLOW_BIND_ANY")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn serve");

    // Wait for readiness (model load can take a while on a cold cache).
    let start = Instant::now();
    let mut ready = false;
    while start.elapsed() < Duration::from_secs(120) {
        if let Ok(Some(status)) = child.try_wait() {
            panic!("serve exited early with {status}");
        }
        if http_status(port, "/health") == Some(200) {
            ready = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    assert!(ready, "server did not become healthy within 120s");

    // Metrics live on their own loopback port.
    assert_eq!(
        http_status(metrics_port, "/metrics"),
        Some(200),
        "metrics endpoint should answer on its dedicated port"
    );

    // Graceful shutdown: SIGTERM should drain and return a clean exit, which is
    // also what flushes the subprocess's coverage profile.
    let _ = Command::new("kill")
        .arg("-TERM")
        .arg(child.id().to_string())
        .status();

    let exit_start = Instant::now();
    loop {
        match child.try_wait().expect("try_wait") {
            Some(status) => {
                assert!(
                    status.success(),
                    "serve should exit cleanly on SIGTERM, got {status}"
                );
                break;
            }
            None => {
                if exit_start.elapsed() > Duration::from_secs(20) {
                    let _ = child.kill();
                    panic!("serve did not exit within 20s of SIGTERM");
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }
}
