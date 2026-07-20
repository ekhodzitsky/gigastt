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

    // Metrics live on their own loopback port, bound by a separate listener that
    // can come up slightly after the main server reports `/health` ready — more
    // so under the instrumented coverage build. Poll it instead of probing once,
    // to avoid a readiness race (the single-probe version flaked in CI's
    // `Coverage (E2E)` job while the non-instrumented `E2E Tests` job passed).
    let metrics_start = Instant::now();
    let mut metrics_ok = false;
    while metrics_start.elapsed() < Duration::from_secs(30) {
        if http_status(metrics_port, "/metrics") == Some(200) {
            metrics_ok = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    assert!(
        metrics_ok,
        "metrics endpoint should answer 200 on its dedicated port within 30s"
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

// ─── model-gated: download --progress=json (NDJSON contract) ────────────────

/// Symlink the named files from the real model dir into `dst`. Returns `false`
/// (test must self-skip) when any of them is absent — e.g. a machine installed
/// via `download --prequantized` has no FP32 encoder; running anyway would
/// silently turn the test into a live ~850 MB network download.
#[cfg(unix)]
#[must_use]
fn link_model_files(dst: &std::path::Path, names: &[&str]) -> bool {
    let src = std::path::PathBuf::from(common::model_dir());
    for name in names {
        let from = src.join(name);
        if !from.exists() {
            eprintln!("skipping: {} not present in {}", name, src.display());
            return false;
        }
        std::os::unix::fs::symlink(&from, dst.join(name)).expect("symlink model file");
    }
    true
}

/// Kill the child on drop so a panicking assertion can never leak a running
/// subprocess (nor let a TempDir be deleted out from under it mid-quantize).
struct KillOnDrop(std::process::Child);

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Happy path with everything already present: stdout must be pure NDJSON —
/// every line parses as a JSON object with a `phase` tag and the final line is
/// `{"phase":"done",…}`. Human `\r`-progress and tracing logs must not leak
/// into stdout.
#[cfg(unix)]
#[ignore = "requires the GigaAM model (~850MB)"]
#[test]
fn cli_download_json_happy_path_stdout_is_pure_ndjson() {
    let tmp = tempfile::tempdir().expect("tempdir");
    if !link_model_files(
        tmp.path(),
        &[
            "v3_rnnt_encoder.onnx",
            "v3_rnnt_decoder.onnx",
            "v3_rnnt_joint.onnx",
            "v3_vocab.txt",
            "v3_rnnt_encoder_int8.onnx",
        ],
    ) {
        return;
    }

    let out = Command::new(bin())
        .args([
            "download",
            "--model-dir",
            tmp.path().to_str().unwrap(),
            "--model-variant",
            "rnnt",
            "--progress=json",
            "--skip-diarization",
        ])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "download failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8(out.stdout).expect("stdout must be UTF-8");
    let lines: Vec<&str> = stdout.lines().filter(|l| !l.is_empty()).collect();
    assert!(!lines.is_empty(), "json mode must emit at least one event");
    for line in &lines {
        let v: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("non-JSON stdout line {line:?}: {e}"));
        assert!(
            v.get("phase").and_then(|p| p.as_str()).is_some(),
            "every event carries a phase tag: {line}"
        );
    }
    let last: serde_json::Value = serde_json::from_str(lines.last().unwrap()).unwrap();
    assert_eq!(last["phase"], "done", "final event must be done: {last}");
    assert!(
        last["model_dir"].as_str().is_some(),
        "done event carries model_dir: {last}"
    );
}

/// The ~2-minute INT8 quantization pass must announce itself as its own
/// `quantize` phase as soon as it starts (a sidecar reading the stream would
/// otherwise mistake it for a hang). FP32 weights are linked in without the
/// INT8 encoder to force the pass; the subprocess is killed right after the
/// event arrives — the test asserts the emission, not the quantization.
#[cfg(unix)]
#[ignore = "requires the GigaAM model (~850MB)"]
#[test]
fn cli_download_json_emits_quantize_phase() {
    use std::io::BufRead;

    let tmp = tempfile::tempdir().expect("tempdir");
    if !link_model_files(
        tmp.path(),
        &[
            "v3_rnnt_encoder.onnx",
            "v3_rnnt_decoder.onnx",
            "v3_rnnt_joint.onnx",
            "v3_vocab.txt",
        ],
    ) {
        return;
    }

    let mut child = KillOnDrop(
        Command::new(bin())
            .args([
                "download",
                "--model-dir",
                tmp.path().to_str().unwrap(),
                "--model-variant",
                "rnnt",
                "--progress=json",
                "--skip-diarization",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn"),
    );

    let stdout = child.0.stdout.take().expect("piped stdout");
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in std::io::BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            if tx.send(line).is_err() {
                break;
            }
        }
    });

    let deadline = Instant::now() + Duration::from_secs(120);
    let mut saw_quantize = false;
    while Instant::now() < deadline {
        match rx.recv_timeout(Duration::from_millis(500)) {
            Ok(line) if line.is_empty() => continue,
            Ok(line) => {
                let v: serde_json::Value = serde_json::from_str(&line)
                    .unwrap_or_else(|e| panic!("non-JSON stdout line {line:?}: {e}"));
                if v["phase"] == "quantize" {
                    assert_eq!(
                        v["file"], "v3_rnnt_encoder.onnx",
                        "quantize event names the encoder being quantized: {v}"
                    );
                    saw_quantize = true;
                    break;
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if let Some(status) = child.0.try_wait().expect("try_wait") {
                    panic!("download exited ({status}) before emitting a quantize event");
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    drop(child);
    assert!(
        saw_quantize,
        "quantize phase event must be emitted when the INT8 encoder is missing"
    );
}

/// The json error contract end-to-end without a model or network: an
/// impossible `--model-dir` (a path under `/dev/null`) makes the download flow
/// fail locally; stdout must then carry a single `{"phase":"error",…}` event
/// whose `kind` matches the documented per-kind process exit code.
#[cfg(unix)]
#[test]
fn cli_download_json_error_event_and_exit_code_are_consistent() {
    let out = Command::new(bin())
        .args([
            "download",
            "--model-dir",
            "/dev/null/impossible",
            "--model-variant",
            "rnnt",
            "--progress=json",
            "--skip-diarization",
        ])
        .output()
        .expect("run");
    assert!(!out.status.success(), "impossible model dir must fail");

    let stdout = String::from_utf8(out.stdout).expect("stdout must be UTF-8");
    let lines: Vec<&str> = stdout.lines().filter(|l| !l.is_empty()).collect();
    let last: serde_json::Value =
        serde_json::from_str(lines.last().expect("json mode must emit an error event"))
            .expect("last stdout line must be JSON");
    assert_eq!(last["phase"], "error", "final event must be error: {last}");
    let kind = last["kind"].as_str().expect("error event carries kind");
    let expected_exit = match kind {
        "network" => 69,
        "disk" => 74,
        "checksum" => 65,
        "interrupted" => 130,
        "other" => 1,
        other => panic!("unknown error kind {other}"),
    };
    assert_eq!(
        out.status.code(),
        Some(expected_exit),
        "exit code must match the documented mapping for kind={kind}"
    );
}

// ─── no-model: air-gapped mode (GIGASTT_OFFLINE / --offline) ────────────────

/// With the env var set and an empty model dir, download must fail fast with
/// an instruction naming the missing file — not a network timeout. Hermetic:
/// no model, no network (the guard fires before any connection attempt).
#[test]
fn cli_download_offline_env_fails_fast_with_instruction() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = Command::new(bin())
        .env("GIGASTT_OFFLINE", "1")
        .args([
            "download",
            "--model-dir",
            tmp.path().to_str().unwrap(),
            "--model-variant",
            "rnnt",
            "--skip-diarization",
        ])
        .output()
        .expect("run");
    assert!(!out.status.success(), "offline download must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("offline mode"),
        "error must explain offline mode, got: {stderr}"
    );
    assert!(
        stderr.contains(tmp.path().to_str().unwrap()),
        "error must name where to place the file, got: {stderr}"
    );
}

/// The --offline flag is equivalent to GIGASTT_OFFLINE=1.
#[test]
fn cli_download_offline_flag_fails_fast() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = Command::new(bin())
        .args([
            "download",
            "--offline",
            "--model-dir",
            tmp.path().to_str().unwrap(),
            "--model-variant",
            "rnnt",
            "--skip-diarization",
        ])
        .output()
        .expect("run");
    assert!(!out.status.success(), "--offline download must fail");
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("offline mode"),
        "error must explain offline mode"
    );
}
