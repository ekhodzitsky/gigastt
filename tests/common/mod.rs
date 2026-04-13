//! Shared test helpers for e2e tests.
//!
//! Provides server startup with clean shutdown, WAV generation,
//! WebSocket helpers, and readiness polling.

use std::path::PathBuf;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::oneshot;

/// Platform-specific home directory.
pub fn home_dir() -> Option<PathBuf> {
    #[cfg(unix)]
    {
        std::env::var_os("HOME").map(PathBuf::from)
    }
    #[cfg(windows)]
    {
        std::env::var_os("USERPROFILE").map(PathBuf::from)
    }
}

/// Return model directory if model files exist, or panic with helpful message.
pub fn model_dir() -> String {
    let dir = home_dir()
        .expect("Cannot determine home directory")
        .join(".gigastt")
        .join("models");
    assert!(
        dir.join("v3_e2e_rnnt_encoder.onnx").exists(),
        "Model not found at {}. Run `cargo run -- download` first.",
        dir.display()
    );
    dir.to_string_lossy().into_owned()
}

/// Find a free TCP port by binding to port 0.
pub async fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    port
}

/// Start the server with a shutdown handle. Returns (port, shutdown_sender).
///
/// Blocks until the server is accepting connections (polls /health).
/// Drop or send on the returned sender to shut the server down.
pub async fn start_server(model_dir: &str) -> (u16, oneshot::Sender<()>) {
    let port = free_port().await;
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let engine = gigastt::inference::Engine::load(model_dir).unwrap();
    tokio::spawn(gigastt::server::run_with_shutdown(
        engine,
        port,
        "127.0.0.1",
        Some(shutdown_rx),
    ));

    wait_for_ready(port, Duration::from_secs(30)).await;
    (port, shutdown_tx)
}

/// Poll GET /health with exponential backoff until 200 OK or timeout.
pub async fn wait_for_ready(port: u16, timeout: Duration) {
    let url = format!("http://127.0.0.1:{port}/health");
    let client = reqwest::Client::new();
    let start = tokio::time::Instant::now();
    let mut delay = Duration::from_millis(10);

    loop {
        if start.elapsed() > timeout {
            panic!(
                "Server on port {port} did not become ready within {timeout:?}"
            );
        }
        match client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => return,
            _ => {
                tokio::time::sleep(delay).await;
                delay = (delay * 2).min(Duration::from_millis(500));
            }
        }
    }
}

/// Generate an in-memory WAV file with a 440Hz sine wave.
pub fn generate_wav(duration_s: u32, sample_rate: u32) -> Vec<u8> {
    let num_samples = sample_rate * duration_s;
    let data_size = num_samples * 2; // 16-bit PCM
    let file_size = 44 + data_size;

    let mut wav = Vec::with_capacity(file_size as usize);
    // RIFF header
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(file_size - 8).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    // fmt chunk
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    wav.extend_from_slice(&2u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    // data chunk
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());
    for i in 0..num_samples {
        let sample =
            (440.0_f64 * 2.0 * std::f64::consts::PI * i as f64 / sample_rate as f64).sin()
                * 1000.0;
        wav.extend_from_slice(&(sample as i16).to_le_bytes());
    }
    wav
}

/// Generate raw PCM16 silence bytes.
pub fn generate_pcm16_silence(duration_s: f32, sample_rate: u32) -> Vec<u8> {
    let num_samples = (sample_rate as f32 * duration_s) as usize;
    vec![0u8; num_samples * 2]
}

/// Connect to WebSocket, receive Ready message, return (sink, stream, ready_value).
pub async fn ws_connect(
    port: u16,
) -> (
    futures_util::stream::SplitSink<
        tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
        tokio_tungstenite::tungstenite::Message,
    >,
    futures_util::stream::SplitStream<
        tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
    >,
    serde_json::Value,
) {
    use futures_util::StreamExt;

    let (ws, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}/ws"))
        .await
        .expect("WebSocket connection failed");
    let (sink, mut stream) = ws.split();

    let msg = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("timeout waiting for Ready")
        .expect("stream ended")
        .expect("ws error");

    let text = msg.into_text().expect("Ready should be text");
    let ready: serde_json::Value = serde_json::from_str(&text).expect("Ready should be JSON");
    assert_eq!(ready["type"], "ready", "First message should be Ready");

    (sink, stream, ready)
}

/// Parse a WebSocket message as JSON and assert its "type" field.
pub fn assert_msg_type(
    msg: tokio_tungstenite::tungstenite::Message,
    expected_type: &str,
) -> serde_json::Value {
    let text = msg.into_text().unwrap_or_else(|_| {
        panic!("Expected text message with type={expected_type}");
    });
    let v: serde_json::Value =
        serde_json::from_str(&text).unwrap_or_else(|_| panic!("Invalid JSON: {text}"));
    assert_eq!(
        v["type"], expected_type,
        "Expected type={expected_type}, got {:?} in {text}",
        v["type"]
    );
    v
}
