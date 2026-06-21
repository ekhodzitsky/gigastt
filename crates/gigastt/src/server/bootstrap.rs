//! Minimal HTTP bootstrap responder used while the engine is still loading.
//!
//! On first run gigastt downloads ~850 MB of model files and generates the INT8
//! encoder before it can serve real requests. Without this, the TCP port is
//! unbound during that window, so `curl --fail /health` and Docker `HEALTHCHECK`
//! probes see connection-refused — indistinguishable from a crashed container.
//!
//! [`super::run_with_config_loading`] binds the listener up front and answers
//! probes with this responder until the engine is ready, then hands the *same*
//! socket to the full server (no rebind, no gap). The responder is deliberately
//! tiny — a hand-written HTTP/1.1 reply per connection — so it has zero engine
//! state and cannot itself fail to start:
//!
//! - `GET /health` → `200 {"status":"ok","model":"loading","version":"…"}`
//! - `GET /ready`  → `503 {"status":"not_ready","reason":"initializing"}`
//! - anything else → `503 {"error":"server is starting up","code":"initializing"}`

use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// Read budget for the request line. Probes send a tiny request; we only need
/// the first line, so a slow or oversized client is dropped rather than served.
const READ_TIMEOUT: Duration = Duration::from_secs(2);

/// Serve a single bootstrap response on `stream`, then let the connection close.
///
/// Generic over the stream type so it can be unit-tested over an in-memory
/// duplex pipe without a real socket.
pub(crate) async fn handle_bootstrap_conn<S>(mut stream: S, version: &str)
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let mut buf = [0u8; 1024];
    let n = match tokio::time::timeout(READ_TIMEOUT, stream.read(&mut buf)).await {
        Ok(Ok(n)) if n > 0 => n,
        // Timeout, EOF, or read error: nothing actionable, just drop the conn.
        _ => return,
    };

    let path = request_path(&buf[..n]);
    let (status_line, body) = bootstrap_response(path, version);

    let response = format!(
        "HTTP/1.1 {status_line}\r\n\
         content-type: application/json\r\n\
         content-length: {len}\r\n\
         connection: close\r\n\
         \r\n\
         {body}",
        len = body.len(),
    );
    let _ = stream.write_all(response.as_bytes()).await;
    let _ = stream.flush().await;
}

/// Extract the request-target path from a raw HTTP/1.1 request, stripping any
/// query string. Returns `/` when the request can't be parsed.
fn request_path(req: &[u8]) -> &str {
    let line_end = req
        .iter()
        .position(|&b| b == b'\r' || b == b'\n')
        .unwrap_or(req.len());
    let line = std::str::from_utf8(&req[..line_end]).unwrap_or("");
    // Request line: "<METHOD> <PATH> HTTP/1.1" — take the second whitespace token.
    let raw_path = line.split_whitespace().nth(1).unwrap_or("/");
    raw_path.split('?').next().unwrap_or("/")
}

/// Map a request path to a `(status-line, json-body)` bootstrap response.
fn bootstrap_response(path: &str, version: &str) -> (&'static str, String) {
    match path {
        "/health" => (
            "200 OK",
            format!(r#"{{"status":"ok","model":"loading","version":"{version}"}}"#),
        ),
        "/ready" => (
            "503 Service Unavailable",
            r#"{"status":"not_ready","reason":"initializing"}"#.to_string(),
        ),
        _ => (
            "503 Service Unavailable",
            r#"{"error":"server is starting up","code":"initializing"}"#.to_string(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path_of(req: &str) -> &str {
        request_path(req.as_bytes())
    }

    #[test]
    fn test_request_path_parsing() {
        assert_eq!(
            path_of("GET /health HTTP/1.1\r\nHost: x\r\n\r\n"),
            "/health"
        );
        assert_eq!(path_of("GET /ready?x=1 HTTP/1.1\r\n\r\n"), "/ready");
        assert_eq!(
            path_of("POST /v1/transcribe HTTP/1.1\r\n"),
            "/v1/transcribe"
        );
        assert_eq!(path_of("garbage"), "/");
        assert_eq!(path_of(""), "/");
    }

    #[test]
    fn test_bootstrap_response_codes() {
        assert_eq!(bootstrap_response("/health", "9.9.9").0, "200 OK");
        assert!(
            bootstrap_response("/health", "9.9.9")
                .1
                .contains("\"model\":\"loading\"")
        );
        assert!(
            bootstrap_response("/health", "9.9.9")
                .1
                .contains("\"version\":\"9.9.9\"")
        );
        assert_eq!(
            bootstrap_response("/ready", "9.9.9").0,
            "503 Service Unavailable"
        );
        assert!(
            bootstrap_response("/ready", "9.9.9")
                .1
                .contains("\"reason\":\"initializing\"")
        );
        assert_eq!(
            bootstrap_response("/v1/transcribe", "9.9.9").0,
            "503 Service Unavailable"
        );
        assert!(
            bootstrap_response("/v1/transcribe", "9.9.9")
                .1
                .contains("\"code\":\"initializing\"")
        );
    }

    #[tokio::test]
    async fn test_handle_bootstrap_conn_health_over_duplex() {
        let (mut client, server) = tokio::io::duplex(2048);
        let task = tokio::spawn(async move {
            handle_bootstrap_conn(server, "1.2.3").await;
        });
        client
            .write_all(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")
            .await
            .unwrap();
        let mut out = Vec::new();
        client.read_to_end(&mut out).await.unwrap();
        task.await.unwrap();
        let resp = String::from_utf8(out).unwrap();
        assert!(resp.starts_with("HTTP/1.1 200 OK"), "got: {resp}");
        assert!(resp.contains("content-length:"));
        assert!(resp.contains("\"status\":\"ok\""));
        assert!(resp.contains("\"model\":\"loading\""));
    }

    #[tokio::test]
    async fn test_handle_bootstrap_conn_ready_is_503() {
        let (mut client, server) = tokio::io::duplex(2048);
        let task = tokio::spawn(async move {
            handle_bootstrap_conn(server, "1.2.3").await;
        });
        client
            .write_all(b"GET /ready HTTP/1.1\r\n\r\n")
            .await
            .unwrap();
        let mut out = Vec::new();
        client.read_to_end(&mut out).await.unwrap();
        task.await.unwrap();
        let resp = String::from_utf8(out).unwrap();
        assert!(resp.starts_with("HTTP/1.1 503"), "got: {resp}");
        assert!(resp.contains("\"status\":\"not_ready\""));
        assert!(resp.contains("\"reason\":\"initializing\""));
    }
}
