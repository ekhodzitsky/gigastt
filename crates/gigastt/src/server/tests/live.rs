//! In-process server + mock engine: WS / REST / OpenAI without the INT8 model.

use super::mock_engine;
use crate::server::{ServerConfig, run_with_config_listener};
use futures_util::{SinkExt, StreamExt};
use std::time::Duration;
use tokio_tungstenite::tungstenite::Message;

async fn spawn_mock_server() -> (u16, tokio::sync::oneshot::Sender<()>, tempfile::TempDir) {
    let (engine, tmp) = mock_engine();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let config = ServerConfig::local(port);
    tokio::spawn(async move {
        run_with_config_listener(engine, config, Some(shutdown_rx), listener)
            .await
            .expect("server");
    });
    for _ in 0..100 {
        if reqwest::get(format!("http://127.0.0.1:{port}/health"))
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
        {
            return (port, shutdown_tx, tmp);
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("mock server did not become ready on port {port}");
}

#[tokio::test]
async fn test_live_health_and_openai_transcriptions() {
    let (port, shutdown, _tmp) = spawn_mock_server().await;
    let health: serde_json::Value = reqwest::get(format!("http://127.0.0.1:{port}/health"))
        .await
        .expect("health")
        .json()
        .await
        .expect("json");
    assert_eq!(health["status"], "ok");
    assert_eq!(health["variant"], "rnnt");

    let wav = gigastt_core::test_support::pcm16_wav(&[0i16; 1600], 16_000);
    let form = reqwest::multipart::Form::new().part(
        "file",
        reqwest::multipart::Part::bytes(wav)
            .file_name("silence.wav")
            .mime_str("audio/wav")
            .expect("mime"),
    );
    let resp = reqwest::Client::new()
        .post(format!("http://127.0.0.1:{port}/v1/audio/transcriptions"))
        .multipart(form)
        .send()
        .await
        .expect("openai");
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.expect("openai json");
    assert!(body.get("text").is_some(), "openai json has text: {body}");

    let _ = shutdown.send(());
}

#[tokio::test]
async fn test_live_ws_ready_audio_stop_final() {
    let (port, shutdown, _tmp) = spawn_mock_server().await;
    let (ws, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}/v1/ws"))
        .await
        .expect("ws connect");
    let (mut sink, mut stream) = ws.split();

    let ready = next_json(&mut stream).await;
    assert_eq!(ready["type"], "ready");
    assert_eq!(ready["version"], "1.1");
    assert_eq!(ready["min_protocol_version"], "1.0");

    sink.send(Message::Text(
        serde_json::json!({"type": "configure", "sample_rate": 16000})
            .to_string()
            .into(),
    ))
    .await
    .expect("configure");

    let mut pcm = Vec::with_capacity(320 * 2);
    for _ in 0..320 {
        pcm.extend_from_slice(&0i16.to_le_bytes());
    }
    sink.send(Message::Binary(pcm.into())).await.expect("audio");
    sink.send(Message::Text(
        serde_json::json!({"type": "stop"}).to_string().into(),
    ))
    .await
    .expect("stop");

    let mut saw_final = false;
    for _ in 0..8 {
        let v = next_json(&mut stream).await;
        match v["type"].as_str() {
            Some("partial") => continue,
            Some("final") => {
                assert!(v["text"].is_string());
                saw_final = true;
                break;
            }
            other => panic!("unexpected ws message type {other:?}: {v}"),
        }
    }
    assert!(saw_final, "stop must produce a final");
    let _ = shutdown.send(());
}

#[tokio::test]
async fn test_live_ws_invalid_sample_rate_is_rejected() {
    let (port, shutdown, _tmp) = spawn_mock_server().await;
    let (ws, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}/v1/ws"))
        .await
        .expect("ws connect");
    let (mut sink, mut stream) = ws.split();
    let ready = next_json(&mut stream).await;
    assert_eq!(ready["type"], "ready");

    sink.send(Message::Text(
        serde_json::json!({"type": "configure", "sample_rate": 12345})
            .to_string()
            .into(),
    ))
    .await
    .expect("configure");

    let err = next_json(&mut stream).await;
    assert_eq!(err["type"], "error");
    assert_eq!(err["code"], "invalid_sample_rate");
    let _ = shutdown.send(());
}

async fn next_json<S>(stream: &mut S) -> serde_json::Value
where
    S: StreamExt<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    let msg = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("timeout")
        .expect("eof")
        .expect("ws err");
    let text = msg.into_text().expect("text");
    serde_json::from_str(&text).unwrap_or_else(|_| panic!("json: {text}"))
}
