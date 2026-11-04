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
async fn test_live_ws_invalid_commit_policy_keeps_session_usable() {
    let (port, shutdown, _tmp) = spawn_mock_server().await;
    let (ws, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}/v1/ws"))
        .await
        .unwrap();
    let (mut sink, mut stream) = ws.split();
    assert_eq!(next_json(&mut stream).await["type"], "ready");
    sink.send(Message::Text(
        serde_json::json!({"type":"configure","commit_policy":"unknown"})
            .to_string()
            .into(),
    ))
    .await
    .unwrap();
    assert_eq!(
        next_json(&mut stream).await["code"],
        "invalid_commit_policy"
    );
    sink.send(Message::Text(
        serde_json::json!({"type":"stop"}).to_string().into(),
    ))
    .await
    .unwrap();
    let final_ = next_json(&mut stream).await;
    assert_eq!(final_["type"], "final");
    assert_eq!(final_["committed"], "");
    assert_eq!(final_["tentative"], "");
    let _ = shutdown.send(());
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
    assert_eq!(ready["version"], "1.3");
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

#[tokio::test]
async fn test_live_ws_tail_error_preserves_partial_without_final() {
    for prior in [false, true] {
        for joiner in [false, true] {
            for panic in [false, true] {
                let tmp = tempfile::tempdir().unwrap();
                gigastt_core::test_support::write_rnnt_layout(tmp.path()).unwrap();
                let factory = gigastt_core::test_support::FailingStreamFactory::new(joiner, panic);
                let engine = gigastt_core::test_support::load_rnnt_engine_with_factory(
                    tmp.path(),
                    1,
                    Box::new(factory.clone()),
                )
                .unwrap();
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                let port = listener.local_addr().unwrap().port();
                let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
                let server = tokio::spawn(run_with_config_listener(
                    engine,
                    ServerConfig::local(port),
                    Some(shutdown_rx),
                    listener,
                ));
                let (mut ws, _) =
                    tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}/v1/ws"))
                        .await
                        .unwrap();
                assert!(matches!(
                    ws.next().await.unwrap().unwrap(),
                    Message::Text(_)
                ));
                factory.arm(if prior { 2 } else { 1 });
                ws.send(Message::Text(
                    serde_json::json!({"type":"configure","sample_rate":16000})
                        .to_string()
                        .into(),
                ))
                .await
                .unwrap();
                if prior {
                    ws.send(Message::Binary(vec![0; 32_000].into()))
                        .await
                        .unwrap();
                }
                ws.send(Message::Binary(vec![0; 3_200].into()))
                    .await
                    .unwrap();
                ws.send(Message::Text(
                    serde_json::json!({"type":"stop"}).to_string().into(),
                ))
                .await
                .unwrap();
                let mut saw_error = false;
                let mut saw_partial = false;
                loop {
                    match tokio::time::timeout(Duration::from_secs(5), ws.next())
                        .await
                        .unwrap()
                        .unwrap()
                        .unwrap()
                    {
                        Message::Text(text) => {
                            let json: serde_json::Value = serde_json::from_str(&text).unwrap();
                            match json["type"].as_str() {
                                Some("partial") => {
                                    saw_partial = true;
                                    assert!(json["text"].as_str().unwrap().contains("hi"));
                                }
                                Some("error") => {
                                    saw_error = true;
                                    assert_eq!(
                                        json["code"],
                                        if panic {
                                            "inference_panic"
                                        } else {
                                            "inference_error"
                                        }
                                    );
                                }
                                other => panic!("unexpected tail response {other:?}: {json}"),
                            }
                        }
                        Message::Close(frame) => {
                            assert_eq!(u16::from(frame.unwrap().code), 1011);
                            break;
                        }
                        other => panic!("unexpected frame: {other:?}"),
                    }
                }
                assert!(saw_error);
                assert_eq!(saw_partial, prior);
                let _ = shutdown_tx.send(());
                server.await.unwrap().unwrap();
            }
        }
    }
}

#[tokio::test]
async fn test_live_ws_endpoint_duration_validation_and_legacy_versions() {
    let (port, shutdown, _tmp) = spawn_mock_server().await;
    for version in ["1.0", "1.1", "1.2", "1.3"] {
        let (ws, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}/v1/ws"))
            .await
            .unwrap();
        let (mut sink, mut stream) = ws.split();
        assert_eq!(next_json(&mut stream).await["type"], "ready");
        for ms in [0, 39, 2001, u32::MAX] {
            sink.send(Message::Text(serde_json::json!({"type":"configure","protocol_version":version,"endpoint_silence_ms":ms}).to_string().into())).await.unwrap();
            assert_eq!(
                next_json(&mut stream).await["code"],
                "invalid_endpoint_silence_ms"
            );
        }
        sink.send(Message::Text(serde_json::json!({"type":"configure","protocol_version":version,"endpoint_silence_ms":1500}).to_string().into())).await.unwrap();
        sink.send(Message::Text(
            serde_json::json!({"type":"stop"}).to_string().into(),
        ))
        .await
        .unwrap();
        assert_eq!(next_json(&mut stream).await["type"], "final");
    }
    let _ = shutdown.send(());
}
