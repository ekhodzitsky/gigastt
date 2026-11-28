use super::*;

fn base() -> Selection {
    Selection {
        context: V4StreamContext::Ms320,
        style: TranscriptStyle::Raw,
    }
}

#[test]
fn test_v4_pcm16_even_unaligned_packet_geometry_and_exact_scale() {
    let samples = pcm16(&[0, 128, 255, 255, 0, 0, 255, 127], 8).unwrap();
    assert_eq!(samples, [-1., -1. / 32768., 0., 32767. / 32768.]);
    assert!(pcm16(&[0, 0, 1], 10).is_err());
    assert!(pcm16(&[0, 0, 1, 0], 2).is_err());
    assert!(pcm16(&[], 4).is_err());
    assert!(pcm16(&[0; 32_000], 32_768).is_ok());
    assert!(pcm16(&[0; 32_002], 32_768).is_err());
}

#[test]
fn test_v4_query_rejects_unknown_duplicate_and_cross_family_selectors() {
    let select = |query: &str| {
        let uri = format!("/v1/ws?{query}").parse().unwrap();
        parse_selection(&uri, base())
    };
    assert_eq!(
        select("future_context_ms=640&transcript_style=e2e").unwrap(),
        Selection {
            context: V4StreamContext::Ms640,
            style: TranscriptStyle::E2e
        }
    );
    for q in [
        "future_context_ms=240",
        "future_context_ms=160&future_context_ms=640",
        "language=ru",
        "model_family=gigaam_v3",
        "model_precision=int8",
        "model_head=ctc",
        "transcript_style=unknown",
    ] {
        assert!(select(q).is_err(), "{q}");
    }
}

#[test]
fn test_v4_configure_confirm_before_audio_only_and_stop_distinct() {
    assert!(matches!(
        command(r#"{"type":"stop"}"#, base(), true),
        Ok(Command::Stop)
    ));
    assert!(matches!(
        command(
            r#"{"type":"configure","sample_rate":16000,"protocol_version":"1.0","future_context_ms":320,"transcript_style":"raw"}"#,
            base(),
            false
        ),
        Ok(Command::Configure)
    ));
    for body in [
        r#"{"type":"configure","sample_rate":48000}"#,
        r#"{"type":"configure","future_context_ms":640}"#,
        r#"{"type":"configure","diarization":true}"#,
        r#"{"type":"configure","language":"ru"}"#,
        r#"{"type":"configure","protocol_version":"2.0"}"#,
        r#"{"type":"configure","endpoint_mode":"auto"}"#,
    ] {
        assert!(command(body, base(), false).is_err(), "{body}");
    }
    assert_eq!(
        command(r#"{"type":"configure"}"#, base(), true)
            .unwrap_err()
            .code,
        "configure_too_late"
    );
}

#[test]
fn test_v4_native_subprotocol_requires_exact_offer() {
    let mut headers = HeaderMap::new();
    assert!(!native_protocol_requested(&headers));
    headers.insert(
        axum::http::header::SEC_WEBSOCKET_PROTOCOL,
        "gigastt-native-v4-old".parse().unwrap(),
    );
    assert!(!native_protocol_requested(&headers));
    headers.append(
        axum::http::header::SEC_WEBSOCKET_PROTOCOL,
        "legacy, gigastt-native-v4".parse().unwrap(),
    );
    assert!(native_protocol_requested(&headers));
}

#[test]
fn test_v4_native_publication_keeps_cumulative_envelope_and_exact_new_delta() {
    use gigastt_core::inference::v4::{V4StreamingDelta, V4Token};
    let mut a = Snapshot::default();
    let packet = |n, final_packet, text: &str, words: Vec<WordInfo>, tokens: Vec<V4Token>| {
        V4StreamingPacket {
            raw: Some(V4StreamingDelta {
                text: text.into(),
                words,
                tokens,
            }),
            e2e: None,
            samples: n * 160,
            encoded_frames: n,
            finished: final_packet,
        }
    };
    assert!(
        a.apply(
            packet(1, false, "", vec![], vec![]),
            TranscriptStyle::Raw,
            false
        )
        .unwrap()
        .is_none()
    );
    let word = WordInfo::new("да", 0.0, 0.04, 0.9, None);
    let token = V4Token {
        id: 23,
        frame: 0,
        confidence: 0.9,
    };
    let first = a
        .apply(
            packet(2, false, "да", vec![word.clone()], vec![token]),
            TranscriptStyle::Raw,
            false,
        )
        .unwrap()
        .unwrap();
    let message =
        serde_json::to_value(wire::Transcript::new(&a.text, &a.words, &first, false)).unwrap();
    assert_eq!(message["text"], "да");
    assert_eq!(message["committed"], "да");
    assert_eq!(message["tentative"], "");
    assert_eq!(message["native_delta"]["packet_one_based"], 2);
    assert_eq!(message["native_delta"]["tokens"][0]["id"], 23);
    assert!(message.get("speech_final").is_none());
    let last = a
        .apply(
            packet(3, true, " да", vec![word], vec![]),
            TranscriptStyle::Raw,
            true,
        )
        .unwrap()
        .unwrap();
    let message =
        serde_json::to_value(wire::Transcript::new(&a.text, &a.words, &last, true)).unwrap();
    assert_eq!(message["type"], "final");
    assert_eq!(message["text"], "да да");
    assert_eq!(message["native_delta"]["text"], " да");
    assert_eq!(message["endpoint_reason"], "stop");
    assert_eq!(message["words"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn test_v4_native_cancel_does_not_release_active_worker_reservation() {
    let capacity = Arc::new(Semaphore::new(1));
    let permit = capacity.clone().acquire_owned().await.unwrap();
    let (started, wait) = tokio::sync::oneshot::channel();
    let (release, done) = std::sync::mpsc::channel();
    let worker = tokio::task::spawn_blocking(move || {
        started.send(()).unwrap();
        done.recv().unwrap();
        drop(permit)
    });
    wait.await.unwrap();
    let shutdown = CancellationToken::new();
    let closed = CancellationToken::new();
    let abort = AtomicBool::new(false);
    closed.cancel();
    assert_eq!(
        await_packet(
            worker,
            &abort,
            &shutdown,
            &closed,
            tokio::time::Instant::now() + Duration::from_secs(1),
            0
        )
        .await,
        Err(WaitFailure::Disconnected)
    );
    assert!(abort.load(Ordering::Relaxed));
    assert_eq!(capacity.available_permits(), 0);
    release.send(()).unwrap();
    assert!(
        tokio::time::timeout(Duration::from_secs(1), capacity.acquire())
            .await
            .unwrap()
            .is_ok()
    );
}

#[tokio::test]
async fn test_v4_native_stalled_writer_is_bounded() {
    use std::{
        pin::Pin,
        task::{Context, Poll},
    };
    struct Stalled;
    impl Sink<Message> for Stalled {
        type Error = axum::Error;
        fn poll_ready(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Pending
        }
        fn start_send(self: Pin<&mut Self>, _: Message) -> Result<(), Self::Error> {
            panic!("not writable")
        }
        fn poll_flush(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Pending
        }
        fn poll_close(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Pending
        }
    }
    assert!(
        send_within(
            &mut Stalled,
            Message::Ping(Vec::new().into()),
            Duration::from_millis(1)
        )
        .await
        .is_err()
    );
}
