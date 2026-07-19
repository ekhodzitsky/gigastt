package gigastt

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

// mockServer upgrades every HTTP request to WebSocket and runs the script for
// the connection. script runs in the handler goroutine; n is the 1-based
// connection counter. Returning from script closes the connection without a
// close frame (abnormal close), unless the script already closed it cleanly.
type mockServer struct {
	srv   *httptest.Server
	url   string
	conns atomic.Int32
}

func newMockServer(t *testing.T, script func(conn *websocket.Conn, n int)) *mockServer {
	t.Helper()
	m := &mockServer{}
	up := websocket.Upgrader{}
	m.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := up.Upgrade(w, r, nil)
		if err != nil {
			t.Errorf("upgrade: %v", err)
			return
		}
		n := int(m.conns.Add(1))
		script(conn, n)
		_ = conn.Close()
	}))
	m.url = "ws" + strings.TrimPrefix(m.srv.URL, "http") + "/v1/ws"
	t.Cleanup(m.srv.Close)
	return m
}

// writeJSON sends v as a text frame.
func writeJSON(t *testing.T, conn *websocket.Conn, v map[string]any) {
	t.Helper()
	data, err := json.Marshal(v)
	if err != nil {
		t.Errorf("marshal: %v", err)
		return
	}
	if err := conn.WriteMessage(websocket.TextMessage, data); err != nil {
		t.Errorf("write JSON: %v", err)
	}
}

// readJSON reads one text frame and decodes it.
func readJSON(t *testing.T, conn *websocket.Conn) map[string]any {
	t.Helper()
	mt, raw, err := conn.ReadMessage()
	if err != nil {
		t.Errorf("read JSON: %v", err)
		return nil
	}
	if mt != websocket.TextMessage {
		t.Errorf("expected text frame, got message type %d", mt)
		return nil
	}
	var v map[string]any
	if err := json.Unmarshal(raw, &v); err != nil {
		t.Errorf("unmarshal: %v", err)
		return nil
	}
	return v
}

func readyPayload() map[string]any {
	return map[string]any{
		"type":                 "ready",
		"model":                "gigaam-v3-e2e-rnnt",
		"sample_rate":          48000,
		"version":              "1.0",
		"supported_rates":      []int{8000, 16000, 24000, 44100, 48000},
		"diarization":          true,
		"min_protocol_version": "1.0",
	}
}

// expectConfigure reads the configure frame the client must send first.
func expectConfigure(t *testing.T, conn *websocket.Conn) map[string]any {
	t.Helper()
	msg := readJSON(t, conn)
	if msg["type"] != "configure" {
		t.Errorf("expected configure, got %v", msg["type"])
	}
	return msg
}

// waitFor receives from ch with a timeout.
func waitFor[T any](t *testing.T, ch <-chan T, what string) T {
	t.Helper()
	select {
	case v := <-ch:
		return v
	case <-time.After(5 * time.Second):
		t.Fatalf("timeout waiting for %s", what)
		var zero T
		return zero
	}
}

func TestDialSendsConfigureAndReceivesReady(t *testing.T) {
	configureCh := make(chan map[string]any, 1)
	m := newMockServer(t, func(conn *websocket.Conn, n int) {
		configureCh <- expectConfigure(t, conn)
		writeJSON(t, conn, readyPayload())
		// Keep the session open until the client disconnects.
		_, _, _ = conn.ReadMessage()
	})

	c, err := Dial(context.Background(), m.url,
		WithSampleRate(16000),
		WithDiarization(true),
	)
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}

	cfg := waitFor(t, configureCh, "configure")
	if cfg["protocol_version"] != ProtocolVersion {
		t.Errorf("protocol_version = %v, want %q", cfg["protocol_version"], ProtocolVersion)
	}
	if cfg["sample_rate"] != float64(16000) {
		t.Errorf("sample_rate = %v, want 16000", cfg["sample_rate"])
	}
	if cfg["diarization"] != true {
		t.Errorf("diarization = %v, want true", cfg["diarization"])
	}

	r := c.Ready()
	if r.Model != "gigaam-v3-e2e-rnnt" {
		t.Errorf("Model = %q", r.Model)
	}
	if r.SampleRate != 48000 {
		t.Errorf("SampleRate = %d", r.SampleRate)
	}
	if r.Version != "1.0" {
		t.Errorf("Version = %q", r.Version)
	}
	if len(r.SupportedRates) != 5 || r.SupportedRates[0] != 8000 {
		t.Errorf("SupportedRates = %v", r.SupportedRates)
	}
	if !r.Diarization {
		t.Errorf("Diarization = false, want true")
	}
	if r.MinProtocolVersion != "1.0" {
		t.Errorf("MinProtocolVersion = %q", r.MinProtocolVersion)
	}

	if err := c.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	<-c.Done()
}

func TestPartialAndFinalDispatch(t *testing.T) {
	m := newMockServer(t, func(conn *websocket.Conn, n int) {
		expectConfigure(t, conn)
		writeJSON(t, conn, readyPayload())
		writeJSON(t, conn, map[string]any{
			"type":      "partial",
			"text":      "привет",
			"timestamp": 1712700000.5,
			"is_final":  false,
			"words": []map[string]any{
				{"word": "привет", "start": 0.1, "end": 0.6, "confidence": 0.95, "speaker": 2},
			},
		})
		writeJSON(t, conn, map[string]any{
			"type":      "final",
			"text":      "привет как дела",
			"timestamp": 1712700001.0,
			"is_final":  true,
			"words": []map[string]any{
				{"word": "привет", "start": 0.1, "end": 0.6, "confidence": 0.95},
				{"word": "как", "start": 0.7, "end": 0.9, "confidence": 0.9},
				{"word": "дела", "start": 1.0, "end": 1.4, "confidence": 0.88},
			},
		})
		_, _, _ = conn.ReadMessage()
	})

	partials := make(chan Transcript, 1)
	finals := make(chan Transcript, 1)
	c, err := Dial(context.Background(), m.url, WithHandlers(Handlers{
		OnPartial: func(tr Transcript) { partials <- tr },
		OnFinal:   func(tr Transcript) { finals <- tr },
	}))
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}

	p := waitFor(t, partials, "partial")
	if p.Text != "привет" || p.IsFinal {
		t.Errorf("partial = %+v", p)
	}
	if p.Timestamp != 1712700000.5 {
		t.Errorf("partial timestamp = %v", p.Timestamp)
	}
	if len(p.Words) != 1 {
		t.Fatalf("partial words = %v", p.Words)
	}
	w := p.Words[0]
	if w.Word != "привет" || w.Start != 0.1 || w.End != 0.6 || w.Confidence != 0.95 {
		t.Errorf("word = %+v", w)
	}
	if w.Speaker == nil || *w.Speaker != 2 {
		t.Errorf("speaker = %v, want 2", w.Speaker)
	}

	f := waitFor(t, finals, "final")
	if f.Text != "привет как дела" || !f.IsFinal {
		t.Errorf("final = %+v", f)
	}
	if len(f.Words) != 3 {
		t.Fatalf("final words = %v", f.Words)
	}
	if f.Words[1].Speaker != nil {
		t.Errorf("speaker must be nil when omitted, got %v", f.Words[1].Speaker)
	}

	_ = c.Close()
	<-c.Done()
}

func TestErrorDispatch(t *testing.T) {
	m := newMockServer(t, func(conn *websocket.Conn, n int) {
		expectConfigure(t, conn)
		writeJSON(t, conn, readyPayload())
		writeJSON(t, conn, map[string]any{
			"type":    "error",
			"message": "Inference failed. Please check audio format.",
			"code":    "inference_error",
		})
		_, _, _ = conn.ReadMessage()
	})

	errs := make(chan *ServerError, 1)
	c, err := Dial(context.Background(), m.url, WithHandlers(Handlers{
		OnError: func(se *ServerError) { errs <- se },
	}))
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}

	se := waitFor(t, errs, "error")
	if se.Code != ErrCodeInferenceError {
		t.Errorf("code = %q", se.Code)
	}
	if se.RetryAfterMs != nil {
		t.Errorf("retry_after_ms must be nil when omitted, got %v", *se.RetryAfterMs)
	}
	if !strings.Contains(se.Error(), "inference_error") {
		t.Errorf("Error() = %q", se.Error())
	}

	_ = c.Close()
	<-c.Done()
}

func TestRetryAfterMsHonoredOnReconnect(t *testing.T) {
	const retryAfterMs = 250
	m := newMockServer(t, func(conn *websocket.Conn, n int) {
		expectConfigure(t, conn)
		if n == 1 {
			// Pool saturation: reject with an explicit retry hint.
			writeJSON(t, conn, map[string]any{
				"type":           "error",
				"message":        "Server busy, try again later",
				"code":           "timeout",
				"retry_after_ms": retryAfterMs,
			})
			return
		}
		writeJSON(t, conn, readyPayload())
		_, _, _ = conn.ReadMessage()
	})

	start := time.Now()
	c, err := Dial(context.Background(), m.url,
		WithReconnect(5*time.Millisecond, 100*time.Millisecond, 5),
	)
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	elapsed := time.Since(start)

	if got := m.conns.Load(); got != 2 {
		t.Errorf("connections = %d, want 2", got)
	}
	// The 250ms server hint must be honored exactly; the 5ms exponential
	// backoff would have reconnected almost immediately.
	if elapsed < 200*time.Millisecond {
		t.Errorf("reconnected after %v, want >= ~%dms (retry_after_ms not honored)", elapsed, retryAfterMs)
	}
	if elapsed > 10*time.Second {
		t.Errorf("reconnect took suspiciously long: %v", elapsed)
	}

	_ = c.Close()
	<-c.Done()
}

func TestReconnectAfterAbnormalDrop(t *testing.T) {
	m := newMockServer(t, func(conn *websocket.Conn, n int) {
		expectConfigure(t, conn)
		writeJSON(t, conn, readyPayload())
		if n == 1 {
			// Wait for one audio frame (proving the client processed ready),
			// then drop the connection abnormally: no close frame.
			if _, _, err := conn.ReadMessage(); err != nil {
				t.Errorf("read before drop: %v", err)
			}
			return
		}
		_, _, _ = conn.ReadMessage()
	})

	readyCount := make(chan Ready, 4)
	closeCh := make(chan error, 1)
	c, err := Dial(context.Background(), m.url,
		WithReconnect(10*time.Millisecond, 50*time.Millisecond, 5),
		WithHandlers(Handlers{
			OnReady: func(r Ready) { readyCount <- r },
			OnClose: func(err error) { closeCh <- err },
		}),
	)
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}

	waitFor(t, readyCount, "initial ready")
	if err := c.SendPCM([]byte{0x00, 0x00}); err != nil {
		t.Fatalf("SendPCM: %v", err)
	}
	waitFor(t, readyCount, "ready after reconnect")
	if got := m.conns.Load(); got != 2 {
		t.Errorf("connections = %d, want 2", got)
	}

	_ = c.Close()
	if err := waitFor(t, closeCh, "OnClose"); err != nil {
		t.Errorf("OnClose err = %v, want nil after Close", err)
	}
}

func TestStopFinalizesSessionWithoutReconnect(t *testing.T) {
	stopReceived := make(chan struct{}, 1)
	m := newMockServer(t, func(conn *websocket.Conn, n int) {
		expectConfigure(t, conn)
		writeJSON(t, conn, readyPayload())
		msg := readJSON(t, conn)
		if msg["type"] != "stop" {
			t.Errorf("expected stop, got %v", msg["type"])
			return
		}
		stopReceived <- struct{}{}
		writeJSON(t, conn, map[string]any{
			"type":      "final",
			"text":      "готово",
			"timestamp": 1712700002.0,
			"is_final":  true,
			"words":     []map[string]any{},
		})
		// Server ends the session after the final transcript.
	})

	finals := make(chan Transcript, 1)
	closeCh := make(chan error, 1)
	c, err := Dial(context.Background(), m.url,
		WithReconnect(10*time.Millisecond, 50*time.Millisecond, 5),
		WithHandlers(Handlers{
			OnFinal: func(tr Transcript) { finals <- tr },
			OnClose: func(err error) { closeCh <- err },
		}),
	)
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}

	if err := c.Stop(); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	waitFor(t, stopReceived, "stop frame")
	if f := waitFor(t, finals, "final"); f.Text != "готово" {
		t.Errorf("final text = %q", f.Text)
	}

	select {
	case <-c.Done():
	case <-time.After(5 * time.Second):
		t.Fatal("Done not closed after session end")
	}
	if err := <-closeCh; err != nil {
		t.Errorf("OnClose err = %v, want nil for Stop-finalized session", err)
	}
	// The session ended intentionally: no reconnect must have happened.
	if got := m.conns.Load(); got != 1 {
		t.Errorf("connections = %d, want 1 (no reconnect after stop)", got)
	}
}

func TestFatalServerErrorNotRetried(t *testing.T) {
	m := newMockServer(t, func(conn *websocket.Conn, n int) {
		expectConfigure(t, conn)
		writeJSON(t, conn, map[string]any{
			"type":    "error",
			"message": "Unsupported protocol version",
			"code":    "unsupported_protocol_version",
		})
	})

	_, err := Dial(context.Background(), m.url,
		WithReconnect(5*time.Millisecond, 50*time.Millisecond, 5),
	)
	if err == nil {
		t.Fatal("Dial must fail on unsupported protocol version")
	}
	var se *ServerError
	if !errors.As(err, &se) {
		t.Fatalf("error %v is not a *ServerError", err)
	}
	if se.Code != ErrCodeUnsupportedProtocolVersion {
		t.Errorf("code = %q", se.Code)
	}
	if got := m.conns.Load(); got != 1 {
		t.Errorf("connections = %d, want 1 (fatal errors are not retried)", got)
	}
}

func TestUnknownFieldsIgnored(t *testing.T) {
	m := newMockServer(t, func(conn *websocket.Conn, n int) {
		expectConfigure(t, conn)
		ready := readyPayload()
		ready["future_field"] = map[string]any{"nested": true}
		writeJSON(t, conn, ready)
		writeJSON(t, conn, map[string]any{
			"type":       "partial",
			"text":       "ok",
			"timestamp":  1.0,
			"is_final":   false,
			"confidence": 0.5, // hypothetical additive field
		})
		writeJSON(t, conn, map[string]any{"type": "brand_new_message", "x": 1})
		_, _, _ = conn.ReadMessage()
	})

	partials := make(chan Transcript, 1)
	c, err := Dial(context.Background(), m.url, WithHandlers(Handlers{
		OnPartial: func(tr Transcript) { partials <- tr },
	}))
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	if p := waitFor(t, partials, "partial"); p.Text != "ok" {
		t.Errorf("partial text = %q", p.Text)
	}
	_ = c.Close()
	<-c.Done()
}

func TestContextCancelDuringRetryBackoff(t *testing.T) {
	m := newMockServer(t, func(conn *websocket.Conn, n int) {
		expectConfigure(t, conn)
		writeJSON(t, conn, map[string]any{
			"type":           "error",
			"message":        "Server busy, try again later",
			"code":           "timeout",
			"retry_after_ms": 60000,
		})
	})

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(100 * time.Millisecond)
		cancel()
	}()

	start := time.Now()
	_, err := Dial(ctx, m.url, WithReconnect(5*time.Millisecond, 100*time.Millisecond, 0))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if time.Since(start) > 10*time.Second {
		t.Errorf("cancel during backoff took too long: %v", time.Since(start))
	}
	if got := m.conns.Load(); got != 1 {
		t.Errorf("connections = %d, want 1", got)
	}
}

func TestSendPCM(t *testing.T) {
	frames := make(chan []byte, 3)
	m := newMockServer(t, func(conn *websocket.Conn, n int) {
		expectConfigure(t, conn)
		writeJSON(t, conn, readyPayload())
		for i := 0; i < 3; i++ {
			mt, raw, err := conn.ReadMessage()
			if err != nil {
				t.Errorf("read frame %d: %v", i, err)
				return
			}
			if mt != websocket.BinaryMessage {
				t.Errorf("frame %d type = %d, want binary", i, mt)
				return
			}
			frames <- raw
		}
		_, _, _ = conn.ReadMessage()
	})

	c, err := Dial(context.Background(), m.url)
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}

	payloads := [][]byte{
		{0x01, 0x02, 0x03},
		{0x00, 0x00},
		make([]byte, 320), // 10ms of 16kHz PCM16, silence
	}
	for _, p := range payloads {
		if err := c.SendPCM(p); err != nil {
			t.Fatalf("SendPCM: %v", err)
		}
	}
	for i, want := range payloads {
		got := waitFor(t, frames, "pcm frame")
		if len(got) != len(want) {
			t.Errorf("frame %d len = %d, want %d", i, len(got), len(want))
			continue
		}
		for j := range want {
			if got[j] != want[j] {
				t.Errorf("frame %d byte %d = %d, want %d", i, j, got[j], want[j])
				break
			}
		}
	}

	_ = c.Close()
	<-c.Done()
}

func TestSendAfterCloseFails(t *testing.T) {
	m := newMockServer(t, func(conn *websocket.Conn, n int) {
		expectConfigure(t, conn)
		writeJSON(t, conn, readyPayload())
		_, _, _ = conn.ReadMessage()
	})

	c, err := Dial(context.Background(), m.url)
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	if err := c.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := c.SendPCM([]byte{0x01}); !errors.Is(err, ErrClosed) {
		t.Errorf("SendPCM after Close = %v, want ErrClosed", err)
	}
	if err := c.Stop(); !errors.Is(err, ErrClosed) {
		t.Errorf("Stop after Close = %v, want ErrClosed", err)
	}
	// Close is idempotent.
	if err := c.Close(); err != nil {
		t.Errorf("second Close = %v", err)
	}
}

func TestMaxReconnectAttemptsExceeded(t *testing.T) {
	m := newMockServer(t, func(conn *websocket.Conn, n int) {
		expectConfigure(t, conn)
		writeJSON(t, conn, map[string]any{
			"type":           "error",
			"message":        "Server busy, try again later",
			"code":           "timeout",
			"retry_after_ms": 20,
		})
	})

	_, err := Dial(context.Background(), m.url,
		WithReconnect(5*time.Millisecond, 50*time.Millisecond, 2),
	)
	if err == nil {
		t.Fatal("Dial must fail after exhausting reconnect attempts")
	}
	var se *ServerError
	if !errors.As(err, &se) {
		t.Errorf("wrapped error should still expose *ServerError, got %v", err)
	}
	// 1 initial attempt + 2 retries.
	if got := m.conns.Load(); got != 3 {
		t.Errorf("connections = %d, want 3", got)
	}
}
