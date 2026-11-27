package gigastt

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gorilla/websocket"
)

func TestNativeV4ReadyAndDelta(t *testing.T) {
	configured := make(chan map[string]any, 1)
	final := make(chan Transcript, 1)
	up := websocket.Upgrader{Subprotocols: []string{NativeV4Subprotocol}}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("future_context_ms") != "640" || r.URL.Query().Get("transcript_style") != "e2e" {
			t.Error("missing explicit native query")
		}
		ws, err := up.Upgrade(w, r, nil)
		if err != nil {
			t.Error(err)
			return
		}
		defer ws.Close()
		if ws.Subprotocol() != NativeV4Subprotocol {
			t.Error("wrong subprotocol")
			return
		}
		configured <- expectConfigure(t, ws)
		writeJSON(t, ws, map[string]any{"type": "ready", "model": "gigaam-v4-rnnt-e2e-fp32-streaming", "version": "1.4", "sample_rate": 16000, "supported_rates": []int{16000}, "native_v4": NativeV4Capability{ModelFamily: "gigaam_v4", ModelHead: "rnnt", ModelPrecision: "fp32", TranscriptStyle: "e2e", FutureContextMS: 640, SupportedFutureContextMS: []uint16{160, 320, 640}, MaxPacketBytes: 32000, EndpointMode: "manual"}})
		if msg := readJSON(t, ws); msg["type"] != "stop" {
			t.Errorf("stop=%v", msg)
			return
		}
		writeJSON(t, ws, map[string]any{"type": "final", "text": "да да", "is_final": true, "native_delta": NativeV4Delta{Text: " да", PacketOneBased: 2}})
	}))
	defer server.Close()
	c, err := Dial(context.Background(), "ws"+strings.TrimPrefix(server.URL, "http")+"/v1/ws", WithNativeV4("e2e", 640), WithHandlers(Handlers{OnFinal: func(v Transcript) { final <- v }}))
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	cfg := waitFor(t, configured, "native configure")
	if cfg["protocol_version"] != "1.4" || cfg["sample_rate"] != float64(16000) || cfg["future_context_ms"] != float64(640) {
		t.Errorf("configure=%v", cfg)
	}
	if c.SendPCM([]byte{0, 1, 2}) == nil {
		t.Error("odd PCM16 must fail")
	}
	if err := c.Stop(); err != nil {
		t.Fatal(err)
	}
	out := waitFor(t, final, "native final")
	if out.Text != "да да" || out.NativeDelta == nil || out.NativeDelta.Text != " да" {
		t.Errorf("final=%+v", out)
	}
}

func TestNativeV4RejectsLegacySubprotocolWithoutSendingPCM(t *testing.T) {
	m := newMockServer(t, func(conn *websocket.Conn, _ int) {
		if _, _, err := conn.ReadMessage(); err == nil {
			t.Error("client sent a frame without negotiated v4")
		}
	})
	_, err := Dial(context.Background(), m.url, WithNativeV4("raw", 320))
	var response *ServerError
	if !errors.As(err, &response) || response.Code != "native_v4_mismatch" {
		t.Fatalf("error=%v", err)
	}
}
