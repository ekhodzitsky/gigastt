package gigastt

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"testing"
	"time"
)

// TestLiveServer exercises the full wire protocol against a live gigastt
// server. It is skipped unless GIGASTT_TEST_WS_URL points at a running server
// (which requires the model, ~850 MB):
//
//	cargo run --release -- serve   # in the repository root
//	GIGASTT_TEST_WS_URL=ws://127.0.0.1:9876/v1/ws go test -run Live -v
func TestLiveServer(t *testing.T) {
	url := os.Getenv("GIGASTT_TEST_WS_URL")
	if url == "" {
		t.Skip("GIGASTT_TEST_WS_URL not set; skipping live-server integration test")
	}

	finals := make(chan Transcript, 1)
	c, err := Dial(context.Background(), url,
		WithSampleRate(16000),
		WithHandlers(Handlers{
			OnFinal: func(tr Transcript) { finals <- tr },
			OnError: func(se *ServerError) { t.Errorf("server error: %v", se) },
		}),
	)
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer c.Close()

	if c.Ready().Model == "" {
		t.Error("ready.model is empty")
	}

	// Stream 2 seconds of a 440 Hz sine tone as 16 kHz PCM16 in 20ms frames.
	const (
		rate       = 16000
		frameSamps = rate / 50
		duration   = 2 * time.Second
	)
	frame := make([]byte, frameSamps*2)
	totalFrames := int(duration.Seconds() * 50)
	for i := 0; i < totalFrames; i++ {
		for j := 0; j < frameSamps; j++ {
			sample := int16(12000 * math.Sin(2*math.Pi*440*float64(i*frameSamps+j)/rate))
			binary.LittleEndian.PutUint16(frame[j*2:], uint16(sample))
		}
		if err := c.SendPCM(frame); err != nil {
			t.Fatalf("SendPCM: %v", err)
		}
		time.Sleep(10 * time.Millisecond)
	}

	if err := c.Stop(); err != nil {
		t.Fatalf("Stop: %v", err)
	}

	select {
	case f := <-finals:
		if !f.IsFinal {
			t.Error("final transcript has is_final=false")
		}
		t.Logf("live final transcript: %q (%d words)", f.Text, len(f.Words))
	case <-time.After(30 * time.Second):
		t.Fatal("no final transcript within 30s")
	}
}
