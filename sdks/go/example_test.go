package gigastt_test

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	gigastt "github.com/ekhodzitsky/gigastt/sdks/go"
)

// ExampleDial streams a WAV file to a local gigastt server and prints partial
// and final transcripts. It is compile-checked only; run it against a live
// server as described in the README.
func ExampleDial() {
	ctx := context.Background()

	finals := make(chan gigastt.Transcript, 1)
	client, err := gigastt.Dial(ctx, gigastt.DefaultURL,
		gigastt.WithSampleRate(16000),
		gigastt.WithReconnect(250*time.Millisecond, 5*time.Second, 10),
		gigastt.WithHandlers(gigastt.Handlers{
			OnPartial: func(t gigastt.Transcript) {
				fmt.Printf("\r  ... %s", t.Text)
			},
			OnFinal: func(t gigastt.Transcript) {
				fmt.Printf("\r  >>> %s\n", t.Text)
				finals <- t
			},
			OnError: func(e *gigastt.ServerError) {
				log.Printf("server error: %v", e)
			},
		}),
	)
	if err != nil {
		log.Fatalf("dial: %v", err)
	}
	defer client.Close()

	fmt.Printf("connected: %s @ %d Hz\n", client.Ready().Model, client.Ready().SampleRate)

	// Stream a 16 kHz mono WAV file (skipping the 44-byte header) in chunks.
	wav, err := os.ReadFile("recording.wav")
	if err != nil {
		log.Fatalf("read wav: %v", err)
	}
	const header, chunk = 44, 32768
	for offset := header; offset < len(wav); offset += chunk {
		end := min(offset+chunk, len(wav))
		if err := client.SendPCM(wav[offset:end]); err != nil {
			log.Fatalf("send: %v", err)
		}
	}

	// Ask the server to finalize, then wait for the last transcript.
	if err := client.Stop(); err != nil {
		log.Fatalf("stop: %v", err)
	}
	<-finals
}
