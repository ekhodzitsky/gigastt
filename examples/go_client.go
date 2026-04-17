// WebSocket client for gigastt — streams a WAV file and prints transcription.
//
// Setup:
//   go mod init gigastt-client
//   go get github.com/gorilla/websocket
//
// Usage:
//   go run examples/go_client.go <audio.wav> [ws://host:port]

package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/url"
	"os"

	"github.com/gorilla/websocket"
)

const (
	wavHeaderBytes = 44
	chunkBytes     = 32768 // ~1s at 16kHz PCM16
)

type serverMsg struct {
	Type    string `json:"type"`
	Text    string `json:"text"`
	Model   string `json:"model"`
	Rate    int    `json:"sample_rate"`
	Message string `json:"message"`
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <audio.wav> [ws://host:port]\n", os.Args[0])
		os.Exit(1)
	}

	wavPath := os.Args[1]
	server := "ws://127.0.0.1:9876/ws"
	if len(os.Args) > 2 {
		server = os.Args[2] + "/ws"
		if _, err := url.Parse(server); err != nil {
			log.Fatalf("invalid server URL: %v", err)
		}
	}

	data, err := os.ReadFile(wavPath)
	if err != nil {
		log.Fatalf("read WAV: %v", err)
	}
	if len(data) <= wavHeaderBytes {
		log.Fatal("file too small to be a valid WAV")
	}
	pcm := data[wavHeaderBytes:]

	conn, _, err := websocket.DefaultDialer.Dial(server, nil)
	if err != nil {
		log.Fatalf("dial %s: %v", server, err)
	}
	defer conn.Close()

	// Receive loop in goroutine
	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			_, raw, err := conn.ReadMessage()
			if err != nil {
				return
			}
			var msg serverMsg
			if err := json.Unmarshal(raw, &msg); err != nil {
				continue
			}
			switch msg.Type {
			case "ready":
				fmt.Printf("Connected: %s @ %dHz\n\n", msg.Model, msg.Rate)
			case "partial":
				fmt.Printf("\r  ... %s", msg.Text)
			case "final":
				fmt.Printf("\r  >>> %s\n", msg.Text)
				return
			case "error":
				fmt.Fprintf(os.Stderr, "\n  ERR: %s\n", msg.Message)
				return
			}
		}
	}()

	// Wait for ready, then send audio
	// gorilla/websocket is synchronous; send after connection is established
	for offset := 0; offset < len(pcm); offset += chunkBytes {
		end := offset + chunkBytes
		if end > len(pcm) {
			end = len(pcm)
		}
		if err := conn.WriteMessage(websocket.BinaryMessage, pcm[offset:end]); err != nil {
			log.Fatalf("send chunk: %v", err)
		}
	}

	stop, _ := json.Marshal(map[string]string{"type": "stop"})
	if err := conn.WriteMessage(websocket.TextMessage, stop); err != nil {
		log.Fatalf("send stop: %v", err)
	}

	<-done
}
