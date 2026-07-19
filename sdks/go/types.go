// Package gigastt provides a typed WebSocket client for the gigastt
// speech-to-text server (WebSocket protocol version 1.0).
//
// The client connects to the server's /v1/ws endpoint, negotiates the input
// sample rate, streams raw PCM16 mono audio as binary frames, and dispatches
// typed Ready/Partial/Final/Error events to user callbacks. Optional
// automatic reconnect uses exponential backoff and honors the server's
// retry_after_ms hint on transient backpressure errors (pool saturation).
//
// Wire protocol reference: docs/asyncapi.yaml in the gigastt repository.
package gigastt

// ProtocolVersion is the WebSocket protocol version this SDK speaks. The SDK
// pins to protocol "1.0" and announces it in the configure message; the
// server rejects the session with an unsupported_protocol_version error if it
// cannot speak it. Additive wire changes (new fields on existing messages)
// are tolerated: unknown JSON fields are ignored on decode.
const ProtocolVersion = "1.0"

// DefaultURL is the default server endpoint (loopback bind, default port).
const DefaultURL = "ws://127.0.0.1:9876/v1/ws"

// Machine-readable error codes sent by the server in Error messages
// (see docs/asyncapi.yaml, ErrorMessage.code).
const (
	ErrCodeInferenceError             = "inference_error"
	ErrCodeInferencePanic             = "inference_panic"
	ErrCodeInferenceTimeout           = "inference_timeout"
	ErrCodeConfigureTooLate           = "configure_too_late"
	ErrCodeInvalidSampleRate          = "invalid_sample_rate"
	ErrCodeTimeout                    = "timeout" // pool saturation; carries RetryAfterMs
	ErrCodePoolClosed                 = "pool_closed"
	ErrCodeMaxSessionDurationExceeded = "max_session_duration_exceeded"
	ErrCodePayloadTooLarge            = "payload_too_large"
	ErrCodeUnsupportedProtocolVersion = "unsupported_protocol_version"
	ErrCodeIdleTimeout                = "idle_timeout"
	ErrCodePolicyViolation            = "policy_violation"
)

// WordInfo is a single recognized word with timing and confidence metadata.
type WordInfo struct {
	// Word is the recognized word text (BPE tokens joined).
	Word string `json:"word"`
	// Start is the word start time in seconds from the stream beginning.
	Start float64 `json:"start"`
	// End is the word end time in seconds from the stream beginning.
	End float64 `json:"end"`
	// Confidence is the mean softmax confidence over the word's tokens (0.0-1.0).
	Confidence float64 `json:"confidence"`
	// Speaker is the zero-based diarization speaker label; nil when
	// diarization is disabled.
	Speaker *uint32 `json:"speaker,omitempty"`
}

// Ready is sent by the server immediately after the WebSocket handshake.
type Ready struct {
	// Model is the identifier of the loaded model (e.g. "gigaam-v3-e2e-rnnt").
	Model string `json:"model"`
	// SampleRate is the audio sample rate in Hz the server expects by default.
	SampleRate uint32 `json:"sample_rate"`
	// Version is the server's WebSocket protocol version (e.g. "1.0").
	Version string `json:"version"`
	// SupportedRates lists the accepted input sample rates in Hz.
	SupportedRates []uint32 `json:"supported_rates,omitempty"`
	// Diarization reports whether speaker diarization is available.
	Diarization bool `json:"diarization,omitempty"`
	// MinProtocolVersion is the minimum protocol version the server accepts.
	MinProtocolVersion string `json:"min_protocol_version,omitempty"`
}

// Transcript is a partial (interim, may change) or final (utterance complete)
// transcript segment.
type Transcript struct {
	// Text is the recognized text for this segment.
	Text string `json:"text"`
	// Words carries word-level timing metadata.
	Words []WordInfo `json:"words,omitempty"`
	// IsFinal reports whether this segment is final.
	IsFinal bool `json:"is_final"`
	// Timestamp is the Unix time (seconds since epoch) when the segment was produced.
	Timestamp float64 `json:"timestamp"`
}

// ServerError is an error message sent by the server. It implements the error
// interface; use errors.As to extract it from Dial or callback paths.
type ServerError struct {
	// Message is a user-facing description (internal details are hidden by the server).
	Message string `json:"message"`
	// Code is a machine-readable error code (see the ErrCode* constants).
	Code string `json:"code"`
	// RetryAfterMs is the server-suggested delay before retrying, in
	// milliseconds. Present only for transient backpressure errors (e.g. pool
	// saturation); nil otherwise.
	RetryAfterMs *uint32 `json:"retry_after_ms,omitempty"`
}

// Error returns a human-readable description of the server error.
func (e *ServerError) Error() string {
	return "gigastt: server error " + e.Code + ": " + e.Message
}
