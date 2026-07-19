package gigastt

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

// ErrClosed is returned by SendPCM and Stop after the client was closed.
var ErrClosed = errors.New("gigastt: client is closed")

// ErrReconnecting is returned by SendPCM and Stop while the client is between
// connections during an automatic reconnect. The SDK does not buffer audio
// across reconnects; callers must retry or drop the frame explicitly.
var ErrReconnecting = errors.New("gigastt: reconnecting, not connected")

// Handlers holds the event callbacks invoked by the client.
//
// Callbacks run on the client's internal read goroutine. They must return
// quickly (hand long work off to your own goroutine) and must not call
// Stop/Close synchronously from within a callback — use a goroutine instead.
type Handlers struct {
	// OnReady is called after each successful handshake, including reconnects.
	OnReady func(Ready)
	// OnPartial is called for every partial (interim) transcript.
	OnPartial func(Transcript)
	// OnFinal is called for every final transcript (utterance complete).
	OnFinal func(Transcript)
	// OnError is called for every server error message.
	OnError func(*ServerError)
	// OnClose is called exactly once when the client is permanently done:
	// after Close, after a session ended by Stop, or when a connection
	// failure could not be recovered. err is nil for an intentional shutdown.
	OnClose func(err error)
}

type config struct {
	sampleRate  uint32
	diarization *bool
	dialTimeout time.Duration

	reconnect   bool
	minBackoff  time.Duration
	maxBackoff  time.Duration
	maxAttempts int // 0 = unlimited

	handlers Handlers
}

func defaultConfig() config {
	return config{
		dialTimeout: 10 * time.Second,
	}
}

// Option configures a Client.
type Option func(*config)

// WithSampleRate negotiates the input audio sample rate in Hz (one of the
// server's supported_rates, e.g. 16000). When unset, the server default
// (48000) applies.
func WithSampleRate(hz uint32) Option {
	return func(c *config) { c.sampleRate = hz }
}

// WithDiarization enables speaker diarization for the session (if the server
// was built and started with diarization support).
func WithDiarization(enabled bool) Option {
	return func(c *config) { c.diarization = &enabled }
}

// WithDialTimeout sets the timeout for the WebSocket handshake and for
// waiting for the server's ready message. Default: 10s.
func WithDialTimeout(d time.Duration) Option {
	return func(c *config) { c.dialTimeout = d }
}

// WithReconnect enables automatic reconnect. Transient failures (network
// errors, dropped connections) are retried with exponential backoff from
// minBackoff, doubling up to maxBackoff. When the server asks for a specific
// delay via retry_after_ms (pool saturation backpressure), that delay is
// honored exactly instead of the exponential value. maxAttempts bounds the
// number of retries per outage; 0 retries until the context is cancelled or
// Close is called.
func WithReconnect(minBackoff, maxBackoff time.Duration, maxAttempts int) Option {
	return func(c *config) {
		c.reconnect = true
		c.minBackoff = minBackoff
		c.maxBackoff = maxBackoff
		c.maxAttempts = maxAttempts
	}
}

// WithHandlers registers the event callbacks.
func WithHandlers(h Handlers) Option {
	return func(c *config) { c.handlers = h }
}

// Client is a typed WebSocket client for one gigastt transcription session.
//
// A Client is safe for concurrent use: SendPCM, Stop and Close may be called
// from any goroutine. All writes to the socket are serialized through a
// single internal writer path.
type Client struct {
	url string
	cfg config

	ctx    context.Context
	cancel context.CancelFunc

	// stateMu guards conn, ready, stopSent, closed.
	stateMu sync.Mutex
	conn    *websocket.Conn
	ready   Ready
	// stopSent marks a session that was finalized via Stop: the following
	// server-side close is a normal end, not a reconnect trigger.
	stopSent bool
	closed   bool

	// writeMu serializes all frame writes (single writer).
	writeMu sync.Mutex

	// onCloseOnce guarantees OnClose fires exactly once.
	onCloseOnce sync.Once
	// done is closed when the client is permanently done.
	done chan struct{}
}

// Dial connects to url (pass DefaultURL for a default local server), performs
// the handshake, sends the configure message (protocol version pinning plus
// any negotiated options), and waits for the server's ready message.
//
// The passed context governs the whole client lifetime: cancelling it shuts
// the client down, including any in-flight reconnect backoff.
//
// On failure Dial returns an error; a *ServerError (check with errors.As) is
// returned when the server rejected the session. With WithReconnect enabled,
// transient failures are retried before Dial gives up.
func Dial(ctx context.Context, url string, opts ...Option) (*Client, error) {
	cfg := defaultConfig()
	for _, o := range opts {
		o(&cfg)
	}
	ctx, cancel := context.WithCancel(ctx)
	c := &Client{
		url:    url,
		cfg:    cfg,
		ctx:    ctx,
		cancel: cancel,
		done:   make(chan struct{}),
	}
	conn, ready, err := c.connectWithRetry(ctx)
	if err != nil {
		cancel()
		close(c.done)
		return nil, err
	}
	c.conn = conn
	c.ready = ready
	go c.readLoop(conn)
	if h := cfg.handlers.OnReady; h != nil {
		h(ready)
	}
	return c, nil
}

// Ready returns the ready message from the current connection.
func (c *Client) Ready() Ready {
	c.stateMu.Lock()
	defer c.stateMu.Unlock()
	return c.ready
}

// Done returns a channel that is closed when the client is permanently done
// (after Close, after a Stop-finalized session ends, or after an
// unrecoverable failure).
func (c *Client) Done() <-chan struct{} {
	return c.done
}

// SendPCM sends one binary frame of raw PCM16 signed little-endian mono audio
// at the negotiated sample rate. Frames may be up to the server's
// --ws-frame-max-bytes (default 512 KiB). While a reconnect is in flight,
// SendPCM fails fast with ErrReconnecting; after Close, with ErrClosed.
func (c *Client) SendPCM(pcm []byte) error {
	return c.write(websocket.BinaryMessage, pcm)
}

// Stop asks the server to finalize the session. The server flushes any
// buffered audio, emits a final transcript (delivered via OnFinal), and ends
// the session; OnClose then fires with a nil error. The connection is not
// reused after Stop; call Close to release resources.
func (c *Client) Stop() error {
	c.stateMu.Lock()
	if c.closed {
		c.stateMu.Unlock()
		return ErrClosed
	}
	c.stopSent = true
	c.stateMu.Unlock()
	return c.write(websocket.TextMessage, []byte(`{"type":"stop"}`))
}

// Close terminates the client: it cancels the internal context, sends a
// WebSocket close frame on a best-effort basis, and closes the connection.
// It is idempotent and safe to call multiple times.
func (c *Client) Close() error {
	c.stateMu.Lock()
	if c.closed {
		c.stateMu.Unlock()
		return nil
	}
	c.closed = true
	conn := c.conn
	c.stateMu.Unlock()

	c.cancel()
	if conn != nil {
		// Best-effort close handshake; ignore errors — we are tearing down anyway.
		_ = c.writeFrame(conn, websocket.CloseMessage,
			websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
		_ = conn.Close()
	}
	c.finish(nil)
	return nil
}

// write sends a frame on the current connection through the single writer path.
func (c *Client) write(messageType int, data []byte) error {
	c.stateMu.Lock()
	if c.closed {
		c.stateMu.Unlock()
		return ErrClosed
	}
	conn := c.conn
	c.stateMu.Unlock()
	if conn == nil {
		return ErrReconnecting
	}
	return c.writeFrame(conn, messageType, data)
}

// writeFrame is the single write path to the socket. gorilla/websocket
// requires one concurrent writer; writeMu serializes all callers.
func (c *Client) writeFrame(conn *websocket.Conn, messageType int, data []byte) error {
	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	return conn.WriteMessage(messageType, data)
}

// connectWithRetry dials until success, a non-transient error, the attempt
// budget is exhausted, or ctx is cancelled.
func (c *Client) connectWithRetry(ctx context.Context) (*websocket.Conn, Ready, error) {
	backoff := c.cfg.minBackoff
	for attempt := 0; ; {
		conn, ready, err := c.dialAndHandshake(ctx)
		if err == nil {
			return conn, ready, nil
		}
		if !c.cfg.reconnect || !isTransient(err) {
			return nil, Ready{}, err
		}
		attempt++
		if c.cfg.maxAttempts > 0 && attempt > c.cfg.maxAttempts {
			return nil, Ready{}, fmt.Errorf("gigastt: giving up after %d attempts: %w", attempt, err)
		}
		wait := backoff
		var se *ServerError
		if errors.As(err, &se) && se.RetryAfterMs != nil {
			// Honor the server's explicit retry hint exactly.
			wait = time.Duration(*se.RetryAfterMs) * time.Millisecond
		}
		select {
		case <-ctx.Done():
			return nil, Ready{}, ctx.Err()
		case <-time.After(wait):
		}
		backoff *= 2
		if c.cfg.maxBackoff > 0 && backoff > c.cfg.maxBackoff {
			backoff = c.cfg.maxBackoff
		}
	}
}

// isTransient reports whether a dial/handshake failure is worth retrying.
// Network-level failures are transient; server errors are transient only when
// the server attached a retry_after_ms hint (backpressure).
func isTransient(err error) bool {
	var se *ServerError
	if errors.As(err, &se) {
		return se.RetryAfterMs != nil
	}
	return true
}

// dialAndHandshake establishes one WebSocket connection, sends the configure
// message, and waits for the server's ready message. A server error received
// before ready (e.g. pool saturation) is returned as a *ServerError.
func (c *Client) dialAndHandshake(ctx context.Context) (*websocket.Conn, Ready, error) {
	dialer := websocket.Dialer{HandshakeTimeout: c.cfg.dialTimeout}
	conn, _, err := dialer.DialContext(ctx, c.url, nil)
	if err != nil {
		return nil, Ready{}, fmt.Errorf("gigastt: dial %s: %w", c.url, err)
	}
	fail := func(err error) (*websocket.Conn, Ready, error) {
		_ = conn.Close()
		return nil, Ready{}, err
	}
	if err := c.writeFrame(conn, websocket.TextMessage, c.configureMessage()); err != nil {
		return fail(fmt.Errorf("gigastt: send configure: %w", err))
	}
	if err := conn.SetReadDeadline(time.Now().Add(c.cfg.dialTimeout)); err != nil {
		return fail(fmt.Errorf("gigastt: set read deadline: %w", err))
	}
	for {
		_, raw, err := conn.ReadMessage()
		if err != nil {
			return fail(fmt.Errorf("gigastt: waiting for ready: %w", err))
		}
		var env envelope
		if err := json.Unmarshal(raw, &env); err != nil {
			continue // not JSON or no type tag: ignore, keep waiting
		}
		switch env.Type {
		case "ready":
			var r Ready
			if err := json.Unmarshal(raw, &r); err != nil {
				return fail(fmt.Errorf("gigastt: decode ready: %w", err))
			}
			// Clear the handshake deadline; the session has no read deadline.
			if err := conn.SetReadDeadline(time.Time{}); err != nil {
				return fail(fmt.Errorf("gigastt: clear read deadline: %w", err))
			}
			return conn, r, nil
		case "error":
			var se ServerError
			if err := json.Unmarshal(raw, &se); err != nil {
				return fail(fmt.Errorf("gigastt: decode error message: %w", err))
			}
			return fail(&se)
		default:
			// Unexpected pre-ready message type: ignore (forward compatibility).
		}
	}
}

// configureMessage builds the configure frame. It is always sent: it pins the
// protocol version so a version mismatch fails loudly instead of silently.
func (c *Client) configureMessage() []byte {
	msg := map[string]any{
		"type":             "configure",
		"protocol_version": ProtocolVersion,
	}
	if c.cfg.sampleRate != 0 {
		msg["sample_rate"] = c.cfg.sampleRate
	}
	if c.cfg.diarization != nil {
		msg["diarization"] = *c.cfg.diarization
	}
	// The map is fully controlled above; marshaling cannot fail.
	data, _ := json.Marshal(msg)
	return data
}

// envelope is the type tag shared by all server messages.
type envelope struct {
	Type string `json:"type"`
}

// readLoop dispatches server messages until the connection fails or closes.
func (c *Client) readLoop(conn *websocket.Conn) {
	for {
		_, raw, err := conn.ReadMessage()
		if err != nil {
			c.handleDisconnect(conn, err)
			return
		}
		c.dispatch(raw)
	}
}

// dispatch parses one server message and invokes the matching handler.
// Unknown message types and unparseable frames are ignored: additive protocol
// evolution must never break the session.
func (c *Client) dispatch(raw []byte) {
	var env envelope
	if err := json.Unmarshal(raw, &env); err != nil {
		return
	}
	switch env.Type {
	case "ready":
		// A second ready within one connection is unexpected; ignore it.
	case "partial":
		var t Transcript
		if err := json.Unmarshal(raw, &t); err == nil {
			if h := c.cfg.handlers.OnPartial; h != nil {
				h(t)
			}
		}
	case "final":
		var t Transcript
		if err := json.Unmarshal(raw, &t); err == nil {
			if h := c.cfg.handlers.OnFinal; h != nil {
				h(t)
			}
		}
	case "error":
		var se ServerError
		if err := json.Unmarshal(raw, &se); err == nil {
			if h := c.cfg.handlers.OnError; h != nil {
				h(&se)
			}
		}
	default:
		// Unknown type: additive protocol change, ignore.
	}
}

// handleDisconnect decides whether a read failure ends the client or triggers
// a reconnect. conn is the connection whose read loop failed.
func (c *Client) handleDisconnect(conn *websocket.Conn, readErr error) {
	c.stateMu.Lock()
	if c.closed {
		c.stateMu.Unlock()
		return
	}
	if c.conn == conn {
		c.conn = nil
	}
	intentional := c.stopSent
	reconnect := c.cfg.reconnect && !c.stopSent
	c.stateMu.Unlock()

	if intentional {
		// Session finalized via Stop: the server closes after the final
		// transcript. This is the normal end of a session.
		c.finish(nil)
		return
	}
	if !reconnect {
		c.finish(normalizeCloseError(readErr))
		return
	}

	newConn, ready, err := c.connectWithRetry(c.ctx)
	if err != nil {
		c.finish(err)
		return
	}
	c.stateMu.Lock()
	if c.closed {
		c.stateMu.Unlock()
		_ = newConn.Close()
		return
	}
	c.conn = newConn
	c.ready = ready
	c.stateMu.Unlock()
	if h := c.cfg.handlers.OnReady; h != nil {
		h(ready)
	}
	go c.readLoop(newConn)
}

// finish terminates the client permanently and fires OnClose exactly once.
func (c *Client) finish(err error) {
	c.stateMu.Lock()
	alreadyClosed := c.closed
	c.closed = true
	conn := c.conn
	c.conn = nil
	c.stateMu.Unlock()

	c.cancel()
	if conn != nil && !alreadyClosed {
		_ = conn.Close()
	}
	c.onCloseOnce.Do(func() {
		if h := c.cfg.handlers.OnClose; h != nil {
			h(err)
		}
		close(c.done)
	})
}

// normalizeCloseError converts WebSocket close frames into a compact error.
func normalizeCloseError(err error) error {
	var closeErr *websocket.CloseError
	if errors.As(err, &closeErr) {
		return fmt.Errorf("gigastt: connection closed by server (code %d: %s)", closeErr.Code, closeErr.Text)
	}
	return fmt.Errorf("gigastt: connection lost: %w", err)
}
