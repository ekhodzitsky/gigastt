import {
  PROTOCOL_VERSION,
  ServerError,
  parseServerMessage,
} from "./types.js";
import type { ReadyMessage, Transcript } from "./types.js";

/** Automatic-reconnect policy. Presence of this option enables reconnects. */
export interface ReconnectOptions {
  /** Initial backoff in milliseconds (default 250). */
  minDelayMs?: number;
  /** Maximum backoff in milliseconds (default 5000). */
  maxDelayMs?: number;
  /** Retry budget per outage; 0 retries forever (default 0). */
  maxAttempts?: number;
}

/** Terminal close notification delivered exactly once. */
export interface CloseInfo {
  /**
   * True when the session ended intentionally: `close()` was called, the
   * session was finalized via `stop()`, or the AbortSignal fired.
   */
  intentional: boolean;
  /** Underlying error for unintentional closes. */
  error?: unknown;
}

/** Event callbacks. Callbacks must not call `stop()`/`close()` re-entrantly. */
export interface ClientHandlers {
  /** Called after each successful handshake, including reconnects. */
  onReady?: (msg: ReadyMessage) => void;
  /** Called for every partial (interim) transcript. */
  onPartial?: (t: Transcript) => void;
  /** Called for every final transcript (utterance complete). */
  onFinal?: (t: Transcript) => void;
  /** Called for every server error message. */
  onError?: (err: ServerError) => void;
  /** Called exactly once when the client is permanently done. */
  onClose?: (info: CloseInfo) => void;
}

/** Minimal structural subset of the WebSocket API used by this SDK. */
export interface WebSocketLike {
  binaryType: string;
  readonly readyState: number;
  send(data: string | ArrayBufferLike | ArrayBufferView): void;
  close(code?: number, reason?: string): void;
  onopen: ((ev: unknown) => void) | null;
  onmessage: ((ev: { data: unknown }) => void) | null;
  onerror: ((ev: unknown) => void) | null;
  onclose:
    | ((ev: { code?: number; reason?: string; wasClean?: boolean }) => void)
    | null;
}

/** Factory for WebSocket instances (inject a custom transport). */
export type WebSocketFactory = (url: string) => WebSocketLike;

export interface ClientOptions {
  /**
   * Input audio sample rate in Hz; must be one of the server's
   * `supported_rates`. Default: server default (48000).
   */
  sampleRate?: number;
  /** Enable speaker diarization for the session (if the server supports it). */
  diarization?: boolean;
  /**
   * AbortSignal governing the whole client lifetime: aborting rejects a
   * pending `connect`, interrupts reconnect backoff, and closes the socket.
   */
  signal?: AbortSignal;
  /** Reconnect policy. Absent: a dropped connection ends the client. */
  reconnect?: ReconnectOptions;
  /** Event callbacks. */
  handlers?: ClientHandlers;
  /**
   * Custom WebSocket transport. Default: the global `WebSocket` (browsers,
   * Node 22+), falling back to the `ws` package when no global exists.
   */
  webSocketFactory?: WebSocketFactory;
  /** Handshake + ready-wait timeout in milliseconds (default 10000). */
  dialTimeoutMs?: number;
}

/** PCM audio accepted by {@link GigasttClient.sendPCM}. */
export type PcmData = ArrayBufferLike | ArrayBufferView;

const WS_OPEN = 1;

/**
 * Typed WebSocket client for one gigastt transcription session.
 *
 * Create with {@link GigasttClient.connect}, stream PCM16 audio with
 * {@link sendPCM}, finalize with {@link stop}, release with {@link close}.
 */
export class GigasttClient {
  private ws: WebSocketLike | null = null;
  private readyMessage: ReadyMessage | null = null;
  private closed = false;
  private stopSent = false;
  private closeFired = false;
  private abortHandler: (() => void) | null = null;
  private factoryPromise: Promise<WebSocketFactory> | null = null;

  private constructor(
    private readonly url: string,
    private readonly opts: ClientOptions,
  ) {}

  /**
   * Connects to `url` (pass {@link DEFAULT_URL} for a default local server),
   * sends the `configure` frame (protocol version pinning plus negotiated
   * options), and waits for the server's `ready` message.
   *
   * Throws a {@link ServerError} when the server rejects the session. With
   * `reconnect` set, transient failures are retried first (see
   * {@link ReconnectOptions}); the server's `retry_after_ms` hint is honored
   * exactly.
   */
  static async connect(
    url: string,
    options: ClientOptions = {},
  ): Promise<GigasttClient> {
    if (options.signal?.aborted === true) throw abortError();
    const client = new GigasttClient(url, options);
    client.registerAbortHandler();
    try {
      const ready = await client.dialWithRetry();
      client.readyMessage = ready;
      options.handlers?.onReady?.(ready);
      return client;
    } catch (err) {
      // The session never started; onClose is not fired for connect failures.
      client.unregisterAbortHandler();
      client.closed = true;
      throw err;
    }
  }

  /** The ready message from the current connection. */
  get ready(): ReadyMessage {
    if (this.readyMessage === null) {
      throw new Error("gigastt: not connected");
    }
    return this.readyMessage;
  }

  /** Whether the socket is currently connected. */
  get connected(): boolean {
    return this.ws !== null && this.ws.readyState === WS_OPEN;
  }

  /**
   * Sends one binary frame of raw PCM16 signed little-endian mono audio at
   * the negotiated sample rate. Frames may be up to the server's
   * `--ws-frame-max-bytes` (default 512 KiB).
   *
   * Throws while a reconnect is in flight and after `close()` — the SDK does
   * not buffer audio across reconnects; callers must retry or drop frames.
   */
  sendPCM(data: PcmData): void {
    if (this.closed) throw new Error("gigastt: client is closed");
    const ws = this.ws;
    if (ws === null || ws.readyState !== WS_OPEN) {
      throw new Error("gigastt: reconnecting, not connected");
    }
    ws.send(data);
  }

  /**
   * Asks the server to finalize the session: it flushes buffered audio,
   * emits a final transcript (delivered via `onFinal`), and ends the session.
   * The following server-side close is treated as a normal end — no reconnect
   * is attempted and `onClose` fires with `intentional: true`.
   */
  stop(): void {
    if (this.closed) throw new Error("gigastt: client is closed");
    const ws = this.ws;
    if (ws === null || ws.readyState !== WS_OPEN) {
      throw new Error("gigastt: reconnecting, not connected");
    }
    this.stopSent = true;
    ws.send(JSON.stringify({ type: "stop" }));
  }

  /**
   * Terminates the client: closes the socket and cancels any in-flight
   * reconnect backoff. Idempotent.
   */
  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.unregisterAbortHandler();
    const ws = this.ws;
    this.ws = null;
    if (ws !== null) {
      try {
        ws.close(1000, "client closing");
      } catch {
        // Tearing down anyway.
      }
    }
    this.fireClose({ intentional: true });
  }

  // ---------------------------------------------------------------------

  private registerAbortHandler(): void {
    if (this.opts.signal === undefined) return;
    this.abortHandler = () => this.close();
    if (this.opts.signal.aborted) {
      // Defer so callers observe a consistent async failure path.
      queueMicrotask(() => this.abortHandler?.());
      return;
    }
    this.opts.signal.addEventListener("abort", this.abortHandler, {
      once: true,
    });
  }

  private unregisterAbortHandler(): void {
    if (this.opts.signal !== undefined && this.abortHandler !== null) {
      this.opts.signal.removeEventListener("abort", this.abortHandler);
      this.abortHandler = null;
    }
  }

  /** Fires onClose exactly once. */
  private fireClose(info: CloseInfo): void {
    if (this.closeFired) return;
    this.closeFired = true;
    this.opts.handlers?.onClose?.(info);
  }

  private async resolveFactory(): Promise<WebSocketFactory> {
    if (this.opts.webSocketFactory !== undefined) {
      return this.opts.webSocketFactory;
    }
    if (this.factoryPromise === null) {
      this.factoryPromise = (async () => {
        if (typeof WebSocket !== "undefined") {
          return (url: string) =>
            new WebSocket(url) as unknown as WebSocketLike;
        }
        // Node without a global WebSocket: use the `ws` package.
        const mod = await import("ws");
        const WS = mod.WebSocket;
        return (url: string) => new WS(url) as unknown as WebSocketLike;
      })();
    }
    return this.factoryPromise;
  }

  /**
   * Dials until success, a non-transient error, the attempt budget is
   * exhausted, or the AbortSignal fires.
   */
  private async dialWithRetry(): Promise<ReadyMessage> {
    const rec = this.opts.reconnect;
    const minDelay = rec?.minDelayMs ?? 250;
    const maxDelay = rec?.maxDelayMs ?? 5000;
    const maxAttempts = rec?.maxAttempts ?? 0;
    let backoff = minDelay;
    let attempt = 0;
    for (;;) {
      try {
        return await this.dialOnce();
      } catch (err) {
        if (rec === undefined || !isTransient(err)) throw err;
        attempt++;
        if (maxAttempts > 0 && attempt > maxAttempts) {
          throw err instanceof Error
            ? err
            : new Error(`gigastt: giving up after ${attempt} attempts`);
        }
        let wait = backoff;
        if (err instanceof ServerError && err.retryAfterMs !== undefined) {
          // Honor the server's explicit retry hint exactly.
          wait = err.retryAfterMs;
        }
        await sleep(wait, this.opts.signal);
        backoff = Math.min(backoff * 2, maxDelay);
      }
    }
  }

  /**
   * Opens one WebSocket, sends the configure frame, and waits for `ready`.
   * A server error received before ready rejects with a {@link ServerError}.
   */
  private async dialOnce(): Promise<ReadyMessage> {
    this.throwIfAborted();
    const factory = await this.resolveFactory();
    const ws = factory(this.url);
    ws.binaryType = "arraybuffer";
    const dialTimeoutMs = this.opts.dialTimeoutMs ?? 10_000;

    return await new Promise<ReadyMessage>((resolve, reject) => {
      let settled = false;
      const timer = setTimeout(() => {
        fail(new Error("gigastt: timed out waiting for ready"));
      }, dialTimeoutMs);

      const onAbort = (): void => fail(abortError());

      const cleanup = (): void => {
        clearTimeout(timer);
        this.opts.signal?.removeEventListener("abort", onAbort);
        ws.onopen = null;
        ws.onmessage = null;
        ws.onerror = null;
        ws.onclose = null;
      };
      const fail = (err: unknown): void => {
        if (settled) return;
        settled = true;
        cleanup();
        try {
          ws.close();
        } catch {
          // Ignore.
        }
        reject(err);
      };

      this.opts.signal?.addEventListener("abort", onAbort, { once: true });

      ws.onopen = () => {
        ws.send(
          JSON.stringify({
            type: "configure",
            protocol_version: PROTOCOL_VERSION,
            ...(this.opts.sampleRate !== undefined
              ? { sample_rate: this.opts.sampleRate }
              : {}),
            ...(this.opts.diarization !== undefined
              ? { diarization: this.opts.diarization }
              : {}),
          }),
        );
      };
      ws.onmessage = (ev) => {
        const msg = parseServerMessage(normalizeData(ev.data));
        if (msg === null) return; // Not a server message: keep waiting.
        if (settled) return;
        if (msg.type === "ready") {
          settled = true;
          const { type: _type, ...ready } = msg;
          cleanup();
          if (this.closed) {
            // Aborted while the handshake was completing: do not leak the socket.
            try {
              ws.close();
            } catch {
              // Ignore.
            }
            reject(abortError());
            return;
          }
          this.ws = ws;
          this.attachSessionHandlers(ws);
          resolve(ready);
        } else if (msg.type === "error") {
          fail(new ServerError(msg));
        }
        // Other pre-ready message types: ignore (forward compatibility).
      };
      ws.onerror = () => {
        // A close event follows; failure is reported there.
      };
      ws.onclose = (ev) => {
        fail(
          new Error(
            `gigastt: connection closed before ready (code ${ev.code ?? "unknown"})`,
          ),
        );
      };
    });
  }

  /** Attaches the steady-state handlers to an established connection. */
  private attachSessionHandlers(ws: WebSocketLike): void {
    ws.onmessage = (ev) => {
      const msg = parseServerMessage(normalizeData(ev.data));
      if (msg === null) return;
      const handlers = this.opts.handlers;
      switch (msg.type) {
        case "ready":
          // A second ready within one connection is unexpected; ignore it.
          break;
        case "partial":
        case "final": {
          const { type, ...segment } = msg;
          if (type === "partial") {
            handlers?.onPartial?.(segment);
          } else {
            handlers?.onFinal?.(segment);
          }
          break;
        }
        case "error":
          handlers?.onError?.(new ServerError(msg));
          break;
      }
    };
    ws.onerror = () => {
      // A close event follows; handled there.
    };
    ws.onclose = (ev) => {
      this.handleSessionClose(ev);
    };
  }

  /** Decides whether a session-level close ends the client or reconnects. */
  private handleSessionClose(ev: {
    code?: number;
    reason?: string;
  }): void {
    this.ws = null;
    if (this.closed) return;
    if (this.stopSent) {
      // Session finalized via stop(): the server closes after the final
      // transcript. Normal end of a session.
      this.closed = true;
      this.unregisterAbortHandler();
      this.fireClose({ intentional: true });
      return;
    }
    if (this.opts.reconnect === undefined) {
      this.closed = true;
      this.unregisterAbortHandler();
      this.fireClose({
        intentional: false,
        error: new Error(
          `gigastt: connection closed by server (code ${ev.code ?? "unknown"})`,
        ),
      });
      return;
    }
    // Reconnect in the background; handlers observe onReady on success or
    // onClose on failure.
    void this.dialWithRetry()
      .then((ready) => {
        if (this.closed) return;
        this.readyMessage = ready;
        this.opts.handlers?.onReady?.(ready);
      })
      .catch((err: unknown) => {
        if (this.closed) return;
        this.closed = true;
        this.unregisterAbortHandler();
        this.fireClose({ intentional: isAbortError(err), error: err });
      });
  }

  private throwIfAborted(): void {
    if (this.opts.signal?.aborted === true) throw abortError();
  }
}

/** Converts any incoming frame payload to a string for JSON parsing. */
function normalizeData(data: unknown): unknown {
  if (typeof data === "string") return data;
  if (data instanceof ArrayBuffer) {
    return new TextDecoder().decode(data);
  }
  if (ArrayBuffer.isView(data)) {
    return new TextDecoder().decode(data);
  }
  return null;
}

/**
 * Network-level failures are transient; server errors are transient only
 * when the server attached a retry_after_ms hint (backpressure).
 */
function isTransient(err: unknown): boolean {
  if (isAbortError(err)) return false;
  if (err instanceof ServerError) return err.retryAfterMs !== undefined;
  return true;
}

function abortError(): Error {
  const err = new Error("gigastt: aborted");
  err.name = "AbortError";
  return err;
}

function isAbortError(err: unknown): boolean {
  return err instanceof Error && err.name === "AbortError";
}

/** Sleeps for `ms`, rejecting early if the signal aborts. */
function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted === true) {
      reject(abortError());
      return;
    }
    const timer = setTimeout(() => {
      cleanup();
      resolve();
    }, ms);
    const onAbort = (): void => {
      cleanup();
      reject(abortError());
    };
    const cleanup = (): void => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}
