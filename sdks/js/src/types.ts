/**
 * Wire protocol types for the gigastt WebSocket API, protocol version 1.0.
 *
 * Field names match the JSON wire format exactly (see docs/asyncapi.yaml in
 * the gigastt repository). Additive protocol changes are tolerated: unknown
 * fields are ignored on decode and unknown message types are dropped.
 */

/**
 * Protocol version this SDK speaks. It is pinned into the `configure` frame
 * so a version mismatch fails loudly with an `unsupported_protocol_version`
 * error instead of silently misbehaving.
 */
export const PROTOCOL_VERSION = "1.0";

/** Default server endpoint (loopback bind, default port). */
export const DEFAULT_URL = "ws://127.0.0.1:9876/v1/ws";

/** A single recognized word with timing and confidence metadata. */
export interface WordInfo {
  /** The recognized word text (BPE tokens joined). */
  word: string;
  /** Word start time in seconds from the stream beginning. */
  start: number;
  /** Word end time in seconds from the stream beginning. */
  end: number;
  /** Mean softmax confidence over the word's tokens (0.0–1.0). */
  confidence: number;
  /** Zero-based diarization speaker label; omitted when diarization is off. */
  speaker?: number;
}

/** Sent by the server immediately after the WebSocket handshake. */
export interface ReadyMessage {
  /** Identifier of the loaded model (e.g. "gigaam-v3-e2e-rnnt"). */
  model: string;
  /** Audio sample rate in Hz the server expects by default. */
  sample_rate: number;
  /** Server's WebSocket protocol version (e.g. "1.0"). */
  version: string;
  /** Accepted input sample rates in Hz. */
  supported_rates?: number[];
  /** Whether speaker diarization is available. */
  diarization?: boolean;
  /** Minimum protocol version the server accepts. */
  min_protocol_version?: string;
}

/** A partial (interim, may change) or final (utterance complete) segment. */
export interface Transcript {
  /** Recognized text for this segment. */
  text: string;
  /** Word-level timing metadata. */
  words?: WordInfo[];
  /** Whether this segment is final. */
  is_final: boolean;
  /** Unix time (seconds since epoch) when the segment was produced. */
  timestamp: number;
}

/** Machine-readable error codes sent by the server (docs/asyncapi.yaml). */
export type ServerErrorCode =
  | "inference_error"
  | "inference_panic"
  | "inference_timeout"
  | "configure_too_late"
  | "invalid_sample_rate"
  /** Pool saturation backpressure; the message carries `retry_after_ms`. */
  | "timeout"
  | "pool_closed"
  | "max_session_duration_exceeded"
  | "payload_too_large"
  | "unsupported_protocol_version"
  | "idle_timeout"
  | "policy_violation";

/** Error message payload as sent by the server. */
export interface ErrorMessage {
  /** User-facing description (internal details are hidden by the server). */
  message: string;
  /** Machine-readable error code. */
  code: ServerErrorCode | (string & {});
  /**
   * Server-suggested delay before retrying, in milliseconds. Present only for
   * transient backpressure errors (e.g. pool saturation).
   */
  retry_after_ms?: number;
}

/** Error raised from a server `error` message. */
export class ServerError extends Error {
  /** Machine-readable error code. */
  readonly code: string;
  /** Server-suggested retry delay in milliseconds, when present. */
  readonly retryAfterMs?: number;

  constructor(msg: ErrorMessage) {
    super(`gigastt: server error ${msg.code}: ${msg.message}`);
    this.name = "ServerError";
    this.code = msg.code;
    this.retryAfterMs = msg.retry_after_ms;
  }
}

/** Raw server message, discriminated by the `type` tag. */
export type ServerWireMessage =
  | ({ type: "ready" } & ReadyMessage)
  | ({ type: "partial" } & Transcript)
  | ({ type: "final" } & Transcript)
  | ({ type: "error" } & ErrorMessage);

/**
 * Parses one raw WebSocket text payload into a typed server message.
 * Returns null for non-JSON payloads, unknown message types (additive
 * protocol evolution), and malformed frames — callers must ignore them.
 */
export function parseServerMessage(data: unknown): ServerWireMessage | null {
  if (typeof data !== "string") return null;
  let msg: unknown;
  try {
    msg = JSON.parse(data);
  } catch {
    return null;
  }
  if (msg === null || typeof msg !== "object") return null;
  switch ((msg as { type?: unknown }).type) {
    case "ready":
    case "partial":
    case "final":
    case "error":
      return msg as ServerWireMessage;
    default:
      return null;
  }
}
