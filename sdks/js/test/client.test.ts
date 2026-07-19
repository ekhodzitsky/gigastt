import type { AddressInfo } from "node:net";
import { describe, expect, it } from "vitest";
import { WebSocketServer, WebSocket as WsSocket } from "ws";
import {
  GigasttClient,
  PROTOCOL_VERSION,
  ServerError,
} from "../src/index.js";
import type { Transcript, WebSocketLike } from "../src/index.js";

// ---------------------------------------------------------------------------
// Mock gigastt server: speaks the wire protocol per a per-connection script.
// ---------------------------------------------------------------------------

type Script = (ws: WsSocket, n: number) => void;

class MockServer {
  readonly url: string;
  connections = 0;

  private constructor(private readonly wss: WebSocketServer, port: number) {
    this.url = `ws://127.0.0.1:${port}/v1/ws`;
  }

  static async create(script: Script): Promise<MockServer> {
    const wss = new WebSocketServer({ host: "127.0.0.1", port: 0 });
    await new Promise<void>((resolve, reject) => {
      wss.once("listening", () => resolve());
      wss.once("error", reject);
    });
    const server = new MockServer(wss, (wss.address() as AddressInfo).port);
    wss.on("connection", (ws) => {
      server.connections += 1;
      script(ws, server.connections);
    });
    return server;
  }

  async close(): Promise<void> {
    for (const client of this.wss.clients) client.terminate();
    await new Promise<void>((resolve) => this.wss.close(() => resolve()));
  }
}

function sendJson(ws: WsSocket, value: unknown): void {
  ws.send(JSON.stringify(value));
}

/** Narrows a ws RawData payload to a Buffer. */
function rawToBuffer(data: unknown): Buffer {
  if (Buffer.isBuffer(data)) return data;
  if (Array.isArray(data)) return Buffer.concat(data);
  if (data instanceof ArrayBuffer) return Buffer.from(data);
  throw new Error("unexpected frame payload type");
}

/** Resolves with the next incoming text frame parsed as JSON. */
function nextJson(ws: WsSocket): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    ws.once("message", (data, isBinary) => {
      if (isBinary) {
        reject(new Error("expected a text frame"));
        return;
      }
      resolve(JSON.parse(rawToBuffer(data).toString()) as Record<string, unknown>);
    });
    ws.once("error", reject);
  });
}

/** Resolves with the next incoming binary frame. */
function nextBinary(ws: WsSocket): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    ws.once("message", (data, isBinary) => {
      if (!isBinary) {
        reject(new Error("expected a binary frame"));
        return;
      }
      resolve(rawToBuffer(data));
    });
    ws.once("error", reject);
  });
}

/** Reads the configure frame the client must send first. */
async function expectConfigure(
  ws: WsSocket,
): Promise<Record<string, unknown>> {
  const msg = await nextJson(ws);
  expect(msg.type).toBe("configure");
  return msg;
}

const READY_PAYLOAD = {
  type: "ready",
  model: "gigaam-v3-e2e-rnnt",
  sample_rate: 48000,
  version: "1.0",
  supported_rates: [8000, 16000, 24000, 44100, 48000],
  diarization: true,
  min_protocol_version: "1.0",
} as const;

function deferred<T>(): {
  promise: Promise<T>;
  resolve: (v: T) => void;
  reject: (e: unknown) => void;
} {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

/** Rejects if `promise` does not settle within `ms`. */
function withTimeout<T>(promise: Promise<T>, ms = 5000): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) =>
      setTimeout(() => reject(new Error("timed out waiting for event")), ms),
    ),
  ]);
}

// ---------------------------------------------------------------------------

describe("GigasttClient", () => {
  it("sends configure with protocol pinning and receives ready", async () => {
    const configureSeen = deferred<Record<string, unknown>>();
    const server = await MockServer.create((ws) => {
      void expectConfigure(ws).then((msg) => {
        configureSeen.resolve(msg);
        sendJson(ws, READY_PAYLOAD);
      });
    });
    try {
      const client = await GigasttClient.connect(server.url, {
        sampleRate: 16000,
        diarization: true,
      });

      const configure = await withTimeout(configureSeen.promise);
      expect(configure.protocol_version).toBe(PROTOCOL_VERSION);
      expect(configure.sample_rate).toBe(16000);
      expect(configure.diarization).toBe(true);

      expect(client.ready.model).toBe("gigaam-v3-e2e-rnnt");
      expect(client.ready.sample_rate).toBe(48000);
      expect(client.ready.version).toBe("1.0");
      expect(client.ready.supported_rates).toEqual([
        8000, 16000, 24000, 44100, 48000,
      ]);
      expect(client.ready.diarization).toBe(true);
      expect(client.ready.min_protocol_version).toBe("1.0");
      expect(client.connected).toBe(true);

      client.close();
    } finally {
      await server.close();
    }
  });

  it("dispatches partial and final transcripts with words", async () => {
    const server = await MockServer.create((ws) => {
      void expectConfigure(ws).then(() => {
        sendJson(ws, READY_PAYLOAD);
        sendJson(ws, {
          type: "partial",
          text: "привет",
          timestamp: 1712700000.5,
          is_final: false,
          words: [
            { word: "привет", start: 0.1, end: 0.6, confidence: 0.95, speaker: 2 },
          ],
        });
        sendJson(ws, {
          type: "final",
          text: "привет как дела",
          timestamp: 1712700001.0,
          is_final: true,
          words: [
            { word: "привет", start: 0.1, end: 0.6, confidence: 0.95 },
            { word: "как", start: 0.7, end: 0.9, confidence: 0.9 },
            { word: "дела", start: 1.0, end: 1.4, confidence: 0.88 },
          ],
        });
      });
    });
    try {
      const partialSeen = deferred<Transcript>();
      const finalSeen = deferred<Transcript>();
      const client = await GigasttClient.connect(server.url, {
        handlers: {
          onPartial: (t) => partialSeen.resolve(t),
          onFinal: (t) => finalSeen.resolve(t),
        },
      });

      const partial = await withTimeout(partialSeen.promise);
      expect(partial.text).toBe("привет");
      expect(partial.is_final).toBe(false);
      expect(partial.timestamp).toBe(1712700000.5);
      expect(partial.words).toHaveLength(1);
      expect(partial.words?.[0]).toEqual({
        word: "привет",
        start: 0.1,
        end: 0.6,
        confidence: 0.95,
        speaker: 2,
      });

      const final = await withTimeout(finalSeen.promise);
      expect(final.text).toBe("привет как дела");
      expect(final.is_final).toBe(true);
      expect(final.words).toHaveLength(3);
      // speaker is omitted when diarization is disabled.
      expect(final.words?.[1]?.speaker).toBeUndefined();

      client.close();
    } finally {
      await server.close();
    }
  });

  it("dispatches server errors with code and no retry hint", async () => {
    const server = await MockServer.create((ws) => {
      void expectConfigure(ws).then(() => {
        sendJson(ws, READY_PAYLOAD);
        sendJson(ws, {
          type: "error",
          message: "Inference failed. Please check audio format.",
          code: "inference_error",
        });
      });
    });
    try {
      const errorSeen = deferred<ServerError>();
      const client = await GigasttClient.connect(server.url, {
        handlers: { onError: (e) => errorSeen.resolve(e) },
      });

      const err = await withTimeout(errorSeen.promise);
      expect(err).toBeInstanceOf(ServerError);
      expect(err.code).toBe("inference_error");
      expect(err.retryAfterMs).toBeUndefined();
      expect(err.message).toContain("inference_error");

      client.close();
    } finally {
      await server.close();
    }
  });

  it("honors retry_after_ms exactly when reconnecting", async () => {
    const server = await MockServer.create((ws, n) => {
      void expectConfigure(ws).then(() => {
        if (n === 1) {
          // Pool saturation: reject with an explicit retry hint.
          sendJson(ws, {
            type: "error",
            message: "Server busy, try again later",
            code: "timeout",
            retry_after_ms: 250,
          });
          ws.close();
          return;
        }
        sendJson(ws, READY_PAYLOAD);
      });
    });
    try {
      const start = Date.now();
      const client = await GigasttClient.connect(server.url, {
        reconnect: { minDelayMs: 5, maxDelayMs: 100, maxAttempts: 5 },
      });
      const elapsed = Date.now() - start;

      expect(server.connections).toBe(2);
      // The 250ms server hint must win over the 5ms exponential backoff.
      expect(elapsed).toBeGreaterThanOrEqual(200);

      client.close();
    } finally {
      await server.close();
    }
  });

  it("reconnects after an abnormal drop and re-fires onReady", async () => {
    const server = await MockServer.create((ws, n) => {
      void expectConfigure(ws).then(() => {
        sendJson(ws, READY_PAYLOAD);
        if (n === 1) {
          // Wait for one audio frame (proving the client processed ready),
          // then drop the connection abnormally.
          void nextBinary(ws).then(() => ws.terminate());
        }
      });
    });
    try {
      const readyEvents: string[] = [];
      const secondReady = deferred<void>();
      const client = await GigasttClient.connect(server.url, {
        reconnect: { minDelayMs: 10, maxDelayMs: 50, maxAttempts: 5 },
        handlers: {
          onReady: (r) => {
            readyEvents.push(r.model);
            if (readyEvents.length === 2) secondReady.resolve();
          },
        },
      });

      client.sendPCM(new Uint8Array([0, 0]));
      await withTimeout(secondReady.promise);
      expect(server.connections).toBe(2);
      expect(readyEvents).toHaveLength(2);

      client.close();
    } finally {
      await server.close();
    }
  });

  it("treats the close after stop() as a normal end (no reconnect)", async () => {
    const server = await MockServer.create((ws) => {
      void expectConfigure(ws).then(async () => {
        sendJson(ws, READY_PAYLOAD);
        const stop = await nextJson(ws);
        expect(stop.type).toBe("stop");
        sendJson(ws, {
          type: "final",
          text: "готово",
          timestamp: 1712700002.0,
          is_final: true,
          words: [],
        });
        // The server ends the session after the final transcript.
        ws.close(1000);
      });
    });
    try {
      const events: string[] = [];
      const closed = deferred<{ intentional: boolean }>();
      const client = await GigasttClient.connect(server.url, {
        reconnect: { minDelayMs: 10, maxDelayMs: 50, maxAttempts: 5 },
        handlers: {
          onFinal: () => events.push("final"),
          onClose: (info) => {
            events.push("close");
            closed.resolve(info);
          },
        },
      });

      client.stop();
      const info = await withTimeout(closed.promise);
      expect(events).toEqual(["final", "close"]);
      expect(info.intentional).toBe(true);
      expect(server.connections).toBe(1);
    } finally {
      await server.close();
    }
  });

  it("does not retry fatal server errors", async () => {
    const server = await MockServer.create((ws) => {
      void expectConfigure(ws).then(() => {
        sendJson(ws, {
          type: "error",
          message: "Unsupported protocol version",
          code: "unsupported_protocol_version",
        });
        ws.close();
      });
    });
    try {
      await expect(
        GigasttClient.connect(server.url, {
          reconnect: { minDelayMs: 5, maxDelayMs: 50, maxAttempts: 5 },
        }),
      ).rejects.toSatisfy(
        (err) =>
          err instanceof ServerError &&
          err.code === "unsupported_protocol_version",
      );
      expect(server.connections).toBe(1);
    } finally {
      await server.close();
    }
  });

  it("ignores unknown fields and unknown message types", async () => {
    const server = await MockServer.create((ws) => {
      void expectConfigure(ws).then(() => {
        sendJson(ws, { ...READY_PAYLOAD, future_field: { nested: true } });
        sendJson(ws, { type: "brand_new_message", x: 1 });
        sendJson(ws, {
          type: "partial",
          text: "ok",
          timestamp: 1.0,
          is_final: false,
          confidence: 0.5, // hypothetical additive field
        });
      });
    });
    try {
      const partialSeen = deferred<Transcript>();
      const client = await GigasttClient.connect(server.url, {
        handlers: { onPartial: (t) => partialSeen.resolve(t) },
      });

      const partial = await withTimeout(partialSeen.promise);
      expect(partial.text).toBe("ok");

      client.close();
    } finally {
      await server.close();
    }
  });

  it("AbortSignal cancels connect during reconnect backoff", async () => {
    const server = await MockServer.create((ws) => {
      void expectConfigure(ws).then(() => {
        sendJson(ws, {
          type: "error",
          message: "Server busy, try again later",
          code: "timeout",
          retry_after_ms: 60_000,
        });
        ws.close();
      });
    });
    try {
      const controller = new AbortController();
      setTimeout(() => controller.abort(), 100);
      const start = Date.now();
      await expect(
        GigasttClient.connect(server.url, {
          signal: controller.signal,
          reconnect: { minDelayMs: 5, maxDelayMs: 100 },
        }),
      ).rejects.toSatisfy(
        (err) => err instanceof Error && err.name === "AbortError",
      );
      expect(Date.now() - start).toBeLessThan(10_000);
      expect(server.connections).toBe(1);
    } finally {
      await server.close();
    }
  });

  it("sendPCM delivers binary frames unchanged", async () => {
    const frames: Buffer[] = [];
    const gotTwo = deferred<void>();
    const server = await MockServer.create((ws) => {
      void expectConfigure(ws).then(() => {
        sendJson(ws, READY_PAYLOAD);
        // Persistent collector: back-to-back frames can arrive in one
        // synchronous socket drain, faster than promise re-registration.
        ws.on("message", (data, isBinary) => {
          if (!isBinary) return;
          frames.push(rawToBuffer(data));
          if (frames.length === 2) gotTwo.resolve();
        });
      });
    });
    try {
      const client = await GigasttClient.connect(server.url);

      client.sendPCM(new Uint8Array([1, 2, 3]));
      client.sendPCM(new Int16Array([1000, -1000]).buffer);

      await withTimeout(gotTwo.promise);
      expect([...frames[0]!]).toEqual([1, 2, 3]);
      expect(frames[1]).toHaveLength(4);
      expect(frames[1]!.readInt16LE(0)).toBe(1000);
      expect(frames[1]!.readInt16LE(2)).toBe(-1000);

      client.close();
    } finally {
      await server.close();
    }
  });

  it("throws when sending after close; close is idempotent", async () => {
    const server = await MockServer.create((ws) => {
      void expectConfigure(ws).then(() => sendJson(ws, READY_PAYLOAD));
    });
    try {
      const client = await GigasttClient.connect(server.url);
      client.close();
      expect(() => client.sendPCM(new Uint8Array([1]))).toThrow(/closed/);
      expect(() => client.stop()).toThrow(/closed/);
      expect(() => client.close()).not.toThrow();
      expect(client.connected).toBe(false);
    } finally {
      await server.close();
    }
  });

  it("gives up after maxAttempts reconnects", async () => {
    const server = await MockServer.create((ws) => {
      void expectConfigure(ws).then(() => {
        sendJson(ws, {
          type: "error",
          message: "Server busy, try again later",
          code: "timeout",
          retry_after_ms: 20,
        });
        ws.close();
      });
    });
    try {
      await expect(
        GigasttClient.connect(server.url, {
          reconnect: { minDelayMs: 5, maxDelayMs: 50, maxAttempts: 2 },
        }),
      ).rejects.toBeInstanceOf(ServerError);
      // 1 initial attempt + 2 retries.
      expect(server.connections).toBe(3);
    } finally {
      await server.close();
    }
  });

  it("works over an injected ws-package transport", async () => {
    const server = await MockServer.create((ws) => {
      void expectConfigure(ws).then(() => sendJson(ws, READY_PAYLOAD));
    });
    try {
      const client = await GigasttClient.connect(server.url, {
        webSocketFactory: (url) =>
          new WsSocket(url) as unknown as WebSocketLike,
      });
      expect(client.ready.model).toBe("gigaam-v3-e2e-rnnt");
      client.close();
    } finally {
      await server.close();
    }
  });
});
