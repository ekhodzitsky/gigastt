import type { AddressInfo } from "node:net";
import { expect, it } from "vitest";
import { WebSocketServer } from "ws";
import { GigasttClient, NATIVE_V4_SUBPROTOCOL } from "../src/index.js";

async function listener() {
  const server = new WebSocketServer({ host: "127.0.0.1", port: 0 });
  await new Promise<void>((resolve, reject) => {
    server.once("listening", resolve);
    server.once("error", reject);
  });
  return { server, url: `ws://127.0.0.1:${(server.address() as AddressInfo).port}/v1/ws`,
    close: () => { for (const ws of server.clients) ws.terminate(); server.close(); } };
}
const ready = { type: "ready", model: "gigaam-v4-rnnt-e2e-fp32-streaming", version: "1.4",
  sample_rate: 16000, supported_rates: [16000], native_v4: { model_family: "gigaam_v4", model_head: "rnnt",
    model_precision: "fp32", future_context_ms: 640, transcript_style: "e2e", endpoint_mode: "manual",
    max_packet_bytes: 32000, supported_future_context_ms: [160, 320, 640] } };

it("requires native v4 subprotocol and exact Ready, preserving snapshot and delta", async () => {
  const ws = await listener();
  let seen: unknown;
  ws.server.on("connection", (conn, req) => {
    const query = new URL(req.url ?? "", ws.url).searchParams;
    expect(conn.protocol).toBe(NATIVE_V4_SUBPROTOCOL);
    expect(query.get("transcript_style")).toBe("e2e");
    expect(query.get("future_context_ms")).toBe("640");
    conn.once("message", (data) => {
      seen = JSON.parse(String(data));
      conn.send(JSON.stringify(ready));
      conn.once("message", () => {
        conn.send(JSON.stringify({ type: "final", text: "да да", is_final: true,
          timestamp: 1, native_delta: { packet_one_based: 2, samples: 2, encoded_frames: 0,
            text: " да", words: [], tokens: [] } }));
        conn.close(1000);
      });
    });
  });
  try {
    let accept!: (event: { text: string; native_delta?: { text: string } }) => void;
    const final = new Promise<{ text: string; native_delta?: { text: string } }>((resolve) => { accept = resolve; });
    const c = await GigasttClient.connect(ws.url, { nativeV4: { transcriptStyle: "e2e", futureContextMs: 640 }, handlers: { onFinal: accept } });
    expect(seen).toEqual({ type: "configure", protocol_version: "1.4", sample_rate: 16000,
      transcript_style: "e2e", future_context_ms: 640 });
    expect(() => c.sendPCM(new Uint8Array(3))).toThrow("PCM16");
    c.stop();
    const f = await final;
    expect(f.text).toBe("да да");
    expect(f.native_delta?.text).toBe(" да");
  } finally { ws.close(); }
});

it("rejects an incorrect native Ready even when the server echoes the subprotocol", async () => {
  const ws = await listener();
  ws.server.on("connection", (conn) => conn.once("message", () => conn.send(JSON.stringify({ ...ready, native_v4: { ...ready.native_v4, future_context_ms: 160 } }))));
  try {
    await expect(GigasttClient.connect(ws.url, { nativeV4: { transcriptStyle: "e2e", futureContextMs: 640 } })).rejects.toMatchObject({ code: "native_v4_mismatch" });
  } finally { ws.close(); }
});
