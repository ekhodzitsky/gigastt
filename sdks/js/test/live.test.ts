import { describe, expect, it } from "vitest";
import { GigasttClient } from "../src/index.js";
import type { Transcript } from "../src/index.js";

// Live-server integration test. Skipped unless GIGASTT_TEST_WS_URL points at
// a running gigastt server (which requires the model, ~850 MB):
//
//   cargo run --release -- serve   # in the repository root
//   GIGASTT_TEST_WS_URL=ws://127.0.0.1:9876/v1/ws npx vitest run test/live.test.ts

const LIVE_URL = process.env.GIGASTT_TEST_WS_URL;

/** Generates `seconds` of a 440 Hz sine tone as 16 kHz PCM16 frames (20 ms). */
function* sineFrames(seconds: number): Generator<Uint8Array> {
  const rate = 16_000;
  const frameSamples = rate / 50;
  const totalFrames = seconds * 50;
  for (let i = 0; i < totalFrames; i++) {
    const frame = new Int16Array(frameSamples);
    for (let j = 0; j < frameSamples; j++) {
      frame[j] = Math.round(
        12_000 * Math.sin((2 * Math.PI * 440 * (i * frameSamples + j)) / rate),
      );
    }
    yield new Uint8Array(frame.buffer);
  }
}

describe("GigasttClient against a live server", () => {
  it.skipIf(LIVE_URL === undefined)(
    "completes a ready -> audio -> stop -> final session",
    async () => {
      const finals: Transcript[] = [];
      const errors: string[] = [];

      let resolveFinal!: () => void;
      const finalPromise = new Promise<void>((resolve) => {
        resolveFinal = resolve;
      });

      const client = await GigasttClient.connect(LIVE_URL as string, {
        sampleRate: 16_000,
        handlers: {
          onFinal: (t) => {
            finals.push(t);
            resolveFinal();
          },
          onError: (e) => errors.push(`${e.code}: ${e.message}`),
        },
      });

      expect(client.ready.model).not.toBe("");

      for (const frame of sineFrames(2)) {
        client.sendPCM(frame);
        await new Promise((resolve) => setTimeout(resolve, 10));
      }
      client.stop();

      await Promise.race([
        finalPromise,
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error("no final within 30s")), 30_000),
        ),
      ]);

      expect(errors).toEqual([]);
      expect(finals.length).toBeGreaterThanOrEqual(1);
      expect(finals.at(-1)?.is_final).toBe(true);

      client.close();
    },
    45_000,
  );
});
