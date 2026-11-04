// Model-backed smoke and optional matched-revision benchmark; both run manually.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const crypto = require('node:crypto');
const { performance } = require('node:perf_hooks');
const os = require('node:os');
const path = require('node:path');
const override = process.env.GIGASTT_TEST_ADDON;
const binding = require(override ? path.resolve(override) : '../loader.js');
const { Engine, Stream } = binding;
const addon = override ? path.resolve(override) : path.resolve(
  __dirname, `../gigastt.${binding.platformKey()}.node`
);
const models = process.env.GIGASTT_MODEL_DIR || path.join(os.homedir(), '.gigastt', 'models');
const fixture = path.resolve(__dirname, '../../gigastt/tests/fixtures/golos_00.wav');
const benchmark = process.argv.includes('--benchmark');
const hash = (file) => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
(async () => {
  assert.equal(typeof global.gc, 'function', 'run with --expose-gc');
  const start = performance.now();
  let engine = new Engine(models, 2);
  const loadMs = performance.now() - start;
  const expected = await engine.transcribeFile(fixture);
  assert.ok(expected.text.trim());
  assert.ok(expected.words.length);
  for (const word of expected.words) {
    assert.ok(Number.isFinite(word.startS) && Number.isFinite(word.endS));
    assert.ok(word.endS >= word.startS);
  }
  assert.throws(() => new Engine('/nonexistent/gigastt-audit-model'), /ModelNotFound:/);
  await assert.rejects(engine.transcribeFile('/nonexistent/gigastt-audit-audio'), /Inference:/);
  for (const result of await Promise.all([
    engine.transcribeFile(fixture), engine.transcribeFile(fixture),
  ])) assert.deepEqual(result, expected);
  const timings = [];
  let ticks = 0;
  const timer = setInterval(() => ticks++, 1);
  try {
    for (let i = 0; i < (benchmark ? 21 : 3); i++) {
      let start = performance.now();
      const serial = await engine.transcribeFile(fixture);
      const serialMs = performance.now() - start;
      assert.deepEqual(serial, expected);
      start = performance.now();
      const results = await Promise.all([
        engine.transcribeFile(fixture), engine.transcribeFile(fixture),
      ]);
      const concurrentMs = performance.now() - start;
      for (const result of results) assert.deepEqual(result, expected);
      global.gc();
      timings.push({ round: i, serialMs, concurrentMs, rss: process.memoryUsage().rss });
    }
  } finally {
    clearInterval(timer);
  }
  assert.ok(ticks > 0, 'async inference must leave the event loop responsive');
  let stream = new Stream(engine);
  const wav = fs.readFileSync(fixture);
  assert.equal(wav.toString('ascii', 0, 4), 'RIFF');
  assert.equal(wav.toString('ascii', 8, 12), 'WAVE');
  let pcm, format;
  for (let offset = 12; offset + 8 <= wav.length;) {
    const size = wav.readUInt32LE(offset + 4);
    const end = offset + 8 + size;
    assert.ok(end <= wav.length, 'truncated fixture chunk');
    const id = wav.toString('ascii', offset, offset + 4);
    if (id === 'fmt ') format = wav.subarray(offset + 8, end);
    if (id === 'data') pcm = wav.subarray(offset + 8, end);
    offset = end + (size & 1);
  }
  assert.ok(format && format.length >= 16);
  assert.equal(format.readUInt16LE(0), 1, 'fixture must be PCM');
  assert.equal(format.readUInt16LE(2), 1, 'fixture must be mono');
  assert.equal(format.readUInt32LE(4), 16000);
  assert.equal(format.readUInt16LE(14), 16);
  assert.ok(pcm && pcm.length > 0 && pcm.length % 2 === 0);
  const segments = [];
  for (let off = 0; off < pcm.length; off += 16000) {
    segments.push(...await stream.processChunk(pcm.subarray(off, off + 16000), 16000));
  }
  segments.push(...await stream.flush());
  assert.ok(segments.some((s) => s.isFinal && s.text.trim()));
  stream = null;
  // An outstanding task must retain its engine even when JS releases it.
  const pending = engine.transcribeFile(fixture);
  engine = null;
  global.gc();
  assert.deepEqual(await pending, expected);
  await new Promise((resolve) => setImmediate(resolve));
  global.gc();
  if (benchmark) {
    console.log(JSON.stringify({
      poolSize: 2, node: process.version, addonSha256: hash(addon),
      fixtureSha256: hash(fixture), loadMs, timings, expected, segments, ticks,
      afterGcRss: process.memoryUsage().rss, maxRssKiB: process.resourceUsage().maxRSS,
    }));
  } else {
    console.log('SMOKE OK:', JSON.stringify(expected.text));
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
