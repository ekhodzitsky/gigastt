// Local smoke test: build the addon (`npm run build`) then `npm run smoke`.
// Requires a side-loaded model dir (default ~/.gigastt/models) and the
// committed golos_00.wav fixture (16 kHz mono PCM16). Asserts a non-empty
// Russian transcript with word timings from `transcribeFile`, a non-empty final
// transcript from a `Stream` fed the fixture's PCM, and that a missing model
// throws a typed JS error.
const fs = require('fs');
const os = require('os');
const path = require('path');
const { Engine, Stream } = require('../loader.js');

const modelDir =
  process.env.GIGASTT_MODEL_DIR || path.join(os.homedir(), '.gigastt', 'models');
const fixture = path.resolve(
  __dirname,
  '../../gigastt/tests/fixtures/golos_00.wav'
);

function fail(msg) {
  console.error('SMOKE FAIL:', msg);
  process.exit(1);
}

(async () => {
  // 1. file transcription
  const engine = new Engine(modelDir);
  const t = await engine.transcribeFile(fixture);
  if (!t.text || t.text.trim().length === 0) fail('empty transcript');
  if (!Array.isArray(t.words) || t.words.length === 0) fail('no words');
  const w0 = t.words[0];
  if (typeof w0.startS !== 'number' || typeof w0.endS !== 'number') {
    fail('word timings missing');
  }
  console.log('transcript:', JSON.stringify(t.text));
  console.log('words:', t.words.length, '| duration_s:', t.durationS);

  // 2. typed error on a missing model
  let threw = false;
  try {
    new Engine('/no/such/model/dir');
  } catch (e) {
    threw = true;
    console.log('error path OK, code prefix:', e.message.split(':')[0]);
  }
  if (!threw) fail('expected a throw for a missing model dir');

  // 3. streaming: feed the fixture's PCM through a Stream in 0.5 s chunks
  // (little-endian mono PCM16 @ 16 kHz), then flush what remains buffered.
  // Segments finalize mid-stream at endpoint boundaries, so finals arrive from
  // both processChunk and flush — collect them from each.
  const wav = fs.readFileSync(fixture);
  const dataOff = wav.indexOf('data', 12); // data chunk descriptor
  if (dataOff < 0) fail('fixture has no data chunk');
  const dataSize = wav.readUInt32LE(dataOff + 4);
  const pcm = wav.subarray(dataOff + 8, dataOff + 8 + dataSize);
  const stream = new Stream(engine);
  const CHUNK = 16000; // 0.5 s of 16 kHz mono PCM16 (8000 samples * 2 bytes)
  const finals = [];
  let partials = 0;
  for (let off = 0; off < pcm.length; off += CHUNK) {
    // Await each chunk before sending the next to preserve ordering.
    for (const s of await stream.processChunk(pcm.subarray(off, off + CHUNK), 16000)) {
      s.isFinal ? finals.push(s) : partials++;
    }
  }
  for (const s of await stream.flush()) {
    s.isFinal ? finals.push(s) : partials++;
  }
  const streamed = finals.map((s) => s.text).join(' ').trim();
  if (!streamed) fail('stream transcription empty');
  console.log('streamed:', JSON.stringify(streamed), '| partial segs:', partials);

  console.log('SMOKE OK');
})().catch((e) => fail(e && e.stack ? e.stack : String(e)));
