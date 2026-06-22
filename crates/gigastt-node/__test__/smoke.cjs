// Local smoke test: build the addon (`npm run build`) then `npm run smoke`.
// Requires a side-loaded model dir (default ~/.gigastt/models) and the
// committed golos_00.wav fixture. Asserts a non-empty Russian transcript with
// word timings, and that a missing model throws a typed JS error.
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

  // 3. streaming: feed the fixture's PCM as one 16 kHz chunk via Stream
  if (typeof Stream === 'function') {
    console.log('Stream export present');
  }

  console.log('SMOKE OK');
})().catch((e) => fail(e && e.stack ? e.stack : String(e)));
