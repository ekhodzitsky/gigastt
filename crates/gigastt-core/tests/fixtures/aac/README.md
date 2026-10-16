# AAC transport regression fixtures

Derived from the existing repository speech fixture
`crates/gigastt/tests/fixtures/golos_00.wav`; no new recording was introduced.
Generated with FFmpeg 7.0.2-static (johnvansickle.com, GCC 8, native AAC and
libmp3lame encoders). The tests consume the files without requiring FFmpeg.

From the repository root:

```sh
ffmpeg -i crates/gigastt/tests/fixtures/golos_00.wav -ar 48000 -ac 1 -c:a aac -b:a 64k speech.m4a
ffmpeg -i speech.m4a -c:a copy -f matroska speech.mka
ffmpeg -i crates/gigastt/tests/fixtures/golos_00.wav -c:a libmp3lame -f mp4 mp3.mp4
```

The Matroska fixture was muxed with FFmpeg's random segment identifier; a
regeneration need not have the same container hash. Decoded content is the test
input. Exact committed SHA-256 hashes:

- `golos_00.wav`: `500d2e88a1634ee766fec46f83cb903f3a1c276b2cedb36f11db800a1c4c14d0`
- `mp3.mp4`: `4c59b4c1960be5ddc20eda48c0720c2c35b215281c1ee68cf91db0eb3287e282`
- `speech.m4a`: `e7f0f3ee0300ca9951e22ca79a4ee0a22619eba37b56f208d4f644cd24781f13`
- `speech.mka`: `e3d95cb170cdad51991b4b9111a104a35122ff98c98b137f6dab5b70fdf79946`
