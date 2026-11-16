# Non-Opus Matroska routing fixture

`nonopus-vorbis.mka` is a synthetic 997 Hz mono tone, 48 kHz, 0.12 seconds.
Generated with FFmpeg 7.0.2 (libvorbis), without input media:

```sh
ffmpeg -v error -f lavfi -i sine=frequency=997:sample_rate=48000:duration=0.12 \
  -map_metadata -1 -c:a libvorbis -f matroska nonopus-vorbis.mka
```

Its Matroska metadata includes CRC-32 elements accepted by the existing
Symphonia route but unsupported by unlace 0.2.4. It verifies that enabling the
optional Opus demuxer does not apply its stricter admission to non-Opus files.
The fixture is checked in; regeneration can vary container metadata and bytes.

`nonopus-late-tracks.mka` retains the same Info, Cluster and Tracks elements,
but places Tracks after the Cluster with a SeekHead entry pointing to it. It
checks legacy seekable codec identification before the optional Opus route.
Its decoded PCM must equal the original synthetic fixture.
