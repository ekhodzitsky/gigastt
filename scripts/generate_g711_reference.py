#!/usr/bin/env python3
"""Regenerate the independent mu-law codepoint oracle (FFmpeg 7.0.2-static)."""
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent.parent
output = root / 'crates/gigastt-core/tests/fixtures/telephony/pcmu_codepoints_ffmpeg.pcm'
subprocess.run([
    'ffmpeg', '-y', '-v', 'error', '-f', 'mulaw', '-ar', '8000', '-ac', '1',
    '-i', 'pipe:0', '-f', 's16le', str(output),
], input=bytes(range(256)), check=True)
if output.stat().st_size != 512:
    raise RuntimeError('expected 256 signed PCM16 codepoints')
