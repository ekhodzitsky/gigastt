#!/usr/bin/env python3
"""Reproduce seam capitals using individual public podcast excerpts plus silence.

Prepare the frozen long-form corpus with longform_stitch.py first. --parts points
at its ruls-parts directory. Audio is not redistributed. These targeted examples
are diagnostic cases, not an unbiased quality or performance benchmark.
"""
import argparse
import io
import json
from pathlib import Path
import signal
import subprocess
import wave

from benchmark_stitch_case import ENCODERS, delta, digest, require, save, start_server, request

ROOT = Path(__file__).resolve().parents[1]


def wav_bytes(params, data):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as dest:
        dest.setparams(params)
        dest.writeframes(data)
    return buffer.getvalue()


def prepare(case, parts):
    source = parts / case['part']
    require(digest(source) == case['part_sha256'], f'source checksum: {source}')
    with wave.open(str(source)) as stream:
        params = stream.getparams()
        require((params.framerate, params.nchannels, params.sampwidth) ==
                (case['sample_rate'], case['channels'], 2), 'unexpected audio format')
        stream.setpos(case['source_offset_frames'])
        speech = stream.readframes(case['source_frames'])
    frame_bytes = params.nchannels * params.sampwidth
    require(len(speech) == case['source_frames'] * frame_bytes, 'truncated source')
    tail = case['total_frames'] - case['leading_silence_frames'] - case['source_frames']
    require(tail >= 0, 'negative trailing silence')
    data = bytes(case['leading_silence_frames'] * frame_bytes) + speech + bytes(tail * frame_bytes)
    second = params.framerate * frame_bytes
    return {key: wav_bytes(params, pcm) for key, pcm in {
        'full': data, 'old': data[:24 * second], 'new': data[22 * second:],
        'speech': speech,
    }.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'models', 'parts', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--manifest', type=Path,
                        default=ROOT / 'benchmark/results/seam_reproduction_20261024/inputs.json')
    args = parser.parse_args()
    # Popen must receive an absolute binary path, including for local binaries.
    args.baseline, args.candidate = args.baseline.resolve(), args.candidate.resolve()
    require(digest(args.models / 'v3_e2e_rnnt_encoder_int8.onnx') == ENCODERS['e2e_rnnt'],
            'encoder checksum mismatch')
    cases = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    for case in cases:
        for kind, data in prepare(case, args.parts).items():
            path = args.output / f"{case['id']}-{kind}.wav"
            path.write_bytes(data)
            if kind == 'full':
                require(digest(path) == case['wav_sha256'], 'prepared WAV checksum mismatch')
    provenance = {'script_sha256': digest(__file__), 'manifest_sha256': digest(args.manifest),
                  'encoder_sha256': ENCODERS['e2e_rnnt'], 'servers': {}}
    for arm in ('baseline', 'candidate'):
        binary = getattr(args, arm)
        with (args.output / f'{arm}.log').open('w') as log:
            proc, command, _ = start_server(binary, 19886, 'e2e_rnnt', args, log)
            provenance['servers'][arm] = {'command': command, 'binary_sha256': digest(binary)}
            save(args.output / 'provenance.json', provenance)
            try:
                for case in cases:
                    for kind in ('full', 'old', 'new', 'speech'):
                        data = (args.output / f"{case['id']}-{kind}.wav").read_bytes()
                        body, _ = request(19886, data)
                        save(args.output / f"{case['id']}-{arm}-{kind}.json", body)
                        if kind == 'full':
                            for repeat in range(2):
                                repeated, _ = request(19886, data)
                                save(args.output / f"{case['id']}-{arm}-repeat-{repeat}.json", repeated)
                                require(repeated == body, 'nondeterministic full response')
                    print(arm, case['id'], 'completed', flush=True)
            finally:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    summary = []
    for case in cases:
        bodies = {arm: json.loads((args.output / f"{case['id']}-{arm}-full.json").read_text())
                  for arm in ('baseline', 'candidate')}
        summary.append({'id': case['id'], 'changes': delta(bodies['baseline'], bodies['candidate'])})
    save(args.output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
