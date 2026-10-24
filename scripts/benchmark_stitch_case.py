#!/usr/bin/env python3
"""Compare two release binaries on frozen long-form audio, casing and warm REST latency.

Run under a resource-limited systemd user scope; uses only local assets and stdlib.
Raw responses, commands, checksums and every timing are retained in --output.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import signal
import statistics
import subprocess
import time
import urllib.error
import urllib.request
import wave

from longform_stitch import edits, normalize

ROOT = Path(__file__).resolve().parents[1]
ENCODERS = {
    'rnnt': '1d5a6f580b692e38ba35ed98d0f54648808f27b0a147dc11bd8978c15dc1a18f',
    'e2e_rnnt': '16805903b102d8b27bb0253de044f25e33a8a5bf07fce0c4da4ac1a5a7544091',
}
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def shifted_wav(path, shift):
    with wave.open(str(path)) as source:
        params = source.getparams()
        frames = source.readframes(source.getnframes())
    out = io.BytesIO()
    with wave.open(out, 'wb') as dest:
        dest.setparams(params)
        dest.writeframes(bytes(shift * params.framerate * params.nchannels * params.sampwidth) + frames)
    return out.getvalue(), params.nframes / params.framerate + shift


def casing(reference, hypothesis):
    """Case-sensitive WER and capitalization counts on lexically aligned matches.

    Align normalized words first, so a recognition insertion does not shift every
    subsequent capitalization comparison. Unmatched words remain in WER, not F1.
    """
    ref = re.findall(r'[^\W_]+', reference, re.UNICODE)
    hyp = re.findall(r'[^\W_]+', hypothesis, re.UNICODE)
    a = [w.lower().replace('ё', 'е') for w in ref]
    b = [w.lower().replace('ё', 'е') for w in hyp]
    matrix = [list(range(len(b) + 1))]
    for i, word in enumerate(a, 1):
        row = [i]
        for j, other in enumerate(b, 1):
            row.append(min(matrix[-1][j] + 1, row[-1] + 1,
                           matrix[-1][j - 1] + (word != other)))
        matrix.append(row)
    i, j = len(a), len(b)
    tp = fp = fn = matched = errors = 0
    while i and j:
        if matrix[i][j] == matrix[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            if a[i - 1] == b[j - 1]:
                r, h = ref[i - 1][0].isupper(), hyp[j - 1][0].isupper()
                matched += 1
                errors += r != h
                tp += r and h
                fp += not r and h
                fn += r and not h
            i, j = i - 1, j - 1
        elif matrix[i][j] == matrix[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return {'reference_words': len(ref), 'case_sensitive_sdi': edits(ref, hyp),
            'aligned_words': matched, 'initial_case_errors': errors,
            'capitalization_tp_fp_fn': [tp, fp, fn]}


def require(condition, message):
    """Validation must remain active under python -O / PYTHONOPTIMIZE."""
    if not condition:
        raise ValueError(message)


def warmup_pair(ports, audio, output, head, sample):
    expected = {}
    for arm, port in ports.items():
        expected[arm], _ = request(port, audio)
        save(output / f'warmup-{head}-{arm}-{sample}.json', expected[arm])
    # Timing-only runs must not silently accept a faster but incorrect result.
    changes = delta(expected['baseline'], expected['candidate'])
    save(output / f'warmup-{head}-{sample}-changes.json', changes)
    return expected


def delta(base, candidate):
    differences = []
    require(len(base['words']) == len(candidate['words']), 'word count changed')
    for index, (a, b) in enumerate(zip(base['words'], candidate['words'])):
        require({k: v for k, v in a.items() if k != 'word'} == {
            k: v for k, v in b.items() if k != 'word'}, 'word metadata changed')
        require(a['word'].lower() == b['word'].lower(), 'lexical/punctuation change')
        if a['word'] != b['word']:
            differences.append({'index': index, 'start': a['start'], 'base': a['word'],
                                'candidate': b['word'], 'context': ' '.join(
                                    w['word'] for w in base['words'][max(0, index - 5):index + 6])})
    require(base['text'].lower() == candidate['text'].lower(), 'rendered text changed beyond case')
    require({k: v for k, v in base.items() if k not in ('words', 'text')} == {
        k: v for k, v in candidate.items() if k not in ('words', 'text')}, 'response metadata changed')
    return differences


def rss(pid):
    fields = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/status').read_text().splitlines())
    return {key: int(fields[key].split()[0]) for key in ('VmRSS', 'VmHWM')}


def cpu_seconds(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK')


def request(port, data):
    start = time.perf_counter()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/transcribe', data=data,
                                 headers={'Content-Type': 'audio/wav'}, method='POST')
    with HTTP.open(req, timeout=120) as response:
        body = json.load(response)
    return body, time.perf_counter() - start


def start_server(binary, port, head, args, log):
    command = [str(binary), 'serve', '--offline', '--model-dir', str(args.models),
               '--model-variant', head, '--execution-provider', 'cpu', '--port', str(port),
               '--pool-size', '1', '--encoder-intra-threads', '2',
               '--file-window-concurrency', '1', '--punctuation', 'off', '--itn', 'off']
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIGASTT_')}
    env['RUST_LOG'] = 'warn'
    begin = time.perf_counter()
    proc = subprocess.Popen(command, stdout=log, stderr=log, env=env)
    try:
        while time.perf_counter() - begin < 120:
            if proc.poll() is not None:
                raise RuntimeError(f'server exited: {command}')
            try:
                with HTTP.open(f'http://127.0.0.1:{port}/ready', timeout=1) as response:
                    if response.status == 200:
                        return proc, command, time.perf_counter() - begin
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.05)
        raise TimeoutError('server startup')
    except BaseException:
        proc.kill()
        proc.wait()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('baseline', 'candidate', 'models', 'audio', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'benchmark/manifests/longform_stitch.json')
    parser.add_argument('--baseline-revision', required=True)
    parser.add_argument('--candidate-revision', required=True)
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--phase', choices=('all', 'quality', 'timing'), default='all')
    parser.add_argument('--cpus', default='2,3', help='two physical CPU IDs of the same core type, comma-separated')
    args = parser.parse_args()
    if args.rounds < 3:
        parser.error('at least three timing rounds are required')
    args.output.mkdir(parents=True, exist_ok=False)
    cpus = {int(value) for value in args.cpus.split(',')}
    if len(cpus) != 2:
        parser.error('select two distinct physical CPU IDs')
    os.sched_setaffinity(0, cpus)
    manifest = json.loads(args.manifest.read_text())
    for sample in manifest['samples']:
        require(digest(args.audio / sample['file']) == sample['sha256'],
                f"audio checksum mismatch: {sample['id']}")
    for head, checksum in ENCODERS.items():
        require(digest(args.models / f'v3_{head}_encoder_int8.onnx') == checksum,
                f'encoder checksum mismatch: {head}')
    metadata = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'rustc': subprocess.check_output(['rustc', '-Vv'], text=True),
                'baseline_revision': args.baseline_revision, 'candidate_revision': args.candidate_revision,
                'binaries': {arm: digest(getattr(args, arm)) for arm in ('baseline', 'candidate')},
                'manifest_sha256': digest(args.manifest), 'script_sha256': digest(__file__),
                'models': {p.name: digest(p) for p in args.models.glob('v3_*') if p.is_file()},
                'platform': platform.platform(), 'cpu': subprocess.check_output(['lscpu'], text=True),
                'affinity': sorted(os.sched_getaffinity(0)), 'rounds': args.rounds, 'phase': args.phase,
                'timing': 'Warm sequential REST; alternating AB/BA rounds; one warmup per file and arm; no concurrent requests',
                'startup': 'Optimized cache primed before both servers; OS page cache uncontrolled; startup timings descriptive only'}
    save(args.output / 'metadata.json', metadata)
    quality, timings, servers = [], [], []
    for head in ENCODERS:
        processes, ports, logs = {}, {'baseline': 19876, 'candidate': 19877}, []
        try:
            # A freshly optimized source session and an ORT-cache session can
            # differ in kernel prepacking. Warm requests do not remove that
            # difference: prime the disk cache, then restart before comparison.
            with (args.output / f'{head}-cache-prime.log').open('w') as log:
                prime, _, _ = start_server(args.baseline, 19878, head, args, log)
                prime.send_signal(signal.SIGTERM)
                try:
                    prime.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    prime.kill()
                    prime.wait()
            for arm, port in ports.items():
                log = (args.output / f'{head}-{arm}.log').open('w')
                logs.append(log)
                proc, command, startup = start_server(getattr(args, arm), port, head, args, log)
                processes[arm] = proc
                if 'loaded fresh optimized graph cache' not in (args.output / f'{head}-{arm}.log').read_text():
                    raise RuntimeError('Both measured servers must load an existing optimized graph cache')
                with HTTP.open(f'http://127.0.0.1:{port}/v1/models', timeout=10) as response:
                    model_status = json.load(response)
                servers.append({'head': head, 'arm': arm, 'command': command,
                                'startup_seconds': startup, 'model_status': model_status})
            for sample in (manifest['samples'] if args.phase != 'timing' else []):
                for shift in (0, 3, 22):
                    key = f"{head}-{sample['id']}-{shift}"
                    audio, seconds = shifted_wav(args.audio / sample['file'], shift)
                    bodies = {}
                    order = ['baseline', 'candidate'] if len(quality) % 2 == 0 else ['candidate', 'baseline']
                    for arm in order:
                        bodies[arm], _ = request(ports[arm], audio)
                        save(args.output / f'{key}-{arm}.json', bodies[arm])
                    row = {'id': key, 'head': head, 'sample': sample['id'], 'shift': shift,
                           'audio_seconds': seconds, 'reference_words': len(normalize(sample['reference'])),
                           'changes': delta(bodies['baseline'], bodies['candidate'])}
                    for arm, body in bodies.items():
                        row[arm] = {'normalized_sdi': edits(normalize(sample['reference']), normalize(body['text']))}
                        # Only the podcast sources have sentence-level cased references.
                        if sample['id'].startswith('podlodka-'):
                            row[arm]['casing'] = casing(sample['reference'], body['text'])
                    quality.append(row)
                    save(args.output / 'quality.json', quality)
                    print('quality', key, 'case changes', len(row['changes']), flush=True)
            # Two durations exercise both a few windows and a sustained long file.
            perf_samples = [s for s in manifest['samples'] if s['id'] in ('long_example', 'podlodka-306')]
            for sample in (perf_samples if args.phase != 'quality' else []):
                audio, seconds = shifted_wav(args.audio / sample['file'], 0)
                expected = warmup_pair(ports, audio, args.output, head, sample['id'])
                for round_index in range(args.rounds):
                    order = ['baseline', 'candidate'] if round_index % 2 == 0 else ['candidate', 'baseline']
                    for arm in order:
                        cpu_before = cpu_seconds(processes[arm].pid)
                        body, elapsed = request(ports[arm], audio)
                        cpu_used = cpu_seconds(processes[arm].pid) - cpu_before
                        if body != expected[arm]:
                            save(args.output / f"nondeterministic-{head}-{arm}-{sample['id']}-{round_index}.json", body)
                            raise AssertionError('non-deterministic repeated transcription')
                        timings.append({'head': head, 'sample': sample['id'], 'round': round_index,
                                        'arm': arm, 'seconds': elapsed, 'rtf': elapsed / seconds,
                                        'cpu_seconds': cpu_used, 'system_loadavg': list(os.getloadavg()),
                                        'audio_seconds': seconds, **rss(processes[arm].pid)})
                        save(args.output / 'timings.json', timings)
                    print('timing', head, sample['id'], round_index, flush=True)
        finally:
            for proc in processes.values():
                proc.send_signal(signal.SIGTERM)
            for proc in processes.values():
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            for log in logs:
                log.close()
            save(args.output / 'servers.json', servers)
    summary = {'quality': {}, 'timing': {}}
    for head in ENCODERS:
        rows = [r for r in quality if r['head'] == head]
        summary['quality'][head] = {
            'cases': len(rows), 'audio_seconds': sum(r['audio_seconds'] for r in rows),
            'reference_words': sum(r['reference_words'] for r in rows),
            'changed_words': sum(len(r['changes']) for r in rows),
            **{arm: {'normalized_sdi': [sum(r[arm]['normalized_sdi'][i] for r in rows) for i in range(3)],
                     'case_errors_on_aligned_podcast_words': sum(r[arm].get('casing', {}).get('initial_case_errors', 0) for r in rows)}
               for arm in ('baseline', 'candidate')}}
        for sample in (('long_example', 'podlodka-306') if timings else ()):
            rows = [r for r in timings if r['head'] == head and r['sample'] == sample]
            b = [r for r in rows if r['arm'] == 'baseline']
            c = [r for r in rows if r['arm'] == 'candidate']
            summary['timing'][head + '/' + sample] = {
                'median_paired_ratio': statistics.median(y['seconds'] / x['seconds'] for x, y in zip(b, c)),
                **{arm: {key: statistics.median(r[key] for r in rows if r['arm'] == arm)
                         for key in ('seconds', 'rtf', 'cpu_seconds', 'VmRSS', 'VmHWM')} for arm in ('baseline', 'candidate')}}
    save(args.output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
