#!/usr/bin/env python3
"""Freeze local current-head packs and available Russian benchmark recordings.

No downloads and no mutation of the source model directory. Missing recordings
are explicitly recorded, never replaced by another utterance with the same text.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


def prepare(repo, source, v3, destination, probe):
    spec = importlib.util.spec_from_file_location('punctuation_runner', repo / 'scripts/compare-punctuation-heads.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    scoring = helper.load_module(repo / 'scripts/score-current-asr.py')
    catalog = json.loads(subprocess.check_output([str(probe), 'catalog']))
    verified = {entry['head']: scoring.verify_assets(source, entry['assets']) for entry in catalog}
    upstream = json.loads((v3 / 'download-protocol.json').read_text())
    for head, names in [('v3_ctc', ['v3_ctc.int8.onnx', 'v3_vocab.txt']),
                        ('v3_e2e_ctc', ['v3_e2e_ctc.int8.onnx', 'v3_e2e_ctc_vocab.txt'])]:
        verified[head] = scoring.verify_assets(v3, {name: upstream['files'][name]['sha256'] for name in names})
    destination.mkdir(parents=True, exist_ok=False)
    cases, datasets = [], {}
    for name in ['fleurs_ru_punct', 'golos_crowd_1k', 'golos_farfield', 'podlodka']:
        path = repo / 'benchmark/manifests' / (name + '.json')
        manifest = json.loads(path.read_text())
        audio_root = Path(manifest['audio_root']).expanduser()
        if name == 'fleurs_ru_punct':
            audio_root = audio_root.with_name('fleurs_ru')
        missing = []
        for row in manifest['samples']:
            audio = audio_root / row['filename']
            if not audio.is_file():
                missing.append(row['filename'])
                continue
            cases.append(dict(id=name + ':' + row['filename'], dataset=name, audio=str(audio),
                              reference=row['reference'], duration=helper.duration(audio),
                              sha256=hashlib.sha256(audio.read_bytes()).hexdigest()))
        datasets[name] = {key: manifest.get(key) for key in ['source', 'license', 'attribution']}
        datasets[name].update(manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              requested=len(manifest['samples']), missing=missing,
                              included=sum(r['dataset'] == name for r in cases))
        if datasets[name]['included'] == 0:
            raise FileNotFoundError(f'no available recordings for {name}: {audio_root}')
    (destination / 'cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    # A fixed, duration-stratified ten recordings per domain for timing repeats.
    timing = []
    for name in datasets:
        rows = sorted((r for r in cases if r['dataset'] == name), key=lambda r: (r['duration'], r['id']))
        timing.extend(rows[(len(rows)-1)*i//9] for i in range(10))
    (destination / 'timing-cases.json').write_text(json.dumps(timing, ensure_ascii=False, indent=2))
    profiles = []
    for head, architecture, files, origin in [
        ('rnnt', 'rnnt', ['v3_rnnt_encoder_int8.onnx', 'v3_rnnt_decoder.onnx', 'v3_rnnt_joint.onnx', 'v3_vocab.txt'], source),
        ('e2e_rnnt', 'e2e_rnnt', ['v3_e2e_rnnt_encoder_int8.onnx', 'v3_e2e_rnnt_decoder.onnx', 'v3_e2e_rnnt_joint.onnx', 'v3_e2e_rnnt_vocab.txt'], source),
        ('v3_ctc', 'ml_ctc', ['v3_ctc.int8.onnx', 'v3_vocab.txt'], v3),
        ('v3_e2e_ctc', 'ml_ctc', ['v3_e2e_ctc.int8.onnx', 'v3_e2e_ctc_vocab.txt'], v3),
        ('ml_ctc', 'ml_ctc', ['multilingual_ctc.int8.onnx', 'multilingual_vocab.txt'], source),
        ('ml_ctc_large', 'ml_ctc_large', ['multilingual_large_ctc.int8.onnx', 'multilingual_vocab.txt'], source),
    ]:
        pack = destination / 'models' / head
        pack.mkdir(parents=True)
        assets = verified[head]
        for name in files:
            path = origin / name
            (pack / name).symlink_to(path.resolve())
        text = f'architecture = "{architecture}"\n[files]\nencoder = "{files[0]}"\nencoder_int8 = "{files[0]}"\nvocab = "{files[-1]}"\n'
        if architecture in ['rnnt', 'e2e_rnnt']:
            text += f'decoder = "{files[1]}"\njoint = "{files[2]}"\n'
        (pack / 'manifest.toml').write_text(text)
        profiles.append(dict(head=head, architecture=architecture, model_dir=str(pack.resolve()),
                             builtin_punctuation=head in ['e2e_rnnt', 'v3_e2e_ctc'], assets=assets, manifest=text))
    protocol = dict(datasets=datasets, profiles=profiles, active_catalog=catalog, upstream_ctc=upstream, encoder_threads=4, pool_size=1,
                    audio_seconds=sum(r['duration'] for r in cases), timing_cases=len(timing))
    (destination / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    return protocol


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['models', 'v3', 'output', 'probe']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(Path(__file__).resolve().parents[1], args.models, args.v3, args.output, args.probe), ensure_ascii=False, indent=2))
