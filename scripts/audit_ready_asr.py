#!/usr/bin/env python3
"""Certify paired ready-ASR results without modifying hypotheses or references.

Partial ledgers may be inspected explicitly; they never become completed runs.
Scores use the established normalization, independent word DP and character
Levenshtein distance. No automatic language-level winner or pooled WER is emitted.
"""
import argparse
import hashlib
from functools import lru_cache
import json
from pathlib import Path

import soundfile as sf

from audit_multilingual_1000 import digest, require
from audit_multilingual_telephony import totals, verify_scores
from wer_unicode import normalize

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'benchmark/results/multilingual_model_comparison_20261001'
CORPUS_KEYS = {
    'kk_conversations_expanded': 'kk_conversation',
    'kk': 'kk_codeswitch',
    'uz': 'uz_conversation',
    'ky': 'ky_read',
}
EXPECTED = [('whisper', c) for c in ('kk_conversations_expanded', 'kk', 'uz')]
EXPECTED += [('omni', c) for c in CORPUS_KEYS]
SAMPLE_KEYS = ('id', 'reference', 'sha256', 'path', 'language', 'dataset', 'split')


@lru_cache(maxsize=None)
def verified_digest(path):
    return digest(Path(path))


def verify_pairing(samples, rows, *, complete):
    require(len(rows) <= len(samples), 'Ledger exceeds frozen population')
    require(len({r['id'] for r in rows}) == len(rows), 'Duplicate ledger IDs')
    require(len({s['id'] for s in samples}) == len(samples), 'Duplicate manifest IDs')
    if complete:
        require(len(rows) == len(samples), 'Incomplete final population')
    for sample, row in zip(samples, rows):
        require(all(row.get(k) == sample[k] for k in SAMPLE_KEYS), 'Changed sample or ledger order')
        require(row.get('condition', 'original') == 'original', 'Non-original audio condition')
        require(row.get('status') in ('ok', 'failed'), 'Invalid request status')
        require(isinstance(row.get('hypothesis'), str), 'Hypothesis must be text')
        require(row['status'] != 'failed' or row['hypothesis'] == '', 'Failed row has text')
        require(row['status'] != 'ok' or row.get('error') is None, 'Successful row has error')


def read_ledger(path, allow_partial):
    data = path.read_bytes()
    uncommitted_tail = bool(data and not data.endswith(b'\n'))
    if uncommitted_tail:
        require(allow_partial, 'Uncommitted ledger tail')
        data = data[:data.rfind(b'\n') + 1]
    return [json.loads(line) for line in data.splitlines()], uncommitted_tail


def paired_metrics(base, rows):
    before, after = totals(base), totals(rows)
    delta = [b['scores']['word_errors'] - a['scores']['word_errors'] for a, b in zip(base, rows)]
    return dict(baseline=before, candidate=after,
                wer_delta_pp=None if before['wer_pct'] is None else after['wer_pct'] - before['wer_pct'],
                cer_delta_pp=None if before['cer_pct'] is None else after['cer_pct'] - before['cer_pct'],
                clips_improved=sum(d < 0 for d in delta), clips_worsened=sum(d > 0 for d in delta),
                clips_unchanged=sum(d == 0 for d in delta),
                newly_empty=sum(bool(normalize(a['hypothesis'])) and not normalize(b['hypothesis'])
                                for a, b in zip(base, rows)))


def verify_model(metadata, backend, model_dir):
    """Verify actual adapter-pinned weights and tokenizer bytes independently."""
    files = metadata.get('model_files')
    require(bool(files), 'Missing verifiable model file hashes')
    if backend == 'whisper':
        require(isinstance(files, dict) and set(files) == {'model.bin', 'config.json', 'tokenizer.json', 'vocabulary.json', 'preprocessor_config.json', 'README.md'}, 'Unexpected Whisper file inventory')
        require(metadata['model'] == 'Systran/faster-whisper-large-v3', 'Wrong Whisper model')
        revision = metadata['revision']
        require(revision == 'edaa852ec7e145841d8ffdb056a99866b5f0a478', 'Wrong Whisper revision')
        require(files['model.bin']['sha256'] == '69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1', 'Wrong Whisper weights')
        require(metadata['device'] == 'cpu' and metadata['requested_compute_type'] == 'int8'
                and metadata['actual_compute_type'] in ('int8', 'int8_float32')
                and metadata['cpu_threads'] == 3 and metadata['num_workers'] == 1,
                'Unexpected Whisper execution configuration')
        require(verified_digest(str(model_dir/'provenance.json')) == metadata['provenance_sha256'], 'Whisper provenance changed')
        inventory = [(name, item['sha256'], item['size_bytes']) for name, item in files.items()]
    else:
        require(isinstance(files, list), 'Unexpected Omni file inventory')
        revision = metadata['source_revision']
        require(revision == '81f51e224ce9e74b02cc2a3eaf21b2d91d743455', 'Wrong Omni source revision')
        require(metadata['device'] == 'cpu' and metadata['dtype'] == 'float32'
                and metadata['threads'] == 3 and metadata['interop_threads'] == 1
                and metadata['batch_size'] == 1 and metadata['language_conditioning'] is False,
                'Unexpected Omni execution configuration')
        inventory = [(item['filename'], item['sha256'], item['bytes']) for item in files]
        expected_files = {
            'omniASR-CTC-1B-v2.pt': (3902956068, '354f981756aa8f41591ea363e45b9c4eba1ec5144c2273af82e747efbb08919c'),
            'omniASR_tokenizer_written_v2.model': (91481, '8aa11a1092142ef472537476ef6e76541123e2f0d789b79f3ebd119008240b1e'),
        }
        require({name: (size, sha) for name, sha, size in inventory} == expected_files,
                'Omni pinned weight/tokenizer identity differs')
        require(metadata['source_archive_sha256'] == '67c484bba6502454bf192f549d0d0aee974f707179f5c35649679a44eaf0d435'
                and metadata['installed_source_tree_sha256'] == '947069e894e9adea2ba075346dd95eec3eeab7f64cae88eb4eb46600c12d9faa',
                'Omni official source identity differs')
        source_root = model_dir.parent/'omni-venv/lib/python3.12/site-packages/omnilingual_asr'
        require(source_root.is_dir(), 'Installed Omni source unavailable for independent verification')
        tree_hash = hashlib.sha256()
        for path in sorted(p for p in source_root.rglob('*') if p.suffix in {'.py', '.yaml'}):
            tree_hash.update(path.relative_to(source_root).as_posix().encode() + b'\0')
            tree_hash.update(hashlib.sha256(path.read_bytes()).digest())
        require(tree_hash.hexdigest() == metadata['installed_source_tree_sha256'], 'Installed Omni source files changed')
        require(metadata['model_card'] == 'omniASR_CTC_1B_v2'
                and metadata['tokenizer_card'] == 'omniASR_tokenizer_written_v2'
                and metadata['training'] is False and metadata['adaptation'] is False,
                'Omni model card or adaptation differs')

    for name, expected, size in inventory:
        require(Path(name).name == name, 'Model filename is not a basename')
        path = model_dir / name
        require(path.stat().st_size == size and verified_digest(str(path)) == expected, 'Model bytes changed')
    return dict(model_file_count=len(inventory), revision=revision, backend=backend,
                model_directory=str(model_dir.resolve()))


def verify_identity(meta, manifest_path, samples, backend, model_dir):
    require(meta['backend'] == backend and meta['n'] == len(samples), 'Backend/population changed')
    require(meta['language'] == samples[0]['language'], 'Wrong language identity')
    if backend == 'whisper':
        require(meta['model']['language'] == meta['language'], 'Whisper forced language differs')
        options = meta['model']['transcription_options']
        require(options['task'] == 'transcribe' and options['beam_size'] == 5
                and options['temperature'] == 0.0 and options['vad_filter'] is False
                and options['condition_on_previous_text'] is False
                and options['initial_prompt'] is None and options['hotwords'] is None,
                'Whisper decoder settings differ')
    if backend == 'omni':
        require(meta['model']['corpus_language'] == meta['language'], 'Omni corpus language differs')
    require(meta['manifest_sha256'] == verified_digest(str(manifest_path)), 'Manifest hash changed')
    require(meta['script_sha256'] == verified_digest(str(ROOT/'scripts/benchmark_ready_asr.py')), 'Runner changed')
    require(meta['backend_script_sha256'] == verified_digest(str(ROOT/'scripts'/f'ready_asr_{backend}.py')), 'Adapter changed')
    for name in ('benchmark_multilingual_public.py', 'wer_unicode.py'):
        require(meta['scorer_sha256'][name] == verified_digest(str(ROOT/'scripts'/name)), 'Scorer changed')
    require(meta['jiwer'] == '4.0.0' and meta['threads'] == 3, 'Scorer/thread configuration changed')
    require(meta['quality_workers'] == (4 if backend == 'whisper' else 1),
            'Quality worker count differs from pre-inference amendment')
    return verify_model(meta['model'], backend, model_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DIRECTORY)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--whisper-model-dir', type=Path, default=Path.home()/'.cache/gigastt-ready-comparison/models/whisper-large-v3')
    parser.add_argument('--omni-model-dir', type=Path, default=Path.home()/'.cache/gigastt-ready-comparison/omni-model')
    args = parser.parse_args()
    frozen = json.loads((args.directory/'baseline_pairing_audit.json').read_text())
    specs = {r['corpus']: r for r in frozen['corpora']}
    corpora = {}
    for key, name in CORPUS_KEYS.items():
        spec = specs[name]
        mp, bp = Path(spec['manifest']), Path(spec['baseline'])
        require(verified_digest(str(mp)) == spec['manifest_sha256'], 'Frozen manifest changed')
        require(verified_digest(str(bp)) == spec['baseline_sha256'], 'Frozen GigaAM baseline changed')
        samples = json.loads(mp.read_text())['samples']
        base = ([json.loads(line) for line in bp.read_text().splitlines()] if bp.suffix == '.jsonl'
                else json.loads(bp.read_text())['details'])
        require(len(base) == len({r['id'] for r in base}) == len(samples), 'Duplicate/incomplete baseline')
        baseline_by_id = {r['id']: r for r in base}
        require(set(baseline_by_id) == {s['id'] for s in samples}, 'Baseline ID set differs')
        base = [baseline_by_id[s['id']] for s in samples]
        verify_pairing(samples, base, complete=True)
        verify_scores(base)
        for sample in samples:
            require(verified_digest(sample['path']) == sample['sha256'], 'Input audio bytes changed')
        corpora[key] = (mp, samples, base)
    reports = []
    for backend, corpus in EXPECTED:
        prefix = args.directory / f'{backend}_{corpus}'
        final_path, ledger_path, meta_path = [prefix.with_suffix(s) for s in ('.json', '.jsonl', '.meta.json')]
        mp, samples, base = corpora[corpus]
        if not ledger_path.exists():
            require(args.allow_partial, f'Missing ledger: {ledger_path}')
            reports.append(dict(backend=backend, corpus=corpus, completed=False, n=0,
                                expected_n=len(samples), state='not_started'))
            continue
        require(meta_path.exists(), 'Ledger without identity')
        meta = json.loads(meta_path.read_text())
        model_audit = verify_identity(meta, mp, samples, backend,
                                      args.whisper_model_dir if backend == 'whisper' else args.omni_model_dir)
        # Snapshot final publication before reading the append-only ledger. If the
        # writer finishes afterward, this audit remains an honest partial snapshot.
        final = json.loads(final_path.read_text()) if final_path.exists() else None
        rows, tail = read_ledger(ledger_path, args.allow_partial)
        complete = final is not None and final.get('completed') is True
        require(args.allow_partial or complete, 'Missing completed final result')
        verify_pairing(samples, rows, complete=complete)
        verify_scores(rows)
        for sample, row in zip(samples, rows):
            info = sf.info(sample['path'])
            require(abs(row['duration_s'] - info.duration) <= 1/info.samplerate + 1e-9, 'Audio duration changed')
        if final is not None:
            require(final['metadata'] == meta and final['details'] == rows, 'Final/ledger/meta divergence')
            require(final['n'] == len(rows) and final['summary'] == totals(rows), 'Final count/summary mismatch')
        require(not complete or not tail, 'Completed result has uncommitted tail')
        report = dict(backend=backend, corpus=corpus, completed=complete, n=len(rows), expected_n=len(samples),
                      state='complete' if complete else 'incomplete', uncommitted_tail_ignored=tail,
                      manifest_sha256=verified_digest(str(mp)),
                      ledger_sha256=digest(ledger_path) if complete else None,
                      audited_rows_canonical_sha256=hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                      metadata_sha256=digest(meta_path), model_audit=model_audit,
                      primary=paired_metrics(base[:len(rows)], rows),
                      independent_word_dp_and_character_distances_verified=True)
        if final is not None:
            report['result_sha256'] = digest(final_path)
        reports.append(report)
        print(backend, corpus, report['state'], len(rows), '/', len(samples), flush=True)
    completed = all(r['completed'] for r in reports)
    output = dict(completed=completed, expected_runs=len(EXPECTED), completed_runs=sum(r['completed'] for r in reports),
                  partial_report=not completed, runs=reports,
                  not_supported=[dict(backend='whisper', language='ky', reason='Kyrgyz is not a native Whisper language token; no forced neighboring-language substitute')],
                  limitations=['Corpus-specific comparisons only; Kyrgyz FLEURS is read speech.',
                               'Partial metrics pair only the committed source-order prefix and are provisional.',
                               'Four Kazakh conversation recordings create clustered observations; clip counts are not independent speaker counts.',
                               'Precision and runtime differ by ready model; this evaluates deployed configurations.',
                               'Exploratory quality-run latency is confounded by concurrent workloads, including an unrelated server that must not be stopped.'],
                  auditor_sha256=digest(Path(__file__)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    require(args.allow_partial or completed, 'Incomplete comparison')


if __name__ == '__main__':
    main()
