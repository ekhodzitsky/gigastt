#!/usr/bin/env python3
"""Compare official short-API scores with exactly matching gigastt samples.

Only completed result JSON files are read. Missing groups are reported, never
silently replaced by different samples. Run again after all benchmarks finish.
"""
import argparse
import json
from pathlib import Path

from wer_unicode import normalize, word_edit_distance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--degraded', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frontend-private', type=Path, help='Optional private factorial results for exact helper/server hypothesis checks')
    args = p.parse_args()
    source = json.loads(args.source.read_text())
    comparisons, missing = [], []
    for key, group in source['groups'].items():
        language, condition = key.split('/')
        if language == 'kk_real_call':
            continue
        selected = {r['id']: r for r in source['rows'] if r['language'] == language and r['condition'] == condition}
        for variant in ['ml_ctc', 'ml_ctc_large']:
            folder = args.baseline if condition == 'original' else args.degraded
            name = f'{variant}_{language}' + ('' if condition == 'original' else '_' + condition) + '.json'
            path = folder / name
            if not path.exists():
                missing.append(name)
                continue
            result = json.loads(path.read_text())
            if not result['completed']:
                missing.append(name)
                continue
            found = {r['id']: r for r in result['details'] if r['id'] in selected}
            if found.keys() != selected.keys():
                raise ValueError(f'Missing selected IDs: {path}')
            errors = words = 0
            for ident, row in found.items():
                if row['sha256'] != selected[ident]['sha256'] or row['status'] != 'ok':
                    raise ValueError(f'Input/result mismatch: {path}: {ident}')
                ref = normalize(row['reference'])
                words += len(ref)
                errors += word_edit_distance(ref, normalize(row['hypothesis']))
            if words != group['ref_words']:
                raise ValueError('Reference word count mismatch')
            comparisons.append({'language': language, 'condition': condition, 'variant': variant,
                                'n': len(found), 'ref_words': words, 'source_errors': group['errors'],
                                'source_wer_percent': group['wer_percent'], 'gigastt_errors': errors,
                                'gigastt_wer_percent': 100 * errors / words,
                                'gigastt_result': str(path), 'source_ids': sorted(selected)})
    frontend_checks = []
    if args.frontend_private:
        for row in json.loads(args.frontend_private.read_text())['rows']:
            if row['chain'] != 'rust_pcm_rust_mel' or row['precision'] != 'int8':
                continue
            condition, language = row['condition'], row['language']
            folder = args.baseline if condition == 'original' else args.degraded
            name = f'ml_ctc_{language}' + ('' if condition == 'original' else '_' + condition) + '.json'
            path = folder / name
            check = {'id': row['id'], 'condition': condition, 'language': language}
            if not path.exists():
                frontend_checks.append({**check, 'status': 'missing_product_result'})
                continue
            result = json.loads(path.read_text())
            if not result['completed']:
                frontend_checks.append({**check, 'status': 'incomplete_product_result'})
                continue
            product = next(r for r in result['details'] if r['id'] == row['id'])
            if product['sha256'] != row['audio_sha256'] or product['status'] != 'ok':
                raise ValueError(f'Frontend input/result mismatch: {name}')
            matches = product['hypothesis'].strip() == row['hypothesis'].strip()
            frontend_checks.append({**check, 'status': 'checked', 'hypotheses_match_except_outer_whitespace': matches,
                                    'helper_word_errors': row['errors'],
                                    'product_word_errors': word_edit_distance(normalize(product['reference']), normalize(product['hypothesis'])),
                                    'ref_words': row['ref_words']})
    args.output.write_text(json.dumps({'selection': 'Identical source-success IDs and verified audio SHA256; official API length rejections excluded from both sides', 'missing_groups': missing, 'comparisons': comparisons, 'frontend_server_checks': frontend_checks}, indent=2) + '\n')
    print(f'Matched groups: {len(comparisons)}; missing: {len(missing)}')


if __name__ == '__main__':
    main()
