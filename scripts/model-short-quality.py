#!/usr/bin/env python3
"""Validate saved Golos transcripts against the exact proposed model bundle.

Long-form gains cannot conceal short-form regressions. Model publication needs
complete short-form evidence with reproducible scores and matching model bytes.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmark'))
from common import compute_wer, compute_wer_naive


def evidence_digest(report, head):
    """Bind an explicit trade-off to model bytes, references and every transcript."""
    evidence = dict(manifest_sha256=report['manifest_sha256'], models=report['models'][head],
                    transcripts={arm: report['results'][head][arm]['details']
                                 for arm in ('baseline', 'candidate')})
    return hashlib.sha256(json.dumps(evidence, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check(report, manifest_path, release):
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    samples = [sample for sample in manifest['samples'] if sample['reference'].strip()]
    references = {sample['filename']: sample['reference'] for sample in samples}
    if (report.get('schema') != 1 or report.get('wer_unit') != 'fraction'
            or report.get('manifest_sha256') != hashlib.sha256(manifest_bytes).hexdigest()
            or report.get('samples') != len(samples)
            or report.get('nominal_samples') != len(manifest['samples'])
            or report.get('skipped_empty_references') != len(manifest['samples']) - len(samples)
            or len(references) != len(samples) or not samples
            or set(report.get('audio_sha256', {})) != set(references)
            or any(not re.fullmatch('[0-9a-f]{64}', value) for value in report['audio_sha256'].values())):
        raise ValueError('incomplete or mismatched short-form corpus')
    failures = []
    for head, recipe in release['heads'].items():
        totals = {}
        encoder = f'v3_{head}_encoder_int8.onnx'
        companions = {name: value for name, value in recipe['source_files'].items()
                      if name != f'v3_{head}_encoder.onnx'}
        for arm in ('baseline', 'candidate'):
            expected = {**companions, encoder: recipe[f'{arm}_encoder_sha256']}
            if report['models'][head][arm]['files'] != expected:
                raise ValueError(f'{head}/{arm}: evidence does not measure the proposed model files')
            result = report['results'][head][arm]
            rows = {row['file']: row for row in result['details']}
            if len(rows) != len(result['details']) or set(rows) != set(references):
                raise ValueError(f'{head}/{arm}: incomplete or duplicate transcripts')
            for prefix, scorer in [('', compute_wer), ('raw_', compute_wer_naive)]:
                errors = words = 0
                for name, row in rows.items():
                    _, edits, count = scorer(references[name], row['text'])
                    if row[prefix + 'errors'] != edits or row[prefix + 'reference_words'] != count:
                        raise ValueError(f'{head}/{arm}/{name}: saved scores disagree with transcript')
                    errors += edits
                    words += count
                if (not words or result[prefix + 'errors'] != errors
                        or result[prefix + 'reference_words'] != words
                        or not math.isclose(result[prefix + 'wer'], errors / words, abs_tol=1e-12)):
                    raise ValueError(f'{head}/{arm}: incorrect aggregate score')
                if not prefix:
                    totals[arm] = errors
        if totals['candidate'] > totals['baseline']:
            approval = release.get('short_form_transition', {}).get(head, {})
            approved = (approval.get('evidence_sha256') == evidence_digest(report, head)
                        and totals['baseline'] == approval.get('baseline_errors')
                        and totals['candidate'] <= approval.get('max_candidate_errors', -1))
            if not approved:
                failures.append(f"{head}: Golos word errors increased from {totals['baseline']} to {totals['candidate']}")
    return failures


def main():
    release = json.loads((ROOT / 'benchmark/model-release.json').read_text())
    report = json.loads((ROOT / release['short_form_evidence']).read_text())
    failures = check(report, ROOT / 'benchmark/manifests/golos_crowd_1k.json', release)
    for failure in failures:
        print(failure, file=sys.stderr)
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
