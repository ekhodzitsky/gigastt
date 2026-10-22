#!/usr/bin/env python3
"""Audit frozen current-head recognition and six steady rounds of CPU timing evidence."""
from collections import defaultdict
import functools
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import sys


def module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


ROOT = Path(__file__).resolve().parents[1]
common = module(ROOT / 'benchmark/common.py')
punctuation = module(ROOT / 'scripts/punctuation-heads.py')


def verify_assets(directory, expected):
    """A loadable local model is not necessarily the active release artifact."""
    result = {}
    for name, digest in expected.items():
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f'missing authoritative checksum for {name}')
        path = directory / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f'nonmatching model asset {path}: expected {digest}, got {actual}')
        result[name] = dict(sha256=actual, bytes=path.stat().st_size)
    return result


@functools.lru_cache(maxsize=32768)
def measure_pair(reference, hypothesis):
    ref, hyp = punctuation.tokenize(reference)[0], punctuation.tokenize(hypothesis)[0]
    ref_chars, hyp_chars = ' '.join(ref), ' '.join(hyp)
    normalized_ref, normalized_hyp = common.normalize_for_wer(reference), common.normalize_for_wer(hypothesis)
    errors, alignment = punctuation.align(ref, hyp)
    normalized_errors, normalized_alignment = punctuation.align(normalized_ref, normalized_hyp)
    return dict(word_errors=errors, words=len(ref),
                substitutions=sum(ref[i] != hyp[j] for i, j in alignment),
                deletions=len(ref)-len(alignment), insertions=len(hyp)-len(alignment),
                char_errors=common.word_edit_distance(ref_chars, hyp_chars), chars=len(ref_chars),
                normalized_errors=normalized_errors, normalized_words=len(normalized_ref),
                normalized_char_errors=common.word_edit_distance(' '.join(normalized_ref), ' '.join(normalized_hyp)),
                normalized_chars=len(' '.join(normalized_ref)),
                normalized_substitutions=sum(normalized_ref[i] != normalized_hyp[j] for i, j in normalized_alignment),
                normalized_deletions=len(normalized_ref)-len(normalized_alignment),
                normalized_insertions=len(normalized_hyp)-len(normalized_alignment))


def numeral_free(reference):
    """Reference-only sensitivity slice, using the benchmark's fixed numeral inventory."""
    tokens = punctuation.tokenize(reference)[0]
    number_words = {word.replace('ё', 'е') for word in common._NUMBER_WORDS}
    return not any(c.isdigit() for c in reference) and not any(word in number_words for word in tokens)


def indexed(cases, rows):
    if not cases or len({r['id'] for r in cases}) != len(cases):
        raise ValueError('empty cohort or duplicate case IDs')
    outputs = {r['id']: r for r in rows}
    if len(outputs) != len(rows) or set(outputs) != {r['id'] for r in cases}:
        raise ValueError('missing, duplicate or unknown hypotheses')
    if any(r.get('error') and r['text'] for r in rows):
        raise ValueError('failed requests must have empty hypotheses')
    return outputs


def score_cases(cases, rows):
    outputs = indexed(cases, rows)
    pairs = [measure_pair(r['reference'], outputs[r['id']]['text']) for r in cases]
    totals = {key: sum(r[key] for r in pairs) for key in pairs[0]}
    totals.update(wer=totals['word_errors']/totals['words'] if totals['words'] else None,
                  cer=totals['char_errors']/totals['chars'] if totals['chars'] else None,
                  normalized_wer=totals['normalized_errors']/totals['normalized_words'] if totals['normalized_words'] else None,
                  normalized_cer=totals['normalized_char_errors']/totals['normalized_chars'] if totals['normalized_chars'] else None,
                  failures=sum(bool(r.get('error')) for r in rows), count=len(rows),
                  empty_lexical_outputs=sum(not punctuation.tokenize(r['text'])[0] for r in rows))
    return totals


def read_report(path, cases):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if len(records) != len(cases)+2 or records[-1].get('complete') is not True or records[-1].get('count') != len(cases):
        raise ValueError(f'incomplete report: {path}')
    metadata, rows, footer = records[0], records[1:-1], records[-1]
    indexed(cases, rows)
    if any(not math.isfinite(r['seconds']) or r['seconds'] <= 0 for r in rows):
        raise ValueError(f'invalid timing: {path}')
    return metadata, rows, footer


def interval(cases, rows):
    """Descriptive bootstrap, clustering identical lexical reference text."""
    outputs = indexed(cases, rows)
    groups = defaultdict(lambda: [0, 0])
    for case in cases:
        pair = measure_pair(case['reference'], outputs[case['id']]['text'])
        key = ' '.join(punctuation.tokenize(case['reference'])[0])
        groups[key][0] += pair['normalized_errors']
        groups[key][1] += pair['normalized_words']
    groups = list(groups.values())
    rng = random.Random(42)
    estimates = []
    for _ in range(1000):
        sample = rng.choices(groups, k=len(groups))
        errors, words = map(sum, zip(*sample))
        if words:estimates.append(errors/words)
    estimates.sort()
    return dict(groups=len(groups), replicates=1000, seed=42,
                normalized_wer_95=[estimates[int(.025*len(estimates))], estimates[min(int(.975*len(estimates)),len(estimates)-1)]])


def summarize(directory):
    protocol = json.loads((directory/'protocol.json').read_text())
    runs = json.loads((directory/'run-protocol.json').read_text())['runs']
    quality_cpus = {r['head']: r.get('cpus') for r in runs if r['mode'] == 'quality'}
    cases = json.loads((directory/'cases.json').read_text())
    timing_cases = json.loads((directory/'timing-cases.json').read_text())
    result = dict(cases=len(cases), audio_seconds=sum(r['duration'] for r in cases), heads={})
    for profile in protocol['profiles']:
        head = profile['head']
        metadata, rows, footer = read_report(directory/'runs'/f'{head}-quality.jsonl', cases)
        if metadata['head'] != head:raise ValueError('wrong head in report')
        by_corpus = {}
        for name in protocol['datasets']:
            subset = [r for r in cases if r['dataset'] == name]
            ids = {r['id'] for r in subset}
            hypotheses = [r for r in rows if r['id'] in ids]
            score = score_cases(subset, hypotheses)
            score.update(interval(subset, hypotheses))
            seconds = sum(r['seconds'] for r in hypotheses)
            score.update(quality_observed_seconds=seconds, quality_observed_rtf=seconds/sum(r['duration'] for r in subset))
            digit_free = [r for r in subset if not any(c.isdigit() for c in r['reference'])]
            ids_free = {r['id'] for r in digit_free}
            score['digit_free_references'] = score_cases(digit_free, [r for r in hypotheses if r['id'] in ids_free]) if digit_free else None
            numeral_subset = [r for r in subset if numeral_free(r['reference'])]
            numeral_ids = {r['id'] for r in numeral_subset}
            score['numeral_free_references'] = score_cases(numeral_subset, [r for r in hypotheses if r['id'] in numeral_ids]) if numeral_subset else None
            if name == 'fleurs_ru_punct':
                score['formatting'] = punctuation.score(subset, hypotheses)
            by_corpus[name] = score
        timing, outputs = [], []
        duration = sum(r['duration'] for r in timing_cases)
        for i in range(7):
            meta, timed, end = read_report(directory/'runs'/f'{head}-timing-{i}.jsonl', timing_cases)
            if meta['head'] != head:raise ValueError('wrong timing head')
            if any(r.get('error') for r in timed):raise ValueError('timing request failed')
            corpus_rtf = {}
            for name in protocol['datasets']:
                ids = {r['id'] for r in timing_cases if r['dataset'] == name}
                corpus_rtf[name] = sum(r['seconds'] for r in timed if r['id'] in ids)/sum(r['duration'] for r in timing_cases if r['id'] in ids)
            timing.append(dict(round=i, seconds=sum(r['seconds'] for r in timed), rtf=sum(r['seconds'] for r in timed)/duration,
                               corpus_rtf=corpus_rtf, load_seconds=meta['load_seconds'], warmup_seconds=meta['warmup_seconds'],
                               ready_memory=meta['ready_memory'], peak_rss_kib=end['memory']['peak_rss_kib']))
            outputs.append({r['id']:r['text'] for r in timed})
        consistency = sum(any(out[key] != outputs[0][key] for out in outputs[1:]) for key in outputs[0])
        quality_outputs = {r['id']:r['text'] for r in rows}
        steady = timing[1:]
        result['heads'][head] = dict(builtin_punctuation=profile['builtin_punctuation'],
            model_mib=sum(v['bytes'] for v in profile['assets'].values())/1048576,
            quality_metadata=metadata, quality_cpu_affinity=quality_cpus.get(head),
            quality_memory=footer['memory'], corpus=by_corpus,
            timing_rounds=timing, timing_conditioning_round=0, timing_steady_rounds=list(range(1,7)),
            timing_ready_rss_mib=statistics.median(r['ready_memory']['rss_kib'] for r in steady)/1024,
            timing_peak_warm_mib=max(r['peak_rss_kib'] for r in steady)/1024,
            warm_load_seconds_median=statistics.median(r['load_seconds'] for r in steady),
            timing_corpus_rtf_median={name:statistics.median(r['corpus_rtf'][name] for r in steady) for name in protocol['datasets']},
            timing_rtf_median=statistics.median(r['rtf'] for r in steady),
            timing_rtf_range=[min(r['rtf'] for r in steady),max(r['rtf'] for r in steady)],
            nondeterministic_timing_cases=consistency,
            quality_timing_text_differences=sum(quality_outputs[key] != outputs[0][key] for key in outputs[0]))
    return result


if __name__ == '__main__':
    print(json.dumps(summarize(Path(sys.argv[1])), ensure_ascii=False, indent=2))
