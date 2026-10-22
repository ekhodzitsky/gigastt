#!/usr/bin/env python3
"""Alignment-aware punctuation evaluation on frozen ASR hypotheses.

Word Levenshtein alignment maps punctuation attached to the preceding word.
Substitutions preserve word-end boundaries; deletions/insertions cannot match.
Capital letters match only inside lexically equal aligned words. This separates
recognition WER from formatting and prevents error-induced global position shifts.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

MARKS = set('.,!?;:—…«»"()')
WORDS = re.compile(r'[^\W_]+', re.UNICODE)


def normalize(word):
    return word.lower().replace('ё', 'е')


def tokenize(text):
    spans = list(WORDS.finditer(text))
    words = [normalize(m.group()) for m in spans]
    punctuation, capitals = Counter(), Counter()
    for i, span in enumerate(spans):
        for offset, char in enumerate(span.group()):
            if char.isupper():
                capitals[(i, offset)] += 1
        end = spans[i + 1].start() if i + 1 < len(spans) else len(text)
        for char in text[span.end():end]:
            if char in MARKS:
                punctuation[(i, char)] += 1
    for char in text[:spans[0].start()] if spans else text:
        if char in MARKS:
            punctuation[(-1, char)] += 1
    return words, punctuation, capitals


def align(reference, hypothesis):
    matrix = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]
    for i in range(len(reference) + 1):
        matrix[i][0] = i
    for j in range(len(hypothesis) + 1):
        matrix[0][j] = j
    for i, ref in enumerate(reference, 1):
        for j, hyp in enumerate(hypothesis, 1):
            matrix[i][j] = min(matrix[i-1][j] + 1, matrix[i][j-1] + 1,
                               matrix[i-1][j-1] + (ref != hyp))
    i, j, pairs = len(reference), len(hypothesis), []
    while i or j:
        if i and j and matrix[i][j] == matrix[i-1][j-1] + (reference[i-1] != hypothesis[j-1]):
            pairs.append((i-1, j-1))
            i, j = i-1, j-1
        elif i and matrix[i][j] == matrix[i-1][j] + 1:
            i -= 1
        else:
            j -= 1
    return matrix[-1][-1], pairs


def counts(reference, hypothesis):
    tp = sum((reference & hypothesis).values())
    return {'tp': tp, 'fp': sum(hypothesis.values()) - tp, 'fn': sum(reference.values()) - tp}


def score_pair(reference, hypothesis):
    rw, rp, rc = tokenize(reference)
    hw, hp, hc = tokenize(hypothesis)
    errors, pairs = align(rw, hw)
    mapping = {j: i for i, j in pairs}
    mapping[-1] = -1
    punct = Counter({(mapping.get(j, ('inserted', j)), mark): n for (j, mark), n in hp.items()})
    exact = {j: i for i, j in pairs if rw[i] == hw[j]}
    caps = Counter({(exact.get(j, ('unmatched', j)), offset): n for (j, offset), n in hc.items()})
    matched_reference = set(exact.values())
    matched_reference.add(-1)
    matched_hypothesis = set(exact)
    matched_hypothesis.add(-1)
    def filtered(events, anchors):
        return Counter({key: value for key, value in events.items() if key[0] in anchors})
    return {'errors': errors, 'words': len(rw), 'punctuation': counts(rp, punct), 'casing': counts(rc, caps),
            'matched_punctuation': counts(filtered(rp, matched_reference),
                                          Counter({(mapping[j], mark): n for (j, mark), n in hp.items() if j in matched_hypothesis})),
            'matched_casing': counts(filtered(rc, matched_reference), filtered(caps, matched_reference))}


def aggregate(rows):
    result = {key: sum(r[key] for r in rows) for key in ['errors', 'words']}
    result['wer'] = result['errors'] / result['words']
    for metric in ['punctuation', 'casing', 'matched_punctuation', 'matched_casing']:
        values = {key: sum(r[metric][key] for r in rows) for key in ['tp', 'fp', 'fn']}
        tp, fp, fn = (values[key] for key in ['tp', 'fp', 'fn'])
        values['f1'] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
        result[metric] = values
    return result


def score(cases, rows):
    outputs = {row['id']: row['text'] for row in rows}
    if len(outputs) != len(rows) or set(outputs) != {r['id'] for r in cases}:
        raise ValueError('missing, duplicate or unexpected hypotheses')
    return aggregate([score_pair(row['reference'], outputs[row['id']]) for row in cases])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', type=Path)
    parser.add_argument('hypotheses', type=Path)
    args = parser.parse_args()
    document = json.loads(args.hypotheses.read_text())
    print(json.dumps(score(json.loads(args.cases.read_text()), document['rows']), indent=2))
