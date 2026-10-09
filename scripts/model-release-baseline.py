#!/usr/bin/env python3
"""Refuse publication measured against an obsolete runtime model bundle."""
import json
from pathlib import Path
import re


def check(recipe, variant_source):
    match = re.search(r'const PREQUANT_RELEASE_BASE:\s*&str\s*=\s*"([^"]+)"', variant_source)
    if not match or recipe.get('baseline_release') != match[1]:
        raise ValueError('model release baseline must match the current runtime download pin; '
                         'update the comparison recipe and evidence before publishing a new candidate')


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    check(json.loads((root / 'benchmark/model-release.json').read_text()),
          (root / 'crates/gigastt-core/src/model/variant.rs').read_text())
    print('Model release baseline matches the current runtime bundle')
