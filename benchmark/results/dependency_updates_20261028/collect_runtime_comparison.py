"""Exploratory comparison: retain changes for review, never claim parity from completion."""
import sys
sys.path.insert(0, '/home/ekhodzitsky/Projects/gigastt-oct-deps/scripts')
import benchmark_stitch_case as benchmark

def changes(base, candidate):
    return [{'field': key, 'baseline': base.get(key), 'candidate': candidate.get(key)}
            for key in base.keys() | candidate.keys() if base.get(key) != candidate.get(key)]
benchmark.delta = changes
benchmark.main()
