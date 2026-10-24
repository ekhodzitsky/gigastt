import hashlib, json, random, statistics, sys
from pathlib import Path
r = Path(sys.argv[1])
timings = json.loads((r/'timings.json').read_text())
quality = json.loads((r/'quality.json').read_text())
if len(quality) != 42 or len(timings) != 168:
    raise ValueError('Expected 42 quality pairs and 168 timing requests')
analysis = {}
for head in ['rnnt', 'e2e_rnnt']:
    for sample in ['long_example', 'podlodka-306']:
        selected = [x for x in timings if (x['head'], x['sample']) == (head, sample)]
        pairs = {arm: {x['round']: x for x in selected if x['arm'] == arm} for arm in ['baseline', 'candidate']}
        if not set(pairs['baseline']) == set(pairs['candidate']) == set(range(21)):
            raise ValueError('Missing or duplicate timing pairs')
        ratios = [pairs['candidate'][i]['seconds']/pairs['baseline'][i]['seconds'] for i in range(21)]
        rng = random.Random(455)
        boot = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(10000))
        analysis[head+'/'+sample] = {
            'ratios': ratios, 'median': statistics.median(ratios),
            'bootstrap_median_95_interval': [boot[250], boot[9750]],
            'bootstrap_resamples': 10000, 'seed': 455,
            'interval_upper_at_most_1_05': boot[9750] <= 1.05,
            **{arm: {key: statistics.median(x[key] for x in pairs[arm].values())
                     for key in ['seconds', 'cpu_seconds', 'VmRSS', 'VmHWM']}
               for arm in ['baseline', 'candidate']}}
checks = []
for row in quality:
    paths = {arm: r/(row['id']+'-'+arm+'.json') for arm in ['baseline','candidate']}
    hashes = {arm: hashlib.sha256(p.read_bytes()).hexdigest() for arm,p in paths.items()}
    checks.append({'id': row['id'], **hashes, 'identical': len(set(hashes.values())) == 1})
(r/'paired-analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
(r/'response-checksums.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(analysis,indent=2))
