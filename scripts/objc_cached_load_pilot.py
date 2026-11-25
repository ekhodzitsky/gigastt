"""Validation and fixed sampling schedule for the optional Core ML pilot."""
import math
import hashlib
import re
import shutil
import statistics

ROUNDS = 7
REPEATED_LOADS = 5
LIMIT = 1.05
GROUPS = ('ab', 'aa')
ARMS = ('base', 'candidate')
MODES = ('cpu', 'cpu_and_ne')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def snapshot(root):
    """Content identity across copies; filesystem identity within each arm."""
    require(root.is_dir(), f'missing compiled cache: {root}')
    content, identity = {}, {}
    for path in (root, *sorted(root.rglob('*'))):
        name = path.relative_to(root).as_posix()
        stat = path.lstat()
        identity[name] = [stat.st_ino, stat.st_size, stat.st_mtime_ns]
        if path.is_symlink():
            # External links would not be a self-contained cache copy.
            require(path.resolve().is_relative_to(root.resolve()), f'external cache symlink: {name}')
            content[name] = {'symlink': str(path.readlink())}
        elif path.is_file():
            content[name] = {'sha256': file_sha256(path)}
        else:
            require(path.is_dir(), f'unsupported cache entry: {name}')
            content[name] = {'directory': True}
    require(any('sha256' in entry for entry in content.values()), f'empty compiled cache: {root}')
    return {'content': content, 'identity': identity}


def copy_cache(source, target):
    expected = snapshot(source)['content']
    shutil.copytree(source, target, symlinks=True)
    require(snapshot(target)['content'] == expected, 'copied cache differs')


def schedule():
    """Interleave A/B and A/A, reverse both group and arm order each round."""
    for phase in (*MODES, 'key'):
        for index in range(ROUNDS):
            groups = GROUPS if index % 2 == 0 else GROUPS[::-1]
            arms = ARMS if index % 2 == 0 else ARMS[::-1]
            for group in groups:
                for arm in arms:
                    yield phase, index, group, arm


def timing(value):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0, 'invalid pilot timing')
    return value


def summarize(values):
    ratios = [candidate / base for base, candidate in values]
    return {'paired_ratios': ratios, 'median_ratio': statistics.median(ratios),
            'range_ratio': [min(ratios), max(ratios)],
            'median_base_seconds': statistics.median(a for a, _ in values),
            'median_candidate_seconds': statistics.median(b for _, b in values)}


def validate(report):
    expected = list(schedule())
    actual = [(row['phase'], row['round'], row['group'], row['arm']) for row in report['samples']]
    require(actual == expected, 'missing, duplicated or reordered pilot rounds')
    binaries = report['binary_sha256']
    require(all(re.fullmatch('[0-9a-f]{64}', value) is not None for value in binaries.values()),
            'invalid executable hash')
    require(binaries['aa_base'] == binaries['aa_candidate'] == binaries['ab_base'],
            'A/A executable identity mismatch')
    before, after = report['cache_before'], report['cache_after']
    for group in GROUPS:
        for arm in ARMS:
            name = f'{group}_{arm}'
            require(before[name] == after[name], f'cache changed during measurement: {name}')
    require(before['aa_base']['content'] == before['aa_candidate']['content']
            == before['ab_base']['content'], 'A/A cache content mismatch')
    require(before['aa_base']['identity']['.'] != before['aa_candidate']['identity']['.'],
            'A/A requires distinct copied caches')
    for row in report['samples']:
        name = f"{row['group']}_{row['arm']}"
        require(row['binary_sha256'] == binaries[name], 'sample executable changed')
    rows = {(row['phase'], row['round'], row['group'], row['arm']): row['data']
            for row in report['samples']}
    result = {}
    for group in GROUPS:
        metrics = {}
        for mode in MODES:
            for name, get in [('first', lambda x: x[0]),
                              ('repeated_median', lambda x: statistics.median(x[1:]))]:
                pairs = []
                for index in range(ROUNDS):
                    pair = []
                    for arm in ARMS:
                        loads = rows[(mode, index, group, arm)]['load_seconds']
                        require(len(loads) == REPEATED_LOADS + 1, 'missing repeated loads')
                        pair.append(get([timing(value) for value in loads]))
                    pairs.append(pair)
                metrics[f'{mode}/{name}'] = summarize(pairs)
        for metric in ('key_seconds', 'meta_seconds'):
            metrics[metric] = summarize([
                [timing(rows[('key', index, group, arm)][metric]) for arm in ARMS]
                for index in range(ROUNDS)
            ])
        result[group] = metrics
    # Seven paired ratios give a conservative min/max diagnostic range.
    # It is not a validated confidence interval on a shared runner. In
    # particular, a stable-looking median cannot excuse large control spread.
    qualification = {}
    control_within_limit = {}
    cache_equal = before['ab_base']['content'] == before['ab_candidate']['content']
    for name in result['ab']:
        aa_low, aa_high = result['aa'][name]['range_ratio']
        ab_low, ab_high = result['ab'][name]['range_ratio']
        control_within_limit[name] = sum(
            1 / LIMIT <= ratio <= LIMIT for ratio in result['aa'][name]['paired_ratios'])
        if aa_low < 1 / LIMIT or aa_high > LIMIT:
            qualification[name] = 'unstable_control'
        elif not cache_equal:
            qualification[name] = 'different_ab_caches'
        elif ab_low > LIMIT:
            qualification[name] = 'regression'
        elif ab_high > LIMIT:
            qualification[name] = 'inconclusive'
        else:
            qualification[name] = 'within_limit'
    return {'metrics': result, 'qualification': qualification,
            'control_pairs_within_limit': control_within_limit,
            'qualification_note': 'Conservative range diagnostic; independence is unproven. '
                                  'No change to the required revision gate.',
            'ab_cache_content_equal': cache_equal}
