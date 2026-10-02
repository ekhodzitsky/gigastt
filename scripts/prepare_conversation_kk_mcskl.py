#!/usr/bin/env python3
"""Freeze first25 eligible human-annotated IUs from each MCSKL030–033.

Preparation only: source PCM frames are sliced without resampling/filtering.
The full source ZIP must already have been downloaded to the task cache.
"""
import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import random
import wave
import xml.etree.ElementTree as ET

ARCHIVE_SHA = 'c3368b896bb829ccda96444a297cd8cae37678a1857e4ebc6bf66fa0ffc5c7fd'
REVISION = 'ada9282bb3720eef7001f4f1fc4060f7541b2b31'
TSV_SHA = {
    'MCSKL030': 'b132da4c2525adbad807198381fdc67e372e8e9b3cf2cd7d87b67ce4f33ae222',
    'MCSKL031': '33dc1c45609e64fea9f438c7352cc755c86524067909cac480911389532e160f',
    'MCSKL032': 'eec403e54d4dd27f6cd9720c317c9582c14c4a316c53cabf7fa145c9291c2353',
    'MCSKL033': '2d67bb2d8fef75ba5dd87a52a43af296334df2686c00a770c68885b9392c2a10',
}
NONLEXICAL_FORMS = {'(H)', '(Hx)', '(TSK)', '<lag>'}
NONLEXICAL_TYPES = {'inhale', 'exhale', 'click', 'paraverbal'}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def units(path):
    groups = defaultdict(list)
    with path.open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            groups[row['iu_id']].append(row)
    result = []
    for key, rows in groups.items():
        if len({r['speaker'] for r in rows}) != 1:
            raise ValueError('IU contains multiple speaker annotations')
        times = defaultdict(list)
        for row in rows:
            for kind, seconds in re.findall(r'(Begin|End)=([0-9.]+)', row['align']):
                times[kind].append(float(seconds))
        if len(times['Begin']) != 1 or len(times['End']) != 1:
            raise ValueError('Missing/ambiguous timestamp in IU ' + key)
        clean = [r['form'] for r in rows if r['type'] not in NONLEXICAL_TYPES
                 and r['form'] not in NONLEXICAL_FORMS and not re.fullmatch('@+', r['form'])]
        result.append({'iu_id': key, 'speaker': rows[0]['speaker'], 'start_s': times['Begin'][0],
                       'end_s': times['End'][0], 'rows': rows, 'reference': ' '.join(clean),
                       'removed_nonlexical_tokens': len(rows) - len(clean)})
    return sorted(result, key=lambda x: (x['start_s'], x['end_s'], int(x['iu_id'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expanded', action='store_true', help='Pre-inference amendment: seeded up to250 IUs per source and complete EAF overlap coverage')
    args = parser.parse_args()
    root = Path.home() / '.cache/gigastt-conversation/kk'
    source = root / 'mcskl'
    destination = root / 'mcskl_segments'
    destination.mkdir(exist_ok=True)
    output = Path('benchmark/results/multilingual_conversation_20261001')
    if digest(root / 'MCSKL-30-33.zip') != ARCHIVE_SHA:
        raise ValueError('OSF source archive changed')
    samples, source_reports = [], []
    for recording in TSV_SHA:
        tsv = source / 'tsv' / (recording + '.vert.tsv')
        if digest(tsv) != TSV_SHA[recording]:
            raise ValueError('Human transcription changed')
        candidates = units(tsv)
        eaf_path = source / 'eaf' / (recording + '.eaf')
        eaf = ET.parse(eaf_path).getroot()
        time_slots = {x.attrib['TIME_SLOT_ID']: int(x.attrib['TIME_VALUE']) / 1000
                      for x in eaf.findall('.//TIME_SLOT')}
        complete_intervals = []
        for tier in eaf.findall('TIER'):
            if tier.attrib.get('LINGUISTIC_TYPE_REF') != 'Intonation Units':
                continue
            for annotation in tier.findall('.//ALIGNABLE_ANNOTATION'):
                complete_intervals.append({'speaker': tier.attrib['TIER_ID'].split('@', 1)[1],
                                           'start_s': time_slots[annotation.attrib['TIME_SLOT_REF1']],
                                           'end_s': time_slots[annotation.attrib['TIME_SLOT_REF2']]})
        interval_keys = {(x['speaker'], x['start_s'], x['end_s']) for x in complete_intervals}
        audio = next(p for p in (source / 'audio').iterdir() if p.stem == recording)
        audio_sha = digest(audio)
        reasons = Counter()
        rejected, eligible = [], []
        with wave.open(str(audio)) as reader:
            params = reader.getparams()
            audio_duration = reader.getnframes() / params.framerate
            for row in candidates:
                why = []
                if not 1 <= row['end_s'] - row['start_s'] <= 30:
                    why.append('duration_outside_1_30_seconds')
                if row['start_s'] < 0 or row['end_s'] > audio_duration:
                    why.append('outside_audio')
                if (row['speaker'], row['start_s'], row['end_s']) not in interval_keys:
                    why.append('tsv_eaf_alignment_mismatch')
                if any(r['type'] == 'unknown' or re.fullmatch(r'Q{2,}', r['form'])
                       or '#' in r['form'] or '???' in r['form'] for r in row['rows']):
                    why.append('unknown_or_placeholder')
                if any('Anonymized=Yes' in r['features'] for r in row['rows']):
                    why.append('anonymized')
                if any(re.search(r'[<>@\[\]]', token) for token in row['reference'].split()):
                    why.append('unexplained_residual_transcription_markup')
                if not any(c.isalpha() for c in row['reference']):
                    why.append('nonlexical_or_missing_reference')
                if any(other['speaker'] != row['speaker']
                       and max(other['start_s'], row['start_s']) < min(other['end_s'], row['end_s'])
                       for other in complete_intervals):
                    why.append('other_speaker_annotation_overlap')
                if why:
                    reasons.update(why)
                    rejected.append({'iu_id': row['iu_id'], 'reasons': why})
                else:
                    eligible.append(row)
            if args.expanded:
                selected_ids = {x['iu_id'] for x in random.Random(42).sample(eligible, min(250, len(eligible)))}
                chosen = [x for x in eligible if x['iu_id'] in selected_ids]
            else:
                chosen = eligible[:25]
            for row in chosen:
                key = recording.lower() + '_iu_' + row['iu_id']
                path = destination / (key + '.wav')
                start = round(row['start_s'] * params.framerate)
                end = round(row['end_s'] * params.framerate)
                reader.setpos(start)
                pcm = reader.readframes(end - start)
                if len(pcm) != (end - start) * params.nchannels * params.sampwidth:
                    raise ValueError('Truncated source PCM')
                with wave.open(str(path), 'wb') as writer:
                    writer.setparams(params)
                    writer.writeframes(pcm)
                with wave.open(str(path)) as verify:
                    if verify.readframes(verify.getnframes()) != pcm:
                        raise ValueError('Lossless segment verification failed')
                samples.append({'id': key, 'language': 'kk', 'dataset': 'mcskl_natural_conversation',
                                'split': 'module1_seed42_up_to250_per_source' if args.expanded else 'module1_fixed_first25_per_source', 'path': str(path),
                                'sha256': digest(path), 'reference': row['reference'], 'condition': 'original',
                                'duration_s': (end - start) / params.framerate, 'recording_id': recording,
                                'speaker_id': row['speaker'], 'iu_id': row['iu_id'],
                                'speech_type': 'online_interview' if recording == 'MCSKL032' else 'friends_conversation',
                                'source': 'https://osf.io/6zjdq/', 'source_audio_sha256': audio_sha,
                                'source_tsv_sha256': TSV_SHA[recording], 'start_frame': start, 'end_frame': end,
                                'start_s': row['start_s'], 'end_s': row['end_s'],
                                'sample_rate': params.framerate, 'channels': params.nchannels,
                                'sample_width_bytes': params.sampwidth,
                                'removed_nonlexical_tokens': row['removed_nonlexical_tokens'],
                                'reference_token_ids': [r['token_id'] for r in row['rows']]})
        source_reports.append({'recording_id': recording, 'audio_sha256': audio_sha,
                               'audio_duration_s': audio_duration, 'sample_rate': params.framerate,
                               'channels': params.nchannels, 'sample_width_bytes': params.sampwidth,
                               'annotation_units': len(candidates), 'eligible': len(eligible), 'selected': len(chosen),
                               'all_eaf_iu_intervals_including_empty': len(complete_intervals),
                               'eaf_sha256': digest(eaf_path),
                               'eligible_not_selected_quota': len(eligible) - len(chosen),
                               'exclusion_reason_counts_nonexclusive': dict(reasons), 'excluded_units': rejected})
    selection = [{k: s[k] for k in ['id', 'source_audio_sha256', 'source_tsv_sha256', 'start_frame', 'end_frame', 'reference']} for s in samples]
    selection_sha = hashlib.sha256(json.dumps(selection, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    provenance = {'source': 'https://osf.io/6zjdq/', 'source_audio_archive_sha256': ARCHIVE_SHA,
                  'github_revision': REVISION, 'github_tsv_matches_osf_release_archive': True,
                  'human_reference_evidence': 'https://reference-global.com/article/10.5334/johd.529?tab=article',
                  'human_reference_description': 'Native-speaker manual ELAN transcripts; senior Kazakh-speaking reviewers checked transcription. Orthographic form column from publisher TSV, not translations or ASR hypotheses.',
                  'license': 'CC BY-NC-SA 4.0; attribution: Giorgia Troiani, Andrey Filchenko and MCSKL annotation team. Non-commercial research dataset; not a commercial data license.',
                  'selection_rule': ('Before inference: random.Random(42) independently per source samples min(250, eligible count), then restore time order.' if args.expanded else 'Before inference: first25 eligible time-ordered IUs per source.') + ' Sources MCSKL030–033; duration1–30s; no overlap with ANY other-speaker EAF IU including empty annotations. Reject unknown/placeholders/anonymized/nonlexical-only, unexplained residual angle-bracket/laughter/bracket markup, out-of-audio and TSV/EAF timestamp mismatch. No outcome-based replacements.',
                  'nonlexical_forms_removed': sorted(NONLEXICAL_FORMS) + ['@+ (laughter-only token)'],
                  'nonlexical_types_removed': sorted(NONLEXICAL_TYPES),
                  'reference_handling': 'Join publisher orthographic forms; retain spoken fragments and Russian/English insertions. No transliteration, translation, alternate-reference selection or language-based exclusion.',
                  'audio_handling': 'Keep original sample rate, channel count and sample width; all source channels sliced together at annotation boundaries. No channel-to-speaker assumption. Byte-exact PCM verified, only WAV container rewritten.',
                  'limitation': 'Cleanly segmented non-overlapping utterances from four recordings; not full-conversation diarization or a population-representative quality estimate. English/Russian code switching occurs; language is corpus label.',
                  'n': len(samples), 'duration_s': sum(s['duration_s'] for s in samples),
                  'hours': sum(s['duration_s'] for s in samples) / 3600,
                  'selection_sha256': selection_sha, 'preparation_script_sha256': digest(Path(__file__)),
                  'sources': source_reports}
    prefix = 'kk_conversations_expanded' if args.expanded else 'kk_conversations'
    if args.expanded:
        provenance['pre_inference_amendment'] = {'reason': 'The initial100 IUs had only151.22s of speech; expand coverage within the same four recordings before any MCSKL inference.',
                                               'superseded_manifest_sha256': digest(output / 'kk_conversations_manifest.json'),
                                               'superseded_measured': False,
                                               'overlap_audit_correction': 'Initial TSV-only overlap check missed two empty EAF IUs. Expanded selection uses all EAF Intonation Units, including empty ones.',
                                               'markup_audit_correction': 'Before inference, pre-audit expanded selection SHA7d9cc8ab96c9f4009de80dfc7c1058f1d34b7d12fc3e7c0fb87a765a594c71d7 contained one residual <rush> reference. Unknown residual markup is now an explicit reference-eligibility exclusion; original pre-audit artifact preserved in private cache.',
                                               'no_model_outcomes_used': True}
    for name, value in [(prefix + '_manifest.json', {**provenance, 'samples': samples}),
                        (prefix + '_provenance.json', provenance)]:
        p = output / name
        text = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
        if p.exists() and p.read_text() != text:
            raise FileExistsError('Existing frozen selection differs')
        p.write_text(text)
    print(json.dumps({'n': len(samples), 'duration_s': provenance['duration_s'], 'selection_sha256': selection_sha,
                      'manifest_sha256': digest(output / (prefix + '_manifest.json'))}, indent=2))


if __name__ == '__main__':
    main()
