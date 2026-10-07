"""VPD on the edits driver's immutable manifest (mpd_battery_2951 site_edits): check that the battery
applies the manifest's operations as the driver does (per experiment its family, position and effect
KL(M_e || M) at the edited token, against an EDITS experiments file the driver scored on the same
manifest), write each form's scores as an EDITS-like file carrying the manifest, with its scores under
the manifest's native weight edits where given (mpd_battery_2951 weight_edits), and add VPD as
published to ~/mpd-data/compare/frontier_points.json as the one labelled reference point.

usage: vpd_ops_points.py BATTERY_OUT.json DRIVER_EDITS.experiments.jsonl [WEIGHTS_OUT.json]"""
import json, sys, os
out, ref = sys.argv[1], sys.argv[2]
weights = json.load(open(sys.argv[3]))['weight_edits'] if len(sys.argv) > 3 else None
r = json.load(open(out))
mine = r['site_edits']['experiments']
theirs = [json.loads(l) for l in open(ref)]
n = min(len(mine), len(theirs))
same_family = all(a['family'] == b['family'] and a['position'] == b['position'] for a, b in zip(mine[:n], theirs[:n]))
diff = max(abs(a['effect_bits_at_edited_token'] - b['effect_bits_at_edited_token']) for a, b in zip(mine[:n], theirs[:n]))
check = {'experiments': [len(mine), len(theirs)], 'same_families_and_positions': same_family, 'max_effect_difference_bits': diff}
print('manifest check', check)
D = os.path.dirname(os.path.abspath(out))
labels = {'published': 'VPD as published (CI reads the edited M, both ways)', 'causal': 'VPD, causal CI on the edited M', 'autonomous': 'VPD autonomous (causal CI on its own run)'}
points = [p for p in json.load(open('/Users/user/mpd-data/compare/frontier_points.json'))] if os.path.exists('/Users/user/mpd-data/compare/frontier_points.json') else []
points = [p for p in points if not p['label'].startswith('VPD')]
for form, label in labels.items():
    path = f'{D}/EDITS_vpd_{form}.json'
    m = r['manifest']
    record = {'sequences': m['sequences'], 'seed': m['seed'], 'edits_per_sequence': m['edits_per_sequence'], 'manifest': {'file': m['file'], 'sha256': m['sha256']},
              'families': r['site_edits'][form]['families'], 'manifest_check': check, 'source_revision': r.get('source_revision')}
    if weights:
        # As the edits driver reports an explanation's weight edits (EDITS "weights").
        record['weights'] = {'edits': weights['edits'], 'not_applicable': weights['not_applicable'], 'sequences': weights['sequences'],
                             'mean_bits_per_token': weights[form]['mean_bits_per_token'], 'ignoring_mean_bits_per_token': weights[form]['ignoring_mean_bits_per_token'],
                             'effect_mean_bits_per_token': weights['effect_mean_bits_per_token'],
                             'records': [{'operator': w['operator'], 'applicable': w['applicable'], 'mean_bits_per_token': w.get(f'{form}_mean_bits_per_token'),
                                          'ignoring_mean_bits_per_token': w.get(f'{form}_ignoring_mean_bits_per_token'), 'effect_mean_bits_per_token': w.get('effect_mean_bits_per_token')} for w in weights['records']]}
    json.dump(record, open(path, 'w'), indent=1)
    if form != 'published':
        continue
    # Executed per token, in rank-one units: every subcomponent (38,912, each computed before its
    # mask), the causal-importance network at each matrix's rank (input 2,048, 8 blocks of q, k, v,
    # o, fc1, fc2 at 2,048, head 2,048: 102,400), and the run its masks read: M's forward at its
    # rank (4 layers of q, k, v, o, c_fc, down_proj at 768: 18,432), or for the autonomous form
    # VPD's own pass with every mask 1 (38,912). Active: the 213 unmasked per token.
    read = 38912 if form == 'autonomous' else 18432
    points.append({'label': label, 'edits': path, 'executed': 38912 + 102400 + read, 'active': 213,
                   'executed_note': f"38,912 subcomponents + 102,400 CI network + {read:,} {'all-on pass' if form == 'autonomous' else 'M forward'}",
                   'description_bits': 11.4e6 + 77.2e6, 'description_note': '11.4M bits subcomponents + 77.2M bits CI network'})
json.dump(points, open('/Users/user/mpd-data/compare/frontier_points.json', 'w'), indent=1)
print('points', [p['label'] for p in points])
