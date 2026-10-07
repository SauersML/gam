"""VPD on the shared manifest (mpd_battery_2951 site_edits): check its experiments against the edits
driver's (the same draws: per experiment its family, position and effect KL(M_e || M) at the edited
token), write each form's scores as an EDITS-like file, and add the three VPD points to
~/mpd-data/compare/frontier_points.json.

usage: vpd_ops_points.py BATTERY_OUT.json REFERENCE_EDITS.experiments.jsonl"""
import json, sys, os
out, ref = sys.argv[1], sys.argv[2]
r = json.load(open(out))
mine = r['site_edits']['experiments']
theirs = [json.loads(l) for l in open(ref)]
n = min(len(mine), len(theirs))
same_family = all(a['family'] == b['family'] and a['position'] == b['position'] for a, b in zip(mine[:n], theirs[:n]))
diff = max(abs(a['effect_bits_at_edited_token'] - b['effect_bits_at_edited_token']) for a, b in zip(mine[:n], theirs[:n]))
check = {'experiments': [len(mine), len(theirs)], 'same_families_and_positions': same_family, 'max_effect_difference_bits': diff}
print('manifest check', check)
D = os.path.dirname(out)
labels = {'published': 'VPD as published (CI reads the edited M, both ways)', 'causal': 'VPD, causal CI on the edited M', 'autonomous': 'VPD autonomous (causal CI on its own run)'}
points = [p for p in json.load(open('/Users/user/mpd-data/compare/frontier_points.json'))] if os.path.exists('/Users/user/mpd-data/compare/frontier_points.json') else []
points = [p for p in points if not p['label'].startswith('VPD')]
for form, label in labels.items():
    path = f'{D}/EDITS_vpd_{form}.json'
    json.dump(dict(r['manifest'], families=r['site_edits'][form]['families'], manifest_check=check, source_revision=r.get('source_revision')), open(path, 'w'), indent=1)
    # Executed per token: every subcomponent (38,912) plus the CI network; active: the masked 213.
    points.append({'label': label, 'edits': path, 'executed': 38912, 'executed_note': 'all 38,912 subcomponents plus the causal-importance network', 'active': 213,
                   'description_bits': 11.4e6 + 77.2e6, 'description_note': '11.4M subcomponents + 77.2M CI network (vpdstart)'})
json.dump(points, open('/Users/user/mpd-data/compare/frontier_points.json', 'w'), indent=1)
print('points', [p['label'] for p in points])
