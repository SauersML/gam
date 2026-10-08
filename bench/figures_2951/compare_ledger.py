"""One ledger per arm against VPD as published (#2951, compare), whole model and MLP only (M's
attention), every number from the one scorer (the edits driver's reports for our arms, the battery's
for VPD, the same held-out sequences and manifests):
- hard KL on held-out text (the activation-edit run's clean experiments, 32 sequences);
- active bits per token (descent's rule: each active part's description bits; VPD's subcomponents
  priced by the same rule) and parts on per token;
- library bits (VPD: its causal-importance network, 77.2M, plus its subcomponents, 11.4M);
- F = library bits / N + hard KL at N = 2^24 and 2^30 tokens;
- the all-on error (every part on; VPD every mask 1, the remainder dropped);
- the activation-edit error (per family of the shared operations) and the strong weight-edit error
  (gap and edit-ignoring baseline over the edits both the arm and VPD support).
Each of our arms is set beside VPD's form of its scope: an arm running M's attention beside VPD's
MLP-only form. usage: compare_ledger.py [OUT.json]"""
import json, os, sys
L = lambda p: json.load(open(p)) if p and os.path.exists(p) else None
MANIFEST = '/Users/user/mpd-data/compare/manifest'
POINTS = '/Users/user/mpd-data/compare/frontier_points.json'
ARMS = '/Users/user/mpd-data/compare/arms'
# VPD's active bits per token under descent's rule (budget_descent.py's K_t at K = VPD's count):
# MLP only, 129 subcomponents; whole model, 213, once descent's whole-model rule gives it.
VPD_ACTIVE_BITS = {'mlp': 3284863.883754805, 'whole': None}
FAMILIES = ('cut', 'push', 'scale', 'swap', 'zero')


def vpd(form, scope):
    s2 = (L(f'{MANIFEST}/vpd_s2_mlp/vpd_s2.json') or {}).get('site_edits', {}).get(form, {}).get('families', {})
    s3 = (L(f'{MANIFEST}/vpd_s3_allon/vpd_s3_weights.json') or L(f'{MANIFEST}/vpd_s3_mlp/vpd_s3_weights.json') or {}).get('weight_edits', {})
    records = [r for r in s3.get('records', []) if r.get('applicable')]
    return {'arm': f"VPD as published, {'MLP only' if scope == 'mlp' else 'whole model'}", 'scope': scope,
            'hard_kl': s2.get('clean', {}).get('mean_bits_per_token'),
            'active_bits': VPD_ACTIVE_BITS[scope], 'parts': 129 if scope == 'mlp' else 213,
            'library_bits': 77.2e6 + 11.4e6,
            'all_on': s3.get('all_on_mlp_mean_bits_per_token' if scope == 'mlp' else 'all_on_mean_bits_per_token'),
            'activation_edits': {f: s2.get(f, {}).get('mean_bits_per_token') for f in FAMILIES},
            'weight_records': [(r['family'], r.get(f'{form}_mean_bits_per_token'), r.get(f'{form}_ignoring_mean_bits_per_token')) for r in records],
            'weight_index': [i for i, r in enumerate(s3.get('records', [])) if r.get('applicable')]}


def ours(point):
    entry = L(point.get('arm_entry'))
    if not entry:
        return None
    log = L(entry['log']) or {}
    at = [t for t in log.get('trace', []) if t.get('step') == point.get('step')]
    t = at[-1] if at else {}
    s2 = L(point.get('edits')) or {}
    fam = s2.get('families') if isinstance(s2.get('families'), dict) else {}
    s3 = (L(point.get('weights_edits')) or {}).get('weights', {})
    records = s3.get('records', [])
    dst = os.path.dirname(point['edits'])
    parity = L(f'{dst}/parity.json') or {}
    rot = t.get('rot')
    return {'arm': point['label'], 'scope': 'whole' if entry.get('blocks', [1]) is None else 'mlp', 'step': point.get('step'),
            'hard_kl': fam.get('clean', {}).get('mean_bits_per_token'),
            # Bits per token where the run reports its budget in bits (rot), else not comparable.
            'active_bits': t.get('active') if rot is not None or t.get('K_t', 0) > 1e5 else None,
            'parts': rot['blocks_on'] if rot else t.get('active'),
            'library_bits': t.get('description_bits'),
            'all_on': parity.get('rust_all_on_bits_per_token') if str(parity.get('stamp')) in point['edits'] else None,
            'activation_edits': {f: fam.get(f, {}).get('mean_bits_per_token') for f in FAMILIES},
            'weight_records': [(r['family'], r.get('mean_bits_per_token'), r.get('ignoring_mean_bits_per_token')) for r in records if r.get('supported', r.get('applicable'))],
            'weight_index': [i for i, r in enumerate(records) if r.get('supported', r.get('applicable'))]}


def weight_error(arm, common):
    keep = [rec for i, rec in zip(arm['weight_index'], arm['weight_records']) if i in common]
    n = len(keep)
    return (n, sum(r[1] for r in keep) / n, sum(r[2] for r in keep) / n) if n else (0, None, None)


def F(arm, n):
    return arm['library_bits'] / n + arm['hard_kl'] if arm.get('library_bits') and arm.get('hard_kl') is not None else None


refs = {'mlp': vpd('published_mlp', 'mlp'), 'whole': vpd('published', 'whole')}
arms = [a for a in (ours(p) for p in (L(POINTS) or []) if p.get('arm_entry') and os.path.exists(p['arm_entry'])) if a]
ledger = []
fmt = lambda v, d=3: '-' if v is None else (f'{v:.1e}' if isinstance(v, float) and abs(v) < 1e-2 else f'{v:.{d}f}' if isinstance(v, float) and abs(v) < 1e4 else f'{v:.4g}')
for arm in arms:
    ref = refs[arm['scope']]
    common = set(arm['weight_index']) & set(ref['weight_index'])
    rows = []
    for a in (ref, arm):
        n, gap, ign = weight_error(a, common)
        rows.append({'arm': a['arm'], 'step': a.get('step'), 'hard_kl': a['hard_kl'], 'active_bits_per_token': a['active_bits'], 'parts_per_token': a['parts'],
                     'library_bits': a['library_bits'], 'F_2^24': F(a, 2 ** 24), 'F_2^30': F(a, 2 ** 30), 'all_on': a['all_on'],
                     'activation_edits': a['activation_edits'], 'weight_edits': {'edits': n, 'gap': gap, 'ignoring': ign}})
    ledger.append(rows)
    print(f"\n{arm['arm']} (step {arm.get('step')}) against {ref['arm']}:")
    keys = [('hard KL', 'hard_kl'), ('active bits/token', 'active_bits_per_token'), ('parts/token', 'parts_per_token'), ('library bits', 'library_bits'),
            ('F at 2^24', 'F_2^24'), ('F at 2^30', 'F_2^30'), ('all-on error', 'all_on')]
    for name, k in keys:
        v, o = rows[0][k], rows[1][k]
        print(f"  {name:20s} VPD {fmt(v):>12s}   ours {fmt(o):>12s}")
    for f in FAMILIES:
        print(f"  {'edit: ' + f:20s} VPD {fmt(rows[0]['activation_edits'][f]):>12s}   ours {fmt(rows[1]['activation_edits'][f]):>12s}")
    w0, w1 = rows[0]['weight_edits'], rows[1]['weight_edits']
    print(f"  {'weight edits':20s} VPD {fmt(w0['gap'])} vs {fmt(w0['ignoring'])}   ours {fmt(w1['gap'])} vs {fmt(w1['ignoring'])}   ({w1['edits']} edits both support)")
json.dump(ledger, open(sys.argv[1] if len(sys.argv) > 1 else '/Users/user/mpd-data/compare/ledger.json', 'w'), indent=1)
