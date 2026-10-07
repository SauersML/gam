"""F on held-out edits per explanation, on the one shared experiment manifest only.

A point is drawn only for an explanation scored on one immutable manifest of shared,
explanation-independent operations. vpd4l: ~/mpd-data/compare/manifest/MANIFEST_vpd4l_s1.json (swap,
zero, scale, push, cut; seed 1; held-out rows 1024 to 1056), the transcoder arms scored by the edits
driver (EDITS_*_m1.json) and VPD's three forms by mpd_battery_2951 site_edits on the same file
(~/mpd-data/compare/manifest/vpd/EDITS_vpd_*.json). Qwen3-0.6B: the c7fa7f6dc5 draw (swap, zero,
scale, push) under ~/mpd-data/compare/new_ops/c7/. An explanation without such a score is listed in
the table with no number and is not drawn; a clean-text error never stands in for an edit gap.

Description bits, one convention for every explanation: a fit's description as its F counts it (KL(q || p)
of every described group: features, threshold groups, sink vectors; the active groups' variances; its
discrete choices and prior parameters), plus 32 bits per real for every executed
fixed piece that is not one of M's own tensors (a transcoder block's b_dec); VPD's 11.4M bits of
subcomponents plus 77.2M bits of causal-importance network; and for an explanation that runs M's
attention unchanged, that attention's description at the Laplace start with its means held at M
(11.4M bits, compare-vpd4l-attn-price). Decomposition arms come from
~/mpd-data/compare/frontier_points.json, drawn when scored on the same manifest (its SHA-256).
F on edits = description bits / N + mean gap, with one N = 2^24 for every arm: description bits per
token compare only at the same N.

The figure plots the description per token (x) against the gap (y), so F = x + y and the dashed
diagonals are lines of equal F.

usage: f_edits.py [OUT.png]"""
import json, os, sys, glob, textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from safetensors import safe_open
from compare_common import experiments_of, latest_scores

C7 = '/Users/user/mpd-data/compare/new_ops/c7'
MANIFEST_DIR = '/Users/user/mpd-data/compare/manifest'
RP = '/Users/user/mpd-data/runpod'
L = lambda p: json.load(open(p)) if p and os.path.exists(p) else None


def manifest_gap(r):
    """Mean gap over every scored token of the shared operations, its by-effect bins, and the
    per-family records (with any diagnostic fields the driver adds)."""
    fams = {k: v for k, v in r['families'].items() if k != 'clean'}
    tok = sum(v['tokens'] for v in fams.values())
    gap = sum(v['mean_bits_per_token'] * v['tokens'] for v in fams.values()) / tok
    bins = {}
    for v in fams.values():
        for b in v.get('by_effect', []):
            key = str(b['effect_bits_at_edited_token'])
            n = b['experiments']
            acc = bins.setdefault(key, [0, 0.0])
            acc[0] += n
            acc[1] += b['mean_bits_per_token'] * n
    return gap, {k: (n, s / n) for k, (n, s) in bins.items() if n}, r['families']


def description(held_out):
    """A fit's description in bits as its F counts it: KL(q || p) of the described groups, the active
    groups' variances, its discrete choices and its prior's parameters (library_mdl::HeldOut)."""
    return sum(held_out.get(k) or 0.0 for k in ('divergence_bits', 'variance_bits', 'choice_bits', 'prior_bits'))


def strong_weights(path):
    """An arm's scores under the strong native weight edits (MANIFEST_vpd4l_s3.json, an EDITS file's
    "weights"): over the applicable edits the gap KL(M_e || P_e), the edit-ignoring baseline
    KL(M_e || P) and the effect KL(M_e || M) in bits per token, in all and per family and per effect bin
    (the effect on the scored tokens: below 0.01, 0.01 to 0.1, 0.1 to 1, above 1 bit); none without the
    file."""
    r = L(path)
    if not r or not r.get('weights'):
        return None
    records = [x for x in r['weights']['records'] if x.get('applicable')]
    def summary(rs):
        n = len(rs)
        mean = lambda k: sum(x[k] for x in rs) / n if n else None
        return {'edits': n, 'gap': mean('mean_bits_per_token'), 'ignoring': mean('ignoring_mean_bits_per_token'), 'effect': mean('effect_mean_bits_per_token')}
    bins = [(0.0, 0.01), (0.01, 0.1), (0.1, 1.0), (1.0, float('inf'))]
    return {'file': path, 'not_applicable': len(r['weights']['records']) - len(records), 'all': summary(records),
            'families': {f: summary([x for x in records if x.get('family') == f]) for f in sorted({x.get('family') for x in records})},
            'bins': {f'[{a}, {b})': summary([x for x in records if a <= x['effect_mean_bits_per_token'] < b]) for a, b in bins}}


def fixed_bits(out):
    """32 bits per real of each transcoder layer's fixed output bias (not M's tensor)."""
    reals = 0
    for f in glob.glob(f'{out}/transcoder_l*.safetensors'):
        with safe_open(f, 'np') as sf:
            reals += int(np.prod(sf.get_slice('bias').get_shape()))
    return 32 * reals


def latest(fit_dir):
    h = L(f'{fit_dir}/checkpoint.json')
    return h


N = 2 ** 24
rows = []
# vpd4l: transcoders as built (priced by the threshold-group fit's Laplace start, the same
# explanation), and the two transcoder fits' final best checkpoints (stopped 10-06 17:15), baselines.
thr2 = latest(f'{RP}/compare-vpd4l-tc4096-thr2/out')
arms = [('vpd4l', 'transcoders as built (4,096 per layer)', f'{MANIFEST_DIR}/vpd4l_as_is', latest_scores(f'{MANIFEST_DIR}/vpd4l_as_is', 'as_is'), thr2['start'] if thr2 else None, f'{RP}/compare-vpd4l-tc4096-thr2/out')]
for name, label in (('compare-vpd4l-tc4096-thr2', 'transcoder baseline: fit by F, read patches'), ('compare-vpd4l-tc4096-thr-edits', 'transcoder baseline: fit by F, read patches and the shared operations')):
    h = latest(f'{RP}/{name}/out')
    best = L(f'{C7}/{name}/checkpoint.best.json')
    rec = None
    if h and best and h['epochs']:
        e = best['best'][1] if best.get('best') else max(best['epoch'] - 1, 0)
        rec = next((x['held_out'] for x in h['epochs'] if x['epoch'] == e), None)
        label += f', epoch {e}'
    arms.append(('vpd4l', label, f'{C7}/{name}', latest_scores(f'{C7}/{name}', 'thr2' if name.endswith('thr2') else 'thr_edits'), rec, f'{RP}/{name}/out'))
arms.append(('Qwen3-0.6B', 'transcoders as built (f >= 1e-3, 28,545 features)', f'{C7}/qwen3_as_is', 'EDITS_as_is_ops.json', None, f'{C7}/qwen3_as_is'))
attn = L('/Users/user/mpd-data/compare/attention_price.json')
for model, label, d, f, rec, out in arms:
    r = L(f'{d}/{f}')
    row = {'model': model, 'method': label, 'edits': f'{d}/{f}' if r else None, 'manifest': experiments_of(r)}
    strong = strong_weights(f"{d}/{f.rsplit('_', 1)[0]}_s3.json")
    if strong:
        row['strong_weights'] = strong
    if r:
        row['gap'], row['gap_by_effect'], row['families'] = manifest_gap(r)
        if r.get('weights'):
            # The manifest's native weight edits (EDITS "weights"): the gap, the edit-ignoring
            # baseline and the effect on M over the applicable edits, and how many were not.
            row['weights'] = {k: r['weights'].get(k) for k in ('edits', 'not_applicable', 'mean_bits_per_token', 'ignoring_mean_bits_per_token', 'effect_mean_bits_per_token')}
    if rec:
        row['description_bits'] = description(rec) + fixed_bits(out)
        row['active_per_token'] = [l['nonzero_per_token'] for l in rec['layers'] if l['functions']]
        # An MLP-only explanation runs M's attention unchanged and is charged its description.
        if model == 'vpd4l':
            row['attention_bits'] = attn['divergence_bits'] if attn else None
    rows.append(row)
reference = next((r['manifest'] for r in rows if r['model'] == 'vpd4l' and r.get('manifest')), None)
# VPD (vpdstart, 10-06): description = its subcomponents' KL(q || p) at 2^24 (vpd-pricing-n2p24b, 11.4M
# bits) plus its causal-importance network priced at its Laplace start (77.2M bits); gaps on the vpd4l
# manifest from mpd_battery_2951 site_edits (each form's masks recomputed under each edit).
for form, label in (('published', 'VPD as published (CI reads the edited M, both ways)'), ('causal', 'VPD, causal CI on the edited M'), ('autonomous', 'VPD autonomous (causal CI on its own run)')):
    # On the weight-edit manifest (s2) where scored, else s1.
    path = next((f for f in (f'{MANIFEST_DIR}/vpd_s2/EDITS_vpd_{form}.json', f'{MANIFEST_DIR}/vpd/EDITS_vpd_{form}.json') if os.path.exists(f)), f'{MANIFEST_DIR}/vpd/EDITS_vpd_{form}.json')
    r = L(path)
    strong = strong_weights(f'{MANIFEST_DIR}/vpd_s3/EDITS_vpd_{form}.json')
    row = {'model': 'vpd4l', 'method': label, 'edits': path if r else None, 'manifest': experiments_of(r), **({'strong_weights': strong} if strong else {}),
           'description_bits': 11.4e6 + 77.2e6, 'description_note': '11.4M subcomponents + 77.2M CI network', 'active': '213 subcomponents unmasked per token'}
    if r:
        row['gap'], row['gap_by_effect'], row['families'] = manifest_gap(r)
        if r.get('weights'):
            # The manifest's native weight edits (EDITS "weights"): the gap, the edit-ignoring
            # baseline and the effect on M over the applicable edits, and how many were not.
            row['weights'] = {k: r['weights'].get(k) for k in ('edits', 'not_applicable', 'mean_bits_per_token', 'ignoring_mean_bits_per_token', 'effect_mean_bits_per_token')}
    rows.append(row)
# Other workstreams' arms (decomp's decompositions), from the frontier's points file.
for p in L('/Users/user/mpd-data/compare/frontier_points.json') or []:
    if p['label'].startswith('VPD'):
        continue
    r = L(p.get('edits'))
    strong = strong_weights(p.get('weights_edits'))
    row = {'model': 'vpd4l', 'method': p['label'], 'edits': p.get('edits') if r else None, 'manifest': experiments_of(r), **({'strong_weights': strong} if strong else {}),
           'description_bits': p.get('description_bits'), 'description_note': p.get('description_note'), 'attention_bits': p.get('attention_bits')}
    if r:
        row['gap'], row['gap_by_effect'], row['families'] = manifest_gap(r)
        if r.get('weights'):
            # The manifest's native weight edits (EDITS "weights"): the gap, the edit-ignoring
            # baseline and the effect on M over the applicable edits, and how many were not.
            row['weights'] = {k: r['weights'].get(k) for k in ('edits', 'not_applicable', 'mean_bits_per_token', 'ignoring_mean_bits_per_token', 'effect_mean_bits_per_token')}
    rows.append(row)
for r in rows:
    if r['model'] == 'vpd4l' and r.get('gap') is not None and r.get('manifest') != reference:
        r['refused'] = f"scored on manifest {r.get('manifest')}, not {reference}"
        del r['gap']
    if r.get('description_bits') is not None:
        r['N'] = N
        r['description_bits_total'] = r['description_bits'] + (r.get('attention_bits') or 0)
        if r.get('gap') is not None:
            r['F_edits'] = r['description_bits_total'] / N + r['gap']
json.dump(rows, open('/Users/user/mpd-data/compare/f_edits_table.json', 'w'), indent=1)
for r in rows:
    print(r['model'], '|', r['method'], '| gap', r.get('gap') and round(r['gap'], 3), '| description', r.get('description_bits_total') and f"{r['description_bits_total']:.4g}",
          '| F on edits (N = 2^24)', r.get('F_edits') and round(r['F_edits'], 3),
          '| weight edits: gap', (r.get('weights') or {}).get('mean_bits_per_token') and round(r['weights']['mean_bits_per_token'], 4),
          'ignoring', (r.get('weights') or {}).get('ignoring_mean_bits_per_token') and round(r['weights']['ignoring_mean_bits_per_token'], 4),
          'not applicable', (r.get('weights') or {}).get('not_applicable'),
          '| strong weight edits', (lambda w: w and {'all': {k: (round(v, 4) if isinstance(v, float) else v) for k, v in w['all'].items()}, 'not applicable': w['not_applicable'],
                                                     'families': {f: (v['edits'], v['gap'] and round(v['gap'], 3), v['ignoring'] and round(v['ignoring'], 3), v['effect'] and round(v['effect'], 3)) for f, v in w['families'].items()},
                                                     'bins': {b: (v['edits'], v['gap'] and round(v['gap'], 3), v['ignoring'] and round(v['ignoring'], 3)) for b, v in w['bins'].items()}})(r.get('strong_weights')), '| bins', {k: (n, round(g, 3)) for k, (n, g) in (r.get('gap_by_effect') or {}).items()}, r.get('refused', ''))
if len(sys.argv) > 1:
    plt.rcParams.update({'font.size': 14})
    models = [m for m in ('vpd4l', 'Qwen3-0.6B') if any(r['model'] == m and r.get('F_edits') is not None for r in rows)]
    fig, axes = plt.subplots(1, len(models), figsize=(14 * len(models), 7.5), facecolor='white', squeeze=False)
    colors = ['#8172b2', '#4c72b0', '#55a868', '#c44e52', '#dd8452', '#937860', '#da8bc3', '#64b5cd', '#8c8c8c']
    for ax, model in zip(axes[0], models):
        # VPD as published is the one labelled reference point; its other forms stay in the table.
        pts = [r for r in rows if r['model'] == model and r.get('F_edits') is not None and not (r['method'].startswith('VPD') and not r['method'].startswith('VPD as published'))]
        xs = [r['description_bits_total'] / N for r in pts]
        top = max(r['gap'] for r in pts) * 1.15
        right = max(xs) * 1.15
        # Lines of equal F = x + y at even F up to the largest point's, each labelled where it enters
        # the axes: left of it on the top edge, above it on the left edge.
        for F in range(2, int(max(r['F_edits'] for r in pts)) + 3, 2):
            ax.plot([0, F], [F, 0], color='#bbbbbb', lw=0.8, ls='--', zorder=0)
            if F > top:
                ax.annotate(f'F = {F}', (F - top, top), textcoords='offset points', xytext=(-4, -14), ha='right', fontsize=11, color='#888888')
            else:
                ax.annotate(f'F = {F}', (0, F), textcoords='offset points', xytext=(4, 3), fontsize=11, color='#888888')
        texts = []
        for r in pts:
            bits = r.get('description_note') or f"{r['description_bits'] / 1e6:.1f}M bits"
            attn_line = [f"+ {r['attention_bits'] / 1e6:.1f}M bits for M's attention it runs"] if r.get('attention_bits') else []
            texts.append(textwrap.wrap(r['method'], 46) + textwrap.wrap(bits, 46) + attn_line + [f"F = {r['F_edits']:.2f} bits/token"])
        # Labels in one column right of the axes, each centred at its point's gap where room allows: a
        # pass down from the highest point keeps each label below the one above it, a pass up from
        # zero keeps the column above the axis; heights in text lines (about 40 to the axis' height).
        line = top / 40
        order = sorted(range(len(pts)), key=lambda i: -pts[i]['gap'])
        placed, below = {}, None
        for i in order:
            y = pts[i]['gap'] if below is None else min(pts[i]['gap'], below[0] - line * (below[1] + len(texts[i]) + 1) / 2)
            placed[i], below = y, (y, len(texts[i]))
        above = None
        for i in reversed(order):
            y = max(placed[i], line * len(texts[i]) / 2) if above is None else max(placed[i], above[0] + line * (above[1] + len(texts[i]) + 1) / 2)
            placed[i], above = y, (y, len(texts[i]))
        for i, (r, x, c) in enumerate(zip(pts, xs, colors * 3)):
            ax.scatter([x], [r['gap']], s=90, color=c, zorder=3)
            ax.annotate('\n'.join(texts[i]), (x, r['gap']), xytext=(1.03, placed[i]), textcoords=('axes fraction', 'data'), va='center', fontsize=10, color=c,
                        arrowprops=dict(arrowstyle='-', lw=0.6, color=c, relpos=(0, 0.5)))
        ax.set_xlim(0, right)
        ax.set_ylim(0, max(top, max(placed[i] + line * len(texts[i]) / 2 for i in placed)))
        ax.set_xlabel('description, bits per token at N = 2^24')
        ax.set_ylabel('held-out edit KL(M_e || P_e), bits/token')
        ax.set_title(model)
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=120, facecolor='white', bbox_inches='tight')
    print('wrote', sys.argv[1])
