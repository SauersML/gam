"""F on held-out edits per explanation, on the one shared experiment manifest only.

A point is drawn only for an explanation scored on one immutable manifest of shared,
explanation-independent operations. vpd4l: ~/mpd-data/compare/manifest/MANIFEST_vpd4l_s1.json (swap,
zero, scale, push, cut; seed 1; held-out rows 1024 to 1056), the transcoder arms scored by the edits
driver (EDITS_*_m1.json) and VPD's three forms by mpd_battery_2951 site_edits on the same file
(~/mpd-data/compare/manifest/vpd/EDITS_vpd_*.json). Qwen3-0.6B: the c7fa7f6dc5 draw (swap, zero,
scale, push) under ~/mpd-data/compare/new_ops/c7/. An explanation without such a score is listed in
the table with no number and is not drawn; a clean-text error never stands in for an edit gap.

Description bits, one convention for every explanation: KL(q || p) of every described group (the
fit's divergence: features, threshold groups, sink vectors), plus 32 bits per real for every executed
fixed piece that is not one of M's own tensors (a transcoder block's b_dec). M's own tensors that an
explanation runs unchanged (attention, embeddings, norms) are charged to no explanation.
F on edits = description bits / N + mean gap, N the fit's scored training tokens.

usage: f_edits.py [OUT.png]"""
import json, os, sys, glob, textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from safetensors import safe_open

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


rows = []
# vpd4l: transcoders as built (priced by the threshold-group fit's Laplace start, the same
# explanation), and the two fits' best checkpoints.
thr2 = latest(f'{RP}/compare-vpd4l-tc4096-thr2/out')
arms = [('vpd4l', 'transcoders as built (4,096 per layer, priced at its Laplace start)', f'{MANIFEST_DIR}/vpd4l_as_is', 'EDITS_as_is_m1.json',
         thr2['start'] if thr2 else None, thr2['tokens'] if thr2 else None, f'{RP}/compare-vpd4l-tc4096-thr2/out')]
for name, label in (('compare-vpd4l-tc4096-thr2', 'ours, fitted by F (read patches)'), ('compare-vpd4l-tc4096-thr-edits', 'ours, fitted by F (read patches and the shared operations)')):
    h = latest(f'{RP}/{name}/out')
    best = L(f'{C7}/{name}/checkpoint.best.json')
    rec = None
    if h and best and h['epochs']:
        e = best['best'][1] if best.get('best') else max(best['epoch'] - 1, 0)
        rec = next((x['held_out'] for x in h['epochs'] if x['epoch'] == e), None)
        label += f' (epoch {e})'
    arms.append(('vpd4l', label, f'{C7}/{name}', f"EDITS_{'thr2' if name.endswith('thr2') else 'thr_edits'}_m1.json", rec, h['tokens'] if h else None, f'{RP}/{name}/out'))
arms.append(('Qwen3-0.6B', 'transcoders as built (f >= 1e-3, 28,545 features)', f'{C7}/qwen3_as_is', 'EDITS_as_is_ops.json', None, None, f'{C7}/qwen3_as_is'))
for model, label, d, f, rec, N, out in arms:
    r = L(f'{d}/{f}')
    row = {'model': model, 'method': label, 'edits': f'{d}/{f}' if r else None}
    if r:
        row['gap'], row['gap_by_effect'], row['families'] = manifest_gap(r)
    if rec and N:
        row['description_bits'] = rec['divergence_bits'] + fixed_bits(out)
        row['N'] = N
        row['active_per_token'] = [l['nonzero_per_token'] for l in rec['layers'] if l['functions']]
        if r:
            row['F_edits'] = row['description_bits'] / N + row['gap']
    rows.append(row)
# VPD (vpdstart, 10-06): description = its subcomponents' KL(q || p) at 2^24 (vpd-pricing-n2p24b, 11.4M
# bits) plus its causal-importance network priced at its Laplace start (77.2M bits), N = 2^24; gaps on
# the vpd4l manifest from mpd_battery_2951 site_edits (each form's masks recomputed under each edit).
for form, label in (('published', 'VPD as published (its CI network reads the whole edited M, bidirectional)'),
                    ('causal', 'VPD with causal CI on the edited M'), ('autonomous', 'VPD autonomous, causal CI on its own activations')):
    path = f'{MANIFEST_DIR}/vpd/EDITS_vpd_{form}.json'
    r = L(path)
    row = {'model': 'vpd4l', 'method': label, 'edits': path if r else None, 'description_bits': 11.4e6 + 77.2e6, 'N': 2 ** 24, 'active': '213 subcomponents unmasked per token'}
    if r:
        row['gap'], row['gap_by_effect'], row['families'] = manifest_gap(r)
        row['F_edits'] = row['description_bits'] / row['N'] + row['gap']
    rows.append(row)
# M's tensors an MLP-only explanation runs unchanged (vpd4l's attention) are charged at the precision an
# explanation needs: their description at the Laplace start with means held at M (compare-vpd4l-attn-price).
attn = L('/Users/user/mpd-data/compare/attention_price.json')
for r in rows:
    if r['model'] == 'vpd4l' and not r['method'].startswith('VPD') and r.get('description_bits'):
        r['attention_bits'] = attn['divergence_bits'] if attn else None
        r['description_bits_with_attention'] = r['description_bits'] + attn['divergence_bits'] if attn else None
        if attn and r.get('gap') is not None:
            r['F_edits_with_attention'] = r['description_bits_with_attention'] / r['N'] + r['gap']
json.dump(rows, open('/Users/user/mpd-data/compare/f_edits_table.json', 'w'), indent=1)
for r in rows:
    print(r['model'], '|', r['method'], '| gap', r.get('gap') and round(r['gap'], 3), '| description', r.get('description_bits') and f"{r['description_bits']:.4g}",
          '| F on edits', r.get('F_edits') and round(r['F_edits'], 3), '| with attention', r.get('F_edits_with_attention') and round(r['F_edits_with_attention'], 3), '| clean', r.get('clean_kl_bits_per_token'), '| bins', {k: (n, round(g, 3)) for k, (n, g) in (r.get('gap_by_effect') or {}).items()}, r.get('note', ''))
if len(sys.argv) > 1:
    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), facecolor='white')
    for ax, model in zip(axes, ('vpd4l', 'Qwen3-0.6B')):
        pts = [r for r in rows if r['model'] == model and r.get('gap') is not None and r.get('description_bits')]
        for r in pts:
            ax.scatter([r['description_bits']], [r['gap']], s=80, color='#4c72b0' if r['method'].startswith('ours') else '#8172b2')
            act = 'active per token by layer: ' + '/'.join(f'{a:.0f}' for a in r['active_per_token']) if 'active_per_token' in r else r.get('active', '')
            ax.annotate('\n'.join(textwrap.wrap(r['method'], 38)) + f'\n{act}', (r['description_bits'], r['gap']), textcoords='offset points', xytext=(8, 6), fontsize=10)
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v / 1e6:g}M'))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel('description bits (log scale)')
        ax.set_ylabel('mean gap on the shared operations, bits/token')
        ax.set_title(model if pts else f'{model}: no explanation scored and priced yet')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=120, facecolor='white')
    print('wrote', sys.argv[1])
