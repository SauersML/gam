"""Head-to-head figure. (a) vpd4l: MLP parts active per token per layer. (b) vpd4l: KL(M_e || P_e) per
edit family, pooled over every scored token from the edited one on (bars) and at the edited token
(dots). (c) Qwen3-0.6B, the same for the transcoder features as built and after the F fit.

Inputs: the compare agent's result files under ~/mpd-data (scratch/compare/tc, compare/, runpod/), skipped
when missing.

usage: compare_edit_faithfulness.py OUT.png"""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

D = '/Users/user/mpd-data/scratch/compare/tc'
out = sys.argv[1]
load = lambda p: json.load(open(p)) if os.path.exists(p) else None
pick = lambda a, b: load(a) or load(b)  # the all-family run (one binary for every arm) when it exists
vpd_counts = load('/Users/user/mpd-data/scratch/toygate/sparsity/vpd_counts.json')
tc = load(f'{D}/eval_tc4096.json')
ours = load(f'{D}/ours_tc4096.json')
plt.rcParams.update({'font.size': 14})
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(21, 6.5), facecolor='white', gridspec_kw={'width_ratios': [1, 1.2, 0.9]})

# (a) active MLP parts per token per layer.
series = [('VPD subcomponents', [vpd_counts[f'layer{l}_mlp']['mean'] for l in range(4)], '#c44e52'),
          ('transcoders as built', tc['M_inputs']['active_per_token'], '#8172b2')]
if ours:
    series.append(('ours, fitted by F', ours['active_per_token'], '#4c72b0'))
w = 0.8 / len(series)
for k, (name, vals, c) in enumerate(series):
    xs = np.arange(4) + (k - (len(series) - 1) / 2) * w
    a1.bar(xs, vals, w, color=c, label=name)
    for x, v in zip(xs, vals):
        a1.text(x, v, f'{v:.0f}' if v >= 10 else f'{v:.1f}', ha='center', va='bottom', fontsize=11)
a1.set_xticks(range(4), [f'layer {l}' for l in range(4)])
a1.set_ylabel('MLP parts active per token')
a1.set_title('vpd4l: sparsity')
a1.legend(frameon=False, fontsize=12, loc='upper left')


def fams_py(r):
    return [(r['pooled_clean']['mean_bits_per_token'], r['pooled_clean']['mean_bits_per_token']),
            (r['pooled_remove']['mean_bits_per_token'], r['pooled_remove']['edited_token_mean_bits']),
            (r['pooled_amplify']['mean_bits_per_token'], r['pooled_amplify']['edited_token_mean_bits'])]


def fams_rs(r):
    f = r['families']
    return [(f['clean']['mean_bits_per_token'], f['clean']['mean_bits_per_token']),
            (f['remove_part']['mean_bits_per_token'], f['remove_part']['edited_token_mean_bits']),
            (f['amplify_part']['mean_bits_per_token'], f['amplify_part']['edited_token_mean_bits'])]


def panel(ax, models, title):
    models = [(n, fn(r), c) for n, r, fn, c in models if r and ('pooled_clean' in r or 'families' in r)]
    w = 0.8 / max(1, len(models))
    for k, (name, vals, c) in enumerate(models):
        xs = np.arange(3) + (k - (len(models) - 1) / 2) * w
        ax.bar(xs, [v[0] for v in vals], w, color=c, label=name)
        ax.scatter(xs[1:], [v[1] for v in vals[1:]], color='black', s=18, zorder=3)
        for x, v in zip(xs, vals):
            ax.text(x, v[0], f'{v[0]:.2f}', ha='center', va='bottom', fontsize=10)
    top = max([max(v[0], v[1]) for _, vals, _ in models for v in vals] + [1e-9])
    ax.set_ylim(0, top * 1.3)
    ax.set_xticks(range(3), ['no edit', 'remove a part', 'scale a part\n(x0.5, 2, 3)'])
    ax.set_ylabel('KL(M_e || P_e), bits per token')
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=11, loc='upper left')


panel(a2, [('VPD, all sites', load(f'{D}/edit_vpd.json'), fams_py, '#c44e52'),
           ('VPD, attention exact', load(f'{D}/edit_vpd_mlp.json'), fams_py, '#dd8452'),
           ('transcoders as built', pick('/Users/user/mpd-data/compare/vpd4l_tc4096_mac/EDITS_as_is_all.json', '/Users/user/mpd-data/compare/vpd4l_tc4096_mac/EDITS_as_is.json'), fams_rs, '#8172b2'),
           ('ours, fitted by F', load(ours['edits']) if ours and ours.get('edits') else None, fams_rs, '#4c72b0')], 'vpd4l: edit faithfulness (dots: edited token)')
Q = '/Users/user/mpd-data/runpod/sparseimport-all28-n2p22b/out'
panel(a3, [('transcoders as built', pick(f'{Q}/EDITS_as_is_all.json', f'{Q}/EDITS_as_is.json'), fams_rs, '#8172b2'),
           ('fitted by F (epoch 6)', pick(f'{Q}/EDITS_fit_best_all.json', f'{Q}/EDITS_fit_best.json'), fams_rs, '#4c72b0')], 'Qwen3-0.6B, 28 MLPs')
for a in (a1, a2, a3):
    a.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(out, dpi=120, facecolor='white')
print('wrote', out)
