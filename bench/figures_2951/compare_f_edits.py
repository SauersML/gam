"""F on held-out edits = description bits / N_train + mean gap KL(M_e || P_e) (bits/token), per method,
from the result files there are now; one figure per model: x = description bits (log), y = gap."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
L = lambda p: json.load(open(p)) if os.path.exists(p) else None
Q = '/Users/user/mpd-data/runpod/sparseimport-all28-n2p22b/out'
V = '/Users/user/mpd-data/runpod/compare-vpd4l-tc4096-n2p24/out'
D = '/Users/user/mpd-data/scratch/compare/tc'


def gap_rs(r):  # mean gap over the edit experiments' scored tokens (all families but clean), and clean
    f = r['families']
    tok = sum(v['tokens'] for k, v in f.items() if k != 'clean')
    return sum(v['mean_bits_per_token'] * v['tokens'] for k, v in f.items() if k != 'clean') / tok, f['clean']['mean_bits_per_token']


def gap_py(r):
    n = r['pooled_remove']['tokens'] + r['pooled_amplify']['tokens']
    return (r['pooled_remove']['mean_bits_per_token'] * r['pooled_remove']['tokens'] + r['pooled_amplify']['mean_bits_per_token'] * r['pooled_amplify']['tokens']) / n, r['pooled_clean']['mean_bits_per_token']


rows = []
qh = L(f'{Q}/checkpoint.json')
best = max(qh['epochs'], key=lambda e: -e['held_out']['objective_bits_per_token'])
for name, desc, r, act in (('Qwen3 transcoders as built (f>=1e-3, 28,545)', qh['start']['divergence_bits'], L(f'{Q}/EDITS_as_is_all.json') or L(f'{Q}/EDITS_as_is.json'), [l['nonzero_per_token'] for l in qh['start']['layers'] if l['functions']]),
                           (f"Qwen3 ours, fitted (epoch {best['epoch']})", best['held_out']['divergence_bits'], L(f'{Q}/EDITS_fit_best_all.json') or L(f'{Q}/EDITS_fit_best.json'), [l['nonzero_per_token'] for l in best['held_out']['layers'] if l['functions']])):
    g, c = gap_rs(r)
    rows.append({'model': 'Qwen3-0.6B', 'method': name, 'N_train': qh['tokens'], 'description_bits': desc, 'gap_edits': g, 'gap_clean': c, 'F_edits': desc / qh['tokens'] + g, 'active_per_token_mean': float(np.mean(act))})
vh = L(f'{V}/checkpoint.json')
N = vh['tokens']
r_tc = L('/Users/user/mpd-data/compare/vpd4l_tc4096_mac/EDITS_as_is_all.json')
g, c = gap_rs(r_tc) if r_tc else gap_py(L(f'{D}/edit_vpd4l-relu4096.json'))
rows.append({'model': 'vpd4l', 'method': 'transcoders as built (4096/layer)', 'N_train': N, 'description_bits': vh['start']['divergence_bits'], 'gap_edits': g, 'gap_clean': c, 'F_edits': vh['start']['divergence_bits'] / N + g, 'active_per_token_mean': float(np.mean([l['nonzero_per_token'] for l in vh['start']['layers'] if l['functions']])), 'gap_source': 'driver' if r_tc else 'python'})
e = vh['epochs'][-1]['held_out']
rows.append({'model': 'vpd4l', 'method': f"ours, fitted (epoch {vh['epochs'][-1]['epoch']}, clean gap only)", 'N_train': N, 'description_bits': e['divergence_bits'], 'gap_edits': None, 'gap_clean': e['mean_bits_per_token'], 'F_edits': None, 'active_per_token_mean': float(np.mean([l['nonzero_per_token'] for l in e['layers'] if l['functions']]))})
for name, f, act in (('VPD, all sites (CI uncharged, gates read M; KL(q||p) of vpd-pricing-n2p24b, attention included)', 'edit_vpd.json', 213.4), ):
    g, c = gap_py(L(f'{D}/{f}'))
    rows.append({'model': 'vpd4l', 'method': name, 'caveat': 'CI uncharged, gates read M (favourable to VPD); the fair point (CI inside P, on P activations, charged) comes from vpdstart', 'N_train': 2 ** 24, 'description_bits': 11.4e6, 'gap_edits': g, 'gap_clean': c, 'F_edits': 11.4e6 / 2 ** 24 + g, 'active_per_token_mean': act / 4})
json.dump(rows, open('/Users/user/mpd-data/compare/f_edits_table.json', 'w'), indent=1)
for r in rows:
    print(r['model'], '|', r['method'], '| desc', f"{r['description_bits']:.3g}", '| gap edits', r['gap_edits'] and round(r['gap_edits'], 3), '| clean', round(r['gap_clean'], 3), '| F_edits', r['F_edits'] and round(r['F_edits'], 3), '| active/token/layer', round(r['active_per_token_mean'], 1))
plt.rcParams.update({'font.size': 14})
fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor='white')
for ax, model in zip(axes, ('vpd4l', 'Qwen3-0.6B')):
    for r in rows:
        if r['model'] != model:
            continue
        y = r['gap_edits'] if r['gap_edits'] is not None else r['gap_clean']
        ax.scatter([r['description_bits']], [y], s=80, color='#4c72b0' if 'ours' in r['method'] else '#c44e52' if 'VPD' in r['method'] else '#8172b2')
        ax.annotate(f"{r['method'].split(' (')[0]}{' (CI uncharged, gates read M)' if 'VPD' in r['method'] else ''}\n{r['active_per_token_mean']:.0f} active/token/layer", (r['description_bits'], y), textcoords='offset points', xytext=(8, 6), fontsize=11)
    ax.set_xscale('log')
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v / 1e6:.0f}M'))
    ax.set_xlim(5e6, 6e7)
    ax.set_xlabel('description bits, KL(q||p) (log scale)')
    ax.set_xticks([5e6, 1e7, 2e7, 5e7])
    ax.set_ylabel('held-out edit gap KL(M_e || P_e), bits/token')
    ax.set_title(model)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig('/Users/user/mpd-data/figures/compare/f_edits_description_vs_gap.png', dpi=120, facecolor='white')
