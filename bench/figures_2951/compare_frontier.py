"""The vpd4l frontier on the shared verbatim manifest (#2951 head-to-head).

x = parts executed per token, counting all executed machinery in rank-one units: an explanation's own
parts that are evaluated on every token (a transcoder's features: every gate is computed; VPD's
subcomponents: every one is computed before its mask) plus every map of M it runs unchanged, counted
at its rank (vpd4l's attention, run as M's own by an MLP-only explanation: q, k, v, o of 768 x 768 per
layer, 4 x 4 x 768 = 12,288), and every network and pass whose output the explanation needs (VPD:
its causal-importance network at each matrix's rank, 102,400, and the run its masks read, M's forward
at its rank, 18,432, or for the autonomous form VPD's own all-on pass, 38,912). A second, open
marker at the parts active per token (nonzero, or unmasked; VPD's 213) plus the same unchanged maps.
y = mean KL(M_e || P_e) in bits/token over every scored token of the manifest's operations.
Labels: description bits under the one convention of compare_f_edits.py, and, for an explanation that runs
M's attention unchanged, that attention's own description at the Laplace start with its means held at M
(compare-vpd4l-attn-price, ~/mpd-data/compare/attention_price.json), labelled separately.

A point is drawn only from an EDITS file scored on the reference arm's manifest: the same immutable
manifest file (its SHA-256) where the edits driver wrote one, else the same held-out sequences, seed,
family names and edits per sequence, scored by the same binary or by one whose experiments are
checked to be the same draws (same families and positions, every edited token's effect KL(M_e || M)
within 1e-3 bits); any other is refused with its difference.
Points that other workstreams score (VPD as published, VPD fair, the decompositions at K = 213, 107,
53) come from ~/mpd-data/compare/frontier_points.json: a list of {"label", "edits" (an EDITS json),
"executed", "active", "description_bits"}.

usage: frontier.py OUT.png"""
import json, os, sys, glob, textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

C7 = '/Users/user/mpd-data/compare/new_ops/c7'
# The vpd4l manifest (edits driver at bfb0e50df4: swap, zero, scale, push, cut; seed 1; held-out
# sequences [0, 32); 32 per sequence), drawn on the transcoders as built; every arm is scored on it.
MANIFEST_DIR = '/Users/user/mpd-data/compare/manifest'
RP = '/Users/user/mpd-data/runpod'
EXTRA = '/Users/user/mpd-data/compare/frontier_points.json'
ATTENTION_RANK_ONE = 4 * 4 * 768
L = lambda p: json.load(open(p)) if p and os.path.exists(p) else None
MANIFEST = ('sequences', 'seed', 'families', 'edits_per_sequence')


def manifest(r):
    """The manifest an EDITS file was scored on: the immutable manifest file's SHA-256 where the edits
    driver wrote one (mpd_library_mdl_2951 EditSettings::manifest), which fixes every experiment;
    else rows, seed, experiments per sequence, family names."""
    sha = (r.get('manifest') or {}).get('sha256')
    if sha:
        return {'sha256': sha}
    return {'sequences': r.get('sequences'), 'seed': r.get('seed'), 'edits_per_sequence': r.get('edits_per_sequence'), 'families': sorted(k for k in r['families'] if k != 'clean')}


def gap(r):
    fams = {k: v for k, v in r['families'].items() if k != 'clean'}
    tok = sum(v['tokens'] for v in fams.values())
    return sum(v['mean_bits_per_token'] * v['tokens'] for v in fams.values()) / tok


def fixed_bits(out):
    from safetensors import safe_open
    reals = 0
    for f in glob.glob(f'{out}/transcoder_l*.safetensors'):
        with safe_open(f, 'np') as sf:
            reals += int(np.prod(sf.get_slice('bias').get_shape()))
    return 32 * reals


def features(out):
    t = L(f'{out}/TRANSCODERS.json')
    return sum(l['kept'] for l in t['layers']) if t else None


points = []
ref = None
thr2 = L(f'{RP}/compare-vpd4l-tc4096-thr2/out/checkpoint.json')
arms = [('transcoders as built (4,096 per layer)', f'{MANIFEST_DIR}/vpd4l_as_is', 'EDITS_as_is_m1.json', thr2['start'] if thr2 else None)]
for name, label in (('compare-vpd4l-tc4096-thr2', 'transcoder fit by F, read patches'), ('compare-vpd4l-tc4096-thr-edits', 'transcoder fit by F, read patches and the shared operations')):
    best = L(f'{C7}/{name}/checkpoint.best.json')
    h = L(f'{RP}/{name}/out/checkpoint.json')
    rec = None
    if best and h:
        e = (best.get('best') or [0, best['epoch'] - 1])[1]
        rec = next((x['held_out'] for x in h['epochs'] if x['epoch'] == e), None)
        label += f', epoch {e}'
    # The pair's fits were stopped at 17:15 on 10-06: their last best epochs are the final baselines.
    arms.append(('baseline: ' + label, f'{C7}/{name}', f"EDITS_{'thr2' if name.endswith('thr2') else 'thr_edits'}_m1.json", rec))
for label, d, f, rec in arms:
    r = L(f'{d}/{f}')
    if not r or not rec:
        continue
    if ref is None:
        ref = dict(manifest(r), source_revision=r.get('source_revision'))
    k = features(d)
    active = sum(l['nonzero_per_token'] for l in rec['layers'] if l['functions'])
    attn = L('/Users/user/mpd-data/compare/attention_price.json')
    points.append({'label': label, 'executed': k + ATTENTION_RANK_ONE, 'active': active + ATTENTION_RANK_ONE, 'gap': gap(r), 'description_bits': rec['divergence_bits'] + fixed_bits(d),
                   'runs_m_attention': True, 'attention_bits': attn['divergence_bits'] if attn else None, 'edits': f'{d}/{f}'})
for p in L(EXTRA) or []:
    r = L(p['edits'])
    if not r:
        continue
    m = manifest(r)
    if ref and ('sha256' in m or 'sha256' in ref):
        # The manifest file fixes every experiment: any binary scoring it scores the same draws.
        if m.get('sha256') != ref.get('sha256'):
            print('refused (another manifest):', p['label'], {'sha256': (m.get('sha256'), ref.get('sha256'))})
            continue
        points.append(dict(p, gap=gap(r)))
        continue
    diff = {k: (m.get(k), ref.get(k)) for k in MANIFEST if ref and m.get(k) != ref.get(k)}
    # Another binary may score the same manifest only if its experiments are shown to be the same
    # draws: the same families and positions, and the same effect KL(M_e || M) at every edited token.
    if ref and r.get('source_revision') != ref['source_revision']:
        c = r.get('manifest_check') or {}
        if not (c.get('same_families_and_positions') and c.get('max_effect_difference_bits', 1.0) <= 1e-3):
            diff['source_revision'] = (r.get('source_revision'), ref['source_revision'], c)
    if diff:
        print('refused (another manifest):', p['label'], diff)
        continue
    points.append(dict(p, gap=gap(r)))
json.dump({'manifest': ref, 'points': points}, open('/Users/user/mpd-data/compare/frontier.json', 'w'), indent=1)
for p in points:
    print(f"{p['label']}: executed {p['executed']}, active {p['active']:.0f}, gap {p['gap']:.3f}, description {p['description_bits'] / 1e6:.2f}M bits")
if len(sys.argv) > 1 and points:
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(15, 7.5), facecolor='white')
    colors = ['#8172b2', '#4c72b0', '#55a868', '#c44e52', '#dd8452', '#937860', '#da8bc3']
    texts = []
    for p in points:
        attn = [f"+ {p['attention_bits'] / 1e6:.1f}M bits for M's attention it runs"] if p.get('attention_bits') else (["+ M's attention it runs (pricing pending)"] if p.get('runs_m_attention') else [])
        bits = p.get('description_note') or f"{p['description_bits'] / 1e6:.1f}M bits"
        texts.append(textwrap.wrap(p['label'], 44) + textwrap.wrap(bits, 44) + attn)
    # Labels in one column right of the axes, each centred at its point's gap where room allows: a
    # pass down from the highest point keeps each label below the one above it, a pass up from zero
    # keeps the column above the axis; heights in text lines (about 40 to the axis' height).
    top = max(p['gap'] for p in points) * 1.1
    line = top / 40
    order = sorted(range(len(points)), key=lambda i: -points[i]['gap'])
    placed, below = {}, None
    for i in order:
        y = points[i]['gap'] if below is None else min(points[i]['gap'], below[0] - line * (below[1] + len(texts[i]) + 1) / 2)
        placed[i], below = y, (y, len(texts[i]))
    above = None
    for i in reversed(order):
        y = max(placed[i], line * len(texts[i]) / 2) if above is None else max(placed[i], above[0] + line * (above[1] + len(texts[i]) + 1) / 2)
        placed[i], above = y, (y, len(texts[i]))
    for i, (p, c) in enumerate(zip(points, colors * 3)):
        ax.scatter([p['executed']], [p['gap']], s=90, color=c)
        ax.scatter([p['active']], [p['gap']], s=90, facecolors='none', edgecolors=c)
        ax.plot([p['active'], p['executed']], [p['gap'], p['gap']], color=c, lw=1)
        ax.annotate('\n'.join(texts[i]), (p['executed'], p['gap']), xytext=(6e5, placed[i]), textcoords='data', va='center', fontsize=10, color=c,
                    arrowprops=dict(arrowstyle='-', lw=0.6, color=c, relpos=(0, 0.5)))
    ax.set_xscale('log')
    ax.set_xlim(30, 5e5)
    ax.set_ylim(0, max(top, max(placed[i] + line * len(texts[i]) / 2 for i in placed)))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.set_xlabel('parts executed per token, rank-one units (filled); active (open)')
    ax.set_ylabel('held-out edit KL(M_e || P_e), bits/token')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=120, facecolor='white', bbox_inches='tight')
    print('wrote', sys.argv[1])
