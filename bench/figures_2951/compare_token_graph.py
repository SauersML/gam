"""One held-out token's graph, VPD beside our wired fit (#2951 head-to-head).

Left: VPD as published, MLP maps only (masks from its causal-importance network on M's clean inputs,
attention exact): the subcomponents active on the token (mask above zero) in each of the 8 MLP maps,
and its implicit edges, the pairs of an active down_proj subcomponent A of an earlier layer and an
active c_fc subcomponent B of a later one whose term in B's read, ((gain_l * v_B) . u_A) a_A / r_l, is
above float32 precision (2^-23 ||gain_l * v_B|| sqrt(d)), as vpd_fair_mlp.py counts them.
Right: our wired fit (budget_descent.py DESCENT_EDGES=1, its last evaluation's example graph): the
parts active on the same token and the kept edges between them, each with the term it adds to its
reader's read.
Each edge is drawn with width and opacity by the size of its term (red adds to the reader's read,
blue subtracts). Panel titles give the parts and edges on the token and the fit's held-out KL.

usage: compare_token_graph.py GAM TARGET VPD_PTH TOKENS OURS.json OUT.png"""
import json, math, sys, types
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, sys.argv[1])
import vpd_model
from vpd_model import load_target, load_vpd, site_names, lower_leaky
vpd_model.TARGET_DIR = Path(sys.argv[2])
vpd_model.VPD_PTH = Path(sys.argv[3])
TOKENS, OURS, OUT = sys.argv[4], sys.argv[5], sys.argv[6]
ours = json.load(open(OURS))
last = ours['trace'][-1]
g_ours = last['example_graph']
first, position = g_ours['row'], g_ours['position']
dev = 'cuda' if torch.cuda.is_available() else 'mps'
T = load_target(dev); V = load_vpd(T, dev)
mlp = [n for n in site_names() if '.mlp.' in n]
tok = np.memmap(TOKENS, dtype=np.uint16 if TOKENS.endswith('.u16') else np.float64, mode='r').reshape(-1, 513)
ids = torch.tensor(tok[first:first + 1, :512].astype(np.int64), device=dev)

# The RMS of the stream at each layer's pre-MLP norm, recorded as the model computes it.
STREAM_RMS = {}
plain_rms = vpd_model.rms
MLP_NORM = {id(T.norms[2 * l + 1]): l for l in range(4)}
def recording_rms(x, w, eps):
    if id(w) in MLP_NORM:
        STREAM_RMS[MLP_NORM[id(w)]] = (x.float().pow(2).mean(-1, keepdim=True) + eps).sqrt()
    return plain_rms(x, w, eps)
vpd_model.rms = recording_rms


def inputs_of(masks):
    """Inputs to all 24 sites on a run with the MLP maps masked (None: M itself)."""
    V.clear()
    for n in V.names:
        st = T.site(n); st.cache_input = True
        if masks is not None and n in masks:
            st.mask = masks[n]
    T(ids)
    acts = {n: T.site(n).last_input for n in V.names}
    V.clear()
    return acts


with torch.no_grad():
    masks = {n: v for n, v in ((n, lower_leaky(v)) for n, v in V.ci_fn(inputs_of(None)).items()) if n in mlp}
    acts = inputs_of(masks)
    active = {n: torch.nonzero(masks[n][0, position] > 0).squeeze(1).tolist() for n in mlp}
    edges = []
    for l in range(1, 4):
        fc = f'h.{l}.mlp.c_fc'
        gv = T.norms[2 * l + 1][:, None] * T.site(fc).V
        floor = 2.0 ** -23 * gv.norm(dim=0) * math.sqrt(gv.shape[0])
        r = STREAM_RMS[l][0, position, 0]
        for k in range(l):
            dn = f'h.{k}.mlp.down_proj'
            a = (acts[dn][0, position] @ T.site(dn).V) * masks[dn][0, position]
            for A in active[dn]:
                terms = (T.site(dn).U[A] @ gv[:, active[fc]]) * a[A] / r
                for B, term in zip(active[fc], terms.tolist()):
                    if abs(term) > floor[B].item():
                        edges.append([dn, A, fc, B, term])
g_vpd = {'active': active, 'edges': edges}
VPD_MLP_KL = 0.737  # VPD's published masks, MLP maps only, on the same held-out text (vpd_fair_mlp.py m_clean)

plt.rcParams.update({'font.size': 16})
fig, axes = plt.subplots(1, 2, figsize=(26, 10), facecolor='white')
maps = [f'h.{l}.mlp.{m}' for l in range(4) for m in ('c_fc', 'down_proj')]
for ax, g, name, kl in ((axes[0], g_vpd, 'VPD as published', VPD_MLP_KL),
                        (axes[1], g_ours, f"ours, parts + edges <= {int(ours['K'])} per token", last['kl'])):
    pos = {}
    for i, n in enumerate(maps):
        nodes = g['active'][n]
        for k, s in enumerate(nodes):
            pos[(n, s)] = (i, (k + 0.5) / max(len(nodes), 1))
    w = np.array([abs(e[4]) for e in g['edges']])
    top = np.quantile(w, 0.99) if len(w) else 1.0
    for a, i, b, j, v in sorted(g['edges'], key=lambda e: abs(e[4])):
        (x0, y0), (x1, y1) = pos[(a, i)], pos[(b, j)]
        s = min(abs(v) / top, 1.0)
        ax.plot([x0, x1], [y0, y1], color='#c44e52' if v > 0 else '#4c72b0', lw=0.3 + 2.5 * s, alpha=0.06 + 0.6 * s)
    for i, n in enumerate(maps):
        ys = [pos[(n, s)][1] for s in g['active'][n]]
        ax.scatter([i] * len(ys), ys, s=22, color='#333333', zorder=3)
        ax.annotate(f'{len(ys)}', (i, 1.03), ha='center', fontsize=15)
    parts = sum(len(v) for v in g['active'].values())
    ax.set_title(f"{name}: {parts} parts and {len(g['edges'])} edges on this token\nheld-out KL {kl:.2f} bits/token; red edges add to their reader's read, blue subtract", fontsize=17)
    ax.set_xticks(range(len(maps)))
    ax.set_xticklabels([f"layer {n.split('.')[1]}\n{n.split('.')[3]}" for n in maps])
    ax.set_yticks([])
    ax.set_ylim(-0.02, 1.08)
    ax.spines[['top', 'right', 'left']].set_visible(False)
axes[0].set_ylabel(f'active parts on token {position} of a held-out sequence')
fig.tight_layout()
fig.savefig(OUT, dpi=110, facecolor='white', bbox_inches='tight')
print('VPD', sum(len(v) for v in active.values()), 'parts,', len(edges), 'edges; ours', sum(len(v) for v in g_ours['active'].values()), 'parts,', len(g_ours['edges']), 'edges; wrote', OUT)
