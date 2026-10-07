"""Export a budget_descent.py explanation (DESCENT_SAVE) to library_vpd's importer (#2951, descent):
a decomposition directory in VPD's export layout (export.json, per site U [C, d_out] and V [d_in, C]
as little-endian float64, all 24 sites in M's order) and a start file with one arm of components,
for mpd_library_mdl_2951's `vpd` setting, scored by the unchanged edits driver.

MLP maps carry the prototype's slices. Its own-read gate Phi((|v.x| ||u|| - tau)/s) is library_vpd's
own read ||V_b^T x|| - tau_b at width s once each slice is rescaled to v ||u||, u/||u|| (the same
rank-one map). The neuron start's gate on the signed pre-activation w_j . x is a direction gate
(g = w_j, c = 0) over the neuron's c_fc and down_proj slices. A whole-model save ('attn': per head
reads V [H, d_in_h, C] and writes U [H, C, d_out_h], thresholds and widths [H, C]) gives each
attention map its heads' slices as full-map rank-one slices (zero outside the head's block), each its
own component under the same own read; otherwise attention maps are copied from an exact
decomposition (ATTN_DIR), and a run scoped to the MLP blocks (`blocks` 1, 3, 5, 7) keeps M's attention.
With --all-on every threshold is -1e9: every component on, so P is M (a gap of 0 checks the import).
library_vpd gates each stage one way (6bb80d5f1b; before it, one way for the whole explanation, and
an own read beside a direction gate anywhere read the constant 0, Φ(0) = 1/2 at a hard gate's width,
which halved every attention map of the neuron start's exports): a neuron start's always-on attention
components read the zero direction (z = 0 - tau > 0), which either way scores them on, and its head
slices keep their own reads at the attention's stages beside the direction-gated MLP.
The rot arm (a save with 'rot': per layer the neuron order 'perm', per group of 32 neurons the
rotation's 'A', the block logits 'L', thresholds 'tau' and widths 's') gives each MLP map M's own
weights (from MODEL, an export of M) in the groups' rotated bases Q = exp(A*U - (A*U)^T), U the strict
upper triangle: c_fc slice i of a group reads W_fc,G^T Q_i and writes Q_i, its down slice reads Q_i
and writes W_dn,G Q_i (rescaled to Q_i ||W_dn,G Q_i||, W_dn,G Q_i / ||W_dn,G Q_i||); each block
(the slices whose argmax over L is it) is one component of its c_fc and down slices, gated on the
norm of its down reads on the layer's all-on activations (library_vpd's Read::Active, c5ebe92bc2),
which is the arm's own read sqrt(sum_i (|abar_G . Q_i| ||W_dn,G Q_i||)^2).

usage: export_to_rust.py STATE.pt ATTN_DIR OUT_DIR ARM [--all-on] [--model MODEL]"""
import sys, json, os, shutil
import numpy as np, scipy.linalg as sl, torch

state_path, attn_dir, out, arm = sys.argv[1:5]
all_on = '--all-on' in sys.argv
model = sys.argv[sys.argv.index('--model') + 1] if '--model' in sys.argv else '/Users/user/mpd-data/engine/vpd4l_pile2p27'
S = torch.load(state_path, map_location='cpu', weights_only=False)
KINDS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'c_fc', 'down_proj')
sites = [f"h.{l}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}" for l in range(4) for k in KINDS]
index = {n: i for i, n in enumerate(sites)}
os.makedirs(out, exist_ok=False)
attn_record = json.load(open(os.path.join(attn_dir, 'export.json')))
heads = S.get('attn') or {}
rot = S.get('rot') or {}
rot_maps, rot_blocks = {}, {}
if rot:
    record = json.load(open(os.path.join(model, 'export.json')))
    weight = lambda name: np.fromfile(os.path.join(model, f'{name}.f64'), dtype='<f8').reshape(record['files'][name]['shape'])
    for l, R in rot.items():
        l = int(l)
        Wf, Wd = weight(f'blocks.{l}.mlp.c_fc'), weight(f'blocks.{l}.mlp.down_proj')       # [3072, 768], [768, 3072]
        perm, A = R['perm'].long().numpy(), R['A'].double().numpy()
        ng, g = A.shape[0], A.shape[1]
        upper = np.triu(np.ones((g, g)), 1)
        Vf, Uf = np.zeros((Wf.shape[1], ng * g)), np.zeros((ng * g, Wf.shape[0]))
        Vd, Ud = np.zeros((Wd.shape[1], ng * g)), np.zeros((ng * g, Wd.shape[0]))
        for n in range(ng):
            G = perm[n * g:(n + 1) * g]
            S_ = A[n] * upper
            Q = sl.expm(S_ - S_.T)                                                        # [g, g], columns Q_i
            k = slice(n * g, (n + 1) * g)
            Vf[:, k] = Wf[G].T @ Q
            Uf[k][:, G] = Q.T
            Vd[G, k] = Q
            Ud[k] = (Wd[:, G] @ Q).T
        norm = np.maximum(np.linalg.norm(Ud, axis=1), 1e-30)
        Vd, Ud = Vd * norm[None, :], Ud / norm[:, None]                                  # |v'.a| = |Q_i . a_G| ||W_dn,G Q_i||
        rot_maps[f'h.{l}.mlp.c_fc'], rot_maps[f'h.{l}.mlp.down_proj'] = (Vf, Uf), (Vd, Ud)
        block = R['L'].double().argmax(-1).numpy()                                        # [ng, g]: each slice's block
        tau, s_ = R['tau'].double().numpy(), R['s'].double().numpy()
        rot_blocks[l] = [(n, j, [n * g + i for i in range(g) if block[n, i] == j], float(tau[n, j]), float(s_[n, j])) for n in range(ng) for j in range(g) if (block[n] == j).any()]
files = {}


def head_slices(n):
    """Map n's head slices as full-map reads [d_in, H C] and writes [H C, d_out], head-major (slice h C + i)."""
    V, U = heads[n]['V'].double(), heads[n]['U'].double()
    H, C = V.shape[0], V.shape[-1]
    if n.endswith('o_proj'):
        hd = V.shape[1]
        Vf = torch.zeros(H * hd, H * C, dtype=torch.float64)
        for h in range(H):
            Vf[h * hd:(h + 1) * hd, h * C:(h + 1) * C] = V[h]
        return Vf, U.reshape(H * C, -1)
    hd = U.shape[-1]
    Uf = torch.zeros(H * C, H * hd, dtype=torch.float64)
    for h in range(H):
        Uf[h * C:(h + 1) * C, h * hd:(h + 1) * hd] = U[h]
    return V.permute(1, 0, 2).reshape(V.shape[1], H * C), Uf


for n in sites:
    if '.attn.' in n and n in heads:
        V, U = head_slices(n)
        norm = U.norm(dim=1).clamp_min(1e-30)
        V, U = V * norm[None, :], U / norm[:, None]                           # |v'.x| = |c| ||u||
        for w, t in (('U', U), ('V', V)):
            t.numpy().astype('<f8').tofile(os.path.join(out, f'{n}.{w}.f64'))
            files[f'{n}.{w}'] = {'shape': list(t.shape)}
        continue
    if '.attn.' in n:
        for w in ('U', 'V'):
            shutil.copyfile(os.path.join(attn_dir, f'{n}.{w}.f64'), os.path.join(out, f'{n}.{w}.f64'))
            files[f'{n}.{w}'] = {'shape': attn_record['files'][f'{n}.{w}']['shape']}
        continue
    if n in rot_maps:
        for w, t in (('U', rot_maps[n][1]), ('V', rot_maps[n][0])):
            t.astype('<f8').tofile(os.path.join(out, f'{n}.{w}.f64'))
            files[f'{n}.{w}'] = {'shape': list(t.shape)}
        continue
    m = S['maps'][n]
    V, U = m['V'].double(), m['U'].double()                                   # [d_in, C], [C, d_out]
    norm = U.norm(dim=1).clamp_min(1e-30)
    if n not in S['tied'] and S['start'] != 'neuron':
        V, U = V * norm[None, :], U / norm[:, None]                           # |v'.x| = |v.x| ||u||
    for w, t in (('U', U), ('V', V)):
        t.numpy().astype('<f8').tofile(os.path.join(out, f'{n}.{w}.f64'))
        files[f'{n}.{w}'] = {'shape': list(t.shape)}
json.dump({'source': {'descent': state_path, 'step': S['step'], 'attention': 'trained' if heads else attn_dir},
           'config': {'sites': sites, 'subcomponents': {n: files[f'{n}.U']['shape'][0] for n in sites}},
           'files': files}, open(os.path.join(out, 'export.json'), 'w'), indent=1)

components = []
# library_vpd needs a component at every stage of every layer: each attention map is one component of
# all its slices, always on (the MLP-scoped run uses M's attention there anyway).
for n in sites:
    if '.attn.' in n and n in heads:
        tau, s = heads[n]['tau'].double().reshape(-1), heads[n]['s'].double().reshape(-1)
        for i in range(tau.numel()):
            components.append({'read': {'own': [index[n], i]}, 'tau': -1e9 if all_on else float(tau[i]), 'width': float(s[i]), 'slices': [[index[n], i]]})
    elif '.attn.' in n:
        C, d = files[f'{n}.U']['shape'][0], files[f'{n}.V']['shape'][0]
        read = {'direction': {'site': index[n], 'coefficients': [0.0] * (d + 1)}} if S['start'] == 'neuron' else {'own': [index[n], 0]}
        components.append({'read': read, 'tau': -1e9, 'width': 1.0, 'slices': [[index[n], i] for i in range(C)]})
for l, blocks in rot_blocks.items():
    fc, dn = index[f'h.{l}.mlp.c_fc'], index[f'h.{l}.mlp.down_proj']
    for _, _, members, tau, s_ in blocks:
        components.append({'read': {'active': {'site': fc}}, 'tau': -1e9 if all_on else tau, 'width': s_, 'slices': [[fc, i] for i in members] + [[dn, i] for i in members]})
tied = {fc: (dn, own) for dn, (fc, own) in S['tied'].items()}
for n in sites:
    if '.attn.' in n or n in S['tied'] or n in rot_maps:
        continue
    m = S['maps'][n]
    tau, s = m['tau'].double(), m['s'].double()
    for i in range(m['V'].shape[1]):
        t = -1e9 if all_on else float(tau[i])
        if S['start'] == 'neuron':
            dn, own = tied[n]
            j = int(torch.nonzero(own == i)[0]) if (own == i).any() else None
            g = m['V'][:, i].double().tolist()
            comp = {'read': {'direction': {'site': index[n], 'coefficients': g + [0.0]}}, 'tau': t, 'width': float(s[i]),
                    'slices': [[index[n], i]] + ([[index[dn], j]] if j is not None else [])}
        else:
            comp = {'read': {'own': [index[n], i]}, 'tau': t, 'width': float(s[i]), 'slices': [[index[n], i]]}
        components.append(comp)
json.dump([{'arm': arm, 'components': components}], open(os.path.join(out, 'start.json'), 'w'))
print(out, len(components), 'components', 'all on' if all_on else 'trained gates')
