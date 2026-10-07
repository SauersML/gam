"""Export a budget_descent.py explanation (DESCENT_SAVE) to library_vpd's importer (#2951, descent):
a decomposition directory in VPD's export layout (export.json, per site U [C, d_out] and V [d_in, C]
as little-endian float64, all 24 sites in M's order) and a start file with one arm of components,
for mpd_library_mdl_2951's `vpd` setting, scored by the unchanged edits driver.

MLP maps carry the prototype's slices. Its own-read gate Phi((|v.x| ||u|| - tau)/s) is library_vpd's
own read ||V_b^T x|| - tau_b at width s once each slice is rescaled to v ||u||, u/||u|| (the same
rank-one map). The neuron start's gate on the signed pre-activation w_j . x is a direction gate
(g = w_j, c = 0) over the neuron's c_fc and down_proj slices. Attention maps are copied from an exact
decomposition (ATTN_DIR); a run scoped to the MLP blocks (`blocks` 1, 3, 5, 7) keeps M's attention.
With --all-on every threshold is -1e9: every component on, so P is M (a gap of 0 checks the import).

usage: export_to_rust.py STATE.pt ATTN_DIR OUT_DIR ARM [--all-on]"""
import sys, json, os, shutil
import numpy as np, torch

state_path, attn_dir, out, arm = sys.argv[1:5]
all_on = '--all-on' in sys.argv
S = torch.load(state_path, map_location='cpu', weights_only=False)
KINDS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'c_fc', 'down_proj')
sites = [f"h.{l}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}" for l in range(4) for k in KINDS]
index = {n: i for i, n in enumerate(sites)}
os.makedirs(out, exist_ok=False)
attn_record = json.load(open(os.path.join(attn_dir, 'export.json')))
files = {}
for n in sites:
    if '.attn.' in n:
        for w in ('U', 'V'):
            shutil.copyfile(os.path.join(attn_dir, f'{n}.{w}.f64'), os.path.join(out, f'{n}.{w}.f64'))
            files[f'{n}.{w}'] = {'shape': attn_record['files'][f'{n}.{w}']['shape']}
        continue
    m = S['maps'][n]
    V, U = m['V'].double(), m['U'].double()                                   # [d_in, C], [C, d_out]
    norm = U.norm(dim=1).clamp_min(1e-30)
    if n not in S['tied'] and S['start'] != 'neuron':
        V, U = V * norm[None, :], U / norm[:, None]                           # |v'.x| = |v.x| ||u||
    for w, t in (('U', U), ('V', V)):
        t.numpy().astype('<f8').tofile(os.path.join(out, f'{n}.{w}.f64'))
        files[f'{n}.{w}'] = {'shape': list(t.shape)}
json.dump({'source': {'descent': state_path, 'step': S['step'], 'attention': attn_dir},
           'config': {'sites': sites, 'subcomponents': {n: files[f'{n}.U']['shape'][0] for n in sites}},
           'files': files}, open(os.path.join(out, 'export.json'), 'w'), indent=1)

components = []
# library_vpd needs a component at every stage of every layer: each attention map is one component of
# all its slices, always on (the MLP-scoped run uses M's attention there anyway).
for n in sites:
    if '.attn.' in n:
        C = files[f'{n}.U']['shape'][0]
        components.append({'read': {'own': [index[n], 0]}, 'tau': -1e9, 'width': 1.0, 'slices': [[index[n], i] for i in range(C)]})
tied = {fc: (dn, own) for dn, (fc, own) in S['tied'].items()}
for n in sites:
    if '.attn.' in n or n in S['tied']:
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
