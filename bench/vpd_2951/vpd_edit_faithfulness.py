"""Edit faithfulness on vpd4l, one code path for VPD and for a transcoder library (prototype of the
edits driver's families, interchange.rs Patch::Part): an experiment takes a held-out row, a position
t >= 1, an MLP layer l, a part active there in P, and a factor alpha in {0 (remove), 0.5, 2, 3}, and
scales the part's write by alpha at row t only, identically on M and on P:
  - transcoder feature i (read g_i, c_i, write u_i): P_e scales its activation at (l, t); M_e adds
    (alpha - 1) relu(g_i . x + c_i) u_i to M's MLP output at (l, t), x being M_e's own MLP input.
  - VPD subcomponent i of site s (c_fc or down_proj of layer l; read V_i, write U_i): P_e (VPD at
    its causal-importance masks) scales mask_i at t by alpha; M_e is M with that subcomponent's
    rank-one term (x . V_i) U_i scaled by alpha at t (mask 1 elsewhere, the delta term kept).
Score: KL(M_e || P_e) in bits per token over positions t..511, and the same positions' KL(M || P)
without the edit (the clean error there). Rows [1024, 1056) of vpd4l_pile2p27 (held out).

usage: vpd_edit_faithfulness.py vpd|vpd_mlp|TC_DIR OUT.json EXPERIMENTS_PER_ROW
(vpd_mlp: VPD's MLP sites at its masks, attention exact.) The transcoder path cross-checks this script
against mpd_library_mdl_2951's edits driver (#2951), which scores transcoder libraries in Rust.
"""
import json, math, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vpd_model import load_target, load_vpd, rms, gelu_tanh
from safetensors.torch import load_file

which, out_path, per_row = sys.argv[1], sys.argv[2], int(sys.argv[3])
dev = 'mps'
T = load_target(dev)
H, hd, eps, L = T.n_head, T.hd, T.eps, 4
W = lambda i, k: T.site(f"h.{i}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}").W
tok = np.memmap(os.path.expanduser('~/mpd-data/engine/vpd4l_pile2p27/tokens.f64'), dtype='<f8', mode='r', shape=(490468, 513))
rows = torch.from_numpy(np.asarray(tok[1024:1056, :512]).astype(np.int64)).to(dev)
FACTORS = [0.0, 0.5, 2.0, 3.0]
rng = np.random.default_rng(7)
LN2 = math.log(2)

if which in ('vpd', 'vpd_mlp'):
    V = load_vpd(T, dev)
else:
    tc = [{k: v.to(dev).float() for k, v in load_file(f'{which}/layer_{i}.safetensors').items()} for i in range(L)]


def kl_rows(lm, lp):
    return (lm.softmax(-1) * (lm.log_softmax(-1) - lp.log_softmax(-1))).sum(-1) / LN2


@torch.no_grad()
def tc_forward(ids, replace, edits=None, collect=False):
    """M (replace False) or the transcoder P (all MLPs replaced after position 0); edits per batch
    element e: (layer, t, feature, alpha)."""
    B, Tn = ids.shape
    x = T.wte[ids]
    acts = []
    for i in range(L):
        h = rms(x, T.norms[2 * i], eps)
        q = (h @ W(i, 'q_proj').T).view(B, Tn, H, hd).transpose(1, 2)
        k = (h @ W(i, 'k_proj').T).view(B, Tn, H, hd).transpose(1, 2)
        v = (h @ W(i, 'v_proj').T).view(B, Tn, H, hd).transpose(1, 2)
        q, k = T._rope(q, Tn), T._rope(k, Tn)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + y.transpose(1, 2).reshape(B, Tn, -1) @ W(i, 'o_proj').T
        hn = rms(x, T.norms[2 * i + 1], eps)
        if collect:
            acts.append(torch.relu(hn @ tc[i]['W_enc'].T + tc[i]['b_enc']))
        if replace:
            a = torch.relu(hn @ tc[i]['W_enc'].T + tc[i]['b_enc'])
            if edits:
                for e, (l, t, f, al) in enumerate(edits):
                    if l == i:
                        a[e, t, f] = a[e, t, f] * al
            m = a @ tc[i]['W_dec'] + tc[i]['b_dec']
            m0 = gelu_tanh(hn[:, :1] @ W(i, 'c_fc').T) @ W(i, 'down_proj').T
            m = torch.cat([m0, m[:, 1:]], dim=1)
        else:
            m = gelu_tanh(hn @ W(i, 'c_fc').T) @ W(i, 'down_proj').T
            if edits:
                for e, (l, t, f, al) in enumerate(edits):
                    if l == i:
                        coef = torch.relu(hn[e, t] @ tc[i]['W_enc'][f] + tc[i]['b_enc'][f])
                        m[e, t] = m[e, t] + (al - 1.0) * coef * tc[i]['W_dec'][f]
        x = x + m
    return rms(x, T.ln_f, eps) @ T.wte.T, acts


results, pooled = [], {}
E = 8
for start in range(0, 32, E):
    ids = rows[start:start + E]
    for rep in range(per_row):
        if which in ('vpd', 'vpd_mlp'):
            lm, ci = V.target_and_ci(ids)
            if which == 'vpd_mlp':
                # Attention exact (all-ones masks with the delta term), MLP sites at VPD's masks.
                ci = {n: (c if '.mlp.' in n else torch.ones_like(c)) for n, c in ci.items()}
                dP = {n: (torch.zeros if '.mlp.' in n else torch.ones)(c.shape[:2], device=dev) for n, c in ci.items()}
            else:
                dP = None
            masks = {n: c.clone() for n, c in ci.items()}
            lp = V.masked(ids, masks, dP)
            if start == 0 and rep == 0:
                ones = {n: torch.ones_like(c) for n, c in ci.items()}
                print('check: KL(M || M through all-ones masks + delta) =', float(kl_rows(lm, V.masked(ids, ones, {n: torch.ones(c.shape[:2], device=dev) for n, c in ci.items()})).mean()), 'clean KL(M||VPD)', float(kl_rows(lm, lp).mean()), flush=True)
        else:
            lm, acts = tc_forward(ids, False, collect=True)
            lp, _ = tc_forward(ids, True)
        clean = kl_rows(lm, lp)
        specs = []
        for e in range(ids.shape[0]):
            t, l = int(rng.integers(1, 512)), int(rng.integers(0, L))
            al = 0.0 if rng.integers(0, 2) == 0 else FACTORS[1 + int(rng.integers(0, 3))]
            if which in ('vpd', 'vpd_mlp'):
                s = f"h.{l}.mlp.{['c_fc', 'down_proj'][int(rng.integers(0, 2))]}"
                row = ci[s][e, t]
            else:
                s = l
                row = acts[l][e, t]
            active = (row > 0).nonzero().flatten().tolist()
            cand = sorted(set(active) | {int(rng.integers(0, row.shape[0]))})
            specs.append((s, l, t, cand[int(rng.integers(0, len(cand)))], al, len(active)))
        if which in ('vpd', 'vpd_mlp'):
            mP = {n: c.clone() for n, c in ci.items()}
            mM = {n: torch.ones_like(c) for n, c in ci.items()}
            dM = {n: torch.ones(c.shape[:2], device=dev) for n, c in ci.items()}
            for e, (s, l, t, i, al, _) in enumerate(specs):
                mP[s][e, t, i] *= al
                mM[s][e, t, i] = al
            lme = V.masked(ids, mM, dM)
            lpe = V.masked(ids, mP, dP)
        else:
            ed = [(l, t, i, al) for (s, l, t, i, al, _) in specs]
            lme, _ = tc_forward(ids, False, ed)
            lpe, _ = tc_forward(ids, True, ed)
        kle = kl_rows(lme, lpe)
        moved = kl_rows(lme, lm)  # KL(M_e || M): how much the edit changes M itself
        for e, (s, l, t, i, al, nact) in enumerate(specs):
            fam = 'remove' if al == 0.0 else 'amplify'
            pooled.setdefault(fam, []).append(kle[e, t:].cpu())
            pooled.setdefault('effect_' + fam, []).append(moved[e, t:].cpu())
            if rep == 0:
                pooled.setdefault('clean', []).append(clean[e].cpu())
            results.append({'row': 1024 + start + e, 'site': str(s), 'layer': l, 't': t, 'part': i, 'alpha': al, 'active': nact,
                            'kl_edit': float(kle[e, t:].mean()), 'kl_clean_same_positions': float(clean[e, t:].mean()),
                            'edit_effect_on_M': float(moved[e, t:].mean()), 'kl_edit_p99_tokens': float(kle[e, t:].quantile(0.99)),
                            'kl_edit_at_t': float(kle[e, t]), 'kl_clean_at_t': float(clean[e, t]), 'edit_effect_on_M_at_t': float(moved[e, t])})
    print(start, len(results), flush=True)

out = {'model': which, 'rows': [1024, 1056], 'experiments': results}
# Pooled as the edits driver pools: every scored token from t on, and the edited tokens alone.
for fam, vals in pooled.items():
    allv = torch.cat(vals)
    first = torch.stack([v[0] for v in vals])
    out['pooled_' + fam] = {'experiments': len(vals), 'tokens': int(allv.numel()), 'mean_bits_per_token': float(allv.mean()), 'p99_bits_per_token': float(allv.quantile(0.99)),
                           'edited_token_mean_bits': float(first.mean()), 'edited_token_p99_bits': float(first.quantile(0.99))}
    print('pooled', fam, out['pooled_' + fam], flush=True)
for name, sel in (('remove', lambda r: r['alpha'] == 0.0), ('amplify', lambda r: r['alpha'] != 0.0), ('all', lambda r: True)):
    rs = [r for r in results if sel(r)]
    out[name] = {k: float(np.mean([r[k] for r in rs])) for k in ('kl_edit', 'kl_clean_same_positions', 'edit_effect_on_M', 'kl_edit_at_t', 'kl_clean_at_t', 'edit_effect_on_M_at_t')}
    out[name]['n'] = len(rs)
    out[name]['kl_edit_p99_experiments'] = float(np.quantile([r['kl_edit'] for r in rs], 0.99))
    print(name, out[name], flush=True)
json.dump(out, open(out_path, 'w'), indent=1)
