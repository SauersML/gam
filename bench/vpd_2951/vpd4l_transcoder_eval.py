"""vpd4l with its MLPs replaced by ReLU transcoder files (layer_{l}.safetensors, circuit-tracer layout)
at every position after each sequence's first (position 0 keeps M's MLP, as the library's block
does), on held-out rows [1024, 1056) of vpd4l_pile2p27 (the fit's held-out rows), context 512.

Per layer, on M's own MLP inputs: FVU of the MLP output, active features per token (positions >= 1).
KL(M || P) in bits per token (all 512 positions) and top-1 agreement with one layer replaced and
with all four replaced; active features per token per layer in the all-replaced run.

usage: vpd4l_transcoder_eval.py TC_DIR OUT.json
"""
import json, math, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vpd_model import load_target, rms, gelu_tanh
from safetensors.torch import load_file

tc_dir, out_path = sys.argv[1], sys.argv[2]
dev = 'mps'
T = load_target(dev)
H, hd, eps, L = T.n_head, T.hd, T.eps, 4
W = lambda i, k: T.site(f"h.{i}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}").W
tc = [{k: v.to(dev).float() for k, v in load_file(f'{tc_dir}/layer_{i}.safetensors').items()} for i in range(L)]
tok = np.memmap(os.path.expanduser('~/mpd-data/engine/vpd4l_pile2p27/tokens.f64'), dtype='<f8', mode='r', shape=(490468, 513))
ids_all = torch.from_numpy(np.asarray(tok[1024:1056, :512]).astype(np.int64)).to(dev)


@torch.no_grad()
def run(ids, replaced):
    B, Tn = ids.shape
    x = T.wte[ids]
    stats = []
    for i in range(L):
        h = rms(x, T.norms[2 * i], eps)
        q = (h @ W(i, 'q_proj').T).view(B, Tn, H, hd).transpose(1, 2)
        k = (h @ W(i, 'k_proj').T).view(B, Tn, H, hd).transpose(1, 2)
        v = (h @ W(i, 'v_proj').T).view(B, Tn, H, hd).transpose(1, 2)
        q, k = T._rope(q, Tn), T._rope(k, Tn)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + y.transpose(1, 2).reshape(B, Tn, -1) @ W(i, 'o_proj').T
        hn = rms(x, T.norms[2 * i + 1], eps)
        m = gelu_tanh(hn @ W(i, 'c_fc').T) @ W(i, 'down_proj').T
        a = torch.relu(hn @ tc[i]['W_enc'].T + tc[i]['b_enc'])
        t = a @ tc[i]['W_dec'] + tc[i]['b_dec']
        later = (m[:, 1:], t[:, 1:])
        fvu = float((later[0] - later[1]).pow(2).sum(-1).mean() / (later[0] - later[0].reshape(-1, m.shape[-1]).mean(0)).pow(2).sum(-1).mean())
        stats.append({'fvu': fvu, 'l0': float((a[:, 1:] > 0).float().sum(-1).mean()), 'fired': (a[:, 1:] > 0).reshape(-1, a.shape[-1]).any(0).cpu()})
        if i in replaced:
            m = torch.cat([m[:, :1], t[:, 1:]], dim=1)
        x = x + m
    return rms(x, T.ln_f, eps) @ T.wte.T, stats


def compare(replaced):
    kl, agree, n, l0 = 0.0, 0.0, 0, np.zeros(L)
    fvu = np.zeros(L)
    fired = [torch.zeros(tc[i]['W_enc'].shape[0], dtype=torch.bool) for i in range(L)]
    per_token = []
    for b in range(0, 32, 4):
        ids = ids_all[b:b + 4]
        lm, _ = run(ids, set())
        lp, st = run(ids, replaced)
        pm, lpm, lpp = lm.softmax(-1), lm.log_softmax(-1), lp.log_softmax(-1)
        k = (pm * (lpm - lpp)).sum(-1) / math.log(2)
        per_token.append(k.flatten().cpu())
        kl += float(k.sum()); n += k.numel()
        agree += float((lm.argmax(-1) == lp.argmax(-1)).float().sum())
        l0 += np.array([s['l0'] for s in st]) / 8
        fvu += np.array([s['fvu'] for s in st]) / 8
        for i in range(L):
            fired[i] |= st[i]['fired']
    pt = torch.cat(per_token)
    return {'replaced': sorted(replaced), 'kl_bits_per_token': kl / n, 'kl_p99': float(pt.quantile(0.99)), 'top1_agreement': agree / n,
            'active_per_token': l0.tolist(), 'fvu_at_inputs_of_this_run': fvu.tolist(), 'features_fired': [int(f.sum()) for f in fired]}


res = {'tc_dir': tc_dir, 'rows': [1024, 1056], 'context': 512, 'width': [int(t['W_enc'].shape[0]) for t in tc]}
res['M_inputs'] = compare(set())
res['single'] = [compare({i}) for i in range(L)]
res['all'] = compare(set(range(L)))
for key in ('M_inputs', 'all'):
    r = res[key]
    print(key, 'KL', round(r['kl_bits_per_token'], 4), 'p99', round(r['kl_p99'], 3), 'top1', round(r['top1_agreement'], 4), 'L0', [round(v, 2) for v in r['active_per_token']], 'FVU', [round(v, 4) for v in r['fvu_at_inputs_of_this_run']], 'fired', r['features_fired'], flush=True)
for r in res['single']:
    print('single', r['replaced'], 'KL', round(r['kl_bits_per_token'], 4), 'top1', round(r['top1_agreement'], 4), flush=True)
json.dump(res, open(out_path, 'w'), indent=1)
