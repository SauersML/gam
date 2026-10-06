"""Why the vpd4l transcoder library got denser in its fit (compare-vpd4l-tc4096-n2p24): checkpoints A
(start) and B (a later epoch) of mpd_library_mdl_2951, per layer:
  - thresholds: r_i = c_i / |g_i| (feature i fires where g_i.x / |g_i| > -r_i), quantiles at A and B;
  - firing frequency per feature on held-out rows [1024, 1056) after position 0, on M's own MLP inputs
    and on P's own inputs; how many features rise more than 10x (and from 0);
  - counterfactual firing on M's inputs with B's c and A's g, and A's c and B's g, and with B's g
    rescaled to A's norms (direction only), to attribute the growth to c, |g| or the direction of g;
  - the description KL(q || p) per layer and operator under the fit's empirical-Bayes prior (per group
    v_G = mean(mu^2 + sigma^2): the gate group = a gate row with its bias, the output group = a column),
    checked against the checkpoint's own total;
  - the data terms per layer: KL(M || P) in bits/token on the held-out rows at the posterior mean and at
    weight samples (mu + sigma eps, 4 draws) with only layer l taken from B (the rest from A), and all
    layers from A and from B.
usage: diag_dense.py A.bin B.bin FIT_OUT OUT.json"""
import json, math, struct, sys, os
import numpy as np, torch
sys.path.insert(0, '/Users/user/gam/bench/vpd_2951')
from vpd_model import load_target, rms, gelu_tanh
from safetensors import safe_open

pa, pb, fit_out, out_path = sys.argv[1:5]
dev = 'mps'
LN2 = math.log(2)


def read_ckpt(path):
    f = open(path, 'rb')
    n = struct.unpack('<Q', f.read(8))[0]
    h = json.loads(f.read(n))
    size = {'F32': 4, 'F64': 8, 'Bf16': 2}
    ops = []
    for (r, c) in h['shapes']:
        arrs = []
        for p in h['precision']:
            raw = f.read(r * c * size[p])
            if p == 'F32':
                a = np.frombuffer(raw, '<f4').reshape(r, c).astype(np.float64)
            elif p == 'F64':
                a = np.frombuffer(raw, '<f8').reshape(r, c)
            else:
                a = (np.frombuffer(raw, '<u2').astype(np.uint32) << 16).view(np.float32).reshape(r, c).astype(np.float64)
            arrs.append(a)
        ops.append({'mean': arrs[0], 'log_sd': arrs[1]})
    return h, ops


ha, A = read_ckpt(pa)
hb, B = read_ckpt(pb)
L = 4
bias = []
for l in range(L):
    with safe_open(f'{fit_out}/transcoder_l{l}.safetensors', 'np') as sf:
        bias.append(sf.get_tensor('bias').astype(np.float64).reshape(-1))
        gate_file = sf.get_tensor('gate').astype(np.float64)
    assert np.abs(gate_file - A[3 * l]['mean']).max() < 1e-6, 'start gate is not the kept file'
lay = lambda ops, l: (ops[3 * l], ops[3 * l + 1], ops[3 * l + 2])  # gate [k,d], gate_bias [k,1], out [d,k]
res = {'A': pa, 'B': pb, 'epochs_B': hb['epoch'], 'layers': []}

# Description: per group v = mean(mu^2 + s^2); KL = 1/2 sum ln(v / s^2) + 1/2 sum ((mu^2 + s^2)/v - 1).
def description(ops):
    out = []
    for l in range(L):
        g, c, u = lay(ops, l)
        mg = np.concatenate([g['mean'], c['mean']], 1); sg2 = np.exp(2 * np.concatenate([g['log_sd'], c['log_sd']], 1))
        mu, su2 = u['mean'], np.exp(2 * u['log_sd'])
        def kl(m, s2, axis):
            v = (m ** 2 + s2).mean(axis, keepdims=True)
            return 0.5 * (np.log(v / s2) + (m ** 2 + s2) / v - 1).sum() / LN2
        out.append({'gate_bits': float(kl(mg, sg2, 1)), 'out_bits': float(kl(mu, su2, 0))})
    return out


dA, dB = description(A), description(B)
print('description total A', sum(x['gate_bits'] + x['out_bits'] for x in dA), 'reported', ha['start']['divergence_bits'] if ha.get('start') else None)
print('description total B', sum(x['gate_bits'] + x['out_bits'] for x in dB), 'reported', hb['epochs'][-1]['held_out']['divergence_bits'] if hb['epochs'] else None)

T = load_target(dev)
H, hd, eps = T.n_head, T.hd, T.eps
W = lambda i, k: T.site(f"h.{i}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}").W
tok = np.memmap(os.path.expanduser('~/mpd-data/engine/vpd4l_pile2p27/tokens.f64'), dtype='<f8', mode='r', shape=(490468, 513))
rows = torch.from_numpy(np.asarray(tok[1024:1056, :512]).astype(np.int64)).to(dev)
t32 = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)


def params(ops, l, sample=None):
    g, c, u = lay(ops, l)
    if sample is None:
        return t32(g['mean']), t32(c['mean'][:, 0]), t32(u['mean'].T), t32(bias[l])
    rng = sample
    draw = lambda o: o['mean'] + np.exp(o['log_sd']) * rng.standard_normal(o['mean'].shape)
    return t32(draw(g)), t32(draw(c)[:, 0]), t32(draw(u).T), t32(bias[l])


@torch.no_grad()
def run(ids, layer_params, collect=False):
    """P with layer l's MLP the transcoder `layer_params[l]` after position 0 (None: M's MLP)."""
    Bn, Tn = ids.shape
    x = T.wte[ids]
    xs = []
    for i in range(L):
        h = rms(x, T.norms[2 * i], eps)
        q = T._rope((h @ W(i, 'q_proj').T).view(Bn, Tn, H, hd).transpose(1, 2), Tn)
        k = T._rope((h @ W(i, 'k_proj').T).view(Bn, Tn, H, hd).transpose(1, 2), Tn)
        v = (h @ W(i, 'v_proj').T).view(Bn, Tn, H, hd).transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + y.transpose(1, 2).reshape(Bn, Tn, -1) @ W(i, 'o_proj').T
        hn = rms(x, T.norms[2 * i + 1], eps)
        if collect:
            xs.append(hn[:, 1:].reshape(-1, hn.shape[-1]))
        m = gelu_tanh(hn @ W(i, 'c_fc').T) @ W(i, 'down_proj').T
        if layer_params[i] is not None:
            g, c, ud, b = layer_params[i]
            t = torch.relu(hn @ g.T + c) @ ud + b
            m = torch.cat([m[:, :1], t[:, 1:]], 1)
        x = x + m
    return rms(x, T.ln_f, eps) @ T.wte.T, xs


def kl_bits(lp_fn):
    tot, n = 0.0, 0
    for b in range(0, 32, 8):
        ids = rows[b:b + 8]
        lm, _ = run(ids, [None] * L)
        lp, _ = run(ids, lp_fn())
        pm = lm.softmax(-1)
        tot += float((pm * (lm.log_softmax(-1) - lp.log_softmax(-1))).sum() / LN2); n += lm.shape[0] * lm.shape[1]
    return tot / n


# Inputs: M's own MLP inputs, and P's own (all layers from A, and all from B).
def inputs(layer_params):
    xs = [[] for _ in range(L)]
    for b in range(0, 32, 8):
        _, x = run(rows[b:b + 8], layer_params, collect=True)
        for l in range(L):
            xs[l].append(x[l])
    return [torch.cat(v) for v in xs]


PA = [params(A, l) for l in range(L)]
PB = [params(B, l) for l in range(L)]
per_layer_data = {}
# Data terms.
res['kl_mean_all_A'] = kl_bits(lambda: PA)
res['kl_mean_all_B'] = kl_bits(lambda: PB)
rng = np.random.default_rng(0)
res['kl_sample_all_A'] = float(np.mean([kl_bits(lambda: [params(A, l, rng) for l in range(L)]) for _ in range(2)]))
res['kl_sample_all_B'] = float(np.mean([kl_bits(lambda: [params(B, l, rng) for l in range(L)]) for _ in range(2)]))
print('all layers: mean A', res['kl_mean_all_A'], 'mean B', res['kl_mean_all_B'], 'sample A', res['kl_sample_all_A'], 'sample B', res['kl_sample_all_B'], flush=True)
for l in range(L):
    e = per_layer_data.setdefault(l, {})
    e['kl_mean_only_this_layer_B'] = kl_bits(lambda: [PB[i] if i == l else PA[i] for i in range(L)])
    e['kl_sample_only_this_layer_B'] = float(np.mean([kl_bits(lambda: [params(B, i, rng) if i == l else PA[i] for i in range(L)]) for _ in range(2)]))
    e['kl_sample_only_this_layer_A'] = float(np.mean([kl_bits(lambda: [params(A, i, rng) if i == l else PA[i] for i in range(L)]) for _ in range(2)]))
    print('layer', l, 'mean with only l from B', e['kl_mean_only_this_layer_B'], 'sample only l: A', e['kl_sample_only_this_layer_A'], 'B', e['kl_sample_only_this_layer_B'], flush=True)
XM, XA, XB = inputs([None] * L), inputs(PA), inputs(PB)
freq = lambda x, g, c: (x @ g.T + c > 0).float().mean(0)
for l in range(L):
    gA, cA, uA, _ = PA[l]
    gB, cB, uB, _ = PB[l]
    nA, nB = gA.norm(dim=1), gB.norm(dim=1)
    rA, rB = (cA / nA).cpu().numpy(), (cB / nB).cpu().numpy()
    fA, fB = freq(XM[l], gA, cA), freq(XM[l], gB, cB)
    f_cB = freq(XM[l], gA, cB)                     # A's g, B's c
    f_gB = freq(XM[l], gB, cA)                     # B's g, A's c
    gdir = gB / nB[:, None] * nA[:, None]          # B's direction at A's norm, A's c
    f_dir = freq(XM[l], gdir, cA)
    f_norm = freq(XM[l], gA / nA[:, None] * nB[:, None], cA)   # A's direction at B's norm, A's c
    rose = ((fB > 10 * fA) & (fB > 0)).cpu().numpy()
    from0 = ((fA == 0) & (fB > 0)).sum().item()
    q = lambda a: [float(np.quantile(a, p)) for p in (0.05, 0.25, 0.5, 0.75, 0.95)]
    cos = (gA * gB).sum(1) / (nA * nB)
    sel = torch.tensor(rose, device=dev)
    entry = {
        'layer': l, 'features': int(len(rA)),
        'ratio_c_over_norm_g_quantiles_A': q(rA), 'ratio_c_over_norm_g_quantiles_B': q(rB),
        'active_per_token_M_inputs': [float(fA.sum()), float(fB.sum())],
        'active_per_token_own_inputs': [float(freq(XA[l], gA, cA).sum()), float(freq(XB[l], gB, cB).sum())],
        'features_rising_10x': int(rose.sum()), 'features_from_zero': int(from0),
        'active_per_token_counterfactual_M_inputs': {'A': float(fA.sum()), 'A_g_B_c': float(f_cB.sum()), 'B_g_A_c': float(f_gB.sum()),
                                                    'B_direction_A_norm_A_c': float(f_dir.sum()), 'A_direction_B_norm_A_c': float(f_norm.sum()), 'B': float(fB.sum())},
        'median_over_rising': {'c_A': float(cA[sel].median()) if rose.any() else None, 'c_B': float(cB[sel].median()) if rose.any() else None,
                               'norm_g_A': float(nA[sel].median()) if rose.any() else None, 'norm_g_B': float(nB[sel].median()) if rose.any() else None,
                               'cos_gA_gB': float(cos[sel].median()) if rose.any() else None,
                               'norm_u_ratio_B_over_A': float((uB.norm(dim=1) / uA.norm(dim=1).clamp_min(1e-12))[sel].median()) if rose.any() else None},
        'median_all': {'c_A': float(cA.median()), 'c_B': float(cB.median()), 'norm_g_A': float(nA.median()), 'norm_g_B': float(nB.median()), 'cos_gA_gB': float(cos.median()),
                       'norm_u_ratio_B_over_A': float((uB.norm(dim=1) / uA.norm(dim=1).clamp_min(1e-12)).median())},
        'description_bits_A': dA[l], 'description_bits_B': dB[l],
    }
    res['layers'].append(entry)
    print(json.dumps(entry), flush=True)

for l in range(L):
    res['layers'][l].update(per_layer_data[l])
res['kl_mean_all_A_recheck'] = kl_bits(lambda: PA)
print('recheck mean A', res['kl_mean_all_A_recheck'], flush=True)
json.dump(res, open(out_path, 'w'), indent=1)
