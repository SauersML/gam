"""Measured removal of single transcoder features at one layer (no linearization).

Model: Qwen3-0.6B with layer L's MLP replaced by its transcoder at positions >= 1 (every other
layer exact). For each sampled feature i, remove i (a_i := 0 everywhere) and measure the increase
in KL(M || model) in bits, summed over the evaluation tokens, and per firing of i.
"""
import argparse, json, math, os, time
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--layers", default="12")
ap.add_argument("--rows", type=int, default=16)
ap.add_argument("--all-rows", type=int, default=64, help="row grid of tc_measure (same windows)")
ap.add_argument("--per-stratum", type=int, default=12)
ap.add_argument("--group", type=int, default=6)
ap.add_argument("--device", default="mps")
ap.add_argument("--out", required=True)
args = ap.parse_args()
dev = torch.device(args.device)
LN2 = math.log(2.0)
TC = os.path.expanduser("~/mpd-data/transcoders/qwen3-0.6b-lowl0")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32, attn_implementation="sdpa").to(dev).eval()
core, layers, lm_w = model.model, model.model.layers, model.lm_head.weight
NL, D = len(layers), 1024
win = np.fromfile(os.path.expanduser("~/mpd-data/qwen3_fineweb/heldout/windows_T512.u32"), dtype=np.uint32).reshape(-1, 512)
rows = np.linspace(0, win.shape[0] - 1, args.all_rows).round().astype(np.int64)[: args.rows]
ids = torch.from_numpy(win[rows].astype(np.int64)).to(dev)
B, T = ids.shape


def attn(L, h, pos):
    lay = layers[L]
    return h + lay.self_attn(hidden_states=lay.input_layernorm(h), position_embeddings=pos, attention_mask=None)[0]


def kl_sum(ref_logp, h):
    """Sum over tokens at positions >= 1 of KL(M || variant) in bits, per sequence group in h (k,B,T,D)."""
    k = h.shape[0]
    hn = core.norm(h[:, :, 1:]).reshape(k, -1, D)
    out = torch.zeros(k, device=dev)
    for j in range(k):
        for s in range(0, hn.shape[1], 1024):
            lq = torch.log_softmax(hn[j, s:s + 1024] @ lm_w.T, -1)
            lp = ref_logp[s:s + 1024]
            out[j] += ((lp.exp() * (lp - lq)).sum(-1) / LN2).sum()
    return out.cpu().numpy()


results = {}
with torch.no_grad():
    pos = core.rotary_emb(core.embed_tokens(ids), torch.arange(T, device=dev)[None])
    for L in [int(v) for v in args.layers.split(",")]:
        t0 = time.time()
        with safe_open(f"{TC}/layer_{L}.safetensors", framework="pt", device="cpu") as f:
            We, be, Wd, bd = (f.get_tensor(k).to(dev).float() for k in ("W_enc", "b_enc", "W_dec", "b_dec"))
        h = core.embed_tokens(ids)
        for L2 in range(L):
            hm = attn(L2, h, pos)
            h = hm + layers[L2].mlp(layers[L2].post_attention_layernorm(hm))
        hm = attn(L, h, pos)
        x = layers[L].post_attention_layernorm(hm)
        y = layers[L].mlp(x)
        href = hm + y
        for L2 in range(L + 1, NL):
            href = attn(L2, href, pos)
            href = href + layers[L2].mlp(layers[L2].post_attention_layernorm(href))
        hn = core.norm(href[:, 1:]).reshape(-1, D)
        ref_logp = torch.empty(hn.shape[0], lm_w.shape[0], device=dev)
        for s in range(0, hn.shape[0], 1024):
            ref_logp[s:s + 1024] = torch.log_softmax(hn[s:s + 1024] @ lm_w.T, -1)
        a = (x[:, 1:] @ We.T).add_(be).relu_()  # (B, T-1, F)
        yrep = y.clone()
        yrep[:, 1:] = a @ Wd + bd
        cnt = (a > 0).sum((0, 1)).cpu().numpy()
        n_tok = B * (T - 1)
        freq = cnt / n_tok
        rng = np.random.default_rng(L)
        strata = {"top": np.argsort(-cnt)[: args.per_stratum]}
        for lo, hi in ((1e-2, 1e-1), (1e-3, 1e-2), (2.0 / n_tok, 1e-3)):
            cand = np.where((freq >= lo) & (freq < hi))[0]
            cand = np.setdiff1d(cand, strata["top"])
            if len(cand):
                strata[f"[{lo:.0e},{hi:.0e})"] = rng.choice(cand, min(args.per_stratum, len(cand)), replace=False)
        feats = np.concatenate(list(strata.values()))
        names = sum(([k] * len(v) for k, v in strata.items()), [])
        a_sel = a[:, :, torch.from_numpy(feats).to(dev)].clone()
        del a
        torch.mps.empty_cache()

        def suffix(hv):
            for L2 in range(L + 1, NL):
                hv = torch.stack([attn(L2, hv[j], pos) for j in range(hv.shape[0])])
                hv = hv + layers[L2].mlp(layers[L2].post_attention_layernorm(hv))
            return hv

        base = kl_sum(ref_logp, suffix((hm + yrep)[None]))[0]
        dkl = []
        for g in range(0, len(feats), args.group):
            fs = feats[g:g + args.group]
            hv = []
            for j, i in enumerate(fs):
                yi = yrep.clone()
                yi[:, 1:] -= a_sel[:, :, g + j:g + j + 1] * Wd[i]
                hv.append(hm + yi)
            dkl.extend((kl_sum(ref_logp, suffix(torch.stack(hv))) - base).tolist())
            print(f"L{L} {g + len(fs)}/{len(feats)} {time.time() - t0:.0f}s", flush=True)
        dkl = np.array(dkl)
        per_fire = dkl / np.maximum(cnt[feats], 1)
        results[L] = {"tokens": n_tok, "base_kl_bits_per_token": float(base / n_tok),
                      "features": feats.tolist(), "stratum": names, "count": cnt[feats].tolist(),
                      "dkl_total_bits": dkl.tolist(), "dkl_per_firing_bits": per_fire.tolist()}
        print(f"L{L} base KL {base / n_tok:.4f} bits/token")
        for k in strata:
            m = np.array([n == k for n in names])
            print(f"  {k:>16}: n={m.sum()} median count {np.median(cnt[feats][m]):.0f}  dKL per firing: median {np.median(per_fire[m]):.4f} "
                  f"mean {per_fire[m].mean():.4f} max {per_fire[m].max():.4f} min {per_fire[m].min():.4f}", flush=True)
        del We, be, Wd, bd, a_sel, ref_logp
        torch.mps.empty_cache()
json.dump(results, open(args.out, "w"), indent=1)
