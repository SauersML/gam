"""Single-layer truncation with two choices of the kept set, measured by KL(M || model) in bits/token:
  write : the k units with the largest |write| on the token
  greedy: matching pursuit on the layer's output: per token, repeatedly add the unit whose (fixed)
          write vector most reduces ||y - y_S||; stop early if no unit reduces it.
Neurons (M's 3072) and transcoder features; position 0 exact; other layers exact.
"""
import argparse, json, math, os, time
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=4)
ap.add_argument("--all-rows", type=int, default=64)
ap.add_argument("--layers", default="6,22")
ap.add_argument("--kf", default="1,2,4,8,16,32,64")
ap.add_argument("--kn", default="1,2,4,8,16,32,64,128,256")
ap.add_argument("--device", default="mps")
ap.add_argument("--out", required=True)
args = ap.parse_args()
dev = torch.device(args.device)
LN2 = math.log(2.0)
TC = os.path.expanduser("~/mpd-data/transcoders/qwen3-0.6b-lowl0")
KF = [int(v) for v in args.kf.split(",")]
KN = [int(v) for v in args.kn.split(",")]
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32, attn_implementation="sdpa").to(dev).eval()
core, layers, lm_w = model.model, model.model.layers, model.lm_head.weight
NL, D = len(layers), 1024
win = np.fromfile(os.path.expanduser("~/mpd-data/qwen3_fineweb/heldout/windows_T512.u32"), dtype=np.uint32).reshape(-1, 512)
rows = np.linspace(0, win.shape[0] - 1, args.all_rows).round().astype(np.int64)[: args.rows]
ids = torch.from_numpy(win[rows].astype(np.int64)).to(dev)
B, T = ids.shape
CH = 512


def attn(L, h, pos):
    lay = layers[L]
    return h + lay.self_attn(hidden_states=lay.input_layernorm(h), position_embeddings=pos, attention_mask=None)[0]


def pursuit(C, target, base, ks):
    """C (n, U, D) candidate write vectors, target (n, D). Returns {k: base + sum of chosen writes}."""
    n, U, _ = C.shape
    ar = torch.arange(n, device=dev)
    r = target - base
    part = base.clone()
    used = torch.zeros(n, U, dtype=torch.bool, device=dev)
    c2 = (C * C).sum(-1)
    out = {}
    for s in range(1, max(ks) + 1):
        score = 2 * torch.einsum("nud,nd->nu", C, r) - c2
        score[used] = -float("inf")
        j = score.argmax(1)
        ok = (score[ar, j] > 0).float()[:, None]
        used[ar, j] = True
        c = C[ar, j] * ok
        r = r - c
        part = part + c
        if s in ks:
            out[s] = part.clone()
    return out


res = {}
with torch.no_grad():
    pos = core.rotary_emb(core.embed_tokens(ids), torch.arange(T, device=dev)[None])
    h = core.embed_tokens(ids)
    cache = {}
    for L in range(NL):
        hm = attn(L, h, pos)
        x = layers[L].post_attention_layernorm(hm)
        y = layers[L].mlp(x)
        if L in [int(v) for v in args.layers.split(",")]:
            cache[L] = (hm, x, y)
        h = hm + y
    hn = core.norm(h[:, 1:]).reshape(-1, D)
    ref_logp = torch.empty(hn.shape[0], lm_w.shape[0], device=dev)
    for s in range(0, hn.shape[0], 1024):
        ref_logp[s:s + 1024] = torch.log_softmax(hn[s:s + 1024] @ lm_w.T, -1)

    def kl(hv):
        hn = core.norm(hv[:, 1:]).reshape(-1, D)
        tot = 0.0
        for s in range(0, hn.shape[0], 1024):
            lq = torch.log_softmax(hn[s:s + 1024] @ lm_w.T, -1)
            lp = ref_logp[s:s + 1024]
            tot += float(((lp.exp() * (lp - lq)).sum(-1) / LN2).sum())
        return tot / hn.shape[0]

    for L, (hm, x, y) in cache.items():
        t0 = time.time()
        with safe_open(f"{TC}/layer_{L}.safetensors", framework="pt", device="cpu") as f:
            We, be, Wd, bd = (f.get_tensor(k).to(dev).float() for k in ("W_enc", "b_enc", "W_dec", "b_dec"))
        dnorm = Wd.norm(dim=1)
        mlp = layers[L].mlp
        Wdown = mlp.down_proj.weight  # (D, 3072)
        xr, yr = x[:, 1:].reshape(-1, D), y[:, 1:].reshape(-1, D)
        n = xr.shape[0]
        variants = {}
        for s in range(0, n, CH):
            xs, ys = xr[s:s + CH], yr[s:s + CH]
            # neurons
            g = mlp.act_fn(mlp.gate_proj(xs)) * mlp.up_proj(xs)
            w = g.abs() * Wdown.norm(dim=0)
            order = torch.topk(w, max(KN), 1).indices
            gv = g.gather(1, order)  # (c, K)
            C = gv[:, :, None] * Wdown.T[order]  # (c, K, D) candidates = top-max(KN) writes
            zero = torch.zeros_like(ys)
            for k in KN:
                variants.setdefault(("n", "write", k), []).append(C[:, :k].sum(1))
            for k, v in pursuit(C, ys, zero, KN).items():
                variants.setdefault(("n", "greedy", k), []).append(v)
            del C
            # features
            a = torch.relu(xs @ We.T + be)
            top = torch.topk(a * dnorm, max(KF), 1)
            av = a.gather(1, top.indices)
            Cf = av[:, :, None] * Wd[top.indices]
            base = bd.expand_as(ys)
            for k in KF:
                variants.setdefault(("f", "write", k), []).append(base + Cf[:, :k].sum(1))
            for k, v in pursuit(Cf, ys, base, KF).items():
                variants.setdefault(("f", "greedy", k), []).append(v)
            del a, Cf
        out = {}
        keys = list(variants)
        for g0 in range(0, len(keys), 6):
            grp = keys[g0:g0 + 6]
            hv = []
            for key in grp:
                yv = y.clone()
                yv[:, 1:] = torch.cat(variants[key]).reshape(B, T - 1, D)
                hv.append(hm + yv)
            hv = torch.cat(hv, 0)
            for L2 in range(L + 1, NL):
                hv = attn(L2, hv, pos)
                hv = hv + layers[L2].mlp(layers[L2].post_attention_layernorm(hv))
            for j, key in enumerate(grp):
                out["%s_%s_%d" % key] = kl(hv[j * B:(j + 1) * B])
        res[L] = out
        print(L, f"{time.time() - t0:.0f}s", json.dumps(out), flush=True)
json.dump(res, open(args.out, "w"), indent=1)
