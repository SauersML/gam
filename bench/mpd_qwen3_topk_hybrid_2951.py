"""All 28 MLPs replaced at once; KL(M || model) in bits per token on positions >= 1 (position 0 exact).

Streams (each its own forward through all 28 layers):
  m k : M's neurons, top-k by deviation write |g_j - gbar_j| * ||w_j||; dropped neurons set to their
        mean activation gbar_j (measured on held-out rows 32..47 of the same window grid)
  h k : transcoder (all active features) + k of M's neurons with their own activations, chosen per
        token by matching pursuit on the transcoder error e = y_M - y_tc (add the neuron whose write
        g_j w_j most reduces ||e - sum||; stop early when none reduces it)
  p k : transcoder + k of M's down-projection directions w_j with free per-token coefficients
        (matching pursuit on e over unit directions w_j/||w_j||, each direction used once)
"""
import argparse, json, math, os, time
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=2)
ap.add_argument("--row-offset", type=int, default=0)
ap.add_argument("--all-rows", type=int, default=64)
ap.add_argument("--km", default="64,128,256,512,1024")
ap.add_argument("--kh", default="0,1,4,16,64")
ap.add_argument("--kp", default="1,4,16,64")
ap.add_argument("--means", default=os.path.expanduser("~/mpd-data/scratch/sparsebase/neuron_means.pt"))
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
grid = np.linspace(0, win.shape[0] - 1, args.all_rows).round().astype(np.int64)
CH = 1024


def attn(L, h, pos):
    lay = layers[L]
    return h + lay.self_attn(hidden_states=lay.input_layernorm(h), position_embeddings=pos, attention_mask=None)[0]


def acts(L, x):
    mlp = layers[L].mlp
    return mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)


with torch.no_grad():
    if not os.path.exists(args.means):
        ids = torch.from_numpy(win[grid[32:48]].astype(np.int64)).to(dev)
        pos = core.rotary_emb(core.embed_tokens(ids), torch.arange(ids.shape[1], device=dev)[None])
        h = core.embed_tokens(ids)
        means = []
        for L in range(NL):
            hm = attn(L, h, pos)
            x = layers[L].post_attention_layernorm(hm)
            means.append(acts(L, x[:, 1:].reshape(-1, D)).mean(0))
            h = hm + layers[L].mlp(x)
        torch.save(torch.stack(means).cpu(), args.means)
        print("saved neuron means", flush=True)
    gbar = torch.load(args.means).to(dev)

W = [l.mlp.down_proj.weight for l in layers]  # (D, 3072)
wn = [w.norm(dim=0) for w in W]
KM = [int(v) for v in args.km.split(",")]
KH = [int(v) for v in args.kh.split(",")]
KP = [int(v) for v in args.kp.split(",")]
names = [("m", k) for k in KM] + [("h", k) for k in KH] + [("p", k) for k in KP]


def load_tc(L):
    with safe_open(f"{TC}/layer_{L}.safetensors", framework="pt", device="cpu") as f:
        return {k: f.get_tensor(k).to(dev).float() for k in ("W_enc", "b_enc", "W_dec", "b_dec")}


def tc_out(t, x):
    out = torch.empty_like(x)
    for s in range(0, x.shape[0], CH):
        out[s:s + CH] = torch.relu(x[s:s + CH] @ t["W_enc"].T + t["b_enc"]) @ t["W_dec"] + t["b_dec"]
    return out


def mean_trunc(L, g, k):
    dev_ = g - gbar[L]
    idx = torch.topk(dev_.abs() * wn[L], k, 1).indices
    keep = torch.zeros_like(g).scatter_(1, idx, dev_.gather(1, idx))
    return (gbar[L] + keep) @ W[L].T


def pursuit_neurons(L, g, e, k):
    """Add up to k neurons (own activations) greedily reducing ||e - sum g_j w_j||."""
    n = g.shape[0]
    ar = torch.arange(n, device=dev)
    r = e.clone()
    c2 = (g * wn[L]) ** 2
    used = torch.zeros_like(g, dtype=torch.bool)
    add = torch.zeros_like(e)
    for _ in range(k):
        score = 2 * g * (r @ W[L]) - c2
        score[used] = -float("inf")
        j = score.argmax(1)
        ok = (score[ar, j] > 0).float()[:, None]
        used[ar, j] = True
        c = g[ar, j][:, None] * W[L][:, j].T * ok
        r = r - c
        add = add + c
    return add


def pursuit_dirs(L, e, k):
    """Matching pursuit on e over unit directions w_j/||w_j||, each used once, free coefficients."""
    n = e.shape[0]
    ar = torch.arange(n, device=dev)
    U = W[L] / wn[L]
    r = e.clone()
    used = torch.zeros(n, U.shape[1], dtype=torch.bool, device=dev)
    for _ in range(k):
        proj = r @ U
        sc = proj.abs()
        sc[used] = -1.0
        j = sc.argmax(1)
        used[ar, j] = True
        r = r - proj[ar, j][:, None] * U[:, j].T
    return e - r


def kl_tok(ref_logp, h):
    hn = core.norm(h[:, 1:]).reshape(-1, D)
    out = torch.empty(hn.shape[0], device=dev)
    for s in range(0, hn.shape[0], CH):
        lq = torch.log_softmax(hn[s:s + CH] @ lm_w.T, -1)
        lp = ref_logp[s:s + CH]
        out[s:s + CH] = (lp.exp() * (lp - lq)).sum(-1) / LN2
    return out.cpu().numpy()


rows = grid[args.row_offset: args.row_offset + args.rows]
ids = torch.from_numpy(win[rows].astype(np.int64)).to(dev)
nb, T = ids.shape
t0 = time.time()
res = {}
with torch.no_grad():
    pos = core.rotary_emb(core.embed_tokens(ids), torch.arange(T, device=dev)[None])
    h = core.embed_tokens(ids)
    streams = {nm: h.clone() for nm in names}
    for L in range(NL):
        t = load_tc(L)
        lay = layers[L]
        hm = attn(L, h, pos)
        h = hm + lay.mlp(lay.post_attention_layernorm(hm))
        hs = torch.cat([streams[nm] for nm in names], 0)
        hms = attn(L, hs, pos)
        xs = lay.post_attention_layernorm(hms)
        ys = lay.mlp(xs)  # exact, used at position 0 and as y_M
        for j, nm in enumerate(names):
            sl = slice(j * nb, (j + 1) * nb)
            x = xs[sl, 1:].reshape(-1, D)
            yM = ys[sl, 1:].reshape(-1, D)
            kind, k = nm
            if kind == "m":
                v = mean_trunc(L, acts(L, x), k)
            else:
                yt = tc_out(t, x)
                if k == 0:
                    v = yt
                elif kind == "h":
                    v = yt + pursuit_neurons(L, acts(L, x), yM - yt, k)
                else:
                    v = yt + pursuit_dirs(L, yM - yt, k)
            y = ys[sl].clone()
            y[:, 1:] = v.reshape(nb, T - 1, D)
            streams[nm] = hms[sl] + y
        del t, hs, hms, xs, ys
        if dev.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()
        print(f"layer {L} {time.time() - t0:.0f}s", flush=True)
    hn = core.norm(h[:, 1:]).reshape(-1, D)
    ref_logp = torch.empty(hn.shape[0], lm_w.shape[0], device=dev)
    for s in range(0, hn.shape[0], CH):
        ref_logp[s:s + CH] = torch.log_softmax(hn[s:s + CH] @ lm_w.T, -1)
    for nm in names:
        res[f"{nm[0]}{nm[1]}"] = float(kl_tok(ref_logp, streams[nm]).mean())
res = {"tokens": nb * (T - 1), "rows": rows.tolist(), "kl": res, "seconds": time.time() - t0}
json.dump(res, open(args.out, "w"), indent=1)
print(json.dumps(res["kl"]), flush=True)
