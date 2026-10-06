"""Units kept per token against faithfulness, for M's MLP neurons and for transcoder features.

For each MLP and each token (positions >= 1), keep the k units with the largest write
|act_j| * ||d_j|| on that token and zero the rest:
  neurons:  y_k = W_down (g masked to its top-k),  g = silu(gate x) * up x   (k = 3072 is exact)
  features: y_k = (a masked to its top-k) W_dec + b_dec,  a = relu(W_enc x + b_enc)
Position 0 (attention sink) keeps M's exact MLP output.
Measured: KL(M || model) in bits per token with (i) one layer truncated, others exact, and
(ii) all 28 layers truncated at the same k.  mpd_qwen3_topk_greedy_2951.py compares the |write|
ranking with a greedy choice on the layer's output error.  The Apple GPU footprint reaches ~34 GiB
with 2 rows per process; run one process per --row-offset.
"""
import argparse, json, math, os, time
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=8)
ap.add_argument("--row-offset", type=int, default=0)
ap.add_argument("--all-rows", type=int, default=64)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--single", default="2,6,14,22,27")
ap.add_argument("--kf", default="1,2,4,8,16,32,64,128,256")
ap.add_argument("--kn", default="1,2,4,8,16,32,64,128,256,512,1024")
ap.add_argument("--device", default="mps")
ap.add_argument("--out", required=True)
args = ap.parse_args()
dev = torch.device(args.device)
LN2 = math.log(2.0)
TC = os.path.expanduser("~/mpd-data/transcoders/qwen3-0.6b-lowl0")
KF = [int(v) for v in args.kf.split(",")]
KN = [int(v) for v in args.kn.split(",")]
SINGLE = [int(v) for v in args.single.split(",")] if args.single else []
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32, attn_implementation="sdpa").to(dev).eval()
core, layers, lm_w = model.model, model.model.layers, model.lm_head.weight
NL, D = len(layers), 1024
down_norm = [l.mlp.down_proj.weight.norm(dim=0) for l in layers]
win = np.fromfile(os.path.expanduser("~/mpd-data/qwen3_fineweb/heldout/windows_T512.u32"), dtype=np.uint32).reshape(-1, 512)
rows = np.linspace(0, win.shape[0] - 1, args.all_rows).round().astype(np.int64)[args.row_offset: args.row_offset + args.rows]
data = torch.from_numpy(win[rows].astype(np.int64))
CH = 1024


def sync():
    if dev.type == "mps":
        torch.mps.synchronize()


def load_tc(L):
    with safe_open(f"{TC}/layer_{L}.safetensors", framework="pt", device="cpu") as f:
        t = {k: f.get_tensor(k).to(dev).float() for k in ("W_enc", "b_enc", "W_dec", "b_dec")}
    t["dnorm"] = t["W_dec"].norm(dim=1)
    return t


def attn(L, h, pos):
    lay = layers[L]
    return h + lay.self_attn(hidden_states=lay.input_layernorm(h), position_embeddings=pos, attention_mask=None)[0]


def neuron_out(L, x, ks):
    """x (n, D) -> list of truncated MLP outputs, one per k (top-k neurons by write)."""
    mlp = layers[L].mlp
    g = mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)
    w = g.abs() * down_norm[L]
    order = torch.topk(w, max(ks), dim=1).indices
    outs = []
    for k in ks:
        idx = order[:, :k]
        gm = torch.zeros_like(g).scatter_(1, idx, g.gather(1, idx))
        outs.append(mlp.down_proj(gm))
    return outs


def feature_out(t, x, ks):
    """x (n, D) -> list of truncated transcoder outputs, one per k (top-k features by write)."""
    outs = [torch.empty_like(x) for _ in ks]
    for s in range(0, x.shape[0], CH):
        a = torch.relu(x[s:s + CH] @ t["W_enc"].T + t["b_enc"])
        order = torch.topk(a * t["dnorm"], max(ks), dim=1).indices
        for j, k in enumerate(ks):
            idx = order[:, :k]
            am = torch.zeros_like(a).scatter_(1, idx, a.gather(1, idx))
            outs[j][s:s + CH] = am @ t["W_dec"] + t["b_dec"]
        del a, order
    return outs


def place(y_full, y_rest, nb, T):
    """Position 0 exact (from y_full), positions >= 1 from y_rest (nb*(T-1), D)."""
    out = y_full.clone()
    out[:, 1:] = y_rest.reshape(nb, T - 1, D)
    return out


def kl_tok(ref_logp, h):
    hn = core.norm(h[:, 1:]).reshape(-1, D)
    out = torch.empty(hn.shape[0], device=dev)
    for s in range(0, hn.shape[0], CH):
        lq = torch.log_softmax(hn[s:s + CH] @ lm_w.T, -1)
        lp = ref_logp[s:s + CH]
        out[s:s + CH] = (lp.exp() * (lp - lq)).sum(-1) / LN2
    return out.cpu().numpy()


names = [("n", k) for k in KN] + [("f", k) for k in KF]
kl_all = {nm: [] for nm in names}
kl_one = {L: {nm: [] for nm in names} for L in SINGLE}
t0 = time.time()
with torch.no_grad():
    for bi in range(0, args.rows, args.batch):
        ids = data[bi:bi + args.batch].to(dev)
        nb, T = ids.shape
        pos = core.rotary_emb(core.embed_tokens(ids), torch.arange(T, device=dev)[None])
        h = core.embed_tokens(ids)
        streams = {nm: h.clone() for nm in names}
        single_cache = {}
        for L in range(NL):
            t = load_tc(L)
            lay = layers[L]
            hm = attn(L, h, pos)
            x = lay.post_attention_layernorm(hm)
            y = lay.mlp(x)
            if L in SINGLE:
                xr = x[:, 1:].reshape(-1, D)
                yn = neuron_out(L, xr, KN)
                yf = feature_out(t, xr, KF)
                single_cache[L] = (hm, [place(y, v, nb, T) for v in yn + yf])
            # all-layers streams: batch the attention over every stream
            hs = torch.cat([streams[nm] for nm in names], 0)
            hms = attn(L, hs, pos)
            xs = lay.post_attention_layernorm(hms)
            ys_exact = lay.mlp(xs)
            for j, nm in enumerate(names):
                sl = slice(j * nb, (j + 1) * nb)
                xr = xs[sl, 1:].reshape(-1, D)
                v = neuron_out(L, xr, [nm[1]])[0] if nm[0] == "n" else feature_out(t, xr, [nm[1]])[0]
                streams[nm] = hms[sl] + place(ys_exact[sl], v, nb, T)
            h = hm + y
            del t, hs, hms, xs, ys_exact
            sync()
            if dev.type == "mps":
                torch.mps.empty_cache()
            print(f"batch {bi // args.batch} layer {L} {time.time() - t0:.0f}s", flush=True)
        hn = core.norm(h[:, 1:]).reshape(-1, D)
        ref_logp = torch.empty(hn.shape[0], lm_w.shape[0], device=dev)
        for s in range(0, hn.shape[0], CH):
            ref_logp[s:s + CH] = torch.log_softmax(hn[s:s + CH] @ lm_w.T, -1)
        for nm in names:
            kl_all[nm].append(kl_tok(ref_logp, streams[nm]))
        for L, (hm, yv) in single_cache.items():
            for g0 in range(0, len(names), 6):
                grp = names[g0:g0 + 6]
                hv = torch.cat([hm + yv[g0 + j] for j in range(len(grp))], 0)
                for L2 in range(L + 1, NL):
                    hv = attn(L2, hv, pos)
                    hv = hv + layers[L2].mlp(layers[L2].post_attention_layernorm(hv))
                for j, nm in enumerate(grp):
                    kl_one[L][nm].append(kl_tok(ref_logp, hv[j * nb:(j + 1) * nb]))
        del ref_logp, single_cache, streams
        sync()
        msg = " ".join(f"{a}{k}:{np.concatenate(kl_all[(a, k)]).mean():.3f}" for a, k in names)
        print(f"batch {bi // args.batch} done {time.time() - t0:.0f}s ALL {msg}", flush=True)

res = {"tokens": int(sum(len(v) for v in kl_all[names[0]])), "kn": KN, "kf": KF,
       "all": {f"{a}{k}": float(np.concatenate(v).mean()) for (a, k), v in kl_all.items()},
       "all_sem": {f"{a}{k}": float(np.concatenate(v).std() / math.sqrt(len(np.concatenate(v)))) for (a, k), v in kl_all.items()},
       "single": {str(L): {f"{a}{k}": float(np.concatenate(v).mean()) for (a, k), v in d.items()} for L, d in kl_one.items()},
       "seconds": time.time() - t0}
json.dump(res, open(args.out, "w"), indent=1)
print(json.dumps(res["all"]))
for L in SINGLE:
    print(L, json.dumps(res["single"][str(L)]))
