"""Measure the circuit-tracer per-layer transcoders (mwhanna/qwen3-0.6b-transcoders-lowl0)
against Qwen3-0.6B's own MLPs on held-out FineWeb windows.

Position 0 of every window is Qwen3's attention-sink position (MLP outputs of norm ~1e3-1e4 at
layers 2 and 27); the transcoders fire 3e4-1.5e5 features there and miss it badly. Every variant
keeps M's exact MLP output at position 0 and every statistic is over positions >= 1.

Per layer L (transcoder applied to M's own MLP input x_L = post_attention_layernorm(h_mid_L)):
  L0 per token, per-feature firing counts, fraction of variance unexplained (FVU) of the
  MLP output y_L, unit counts above relative contribution thresholds for features and for
  M's 3072 SwiGLU neurons.
End to end (first --kl-rows windows), KL(M || variant) in bits per token for
  (a)  layer L's MLP replaced by its transcoder, every other layer exact;
  (a0) layer L's MLP replaced by the transcoder with every feature removed (output b_dec);
  (b)  all 28 MLPs replaced, no error terms.
"""
import argparse, json, math, os, time
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, default=1 << 16)
ap.add_argument("--kl-rows", type=int, default=32)
ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--chunk", type=int, default=2048)
ap.add_argument("--tc", default=os.path.expanduser("~/mpd-data/transcoders/qwen3-0.6b-lowl0"))
ap.add_argument("--windows", default=os.path.expanduser("~/mpd-data/qwen3_fineweb/heldout/windows_T512.u32"))
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
ap.add_argument("--device", default="mps")
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = torch.device(args.device)
LN2 = math.log(2.0)
TAUS = (1e-3, 1e-2, 1e-1)


def sync():
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa").to(dev).eval()
core = model.model
layers = core.layers
NL = len(layers)
D = model.config.hidden_size
lm_w = model.lm_head.weight  # (V, D), tied to the embedding

win = np.fromfile(args.windows, dtype=np.uint32).reshape(-1, args.seq)
nrows = args.tokens // args.seq
rows = np.linspace(0, win.shape[0] - 1, nrows).round().astype(np.int64)
data = torch.from_numpy(win[rows].astype(np.int64))
T1 = args.seq - 1
N = nrows * T1  # tokens at positions >= 1
NK = min(args.kl_rows, nrows) * T1
print(f"windows {win.shape}, using {nrows} rows; stats over {N} tokens (positions >= 1), KL over {NK}", flush=True)

down_norm = [l.mlp.down_proj.weight.norm(dim=0) for l in layers]  # (3072,) column norms
F = 163840


def load_tc(L):
    with safe_open(f"{args.tc}/layer_{L}.safetensors", framework="pt", device="cpu") as f:
        t = {k: f.get_tensor(k).to(dev).float() for k in ("W_enc", "b_enc", "W_dec", "b_dec")}
    t["dnorm"] = t["W_dec"].norm(dim=1)
    return t


b_dec = {}
for L in range(NL):
    with safe_open(f"{args.tc}/layer_{L}.safetensors", framework="pt", device="cpu") as f:
        b_dec[L] = f.get_tensor("b_dec").to(dev).float()


def encode(t, x):
    return torch.relu(x @ t["W_enc"].T + t["b_enc"])


def decode(t, a):
    """Dense decode (torch MPS nonzero returns out-of-range indices on these sizes)."""
    return a @ t["W_dec"] + t["b_dec"]


def tc_out(t, x, dense=False):
    out = torch.empty_like(x)
    for s in range(0, x.shape[0], args.chunk):
        a = encode(t, x[s:s + args.chunk])
        out[s:s + args.chunk] = (a @ t["W_dec"] + t["b_dec"]) if dense else decode(t, a)
    return out


count = np.zeros((NL, F), np.int64)
sum_act = np.zeros((NL, F), np.float64)
sse = np.zeros(NL); sy = np.zeros((NL, D)); syy = np.zeros(NL)
sse_alt = np.zeros((NL, 2)); sse_first = np.zeros(NL); syy_first = np.zeros(NL); sy_first = np.zeros((NL, D)); n_first = 0
l0 = np.zeros((N, NL), np.int32)
fcnt = np.zeros((N, NL, len(TAUS)), np.int32)
ncnt = np.zeros((N, NL, len(TAUS)), np.int32)
ynorm = np.zeros((N, NL), np.float32)
kl_a = np.zeros((NK, NL), np.float32)
kl_a0 = np.zeros((NK, NL), np.float32)
kl_b = np.zeros(NK, np.float32)
timing = {"stats": 0.0, "b": 0.0, "suffix": 0.0, "load": 0.0, "ref": 0.0}


def stats(L, t, x, y, tok0):
    """Transcoder and neuron statistics on M's own MLP input/output (positions >= 1) at layer L."""
    yhat = torch.empty_like(y)
    cnt = torch.zeros(F, dtype=torch.int64, device=dev)
    sa = torch.zeros(F, dtype=torch.float32, device=dev)
    mlp = layers[L].mlp
    for s in range(0, x.shape[0], args.chunk):
        e = min(s + args.chunk, x.shape[0])
        a = encode(t, x[s:e])
        yhat[s:e] = decode(t, a)
        on = a > 0
        cnt += on.sum(0)
        sa += a.sum(0)
        yn = y[s:e].norm(dim=1)
        contrib = a * t["dnorm"]
        g = mlp.act_fn(mlp.gate_proj(x[s:e])) * mlp.up_proj(x[s:e])
        ncontrib = g.abs() * down_norm[L]
        r = slice(tok0 + s, tok0 + e)
        l0[r, L] = on.sum(1).int().cpu().numpy()
        ynorm[r, L] = yn.cpu().numpy()
        for k, tau in enumerate(TAUS):
            thr = (tau * yn)[:, None]
            fcnt[r, L, k] = (contrib > thr).sum(1).int().cpu().numpy()
            ncnt[r, L, k] = (ncontrib > thr).sum(1).int().cpu().numpy()
        del a, contrib, g, ncontrib, on
    count[L] += cnt.cpu().numpy()
    sum_act[L] += sa.cpu().double().numpy()
    e2 = float(((y - yhat) ** 2).sum(1).cpu().double().sum())
    sse[L] += e2
    sy[L] += y.sum(0).cpu().double().numpy()
    syy[L] += float((y ** 2).sum(1).cpu().double().sum())
    return yhat, e2


def attn(L, h, pos):
    lay = layers[L]
    return h + lay.self_attn(hidden_states=lay.input_layernorm(h), position_embeddings=pos, attention_mask=None)[0]


def layer_full(L, h, pos):
    hm = attn(L, h, pos)
    return hm + layers[L].mlp(layers[L].post_attention_layernorm(hm))


def kl_rows(ref_logp, h):
    """KL(M || variant) per token in bits; ref_logp (n,V), h final residual (n,D)."""
    out = torch.empty(h.shape[0], device=dev)
    hn = core.norm(h)
    c = args.chunk // 2
    for s in range(0, h.shape[0], c):
        lq = torch.log_softmax(hn[s:s + c] @ lm_w.T, -1)
        lp = ref_logp[s:s + c]
        out[s:s + c] = (lp.exp() * (lp - lq)).sum(-1) / LN2
    return out


t0 = time.time()
B = args.batch
for bi in range(0, nrows, B):
    ids = data[bi:bi + B].to(dev)
    nb, T = ids.shape
    tok0 = bi * T1
    first = bi == 0
    do_kl = bi < args.kl_rows
    with torch.no_grad():
        pos_ids = torch.arange(T, device=dev)[None]
        h = core.embed_tokens(ids)
        pos = core.rotary_emb(h, pos_ids)
        hb = h.clone()
        hmid_c, yrep_c, y0_c = [], [], []
        for L in range(NL):
            tl = time.time()
            t = load_tc(L)
            sync(); timing["load"] += time.time() - tl
            lay = layers[L]
            tl = time.time()
            hm = attn(L, h, pos)
            x = lay.post_attention_layernorm(hm)
            y = lay.mlp(x)
            sync(); timing["ref"] += time.time() - tl
            tl = time.time()
            xf, yf = x[:, 1:].reshape(-1, D), y[:, 1:].reshape(-1, D)
            yh, e2 = stats(L, t, xf, yf, tok0)
            if first:
                hmf = hm[:, 1:].reshape(-1, D)
                xpre = hmf * torch.rsqrt(hmf.pow(2).mean(-1, keepdim=True) + lay.post_attention_layernorm.variance_epsilon)
                for k, xa in enumerate((xpre, hmf)):
                    sse_alt[L, k] += float(((yf - tc_out(t, xa, dense=True)) ** 2).sum(1).cpu().double().sum())
                sse_first[L] += e2
                syy_first[L] += float((yf ** 2).sum(1).cpu().double().sum()); sy_first[L] += yf.sum(0).cpu().double().numpy()
            sync(); timing["stats"] += time.time() - tl
            if do_kl:
                hmid_c.append(hm)
                yrep = y.clone(); yrep[:, 1:] = yh.reshape(nb, T1, D)
                yrep_c.append(yrep)
                y0 = y.clone(); y0[:, 1:] = b_dec[L]
                y0_c.append(y0)
                tl = time.time()
                hmb = attn(L, hb, pos)
                xb = lay.post_attention_layernorm(hmb)
                yb = lay.mlp(xb)  # exact at position 0
                yb[:, 1:] = tc_out(t, xb[:, 1:].reshape(-1, D)).reshape(nb, T1, D)
                hb = hmb + yb
                sync(); timing["b"] += time.time() - tl
            h = hm + y
            del t
        if first:
            n_first = nb * T1
            ref_logits = model(ids).logits
            mine = core.norm(h) @ lm_w.T
            print("manual forward vs HF logits max|diff|", float((ref_logits - mine).abs().max()), flush=True)
            del ref_logits, mine
        if do_kl:
            tl = time.time()
            k0 = bi * T1
            hn = core.norm(h).reshape(-1, D)
            ref_logp = torch.empty(nb * T, lm_w.shape[0], device=dev)
            c = args.chunk // 2
            for s in range(0, nb * T, c):
                ref_logp[s:s + c] = torch.log_softmax(hn[s:s + c] @ lm_w.T, -1)
            ref_logp = ref_logp.reshape(nb, T, -1)[:, 1:].reshape(nb * T1, -1)
            kl_b[k0:k0 + nb * T1] = kl_rows(ref_logp, hb[:, 1:].reshape(-1, D)).cpu().numpy()
            for L in range(NL):
                hv = torch.cat([hmid_c[L] + yrep_c[L], hmid_c[L] + y0_c[L]], 0)
                for L2 in range(L + 1, NL):
                    hv = layer_full(L2, hv, pos)
                kl_a[k0:k0 + nb * T1, L] = kl_rows(ref_logp, hv[:nb, 1:].reshape(-1, D)).cpu().numpy()
                kl_a0[k0:k0 + nb * T1, L] = kl_rows(ref_logp, hv[nb:, 1:].reshape(-1, D)).cpu().numpy()
            del ref_logp, hmid_c, yrep_c, y0_c
            sync(); timing["suffix"] += time.time() - tl
    done = min((bi + nb), args.kl_rows) * T1
    msg = f"batch {bi // B + 1}/{math.ceil(nrows / B)} {time.time() - t0:.0f}s " + " ".join(f"{k} {v:.0f}" for k, v in timing.items())
    if done:
        msg += f"  KL_b {kl_b[:done].mean():.3f} KL_a(mean over layers) {kl_a[:done].mean():.4f}"
    print(msg, flush=True)

mean_y = sy / N
fvu = sse / (syy - N * (mean_y ** 2).sum(1))
mf = sy_first / max(n_first, 1)
den_first = syy_first - n_first * (mf ** 2).sum(1)
fvu_alt = np.concatenate([(sse_first / den_first)[:, None], sse_alt / den_first[:, None]], 1)
np.savez_compressed(f"{args.out}/tc_measure.npz", count=count, sum_act=sum_act, l0=l0, fcnt=fcnt, ncnt=ncnt,
                    ynorm=ynorm, kl_a=kl_a, kl_a0=kl_a0, kl_b=kl_b, fvu=fvu, fvu_alt=fvu_alt, taus=np.array(TAUS),
                    rows=rows)
summary = {
    "tokens_stats": N, "tokens_kl": NK, "seq": args.seq, "rows": nrows,
    "fvu": fvu.tolist(),
    "fvu_first_batch_[postnorm_with_weight, postnorm_without_weight, raw_resid]": fvu_alt.tolist(),
    "l0_mean": l0.mean(0).tolist(),
    "alive_frac": (count > 0).mean(1).tolist(),
    "feature_count_rel_tau_mean": fcnt.mean(0).tolist(),
    "neuron_count_rel_tau_mean": ncnt.mean(0).tolist(),
    "kl_a_bits": kl_a.mean(0).tolist(), "kl_a0_bits": kl_a0.mean(0).tolist(), "kl_b_bits": float(kl_b.mean()),
    "seconds": time.time() - t0, "timing": timing,
}
json.dump(summary, open(f"{args.out}/summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}), flush=True)
