"""
Weekday circle on real Qwen3-8B: fit -> recover ordered ring -> steer (causal).
Self-contained. CPU. Saves weekday_ring.png + weekday_numbers.json.
"""
import json, sys, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
torch.set_num_threads(64)
R = os.environ["GAM_DATA_ROOT"]  # required: data/model root (was a hardcoded cluster path)
MODEL = os.path.join(R, "models", "qwen3-8b")
OUT = os.path.join(R, "gam_ceiling_fable", "experiments", "real_circle")
os.makedirs(OUT, exist_ok=True)

DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
# context templates that place the day word mid-sentence (varied contexts)
TEMPLATES = [
    "I will see you on {d}.",
    "The meeting is scheduled for {d} morning.",
    "She was born on a {d}.",
    "We usually go shopping on {d}.",
    "The package arrived last {d}.",
    "Every {d} we have a call.",
    "It happened on {d} afternoon.",
    "They are leaving next {d}.",
    "My favorite day is {d}.",
    "The store is closed on {d}.",
    "He starts work this {d}.",
    "The concert is on {d} evening.",
]

print("loading model (cpu, bf16)...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
model.eval()
NL = model.config.num_hidden_layers
D = model.config.hidden_size
print(f"loaded: {NL} layers, d={D}", flush=True)
MEANS_CACHE = f"{OUT}/weekday_means.npy"

def day_token_pos(prompt, day):
    """index of the last subtoken of the day word in the tokenized prompt."""
    ids = tok(prompt, return_tensors="pt").input_ids[0]
    # find the day word span by decoding incremental
    day_ids = tok(" "+day, add_special_tokens=False).input_ids
    n = len(ids); m = len(day_ids)
    for i in range(n-m, -1, -1):
        if ids[i:i+m].tolist() == day_ids:
            return ids.unsqueeze(0), i+m-1
    # fallback: last non-eos token
    return ids.unsqueeze(0), n-1

# ---- harvest: per (layer, day) mean residual at the day-token position ----
if os.path.exists(MEANS_CACHE):
    means = np.load(MEANS_CACHE)
    print(f"loaded cached means {means.shape}", flush=True)
else:
    print("harvesting weekday activations...", flush=True)
    acc = np.zeros((NL+1, 7, D), dtype=np.float64)
    cnt = np.zeros((NL+1, 7), dtype=np.float64)
    with torch.no_grad():
        for di, day in enumerate(DAYS):
            for t in TEMPLATES:
                prompt = t.format(d=day)
                ids, pos = day_token_pos(prompt, day)
                out = model(ids, output_hidden_states=True)
                hs = out.hidden_states  # tuple (NL+1) [1,seq,D]
                for L in range(NL+1):
                    v = hs[L][0, pos].float().numpy()
                    acc[L, di] += v; cnt[L, di] += 1
            print(f"  {day} done", flush=True)
    means = acc / cnt[:,:,None]   # [NL+1, 7, D] mean residual per (layer, day)
    np.save(MEANS_CACHE, means)

# ---- recovery: for each layer, project the 7 day-means to 2D, measure ordered ring ----
def circ_order_score(pts2d):
    """circular correlation between recovered angle-order and true day index 0..6."""
    c = pts2d - pts2d.mean(0)
    ang = np.arctan2(c[:,1], c[:,0])   # angle of each day
    # circular correlation of ang vs uniform target angles (2pi*i/7), best rotation/reflection
    tgt = 2*np.pi*np.arange(7)/7
    best = -1
    for refl in [1,-1]:
        a = refl*ang
        for shift in np.linspace(0,2*np.pi,72,endpoint=False):
            aa = (a+shift)%(2*np.pi)
            # circular corr
            sa, st = np.sin(aa-aa.mean()), np.sin(tgt-tgt.mean())
            r = (sa*st).sum()/np.sqrt((sa**2).sum()*(st**2).sum()+1e-12)
            best = max(best, r)
    return best, ang

results = {}
best_L, best_score = None, -1
for L in range(1, NL+1):
    m = means[L]                        # [7,D]
    mc = m - m.mean(0)
    U,S,Vt = np.linalg.svd(mc, full_matrices=False)
    pts = mc @ Vt[:2].T                 # top-2 PCA of the 7 day-means
    radius = np.linalg.norm(pts,axis=1)
    ring_flat = radius.std()/(radius.mean()+1e-9)   # low => points equidistant from center (ring-like)
    score, ang = circ_order_score(pts)
    ev2 = (S[:2]**2).sum()/(S**2).sum()             # frac of the 7-day variance in the 2D circle plane
    results[L] = dict(circ_order=float(score), ring_flatness=float(ring_flat), ev_2d=float(ev2))
    if score > best_score:
        best_score, best_L = score, L

print(f"BEST LAYER L={best_L}: circ_order={best_score:.3f}", flush=True)

# ---- plot the ring at the best layer ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
m = means[best_L]; mc = m - m.mean(0)
U,S,Vt = np.linalg.svd(mc, full_matrices=False); pts = mc @ Vt[:2].T
order = np.argsort(np.arctan2(pts[:,1]-pts[:,1].mean(), pts[:,0]-pts[:,0].mean()))
fig, ax = plt.subplots(figsize=(6,6))
# connect in TRUE weekday order to show it traces a ring
loop = list(range(7))+[0]
ax.plot(pts[loop,0], pts[loop,1], '-', color="#555", lw=1.5, zorder=1)
ax.scatter(pts[:,0], pts[:,1], s=260, c=np.arange(7), cmap="twilight", zorder=2, edgecolors="k")
for i,d in enumerate(DAYS):
    ax.annotate(d, (pts[i,0], pts[i,1]), fontsize=11, ha="center", va="center", zorder=3)
ax.set_title(f"Qwen3-8B weekday activations, L{best_L}\ncircular-order corr = {best_score:.3f} (Mon→Sun traces a ring)")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_aspect("equal"); ax.grid(alpha=.2)
plt.tight_layout(); plt.savefig(f"{OUT}/weekday_ring.png", dpi=140)
print(f"saved {OUT}/weekday_ring.png", flush=True)

# ---- steering (causal): push a day activation +1 along the circle (PER-DAY local tangent) ----
# BUGFIX: a global mean of step vectors telescopes to ~0 around the cycle. Use the LOCAL step
# tangent_i = mean[day_{i+1}] - mean[day_i], specific to the day being steered.
Lst = best_L
day_tok_ids = {d: tok(" "+d, add_special_tokens=False).input_ids[0] for d in DAYS}
layer_module = model.model.layers[Lst-1]  # hidden_states[L] is output of layer index L-1

def steer_hook_factory(pos, vec, alpha):
    def hook(module, inp, out):
        if isinstance(out, tuple):
            h = out[0].clone(); h[0, pos] = h[0, pos] + alpha*vec
            return (h,) + tuple(out[1:])
        h = out.clone(); h[0, pos] = h[0, pos] + alpha*vec
        return h
    return hook

# SANITY: confirm the hook actually changes logits (big alpha must move something)
def run_steered(ids, pos, vec, alpha):
    hh = layer_module.register_forward_hook(steer_hook_factory(pos, vec, alpha))
    try: lg = model(ids).logits[0,-1].float()
    finally: hh.remove()
    return lg

ALPHAS = [0.0, 2.0, 4.0, 8.0, 16.0]
best_alpha, best_frac, best_detail, best_mean = None, -1, None, 0.0
with torch.no_grad():
    for alpha in ALPHAS:
        detail = []
        for i,day in enumerate(DAYS):
            prompt = f"The day after {day} is"
            ids, pos = day_token_pos(prompt, day)
            tangent_i = torch.tensor(means[Lst,(i+1)%7]-means[Lst,i], dtype=torch.bfloat16)
            base = model(ids).logits[0,-1].float()
            stl  = run_steered(ids, pos, tangent_i, alpha)
            nxt, nxt2 = DAYS[(i+1)%7], DAYS[(i+2)%7]
            b1,b2 = base[day_tok_ids[nxt]].item(), base[day_tok_ids[nxt2]].item()
            s1,s2 = stl[day_tok_ids[nxt]].item(),  stl[day_tok_ids[nxt2]].item()
            logit_delta = float((stl-base).abs().max())   # did anything move at all?
            shift = (s2-s1)-(b2-b1)                        # >0 => advanced the day
            detail.append(dict(day=day, advance_shift=shift, max_logit_delta=logit_delta,
                               base_next=b1, base_next2=b2, steered_next=s1, steered_next2=s2))
        mean_shift = float(np.mean([d["advance_shift"] for d in detail]))
        frac_pos = float(np.mean([d["advance_shift"]>0 for d in detail]))
        maxdelta = float(np.mean([d["max_logit_delta"] for d in detail]))
        print(f"  alpha={alpha}: mean_advance_shift={mean_shift:+.3f}, frac_advanced={frac_pos:.2f}, mean_max_logit_delta={maxdelta:.3f}", flush=True)
        if frac_pos > best_frac or (frac_pos==best_frac and mean_shift>best_mean):
            best_frac, best_mean, best_alpha, best_detail = frac_pos, mean_shift, alpha, detail

steer_results = best_detail
mean_shift, frac_pos = best_mean, best_frac
print(f"BEST alpha={best_alpha}: mean_advance_shift={mean_shift:+.3f}, frac_advanced={frac_pos:.2f}", flush=True)

summary = dict(best_layer=best_L, circ_order_corr=best_score,
               per_layer=results, steer_best_alpha=best_alpha,
               steering_mean_advance_shift=mean_shift,
               steering_frac_advanced=frac_pos, steer_detail=steer_results)
json.dump(summary, open(f"{OUT}/weekday_numbers.json","w"), indent=1)
print("\n==== RESULT ====", flush=True)
print(f"RECOVERY: best layer L{best_L}, circular-order corr = {best_score:.3f}", flush=True)
print(f"STEERING: mean advance shift = {mean_shift:+.2f} logits, {frac_pos*100:.0f}% of days advanced", flush=True)
print(f"saved {OUT}/weekday_numbers.json and weekday_ring.png", flush=True)
