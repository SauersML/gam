"""Pruned-native baselines on the 4L target: keep a subset of MLP neurons (inputs of down_proj)
and attention heads (input slices of o_proj), mean-ablate the rest, and report KL to the target.

  static  one fixed neuron subset per MLP (the k with the largest contribution variance
          E[(a_j - mean a_j)^2] * |W_down[:, j]|^2 on SCORE rows) and one fixed head subset per
          layer (largest variance of the head's o_proj contribution)
  dynamic per token, the k neurons with the largest |a_j - mean a_j| * |W_down[:, j]| (all heads)
Scores and means come from SCORE_ROWS rows at offset 2048; KL is measured on EVAL_ROWS rows at
offset 0 (disjoint).

usage: vpd_prune.py EVAL_ROWS OUT.json
"""

import json
import sys
import time

import torch

from vpd_eval import kl_per_pos
from vpd_model import load_target, val_tokens

EVAL_ROWS, out_path = int(sys.argv[1]), sys.argv[2]
SCORE_ROWS, MB = 64, 4
t0 = time.time()
target = load_target("mps")
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
L, H, HD = 4, 6, 128
down = [target.site(f"h.{i}.mlp.down_proj") for i in range(L)]
oproj = [target.site(f"h.{i}.attn.o_proj") for i in range(L)]

# --- statistics on the score rows
s_ids = val_tokens(SCORE_ROWS, offset=2048)
sum_a = [torch.zeros(3072, device="mps") for _ in range(L)]
sum_a2 = [torch.zeros(3072, device="mps") for _ in range(L)]
sum_y = [torch.zeros(768, device="mps") for _ in range(L)]
head_c2 = [torch.zeros(H, device="mps") for _ in range(L)]
head_c = [torch.zeros(H, 768, device="mps") for _ in range(L)]
n_tok = 0
with torch.no_grad():
    for i in range(0, SCORE_ROWS, MB):
        for s in down + oproj:
            s.cache_input = True
        target(s_ids[i:i + MB].to("mps"))
        for l in range(L):
            a = down[l].last_input.flatten(0, 1)
            sum_a[l] += a.sum(0)
            sum_a2[l] += (a * a).sum(0)
            y = oproj[l].last_input.flatten(0, 1).view(-1, H, HD)
            Wo = oproj[l].W.view(768, H, HD)
            c = torch.einsum("nhd,ohd->nho", y, Wo)  # per-head contribution to the residual
            head_c[l] += c.sum(0)
            head_c2[l] += (c * c).sum((0, 2))
            sum_y[l] += y.flatten(1).sum(0)
        n_tok += s_ids[i:i + MB].numel()
        for s in down + oproj:
            s.cache_input = False
            s.last_input = None
mean_a = [(x / n_tok).float() for x in sum_a]
mean_y = [(x / n_tok).float() for x in sum_y]
wnorm = [down[l].W.norm(dim=0) for l in range(L)]
neuron_score = [((sum_a2[l] / n_tok - (sum_a[l] / n_tok) ** 2).float() * wnorm[l] ** 2) for l in range(L)]
head_score = [(head_c2[l] / n_tok - ((head_c[l] / n_tok) ** 2).sum(1)).float() for l in range(L)]
neuron_rank = [s.argsort(descending=True) for s in neuron_score]
head_rank = [s.argsort(descending=True) for s in head_score]
log(f"stats on {n_tok} tokens; head variance ranks {[r.tolist() for r in head_rank]}")

e_ids = val_tokens(EVAL_ROWS, offset=0)


@torch.no_grad()
def kl_now() -> float:
    """KL(target || pruned), the target logits recomputed per microbatch (no logit cache)."""
    tot, n = 0.0, 0
    for i in range(0, EVAL_ROWS, MB):
        b = e_ids[i:i + MB].to("mps")
        lg = target(b)
        fns = [s.in_fn for s in down + oproj]
        for s in down + oproj:
            s.in_fn = None
        ref = target(b)
        for s, f in zip(down + oproj, fns):
            s.in_fn = f
        tot += kl_per_pos(lg, ref).mean().item()
        n += 1
        del lg, ref
    return tot / n


def clear():
    for s in down + oproj:
        s.in_fn = None


def static_neurons(k: int):
    for l in range(L):
        keep = torch.zeros(3072, dtype=torch.bool, device="mps")
        keep[neuron_rank[l][:k]] = True
        down[l].in_fn = lambda a, keep=keep, m=mean_a[l]: torch.where(keep, a, m)


def static_heads(h: int):
    for l in range(L):
        keep = torch.zeros(H, dtype=torch.bool, device="mps")
        keep[head_rank[l][:h]] = True
        keep = keep.repeat_interleave(HD)
        oproj[l].in_fn = lambda y, keep=keep, m=mean_y[l]: torch.where(keep, y, m)


def dynamic_neurons(k: int):
    for l in range(L):
        def f(a, l=l):
            dev = (a - mean_a[l]).abs() * wnorm[l]
            idx = dev.topk(k, dim=-1).indices
            keep = torch.zeros_like(a, dtype=torch.bool).scatter_(-1, idx, True)
            return torch.where(keep, a, mean_a[l])
        down[l].in_fn = f


results = {"eval_rows": EVAL_ROWS, "score_rows": SCORE_ROWS, "static_neurons": {}, "static_heads": {},
           "dynamic_neurons": {}, "static_joint": {}}
results["mean_ablate_all_mlp_neurons"] = (static_neurons(0), kl_now(), clear())[1]
for k in (8, 32, 128, 512, 1024, 2048, 2560, 3072):
    static_neurons(k)
    results["static_neurons"][k] = kl_now()
    clear()
    log(f"static neurons k={k}/3072 per MLP: KL {results['static_neurons'][k]:.4f}")
for h in range(0, H + 1):
    static_heads(h)
    results["static_heads"][h] = kl_now()
    clear()
    log(f"static heads {h}/6 per layer: KL {results['static_heads'][h]:.4f}")
for k in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
    dynamic_neurons(k)
    results["dynamic_neurons"][k] = kl_now()
    clear()
    log(f"dynamic per-token top-{k} neurons per MLP: KL {results['dynamic_neurons'][k]:.4f}")
for k, h in ((1024, 5), (2048, 5), (2048, 6), (2560, 6), (8, 6)):
    static_neurons(k)
    static_heads(h)
    results["static_joint"][f"{k}n_{h}h"] = kl_now()
    clear()
    log(f"static joint {k} neurons, {h} heads: KL {results['static_joint'][f'{k}n_{h}h']:.4f}")
# one MLP at a time (the other three intact): is "8 of 3072 neurons per MLP" a per-layer claim?
results["single_layer"] = {}
for l in range(L):
    for kind in ("static", "dynamic"):
        for k in (8, 64, 512):
            if kind == "static":
                static_neurons(k)
            else:
                dynamic_neurons(k)
            for j in range(L):
                if j != l:
                    down[j].in_fn = None
            results["single_layer"][f"L{l}_{kind}_{k}"] = kl_now()
            clear()
            log(f"only MLP {l}, {kind} top-{k}: KL {results['single_layer'][f'L{l}_{kind}_{k}']:.4f}")
    down[l].in_fn = lambda a, m=mean_a[l]: m.expand_as(a)
    results["single_layer"][f"L{l}_mean_ablate_all"] = kl_now()
    clear()
    log(f"only MLP {l}, all neurons mean-ablated: KL {results['single_layer'][f'L{l}_mean_ablate_all']:.4f}")
json.dump(results, open(out_path, "w"), indent=1)
log("done")
