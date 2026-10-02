"""Per-step KL(real model || VPD masked model) along engine-eval's 8 model-sampled rollouts
(16-token prefix + 32 sampled), VPD rounded and CI masks, delta off. Gates two ways: from the
whole 48-token rollout at once, and causally (the gate at step t sees only tokens <= t)."""
import json
import os
import sys

import torch
import torch.nn.functional as F

os.chdir(os.path.expanduser("~/mpd-data/vpd"))
sys.path.insert(0, ".")
from vpd_model import load_target, load_vpd  # noqa: E402

DEV = "mps"
R = json.load(open(os.path.expanduser("~/mpd-data/engine/vpd4l_1x16/rollouts.json")))
ids = torch.tensor(R["tokens"], device=DEV)
P, H = R["prefix"], R["horizon"]
vpd = load_vpd(load_target(DEV), DEV)


def kl(tgt, lg):
    lp, lq = F.log_softmax(tgt, -1), F.log_softmax(lg, -1)
    return (lp.exp() * (lp - lq)).sum(-1)


@torch.no_grad()
def masked_logits(b, g, how):
    m = {n: (v > 0).float() if how == "rounded" else v for n, v in g.items()}
    d = {n: torch.zeros(v.shape[:-1], device=DEV) for n, v in g.items()}
    return vpd.masked(b, m, d)


out = {"prefix": P, "horizon": H}
with torch.no_grad():
    tgt, g = vpd.target_and_ci(ids)
    for how in ("rounded", "ci"):
        out[f"{how}_whole"] = kl(tgt, masked_logits(ids, g, how))[:, P - 1:P - 1 + H].cpu().tolist()
    causal = {"rounded": [], "ci": []}
    for t in range(P - 1, P - 1 + H):
        b = ids[:, :t + 1]
        tg, gg = vpd.target_and_ci(b)
        for how in causal:
            causal[how].append(kl(tg[:, -1], masked_logits(b, gg, how)[:, -1]).cpu())
    for how in causal:
        out[f"{how}_causal"] = torch.stack(causal[how], 1).tolist()
for k, v in out.items():
    if isinstance(v, list):
        x = torch.tensor(v)
        print(k, f"mean {x.mean():.4f} max {x.max():.4f} seqsum {x.sum(1).mean():.3f}")
json.dump(out, open(os.path.expanduser("~/mpd-data/figures/data/rollout_kl.json"), "w"))
