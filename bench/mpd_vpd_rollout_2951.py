"""VPD's rollout fidelity on the engine suite's sampled sequences (#2951), matched to the suite's
rollout columns: per rollout the summed per-step KL(target || VPD) over the model-sampled
continuation, its mean and max, and greedy agreement, for VPD's rounded (g > 0) and CI masks with
no delta component (the scoreboard's S1 program).

usage: mpd_vpd_rollout_2951.py ROLLOUTS_JSON OUT_JSON
"""
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/mpd-data/vpd"))
import torch  # noqa: E402
from vpd_model import load_target, load_vpd  # noqa: E402

r = json.load(open(sys.argv[1]))
prefix, horizon = r["prefix"], r["horizon"]
toks = torch.tensor(r["tokens"], dtype=torch.long)
ids = toks[:, :-1].to("mps")
target = load_target("mps")
vpd = load_vpd(target, "mps")
out = {"rollouts": toks.shape[0], "horizon": horizon, "prefix": prefix}
with torch.no_grad():
    tgt, g = vpd.target_and_ci(ids)
    steps = slice(prefix - 1, prefix - 1 + horizon)
    lp = torch.log_softmax(tgt[:, steps].cpu().double(), -1)
    l0 = sum((v[:, steps] > 0).float().sum(-1) for v in g.values()).mean().item()
    for name, masks in (("rounded_masked", {n: (v > 0).float() for n, v in g.items()}), ("ci_masked", g)):
        lg = vpd.masked(ids, masks, None)
        lq = torch.log_softmax(lg[:, steps].cpu().double(), -1)
        kl = (lp.exp() * (lp - lq)).sum(-1)  # rollouts × horizon
        seq = kl.sum(-1)
        out[name] = {
            "sequence_kl_mean": seq.mean().item(),
            "sequence_kl_max": seq.max().item(),
            "teacher_forced_kl_mean": kl.mean().item(),
            "teacher_forced_kl_max": kl.max().item(),
            "greedy_agreement": (lp.argmax(-1) == lq.argmax(-1)).double().mean().item(),
            "l0_per_token": l0,
        }
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps(out))
