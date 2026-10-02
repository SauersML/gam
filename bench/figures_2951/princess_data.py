"""VPD 4L on "The princess lost": every active subcomponent (g > 0) at every position, and what
removing it alone does to the her-vs-his logit gap at " lost" (full model otherwise, delta on)."""
import json
import os
import sys

import torch
import torch.nn.functional as F

os.chdir(os.path.expanduser("~/mpd-data/vpd"))  # tokenizer and runs are read relative to here
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vpd_2951"))
from tokenizers import Tokenizer  # noqa: E402

from vpd_model import load_target, load_vpd  # noqa: E402

DEV = "mps"
tok = Tokenizer.from_file("t-9d2b8f02/tokenizer.json")
vpd = load_vpd(load_target(DEV), DEV)
prompt, pos = "The princess lost", 2
ids = torch.tensor([tok.encode(prompt).ids], device=DEV)
T = ids.shape[1]
her, his = tok.encode(" her").ids[0], tok.encode(" his").ids[0]
_, g = vpd.target_and_ci(ids)
active = [(n, t, c, g[n][0, t, c].item()) for n in vpd.names for t in range(T)
          for c in torch.nonzero(g[n][0, t] > 0).flatten().tolist()]
print(len(active), "active")


@torch.no_grad()
def gaps(abl):
    B = len(abl)
    masks = {n: torch.ones(B, T, vpd.C[n], device=DEV) for n in vpd.names}
    delta = {n: torch.ones(B, T, device=DEV) for n in vpd.names}
    for b, a in enumerate(abl):
        for n, t, c in a:
            masks[n][b, t, c] = 0.0
    lp = F.log_softmax(vpd.masked(ids.expand(B, T), masks, delta)[:, pos], -1)
    return (lp[:, her] - lp[:, his]).cpu(), lp.exp()[:, [her, his]].cpu()


base_gap, base_p = gaps([[]])
effects = []
for i in range(0, len(active), 64):
    chunk = active[i:i + 64]
    gp, _ = gaps([[(n, t, c)] for n, t, c, _ in chunk])
    effects += (gp - base_gap[0]).tolist()
_, p281 = gaps([[("h.3.attn.o_proj", 2, 281)]])
out = {"prompt": prompt, "tokens": [tok.id_to_token(i) for i in ids[0].tolist()], "pos": pos,
       "C": vpd.C, "names": vpd.names, "p_her_his": base_p[0].tolist(), "p_her_his_without_281": p281[0].tolist(),
       "base_gap": base_gap[0].item(),
       "active": [{"site": n, "t": t, "c": c, "ci": ci, "d_gap": e} for (n, t, c, ci), e in zip(active, effects)]}
json.dump(out, open(os.path.expanduser("~/mpd-data/figures/data/princess.json"), "w"), indent=0)
top = sorted(out["active"], key=lambda a: a["d_gap"])[:12]
for a in top:
    print(a)
