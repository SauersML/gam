"""For the subcomponents that most support " her", how much removing each alone shrinks the
activation of 3.attn.o:2:281 (x @ V[:, 281] at " lost"): the edges into the her->his subcomponent."""
import json
import os
import sys

import torch

os.chdir(os.path.expanduser("~/mpd-data/vpd"))  # tokenizer and runs are read relative to here
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vpd_2951"))
from tokenizers import Tokenizer  # noqa: E402

from vpd_model import load_target, load_vpd  # noqa: E402

DEV = "mps"
P = os.path.expanduser("~/mpd-data/figures/data/princess.json")
d = json.load(open(P))
tok = Tokenizer.from_file("t-9d2b8f02/tokenizer.json")
vpd = load_vpd(load_target(DEV), DEV)
ids = torch.tensor([tok.encode(d["prompt"]).ids], device=DEV)
T = ids.shape[1]
o = vpd.target.site("h.3.attn.o_proj")
cand = [a for a in sorted(d["active"], key=lambda a: a["d_gap"])[:40]
        if not (a["site"] == "h.3.attn.o_proj" and a["c"] == 281)]


@torch.no_grad()
def act281(abl):
    B = len(abl)
    masks = {n: torch.ones(B, T, vpd.C[n], device=DEV) for n in vpd.names}
    delta = {n: torch.ones(B, T, device=DEV) for n in vpd.names}
    for b, a in enumerate(abl):
        for n, t, c in a:
            masks[n][b, t, c] = 0.0
    seen = []
    h = o.register_forward_hook(lambda m, inp, out: seen.append(inp[0][:, 2] @ m.V[:, 281]))
    vpd.masked(ids.expand(B, T), masks, delta)
    h.remove()
    return seen[0].cpu()


base = act281([[]])[0].item()
r = act281([[(a["site"], a["t"], a["c"])] for a in cand])
d["act281"] = base
d["edges281"] = [{"site": a["site"], "t": a["t"], "c": a["c"], "frac_left": (x / base)} for a, x in zip(cand, r.tolist())]
for e in d["edges281"]:
    print(e)
json.dump(d, open(P, "w"), indent=0)
