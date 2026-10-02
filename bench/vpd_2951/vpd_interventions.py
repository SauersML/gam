"""The paper's attribution-graph interventions on the target (ablating one rank-one subcomponent
at one position: W -> W - U_c V_c^T there, everything else exact incl. the delta component).

  gender:  "The princess lost" -> P(" her") / P(" his") at " lost"; ablate 3.attn.o:2:281
  bracket: "<u,v" -> P(">") at "v"; ablate 3.mlp.down:3:1414, 2.mlp.down:3:1560, both
Paper: P(>) 0.547 -> 0.158 / 0.243 / 0.046; ablating 3.attn.o:2:281 flips her to his.

usage: vpd_interventions.py OUT.json
"""

import json
import sys

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from vpd_model import load_target, load_vpd

tok = Tokenizer.from_file("t-9d2b8f02/tokenizer.json")
target = load_target("mps")
vpd = load_vpd(target, "mps")
enc = lambda s: torch.tensor([tok.encode(s).ids], device="mps")
tid = lambda s: tok.encode(s).ids[0]


@torch.no_grad()
def run(ids, ablations: list[tuple[str, int, int]]):
    T = ids.shape[1]
    masks = {n: torch.ones(1, T, vpd.C[n], device="mps") for n in vpd.names}
    delta = {n: torch.ones(1, T, device="mps") for n in vpd.names}
    for site, pos, c in ablations:
        masks[site][0, pos, c] = 0.0
    return F.softmax(vpd.masked(ids, masks, delta)[0], -1)


@torch.no_grad()
def ci_at(ids, site, pos, c):
    _, g = vpd.target_and_ci(ids)
    return g[site][0, pos, c].item()


out = {}
cases = {
    "gender": ("The princess lost", 2, {" her": tid(" her"), " his": tid(" his")},
               {"3.attn.o:2:281": [("h.3.attn.o_proj", 2, 281)]}),
    "gender_prince": ("The prince lost", 2, {" her": tid(" her"), " his": tid(" his")},
                      {"3.attn.o:2:281": [("h.3.attn.o_proj", 2, 281)]}),
    "bracket": ("<u,v", 3, {">": tid(">")},
                {"3.mlp.down:3:1414": [("h.3.mlp.down_proj", 3, 1414)],
                 "2.mlp.down:3:1560": [("h.2.mlp.down_proj", 3, 1560)],
                 "both": [("h.3.mlp.down_proj", 3, 1414), ("h.2.mlp.down_proj", 3, 1560)]}),
}
for name, (prompt, pos, toks, abls) in cases.items():
    ids = enc(prompt)
    vpd.clear()
    with torch.no_grad():
        p0 = F.softmax(target(ids)[0], -1)
    r = {"prompt": prompt, "pos": pos, "target": {k: p0[pos, v].item() for k, v in toks.items()},
         "target_top": tok.id_to_token(int(p0[pos].argmax())),
         "exact_masked_check": {k: run(ids, [])[pos, v].item() for k, v in toks.items()}, "ablations": {}}
    for an, spec in abls.items():
        p = run(ids, spec)
        r["ablations"][an] = {**{k: p[pos, v].item() for k, v in toks.items()},
                              "top": tok.id_to_token(int(p[pos].argmax())),
                              "ci": [ci_at(ids, s, q, c) for s, q, c in spec]}
    out[name] = r
    print(json.dumps(r))
json.dump(out, open(sys.argv[1], "w"), indent=1)
