"""S4: the paper's layer-1 attention case studies as behaviour-scoped readouts on held-out rows.

Readouts (all on layer h.1, target model, full-row context):
  B1 previous-token: per head, mean attention from query t to t-1 (t >= 1), and mass on
     offsets 1..4. Ablations remove one rank-one subcomponent from the weight
     (W -> W - U_c V_c^T): q:316, k:329, both; control = the largest effect among the other
     20 largest-norm q subcomponents ablated one at a time.
  B2 it + copula: query = a copula token, key = an `it` token 1..3 positions earlier; per head,
     mean attention query -> it. Ablations: q:308, k:218, k:485, k:218+k:485.
  B3 there + copula: as B2 with key in {there, here}; same ablations.
CI firing of the named subcomponents on those query/key positions comes from the scan.

usage: vpd_cases.py SCAN.pt N_ROWS OUT.json
"""

import json
import sys
import time

import torch
from tokenizers import Tokenizer

from vpd_model import load_target, load_vpd, val_tokens

scan = torch.load(sys.argv[1])
n_rows, out_path = int(sys.argv[2]), sys.argv[3]
t0 = time.time()
target = load_target("mps")
vpd = load_vpd(target, "mps")
vpd.clear()
tok = Tokenizer.from_file("t-9d2b8f02/tokenizer.json")
r0, _ = scan["rows"]
ids = val_tokens(n_rows, offset=r0)
Q, K = "h.1.attn.q_proj", "h.1.attn.k_proj"
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)


def tid(words):
    out = set()
    for w in words:
        e = tok.encode(w).ids
        if len(e) == 1:
            out.add(e[0])
    return out


COPULA = tid([" is", " was", " are", " were", " be", " been", " being", " seems", " seem", " seemed",
              " appears", " appear", " appeared", " becomes", " became", " remains", " remained", "'s"])
IT = tid([" it", " It", "It", "it", " IT"])
THERE = tid([" there", " There", "There", " here", " Here", "Here"])

# query/key pairs for B2 / B3
pairs = {"it": [], "there": []}
for r in range(n_rows):
    row = ids[r].tolist()
    for t in range(1, 512):
        if row[t] in COPULA:
            for d in (1, 2, 3):
                if t - d >= 0 and row[t - d] in IT:
                    pairs["it"].append((r, t, t - d))
                    break
                if t - d >= 0 and row[t - d] in THERE:
                    pairs["there"].append((r, t, t - d))
                    break
log(f"pairs: it+copula {len(pairs['it'])}, there+copula {len(pairs['there'])}")


@torch.no_grad()
def readouts() -> dict:
    """Per-head B1/B2/B3 readouts under the currently installed weights."""
    prev1 = torch.zeros(6, device="mps")
    prev14 = torch.zeros(6, device="mps")
    b2 = {k: torch.zeros(6, device="mps") for k in pairs}
    for i in range(0, n_rows, 4):
        b = ids[i:i + 4].to("mps")
        st_q, st_k = target.site(Q), target.site(K)
        st_q.cache_output = st_k.cache_output = True
        target(b)
        A = target.attention_pattern(st_q.last_output, st_k.last_output)  # [B, H, T, T]
        st_q.cache_output = st_k.cache_output = False
        st_q.last_output = st_k.last_output = None
        d1 = A.diagonal(offset=-1, dim1=-2, dim2=-1)  # [B, H, T-1]
        prev1 += d1.sum((0, 2))
        prev14 += sum(A.diagonal(offset=-o, dim1=-2, dim2=-1).sum((0, 2)) for o in (1, 2, 3, 4))
        for k, pl in pairs.items():
            for r, t, s in pl:
                if i <= r < i + 4:
                    b2[k] += A[r - i, :, t, s]
        del A
    nq = n_rows * 511
    out = {"prev1": (prev1 / nq).tolist(), "prev1to4": (prev14 / nq).tolist()}
    for k, pl in pairs.items():
        out[f"{k}_copula"] = (b2[k] / max(1, len(pl))).tolist()
    return out


def ablate(site_name: str, comps: list[int]):
    st = target.site(site_name)
    W0 = st.W
    st.W = W0 - sum(torch.outer(st.U[c], st.V[:, c]) for c in comps)
    return lambda: setattr(st, "W", W0)


results = {"n_rows": n_rows, "pairs": {k: len(v) for k, v in pairs.items()}}
nt = scan["n_tokens"]
results["density"] = {f"{n}:{c}": f.shape[0] / nt for (n, c), f in scan["fires"].items()}
log(f"density {results['density']}")
# CI of the named subcomponents on the B2/B3 query/key positions (scan rows cover these rows)
fire_sets = {k: {(int(r) - r0, int(p)) for r, p, _ in v.tolist()} for k, v in scan["fires"].items()}
for k, pl in pairs.items():
    results[f"{k}_copula_ci_rate"] = {
        "q:308@query": sum((r, t) in fire_sets[(Q, 308)] for r, t, s in pl) / max(1, len(pl)),
        "k:218@key": sum((r, s) in fire_sets[(K, 218)] for r, t, s in pl) / max(1, len(pl)),
        "k:485@key": sum((r, s) in fire_sets[(K, 485)] for r, t, s in pl) / max(1, len(pl)),
    }
log(f"ci rates { {k: results[f'{k}_copula_ci_rate'] for k in pairs} }")

results["target"] = readouts()
log(f"target {results['target']}")
abl = {"q:316": [(Q, [316])], "k:329": [(K, [329])], "q:316+k:329": [(Q, [316]), (K, [329])],
       "q:308": [(Q, [308])], "k:218": [(K, [218])], "k:485": [(K, [485])],
       "k:218+k:485": [(K, [218, 485])], "q:308+k:218+k:485": [(Q, [308]), (K, [218, 485])]}
results["ablations"] = {}
for name, spec in abl.items():
    undo = [ablate(s, c) for s, c in spec]
    results["ablations"][name] = readouts()
    for u in undo:
        u()
    log(f"ablate {name}: {results['ablations'][name]}")
# control: other large-norm q subcomponents
st = target.site(Q)
norms = (st.U.norm(dim=1) * st.V.norm(dim=0)).cpu()
ctrl = [int(c) for c in norms.argsort(descending=True).tolist() if c not in (316, 308)][:20]
results["control_q"] = {}
for c in ctrl:
    undo = ablate(Q, [c])
    results["control_q"][c] = readouts()
    undo()
results["control_q_norm_rank"] = {"316": int((norms > norms[316]).sum()), "308": int((norms > norms[308]).sum())}
json.dump(results, open(out_path, "w"), indent=1)
log("done")
