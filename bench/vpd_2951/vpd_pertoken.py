"""VPD on its own terms: the per-token code of its explanation, with no Gamma.

Per token t the decoder is sent, at every one of the 24 sites s, the active set A_{s,t} (which of
the C_s components have a nonzero mask) and, unless the mask is binary, each active mask value on a
dyadic lattice. With the library (U, V at every site) the decoder then runs the masked program; the
CI network Gamma is not needed, since its outputs are transmitted. Bits per token:

    sum_s [ omega(|A_{s,t}| + 1) + log2 binom(C_s, |A_{s,t}|) + sum_{c in A_{s,t}} delta(k_c) ]

with k_c = round(g_c 2^p) >= 1 on the lattice of step 2^-p (components that round to 0 leave the
set), no coefficient term for a binary mask, and 32 bits per value for the unquantized gates.
Scored as KL(target || masked program) and top-1 agreement on the frontier's rows (32 val rows at
offset 1024, delta component excluded), with U, V as published (fp32) and on the b = 2 lattice.

usage: vpd_pertoken.py ROWS OFFSET OUT.json
"""

import json
import math
import sys
import time

import numpy as np
import torch

from vpd_bits import delta_len, omega_len, precision_for, quantize
from vpd_eval import MB, gates_and_l0, kl_per_pos
from vpd_model import load_target, load_vpd, val_tokens

rows, offset, out_path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
target = load_target("mps")
vpd = load_vpd(target, "mps")
ids = val_tokens(rows, offset=offset).to("mps")
sg, l0 = gates_and_l0(vpd, ids)
vpd.ci_fn = None  # the decoder never runs Gamma: its outputs are the transmitted gates
torch.mps.empty_cache()
log(f"gates stored; mean L0 {sum(l0.values()):.1f}")
C = vpd.C
lbinom = {n: torch.tensor([math.lgamma(C[n] + 1) - math.lgamma(k + 1) - math.lgamma(C[n] - k + 1)
                           for k in range(C[n] + 1)], dtype=torch.float64) / math.log(2) for n in vpd.names}

# (name, how the transmitted mask is formed from g, coefficient bits per active value)
VARIANTS = [("ci_fp32", None), ("rounded", "binary")] + [(f"ci_lattice_p{p}", p) for p in (1, 2, 3, 4, 6, 8)]


def mask_and_bits(g: torch.Tensor, how):
    """Transmitted mask for one site [B, S, C] and its per-position bits [B, S] (float64, CPU)."""
    if how is None:
        m, coef = g, 32.0 * (g > 0).sum(-1).double()
    elif how == "binary":
        m, coef = (g > 0).float(), torch.zeros(g.shape[:-1], dtype=torch.float64, device=g.device)
    else:
        k = torch.round(g * 2.0**how)
        m = k * 2.0**-how
        kk = k[k > 0].long().cpu().numpy()
        per = torch.from_numpy(delta_len(kk).astype(np.float64)).to(g.device)
        coef = torch.zeros(k.shape, dtype=torch.float64, device=g.device)
        coef[k > 0] = per
        coef = coef.sum(-1)
    return m, coef.cpu()


orig_uv = {n: (target.site(n).U.clone(), target.site(n).V.clone()) for n in vpd.names}


def set_uv(b):
    for n in vpd.names:
        U, V = orig_uv[n]
        st = target.site(n)
        if b == "fp32":
            st.U, st.V = U, V
        else:
            st.U, st.V = quantize(U, precision_for(U, b)), quantize(V, precision_for(V, b))


results = {"rows": rows, "offset": offset, "n_tokens": rows * ids.shape[1], "C": C, "variants": {}}
for b in ("fp32", 2):
    set_uv(b)
    for name, how in VARIANTS:
        kl = top1 = 0.0
        bits_set, bits_coef, l0_tot, n_pos = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for i in range(len(sg.mb)):
                bt = ids[i * MB:(i + 1) * MB]
                g = sg.dense(i)
                tgt = vpd.target_forward(bt)
                masks, set_b, coef_b, l0_b = {}, 0.0, 0.0, 0.0
                for n, v in g.items():
                    m, coef = mask_and_bits(v, how)
                    masks[n] = m
                    a = (m > 0).sum(-1).cpu()
                    set_b = set_b + lbinom[n][a] + torch.from_numpy(omega_len((a + 1).numpy().reshape(-1)).astype(np.float64)).view(a.shape)
                    coef_b = coef_b + coef
                    l0_b = l0_b + a.double()
                lg = vpd.masked(bt, masks, None)
                kl += kl_per_pos(lg, tgt).mean().item() / len(sg.mb)
                top1 += (lg.argmax(-1) == tgt.argmax(-1)).float().mean().item() / len(sg.mb)
                bits_set += set_b.sum().item()
                bits_coef += coef_b.sum().item()
                l0_tot += l0_b.sum().item()
                n_pos += set_b.numel()
                del g, tgt, lg, masks
        r = {"uv": str(b), "kl": kl, "top1": top1, "l0": l0_tot / n_pos, "bits_set_per_token": bits_set / n_pos,
             "bits_coef_per_token": bits_coef / n_pos, "bits_per_token": (bits_set + bits_coef) / n_pos}
        results["variants"][f"{name}@uv{b}"] = r
        log(f"{name} uv={b}: " + " ".join(f"{k}={v:.4g}" for k, v in r.items() if isinstance(v, float)))
        json.dump(results, open(out_path, "w"), indent=1)
        torch.mps.empty_cache()
log("done")
