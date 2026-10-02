"""VPD's points on the bits-vs-error frontier, and the dense/zero endpoints, on the matched spec.

VPD's explanation = subcomponents (U, V at 24 sites) + causal-importance network, every real on
a per-tensor dyadic lattice (gam #2951 LatticeCode, Elias-omega indices; vpd_bits.py). The
declared precision per tensor is p = b - round(log2 rms): b bits of resolution under the RMS.
The delta component (W - V U) is not part of the program: it needs W, which the decoder is
not given. Error is scored against the ORIGINAL fp32 target on held-out Pile rows:
  kl_unmasked      all components on (parameter faithfulness of the decoded program)
  kl_ci_masked     gates as masks
  kl_rounded       binarized gates (g > 0) as masks
  kl_stoch         mask = g + (1-g) U(0,1)
  pgd20            VPD's eval adversary: shared-across-batch source, 20 sign steps of 0.1
L0 is the mean count of gates > 0 per position (all 24 sites).

Dense endpoint: the 24 target matrices themselves on the same lattice code (no gates, so the
adversary has nothing to ablate: its error is its unmasked KL). Zero endpoint: no matrices.

usage: mpd_vpd_frontier.py ROWS OFFSET B[,B..] [out.json]    (B = 'fp32' for the published reals)
"""

import json
import sys
import time

import torch

from vpd_bits import lattice_bits, precision_for, quantize
from vpd_eval import MB, ce_kl_battery, gates_and_l0, kl_per_pos, pgd_recon
from vpd_model import VPD_PTH, load_target, load_vpd, site_names, val_tokens

rows, offset = int(sys.argv[1]), int(sys.argv[2])
bs = sys.argv[3].split(",")
out_path = sys.argv[4] if len(sys.argv) > 4 else None

target = load_target("mps")
vpd = load_vpd(target, "mps")
ids = val_tokens(rows, seq=int(__import__("os").environ.get("VPD_SEQ", "512")), offset=offset).to("mps")


def ci_param_map(ci) -> dict[str, torch.Tensor]:
    m = {"_input_projector.W": ci.proj_in.W, "_input_projector.b": ci.proj_in.b,
         "_output_head.W": ci.head.W, "_output_head.b": ci.head.b}
    for i, blk in enumerate(ci.blocks):
        m.update({f"_blocks.{i}.attn.q_proj.weight": blk.wq, f"_blocks.{i}.attn.k_proj.weight": blk.wk,
                  f"_blocks.{i}.attn.v_proj.weight": blk.wv, f"_blocks.{i}.attn.out_proj.weight": blk.wo,
                  f"_blocks.{i}.mlp.0.W": blk.fc1.W, f"_blocks.{i}.mlp.0.b": blk.fc1.b,
                  f"_blocks.{i}.mlp.2.W": blk.fc2.W, f"_blocks.{i}.mlp.2.b": blk.fc2.b})
    return m


ci_params = ci_param_map(vpd.ci_fn)
TABLE = json.load(open("bits_table.json"))  # {b: {uv, ci, target24: {n_reals, bits}}} from vpd_bits_table.py


def install(b: str) -> dict[str, float]:
    """Load the published reals from the checkpoint, put them on the lattice at resolution b,
    write them into the live modules, and return the code length by part."""
    raw = torch.load(str(VPD_PTH), map_location="cpu", weights_only=True, mmap=True)
    bits = {"uv": 0, "ci": 0, "n_uv": 0, "n_ci": 0}
    for k, x in raw.items():
        if k.startswith("_components."):
            site, which = k[len("_components."):].rsplit(".", 1)
            st = target.site(site.replace("-", "."))
            dst, part = (st.U if which == "U" else st.V), "uv"
        elif "_global_ci_fn." in k and not k.endswith("rope.inv_freq"):
            dst, part = ci_params[k.split("_global_ci_fn.", 1)[1]], "ci"
        else:
            continue  # the target's own weights; rope.inv_freq is a declared constant (base 1e4)
        if b == "fp32":
            q, nb = x, 32 * x.numel()
        else:
            p = precision_for(x, int(b))
            # the code length is the same lattice_bits sum bits_table.json already holds per part
            q, nb = quantize(x, p), (0 if b in TABLE else lattice_bits(x, p))
        with torch.no_grad():
            dst.copy_(q.to(dst.device))
        bits[part] += nb
        bits["n_" + part] += x.numel()
    if b in TABLE:
        bits["uv"], bits["ci"] = TABLE[b]["uv"]["bits"], TABLE[b]["ci"]["bits"]
    del raw
    import gc
    gc.collect()
    torch.mps.empty_cache()
    return bits


def evaluate(expl, gated: bool = True) -> dict:
    t = time.time()
    sg, l0 = gates_and_l0(expl, ids)
    r = {"l0_total": sum(l0.values()), "l0": l0}
    r.update(ce_kl_battery(expl, ids, sg, seed=0, with_delta=False,
                           strategies=("ci_masked", "unmasked", "stoch_masked", "rounded_masked")))
    r["pgd20"] = pgd_recon(expl, ids, sg, [20], with_delta=False, seed=0)[20] if gated else r["kl_unmasked"]
    r["eval_seconds"] = time.time() - t
    return r


class Dense:
    """The trivial explanation: the target matrices on the lattice, always on."""

    def __init__(self, b: int):
        self.names, self.C, self.bits, self.n = site_names(), {}, 0, 0
        self.Wq = {}
        for n in self.names:
            W = target.site(n).W
            p = precision_for(W, b)
            self.Wq[n], self.bits, self.n = quantize(W, p), self.bits + lattice_bits(W.cpu(), p), self.n + W.numel()

    def target_forward(self, x):
        return vpd.target_forward(x)

    def target_and_ci(self, x):
        return self.target_forward(x), {}

    @torch.no_grad()
    def masked(self, x, masks, delta):
        orig = {n: target.site(n).W for n in self.names}
        for n in self.names:
            target.site(n).W = self.Wq[n]
        try:
            return target(x)
        finally:
            for n in self.names:
                target.site(n).W = orig[n]


import os
results = json.load(open(out_path)) if out_path and os.path.exists(out_path) else []
done = {r["b"] for r in results}
for b in bs:
    if b in done:
        continue
    if b.startswith("dense"):
        d = Dense(int(b[5:]))
        kl = sum(kl_per_pos(d.masked(ids[i:i + MB], None, None), d.target_forward(ids[i:i + MB])).mean().item()
                 for i in range(0, rows, MB)) / (rows // MB)
        r = {"b": b, "bits_total": d.bits, "n_reals": d.n, "kl_unmasked": kl, "pgd20": kl, "l0_total": None}
    else:
        bits = install(b)
        r = {"b": b, "bits_uv": bits["uv"], "bits_ci": bits["ci"], "bits_total": bits["uv"] + bits["ci"],
             "n_uv": bits["n_uv"], "n_ci": bits["n_ci"]}
        r.update(evaluate(vpd))
    print(json.dumps({k: v for k, v in r.items() if k != "l0"}), flush=True)
    results.append(r)
    if out_path:
        json.dump(results, open(out_path, "w"), indent=1)
