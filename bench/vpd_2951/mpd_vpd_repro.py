"""Reproduce the VPD paper decomposition's (goodfire/spd/runs/s-55ea3f9b) eval metrics on held-out
Pile val rows: CI_L0, the CEandKL battery, and fresh shared-across-batch PGD recon, per
eval batch of 128 x 512 (the run's eval_batch_size, one batch per eval point).

usage: mpd_vpd_repro.py N_BATCHES BATCH_ROWS PGD_STEPS[,..] [out.json]
"""

import json
import sys
import time

import torch

from vpd_eval import ce_kl_battery, gates_and_l0, pgd_recon
from vpd_model import load_target, load_vpd, val_tokens

n_batches, rows = int(sys.argv[1]), int(sys.argv[2])
steps = [int(x) for x in sys.argv[3].split(",")]
out_path = sys.argv[4] if len(sys.argv) > 4 else None

t0 = time.time()
target = load_target("mps")
vpd = load_vpd(target, "mps")
print(f"loaded in {time.time() - t0:.0f}s; checkpoint target vs t-9d2b8f02 max|dW| = {vpd.target_weight_diff:.2e}", flush=True)

results = []
for bi in range(n_batches):
    ids = val_tokens(rows, offset=bi * rows).to("mps")
    t = time.time()
    sg, l0 = gates_and_l0(vpd, ids)
    r = {"batch": bi, "l0_total": sum(l0.values()), "l0": l0}
    for k in range(4):
        r[f"l0_layer_{k}"] = sum(v for n, v in l0.items() if n.startswith(f"h.{k}."))
    r.update(ce_kl_battery(vpd, ids, sg, seed=bi))
    print(f"[batch {bi}] L0 {r['l0_total']:.1f}  " + "  ".join(f"{k}={r[k]:.4f}" for k in r if k.startswith("kl_")) + f"  ({time.time() - t:.0f}s)", flush=True)
    t = time.time()
    r["pgd_with_delta"] = pgd_recon(vpd, ids, sg, steps, with_delta=True, seed=bi)
    print(f"[batch {bi}] PGD (delta in program) {r['pgd_with_delta']}  ({time.time() - t:.0f}s)", flush=True)
    t = time.time()
    r["pgd_no_delta"] = pgd_recon(vpd, ids, sg, steps, with_delta=False, seed=bi)
    print(f"[batch {bi}] PGD (delta excluded)   {r['pgd_no_delta']}  ({time.time() - t:.0f}s)", flush=True)
    results.append(r)
    if out_path:
        json.dump(results, open(out_path, "w"), indent=1)
    del sg
    torch.mps.empty_cache()
