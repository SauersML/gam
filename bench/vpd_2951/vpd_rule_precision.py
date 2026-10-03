"""The bits of VPD's selection rule (#2951): its causal-importance network sent at the lowest
uniform precision that leaves its selections unchanged, priced like every real in an explanation
(gam_mpd::describe: a dyadic lattice 2^-p, each lattice integer in the signed Elias δ code).

Every parameter w is sent as round(w · 2^p); p is the smallest for which the rounded network's
selections (CI > 0) on the 32 frontier passages, read on the model's clean states as VPD computes
its published sets, equal the unrounded network's in every entry. p is found by bisection over
[0, 40] (exactness is checked at the end). The bits are the signed δ lengths of all lattice
integers plus p once, in the prefix integer code.

The δ length mirrors gam_mpd::codec::elias_delta_len_bits of zigzag(q) + 1:
  low = floor(log2 N), length = low + 2 floor(log2(low + 1)) + 1.

usage: vpd_rule_precision.py OUT.json [--device cuda|mps|cpu]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vpd_model import load_target, load_vpd, site_names, val_tokens

FRONTIER_ROW0, PASSAGES = 1024, 32

parser = argparse.ArgumentParser()
parser.add_argument("out", type=Path)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
args = parser.parse_args()

target = load_target(args.device)
vpd = load_vpd(target, args.device)
names = site_names()
ids = val_tokens(PASSAGES, offset=FRONTIER_ROW0).to(args.device)
params = list(vpd.ci_fn.parameters())
original = [p.detach().double().cpu().clone() for p in params]

# The clean states' site inputs, kept on the device for every rounding.
with torch.no_grad():
    inputs = []
    for r in range(PASSAGES):
        vpd.clear()
        for n in names:
            target.site(n).cache_input = True
        target(ids[r:r + 1])
        inputs.append({n: target.site(n).last_input for n in names})
        vpd.clear()


def selections():
    with torch.no_grad():
        return [torch.cat([(c[0] > 0) for c in vpd.ci_fn(x).values()], dim=-1) for x in inputs]


reference = selections()


def rounded_equal(p: int) -> bool:
    with torch.no_grad():
        for param, w in zip(params, original):
            param.copy_((torch.round(w * 2.0**p) / 2.0**p).to(param.dtype))
    same = all(bool((a == b).all()) for a, b in zip(selections(), reference))
    with torch.no_grad():
        for param, w in zip(params, original):
            param.copy_(w.to(param.dtype))
    return same


def delta_bits(q: np.ndarray) -> float:
    z = np.where(q >= 0, 2 * q, -2 * q - 1).astype(np.uint64) + np.uint64(1)
    low = np.floor(np.log2(z.astype(np.float64))).astype(np.int64)
    return float((low + 2 * np.floor(np.log2(low + 1)).astype(np.int64) + 1).sum())


def prefix_bits(value: int) -> int:
    bits, group = 1, value
    while group > 1:
        width = group.bit_length()
        bits += width
        group = width - 1
    return bits


lo, hi = 0, 40
if not rounded_equal(hi):
    raise SystemExit(f"selections differ even at p = {hi}")
while lo < hi:
    mid = (lo + hi) // 2
    if rounded_equal(mid):
        hi = mid
    else:
        lo = mid + 1
p = hi
bits = sum(delta_bits(np.round(w.numpy() * 2.0**p).astype(np.int64).ravel()) for w in original) + prefix_bits(p + 1)
reals = int(sum(w.numel() for w in original))
record = {"precision": p, "reals": reals, "bits": bits, "bits_per_real": bits / reals,
          "below_changes": not rounded_equal(p - 1) if p > 0 else None,
          "test": f"CI > 0 on the clean states of {PASSAGES} frontier passages, every entry equal"}
json.dump(record, open(args.out, "w"), indent=1)
print(record)
