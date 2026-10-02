"""Scan held-out Pile val rows with the VPD CI network and record where the case-study and
edit subcomponents are causally important (CI > 0).

usage: vpd_scan.py ROW_START ROW_END OUT.pt
"""

import sys
import time

import torch

from vpd_model import load_target, load_vpd, val_tokens

WATCH = {
    "h.2.mlp.down_proj": [1672, 2359, 2623, 3290, 3327, 3382],
    "h.1.attn.q_proj": [308, 316],
    "h.1.attn.k_proj": [119, 218, 329, 485],
}
MB = 8
if "VPD_SCAN_SITE" in __import__("os").environ:  # e.g. h.2.mlp.down_proj:2359,3382 (only what the edit needs)
    _n, _c = __import__("os").environ["VPD_SCAN_SITE"].split(":")
    WATCH = {_n: [int(c) for c in _c.split(",")]}
CHUNK = 1024

r0, r1, out_path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
t0 = time.time()
target = load_target("mps")
vpd = load_vpd(target, "mps")
fires = {(n, c): [] for n, cs in WATCH.items() for c in cs}
n_tokens = 0
with torch.no_grad():
    for i in range(0, r1 - r0, MB):
        if i % CHUNK == 0:
            ids = val_tokens(min(CHUNK, r1 - r0 - i), offset=r0 + i)
        b = ids[i % CHUNK:i % CHUNK + MB]
        n_tokens += b.numel()
        _, g = vpd.target_and_ci(b.to("mps"))
        for n, cs in WATCH.items():
            sub = g[n][..., cs].cpu()
            for j, c in enumerate(cs):
                rr, pp = (sub[..., j] > 0).nonzero(as_tuple=True)
                fires[(n, c)].append(torch.stack([rr + r0 + i, pp, (sub[..., j][rr, pp] * 1e6).long()], 1))
        del g
        if i % 256 == 0:
            print(f"[{time.time() - t0:.0f}s] rows {r0 + i}", flush=True)
fires = {k: torch.cat(v) for k, v in fires.items()}
torch.save({"rows": (r0, r1), "n_tokens": n_tokens, "fires": fires}, out_path)
for (n, c), f in fires.items():
    print(f"{n}:{c}  density {f.shape[0] / n_tokens:.6f}  n={f.shape[0]}")
