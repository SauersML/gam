"""Scan Pile val rows with the VPD CI network for the edit-comparison components, checkpointing
every CKPT rows so a killed run resumes (the rows already scanned are kept).

usage: vpd_scan_edits.py ROW_START ROW_END OUT.pt
"""

import os
import sys
import time

import torch

from vpd_model import load_target, load_vpd, val_tokens

WATCH = {
    "h.2.mlp.down_proj": [2359, 1560],
    "h.3.attn.o_proj": [281],
    "h.3.mlp.down_proj": [1414],
}
MB, CHUNK, CKPT = 8, 1024, 2048

r0, r1, out_path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
t0 = time.time()
fires = {(n, c): [] for n, cs in WATCH.items() for c in cs}
done = r0
if os.path.exists(out_path):
    prev = torch.load(out_path)
    assert prev["rows"][0] == r0
    done = prev["rows"][1]
    fires = {k: [v] for k, v in prev["fires"].items()}
    print(f"resuming at row {done}", flush=True)
target = load_target("mps")
vpd = load_vpd(target, "mps")


def save(upto: int) -> None:
    n_tok = (upto - r0) * 512
    torch.save({"rows": (r0, upto), "n_tokens": n_tok,
                "fires": {k: torch.cat(v) if v else torch.zeros(0, 3, dtype=torch.long) for k, v in fires.items()}},
               out_path + ".tmp")
    os.replace(out_path + ".tmp", out_path)


with torch.no_grad():
    for i in range(done - r0, r1 - r0, MB):
        if (i - (done - r0)) % CHUNK == 0:
            ids = val_tokens(min(CHUNK, r1 - r0 - i), offset=r0 + i)
            base = i
        b = ids[i - base:i - base + MB]
        _, g = vpd.target_and_ci(b.to("mps"))
        for n, cs in WATCH.items():
            sub = g[n][..., cs].cpu()
            for j, c in enumerate(cs):
                rr, pp = (sub[..., j] > 0).nonzero(as_tuple=True)
                fires[(n, c)].append(torch.stack([rr + r0 + i, pp, (sub[..., j][rr, pp] * 1e6).long()], 1))
        del g
        if (i + MB) % CKPT == 0:
            save(r0 + i + MB)
            print(f"[{time.time() - t0:.0f}s] rows {r0 + i + MB}", flush=True)
save(r1)
for (n, c), f in {k: torch.cat(v) for k, v in fires.items()}.items():
    print(f"{n}:{c}  density {f.shape[0] / ((r1 - r0) * 512):.6f}  n={f.shape[0]}")
