"""Weight numbers that ran per word on VPD 4L (#2951): a rank-one subcomponent of an m x n matrix runs
m + n numbers; the dense model runs every weight of the 24 matrices on every word.

Sets: CSR over the frontier rows (128 val rows at offset 1024), global index = site offset + subcomponent,
sites in sites.txt order. Shapes and the uncovered remainder ||W - sum u v^T|| / ||W|| come from the Step A
log. Writes ~/mpd-data/figures/data/weights_ran.json.

usage: weights_ran_data.py [NAME=SETS_DIR ...]      (default: vpd=~/mpd-data/pieces/vpd4l_sets)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

P = Path.home() / "mpd-data/pieces"
LOG = P / "vpd4l/stepA_warm_1024.log"
shape, remainder = {}, {}
for line in LOG.read_text().splitlines():
    m = re.match(r"(blocks\.\d+\.\w+): (\d+)×(\d+), \d+ given pieces, ‖W − Σ u vᵀ‖/‖W‖ = ([\d.e+-]+)", line)
    if m:
        shape[m[1]] = (int(m[2]), int(m[3]))
        remainder[m[1]] = float(m[4])
dense_site = {s: a * b for s, (a, b) in shape.items()}
sets = dict(a.split("=", 1) for a in sys.argv[1:]) or {"vpd": str(P / "vpd4l_sets")}
out = {"dense_per_word": sum(dense_site.values()), "dense_per_site": dense_site, "shape": shape,
       "remainder": remainder, "sets": {}}
for name, d in sets.items():
    d = Path(d)
    sites = [(n, int(c)) for n, c in (l.split() for l in (d / "sites.txt").read_text().splitlines())]
    offs = np.cumsum([0] + [c for _, c in sites])
    cost = np.concatenate([np.full(c, sum(shape[n])) for n, c in sites])  # m + n per subcomponent
    ip = np.fromfile(d / "indptr.i64", "<i8")
    ix = np.fromfile(d / "indices.i64", "<i8")
    T = len(ip) - 1
    per_word = np.add.reduceat(cost[ix], ip[:-1]) * (np.diff(ip) > 0)
    site_of = np.searchsorted(offs, ix, side="right") - 1
    per_site_w = np.bincount(site_of, weights=cost[ix], minlength=len(sites)) / T
    per_site_n = np.bincount(site_of, minlength=len(sites)) / T
    out["sets"][name] = {"dir": str(d), "tokens": T, "per_word": per_word.tolist(),
                         "subcomponents_per_word": float(len(ix) / T),
                         "per_site_weights": {n: float(w) for (n, _), w in zip(sites, per_site_w)},
                         "per_site_subcomponents": {n: float(k) for (n, _), k in zip(sites, per_site_n)}}
    print(name, "mean weights/word", per_word.mean(), "subcomponents/word", len(ix) / T)
json.dump(out, open(Path.home() / "mpd-data/figures/data/weights_ran.json", "w"))
print("dense weights/word", out["dense_per_word"])
