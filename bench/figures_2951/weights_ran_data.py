"""Weight numbers that ran per word on VPD 4L (#2951): a rank-one subcomponent of an m x n matrix runs
m + n numbers; the dense model runs every weight of the 24 matrices on every word.

Sets: CSR over the frontier rows (128 val rows at offset 1024), global index = site offset + subcomponent,
sites in sites.txt order. Shapes and the uncovered remainder ||W - sum u v^T|| / ||W|| come from the Step A
log. Writes ~/mpd-data/figures/data/weights_ran.json.

A set is either a directory (indptr.i64, indices.i64, sites.txt) or a .npy prefix PREFIX with
PREFIX.{indptr,indices,offsets}.npy (sites in vpd4l_sets/sites.txt order). rows=N keeps the first N rows of
512 tokens of every set, so sets dumped on fewer rows are compared on the same words. tag=T writes
weights_ran_T.json instead.

usage: weights_ran_data.py [rows=N] [tag=T] [NAME=SETS ...]      (default: vpd=~/mpd-data/pieces/vpd4l_sets)
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
args = dict(a.split("=", 1) for a in sys.argv[1:])
rows = int(args.pop("rows")) if "rows" in args else None
tag = args.pop("tag", None)
sets = args or {"vpd": str(P / "vpd4l_sets")}
SITES = [(n, int(c)) for n, c in (l.split() for l in (P / "vpd4l_sets/sites.txt").read_text().splitlines())]


def load(spec: str):
    """(sites, indptr, indices) of a set given as a directory or a .npy prefix."""
    d = Path(spec)
    if d.is_dir():
        sites = [(n, int(c)) for n, c in (l.split() for l in (d / "sites.txt").read_text().splitlines())]
        return sites, np.fromfile(d / "indptr.i64", "<i8"), np.fromfile(d / "indices.i64", "<i8")
    ip, ix = np.load(f"{spec}.indptr.npy").astype(np.int64), np.load(f"{spec}.indices.npy").astype(np.int64)
    offs = np.load(f"{spec}.offsets.npy").astype(np.int64)
    assert np.array_equal(offs[:len(SITES) + 1], np.cumsum([0] + [c for _, c in SITES])), "site offsets differ"
    return SITES, ip, ix
out = {"rows": rows, "dense_per_word": sum(dense_site.values()), "dense_per_site": dense_site, "shape": shape,
       "remainder": remainder, "sets": {}}
for name, d in sets.items():
    sites, ip, ix = load(d)
    if rows is not None:
        ip = ip[:rows * 512 + 1]
        ix = ix[:ip[-1]]
    offs = np.cumsum([0] + [c for _, c in sites])
    cost = np.concatenate([np.full(c, sum(shape[n])) for n, c in sites])  # m + n per subcomponent
    T = len(ip) - 1
    per_word = np.add.reduceat(cost[ix], ip[:-1]) * (np.diff(ip) > 0)
    site_of = np.searchsorted(offs, ix, side="right") - 1
    per_site_w = np.bincount(site_of, weights=cost[ix], minlength=len(sites)) / T
    per_site_n = np.bincount(site_of, minlength=len(sites)) / T
    out["sets"][name] = {"source": d, "tokens": T, "per_word": per_word.tolist(),
                         "subcomponents_per_word": float(len(ix) / T),
                         "per_site_weights": {n: float(w) for (n, _), w in zip(sites, per_site_w)},
                         "per_site_subcomponents": {n: float(k) for (n, _), k in zip(sites, per_site_n)}}
    print(name, "mean weights/word", per_word.mean(), "subcomponents/word", len(ix) / T)
json.dump(out, open(Path.home() / f"mpd-data/figures/data/weights_ran{'_' + tag if tag else ''}.json", "w"))
print("dense weights/word", out["dense_per_word"])
