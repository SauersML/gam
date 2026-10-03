"""The box-claim frontier from per-passage sweep runs (#2951): for each run directory
(~/mpd-data/cluster/sweep1e5, …), every finished passage's point (`p{task}.json`) and sets
(`p{task}.pass0.*.npy`) merged in passage order into `merged.pass0.{indptr,indices,offsets}.npy`,
and per n one frontier line for our sets, VPD's sets and everything on: active subcomponents,
weight numbers run per word (Σ over on subcomponents of d_in + d_out − 1), KL at the masks and
under the box claim (the run's worst case), and bits per word. Writes `frontier.json` beside the
runs.

usage: vpd_frontier_merge.py RUN_DIR [RUN_DIR …]
"""

import json
import sys
from pathlib import Path

import numpy as np

library = json.load(open(Path.home() / "mpd-data/pieces/vpd4l_library/manifest.json"))
sites = [line.split()[0] for line in open(Path.home() / "mpd-data/pieces/vpd4l_sets/sites.txt")]
weights_of = np.concatenate([np.full(library[s]["pieces"], library[s]["d_in"] + library[s]["d_out"] - 1) for s in sites])

frontier = []
for run in map(Path, sys.argv[1:]):
    tasks = sorted((int(p.stem[1:]), p) for p in run.glob("p*.json") if p.stem[1:].isdigit())
    points, indptr, indices = [], [np.zeros(1, dtype=np.int64)], []
    total = 0
    for task, path in tasks:
        stem = path.with_suffix("")
        csr = [Path(f"{stem}.pass0.{k}.npy") for k in ("indptr", "indices")]
        doc = json.load(open(path))
        if not doc.get("points") or not all(p.exists() for p in csr):
            continue
        ip, ix = (np.load(p) for p in csr)
        points.append((task, doc["points"][-1], float(weights_of[ix].sum()) / (len(ip) - 1)))
        indptr.append(ip[1:] + total)
        indices.append(ix)
        total += int(ip[-1])
    if not points:
        print(f"{run}: no finished passages")
        continue
    np.save(run / "merged.pass0.indptr.npy", np.concatenate(indptr))
    np.save(run / "merged.pass0.indices.npy", np.concatenate(indices))
    np.save(run / "merged.pass0.offsets.npy", np.cumsum([0] + [library[s]["pieces"] for s in sites]).astype(np.int64))
    n = points[0][1]["observations"]

    def mean(key, part=None):
        values = [(p[part] if part else p).get(key) for _, p, _ in points]
        return float(np.mean(values)) if all(v is not None for v in values) else None

    line = {
        "run": str(run), "n": n, "passages": [t for t, _, _ in points], "claim": points[0][1].get("claim"),
        "ours": {"l0": mean("l0"), "weights": float(np.mean([w for _, _, w in points])), "kl": mean("kl"), "kl_box": mean("kl_box"), "code": mean("code_box") or mean("code")},
        "vpd": {"l0": mean("l0", "start"), "kl": mean("kl", "start"), "kl_box": mean("kl_box", "start"), "code": mean("code_box", "start") or mean("code", "start")},
        "all_on": {"l0": mean("l0", "all_on"), "kl": mean("kl", "all_on"), "code": mean("code", "all_on")},
    }
    frontier.append(line)
    print(json.dumps(line))
if frontier:
    out = Path(sys.argv[1]).parent / "frontier.json"
    json.dump(frontier, open(out, "w"), indent=1)
    print(f"wrote {out}")
