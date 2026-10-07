"""Mechanism atlas (#2951, night plan R7): which pieces of the model the best program of each behavior
uses, and which pieces and connections recur across behaviors: a first look at a decomposition that
emerges from the explanations themselves.

  atlas.py RESULT_DIR... [--model vpd4l] [--out PREFIX]
For every behavior, the program with the lowest total among the result files in RESULT_DIRs (search.py's
<behavior>.<mode><tag>.json, a sweep file's programs, or {"behavior", "source", "score"} files), traced
by mech. Writes PREFIX.tsv (behaviors x pieces: 1 for a head, the fraction of a layer's neurons or
subcomponents used for an MLP), PREFIX.recurring.tsv (pieces and head-to-head connections used by two or
more behaviors) and PREFIX.png (the matrix as a figure).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import mech  # noqa: E402

RUNS = Path.home() / "mpd-data/graph_oracle/runs"
FIGURES = Path.home() / "mpd-data/figures/graph_oracle"


def candidates(directories: list[Path]):
    """(behavior, total bits per token, source) for every program in the result files."""
    for d in directories:
        for path in sorted(d.glob("*.json")):
            r = json.loads(path.read_text())
            if "source" in r and ("heldout" in r or "score" in r):  # search.py's result, or an oracle file
                s = r.get("score") or r["heldout"]
                behavior = r.get("behavior") or ".".join(path.stem.split(".")[:2])
                yield behavior, s["total_bits"] / s.get("N", 2**24), r["source"]
            elif "programs" in r:  # a sweep file: its programs' sources are the reference programs, not stored
                continue


def best_programs(directories: list[Path]) -> dict[str, tuple[float, str]]:
    best: dict[str, tuple[float, str]] = {}
    for behavior, total, source in candidates(directories):
        if behavior not in best or total < best[behavior][0]:
            best[behavior] = (total, source)
    return best


def pieces_of(ir: dict, shapes: dict) -> dict[str, float]:
    """Piece name -> weight: 1 per head; per layer and view, the fraction of its MLP units used."""
    out: dict[str, float] = defaultdict(float)
    for n in ir["nodes"]:
        for p in n["pieces"]:
            idx = p["index"]
            count = None if idx is None else (1 if isinstance(idx, int) else len(idx))
            if p["kind"] == "head":
                for h in ([idx] if isinstance(idx, int) else idx if idx is not None else range(shapes["heads"])):
                    out[f"L{p['layer']}.H{h}"] = 1.0
            elif p["view"] == "native":
                out[f"L{p['layer']}.mlp"] += (count if count is not None else shapes["d_mlp"]) / shapes["d_mlp"]
            else:
                size = shapes["views"][p["view"]][p["layer"]][p["kind"]] if p["view"] == "vpd" else 1
                out[f"L{p['layer']}.{p['view']}.{p['kind']}"] += (count if count is not None else size) / size
    return dict(out)


def head_edges(ir: dict) -> set[tuple[str, str]]:
    """Head-to-head connections the program declares, as piece names (one per head pair)."""
    heads = {}
    for n in ir["nodes"]:
        hs = [f"L{p['layer']}.H{h}" for p in n["pieces"] if p["kind"] == "head"
              for h in ([p["index"]] if isinstance(p["index"], int) else p["index"] or [])]
        heads[n["id"]] = hs
    return {(a, b) for e in ir["edges"] for a in heads.get(e["from"], []) for b in heads.get(e["to"], [])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results", type=Path, nargs="+")
    ap.add_argument("--model", default="vpd4l")
    ap.add_argument("--out", type=Path, default=RUNS / "atlas_vpd4l")
    a = ap.parse_args()
    shapes = mech.shapes(a.model)
    best = best_programs([d.expanduser() for d in a.results])
    rows, edges = {}, {}
    for behavior, (total, source) in sorted(best.items()):
        ir = mech.trace_inline(source, a.model)
        if ir["valid"]:
            rows[behavior] = pieces_of(ir, shapes)
            edges[behavior] = head_edges(ir)
    columns = [f"L{l}.H{h}" for l in range(shapes["layers"]) for h in range(shapes["heads"])]
    columns += sorted({c for r in rows.values() for c in r if c not in columns})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{a.out}.tsv", "w") as f:
        f.write("behavior\ttotal_bits_per_token\t" + "\t".join(columns) + "\n")
        for b, r in rows.items():
            f.write(f"{b}\t{best[b][0]:.4f}\t" + "\t".join(f"{r.get(c, 0.0):.3g}" for c in columns) + "\n")
    use = Counter(c for r in rows.values() for c in r)
    pairs = Counter(e for es in edges.values() for e in es)
    with open(f"{a.out}.recurring.tsv", "w") as f:
        f.write("kind\tpiece\tbehaviors\tnames\n")
        for c, k in use.most_common():
            if k >= 2:
                f.write(f"piece\t{c}\t{k}\t{','.join(b for b, r in rows.items() if c in r)}\n")
        for (x, y), k in pairs.most_common():
            if k >= 2:
                f.write(f"connection\t{x} >> {y}\t{k}\t{','.join(b for b, es in edges.items() if (x, y) in es)}\n")
    figure(rows, columns, Path(f"{a.out}.png") if a.out.parent != RUNS else FIGURES / f"{a.out.name}.png")
    print(f"{len(rows)} behaviors; {sum(1 for k in use.values() if k >= 2)} pieces and {sum(1 for k in pairs.values() if k >= 2)} "
          f"connections recur; {a.out}.tsv")


def figure(rows: dict, columns: list[str], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    used = [c for c in columns if any(c in r for r in rows.values())]
    if not rows or not used:
        return
    m = np.array([[rows[b].get(c, 0.0) for c in used] for b in rows])
    plt.rcParams.update({"font.size": 18, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(1.0 + 0.45 * len(used) + 4, 0.45 * len(rows) + 3))
    ax.imshow(m, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(used)), used, rotation=90, fontsize=14)
    ax.set_yticks(range(len(rows)), list(rows), fontsize=14)
    ax.set_title("Pieces each behavior's best program uses", loc="left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
