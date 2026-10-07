"""Hand-built candidate programs for Qwen3-0.6B mechanisms, scored by the checker (MPD #2951 graph oracle, R6).

Each program declares a chain of components found by patching (set_patch.py, qwen_patch.py, qwen_tc_patch.py) with
every causal edge among them; MLPs enter as their top transcoder features (PD.tc, about 2,000 numbers per feature,
against about 9.4M for a whole native MLP), so the weight price follows what the program uses. Scored on the
Metal/CUDA device (float32) with the transcoder view, on a fit seed and a held-out seed; every result goes to
OUT/programs_r6.jsonl with its source.

    programs_r6.py [--behaviors ioi.argument,sva.simple,...] [--experiments 4] [--root ~/mpd-data/graph_oracle/behaviors_r6]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import score  # noqa: E402

TRANSCODERS = Path.home() / "mpd-data/transcoders/qwen3-0.6b-lowl0"
TC = Path.home() / "mpd-data/graph_oracle/experiments/examples/tc_qwen3"


def head(l, h):
    return (f"h{l}_{h}", f"L[{l}].head[{h}]", 2 * l)


def features(l, ids):
    return (f"f{l}", f"PD.tc[{l}][{', '.join(map(str, ids))}]", 2 * l + 1)


def top_features(behavior: str, layer: int, k: int) -> list[int]:
    """The k features of a layer with the largest measured single-feature recovery (qwen_tc_patch.py)."""
    d = json.loads((TC / f"tc_{behavior}.json").read_text())
    fs = sorted(d["layers"][str(layer)], key=lambda f: -f["recovery_bits"])
    return [f["feature"] for f in fs[:k]]


def source(nodes, doc: str) -> str:
    nodes = sorted(nodes, key=lambda n: (n[2], n[0]))
    lines = [f'"""{doc}"""', "from mech import node, edges, L, PD, embed, logits"]
    lines += [f"{n} = node({p})" for n, p, _ in nodes]
    wires = []
    for i, (n, _, s) in enumerate(nodes):
        wires += [f"    {w} >> {n}," for w in ["embed"] + [m for m, _, t in nodes[:i] if t < s]]
    wires += [f"    {w} >> logits," for w in ["embed"] + [n for n, _, _ in nodes]]
    if nodes:
        lines += ["edges("] + wires + [")"]
    return "\n".join(lines) + "\n"


def programs() -> dict[str, list[tuple[str, list]]]:
    gt = "greater_than.war"
    return {
        "ioi.argument": [
            ("empty", []),
            ("heads6", [head(19, 2), head(21, 0), head(23, 6), head(11, 9), head(11, 13), head(21, 11)]),
            ("heads6+tc16", [head(19, 2), head(21, 0), head(23, 6), head(11, 9), head(11, 13), head(21, 11),
                             features(16, top_features("ioi.argument", 16, 8))]),
            ("heads10", [head(19, 2), head(21, 0), head(23, 6), head(11, 9), head(11, 13), head(21, 11), head(13, 6),
                         head(17, 0), head(22, 8), head(27, 15)]),
        ],
        "sva.simple": [
            ("empty", []),
            ("L0.H3", [head(0, 3)]),
            ("path3", [head(0, 3), head(24, 1), head(25, 0)]),
            ("path5", [head(0, 3), head(24, 1), head(25, 0), head(19, 12), head(11, 9)]),
        ],
        "sva.nounpp": [
            ("empty", []),
            ("path3", [head(0, 3), head(24, 1), head(25, 0)]),
            ("path5", [head(0, 3), head(24, 1), head(25, 0), head(19, 12), head(20, 9)]),
        ],
        gt: [
            ("empty", []),
            ("tc8", [head(0, 7), features(18, top_features(gt, 18, 8)), features(20, top_features(gt, 20, 8)),
                     features(21, top_features(gt, 21, 8))]),
            ("tc8+heads", [head(0, 7), head(0, 3), head(11, 9), head(20, 14), head(21, 9), head(20, 3),
                           features(18, top_features(gt, 18, 8)), features(20, top_features(gt, 20, 8)),
                           features(21, top_features(gt, 21, 8))]),
            ("tc16+heads", [head(0, 7), head(0, 3), head(11, 9), head(20, 14), head(21, 9), head(20, 3),
                            features(18, top_features(gt, 18, 16)), features(20, top_features(gt, 20, 16)),
                            features(21, top_features(gt, 21, 16)), features(22, top_features(gt, 22, 8))]),
        ],
        "induction_random.words8": [
            ("empty", []),
            ("prev+induction", [head(1, 3), head(2, 12), head(3, 10), head(15, 3), head(16, 14), head(20, 0), head(21, 8)]),
            ("prev+induction+L11.H9", [head(1, 3), head(2, 12), head(3, 10), head(15, 3), head(16, 14), head(20, 0), head(21, 8),
                                       head(11, 9), head(0, 12)]),
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behaviors", default="")
    ap.add_argument("--experiments", type=int, default=4)
    ap.add_argument("--seeds", default="0,1", help="fit seed, held-out seed")
    ap.add_argument("--root", type=Path, default=Path.home() / "mpd-data/graph_oracle/behaviors_r6")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--memory-gib", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path.home() / "mpd-data/graph_oracle/runs/r6")
    ap.add_argument("--heads-only", action="store_true", help="skip programs with transcoder features and do not load the view")
    a = ap.parse_args()
    out = a.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    progs = programs()
    if a.heads_only:
        progs = {b: [(n, nodes) for n, nodes in ps if not any(x[1].startswith("PD.tc") for x in nodes)] for b, ps in progs.items()}
    names = [b for b in progs if not a.behaviors or b in a.behaviors.split(",")]
    views = None if a.heads_only else {"transcoders": TRANSCODERS}
    with score.Checker("qwen3-0.6b", memory_gib=a.memory_gib, views=views, device=a.device or None) as c:
        for b in names:
            path = a.root.expanduser() / "qwen3-0.6b" / f"{b}.json"
            c.behavior(path)
            srcs = [source(nodes, f"R6 candidate {name} for {b}") for name, nodes in progs[b]]
            for seed in map(int, a.seeds.split(",")):
                t0 = time.time()
                rs = c.score_batch(srcs, experiments=a.experiments, seed=seed, reader=False)
                dt = time.time() - t0
                for (name, _), src, r in zip(progs[b], srcs, rs):
                    per = {k: (r[k] / r["N"] if isinstance(r.get(k), (int, float)) and k.endswith("_bits") else r.get(k))
                           for k in ("total_bits", "exec_error_bits", "code_bits", "opaque_bits", "opaque_numbers", "valid", "error")}
                    row = {"behavior": b, "program": name, "seed": seed, "experiments": a.experiments, "prompts_file": str(path),
                           "per_token": per, "raw": {k: r.get(k) for k in ("total_bits", "exec_error_bits", "code_bits", "opaque_bits", "N")},
                           "seconds_batch": round(dt, 1), "source": src}
                    with (out / "programs_r6.jsonl").open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(f"{b:24s} seed {seed} {name:22s} total {per['total_bits']:.4f}  exec {per['exec_error_bits']:.4f}  "
                          f"opaque {per['opaque_bits']:.4f} ({per['opaque_numbers']} numbers)  code {per['code_bits']:.4f}  valid {per['valid']}", flush=True)
                print(f"  ({len(srcs)} programs in {dt:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
