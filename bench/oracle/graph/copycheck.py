"""A check of answers against a mechanism measured on the input alone (#2951 graph oracle), independent of the verifier:
copying. A question is a copy when the model's predicted token y follows an earlier occurrence of the current token
(ids[j] == ids[t], ids[j + 1] == y, j + 1 < t) and replacing ids[j + 1] with another token z moves the prediction to z
(the model's probability of z at t rises above its probability of y): the prediction reads position s = j + 1. A graph
that explains such a prediction must read position s: some connection from a subcomponent at s into an attention output
at the target (a complete graph: a value at s and an attention output of its layer at the target). Per method and
question: whether its whole answer reads s, and the description length (Graph.bits) of its first step that does.

  copycheck.py OUT.jsonl --split hard heldout [--answers NAME=DIR ...] [--vpd] [--questions 50]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import native  # noqa: E402


def copy_source(nat, ids: list[int], targets: list[int]) -> tuple[int, int, float] | None:
    """(source position s, the predicted token, how far the prediction follows the swap) for a copy question, or None:
    the latest earlier occurrence j of the current token whose next token is the prediction, confirmed when putting
    another token z at j + 1 makes z the more probable of z and y at the target."""
    t = max(targets)
    with torch.no_grad():
        p = nat.reference([ids], targets)[0, 0].exp()
    y = int(p.argmax())
    js = [j for j in range(t - 1) if ids[j] == ids[t] and ids[j + 1] == y and j + 1 < t]
    if not js:
        return None
    s = js[-1] + 1
    z = int(p.argsort(descending=True)[1])  # the model's second choice: a plausible token, not the copied one
    edited = list(ids)
    edited[s] = z
    with torch.no_grad():
        q = nat.reference([edited], targets)[0, 0].exp()
    if float(q[z]) <= float(q[y]):
        return None
    return s, y, float(q[z] - p[z])


def reads(g, s: int, targets: list[int]) -> bool:
    """Whether the graph connects a subcomponent at position s into an attention output at a target."""
    tset = set(targets)
    if g.complete:
        nodes = g.node_set()
        values = {native._layer_kind(nd)[0] for nd in nodes if nd[1] == s and native._layer_kind(nd)[1] == "v_proj"}
        return any(nd[1] in tset and native._layer_kind(nd)[1] == "o_proj" and native._layer_kind(nd)[0] in values for nd in nodes)
    return any(r[1] in tset and any(w[1] == s for w in ws) for r, ws in g.parents.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--split", nargs="+", default=["hard", "heldout"])
    ap.add_argument("--answers", action="append", default=[], help="NAME=DIR of <question>.py answers")
    ap.add_argument("--vpd", action="store_true", help="also VPD's ranked answer")
    ap.add_argument("--questions", type=int, default=50)
    a = ap.parse_args()
    import mech

    nat = native.Native()
    P_total = sum(nat.C.values())
    methods = dict(spec.split("=", 1) for spec in a.answers)
    rows = []
    for split in a.split:
        for p in native.tasks(split)[:a.questions]:
            task = {**json.loads(p.read_text()), "path": str(p)}
            ids, targets = native.text(p)
            found = copy_source(nat, ids, targets)
            if found is None:
                continue
            s, y, follow = found
            positions = nat.positions(targets)
            row = {"task": p.stem, "split": split, "source": s, "target": max(targets), "follow": follow, "methods": {}}
            for m, d in methods.items():
                f = Path(d) / f"{p.stem}.py"
                if not f.exists():
                    continue
                ir = mech.trace(f.read_text(), "vpd4l", behavior=task)
                if not ir["valid"]:
                    row["methods"][m] = {"valid": False}
                    continue
                first = None
                for k in range(1, ir["graph"]["steps"] + 1):
                    g, _ = native.prefix(ir, k)
                    if reads(g, s, targets):
                        first = g.bits(positions, P_total, targets)
                        break
                row["methods"][m] = {"valid": True, "reads": first is not None, "bits": first}
            if a.vpd:
                first = None
                for g in nat.vpd_steps(ids, targets):
                    if reads(g, s, targets):
                        first = g.bits(positions, P_total, targets)
                        break
                row["methods"]["vpd"] = {"valid": True, "reads": first is not None, "bits": first}
            rows.append(row)
            print(p.stem, "source", s, "->", max(targets), {m: (v.get("reads"), v.get("bits") and round(v["bits"])) for m, v in row["methods"].items()}, flush=True)
    a.out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    names = sorted({m for r in rows for m in r["methods"]})
    print(f"{len(rows)} copy questions")
    for m in names:
        rs = [r["methods"][m] for r in rows if m in r["methods"]]
        hit = sorted(v["bits"] for v in rs if v.get("reads"))
        print(f"{m:12s} reads the source in {len(hit)}/{len(rs)}" + (f", median first at {hit[len(hit) // 2]:.0f} bits" if hit else ""))


if __name__ == "__main__":
    main()
