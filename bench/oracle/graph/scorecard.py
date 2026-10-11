"""The scorecard (#2951 graph oracle): every method's answers to a split's questions, measured on the whole graph each
answer states, never one number. Per answer and question:
  area        score.key: mean KL over log-uniform description length (the graph's: subcomponents and connections)
  kl, size    the whole answer: KL in bits of the prediction from its graph alone, its nodes, its connections, its
              description length (completeness at a cost)
  necessity   KL in bits of the prediction with the answer's subcomponents removed from the model (how much the
              prediction depends on them: what an edit relies on)
  reads       connections the answer states from other positions (attention reading values elsewhere: structure
              beyond the predicted position)
  mistake     (hard split) the fewest of the answer's subcomponents, step by step, whose removal halves the model's
              probability of its wrong token, and the KL of the rest of its prediction there (fig_fix.removals): a
              targeted fix moves little else
VPD's ranked answer ("vpd") is scored the same way. Writes OUT.jsonl (one line per question and method) and prints the
medians.

  scorecard.py OUT.jsonl --split hard --answers NAME=DIR [--answers ...] [--questions 50] [--seed 1000003] [--draws 4]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import native  # noqa: E402
import score  # noqa: E402


def reads(src: str, task: dict) -> int:
    """Connections the answer states from values at other positions into attention outputs."""
    import mech

    ir = mech.trace(src, "vpd4l", behavior=task)
    if not ir["valid"]:
        return 0
    g = ir["graph"]
    nodes = g["nodes"]
    return sum(1 for e in g["parents"] if nodes[e[0]][2] != nodes[e[1]][2])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--split", choices=("heldout", "hard"), required=True)
    ap.add_argument("--answers", action="append", default=[], help="NAME=DIR of <question>.py answers")
    ap.add_argument("--questions", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1_000_003)
    ap.add_argument("--draws", type=int, default=4)
    a = ap.parse_args()
    import fig_fix

    methods = dict(spec.split("=", 1) for spec in a.answers)
    done = {(r["task"], r["method"]) for r in map(json.loads, open(a.out))} if a.out.exists() else set()
    sc = score.Scorer()
    with open(a.out, "a") as out:
        for p in native.tasks(a.split)[:a.questions]:
            tid = p.stem
            task = {**json.loads(p.read_text()), "path": str(p)}
            names = [m for m, d in methods.items() if (Path(d) / f"{tid}.py").exists() and (tid, m) not in done]
            if (tid, "vpd") not in done:
                names.append("vpd")
            if not names:
                continue
            srcs = ["vpd" if m == "vpd" else (Path(methods[m]) / f"{tid}.py").read_text() for m in names]
            scores = sc.score(task, srcs, a.seed, necessity=True)
            for m, src, s in zip(names, srcs, scores):
                row = {"task": tid, "method": m, "valid": s.get("valid"), "area": score.key(s)[1] if s.get("valid") else None,
                       "kl": s.get("kl_bits"), "bits": s.get("bits"), "nodes": s.get("nodes"), "edges": s.get("edges"),
                       "necessity": s.get("necessity_kl_bits"), "reads": reads(src, task) if m != "vpd" else None}
                if a.split == "hard" and m != "vpd" and s.get("valid"):
                    with torch.no_grad():
                        res = fig_fix.removals(sc.nat, tid, src, a.draws)
                    if res:
                        hit = next((r for r in res["steps"] if r["answer_wrong"] < 0.5 * res["p_wrong"]), None)
                        vhit = next((r for r in res["steps"] if r["vpd_wrong"] < 0.5 * res["p_wrong"]), None)
                        row["mistake"] = {"removed": hit and hit["removed"], "rest": hit and hit["answer_rest"],
                                          "vpd_removed": vhit and vhit["removed"], "vpd_rest": vhit and vhit["vpd_rest"]}
                out.write(json.dumps(row) + "\n")
                out.flush()
            print(tid, " ".join(f"{m}:{s.get('valid') and round(score.key(s)[1], 2)}" for m, s in zip(names, scores)), flush=True)
    rows = [json.loads(x) for x in open(a.out)]
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None  # noqa: E731
    for m in sorted({r["method"] for r in rows}):
        rs = [r for r in rows if r["method"] == m and r["valid"]]
        print(f"{m:14s} n={len(rs):3d} area {med([r['area'] for r in rs])} kl {med([r['kl'] for r in rs])} "
              f"nodes {med([r['nodes'] for r in rs])} edges {med([r['edges'] for r in rs if r['edges'] is not None])} "
              f"necessity {med([r['necessity'] for r in rs if r['necessity'] is not None])} reads {med([r['reads'] for r in rs if r['reads'] is not None])}")


if __name__ == "__main__":
    main()
