"""Undoing the model's mistakes (#2951 graph oracle, hard split): on each question of the hard split the model is sure
of a token the text does not continue with. If an answer's mechanism is the one the model uses, removing its
subcomponents should take the probability away from that token. For each prefix of the answer's steps, the whole
model runs with those subcomponents removed (native.without); the control removes as many subcomponents from the same
weight matrices at the same positions, drawn in proportion to how much each writes there on the text
(native.contributions), so it removes about as much activity. The figure: the model's probability of its wrong
token against the number of subcomponents removed, per question (thin) and the median (thick).

  fig_fix.py OUT.png [--answers texts/search_hard | --samples EVAL_SAMPLES.jsonl --program prompted] [--draws 8]
  fig_fix.py OUT.png --replot      (the figure again from OUT.json)
Writes OUT.json with every question's numbers.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mech  # noqa: E402
import native  # noqa: E402

ANSWER, CONTROL = "#2a78b5", "#7f7f7f"


def sources(a) -> dict:
    if a.samples:
        import score

        out = {}
        for r in map(json.loads, open(a.samples)):
            if r["program"].startswith(a.program) and r["score"].get("valid", True) and r["score"].get("curve"):
                if r["behavior"] not in out or score.key(r["score"]) < score.key(out[r["behavior"]][1]):
                    out[r["behavior"]] = (r["source"], r["score"])
        return {k: v[0] for k, v in out.items()}
    return {p.stem: p.read_text() for p in sorted(Path(a.answers).glob("*.py"))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--answers", default=str(native.TEXTS / "search_hard"))
    ap.add_argument("--samples")
    ap.add_argument("--program", default="prompted")
    ap.add_argument("--draws", type=int, default=8, help="control draws per prefix")
    ap.add_argument("--replot", action="store_true")
    a = ap.parse_args()
    if a.replot:
        plot(json.loads(Path(a.out).with_suffix(".json").read_text()), a.out)
        return
    nat = native.Native()
    results = []
    for task_id, src in sources(a).items():
        task = json.loads((native.TEXTS / "vpd4l" / f"{task_id}.json").read_text())
        prompt = task["prompts"][0]
        if "actual_next_id" not in prompt:
            continue
        ids, targets = prompt["token_ids"], prompt["target_positions"]
        ir = mech.trace_inline(src, "vpd4l", task)
        if not ir["valid"] or not ir["graph"]["steps"]:
            continue
        g = ir["graph"]
        nodes = [(native.site_name(layer, kind), pos, c) for layer, kind, pos, c in g["nodes"]]
        with torch.no_grad():
            p0 = nat.reference([ids], targets)[0, 0].exp()
            wrong, actual = int(p0.argmax()), prompt["actual_next_id"]
            weights = nat.contributions(ids, targets)
            gen = torch.Generator(device="cpu").manual_seed(native.task_seed(task_id))
            rows = []
            for k in range(1, g["steps"] + 1):
                removed = sorted({nodes[i] for i, s in enumerate(g["node_step"]) if s < k})
                p = nat.without(ids, targets, removed)[0, 0].exp()
                mine = {(n, t) : set() for n, t, _ in removed}
                for n, t, c in removed:
                    mine[(n, t)].add(c)
                ctrl_wrong, ctrl_actual = [], []
                for _ in range(a.draws):
                    picks = []
                    for (n, t), cs in mine.items():
                        w = weights[n][t].clone().cpu()
                        w[list(cs)] = 0.0
                        m = min(len(cs), int((w > 0).sum()))
                        picks += [(n, t, int(c)) for c in torch.multinomial(w, m, replacement=False, generator=gen)] if m else []
                    q = nat.without(ids, targets, picks)[0, 0].exp()
                    ctrl_wrong.append(float(q[wrong]))
                    ctrl_actual.append(float(q[actual]))
                rows.append({"removed": len(removed), "wrong": float(p[wrong]), "actual": float(p[actual]), "flips": int(p.argmax()) != wrong,
                             "control_wrong": sum(ctrl_wrong) / len(ctrl_wrong), "control_actual": sum(ctrl_actual) / len(ctrl_actual)})
        results.append({"task": task_id, "p_wrong": float(p0[wrong]), "p_actual": float(p0[actual]), "steps": rows})
        print(f"{task_id}: p(wrong) {float(p0[wrong]):.3f} -> " + ", ".join(f"{r['removed']}: {r['wrong']:.3f} (control {r['control_wrong']:.3f})" for r in rows[:6]), flush=True)
    Path(a.out).with_suffix(".json").write_text(json.dumps(results))

    plot(results, a.out)


def plot(results: list, out: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for res in results:
        xs = [1] + [r["removed"] for r in res["steps"]]
        ax.plot(xs, [res["p_wrong"]] + [r["wrong"] for r in res["steps"]], color=ANSWER, lw=0.8, alpha=0.25)
        ax.plot(xs, [res["p_wrong"]] + [r["control_wrong"] for r in res["steps"]], color=CONTROL, lw=0.8, alpha=0.25)
    grid = sorted({r["removed"] for res in results for r in res["steps"]})
    for key, color, label in (("wrong", ANSWER, "the answer's subcomponents"), ("control_wrong", CONTROL, "as many active subcomponents")):
        med = []
        for x in grid:
            vals = []
            for res in results:  # each question's value at x: its last prefix with at most x subcomponents
                before = [r[key] for r in res["steps"] if r["removed"] <= x]
                vals.append(before[-1] if before else res["p_wrong"])
            med.append(sorted(vals)[len(vals) // 2])
        ax.plot(grid, med, color=color, lw=3)
        ax.annotate(label, (grid[-1], med[-1]), xytext=(-6, 10), textcoords="offset points", ha="right", fontsize=15, color=color)
    ax.set_xscale("log")
    ax.set_xlabel("subcomponents removed", fontsize=16)
    ax.set_ylabel("model's probability of its wrong token", fontsize=16)
    ax.tick_params(labelsize=13)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor="white")
    halves = lambda res, key: next((r["removed"] for r in res["steps"] if r[key] < 0.5 * res["p_wrong"]), None)  # noqa: E731
    a_half = [halves(res, "wrong") for res in results]
    c_half = [halves(res, "control_wrong") for res in results]
    found = sorted(x for x in a_half if x is not None)
    print(f"{out}: {len(results)} questions; the answer's subcomponents halve the wrong token's probability on {len(found)} "
          f"(median {found[len(found) // 2] if found else None} removed), the control on {sum(x is not None for x in c_half)}")


if __name__ == "__main__":
    main()
