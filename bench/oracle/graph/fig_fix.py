"""Undoing the model's mistakes (#2951 graph oracle, hard split): on each question of the hard split the model is sure
of a token the text does not continue with. If an answer's mechanism is the one the model uses for that token,
removing its subcomponents should take the probability away from that token and leave the rest of the prediction as
it was. For each prefix of the answer's steps, the whole model runs with those subcomponents removed
(native.without); the control removes as many subcomponents from the same weight matrices at the same positions,
drawn in proportion to how much each writes there on the text (native.contributions), so it removes about as much
activity; VPD removes its own first as many subcomponents (vpd_steps' order: causal importance times what each
writes, the predicted position's first). Two measures against the number of subcomponents removed, per question
(thin) and the median (thick): the model's probability of its wrong token, and the KL (bits) from the model's
prediction without that token (renormalized) to the edited one without it, which is 0 when the removal takes away
only the wrong token.

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

ANSWER, CONTROL, VPD = "#2a78b5", "#7f7f7f", "#d98a1e"  # the palette fig_curves validated


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


def removals(nat, task_id: str, src: str, draws: int = 8) -> dict | None:
    """One hard question's removal test: for each prefix of the answer's steps, (the wrong token's probability, the KL
    of the rest of the prediction) with the prefix's subcomponents removed, with VPD's first as many removed, and with as
    many active subcomponents removed (the control, `draws` draws). None for a text without a wrong prediction or an
    answer without steps."""
    task = json.loads((native.TEXTS / "vpd4l" / f"{task_id}.json").read_text())
    prompt = task["prompts"][0]
    if "actual_next_id" not in prompt:
        return None
    ids, targets = prompt["token_ids"], prompt["target_positions"]
    ir = mech.trace_inline(src, "vpd4l", task)
    if not ir["valid"] or not ir["graph"]["steps"]:
        return None
    g = ir["graph"]
    nodes = [(native.site_name(layer, kind), pos, c) for layer, kind, pos, c in g["nodes"]]
    with torch.no_grad():
        p0 = nat.reference([ids], targets)[0, 0].exp()
        wrong = int(p0.argmax())

        def effect(removed):
            """(the wrong token's probability, the KL in bits of the rest) with these subcomponents removed."""
            q = nat.without(ids, targets, removed)[0, 0].exp()
            a_, b_ = p0.clone(), q.clone()
            a_[wrong], b_[wrong] = 0.0, 0.0
            a_, b_ = a_ / a_.sum(), b_ / b_.sum()
            return float(q[wrong]), float((a_ * (a_.clamp_min(1e-30).log2() - b_.clamp_min(1e-30).log2())).sum())

        weights = nat.contributions(ids, targets)
        rank = nat._vpd_rank(ids, targets)
        vpd_order = sorted(rank, key=rank.get)
        gen = torch.Generator(device="cpu").manual_seed(native.task_seed(task_id))
        rows = []
        for k in range(1, g["steps"] + 1):
            removed = sorted({nodes[i] for i, s in enumerate(g["node_step"]) if s < k})
            mine = {(n, t): set() for n, t, _ in removed}
            for n, t, c in removed:
                mine[(n, t)].add(c)
            ctrl = []
            for _ in range(draws):
                picks = []
                for (n, t), cs in mine.items():
                    w = weights[n][t].clone().cpu()
                    w[list(cs)] = 0.0
                    m = min(len(cs), int((w > 0).sum()))
                    picks += [(n, t, int(c)) for c in torch.multinomial(w, m, replacement=False, generator=gen)] if m else []
                ctrl.append(effect(picks))
            row = {"removed": len(removed)}
            for arm, (pw, rest) in (("answer", effect(removed)), ("vpd", effect(vpd_order[:len(removed)])),
                                    ("control", tuple(sum(x) / len(x) for x in zip(*ctrl)))):
                row[f"{arm}_wrong"], row[f"{arm}_rest"] = pw, rest
            rows.append(row)
    return {"task": task_id, "p_wrong": float(p0[wrong]), "steps": rows}


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
        res = removals(nat, task_id, src, a.draws)
        if res is None:
            continue
        results.append(res)
        print(f"{task_id}: p(wrong) {res['p_wrong']:.3f} -> " + ", ".join(
            f"{r['removed']}: {r['answer_wrong']:.3f}/{r['answer_rest']:.2f} (VPD {r['vpd_wrong']:.3f}/{r['vpd_rest']:.2f}, control {r['control_wrong']:.3f}/{r['control_rest']:.2f})"
            for r in res["steps"][:5]), flush=True)
    Path(a.out).with_suffix(".json").write_text(json.dumps(results))

    plot(results, a.out)


def plot(results: list, out: str) -> None:
    arms = [("answer", ANSWER, "the answer's subcomponents"), ("vpd", VPD, "VPD's first subcomponents"), ("control", CONTROL, "as many active subcomponents")]
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    grid = sorted({r["removed"] for res in results for r in res["steps"]})
    floor = 1e-3  # the KL axis is logarithmic: values below it are drawn at it
    for ax, measure, start, ylabel in ((axes[0], "wrong", lambda res: res["p_wrong"], "model's probability of its wrong token"),
                                       (axes[1], "rest", None, "KL of the rest of the prediction (bits)")):
        shown = (lambda v: v) if start else (lambda v: max(v, floor))
        for key, color, label in arms:
            for res in results:
                xs = ([1] if start else []) + [r["removed"] for r in res["steps"]]
                ys = ([start(res)] if start else []) + [shown(r[f"{key}_{measure}"]) for r in res["steps"]]
                ax.plot(xs, ys, color=color, lw=0.7, alpha=0.18)
            xs, med = [], []
            for x in grid:
                vals = []
                for res in results:  # each question's value at x: its last prefix with at most x subcomponents
                    before = [r[f"{key}_{measure}"] for r in res["steps"] if r["removed"] <= x]
                    if before or start:
                        vals.append(before[-1] if before else start(res))
                if vals:
                    xs.append(x)
                    med.append(shown(sorted(vals)[len(vals) // 2]))
            ax.plot(xs, med, color=color, lw=3, label=label)
        ax.set_xscale("log")
        from matplotlib.ticker import FuncFormatter

        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("subcomponents removed", fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.tick_params(labelsize=13)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[1].set_yscale("log")
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].legend(fontsize=14, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor="white")

    def first(res, key):  # the fewest removed that halve the wrong token's probability
        return next((r for r in res["steps"] if r[f"{key}_wrong"] < 0.5 * res["p_wrong"]), None)

    for key, _, label in arms:
        hits = [first(res, key) for res in results]
        got = [h for h in hits if h is not None]
        n = sorted(h["removed"] for h in got)
        rest = sorted(h[f"{key}_rest"] for h in got)
        print(f"{label}: halve the wrong token's probability on {len(got)}/{len(results)} questions"
              + (f", median {n[len(n) // 2]} removed, rest of the prediction moved by a median {rest[len(rest) // 2]:.2f} bits" if got else ""))


if __name__ == "__main__":
    main()
