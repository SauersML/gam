"""Tables and a figure of prediction-SFT runs (#2951): per held-out set and question type, the answer's
bits per question for the base and the trained oracle (sft.py's eval.json), the learning curve (the
held-out sets' mean bits per question every --eval-every steps, train.jsonl), and, when eval_kl.py has
run, KL(M_e || answer) in bits beside the no-change answer's on the same questions.

  report.py --run NAME=DIR [--run NAME=DIR ...] [--figure PNG]
DIR is an sft.py output directory (eval.json, train.jsonl, eval_kl_base.json, eval_kl_trained.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(run: Path):
    e = run / "eval.json" if (run / "eval.json").exists() else run / "eval_only.json"  # sft.py --eval-only writes the latter
    out = {"eval": json.loads(e.read_text()) if e.exists() else None, "curve": [], "kl": {}}
    if (run / "train.jsonl").exists():
        for line in open(run / "train.jsonl"):
            r = json.loads(line)
            if "heldout" in r:
                out["curve"].append(r)
    for which in ("base", "trained"):
        p = run / f"eval_kl_{which}.json"
        if p.exists():
            d = json.loads(p.read_text())
            out["kl"][which] = d.get("sets") or {"heldout": d.get("per_type", {})}
    return out


def sets_of(e):
    """eval.json from a single-set run ({type: ...}) or a multi-set run ({set: {type: ...}})."""
    first = next(iter(e.values()))
    return e if "bits_per_question" not in first else {"heldout": e}


def table(name, r):
    lines = [f"== {name}"]
    if r["eval"]:
        base, trained = sets_of(r["eval"]["base"]), sets_of(r["eval"]["trained"])
        if "steps" in r["eval"]:
            lines.append(f"steps {r['eval']['steps']}, hours {r['eval']['hours']:.2f}")
        else:
            lines.append(f"adapters {r['eval'].get('adapters')}")
        for s in trained:
            lines.append(f"  [{s}] answer bits per question, base -> trained (se)")
            for k in sorted(trained[s]):
                b, t = base[s][k], trained[s][k]
                lines.append(f"    {k:9s} {b['bits_per_question']:7.1f} -> {t['bits_per_question']:6.1f} ({t['se']:.1f})")
    for which, sets in r["kl"].items():
        for s, types in sets.items():
            lines.append(f"  [{s}] eval_kl {which}: KL(M_e || answer) bits, no-change answer, difference (se)")
            for k, v in sorted(types.items()):
                extra = f" vs {v['no_change_kl_bits']:.3f}, diff {v['difference_bits']:+.3f} ({v['difference_se']:.3f})" if "no_change_kl_bits" in v else ""
                lines.append(f"    {k:9s} {v['kl_bits']:.3f} ({v['se']:.3f}){extra}")
    return "\n".join(lines)


def figure(runs, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 15, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    for name, r in runs.items():
        if not r["curve"]:
            continue
        for s in r["curve"][0]["heldout"]:
            steps = [c["step"] for c in r["curve"]]
            mean = [sum(v["bits_per_question"] for v in c["heldout"][s].values()) / len(c["heldout"][s]) for c in r["curve"]]
            line, = ax.plot(steps, mean, marker="o")
            ax.annotate(f"{name}, {s}", (steps[-1], mean[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=line.get_color())
    ax.set_xlabel("training step")
    ax.set_ylabel("held-out answer bits per question\n(mean over question types)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")


BINS = ((0.0, 0.01), (0.01, 0.1), (0.1, 1.0), (1.0, float("inf")))


def strata(run: Path):
    """eval_kl's per-question records (eval_kl_trained.questions.jsonl) split by the measured change
    KL(M || M_e): per set, type and bin, the trained oracle's mean KL, the no-change answer's, and the
    paired difference with its standard error."""
    p = run / "eval_kl_trained.questions.jsonl"
    if not p.exists():
        return ""
    groups = {}
    for line in open(p):
        r = json.loads(line)
        if r.get("measured_kl_bits") is None or r.get("no_change_kl_bits") is None:
            continue
        b = next(i for i, (lo, hi) in enumerate(BINS) if lo <= max(r["measured_kl_bits"], 0.0) < hi)
        groups.setdefault((r["set"], r["type"], b), []).append(r["oracle_kl_bits"] - r["no_change_kl_bits"])
    lines = ["  eval_kl by measured change (bits): trained - no change, mean (se) [n]"]
    for (s, k, b), d in sorted(groups.items()):
        m = sum(d) / len(d)
        se = (sum((x - m) ** 2 for x in d) / max(1, len(d) - 1) / len(d)) ** 0.5
        lo, hi = BINS[b]
        lines.append(f"    [{s}] {k:7s} {lo:g}-{hi:g}: {m:+.3f} ({se:.3f}) [{len(d)}]")
    return "\n".join(lines)


def strata_figure(run: Path, path: str):
    """Per held-out set (panels): trained minus no-change KL (bits, mean and standard error) for each question
    type against the size of the measured change, from eval_kl's per-question records."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = {}
    for line in open(run / "eval_kl_trained.questions.jsonl"):
        r = json.loads(line)
        if r.get("measured_kl_bits") is None or r.get("no_change_kl_bits") is None:
            continue
        b = next(i for i, (lo, hi) in enumerate(BINS) if lo <= max(r["measured_kl_bits"], 0.0) < hi)
        groups.setdefault(r["set"], {}).setdefault(r["type"], {}).setdefault(b, []).append(r["oracle_kl_bits"] - r["no_change_kl_bits"])
    plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False})
    sets = sorted(groups)
    fig, axes = plt.subplots(1, len(sets), figsize=(5.5 * len(sets), 5), facecolor="white", squeeze=False, sharey=True)
    names = ["< 0.01", "0.01-0.1", "0.1-1", "> 1"]
    all_types = sorted({k for s in sets for k in groups[s]})
    color = {k: f"C{i}" for i, k in enumerate(all_types)}  # one color per type in every panel
    for ax, s in zip(axes[0], sets):
        types = sorted(groups[s])
        for j, k in enumerate(types):
            xs, ms, ses = [], [], []
            for b in range(len(BINS)):
                d = groups[s][k].get(b, [])
                if len(d) >= 3:
                    m = sum(d) / len(d)
                    xs.append(b + (j - len(types) / 2) * 0.08)
                    ms.append(m)
                    ses.append((sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d)) ** 0.5)
            ax.errorbar(xs, ms, yerr=ses, marker="o", capsize=3, label=k, color=color[k])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_xticks(range(len(BINS)))
        ax.set_xticklabels(names)
        ax.set_xlabel("measured change KL(M || M_e), bits")
        ax.set_title(s)
    axes[0][0].set_ylabel("oracle minus no-change KL, bits\n(below 0 = oracle better)")
    from matplotlib.lines import Line2D

    fig.legend([Line2D([], [], color=color[k], marker="o") for k in all_types], all_types, frameon=False, ncol=len(all_types), loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150, facecolor="white")


def kl_figure(name, r, path):
    """Per held-out set (panels) and question type: KL(M_e || answer) in bits for the base oracle, the trained
    oracle and the no-change answer on the same questions (log scale)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False})
    base, trained = r["kl"].get("base", {}), r["kl"].get("trained", {})
    sets = [s for s in trained if trained[s]]
    fig, axes = plt.subplots(1, len(sets), figsize=(5.5 * len(sets), 5), facecolor="white", squeeze=False)
    for ax, s in zip(axes[0], sets):
        types = sorted(trained[s])
        x = range(len(types))
        rows = [("base oracle", [base.get(s, {}).get(k, {}).get("kl_bits", float("nan")) for k in types], "#9aa5b1"),
                ("no change", [trained[s][k].get("no_change_kl_bits", float("nan")) for k in types], "#e0a458"),
                ("trained oracle", [trained[s][k]["kl_bits"] for k in types], "#2b6cb0")]
        for j, (label, vals, color) in enumerate(rows):
            ax.bar([i + (j - 1) * 0.27 for i in x], vals, width=0.27, color=color, label=label)
        ax.set_yscale("log")
        ax.set_xticks(list(x))
        ax.set_xticklabels(types, rotation=30)
        n = min(trained[s][k]["questions"] for k in types)
        ax.set_title(f"{s} (n >= {n} per type)")
        ax.set_ylabel("KL(M_e || answer), bits")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150, facecolor="white")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--figure", default="")
    ap.add_argument("--strata-figure", default="", help="eval_kl by size of the measured change for the first run")
    ap.add_argument("--kl-figure", default="", help="eval_kl bars of the first run (base, trained, no change)")
    args = ap.parse_args()
    runs = {}
    for spec in args.run:
        name, _, d = spec.rpartition("=")
        runs[name or Path(d).name] = dict(read(Path(d)), dir=Path(d))
    for name, r in runs.items():
        print(table(name, r))
        st = strata(r["dir"])
        if st:
            print(st)
    if args.strata_figure:
        strata_figure(next(iter(runs.values()))["dir"], args.strata_figure)
        print("figure", args.strata_figure)
    if args.kl_figure:
        name, r = next(iter(runs.items()))
        kl_figure(name, r, args.kl_figure)
        print("figure", args.kl_figure)
    if args.figure:
        figure(runs, args.figure)
        print("figure", args.figure)


if __name__ == "__main__":
    main()
