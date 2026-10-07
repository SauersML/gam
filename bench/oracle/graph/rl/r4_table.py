"""R4's table and figure (#2951): does the oracle write good programs for held-out behaviors?

  r4_table.py NAME=RUN_DIR|NAME=SUMMARY.jsonl@SET ... [--set heldout_behaviors] [--out PNG] [--tsv TSV]

Per oracle run (its latest evaluation in RUN_DIR/eval.jsonl, or rescore_<tag>_summary.jsonl given as the
path): the valid share of sampled programs, the share below the empty program's S, the signal its best
program recovers (1 - exec_error / the empty program's exec_error, on the same experiments) and its best
S; beside them the baselines scored on the same experiments (empty, full, search). Cost per behavior: the
oracle's samples (generations, no checker call at answer time) against search's checker calls (from
runs/search/<behavior>.<mode>.json). The figure: signal recovered per held-out behavior, one bar group per
behavior (oracle runs and search), white background, direct labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEARCH = Path.home() / "mpd-data/graph_oracle/runs/search"


def rows_of(path: Path, which: str) -> list[dict]:
    """The per-behavior rows of the latest evaluation of set `which` in an eval.jsonl or rescore summary."""
    lines = [json.loads(x) for x in open(path)]
    last = max((r["step"] for r in lines if r.get("set") == which), default=None)
    return [r for r in lines if r.get("set") == which and r["step"] == last]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=RUN_DIR or NAME=summary.jsonl")
    ap.add_argument("--set", default="heldout_behaviors")
    ap.add_argument("--out", default=str(Path.home() / "mpd-data/figures/graph_oracle/r4_heldout_recovered.png"))
    ap.add_argument("--tsv", default=str(Path.home() / "mpd-data/graph_oracle/runs/r4/r4_table.tsv"))
    a = ap.parse_args()
    runs = {}
    for spec in a.runs:
        name, path = spec.split("=", 1)
        path, _, which = path.partition("@")  # NAME=summary.jsonl@SET for a rescore summary's per-run set
        p = Path(path).expanduser()
        runs[name] = rows_of(p / "eval.jsonl" if p.is_dir() else p, which or a.set)
    behaviors = sorted({r["behavior"] for rs in runs.values() for r in rs})
    table = ["method\tbehavior\tvalid_share\tbelow_empty_share\tbest_recovered\tbest_bits\tcost_per_behavior"]
    recovered = {}
    for name, rs in runs.items():
        for r in rs:
            table.append(f"{name}\t{r['behavior']}\t{r['valid_fraction']:.3f}\t{r['below_empty_fraction']}\t{r['best_recovered']}\t{r['best_bits']:.6g}\tsamples")
            recovered.setdefault(name, {})[r["behavior"]] = r["best_recovered"]
            for base, bits in r["baselines"].items():
                calls = None
                if base.startswith("search_"):
                    f = SEARCH / f"{r['behavior']}.{base[len('search_'):]}.json"
                    calls = json.loads(f.read_text()).get("checker_calls") if f.exists() else None
                key = f"{base}"
                if r["behavior"] not in recovered.get(key, {}):
                    table.append(f"{base}\t{r['behavior']}\t\t\t{r['baselines_recovered'].get(base)}\t{bits:.6g}\t{calls if calls is not None else ''}")
                    recovered.setdefault(key, {})[r["behavior"]] = r["baselines_recovered"].get(base)
    Path(a.tsv).parent.mkdir(parents=True, exist_ok=True)
    Path(a.tsv).write_text("\n".join(table) + "\n")
    methods = [m for m in recovered if m != "empty"]
    plt.rcParams.update({"font.size": 15, "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(max(10, 1.1 * len(behaviors) * max(1, len(methods)) / 2), 6))
    width = 0.8 / max(1, len(methods))
    colors = ["#1f5fa8", "#c2410c", "#15803d", "#6b7280", "#7c3aed", "#b45309"]
    for k, m in enumerate(methods):
        xs = [i + k * width for i in range(len(behaviors))]
        ys = [recovered[m].get(b) if recovered[m].get(b) is not None else 0.0 for b in behaviors]
        ax.bar(xs, ys, width=width, color=colors[k % len(colors)])
        if ys:
            ax.text(xs[-1] + width / 2, ys[-1], f" {m}", color=colors[k % len(colors)], va="bottom", rotation=90, fontsize=12)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(behaviors))], behaviors, rotation=35, ha="right", fontsize=11)
    ax.set_ylabel("signal recovered by the best program")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    means = {m: (sum(v for v in recovered[m].values() if v is not None) / max(1, sum(1 for v in recovered[m].values() if v is not None))) for m in recovered}
    print(json.dumps({"behaviors": len(behaviors), "mean_recovered": means, "tsv": a.tsv, "figure": a.out}))


if __name__ == "__main__":
    main()
