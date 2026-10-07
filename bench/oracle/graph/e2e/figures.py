"""Figures of the graph oracle's score (#2951), under ~/mpd-data/figures/graph_oracle/.

  figures.py terms RESULTS.json BEHAVIOR_ID [--names a=label ...] [--out PNG]
      one behavior: each program's total in bits as stacked horizontal bars (execution error, opaque
      numbers, code, reader error). RESULTS.json maps program name -> score dict (run.py --json,
      a sweep file's "programs").
  figures.py modes RESULTS.json --names a=label ...
      each program's total under each stand-in form (RESULTS maps "program|form" -> score dict).
  figures.py compare --sweep DIR [--search DIR] [--oracle DIR] [--out PNG]
      every behavior: the total of the empty program, the hand-written program (where one exists),
      search's best program and the oracle's program, as bits per scored token.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGURES = Path.home() / "mpd-data/figures/graph_oracle"
# Categorical slots 1-4 of the reference palette (dataviz skill), in fixed order.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TERMS = [("exec_error_bits", "execution error"), ("opaque_bits", "weights the program reads"),
         ("code_bits", "code"), ("reader_error_bits", "reader error")]

plt.rcParams.update({"font.size": 20, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})


def terms(results: dict, behavior: str, labels: dict[str, str], out: Path, title: str | None = None) -> Path:
    """Stacked bars of each program's terms, in bits per scored token (each term over N)."""
    names = [n for n in labels if n in results] + [n for n in results if n not in labels and not n.startswith("_")]
    names = [n for n in names if "total_bits" in results[n]]
    results = {n: {k: (results[n].get(k) or 0.0) / results[n].get("N", 2**24) for k, _ in TERMS} for n in names}
    totals = [sum(results[n][k] for k, _ in TERMS) for n in names]
    shown = {k for k, _ in TERMS if max(results[n][k] for n in names) > 0.002 * max(totals)}  # a term too small to see gets no legend entry
    # The axis covers every program but the largest when that one is over twice the next: it is drawn cut
    # off at the edge with its total written beside it.
    order = sorted(totals)
    limit = order[-1] if len(order) < 2 or order[-1] < 2 * order[-2] else 1.25 * order[-2]
    fig, ax = plt.subplots(figsize=(14, 1.1 * len(names) + 2.2))
    for row, n in enumerate(names):
        left = 0.0
        for (k, label), color in zip(TERMS, COLORS):
            v = results[n][k]
            if v <= 0 or k not in shown:
                continue
            ax.barh(row, min(v, max(limit - left, 0)), left=left, color=color, height=0.6, edgecolor="white", linewidth=2,
                    label=label if row == next(r for r, m in enumerate(names) if results[m][k] > 0) else None)
            left += v
        text = f"{totals[row]:.2f}" + (" →" if totals[row] > limit else "")  # an arrow: the bar runs past the axis
        ax.text(min(totals[row], limit) + 0.01 * limit, row, text, va="center", fontsize=18)
    ax.set_yticks(range(len(names)), [labels.get(n, n) for n in names])
    ax.invert_yaxis()
    ax.set_xlim(0, limit * 1.3)
    ax.set_xlabel("bits per scored token (lower is better)")
    ax.set_title(title or f"Score of each program on vpd4l {behavior}", loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=17)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def best(directory: Path | None, behavior: str, key: str = "score") -> float | None:
    """The lowest total among DIR/<behavior>.*.json results (search.py's or the oracle's)."""
    if directory is None:
        return None
    found = []
    for path in directory.glob(f"{behavior}.*.json"):
        r = json.loads(path.read_text())
        s = r.get("heldout") or r.get(key) or r
        if "total_bits" in s:
            found.append(s["total_bits"] / s.get("N", 2**24))
    return min(found) if found else None


def compare(sweeps: list[Path], search: Path | None, oracle: Path | None, out: Path, title: str | None = None) -> Path:
    """Later sweep directories override earlier ones for the same behavior."""
    files = {}
    for sweep in sweeps:
        files.update({p.name: p for p in sorted(sweep.glob("*.json"))})
    rows = []
    for path in sorted(files.values(), key=lambda p: p.name):
        r = json.loads(path.read_text())
        progs = r.get("programs", {})
        per = lambda n: progs[n]["total_bits"] / progs[n].get("N", 2**24) if "total_bits" in progs.get(n, {}) else None
        rows.append((r["behavior"], per("empty"), per("hand"), best(search, r["behavior"]), best(oracle, r["behavior"])))
    series = [("empty program", 1), ("hand-written", 2), ("search's best", 3), ("oracle", 4)]
    fig, ax = plt.subplots(figsize=(14, 0.55 * len(rows) + 2.5))
    for (label, k), color in zip(series, COLORS):
        xs = [(row[k], i) for i, row in enumerate(rows) if row[k] is not None]
        if xs:
            ax.scatter([x for x, _ in xs], [i for _, i in xs], s=110, color=color, label=label, zorder=3,
                       edgecolor="white", linewidth=2)
    ax.set_yticks(range(len(rows)), [row[0] for row in rows], fontsize=15)
    ax.invert_yaxis()
    ax.set_xlabel("total score, bits per scored token (lower is better)")
    ax.set_title(title or "Score per behavior: empty, hand-written, search, oracle", loc="left")
    ax.set_xlim(left=0)
    ax.legend(frameon=True, framealpha=1, edgecolor="white", loc="upper right", fontsize=17)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def modes(results: dict, labels: dict[str, str], out: Path, title: str) -> Path:
    """One dot per (program, stand-in form): total bits per scored token. RESULTS maps "program|form"
    to a score dict."""
    forms = sorted({k.split("|")[1] for k in results})
    names = [n for n in labels if any(k.startswith(n + "|") for k in results)]
    fig, ax = plt.subplots(figsize=(14, 0.9 * len(names) + 2.2))
    for j, (form, color) in enumerate(zip(forms, COLORS)):
        shift = (j - (len(forms) - 1) / 2) * 0.22  # forms side by side within a row, so equal totals stay visible
        pts = [(results[f"{n}|{form}"]["total_bits"] / results[f"{n}|{form}"]["N"], i + shift) for i, n in enumerate(names)
               if f"{n}|{form}" in results]
        ax.scatter([x for x, _ in pts], [i for _, i in pts], s=140, color=color, label=f"{form} stand-ins", zorder=3,
                   edgecolor="white", linewidth=2)
    ax.set_yticks(range(len(names)), [labels[n] for n in names])
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel("total score, bits per scored token (lower is better)")
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=17)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def r1(table_path: Path, out: Path, title: str) -> Path:
    """Per behavior, the shared-family total (bits per scored token) of the empty program, the hand-written
    program, the best search program with native pieces, with VPD subcomponents, and the oracle's, from
    table.py's TSV (rows by program name)."""
    import csv
    rows = list(csv.DictReader(table_path.open(), delimiter="\t"))
    kinds = [("empty program", lambda p: p == "empty"), ("hand-written", lambda p: p == "hand"),
             ("search, native pieces", lambda p: p.startswith("search") and "vpd" not in p),
             ("search, VPD subcomponents", lambda p: p.startswith("search") and "vpd" in p),
             ("oracle", lambda p: p.startswith("oracle"))]
    best: dict[str, dict[str, float]] = {}
    for r in rows:
        if not r.get("shared_total"):
            continue
        for label, test in kinds:
            if test(r["program"]):
                v = float(r["shared_total"])
                best.setdefault(r["behavior"], {})
                best[r["behavior"]][label] = min(v, best[r["behavior"]].get(label, v))
    behaviors = sorted(best, key=lambda b: best[b].get("empty program", 0.0))
    fig, ax = plt.subplots(figsize=(14, 0.45 * len(behaviors) + 2.5))
    for (label, _), color in zip(kinds, COLORS + ["#e87ba4"]):
        pts = [(best[b][label], i) for i, b in enumerate(behaviors) if label in best[b]]
        if pts:
            ax.scatter([x for x, _ in pts], [i for _, i in pts], s=90, color=color, label=label, zorder=3, edgecolor="white", linewidth=1.5)
    ax.set_yticks(range(len(behaviors)), behaviors, fontsize=13)
    ax.set_xlim(left=0)
    ax.set_xlabel("total over experiments every program shares, bits per scored token (lower is better)")
    ax.set_title(title, loc="left")
    ax.legend(frameon=True, framealpha=1, edgecolor="white", loc="lower right", fontsize=15)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def frontier(curves: list[tuple[str, Path]], points: list[tuple[str, Path]], out: Path, title: str, families=None) -> Path:
    """Execution error against opaque cost (bits per scored token) along each search result's prefixes
    (curves: label, search.py result JSON with a prefix trajectory) and for single programs (points:
    label, {"score": ...} or a search result)."""
    import table
    families = families or table.SHARED
    fig, ax = plt.subplots(figsize=(13, 8))
    colors = iter(COLORS + ["#e87ba4", "#008300"])
    for label, path in curves:
        r = json.loads(path.read_text())
        pts = sorted((s["opaque_bits"] / s["N"], table.shared(s, families)[0]) for k, s in r["trajectory"][0]["prefixes"]
                     if s.get("valid", True) and table.shared(s, families))
        c = next(colors)
        ax.plot([x for x, _ in pts], [y for _, y in pts], "-o", color=c, linewidth=2, markersize=5, label=label)
    for label, path in points:
        r = json.loads(path.read_text())
        s = r.get("score") or r
        c = next(colors)
        ax.scatter([s["opaque_bits"] / s["N"]], [table.shared(s, families)[0]], s=160, color=c, marker="D", label=label, zorder=4)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("weights the program reads, bits per scored token")
    ax.set_ylabel("execution error, bits per scored token")
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, fontsize=15, loc="lower left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("terms")
    t.add_argument("results", type=Path)
    t.add_argument("behavior")
    t.add_argument("--names", nargs="*", default=[], help="name=label, in display order")
    t.add_argument("--out", type=Path)
    t.add_argument("--title")
    c = sub.add_parser("compare")
    c.add_argument("--sweep", type=Path, nargs="+", required=True)
    c.add_argument("--title")
    c.add_argument("--search", type=Path)
    c.add_argument("--oracle", type=Path)
    c.add_argument("--out", type=Path, default=FIGURES / "score_per_behavior.png")
    m = sub.add_parser("modes")
    m.add_argument("results", type=Path)
    m.add_argument("--names", nargs="*", default=[], help="name=label, in display order")
    m.add_argument("--title", default="Score of each program under each stand-in form")
    m.add_argument("--out", type=Path, default=FIGURES / "standin_forms.png")
    fr = sub.add_parser("frontier")
    fr.add_argument("--curve", nargs=2, action="append", default=[], metavar=("LABEL", "RESULT"))
    fr.add_argument("--point", nargs=2, action="append", default=[], metavar=("LABEL", "RESULT"))
    fr.add_argument("--title", default="Execution error against the cost of the weights read")
    fr.add_argument("--out", type=Path, default=FIGURES / "frontier.png")
    q = sub.add_parser("r1")
    q.add_argument("table", type=Path)
    q.add_argument("--title", default="Best program per vpd4l behavior vs the empty program")
    q.add_argument("--out", type=Path, default=FIGURES / "r1_best_per_behavior.png")
    a = ap.parse_args()
    if a.command == "frontier":
        print(frontier([(l, Path(p)) for l, p in a.curve], [(l, Path(p)) for l, p in a.point], a.out, a.title))
    elif a.command == "r1":
        print(r1(a.table, a.out, a.title))
    elif a.command == "modes":
        print(modes(json.loads(a.results.read_text()), dict(kv.split("=", 1) for kv in a.names), a.out, a.title))
    elif a.command == "terms":
        results = json.loads(a.results.read_text())
        results = results.get("programs", results)
        labels = dict(kv.split("=", 1) for kv in a.names)
        print(terms(results, a.behavior, labels, a.out or FIGURES / f"terms_{a.behavior}.png", a.title))
    else:
        print(compare(a.sweep, a.search, a.oracle, a.out, a.title))


if __name__ == "__main__":
    main()
