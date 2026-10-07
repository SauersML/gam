"""R1 table and figure (#2951 night plan): per behavior, the best program search found against the empty
program, both scored in the same search run (same prompts, same experiments, counterfactual stand-ins).
The best program is the lower of the search's result and the empty program on the shared families
(table.shared); search is a baseline and a data source here, not the method.

Columns: behavior, pieces kept (heads, MLP neurons), total and empty total (bits per scored token, shared
families), signal recovered = 1 - shared execution error / the empty program's, opaque bits per token,
held-out-family execution error (rank-one, swaps, cuts) of the best program and of the empty program,
the held-out seed's total, checker calls.

  r1.py RESULT_DIR... [--glob '*.prefix_*.json'] [--out ~/mpd-data/graph_oracle/runs/r1_table]
writes OUT.tsv and ~/mpd-data/figures/graph_oracle/<OUT name>.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import table  # noqa: E402
from figures import COLORS, FIGURES, plt  # noqa: E402

RUNS = Path.home() / "mpd-data/graph_oracle/runs"


def pieces(units: list[str]) -> tuple[int, int]:
    heads = sum(1 for u in units if u.startswith("h"))
    neurons = 0
    for u in units:
        if u.startswith("m"):
            _, a, b = u[1:].split("_")
            neurons += int(b) - int(a)
    return heads, neurons


def rows_of(directories: list[Path], pattern: str) -> list[dict]:
    out = {}
    for d in directories:
        for path in sorted(d.expanduser().glob(pattern)):
            r = json.loads(path.read_text())
            if path.name.endswith(".partial.json") or "heldout" not in r or not r.get("trajectory") or "empty" not in r["trajectory"][0]:
                continue
            behavior = path.name.split(".prefix")[0].split(".addition")[0]
            empty, found = r["trajectory"][0]["empty"], r["score"]
            # the shared families both runs have (an empty program scored for the ranking has only the clean
            # and counterfactual prompts)
            fams = tuple(f for f in table.SHARED if f in (empty.get("per_family") or {}) and f in (found.get("per_family") or {}))
            e, f = table.shared(empty, fams), table.shared(found, fams)
            use_empty = f is None or f[1] >= e[1]
            best = empty if use_empty else found
            b = table.shared(best, fams)
            he, hb = table.shared(empty, table.HELDOUT), table.shared(best, table.HELDOUT)
            heads, neurons = (0, 0) if use_empty else pieces(r["units"])
            row = {"behavior": behavior, "heads": heads, "neurons": neurons, "units": [] if use_empty else r["units"],
                   "total": b[1], "empty_total": e[1], "recovered": 1 - b[0] / e[0] if e[0] else 0.0,
                   "opaque": best["opaque_bits"] / best["N"], "heldout_exec": hb[0] if hb else None,
                   "empty_heldout_exec": he[0] if he else None,
                   "heldout_seed_total": None if use_empty else r["heldout"]["total_bits"] / r["heldout"]["N"],
                   "calls": r.get("calls"), "families": ",".join(fams), "file": str(path)}
            if behavior not in out or row["total"] < out[behavior]["total"]:
                out[behavior] = row
    return sorted(out.values(), key=lambda r: r["empty_total"] - r["total"], reverse=True)


def figure(rows: list[dict], out: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(14, 0.42 * len(rows) + 2.5))
    ys = range(len(rows))
    for i, r in enumerate(rows):
        ax.plot([r["total"], r["empty_total"]], [i, i], color="#c3c2b7", linewidth=2, zorder=1)
    won = [i for i, r in enumerate(rows) if r["recovered"] > 0]
    lost = [i for i, r in enumerate(rows) if r["recovered"] <= 0]
    ax.scatter([rows[i]["empty_total"] for i in won], won, s=80, color=COLORS[1], label="empty program", zorder=3)
    ax.scatter([rows[i]["total"] for i in won], won, s=80, color=COLORS[0], label="best program found by search", zorder=3)
    ax.scatter([rows[i]["empty_total"] for i in lost], lost, s=80, facecolor="white", edgecolor="#898781", linewidth=2,
               label="nothing found beats the empty program", zorder=3)
    for i, r in enumerate(rows):
        if r["recovered"] > 0.05:
            ax.text(max(r["total"], r["empty_total"]) + 0.2, i, f"{r['recovered']:.0%} of the signal", va="center", fontsize=12)
    ax.set_yticks(list(ys), [r["behavior"] for r in rows], fontsize=13)
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel("total bits per scored token on shared experiments (lower is better)")
    ax.set_title(title, loc="left")
    ax.legend(frameon=True, framealpha=1, edgecolor="white", loc="lower right", fontsize=15)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results", type=Path, nargs="+")
    ap.add_argument("--glob", default="*.prefix_*.json")
    ap.add_argument("--out", type=Path, default=RUNS / "r1_table")
    ap.add_argument("--title", default="vpd4l: search's best program vs the empty program")
    a = ap.parse_args()
    rows = rows_of(a.results, a.glob)
    cols = ["behavior", "heads", "neurons", "total", "empty_total", "recovered", "opaque", "heldout_exec", "empty_heldout_exec",
            "heldout_seed_total", "calls", "families", "units", "file"]
    fmt = lambda v: "" if v is None else f"{v:.4f}" if isinstance(v, float) else ",".join(v) if isinstance(v, list) else str(v)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{a.out}.tsv").write_text("\t".join(cols) + "\n" + "".join("\t".join(fmt(r[c]) for c in cols) + "\n" for r in rows))
    print(figure(rows, FIGURES / f"{a.out.name}.png", a.title))
    for r in rows:
        print(f"{r['behavior']:30s} {r['heads']:2d} heads {r['neurons']:5d} neurons  {r['total']:6.2f} vs {r['empty_total']:6.2f}  "
              f"recovered {r['recovered']:5.2f}  held-out exec {fmt(r['heldout_exec'])} vs {fmt(r['empty_heldout_exec'])}")


if __name__ == "__main__":
    main()
