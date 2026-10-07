"""R1 table and figure (#2951 night plan): per behavior, the best of the empty program, the hand-written program
and search's best program, each against the empty program scored on the same prompts and experiments
(search: its own run, train prompts; hand: g-mech's rescoring). Search is a baseline and a data source here,
not the method.

In plain units per behavior: the share of the behavior's signal the program reproduces (1 - execution error
/ the empty program's, on the fit families: clean and counterfactual prompts, uniform and targeted weight
edits), the share of the model's weights it names (opaque numbers / every number of every head and MLP),
the total against the empty total (bits per scored token, fit families), the error on held-out families
(rank-one perturbations, site operations), and the checker build.

  r1.py SEARCH_DIR... [--hand JSONL...] [--glob '*.prefix_final.json'] [--out ~/mpd-data/graph_oracle/runs/r1_final]
writes OUT.tsv, OUT_best/<behavior>.json (programs that beat the empty program) and
~/mpd-data/figures/graph_oracle/<OUT name>.png.
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
MODEL_NUMBERS = {"vpd4l": 28_324_608}  # every number of every head and MLP (the full program's count)


def entry(behavior: str, origin: str, empty: dict, found: dict, source: str | None, build: str, model: str = "vpd4l",
          families=table.FIT) -> dict:
    fams = tuple(f for f in families if f in (empty.get("per_family") or {}) and f in (found.get("per_family") or {}))
    e, f = table.shared(empty, fams), table.shared(found, fams)
    he, hf = table.shared(empty, table.HELDOUT), table.shared(found, table.HELDOUT)
    return {"behavior": behavior, "origin": origin, "total": f[1], "empty_total": e[1],
            "reproduced": 1 - f[0] / e[0] if e[0] else 0.0, "weights_named": found.get("opaque_numbers", 0) / MODEL_NUMBERS[model],
            "opaque": found["opaque_bits"] / found["N"], "heldout_exec": hf[0] if hf else None, "empty_heldout_exec": he[0] if he else None,
            "saving": 1 - f[1] / e[1] if e[1] else 0.0, "build": build, "source": source, "families": ",".join(fams)}


def search_entries(directories: list[Path], pattern: str) -> list[dict]:
    out = []
    for d in directories:
        for path in sorted(d.expanduser().glob(pattern)):
            if path.name.endswith(".partial.json"):
                continue
            r = json.loads(path.read_text())
            if "heldout" not in r or not r.get("trajectory"):
                continue
            t = r["trajectory"][0]
            empty = next((s for k, s in t.get("prefixes", []) if k == 0), t.get("empty"))
            behavior = path.name.split(".prefix")[0].split(".addition")[0]
            out.append(entry(behavior, "empty", empty, empty, None, r.get("checker", "")))
            if r["units"]:
                out.append(entry(behavior, "search", empty, r["score"], r["source"], r.get("checker", "")))
    return out


def hand_entries(files: list[Path]) -> list[dict]:
    out = []
    for path in files:
        for line in path.expanduser().read_text().splitlines():
            r = json.loads(line)
            if "program" in r and "empty" in r:
                out.append(entry(r["behavior"], "empty", r["empty"], r["empty"], None, r.get("checker", "")))
                out.append(entry(r["behavior"], "hand", r["empty"], r["program"], r.get("example"), r.get("checker", "")))
    return out


def best_per_behavior(entries: list[dict]) -> list[dict]:
    best = {}
    for e in entries:
        if e["behavior"] not in best or e["saving"] > best[e["behavior"]]["saving"]:
            best[e["behavior"]] = e
    return sorted(best.values(), key=lambda r: -r["saving"])


def figure(rows: list[dict], out: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(14, 0.42 * len(rows) + 2.5))
    won = [i for i, r in enumerate(rows) if r["saving"] > 0]
    lost = [i for i, r in enumerate(rows) if r["saving"] <= 0]
    for i in won:
        ax.plot([rows[i]["total"], rows[i]["empty_total"]], [i, i], color="#c3c2b7", linewidth=2, zorder=1)
    ax.scatter([rows[i]["empty_total"] for i in won], won, s=80, color=COLORS[1], label="empty program", zorder=3)
    for origin, color, label in (("hand", COLORS[2], "hand-written program"), ("search", COLORS[0], "program found by search")):
        pts = [i for i in won if rows[i]["origin"] == origin]
        ax.scatter([rows[i]["total"] for i in pts], pts, s=80, color=color, label=label, zorder=3)
    ax.scatter([rows[i]["empty_total"] for i in lost], lost, s=80, facecolor="white", edgecolor="#898781", linewidth=2,
               label="nothing beats the empty program", zorder=3)
    for i in won:
        r = rows[i]
        ax.text(max(r["total"], r["empty_total"]) + 0.2, i, f"{r['reproduced']:.0%} of the behavior, {r['weights_named']:.1%} of the weights",
                va="center", fontsize=11)
    ax.set_yticks(range(len(rows)), [r["behavior"] for r in rows], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(left=0, right=max(max(r["empty_total"], r["total"]) for r in rows) * 1.6)
    ax.set_xlabel("total bits per scored token (lower is better)")
    ax.set_title(title, loc="left")
    ax.legend(frameon=True, framealpha=1, edgecolor="white", loc="lower right", fontsize=14)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results", type=Path, nargs="*")
    ap.add_argument("--hand", type=Path, nargs="*", default=[])
    ap.add_argument("--glob", default="*.prefix_final.json")
    ap.add_argument("--out", type=Path, default=RUNS / "r1_final")
    ap.add_argument("--title", default="vpd4l: the best program per behavior against the empty program")
    a = ap.parse_args()
    entries = search_entries(a.results, a.glob) + hand_entries(a.hand)
    rows = best_per_behavior(entries)
    cols = ["behavior", "origin", "reproduced", "weights_named", "total", "empty_total", "opaque", "heldout_exec",
            "empty_heldout_exec", "build", "families"]
    fmt = lambda v: "" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{a.out}.tsv").write_text("\t".join(cols) + "\n" + "".join("\t".join(fmt(r[c]) for c in cols) + "\n" for r in rows))
    # every candidate, for the search-vs-hand comparison
    Path(f"{a.out}.all.tsv").write_text("\t".join(cols) + "\n" + "".join("\t".join(fmt(r[c]) for c in cols) + "\n" for r in entries))
    best = Path(f"{a.out}_best")
    best.mkdir(parents=True, exist_ok=True)
    for r in rows:
        if r["saving"] > 0 and r["source"]:
            source = r["source"] if r["origin"] == "search" else (HERE.parent / "examples" / f"{r['source']}.py").read_text()
            (best / f"{r['behavior']}.json").write_text(json.dumps({"behavior": r["behavior"], "source": source, "origin": r["origin"],
                                                                    "reproduced": r["reproduced"], "weights_named": r["weights_named"],
                                                                    "total": r["total"], "empty_total": r["empty_total"], "build": r["build"]}))
    print(figure(rows, FIGURES / f"{a.out.name}.png", a.title))
    for r in rows:
        print(f"{r['behavior']:30s} {r['origin']:6s} reproduced {r['reproduced']:6.1%}  weights {r['weights_named']:6.2%}  "
              f"{r['total']:6.2f} vs {r['empty_total']:6.2f}  held-out {fmt(r['heldout_exec'])} vs {fmt(r['empty_heldout_exec'])}")


if __name__ == "__main__":
    main()
