"""Best program per behavior, exported for the oracle's training (#2951, night plan R4) and the gallery: for
each behavior the program with the lowest total over the experiment families every program shares
(table.shared) among search results, hand-written programs and the empty program, written as
{"behavior", "split", "source", "score", "origin"} to OUT/<behavior>.json and, through g-mech's printer
(measured facts as comments), to OUT/printed/<behavior>.py and .graph.json.

  export_best.py RESULT_DIR... --sweep SWEEP_DIR [--model vpd4l] [--out ~/mpd-data/graph_oracle/runs/best]
                 [--split train] [--no-print]
RESULT_DIRs hold search.py results (<behavior>.<mode><tag>.json) or {"behavior", "source", "score"} files;
SWEEP_DIR holds sweep.py files (the empty program's score per behavior).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import mech  # noqa: E402
import programs  # noqa: E402
import table  # noqa: E402

BEHAVIORS = Path.home() / "mpd-data/graph_oracle/behaviors"


def fair(score: dict) -> float:
    s = table.shared(score)
    return float("inf") if s is None or not score.get("valid", True) else s[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results", type=Path, nargs="*")
    ap.add_argument("--sweep", type=Path, nargs="*", default=[])
    ap.add_argument("--model", default="vpd4l")
    ap.add_argument("--out", type=Path, default=Path.home() / "mpd-data/graph_oracle/runs/best")
    ap.add_argument("--split", choices=["train", "heldout", "all"], default="all")
    ap.add_argument("--no-print", action="store_true")
    a = ap.parse_args()
    best: dict[str, dict] = {}

    def offer(behavior: str, source: str, score: dict, origin: str):
        if behavior not in best or fair(score) < fair(best[behavior]["score"]):
            best[behavior] = {"behavior": behavior, "source": source, "score": score, "origin": origin}

    for d in a.sweep:
        for path in sorted(d.expanduser().glob("*.json")):
            r = json.loads(path.read_text())
            if "empty" in r.get("programs", {}):
                offer(r["behavior"], programs.empty(a.model), r["programs"]["empty"], f"empty ({path})")
    for d in a.results:
        for path in sorted(d.expanduser().glob("*.json")):
            r = json.loads(path.read_text())
            if "source" not in r or not ("score" in r or "heldout" in r):
                continue
            behavior = r.get("behavior") or ".".join(path.stem.split(".")[:2])
            offer(behavior, r["source"], r.get("score") or r["heldout"], str(path))
    out = a.out.expanduser()
    (out / "printed").mkdir(parents=True, exist_ok=True)
    engine = None
    for behavior, b in sorted(best.items()):
        bpath = BEHAVIORS / a.model / f"{behavior}.json"
        if not bpath.exists():
            continue
        record = json.loads(bpath.read_text())
        b["split"] = record.get("split")
        if a.split != "all" and b["split"] != a.split:
            continue
        b["fair_total_bits_per_token"] = fair(b["score"])
        (out / f"{behavior}.json").write_text(json.dumps(b, indent=1))
        if not a.no_print and b["origin"].startswith("empty") is False:
            import printer
            engine = engine or printer.engine_for(a.model)
            src, graph = printer.printed(mech.trace_inline(b["source"], a.model), record, b["score"])
            (out / "printed" / f"{behavior}.py").write_text(src)
            (out / "printed" / f"{behavior}.graph.json").write_text(json.dumps(graph, indent=1))
        print(f"{behavior}\t{b['split']}\t{b['fair_total_bits_per_token']:.3f}\t{b['origin']}")


if __name__ == "__main__":
    main()
