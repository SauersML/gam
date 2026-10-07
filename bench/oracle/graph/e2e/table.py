"""The oracle-vs-search table (#2951): per behavior, every score term of the empty program, the
hand-written program, search's best program (per search mode) and the oracle's best program, in bits
per scored token, with the checker calls search used.

  table.py [--sweep DIR] [--search DIR ...] [--oracle DIR] [--out TSV]
Reads sweep.py's OUT/<behavior>.json, search.py's <behavior>.<mode><tag>.json (its held-out rescore)
and the oracle's <behavior>.<run>.json ({"score": ...}, g-rl); writes a TSV and prints it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS = Path.home() / "mpd-data/graph_oracle/runs"
TERMS = ["total_bits", "exec_error_bits", "opaque_bits", "code_bits", "reader_error_bits"]
# The behavior's own experiment families: drawn from the seed alone, identical for every program, so a
# total over them compares programs on the same experiments (the aimed, swap and cut families depend on
# what the program declares).
SHARED = ("clean", "counterfactual", "edit_uniform", "rank_one")
COLUMNS = ["behavior", "program", "stand_in", "total", "exec_error", "opaque", "code", "reader_error", "opaque_numbers",
           "shared_exec_error", "shared_total", "checker_calls", "source_file"]


def shared(s: dict) -> tuple[float, float] | None:
    """Mean KL per token over the shared families and the total with it in place of the execution error
    (bits per scored token), or None without per-family terms."""
    pf = s.get("per_family") or {}
    tokens = sum(pf[f]["tokens"] for f in SHARED if f in pf)
    if not tokens:
        return None
    mean = sum(pf[f]["mean_kl_bits"] * pf[f]["tokens"] for f in SHARED if f in pf) / tokens
    n = s.get("N", 2**24)
    return mean, mean + (s.get("opaque_bits", 0.0) + s.get("code_bits", 0.0) + (s.get("reader_error_bits") or 0.0)) / n


def row(behavior: str, program: str, s: dict, stand_in, calls, source: str) -> list[str]:
    n = s.get("N", 2**24)
    cells = [behavior, program, str(stand_in or "default")]
    cells += ["" if s.get(k) is None else f"{s[k] / n:.4f}" for k in TERMS]
    sh = shared(s)
    cells += [str(s.get("opaque_numbers", ""))] + (["", ""] if sh is None else [f"{sh[0]:.4f}", f"{sh[1]:.4f}"])
    cells += ["" if calls is None else str(calls), source]
    return cells


def oracle_eval_rows(path: Path) -> list[list[str]]:
    """g-rl's train.py evaluation log (eval.jsonl): per behavior, the latest step's best-of-N and mean
    single-sample totals (bits, N = 2^24 assumed: the score's default size)."""
    latest: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if "behavior" in r and (r["behavior"] not in latest or r["step"] >= latest[r["behavior"]]["step"]):
            latest[r["behavior"]] = r
    rows = []
    for b, r in latest.items():
        for label, key in (("oracle best of n", "best_bits"), ("oracle mean", "mean_bits")):
            rows.append(row(b, f"{label} (step {r['step']})", {"total_bits": r[key]}, None, None, str(path)))
    return rows


def collect(sweep: Path | None, searches: list[Path], oracle: Path | None, oracle_evals: list[Path] = ()) -> list[list[str]]:
    rows = []
    behaviors = set()
    if sweep is not None:
        for path in sorted(sweep.glob("*.json")):
            r = json.loads(path.read_text())
            behaviors.add(r["behavior"])
            for name in ("empty", "hand", "random", "full"):
                if "total_bits" in r["programs"].get(name, {}):
                    rows.append(row(r["behavior"], name, r["programs"][name], r.get("stand_in"), None, str(path)))
    for directory in searches:
        for path in sorted(directory.glob("*.json")):
            r = json.loads(path.read_text())
            if "heldout" not in r:
                continue
            behavior, mode = path.stem.split(".")[0] + "." + path.stem.split(".")[1], ".".join(path.stem.split(".")[2:])
            rows.append(row(behavior, f"search {mode}", r["heldout"], r.get("stand_in"), r.get("calls"), str(path)))
    if oracle is not None:
        best: dict[str, tuple[float, list[str]]] = {}
        for path in sorted(oracle.glob("*.json")):
            r = json.loads(path.read_text())
            s = r.get("score", {})
            if "total_bits" in s and (r["behavior"] not in best or s["total_bits"] < best[r["behavior"]][0]):
                best[r["behavior"]] = (s["total_bits"], row(r["behavior"], "oracle", s, r.get("stand_in"), None, str(path)))
        rows += [v[1] for v in best.values()]
    for path in oracle_evals:
        rows += oracle_eval_rows(path)
    return sorted(rows, key=lambda c: (c[0], c[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep", type=Path, default=RUNS / "sweep")
    ap.add_argument("--search", type=Path, nargs="*", default=[RUNS / "search"])
    ap.add_argument("--oracle", type=Path, default=RUNS / "oracle")
    ap.add_argument("--oracle-eval", type=Path, nargs="*", default=[], help="g-rl's eval.jsonl files")
    ap.add_argument("--out", type=Path, default=RUNS / "oracle_vs_search.tsv")
    a = ap.parse_args()
    rows = collect(a.sweep if a.sweep.exists() else None, [d for d in a.search if d.exists()], a.oracle if a.oracle.exists() else None,
                   [p for p in a.oracle_eval if p.exists()])
    text = "\t".join(COLUMNS) + "\n" + "".join("\t".join(r) + "\n" for r in rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
