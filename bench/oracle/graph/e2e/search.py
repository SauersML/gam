"""Search baseline on the graph oracle's score (#2951): the score the oracle must beat per behavior, found
by exact greedy search through the checker, with the number of checker calls it used.

Units are native pieces: each attention head, and each layer's MLP neurons as one block of which a
move may take any dyadic sub-block (index halves, their halves, ... down to --min-neurons). A program declares its units as
nodes and lists every causal edge among them (embed into every unit, every unit into every later unit
and into the logits), so it equals M with its undeclared pieces replaced by stand-ins.
  addition  from the empty program: each step adds the head or MLP sub-block that lowers the total
            most; stops when no addition lowers it.
  removal   from the full program: each step removes the head or MLP sub-block whose removal lowers
            the total most; stops when no removal lowers it.
Every candidate of a step is scored under the same experiment seed; the final program is rescored
under a held-out seed. Candidates are scored in parallel by --workers checker processes.

  search.py BEHAVIOR.json [--mode addition|removal|both] [--experiments 16] [--workers 4]
Writes ~/mpd-data/graph_oracle/runs/search/<behavior>.<mode>.json (trajectory, final program source,
its terms, checker calls) and appends the final program to status.tsv as program "search_<mode>".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
os.environ.setdefault("MPD_MEM_GIB", "1")

import mech  # noqa: E402
import run as e2e  # noqa: E402
import score  # noqa: E402

OUT = Path.home() / "mpd-data/graph_oracle/runs/search"

# A unit: ("head", layer, head) or ("mlp", layer, start, stop) for neurons start..stop-1.


def site(unit) -> int:
    """The residual stream a unit reads (design.txt section 5): attention at 2l, MLP at 2l+1."""
    return 2 * unit[1] + (unit[0] == "mlp")


def name(unit) -> str:
    return f"h{unit[1]}_{unit[2]}" if unit[0] == "head" else f"m{unit[1]}_{unit[2]}_{unit[3]}"


def unit_of(text: str):
    """The unit a name() denotes: h<l>_<h> or m<l>_<start>_<stop>."""
    parts = [int(x) for x in text[1:].split("_")]
    return ("head", *parts) if text[0] == "h" else ("mlp", *parts)


def source(units) -> str:
    """The program declaring `units` as nodes with every causal edge among them listed."""
    units = sorted(units, key=lambda u: (site(u), u))
    lines = ['"""Found by greedy search on the score (e2e/search.py)."""', "from mech import node, edges, L, embed, logits"]
    for u in units:
        piece = f"L[{u[1]}].head[{u[2]}]" if u[0] == "head" else f"L[{u[1]}].mlp[{u[2]}:{u[3]}]"
        lines.append(f"{name(u)} = node({piece})")
    wires = []
    for i, u in enumerate(units):
        wires += [f"    {w} >> {name(u)}," for w in ["embed"] + [name(v) for v in units[:i] if site(v) < site(u)]]
    wires += [f"    {w} >> logits," for w in ["embed"] + [name(u) for u in units]]
    if units:
        lines += ["edges("] + wires + [")"]
    return "\n".join(lines) + "\n"


def pieces_of(unit, min_neurons: int):
    """Every dyadic sub-block b of `unit` (itself; for an MLP block its halves, their halves, ... down
    to min_neurons) with the rest of the unit as dyadic blocks: [(b, rest)]."""
    out = [(unit, [])]
    if unit[0] == "mlp" and unit[3] - unit[2] >= 2 * min_neurons:
        mid = (unit[2] + unit[3]) // 2
        lo, hi = ("mlp", unit[1], unit[2], mid), ("mlp", unit[1], mid, unit[3])
        out += [(b, rest + [hi]) for b, rest in pieces_of(lo, min_neurons)]
        out += [(b, rest + [lo]) for b, rest in pieces_of(hi, min_neurons)]
    return out


class Pool:
    """`workers` checker processes with the behavior loaded; scores programs in parallel."""

    def __init__(self, model: str, behavior: Path, workers: int, export: Path | None = None, stand_in: str | None = None):
        self.checkers = [score.Checker(model, export) for _ in range(workers)]
        self.extra = {} if stand_in is None else {"stand_in": stand_in}
        for c in self.checkers:
            c.behavior(behavior)
        self.calls = 0

    def score(self, programs: list[str], experiments: int, seed: int) -> list[dict]:
        def one(k):
            c = self.checkers[k % len(self.checkers)]
            out = []
            for i in range(k, len(programs), len(self.checkers)):
                out.append((i, c.score(programs[i], experiments=experiments, seed=seed, reader=False, **self.extra)))
            return out

        with ThreadPoolExecutor(len(self.checkers)) as ex:
            results = [r for part in ex.map(one, range(min(len(self.checkers), len(programs)))) for r in part]
        self.calls += len(programs)
        return [r for _, r in sorted(results, key=lambda x: x[0])]

    def close(self):
        for c in self.checkers:
            c.close()


def all_units(model: str):
    s = mech.shapes(model)
    return [("head", l, h) for l in range(s["layers"]) for h in range(s["heads"])] + [("mlp", l, 0, s["d_mlp"]) for l in range(s["layers"])]


def greedy(pool: Pool, model: str, mode: str, experiments: int, seed: int, min_neurons: int, log, start=None) -> dict:
    """`start`: the units to start from (default: none for addition, every unit for removal)."""
    full = all_units(model)
    current = list(start) if start is not None else [] if mode == "addition" else list(full)
    # Addition draws from `outside`: the pieces not declared yet, as dyadic blocks (whole units the
    # start does not touch; a start's partial MLP blocks leave nothing outside in that layer).
    touched = {(u[0], u[1]) if u[0] == "mlp" else u for u in current}
    outside = [u for u in full if ((u[0], u[1]) if u[0] == "mlp" else u) not in touched] if mode == "addition" else []
    best = pool.score([source(current)], experiments, seed)[0]
    trajectory = [{"step": 0, "units": [name(u) for u in current], "total_bits": best["total_bits"],
                   "exec_error_bits": best["exec_error_bits"], "opaque_bits": best["opaque_bits"], "calls": pool.calls}]
    log(f"{mode} step 0: {best['total_bits']:.6g} bits")
    step = 0
    while True:
        step += 1
        moves = []  # (units, outside, (verb, block))
        if mode == "addition":
            for u in outside:
                for b, rest in pieces_of(u, min_neurons):
                    moves.append((current + [b], [v for v in outside if v != u] + rest, ("add", b)))
        else:
            for u in current:
                for b, rest in pieces_of(u, min_neurons):
                    moves.append(([v for v in current if v != u] + rest, outside + [b], ("remove", b)))
        if not moves:
            break
        t = time.time()
        results = pool.score([source(m[0]) for m in moves], experiments, seed)
        k = min(range(len(moves)), key=lambda i: results[i]["total_bits"])
        log(f"{mode} step {step}: {len(moves)} candidates in {time.time() - t:.0f} s; best {moves[k][2][0]} "
            f"{name(moves[k][2][1])} {results[k]['total_bits']:.6g} vs {best['total_bits']:.6g}")
        if results[k]["total_bits"] >= best["total_bits"]:
            break
        current, outside, best = moves[k][0], moves[k][1], results[k]
        trajectory.append({"step": step, "move": [moves[k][2][0], name(moves[k][2][1])], "units": [name(u) for u in current],
                           "total_bits": best["total_bits"], "exec_error_bits": best["exec_error_bits"],
                           "opaque_bits": best["opaque_bits"], "calls": pool.calls})
    return {"units": current, "source": source(current), "score": best, "trajectory": trajectory}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("behavior", type=Path)
    ap.add_argument("--mode", default="both", choices=["addition", "removal", "both"])
    ap.add_argument("--experiments", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heldout-seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-neurons", type=int, default=384)
    ap.add_argument("--export", type=Path, help="the model's export directory (score.py's default otherwise)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--start", help="comma-separated units to start from (h<l>_<h>, m<l>_<start>_<stop>)")
    ap.add_argument("--tag", default="", help="suffix of the output names")
    ap.add_argument("--stand-in", help="the checker's stand-in option (counterfactual, global, position, ...)")
    a = ap.parse_args()
    path = a.behavior.expanduser()
    behavior = json.loads(path.read_text())
    model = behavior["model"]
    out = a.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    pool = Pool(model, path, a.workers, a.export, a.stand_in)
    try:
        for mode in (["addition", "removal"] if a.mode == "both" else [a.mode]):
            start = pool.calls
            stem = f"{behavior['id']}.{mode}{a.tag}"
            log_path = out / f"{stem}.log"
            with log_path.open("w") as logf:
                def log(msg):
                    print(msg, flush=True)
                    logf.write(msg + "\n")
                    logf.flush()
                start = [unit_of(t) for t in a.start.split(",")] if a.start else None
                found = greedy(pool, model, mode, a.experiments, a.seed, a.min_neurons, log, start)
                heldout = pool.score([found["source"]], a.experiments, a.heldout_seed)[0]
                found.update(units=[name(u) for u in found["units"]], heldout=heldout, calls=pool.calls - start, stand_in=a.stand_in,
                             experiments=a.experiments, seed=a.seed, heldout_seed=a.heldout_seed)
                log(f"{mode}: {len(found['units'])} units, {found['score']['total_bits']:.6g} bits (held-out seed "
                    f"{heldout['total_bits']:.6g}), {found['calls']} checker calls")
            (out / f"{stem}.json").write_text(json.dumps(found, indent=1))
            e2e.record([e2e.status_line(model, behavior["id"], f"search_{mode}{a.tag}", found["heldout"], a.stand_in)])
    finally:
        pool.close()


if __name__ == "__main__":
    main()
