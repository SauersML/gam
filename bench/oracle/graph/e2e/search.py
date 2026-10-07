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
under a held-out seed. Candidates go to --workers checker processes as score_batch requests (each
checker scores its share in parallel threads, RAYON_NUM_THREADS).

  search.py BEHAVIOR.json [--mode addition|removal|both] [--experiments 16] [--workers 1] [--stand-in counterfactual|global]
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


# A VPD unit: ("vpd", layer, kc, kd) = the kc c_fc and kd down_proj subcomponents of layer l's MLP with
# the largest measured removal effect (RANKING: per site, subcomponent indices from largest effect down).
RANKING: dict[str, list[int]] = {}
GROW = 2  # a VPD unit's growth factor per move (--grow)


def load_ranking(path: Path) -> None:
    """Removal effects of VPD subcomponents (g-mech's measure/vpd_induction_removal.py output:
    sites -> {"kl_bits": [per subcomponent]}), as per-site index orders, largest effect first."""
    sites = json.loads(Path(path).read_text())["sites"]
    for key, v in sites.items():
        layer, site_name = int(key.split(".")[1]), key.split(".")[-1]
        kl = v["kl_bits"]
        RANKING[f"{layer}.{site_name}"] = sorted(range(len(kl)), key=lambda i: -kl[i])


def site(unit) -> int:
    """The residual stream a unit reads (design.txt section 5): attention at 2l, MLP at 2l+1."""
    return 2 * unit[1] + (unit[0] in ("mlp", "vpd"))


def name(unit) -> str:
    return {"head": "h", "mlp": "m", "vpd": "v"}[unit[0]] + "_".join(str(x) for x in unit[1:]) if unit[0] != "head" else f"h{unit[1]}_{unit[2]}"


def unit_of(text: str):
    """The unit a name() denotes: h<l>_<h>, m<l>_<start>_<stop> or v<l>_<kc>_<kd>."""
    parts = [int(x) for x in text[1:].split("_")]
    return ({"h": "head", "m": "mlp", "v": "vpd"}[text[0]], *parts)


def piece(unit) -> str:
    if unit[0] == "head":
        return f"L[{unit[1]}].head[{unit[2]}]"
    if unit[0] == "mlp":
        return f"L[{unit[1]}].mlp[{unit[2]}:{unit[3]}]"
    l, kc, kd = unit[1:]
    c = ", ".join(map(str, sorted(RANKING[f"{l}.c_fc"][:kc])))
    d = ", ".join(map(str, sorted(RANKING[f"{l}.down_proj"][:kd])))
    return f"PD.vpd[{l}].c_fc[{c}], PD.vpd[{l}].down_proj[{d}]"


def source(units) -> str:
    """The program declaring `units` as nodes with every causal edge among them listed."""
    units = sorted(units, key=lambda u: (site(u), u))
    lines = ['"""Found by greedy search on the score (e2e/search.py)."""', "from mech import node, edges, L, PD, embed, logits"]
    for u in units:
        lines.append(f"{name(u)} = node({piece(u)})")
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
    """`workers` checker processes with the behavior loaded; each scores its share of a step's programs in
    one score_batch request (the checker's parallel threads, M's run per experiment shared)."""

    def __init__(self, model: str, behavior: Path, workers: int, export: Path | None = None, stand_in: str | None = None,
                 views: dict | None = None, log_path: Path | None = None):
        """views: decomposition views for the checker (score.Checker's); log_path: every scored program is
        appended there as a JSON line {"source", "seed", "experiments", "score"} (search's data)."""
        self.model, self.stand_in, self.log_path = model, stand_in, log_path
        self.checkers = [score.Checker(model, export, views=views) for _ in range(workers)]
        for c in self.checkers:
            e2e.load_behavior(c, behavior)
        self.calls = 0

    def score(self, programs: list[str], experiments: int, seed: int) -> list[dict]:
        irs = [e2e.ir_of(p, self.model, self.stand_in) for p in programs]

        def one(k):
            share = list(range(k, len(irs), len(self.checkers)))
            answer = self.checkers[k].request({"op": "score_batch", "programs": [irs[i] for i in share], "experiments": experiments,
                                               "seed": seed, "routing": "edges", "N": None, "reader_top": 0})
            return list(zip(share, answer["scores"]))

        with ThreadPoolExecutor(len(self.checkers)) as ex:
            results = [r for part in ex.map(one, range(min(len(self.checkers), len(irs)))) for r in part]
        self.calls += len(programs)
        scored = [r for _, r in sorted(results, key=lambda x: x[0])]
        if self.log_path is not None:
            with self.log_path.open("a") as f:
                for p, r in zip(programs, scored):
                    f.write(json.dumps({"source": p, "seed": seed, "experiments": experiments, "score": r}) + "\n")
        return scored

    def close(self):
        for c in self.checkers:
            c.close()


def all_units(model: str, mlp_view: str = "native"):
    """Every unit: the heads, and per layer its MLP as one native block or (mlp_view "vpd") a VPD unit placeholder."""
    s = mech.shapes(model)
    mlps = [("mlp", l, 0, s["d_mlp"]) for l in range(s["layers"])] if mlp_view == "native" else [("vpd", l, 0, 0) for l in range(s["layers"])]
    return [("head", l, h) for l in range(s["layers"]) for h in range(s["heads"])] + mlps


def objective_of(kind: str):
    """The quantity search minimizes: the score's total, or (kind "shared") the total with the execution
    error taken over the experiment families every program shares (table.shared)."""
    if kind == "total":
        return lambda r: r["total_bits"]
    import table
    return lambda r: table.shared(r)[1] * r["N"]


def greedy(pool: Pool, model: str, mode: str, experiments: int, seed: int, min_neurons: int, log, start=None,
           mlp_view: str = "native", checkpoint=None, objective=None) -> dict:
    """`start`: the units to start from (default: none for addition, every unit for removal). With mlp_view
    "vpd" (addition only), MLPs enter as VPD units grown by doubling along the removal ranking."""
    objective = objective or (lambda r: r["total_bits"])
    full = all_units(model, mlp_view)
    current = list(start) if start is not None else [] if mode == "addition" else list(full)
    # Addition draws from `outside`: the pieces not declared yet, as dyadic blocks (whole units the
    # start does not touch; a start's partial MLP blocks leave nothing outside in that layer).
    touched = {(u[0], u[1]) if u[0] in ("mlp", "vpd") else u for u in current}
    outside = [u for u in full if ((u[0], u[1]) if u[0] in ("mlp", "vpd") else u) not in touched] if mode == "addition" else []
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
                if u[0] == "vpd":  # a layer's VPD MLP enters with its strongest subcomponent of each matrix
                    moves.append((current + [("vpd", u[1], 1, 1)], [v for v in outside if v != u], ("add", ("vpd", u[1], 1, 1))))
                    continue
                for b, rest in pieces_of(u, min_neurons):
                    moves.append((current + [b], [v for v in outside if v != u] + rest, ("add", b)))
            for u in [u for u in current if u[0] == "vpd"]:  # grow a VPD MLP: GROW times the c_fc or the down_proj subcomponents
                for grown in (("vpd", u[1], min(GROW * u[2], len(RANKING[f"{u[1]}.c_fc"])), u[3]),
                              ("vpd", u[1], u[2], min(GROW * u[3], len(RANKING[f"{u[1]}.down_proj"]))),
                              ("vpd", u[1], min(GROW * u[2], len(RANKING[f"{u[1]}.c_fc"])), min(GROW * u[3], len(RANKING[f"{u[1]}.down_proj"])))):
                    if grown != u:
                        moves.append(([v for v in current if v != u] + [grown], outside, ("grow", grown)))
        else:
            for u in current:
                for b, rest in pieces_of(u, min_neurons):
                    moves.append(([v for v in current if v != u] + rest, outside + [b], ("remove", b)))
        if not moves:
            break
        t = time.time()
        results = pool.score([source(m[0]) for m in moves], experiments, seed)
        k = min(range(len(moves)), key=lambda i: objective(results[i]))
        log(f"{mode} step {step}: {len(moves)} candidates in {time.time() - t:.0f} s; best {moves[k][2][0]} "
            f"{name(moves[k][2][1])} {objective(results[k]):.6g} vs {objective(best):.6g}")
        candidates = [{"move": [m[2][0], name(m[2][1])], "total_bits": r["total_bits"], "exec_error_bits": r["exec_error_bits"],
                       "opaque_bits": r["opaque_bits"]} for m, r in zip(moves, results)]
        if objective(results[k]) >= objective(best):
            trajectory.append({"step": step, "stopped": True, "candidates": candidates, "calls": pool.calls})
            break
        current, outside, best = moves[k][0], moves[k][1], results[k]
        trajectory.append({"step": step, "move": [moves[k][2][0], name(moves[k][2][1])], "units": [name(u) for u in current],
                           "total_bits": best["total_bits"], "exec_error_bits": best["exec_error_bits"],
                           "opaque_bits": best["opaque_bits"], "calls": pool.calls, "candidates": candidates})
        if checkpoint is not None:  # a job cut off by its time limit keeps every finished step
            checkpoint({"units": [name(u) for u in current], "source": source(current), "score": best, "trajectory": trajectory,
                        "partial": True})
    return {"units": current, "source": source(current), "score": best, "trajectory": trajectory}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("behavior", type=Path)
    ap.add_argument("--mode", default="both", choices=["addition", "removal", "both"])
    ap.add_argument("--experiments", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heldout-seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--min-neurons", type=int, default=384)
    ap.add_argument("--export", type=Path, help="the model's export directory (score.py's default otherwise)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--start", help="comma-separated units to start from (h<l>_<h>, m<l>_<start>_<stop>)")
    ap.add_argument("--tag", default="", help="suffix of the output names")
    ap.add_argument("--stand-in", choices=["counterfactual", "global"], help="the programs' stand-in form (checker default: counterfactual)")
    ap.add_argument("--objective", default="total", choices=["total", "shared"],
                    help="minimize the score's total, or the total over the experiment families every program shares")
    ap.add_argument("--mlp-view", default="native", choices=["native", "vpd"], help="MLP units: native neuron blocks or VPD subcomponents")
    ap.add_argument("--ranking", type=Path, help="with --mlp-view vpd: measured removal effects of VPD subcomponents (sites -> kl_bits)")
    ap.add_argument("--vpd", type=Path, default=Path.home() / "mpd-data/engine/vpd4l_decomposition",
                    help="with --mlp-view vpd: VPD's decomposition export (the checker's vpd view)")
    ap.add_argument("--grow", type=int, default=2, help="with --mlp-view vpd: growth factor of a VPD unit per move")
    ap.add_argument("--prompt-holdout", type=int, default=0,
                    help="drop every K-th prompt (i %% K == 0) before searching: the prompts the oracle is evaluated on (g-rl)")
    a = ap.parse_args()
    path = a.behavior.expanduser()
    behavior = json.loads(path.read_text())
    model = behavior["model"]
    out = a.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if a.prompt_holdout:
        behavior["prompts"] = [p for i, p in enumerate(behavior["prompts"]) if i % a.prompt_holdout != 0]
        path = out / f"{behavior['id']}.train_prompts.json"
        path.write_text(json.dumps(behavior))
    if a.mlp_view == "vpd":
        global GROW
        GROW = a.grow
        load_ranking(a.ranking)
    views = {"vpd": a.vpd} if a.mlp_view == "vpd" else None
    pool = Pool(model, path, a.workers, a.export, a.stand_in, views, out / f"{behavior['id']}{a.tag}.candidates.jsonl")
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
                start_units = [unit_of(t) for t in a.start.split(",")] if a.start else None
                partial = out / f"{stem}.partial.json"
                found = greedy(pool, model, mode, a.experiments, a.seed, a.min_neurons, log, start_units, a.mlp_view,
                               lambda state: partial.write_text(json.dumps(state, indent=1)), objective_of(a.objective))
                heldout = pool.score([found["source"]], a.experiments, a.heldout_seed)[0]
                found.update(units=[name(u) for u in found["units"]], heldout=heldout, calls=pool.calls - start, stand_in=a.stand_in, checker=Path(str(score.BINARY)).name,
                             experiments=a.experiments, seed=a.seed, heldout_seed=a.heldout_seed)
                log(f"{mode}: {len(found['units'])} units, {found['score']['total_bits']:.6g} bits (held-out seed "
                    f"{heldout['total_bits']:.6g}), {found['calls']} checker calls")
            (out / f"{stem}.json").write_text(json.dumps(found, indent=1))
            e2e.record([e2e.status_line(model, behavior["id"], f"search_{mode}{a.tag}", found["heldout"], a.stand_in)])
    finally:
        pool.close()


if __name__ == "__main__":
    main()
