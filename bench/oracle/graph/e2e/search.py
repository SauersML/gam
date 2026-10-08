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
  prefix    the units of a ranking (--ranking: native heads and neurons, or the attached decomposition's
            subcomponents, e.g. VPD's ranked by its importance, patch_ranking.py), its geometric prefixes,
            then pruning (prune); with subcomponents the VPD view is attached (deletion: unnamed parts
            contribute zero).
Every candidate of a step is scored under the same experiment seed; the final program is rescored
under a held-out seed. Candidates go to --workers checker processes as score_batch requests (each
checker scores its share in parallel threads, RAYON_NUM_THREADS).

  search.py BEHAVIOR.json [--mode addition|removal|both|prefix] [--ranking FILE] [--experiments 16] [--workers 1] [--stand-in counterfactual]
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

# A unit: ("head", layer, head), ("mlp", layer, start, stop) for neurons start..stop-1, or ("sub", layer,
# site, index) for one subcomponent of the attached decomposition (VPD).


def site(unit) -> int:
    """The residual stream a unit reads (design.txt section 5): attention at 2l, MLP at 2l+1."""
    return 2 * unit[1] + (unit[0] == "mlp" or (unit[0] == "sub" and unit[2] in ("c_fc", "down_proj")))


def name(unit) -> str:
    return {"head": "h", "mlp": "m", "sub": "s"}[unit[0]] + "_".join(str(x) for x in unit[1:]) if unit[0] != "head" else f"h{unit[1]}_{unit[2]}"


def unit_of(text: str):
    """The unit a name() denotes: h<l>_<h>, m<l>_<start>_<stop> or s<l>_<matrix>_<index>."""
    if text[0] == "s":
        layer, rest = text[1:].split("_", 1)
        matrix, index = rest.rsplit("_", 1)
        return ("sub", int(layer), matrix, int(index))
    parts = [int(x) for x in text[1:].split("_")]
    return ({"h": "head", "m": "mlp"}[text[0]], *parts)


def ranked_native(path: Path, model: str = "vpd4l", per_number: bool = True) -> list[tuple]:
    """Heads and single MLP neurons ("mlp", layer, i, i + 1), largest counterfactual write change (per opaque
    number, by default) first (vpd_cf_ranking.py's "native" entry)."""
    native = json.loads(Path(path).read_text())["native"]
    units = [(v, ("head", l, h)) for l, row in enumerate(native["heads"]) for h, v in enumerate(row)]
    units += [(v, ("mlp", l, i, i + 1)) for l, row in enumerate(native["neurons"]) for i, v in enumerate(row)]
    if per_number:  # change per opaque number the unit costs (a head's q, k, v, o maps; a neuron's rows)
        s_ = mech.shapes(model)
        cost = {"head": 4 * s_["d_model"] * s_["head_dim"], "mlp": 2 * s_["d_model"] + 1}
        units = [(v / cost[u[0]], u) for v, u in units]
    return [u for v, u in sorted(units, key=lambda x: -x[0])]


def ranked_subcomponents(path: Path) -> list[tuple]:
    """Every VPD subcomponent ("sub", layer, matrix, index), largest measured removal effect first (g-mech's
    measure/vpd_induction_removal.py output: sites -> {"kl_bits": [per subcomponent]})."""
    sites = json.loads(Path(path).read_text())["sites"]
    # a removal scan's "kl_bits", or vpd_cf_ranking.py's "cf_write_change"
    subs = [(kl, ("sub", int(key.split(".")[1]), key.split(".")[-1], i)) for key, v in sites.items()
            for i, kl in enumerate(v.get("kl_bits") or v["cf_write_change"])]
    return [u for kl, u in sorted(subs, key=lambda x: -x[0])]


def piece(unit) -> str:
    if unit[0] == "head":
        return f"L[{unit[1]}].head[{unit[2]}]"
    return f"L[{unit[1]}].mlp[{unit[2]}:{unit[3]}]"


def _slices(indices) -> str:
    """Sorted unique indices in mech's slice syntax: consecutive runs as start:stop."""
    xs, out, i = sorted(set(indices)), [], 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(str(xs[i]) if i == j else f"{xs[i]}:{xs[j] + 1}")
        i = j + 1
    return ", ".join(out)


SITES_OF = {"attn": ("q_proj", "k_proj", "v_proj", "o_proj"), "mlp": ("c_fc", "down_proj")}


def nodes_of(units) -> list[tuple[str, str, int, bool, bool]]:
    """(node name, pieces, residual site, reads the residual, writes it) per node: a head per node; a layer's
    native MLP blocks merged into one node, and a layer's VPD subcomponents ("sub", layer, matrix, index)
    into one node per block (attention or MLP), since every causal edge is declared and a merged node then
    computes the same. A VPD node reads the residual only through c_fc / q, k, v subcomponents and writes
    it only through down_proj / o_proj ones."""
    out, mlp, sub = [], {}, {}
    for u in units:
        if u[0] == "mlp":
            mlp.setdefault(u[1], set()).update(range(u[2], u[3]))
        elif u[0] == "sub":
            block = "mlp" if u[2] in SITES_OF["mlp"] else "attn"
            sub.setdefault((u[1], block), {}).setdefault(u[2], set()).add(u[3])
        else:
            out.append((name(u), piece(u), site(u), True, True))
    for l, idx in mlp.items():
        out.append((f"m{l}", f"L[{l}].mlp[{_slices(idx)}]", 2 * l + 1, True, True))
    for (l, block), sites in sub.items():
        # one part token per subcomponent (a part token is one Python token of the code length)
        pieces = ", ".join(f"<p:{l}.{mech.CODES[m]}.{i}>" for m in SITES_OF[block] if m in sites for i in sorted(sites[m]))
        reads = bool(set(sites) & {"c_fc", "q_proj", "k_proj", "v_proj"})
        writes = bool(set(sites) & {"down_proj", "o_proj"})
        out.append((f"{'va' if block == 'attn' else 'vm'}{l}", pieces, 2 * l + (block == "mlp"), reads, writes))
    return sorted(out, key=lambda n: (n[2], n[0]))


def source(units) -> str:
    """The program declaring `units` as nodes with every causal edge among them listed."""
    nodes = nodes_of(units)
    lines = ['"""Found by greedy search on the score (e2e/search.py)."""', "from mech import node, edges, L, PD, embed, logits"]
    lines += [f"{n[0]} = node({n[1]})" for n in nodes]
    wires = []
    for i, (n, _, s_, reads, _) in enumerate(nodes):
        if reads:
            wires += [f"    {w} >> {n}," for w in ["embed"] + [m[0] for m in nodes[:i] if m[2] < s_ and m[4]]]
    wires += [f"    {w} >> logits," for w in ["embed"] + [n[0] for n in nodes if n[4]]]
    if nodes:
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
                 views: dict | None = None, log_path: Path | None = None, device: str | None = None, necessity: bool = True):
        """views: decomposition views for the checker (score.Checker's); log_path: every scored program is
        appended there as a JSON line {"source", "seed", "experiments", "score"} (search's data); necessity:
        False skips the checker's necessity runs (necessity_error_bits 0)."""
        self.model, self.stand_in, self.log_path, self.necessity = model, stand_in, log_path, necessity
        self.checkers = [score.Checker(model, export, views=views, device=device) for _ in range(workers)]
        for c in self.checkers:
            e2e.load_behavior(c, behavior)
        self.calls = 0

    def score(self, programs: list[str], experiments: int, seed: int) -> list[dict]:
        irs = [e2e.ir_of(p, self.model, self.stand_in) for p in programs]

        def one(k):
            share = list(range(k, len(irs), len(self.checkers)))
            request = {"op": "score_batch", "programs": [irs[i] for i in share], "experiments": experiments,
                       "seed": seed, "routing": "edges", "N": None, "reader_top": 0}
            if not self.necessity:
                request["necessity"] = False
            answer = self.checkers[k].request(request)
            return list(zip(share, answer["scores"]))

        with ThreadPoolExecutor(len(self.checkers)) as ex:
            results = [r for part in ex.map(one, range(min(len(self.checkers), len(irs)))) for r in part]
        self.calls += len(programs)
        scored = [r for _, r in sorted(results, key=lambda x: x[0])]
        if self.log_path is not None:
            with self.log_path.open("a") as f:
                for p, r in zip(programs, scored):
                    f.write(json.dumps({"source": p, "seed": seed, "experiments": experiments, "necessity": self.necessity, "score": r}) + "\n")
        return scored

    def close(self):
        for c in self.checkers:
            c.close()


def all_units(model: str):
    """Every native unit: the heads, and per layer its MLP as one block."""
    s = mech.shapes(model)
    return [("head", l, h) for l in range(s["layers"]) for h in range(s["heads"])] + [("mlp", l, 0, s["d_mlp"]) for l in range(s["layers"])]


def objective_of(kind: str):
    """The quantity search minimizes: the score's total, or (kind "shared") the total with the execution
    error taken over the experiment families every program shares (table.shared)."""
    if kind == "total":
        return lambda r: r["total_bits"] if r.get("valid", True) else float("inf")
    import table
    families = {"shared": table.SHARED, "fit": table.FIT}[kind]
    return lambda r: table.shared(r, families)[1] * r["N"] if r.get("valid", True) else float("inf")


def greedy(pool: Pool, model: str, mode: str, experiments: int, seed: int, min_neurons: int, log, start=None,
           checkpoint=None, objective=None) -> dict:
    """`start`: the units to start from (default: none for addition, every unit for removal)."""
    objective = objective or (lambda r: r["total_bits"])
    full = all_units(model)
    current = list(start) if start is not None else [] if mode == "addition" else list(full)
    # Addition draws from `outside`: the pieces not declared yet, as dyadic blocks (whole units the
    # start does not touch; a start's partial MLP blocks leave nothing outside in that layer).
    touched = {(u[0], u[1]) if u[0] == "mlp" else u for u in current}
    outside = [u for u in full if ((u[0], u[1]) if u[0] == "mlp" else u) not in touched] if mode == "addition" else []
    best = pool.score([source(current)], experiments, seed)[0]
    trajectory = [{"step": 0, "units": [name(u) for u in current], "total_bits": best["total_bits"],
                   "exec_error_bits": best["exec_error_bits"], "complexity_bits": best.get("complexity_bits"), "calls": pool.calls}]
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
        k = min(range(len(moves)), key=lambda i: objective(results[i]))
        log(f"{mode} step {step}: {len(moves)} candidates in {time.time() - t:.0f} s; best {moves[k][2][0]} "
            f"{name(moves[k][2][1])} {objective(results[k]):.6g} vs {objective(best):.6g}")
        candidates = [{"move": [m[2][0], name(m[2][1])], "total_bits": r["total_bits"], "exec_error_bits": r["exec_error_bits"],
                       "complexity_bits": r.get("complexity_bits")} for m, r in zip(moves, results)]
        if objective(results[k]) >= objective(best):
            trajectory.append({"step": step, "stopped": True, "candidates": candidates, "calls": pool.calls})
            break
        current, outside, best = moves[k][0], moves[k][1], results[k]
        trajectory.append({"step": step, "move": [moves[k][2][0], name(moves[k][2][1])], "units": [name(u) for u in current],
                           "total_bits": best["total_bits"], "exec_error_bits": best["exec_error_bits"],
                           "complexity_bits": best.get("complexity_bits"), "calls": pool.calls, "candidates": candidates})
        if checkpoint is not None:  # a job cut off by its time limit keeps every finished step
            checkpoint({"units": [name(u) for u in current], "source": source(current), "score": best, "trajectory": trajectory,
                        "partial": True})
    return {"units": current, "source": source(current), "score": best, "trajectory": trajectory}


def prune(pool: Pool, current: list, best: dict, experiments: int, seed: int, objective, log, max_prune: int = 64,
          cap: int = 32, trajectory: list | None = None, checkpoint=None) -> tuple[list, dict]:
    """Removals while one lowers the objective: of a whole group (a head, a layer's native neurons, a matrix's
    subcomponents) above max_prune units; then of runs of consecutive units of `current` (the ranking's order)
    of n/8 units, n/16, ... down to n/cap (at most about `cap` candidates per step); then, at max_prune units or
    fewer, of one unit at a time."""
    trajectory = trajectory if trajectory is not None else []
    note = lambda: checkpoint and checkpoint({"units": [name(u) for u in current], "source": source(current), "score": best,
                                              "trajectory": trajectory, "partial": True})
    if len(current) > max_prune:  # coarse pruning: drop a whole group (a head, a layer's native neurons, a matrix's subcomponents)
        group = lambda u: u if u[0] == "head" else (u[0], u[1]) if u[0] == "mlp" else (u[0], u[1], u[2])
        while True:
            groups = sorted({group(u) for u in current}, key=str)
            if len(groups) < 2:
                break
            moves = [[u for u in current if group(u) != g] for g in groups]
            rs = pool.score([source(m) for m in moves], experiments, seed)
            i = min(range(len(moves)), key=lambda i: objective(rs[i]))
            log(f"group prune: {len(moves)} candidates; best drops {groups[i]} {objective(rs[i]):.6g} vs {objective(best):.6g}")
            if objective(rs[i]) >= objective(best):
                break
            trajectory.append({"drop_group": str(groups[i]), "score": rs[i]})
            current, best = moves[i], rs[i]
            note()
    runs = 8
    while len(current) > max_prune and runs <= cap:
        size = max(1, len(current) // runs)
        moves = [current[:i] + current[i + size:] for i in range(0, len(current), size)]
        rs = pool.score([source(m) for m in moves], experiments, seed)
        i = min(range(len(moves)), key=lambda i: objective(rs[i]))
        log(f"run prune: {len(moves)} runs of {size}; best drops units {i * size}-{i * size + size - 1} {objective(rs[i]):.6g} vs {objective(best):.6g}")
        if objective(rs[i]) < objective(best):
            trajectory.append({"drop_run": [name(u) for u in current[i * size:i * size + size]], "score": rs[i]})
            current, best = moves[i], rs[i]
            note()
        elif size == 1:
            break
        else:
            runs *= 2
    while 1 < len(current) <= max_prune:
        moves = [[v for v in current if v != u] for u in current]
        rs = pool.score([source(m) for m in moves], experiments, seed)
        i = min(range(len(moves)), key=lambda i: objective(rs[i]))
        log(f"prune: {len(moves)} candidates; best removes {name(current[i])} {objective(rs[i]):.6g} vs {objective(best):.6g}")
        if objective(rs[i]) >= objective(best):
            break
        trajectory.append({"remove": name(current[i]), "score": rs[i]})
        current, best = moves[i], rs[i]
        note()
    return current, best


def prefix_search(pool: Pool, model: str, experiments: int, seed: int, block: int, log, rank_experiments: int = 0,
                  checkpoint=None, objective=None, units=None, ranked=None, max_prune: int = 64, growth: float = 1.5,
                  cap: int = 32) -> dict:
    """Measured ranking, then prefixes, then pruning, all exact through the checker:
    1. every unit alone (heads, MLP neuron blocks of `block`) as a one-node program: under counterfactual
       stand-ins this is the unit's activation patch from x into x', and the drop in KL on the clean and
       counterfactual prompts from the empty program's is its measured effect;
    2. the programs of the k most effective units for k = 1, 2, 3, 4, 6, 8, 12, ... (pieces that pay only
       together enter together, which one-piece-at-a-time addition misses), the best kept;
    3. pruning (prune) until no removal lowers the objective."""
    objective = objective or (lambda r: r["total_bits"])
    s_ = mech.shapes(model)
    units = units or ([("head", l, h) for l in range(s_["layers"]) for h in range(s_["heads"])]
                      + [("mlp", l, i, min(i + block, s_["d_mlp"])) for l in range(s_["layers"]) for i in range(0, s_["d_mlp"], block)])
    import table
    if ranked is None:
        empty, *alone = pool.score([source([])] + [source([u]) for u in units], rank_experiments, seed)
        # The effect on the clean and counterfactual prompts only: every program has those two experiments,
        # while the rest of a program's draws depend on what it declares.
        kl = lambda r: table.shared(r, ("clean", "counterfactual"))[0] * r["N"]
        effect = {u: kl(empty) - kl(r) for u, r in zip(units, alone)}
        ranked = sorted(units, key=lambda u: -effect[u])
        log(f"ranked {len(units)} units by their patch; top: " + ", ".join(f"{name(u)} {effect[u] / empty['N']:.3f}" for u in ranked[:8]))
    else:  # a ranking measured elsewhere (a removal scan): its order only
        empty = pool.score([source([])], experiments, seed)[0]
        effect = {u: float(len(ranked) - i) for i, u in enumerate(ranked)}
        log(f"{len(ranked)} units ranked by a given measurement; top: " + ", ".join(name(u) for u in ranked[:8]))
    ks, k = [], 1
    while k < len(ranked):
        ks.append(k)
        k = max(k + 1, int(k * growth))
    ks.append(len(ranked))
    # k = 0, the empty program, competes too: no prefix may be worse than declaring nothing
    ks = [0] + ks
    results = pool.score([source(ranked[:k]) for k in ks], experiments, seed)
    j = min(range(len(ks)), key=lambda i: objective(results[i]))
    log("prefixes: " + ", ".join(f"{k}:{objective(r) / r['N']:.3f}" for k, r in zip(ks, results)) + f"; best k = {ks[j]}")
    current, best = ranked[:ks[j]], results[j]
    trajectory = [{"ranking": [[name(u), effect[u]] for u in ranked], "empty": empty, "prefixes": [[k, r] for k, r in zip(ks, results)]}]
    if checkpoint:
        checkpoint({"units": [name(u) for u in current], "source": source(current), "score": best, "trajectory": trajectory, "partial": True})
    if len(current) > max_prune:  # too many units to drop one by one: refine k between the best prefix's neighbours
        lo, hi = ks[max(j - 1, 0)], ks[min(j + 1, len(ks) - 1)]
        fine = sorted({lo + (hi - lo) * i // 8 for i in range(9)} - {ks[j]} - {0})
        rs = pool.score([source(ranked[:k]) for k in fine], experiments, seed)
        log("refine: " + ", ".join(f"{k}:{objective(r) / r['N']:.3f}" for k, r in zip(fine, rs)))
        trajectory.append({"refine": [[k, r] for k, r in zip(fine, rs)]})
        i = min(range(len(fine)), key=lambda i: objective(rs[i]))
        if objective(rs[i]) < objective(best):
            current, best = ranked[:fine[i]], rs[i]
    current, best = prune(pool, current, best, experiments, seed, objective, log, max_prune, cap, trajectory, checkpoint)
    return {"units": current, "source": source(current), "score": best, "trajectory": trajectory}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("behavior", type=Path)
    ap.add_argument("--mode", default="both", choices=["addition", "removal", "both", "prefix"])
    ap.add_argument("--block", type=int, default=96, help="prefix mode: MLP neurons per unit")
    ap.add_argument("--prefix-growth", type=float, default=1.5, help="prefix mode: ratio between successive prefix sizes")
    ap.add_argument("--max-prune", type=int, default=64, help="prefix mode: prune one unit at a time only up to this many units (else refine k)")
    ap.add_argument("--cap", type=int, default=32, help="prefix mode: the most candidates of a run-pruning step")
    ap.add_argument("--search-necessity", action="store_true",
                    help="prefix mode: score necessity throughout (default: only the final pruning and the result, from the "
                         "program found without it)")
    ap.add_argument("--device", help="the checker's device (gpu: the single-precision device path)")
    ap.add_argument("--max-units", type=int, default=8192, help="prefix mode with a ranking: the top ranked units considered")
    ap.add_argument("--rank-experiments", type=int, default=0, help="prefix mode: draws beyond clean and counterfactual per one-unit ranking program")
    ap.add_argument("--experiments", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heldout-seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--min-neurons", type=int, default=384)
    ap.add_argument("--export", type=Path, help="the model's export directory (score.py's default otherwise)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--start", help="comma-separated units to start from (h<l>_<h>, m<l>_<start>_<stop>)")
    ap.add_argument("--tag", default="", help="suffix of the output names")
    ap.add_argument("--stand-in", choices=["counterfactual"], help="the programs' stand-in form (counterfactual, the only one since the average stand-ins were deleted)")
    ap.add_argument("--objective", default="total", choices=["total", "shared", "fit"],
                    help="minimize the score's total, the total over the families every program shares, or over the fit "
                         "families (table.FIT; the held-out ones, table.HELDOUT, stay for reporting)")
    ap.add_argument("--ranking", type=Path, help="prefix mode: the units' order, {\"mixed\": [[unit name, value], ...]} (largest "
                                                 "first; patch_ranking.py), {\"native\": ...} (ranked_native) or a VPD removal scan "
                                                 "{\"sites\": ...} (ranked_subcomponents)")
    ap.add_argument("--vpd", type=Path, default=Path.home() / "mpd-data/engine/vpd4l_decomposition",
                    help="VPD's decomposition export (the checker's vpd view, attached when the ranking names subcomponents)")
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
    data = None if a.ranking is None else json.loads(a.ranking.read_text())
    subs = data is not None and ("sites" in data or any(n.startswith("s") for n, _ in data.get("mixed", [])))
    views = {"vpd": a.vpd} if subs else None
    pool = Pool(model, path, a.workers, a.export, a.stand_in, views, out / f"{behavior['id']}{a.tag}.candidates.jsonl", a.device,
                necessity=a.search_necessity or a.mode != "prefix")
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
                save = lambda state: partial.write_text(json.dumps(state, indent=1))
                if mode == "prefix":
                    ranked = None if data is None else ([unit_of(n) for n, _ in data["mixed"]] if "mixed" in data
                                                        else ranked_subcomponents(a.ranking) if "sites" in data
                                                        else ranked_native(a.ranking, model))[: a.max_units]
                    objective = objective_of(a.objective)
                    found = prefix_search(pool, model, a.experiments, a.seed, a.block, log, a.rank_experiments, save,
                                          objective, ranked=ranked, max_prune=a.max_prune, growth=a.prefix_growth, cap=a.cap)
                    if not pool.necessity:  # the final pruning with necessity scored, from the program found without it
                        pool.necessity = True
                        best = pool.score([found["source"]], a.experiments, a.seed)[0]
                        log(f"with necessity: {len(found['units'])} units, {objective(best):.6g} (necessity {best.get('necessity_error_bits', 0):.6g} bits)")
                        units, best = prune(pool, found["units"], best, a.experiments, a.seed, objective, log, a.max_prune, a.cap,
                                            found["trajectory"], save)
                        found.update(units=units, source=source(units), score=best)
                else:
                    found = greedy(pool, model, mode, a.experiments, a.seed, a.min_neurons, log, start_units, save,
                                   objective_of(a.objective))
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
