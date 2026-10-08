"""Subcomponents ranked by the contrast they carry, measured (#2951 graph oracle, the lead's 10-08 plan), and the best
answer at each size k.

Under counterfactual stand-ins a part's worth is the prompt/counterfactual contrast that flows through it. VPD's
importance ranks parts active on the prompt, mostly as active on the counterfactual. Here each chunk of a site's
subcomponents is measured by its removal: the program naming every subcomponent but the chunk (the chunk at its
counterfactual values, everything else on the prompt), whose execution error is the contrast the chunk carries,
mediated paths included, its partners in the block (c_fc's down_proj, a head's q/k with v/o) all on. Chunks start
at --chunk subcomponents per site; the --keep largest are split in half and measured again, down to --leaf; the
ranking lists the leaf chunks by effect, then the larger chunks, members by VPD importance. Necessity is off while
ranking (it scores the program, not the chunk), and a removal is scored on the clean and counterfactual prompts
alone (--rank-experiments 0) unless asked for more.

Then the k-curve: per k in --ks, align(answer, the ranking's first k) in the family algorithm (closed as
teacher_run.closed makes it valid; held-out behaviors, which have no algorithm, as node programs) scored in full,
with the share of the signal it reproduces (1 - execution error / the program without parts') and its total
against the program without parts.

  contrast.py BEHAVIOR_ID... [--experiments 16] [--device gpu] [--out ~/mpd-data/graph_oracle/runs/kcurve]
writes OUT/<behavior>.json {"ranking", "chunks" (every measured chunk), "curve", "empty", settings} and
OUT/rankings/<behavior>.json (teacher_run --rankings format).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import mech  # noqa: E402
import score as score_module  # noqa: E402
import search  # noqa: E402
import teacher  # noqa: E402
import teacher_run  # noqa: E402

DATA = Path.home() / "mpd-data/graph_oracle"
BLOCKS = {"attn": ("q_proj", "k_proj", "v_proj", "o_proj"), "mlp": ("c_fc", "down_proj")}


def ranges(indices: list[int]) -> str:
    """Sorted indices as PD slice text: 0:128, 300, 302:310."""
    out, i = [], 0
    while i < len(indices):
        j = i
        while j + 1 < len(indices) and indices[j + 1] == indices[j] + 1:
            j += 1
        out.append(str(indices[i]) if i == j else f"{indices[i]}:{indices[j] + 1}")
        i = j + 1
    return ", ".join(out)


def nodes_program(kept: dict) -> str:
    """The node program naming the subcomponents `kept` ({(layer, site): indices}): a node per (layer, block), and every
    causal edge from embed and each earlier residual writer into each node that reads, and from every writer into the
    logits."""
    nodes = []
    for l in sorted({l for l, _ in kept}):
        for block, sites in BLOCKS.items():
            pieces = [f"PD[{l}].{s}[{ranges(sorted(kept[(l, s)]))}]" for s in sites if kept.get((l, s))]
            if pieces:
                present = {s for s in sites if kept.get((l, s))}
                nodes.append((f"{'va' if block == 'attn' else 'vm'}{l}", ", ".join(pieces), bool(present - {"o_proj", "down_proj"}),
                              bool(present & {"o_proj", "down_proj"})))
    lines = ["from mech import node, edges, L, PD, embed, logits"] + [f"{n} = node({p})" for n, p, _, _ in nodes]
    wires = [f"    {w} >> {n}," for k, (n, _, reads, _) in enumerate(nodes) if reads
             for w in ["embed"] + [m for m, _, _, writes in nodes[:k] if writes]]
    wires += [f"    {w} >> logits," for w in ["embed"] + [n for n, _, _, writes in nodes if writes]]
    return "\n".join(lines + (["edges("] + wires + [")"] if nodes else [])) + "\n"


def kept_of(chunks) -> dict:
    out: dict = {}
    for l, s, i, j in chunks:
        out.setdefault((l, s), set()).update(range(i, j))
    return out


def without(sizes: dict, chunk: tuple) -> str:
    """The node program naming every subcomponent but `chunk` (layer, site, start, stop)."""
    return nodes_program(kept_of([c for c in ((l, s, 0, n) for (l, s), n in sizes.items())
                                  for c in ([c] if (c[0], c[1]) != chunk[:2] else [(c[0], c[1], 0, chunk[2]), (c[0], c[1], chunk[3], c[3])])
                                  if c[3] > c[2]]))


def removals(checker, programs: list[str], a) -> list[float]:
    """Each program's execution error on the clean and counterfactual prompts alone (--rank-experiments), necessity off."""
    results = []
    for k in range(0, len(programs), a.batch):
        results += checker.score_batch(programs[k:k + a.batch], experiments=a.rank_experiments, seed=0, reader=False,
                                       stand_in="counterfactual", options={"necessity": False})
    return [r["exec_error_bits"] if r.get("valid", True) else float("inf") for r in results]


def prune(b: str, checker, sizes: dict, a, log) -> tuple[dict, list]:
    """Iterative pruning from every subcomponent: each round measures every chunk's removal from the current set (the
    error of the set without it), drops the chunks of least effect until a share --drop of the parts is gone (or the
    next k remains), and splits the rest in half; the set at each k in --ks ({k: units}) and every round's log."""
    chunks = [(l, s, i, min(i + a.chunk, n)) for (l, s), n in sorted(sizes.items()) for i in range(0, n, a.chunk)]
    wanted, sets, rounds = sorted(a.ks, reverse=True), {}, []
    while wanted:
        parts = sum(j - i for _, _, i, j in chunks)
        errors = removals(checker, [nodes_program(kept_of([d for d in chunks if d != c])) for c in chunks], a)
        effect = dict(zip(chunks, errors))
        target = max(int(parts * (1 - a.drop)), wanted[0])
        kept, total = [], parts
        for c in sorted(chunks, key=lambda c: effect[c]):
            if total - (c[3] - c[2]) >= target and total > wanted[-1]:
                total -= c[3] - c[2]
            else:
                kept.append(c)
        chunks = kept
        current = removals(checker, [nodes_program(kept_of(chunks))], a)[0]
        rounds.append({"parts": total, "chunks": len(chunks), "error_bits": current})
        log(f"{total} parts in {len(chunks)} chunks, error {current:.5g} bits")
        while wanted and total <= wanted[0]:
            sets[wanted.pop(0)] = [("sub", l, s, i) for l, s, i0, j in chunks for i in range(i0, j)]
        chunks = [h for (l, s, i, j) in chunks for h in (((l, s, i, (i + j) // 2), (l, s, (i + j) // 2, j)) if j - i > 1 else ((l, s, i, j),))]
    return sets, rounds


def rank(b: str, checker, sizes: dict, importance: dict, a, log) -> tuple[list, list]:
    """(the ranking as units ("sub", layer, site, index) with each one's chunk effect, every measured chunk)."""
    def measure(chunks):
        return [e if e != float("inf") else -1.0 for e in removals(checker, [without(sizes, c) for c in chunks], a)]  # invalid: last

    chunks = [(l, s, i, min(i + a.chunk, n)) for (l, s), n in sorted(sizes.items()) for i in range(0, n, a.chunk)]
    effect = dict(zip(chunks, measure(chunks)))
    log(f"{len(chunks)} chunks of {a.chunk}; largest effects " + ", ".join(f"{c[0]}.{c[1]}[{c[2]}:{c[3]}] {effect[c]:.4g}"
        for c in sorted(chunks, key=lambda c: -effect[c])[:6]))
    open_ = sorted(chunks, key=lambda c: -effect[c])
    size = a.chunk
    while size > a.leaf:
        size //= 2
        top = [c for c in open_ if c[3] - c[2] > size][: a.keep]
        halves = [h for (l, s, i, j) in top for h in ((l, s, i, min(i + size, j)), (l, s, min(i + size, j), j)) if h[3] > h[2]]
        effect.update(zip(halves, measure(halves)))
        split = set(top)
        open_ = sorted([c for c in open_ if c not in split] + halves, key=lambda c: -effect[c])
        log(f"chunks of {size}: {len(halves)} measured; largest " + ", ".join(f"{c[0]}.{c[1]}[{c[2]}:{c[3]}] {effect[c]:.4g}"
            for c in sorted(halves, key=lambda c: -effect[c])[:6]))
    leaves = sorted((c for c in effect if c[3] - c[2] <= a.leaf), key=lambda c: -effect[c])
    rest = sorted((c for c in open_ if c[3] - c[2] > a.leaf), key=lambda c: -effect[c])
    ranking = []
    for c in leaves + rest:
        members = sorted(range(c[2], c[3]), key=lambda i: -importance[(c[0], c[1])][i])
        ranking += [(("sub", c[0], c[1], i), effect[c]) for i in members]
    seen, out = set(), []
    for u, e in ranking:
        if u not in seen:
            seen.add(u)
            out.append((u, e))
    return out, [{"chunk": list(c), "effect_bits": effect[c]} for c in sorted(effect, key=lambda c: -effect[c])]


def writing(units: list) -> list:
    """`units` less the parts of blocks with no residual writer among them: mech rejects an alignment whose block writes
    no residual stream, and under counterfactual stand-ins such parts reach nothing (their block's unnamed writers keep
    their counterfactual values)."""
    blocks = {(u[1], teacher_run.BLOCK[u[2]]) for u in units if teacher_run.WRITER[teacher_run.BLOCK[u[2]]] == u[2]}
    return [u for u in units if (u[1], teacher_run.BLOCK[u[2]]) in blocks]


def curve(b: str, behavior: dict, checker, chosen: dict, a) -> tuple[dict, list]:
    """The program without parts' score and, per k, the answer aligning chosen[k] (units) and its score."""
    try:
        algorithm = teacher.algorithm_of(behavior)
    except ValueError:
        algorithm = None  # a held-out family: node programs

    def program(chosen):
        if algorithm is None:
            return search.source(chosen)
        return algorithm.rstrip() + "\n\n\n" + (f"align(answer, {', '.join(map(teacher_run.token_of, chosen))})\n" if chosen else "")

    ks = sorted(chosen)
    sets = [[]] + [chosen[k] for k in ks]
    results = []
    for k in range(0, len(sets), a.batch):
        results += checker.score_batch([program(s) for s in sets[k:k + a.batch]], experiments=a.experiments, seed=1,
                                       reader=False, stand_in="counterfactual")
    terms = ("total_bits", "exec_error_bits", "necessity_error_bits", "alignment_error_bits", "complexity_bits", "structure_bits",
             "code_bits", "N", "parts", "valid")
    empty = {t: results[0].get(t) for t in terms}
    rows = []
    for k, s, r in zip(ks, sets[1:], results[1:]):
        rows.append({"k": k, "parts": len(s), "units": [search.name(u) for u in s], "source_format": "answer" if algorithm else "nodes",
                     "reproduced": 1 - r["exec_error_bits"] / empty["exec_error_bits"] if empty["exec_error_bits"] else None,
                     "beats_empty": r["total_bits"] < empty["total_bits"], "score": {t: r.get(t) for t in terms},
                     "error": r.get("error")})
    return empty, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("behaviors", nargs="+")
    ap.add_argument("--behaviors-dir", type=Path, default=DATA / "behaviors/vpd4l")
    ap.add_argument("--export", type=Path, default=Path.home() / "mpd-data/engine/vpd4l")
    ap.add_argument("--vpd", type=Path, default=Path.home() / "mpd-data/engine/vpd4l_decomposition")
    ap.add_argument("--importance", type=Path, default=DATA / "experiments/importance", help="mpd_vpd_importance_2951's tables (site sizes, order within a chunk)")
    ap.add_argument("--device")
    ap.add_argument("--experiments", type=int, default=16)
    ap.add_argument("--method", choices=["prune", "removal"], default="prune", help="prune: iterative pruning from every "
                    "subcomponent, removals re-measured in the current set each round; removal: one ranking by removal "
                    "from the full model, split down to --leaf, its prefixes")
    ap.add_argument("--drop", type=float, default=0.5, help="prune: the share of the parts dropped per round")
    ap.add_argument("--rank-experiments", type=int, default=0, help="experiments drawn per chunk removal (0: the clean and "
                    "counterfactual prompts alone, the contrast itself)")
    ap.add_argument("--chunk", type=int, default=256, help="subcomponents per first chunk")
    ap.add_argument("--leaf", type=int, default=8, help="the chunk size splitting stops at")
    ap.add_argument("--keep", type=int, default=32, help="chunks split per level (largest effect first)")
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--batch", type=int, default=3, help="programs per checker request")
    ap.add_argument("--out", type=Path, default=DATA / "runs/kcurve")
    a = ap.parse_args()
    (a.out / "rankings").mkdir(parents=True, exist_ok=True)
    for b in a.behaviors:
        t0 = time.time()
        path = a.behaviors_dir / f"{b}.json"
        behavior = json.loads(path.read_text())
        table = json.loads((a.importance / f"{b}.json").read_text())["sites"]
        importance = {(site["layer"], key.split(".")[-1]): site["mean"] for key, site in table.items()}
        sizes = {k: len(v) for k, v in importance.items()}
        log = lambda m: print(f"{b}: {m}", flush=True)  # noqa: E731
        with score_module.Checker(behavior["model"], export=a.export, views={"vpd": a.vpd}, device=a.device) as checker:
            checker.behavior(path)
            if a.method == "prune":
                chosen, rounds = prune(b, checker, sizes, a, log)
                ranked, chunks = [], rounds
            else:
                ranked, chunks = rank(b, checker, sizes, importance, a, log)
                units = [u for u, _ in ranked]
                chosen = {k: teacher_run.closed(units[:k], units) for k in a.ks}
            if behavior.get("split") == "train":
                chosen = {k: writing(v) for k, v in chosen.items()}  # an answer mech accepts
            empty, rows = curve(b, behavior, checker, chosen, a)
        for r in rows:
            log(f"k={r['k']} ({r['parts']} parts): reproduced {r['reproduced']:.1%}, total {r['score']['total_bits']:.6g} vs empty "
                f"{empty['total_bits']:.6g} (exec {r['score']['exec_error_bits']:.4g}, necessity {r['score']['necessity_error_bits']:.4g}, "
                f"alignment {r['score'].get('alignment_error_bits') or 0:.4g}, complexity {r['score']['complexity_bits']:.4g})")
        record = {"behavior": b, "method": a.method, "drop": a.drop, "semantics": "counterfactual", "experiments": a.experiments, "rank_experiments": a.rank_experiments,
                  "chunk": a.chunk, "leaf": a.leaf,
                  "keep": a.keep, "checker": str(score_module.BINARY), "empty": empty, "curve": rows,
                  "ranking": [[search.name(u), e] for u, e in ranked[:2048]], "chunks": chunks, "seconds": round(time.time() - t0)}
        (a.out / f"{b}.{a.method}{'' if a.drop == 0.5 else f'_drop{a.drop}'}.json").write_text(json.dumps(record, indent=1))
        if ranked:
            (a.out / "rankings" / f"{b}.json").write_text(json.dumps({"behavior": b, "source": f"{a.out / b}.json (measured contrast by chunk removal)",
                                                                  "mixed": [[search.name(u), e] for u, e in ranked]}))
        log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
