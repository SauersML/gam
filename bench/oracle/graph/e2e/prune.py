"""The subcomponent search (#2951 graph oracle): refine the whole model's VPD subcomponents down to single subcomponents,
keeping the set at each size k. The search behind the teacher answers.

Every error is the execution error of the explanation naming the set (one output group, e2e/explain.ir; everything
left out runs on the changed prompt), scored on the behavior's first --search-prompts prompts and their changed
prompts alone (--rank-experiments 0) with necessity off; the signal is the error of naming nothing. Removing chunks from the whole model by their single
removals fails here: the difference between prompt and changed prompt travels many parallel paths, so each chunk's
removal alone costs little and their joint removal costs everything. Growing from nothing named fails too: what is
left out runs on the changed prompt, so a named block reaches the output only directly until a whole path from the
input to the output is named, and no single block or matrix lowers the error. So, from the whole model (the eight
blocks, one layer's attention or MLP each):

  refine  split every piece in half, measure each half's removal, and drop the halves of least effect, as many as
          can go together for at most --tol of the signal more error (found by bisection on their order); repeat
          down to single subcomponents, then keep dropping the least costly one at a time until the smallest k.

The set at each k in --ks is the first set of at most k subcomponents, and "knee" the set where refining stopped
within the tolerance. Each kept set is then scored in full as an explanation (--experiments, seeds 0 and 1):
reproduces and removes against nothing named.

  prune.py BEHAVIOR_ID... [--behaviors-dir ~/mpd-data/graph_oracle/behaviors_v3/vpd4l] [--out runs/prune_v4]
writes OUT/<behavior>.json {"rounds", "sets": {k: [part token, ...]}, "curve": [{"k", "score", "shares"}]}.
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

import atlas  # noqa: E402
import explain  # noqa: E402
import mech  # noqa: E402
import score as score_module  # noqa: E402

DATA = Path.home() / "mpd-data/graph_oracle"
TERMS = ("total_bits", "exec_error_bits", "necessity_error_bits", "alignment_error_bits", "complexity_bits", "N", "parts", "valid")


def units(chunks) -> list[tuple]:
    return [u for c in chunks for u in c]


def errors(checker, sets: list[list[tuple]], a) -> list[float]:
    """Each set's execution error on the clean and changed prompts alone, necessity off."""
    out = []
    for k in range(0, len(sets), a.batch):
        out += checker.score_batch([explain.ir(units(s)) for s in sets[k:k + a.batch]], experiments=a.rank_experiments, seed=0,
                                   options={"necessity": False})
    return [r["exec_error_bits"] if r.get("valid", True) else float("inf") for r in out]


def blocks(model: str) -> list[tuple]:
    """Every subcomponent, one tuple per block (layer, attention or MLP), in matrix order."""
    out = {}
    for l, row in enumerate(mech.shapes(model)["views"]["vpd"]):
        for site, n in sorted(row.items()):
            out.setdefault((l, site in ("c_fc", "down_proj")), []).extend((l, site, i) for i in range(n))
    return [tuple(v) for _, v in sorted(out.items())]


def refine(checker, chunks: list[tuple], error: float, signal: float, a, log) -> tuple[dict, list]:
    """The set at each k and "knee" ({k: units}) and every round's (subcomponents, chunks, error)."""
    wanted, sets, rounds = sorted(a.ks, reverse=True), {}, []
    within = True

    def keep(chunks, error):
        total = len(units(chunks))
        rounds.append({"parts": total, "chunks": len(chunks), "error_bits": error})
        while wanted and total <= wanted[0]:
            sets[wanted.pop(0)] = units(chunks)

    keep(chunks, error)
    while wanted and chunks:
        chunks = [h for c in chunks for h in ((c[:len(c) // 2], c[len(c) // 2:]) if len(c) > 1 else (c,))]
        effect = dict(zip(chunks, errors(checker, [[d for d in chunks if d != c] for c in chunks], a)))
        order = sorted(chunks, key=effect.__getitem__)
        budget = error + a.tol * signal
        lo, hi, found = 0, len(order) - 1, {}  # the most of `order`'s first pieces that go together within the budget
        while lo < hi:
            m = (lo + hi + 1) // 2
            found[m] = errors(checker, [order[m:]], a)[0]
            lo, hi = (m, hi) if found[m] <= budget else (lo, m - 1)
        singles = all(len(c) == 1 for c in chunks)
        if lo == 0 and singles:  # nothing goes within the tolerance: past the knee, the least costly piece at a time
            if within:
                sets["knee"], within = units(chunks), False
            lo, found[1] = 1, effect[order[0]]
        if lo:
            chunks, error = order[lo:], found[lo]
        keep(chunks, error)
        log(f"refine: {len(units(chunks))} subcomponents in {len(chunks)} pieces, error {error:.5g} bits ({1 - error / signal:.1%} reproduced)")
    if within:
        sets["knee"] = units(chunks)
    return sets, rounds


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("behaviors", nargs="+")
    ap.add_argument("--behaviors-dir", type=Path, default=DATA / "behaviors_v3/vpd4l")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--experiments", type=int, default=64)
    ap.add_argument("--rank-experiments", type=int, default=0)
    ap.add_argument("--search-prompts", type=int, default=24, help="the prompts the search scores on (the kept sets are scored on all)")
    ap.add_argument("--tol", type=float, default=0.01, help="the share of the signal a refining round may lose")
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--candidates", type=int, default=0, help="refine the atlas's first N subcomponents of the behavior instead, one per piece")
    ap.add_argument("--out", type=Path, default=DATA / "runs/prune_v4")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for b in a.behaviors:
        t0 = time.time()
        path = a.behaviors_dir / f"{b}.json"
        behavior = json.loads(path.read_text())
        if a.candidates:
            chunks = [(u,) for u in explain.units_of(" ".join(atlas.ranking(atlas.table(behavior))[:a.candidates]))]
        else:
            chunks = blocks(behavior["model"])
        log = lambda m: print(f"{b}: {m}", flush=True)  # noqa: E731
        searched = a.out / "search" / f"{b}.json"  # the prompts the search scores on
        searched.parent.mkdir(exist_ok=True)
        searched.write_text(json.dumps({**behavior, "prompts": behavior["prompts"][:a.search_prompts]}))
        with score_module.Checker(behavior["model"], device=a.device) as checker:
            checker.behavior(searched)
            signal, error = errors(checker, [[], chunks], a)
            sets, rounds = refine(checker, chunks, error, signal, a, log)
            checker.behavior(path)
            curve = []
            for seed in (0, 1):
                empty = checker.score_batch([explain.ir([])], experiments=a.experiments, seed=seed)[0]
                for k, s in sets.items():
                    r = checker.score_batch([explain.ir(s)], experiments=a.experiments, seed=seed)[0]
                    curve.append({"k": k, "seed": seed, "score": {t: r.get(t) for t in TERMS},
                                  "shares": {"reproduces": 1 - r["exec_error_bits"] / empty["exec_error_bits"],
                                             "removes": 1 - r["necessity_error_bits"] / empty["necessity_error_bits"]}})
                    log(f"{k} ({len(s)}) seed {seed}: total {r['total_bits']:.6g}, reproduces {curve[-1]['shares']['reproduces']:.1%}, "
                        f"removes {curve[-1]['shares']['removes']:.1%}")
        (a.out / f"{b}.json").write_text(json.dumps({"behavior": b, "signal_bits": signal, "rounds": rounds,
                                                    "sets": {k: [explain.token(u) for u in v] for k, v in sets.items()},
                                                    "curve": curve, "experiments": a.experiments, "checker": str(score_module.BINARY),
                                                    "seconds": round(time.time() - t0)}, indent=1))
        log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
