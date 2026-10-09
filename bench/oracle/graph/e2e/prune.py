"""The pruning search (#2951 graph oracle): from every VPD subcomponent, repeatedly drop the chunks whose removal costs
least, and keep the set at each size k. The search that found the teacher answers' output groups.

Each round measures every chunk's removal from the current set: the error of the explanation naming the set without
the chunk (one output group, e2e/explain.ir; everything left out runs on the changed prompt), scored on the clean and
changed prompts alone (--rank-experiments 0) with necessity off. The chunks of least effect are dropped until a share
--drop of the subcomponents is gone (or the next k remains), and the rest are split in half. Then each kept set is
scored in full as an explanation (--experiments, seeds 0 and 1): reproduces and removes against nothing named.

  prune.py BEHAVIOR_ID... [--behaviors-dir ~/mpd-data/graph_oracle/behaviors_v3/vpd4l] [--out runs/prune_v4]
writes OUT/<behavior>.json {"rounds", "sets": {k: [part token, ...]}, "curve": [{"k", "score", "shares"}]} (default runs/prune_v4).
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


def prune(checker, start: list[tuple], a, log) -> tuple[dict, list]:
    """The kept set at each k in --ks ({k: units}) and every round's (subcomponents, chunks, error), from the units
    `start` (in chunks of --chunk consecutive units of one matrix)."""
    chunks = []
    for u in start:
        if chunks and len(chunks[-1]) < a.chunk and chunks[-1][-1][:2] == u[:2]:
            chunks[-1] = chunks[-1] + (u,)
        else:
            chunks.append((u,))
    wanted, sets, rounds = sorted(a.ks, reverse=True), {}, []
    while wanted:
        parts = len(units(chunks))
        effect = dict(zip(chunks, errors(checker, [[d for d in chunks if d != c] for c in chunks], a)))
        target = max(int(parts * (1 - a.drop)), wanted[0])
        kept, total = [], parts
        for c in sorted(chunks, key=lambda c: effect[c]):
            if total - len(c) >= target and total > wanted[-1]:
                total -= len(c)
            else:
                kept.append(c)
        chunks = kept
        rounds.append({"parts": total, "chunks": len(chunks), "error_bits": errors(checker, [chunks], a)[0]})
        log(f"{total} subcomponents in {len(chunks)} chunks, error {rounds[-1]['error_bits']:.5g} bits")
        while wanted and total <= wanted[0]:
            sets[wanted.pop(0)] = units(chunks)
        chunks = [h for c in chunks for h in ((c[:len(c) // 2], c[len(c) // 2:]) if len(c) > 1 else (c,))]
    return sets, rounds


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("behaviors", nargs="+")
    ap.add_argument("--behaviors-dir", type=Path, default=DATA / "behaviors_v3/vpd4l")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--experiments", type=int, default=64)
    ap.add_argument("--rank-experiments", type=int, default=0)
    ap.add_argument("--drop", type=float, default=0.5, help="the share of the subcomponents dropped per round")
    ap.add_argument("--chunk", type=int, default=256, help="subcomponents per first chunk")
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--candidates", type=int, default=0, help="search only the atlas's first N subcomponents of the behavior")
    ap.add_argument("--out", type=Path, default=DATA / "runs/prune_v4")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for b in a.behaviors:
        t0 = time.time()
        path = a.behaviors_dir / f"{b}.json"
        behavior = json.loads(path.read_text())
        if a.candidates:  # the oracle's candidate list (atlas.py), in matrix order
            start = explain.order(explain.units_of(" ".join(atlas.ranking(atlas.scores(behavior))[:a.candidates])))
        else:
            start = [(l, site, i) for l, row in enumerate(mech.shapes(behavior["model"])["views"]["vpd"]) for site, n in sorted(row.items())
                     for i in range(n)]
        log = lambda m: print(f"{b}: {m}", flush=True)  # noqa: E731
        with score_module.Checker(behavior["model"], device=a.device) as checker:
            checker.behavior(path)
            sets, rounds = prune(checker, start, a, log)
            curve = []
            for seed in (0, 1):
                empty = checker.score_batch([explain.ir([])], experiments=a.experiments, seed=seed)[0]
                for k in sorted(sets):
                    r = checker.score_batch([explain.ir(sets[k])], experiments=a.experiments, seed=seed)[0]
                    curve.append({"k": k, "seed": seed, "score": {t: r.get(t) for t in TERMS},
                                  "shares": {"reproduces": 1 - r["exec_error_bits"] / empty["exec_error_bits"],
                                             "removes": 1 - r["necessity_error_bits"] / empty["necessity_error_bits"]}})
                    log(f"k={k} seed {seed}: reproduces {curve[-1]['shares']['reproduces']:.1%}, removes {curve[-1]['shares']['removes']:.1%}")
        (a.out / f"{b}.json").write_text(json.dumps({"behavior": b, "rounds": rounds, "sets": {k: [explain.token(u) for u in v] for k, v in sets.items()},
                                                    "curve": curve, "experiments": a.experiments, "checker": str(score_module.BINARY),
                                                    "seconds": round(time.time() - t0)}, indent=1))
        log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
