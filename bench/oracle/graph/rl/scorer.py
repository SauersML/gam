"""Scorers for the oracle's training (#2951): a batch of items (an explanation's source, or an IR, and its task) -> the
checker's score dicts ({"exec_error_bits", "pairs", "valid", "error", ...}; score.order ranks them).

  checker  score.py's Checker: the score the oracle is trained on. An item with "ir" (the nothing-named baseline) is
           scored as that IR.
  mock     no checker: KL 0 for a valid explanation and its pairs from mech.trace. Plumbing only: its optimum is the
           valid explanation naming the fewest pairs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

GRAPH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GRAPH))


def mock(items: list[dict]) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor

    import mech
    import score

    with ThreadPoolExecutor(8) as ex:  # each trace is a sandboxed child process
        irs = list(ex.map(lambda it: it.get("ir") or mech.trace(it["source"], it["behavior"]["model"], behavior=it["behavior"]), items))
    return [{"exec_error_bits": 0.0, "pairs": score.pairs(ir), "valid": ir["valid"], "error": ir.get("error"), "scorer": "mock"} for ir in irs]


_CHECKERS = {}
WORKERS = 1
EXPORT = None
BATCH = 4  # programs per score request (vpd4l: 11 programs of 16 experiments passed a 20 GiB lease on the Mac)
MEMORY_GIB = None  # the checker server's lease (score.py's default when None)
VIEWS = None  # decomposition views the checker attaches (score.Checker's default when None)
DEVICE = None  # "gpu": the checker's single-precision device path


def checker(items: list[dict]) -> list[dict]:
    """score.py's Checker: WORKERS long-lived servers per target model, each taking whole behaviors (it loads a behavior
    once and caches M's outcomes on its experiments) and scoring a behavior's programs of one seed in one score_batch
    request, so a group's scores differ by the programs only."""
    from concurrent.futures import ThreadPoolExecutor

    import score

    groups = {}
    for k, it in enumerate(items):
        groups.setdefault((it["behavior"]["model"], it["behavior"]["path"]), []).append(k)
    out = [None] * len(items)

    def key(k):
        it = items[k]
        return it.get("seed", 0), it.get("uniform_seeds") or 0, it.get("experiments", 0), json.dumps(it.get("options"), sort_keys=True)

    def run(w: int, model: str, path: str, ks: list[int]):
        c = _CHECKERS.get((model, w))
        if c is None:
            c = _CHECKERS[(model, w)] = score.Checker(model, EXPORT, memory_gib=MEMORY_GIB, views=VIEWS, **({"device": DEVICE} if DEVICE else {}))
            c.loaded = None
        if c.loaded != path:
            c.behavior(path)
            c.loaded = path
        for seed, uniform, experiments, options in sorted({key(k) for k in ks}):  # one request per seed: M once per experiment
            batch = [k for k in ks if key(k) == (seed, uniform, experiments, options)]
            for s in range(0, len(batch), BATCH):  # a server's memory grows with the programs of one request
                chunk = batch[s: s + BATCH]
                programs = [items[k].get("ir") or items[k]["source"] for k in chunk]
                for k, r in zip(chunk, c.score_batch(programs, experiments=experiments, seed=seed, uniform_seeds=uniform or None,
                                                     options=json.loads(options))):
                    out[k] = r

    per_worker = [[] for _ in range(WORKERS)]
    for g, (key_, ks) in enumerate(groups.items()):
        per_worker[g % WORKERS].append((key_, ks))

    def worker(w: int):
        for (model, path), ks in per_worker[w]:
            run(w, model, path, ks)

    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(worker, range(WORKERS)))
    return out


SCORERS = {"mock": mock, "checker": checker}
