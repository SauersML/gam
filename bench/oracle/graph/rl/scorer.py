"""Scorers for the oracle's training (#2951): a batch of items (an explanation's source, or an IR, and its behavior) ->
the checker's score dicts ({"total_bits", "exec_error_bits", "necessity_error_bits", "alignment_error_bits",
"complexity_bits", "valid", "error", ...}).

  checker  score.py's Checker: the score the oracle is trained on. An item with "ir" (the nothing-named baseline) is
           scored as that IR.
  mock     no checker: code bits from mech.trace and no error terms; an invalid explanation pays MOCK_EMPTY_BITS.
           Plumbing only: its optimum is the shortest valid explanation.
  none     no score: evaluation only samples and saves (train.py --mode rescore scores later).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

GRAPH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GRAPH))

MOCK_EMPTY_BITS = 1.0e4


def mock(items: list[dict]) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor

    import mech

    with ThreadPoolExecutor(8) as ex:  # each trace is a sandboxed child process
        irs = list(ex.map(lambda it: mech.trace(it["source"], it["behavior"]["model"]), items))
    out = []
    for ir in irs:
        valid, tokens, types = ir["valid"], ir.get("python_tokens", 0), ir.get("token_types", 1)
        code = tokens * math.log2(max(types, 2)) if valid else 0.0
        out.append({"total_bits": code if valid else MOCK_EMPTY_BITS, "exec_error_bits": 0.0 if valid else MOCK_EMPTY_BITS,
                    "code_bits": code, "python_tokens": tokens if valid else 0, "valid": valid, "error": ir.get("error"), "scorer": "mock"})
    return out


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
        return it.get("seed", 0), it.get("uniform_seeds") or 0, it.get("experiments") or 32, json.dumps(it.get("options"), sort_keys=True)

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
                programs = [items[k].get("ir") or {"source": items[k]["source"], "explanation": items[k].get("explanation", "")} for k in chunk]
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


def none(items: list[dict]) -> list[dict]:
    return [{"total_bits": float("nan"), "valid": None, "scorer": "none"} for _ in items]


SCORERS = {"mock": mock, "checker": checker, "none": none}
