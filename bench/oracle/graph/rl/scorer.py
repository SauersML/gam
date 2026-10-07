"""Scorers for the oracle's program training (#2951): a batch of (program source, behavior) -> score
dicts in design.txt section 5's format ({"total_bits", "exec_error_bits", "reader_error_bits",
"code_bits", "python_tokens", "opaque_numbers", "opaque_bits", "valid", "error", ...}).

  checker  g-exec's score.py (the Rust checker's execution error + g-reader's reader error + code and
           opaque-number bits): the score the oracle is trained on.
  mock     until the checker runs: code bits from mech.trace (python_tokens x log2(token_types)) and no
           execution or reader error; an invalid program pays MOCK_EMPTY_BITS, standing in for the empty
           program's execution error, so validity and length are the only signal. Plumbing only:
           its optimum is the shortest valid program.
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
        valid, error, tokens, types = ir["valid"], ir.get("error"), ir.get("python_tokens", 0), ir.get("token_types", 1)
        code = tokens * math.log2(max(types, 2)) if valid else 0.0
        out.append({"total_bits": code if valid else MOCK_EMPTY_BITS, "exec_error_bits": 0.0 if valid else MOCK_EMPTY_BITS, "reader_error_bits": 0.0,
                    "code_bits": code, "python_tokens": tokens if valid else 0, "opaque_numbers": 0, "opaque_bits": 0.0, "valid": valid, "error": error,
                    "scorer": "mock"})
    return out


_CHECKERS = {}
WORKERS = 1
EXPORT = None


def checker(items: list[dict]) -> list[dict]:
    """score.py's Checker: WORKERS long-lived servers per target model, each taking whole behaviors (it loads
    a behavior once and caches M's outcomes on its seed-shared experiments) and scoring a behavior's
    programs of one seed in one score_batch request; the reader term when
    GRAPH_READER (reader_score.py serve's HOST:PORT) is set. Programs of one behavior and step share the
    experiments' seed, so a group's scores differ by the programs only."""
    from concurrent.futures import ThreadPoolExecutor

    import score

    sys.path.insert(0, str(GRAPH / "e2e"))
    from run import load_behavior  # g-int's: the behavior without the site manifest, which the vpd4l export cannot serve

    groups = {}
    for k, it in enumerate(items):
        groups.setdefault((it["behavior"]["model"], it["behavior"]["path"]), []).append(k)
    out = [None] * len(items)

    def run(w: int, model: str, path: str, ks: list[int]):
        c = _CHECKERS.get((model, w))
        if c is None:
            c = _CHECKERS[(model, w)] = score.Checker(model, EXPORT) if EXPORT else score.Checker(model)
            c.loaded = None
        if c.loaded != path:
            load_behavior(c, path)
            c.loaded = path
        def key(k):
            return items[k].get("seed", 0), items[k].get("uniform_seeds") or 0, items[k].get("experiments") or 32, json.dumps(items[k].get("options"), sort_keys=True)

        for seed, uniform, experiments, options in sorted({key(k) for k in ks}):  # one batch request per seed: M once per experiment, the programs in parallel
            batch = [k for k in ks if key(k) == (seed, uniform, experiments, options)]
            extra = {"options": json.loads(options)} if json.loads(options) else {}
            for k, r in zip(batch, c.score_batch([items[k]["source"] for k in batch], experiments=experiments, seed=seed, uniform_seeds=uniform or None, **extra)):
                out[k] = r

    per_worker = [[] for _ in range(WORKERS)]
    for g, (key, ks) in enumerate(groups.items()):
        per_worker[g % WORKERS].append((key, ks))

    def worker(w: int):
        for (model, path), ks in per_worker[w]:
            run(w, model, path, ks)

    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(worker, range(WORKERS)))
    return out


SCORERS = {"mock": mock, "checker": checker}
