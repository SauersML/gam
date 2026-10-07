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


def checker(items: list[dict]) -> list[dict]:
    import score

    if hasattr(score, "score_many"):
        return score.score_many(items)
    return [score.score(it["source"], it["behavior"]) for it in items]


SCORERS = {"mock": mock, "checker": checker}
