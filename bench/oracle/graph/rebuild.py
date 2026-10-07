"""Rebuild-from-English (#2951 graph oracle): a frozen local coder model sees only a program's English
(its comments and docstrings, mech.english), the mech reference and the target model's sizes, and
writes the program again; the rebuild is traced and, with --behavior, scored by the checker beside
the original on the same experiments (same seed). English that rebuilds a program scoring like the
original states the mechanism in words; English that only restates the code's surface does not.

  rebuild.py PROGRAM.py [...] --model vpd4l --coder Qwen/Qwen3-8B --backend vllm \\
             [--behavior BEHAVIOR.json --experiments 32] --out OUT.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mech  # noqa: E402
import prompt  # noqa: E402

ASK = """\
Below is the English of an explanation of how the model produces a behavior: the comments and
docstrings of a mech program whose code was removed. Write that program again: the nodes, pieces and
edges the English states, nothing it does not state. Keep the English as its comments and docstrings.

{sizes}

English of the explanation:
<<<
{english}
>>>

Write the program."""


def request(source: str, model: str) -> str:
    """The coder's message for `source`: the mech reference, the model's sizes, the program's English."""
    sizes, pieces = prompt.views(model)
    return prompt.REFERENCE.format(pieces=pieces) + "\n\n" + ASK.format(sizes=sizes, english=mech.english(source))


def rebuild(sources: list[str], model: str, generate) -> list[dict]:
    """Per source: {"english", "request", "answer", "source" (the rebuild), "ir" (its trace)};
    `generate` maps a list of user messages to a list of answers (textgen.Generator)."""
    asks = [request(s, model) for s in sources]
    out = []
    for source, ask, answer in zip(sources, asks, generate(asks)):
        program = prompt.program_of(answer)
        out.append({"english": mech.english(source), "request": ask, "answer": answer, "source": program,
                    "ir": mech.trace(program, model)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("programs", nargs="+", type=Path)
    ap.add_argument("--model", required=True, choices=mech.MODELS)
    ap.add_argument("--coder", default="Qwen/Qwen3-8B")
    ap.add_argument("--backend", default="transformers", choices=["transformers", "vllm"])
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--behavior", type=Path, help="score the original and the rebuild with the checker")
    ap.add_argument("--experiments", type=int, default=32)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from textgen import Generator

    sources = [p.read_text() for p in a.programs]
    results = rebuild(sources, a.model, Generator(a.coder, a.backend, a.max_tokens))
    if a.behavior:
        from score import Checker

        with Checker(a.model) as checker:
            checker.behavior(str(a.behavior))
            for source, r in zip(sources, results):
                r["original_score"] = checker.score(source, experiments=a.experiments, seed=0, reader=False)
                r["rebuild_score"] = (checker.score(r["source"], experiments=a.experiments, seed=0, reader=False)
                                      if r["ir"]["valid"] else None)
    with a.out.open("w") as f:
        for path, r in zip(a.programs, results):
            f.write(json.dumps({"program": str(path), **r}) + "\n")
            print(path, "rebuild valid:", r["ir"]["valid"], r["ir"]["error"] or "",
                  "| original", (r.get("original_score") or {}).get("total_bits"),
                  "rebuild", (r.get("rebuild_score") or {}).get("total_bits"))


if __name__ == "__main__":
    main()
