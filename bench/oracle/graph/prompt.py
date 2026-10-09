"""The graph oracle's input (#2951): the ask and the text, nothing else. The oracle answers with a gate program
(mech.gates): Python defining on(tokens, targets), the subcomponents acting at each position, the circuit that computes
the model's prediction at the targets.

A task (text.py) is shown as what on() receives, the token strings and the positions whose next token is asked, with
the model's most probable next tokens there. A text example (activity()) asks instead which subcomponents act most
strongly at each position.

  prompt.py TASK.json      prints the prompt
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mech  # noqa: E402

ASK = ("Which of {model}'s subcomponents, at which positions, compute its prediction of the next token at the targets "
       "of this text? Answer with a Python function on(tokens, targets) returning, for each position, the subcomponents "
       "acting there.")
ACTIVITY = "Which of {model}'s subcomponents act most strongly at each position of this sequence? Answer with on(tokens, targets)."


def tokens_of(model: str, ids: list[int]) -> list[str]:
    tk = mech.tokenizer(model)
    return [tk.decode([i]) for i in ids]


def line(model: str, ids: list[int], targets: list[int], top: list) -> str:
    """One sequence as on() sees it, and the model's most probable next tokens at its targets."""
    nexts = "; ".join(f"at {t}: " + ", ".join(f"{tok!r} {p:.2f}" for tok, p in row[:3]) for t, row in zip(targets, top))
    return f"tokens = {tokens_of(model, ids)!r}, targets = {targets!r} -> {nexts}"


def render(task: dict) -> str:
    """The oracle's prompt for a task: the ask, then each of its texts as on() receives it."""
    model = task["model"]
    return "\n".join([ASK.format(model=model)] + [line(model, p["token_ids"], p["target_positions"], p["model_top"]) for p in task["prompts"]])


def activity(model: str, ids: list[int]) -> str:
    """The prompt of a text example: which subcomponents act most strongly at each position of `ids`."""
    return ACTIVITY.format(model=model) + f"\ntokens = {tokens_of(model, ids)!r}, targets = []"


def split_answer(answer: str) -> tuple[str, str]:
    """(program, the text after it) of an oracle's answer: the last fenced block that parses as Python. Without such
    a block: (answer, "")."""
    parts = answer.split("```")
    for k in range(len(parts) - 2, 0, -2):  # fenced blocks are the odd parts; take the last one that parses
        block = parts[k].split("\n", 1)[1] if "\n" in parts[k] else ""
        try:
            ast.parse(block)
        except SyntaxError:
            continue
        return block, "```".join(parts[k + 1:]).strip()
    return answer, ""


def program_of(answer: str) -> str:
    """The program in an oracle's answer (split_answer's first part)."""
    return split_answer(answer)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("task", type=Path)
    a = ap.parse_args()
    print(render(json.loads(a.task.read_text())))


if __name__ == "__main__":
    main()
